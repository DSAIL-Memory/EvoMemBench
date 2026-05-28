"""GraphRAG memory backend for the BFCL cross-episode pipeline.

Concept-graph augmented dense retrieval over chunked agent trajectories.
Mirrors Flash-Searcher's graphrag_provider.py, ported from AgentGym's
graphrag_adapter.py to the BFCL Memory ABC (utilize/update/load_from_disk).

Algorithm:
  update() — format trajectory → chunk → extract concepts (1 Ark batch LLM call,
             optional; degrades to empty concepts if Ark creds absent) → embed
             chunks (OpenAI-compatible endpoint) → add nodes+edges to networkx.Graph
             (weight = 0.7·cosine + 0.3·Jaccard(concepts)) → persist
             memory_bank.jsonl + embeddings.jsonl + graph.gpickle

  utilize() — embed query (system_prompt) → cosine top-k seeds → Dijkstra-style
              graph walk → inject results under _UTILIZE_PREFIX (no LLM calls)

Persistence (per storage_dir/):
  memory_bank.jsonl  — {task_id, chunk_index, chunk_text, context_category,
                         sub_category, concepts}
  embeddings.jsonl   — {id, text, embedding}
  graph.gpickle      — raw pickle of networkx.Graph

Chunking and trajectory formatting are imported from bm25_backend to guarantee
identical preprocessing across all baselines (same as AgentGym reference).
"""
from __future__ import annotations

import heapq
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.bm25_backend import (
    _UTILIZE_PREFIX,
    _chunk_text,
    _format_trajectory,
    _token_count,
)

_EMBED_BATCH_SIZE = 10  # DashScope hard limit: ≤10 texts per embedding API call

# text-embedding-v4 documents an 8192-token input limit.
# Stay well below to absorb differences between cl100k_base and DashScope's internal tokenizer
# (trading_bot JSON payloads can tokenize very differently across tokenizers).
_EMBED_BUDGET_TOKENS = 6500

try:
    import tiktoken as _tiktoken
    _EMBED_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _EMBED_ENC = None


def _truncate_for_embed(text: str, keep_tail: bool = False) -> str:
    """Truncate text to fit text-embedding-v4's 8192-token limit.

    keep_tail=True preserves the end of the text (used for queries, where the
    task-specific framing appended last is most important).
    keep_tail=False preserves the head (used for indexing chunks).
    """
    if _EMBED_ENC is not None:
        tokens = _EMBED_ENC.encode(text)
        if len(tokens) <= _EMBED_BUDGET_TOKENS:
            return text
        kept = tokens[-_EMBED_BUDGET_TOKENS:] if keep_tail else tokens[:_EMBED_BUDGET_TOKENS]
        return _EMBED_ENC.decode(kept)
    char_limit = _EMBED_BUDGET_TOKENS * 4
    if len(text) <= char_limit:
        return text
    return text[-char_limit:] if keep_tail else text[:char_limit]

_CONCEPT_EXTRACT_SYSTEM = (
    "You are a concept-extraction assistant. "
    "Given a text, extract up to {max_concepts} key concepts: topics, entities, "
    "strategies, domain terms, or task types. "
    'Respond ONLY with a JSON object with a single key "concepts" whose value '
    "is a list of short strings (1-5 words each). No prose, no extra keys."
)


class GraphRagMemory(Memory):
    """Concept-graph augmented dense retrieval memory backend.

    Thread-safe via threading.RLock. Slow I/O (LLM extraction + embedding API
    calls) is performed outside the lock; lock is held only for state mutation,
    JSONL appends, graph save, and matrix build.
    """

    EDGES_THRESHOLD: float = 0.8
    CONCEPTS_WEIGHT: float = 0.3       # Jaccard weight; semantic weight = 1 - 0.3
    WALK_MAX_NODES: int = 20
    CONCEPTS_PER_NODE_MAX: int = 10

    def __init__(
        self,
        storage_dir: Path | str,
        # Embedding creds (constructor → DASHSCOPE_* → OPENAI_*)
        dashscope_api_key: str = "",
        dashscope_base_url: str = "",
        embed_model: Optional[str] = None,
        # Concept-extraction creds (soft — degrades if absent)
        ark_api_key: str = "",          # → DEEPSEEK_API_KEY env
        ark_model: str = "",            # → BATCH_MODEL env
        # Retrieval / chunking
        top_k: int = 3,
        chunk_size: int = 1024,
        embed_batch_size: int = _EMBED_BATCH_SIZE,
        embed_max_retries: int = 10,
        embed_retry_delay: float = 3.0,
        # Concept extraction
        extract_max_retries: int = 10,
        extract_retry_delay: float = 5.0,
        # Graph hyperparameters (None → class constants)
        edges_threshold: Optional[float] = None,
        concepts_weight: Optional[float] = None,
        walk_max_nodes: Optional[int] = None,
        concepts_per_node_max: Optional[int] = None,
        clear: bool = False,
        readonly: bool = False,
        **unused,  # absorb mem_model, etc. passed by the driver
    ):
        super().__init__(clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._bank_path = self._storage_dir / "memory_bank.jsonl"
        self._embeddings_path = self._storage_dir / "embeddings.jsonl"
        self._graph_path = self._storage_dir / "graph.gpickle"

        # Embedding credentials: constructor args → DASHSCOPE_* → OPENAI_*
        self._embed_api_key = (
            dashscope_api_key
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._embed_base_url = (
            dashscope_base_url
            or os.environ.get("DASHSCOPE_BASE_URL", "")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        self._embed_model = (
            embed_model
            or os.environ.get("BFCL_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
        )
        if not self._embed_api_key or not self._embed_base_url:
            raise RuntimeError(
                "GraphRagMemory requires an embedding API key + base URL. "
                "Set OPENAI_API_KEY + OPENAI_BASE_URL (or DASHSCOPE_API_KEY + "
                "DASHSCOPE_BASE_URL) in the environment, or pass dashscope_api_key / "
                "dashscope_base_url to the constructor."
            )

        # Ark concept-extraction credentials (soft)
        self._ark_api_key = (
            ark_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self._ark_model = (
            ark_model or os.environ.get("BATCH_MODEL", "")
        )

        self._top_k = top_k
        self._chunk_size = chunk_size
        self._embed_batch_size = embed_batch_size
        self._embed_max_retries = embed_max_retries
        self._embed_retry_delay = embed_retry_delay
        self._extract_max_retries = extract_max_retries
        self._extract_retry_delay = extract_retry_delay

        self._edges_threshold = (
            edges_threshold if edges_threshold is not None else self.EDGES_THRESHOLD
        )
        self._concepts_weight = (
            concepts_weight if concepts_weight is not None else self.CONCEPTS_WEIGHT
        )
        self._walk_max_nodes = (
            walk_max_nodes if walk_max_nodes is not None else self.WALK_MAX_NODES
        )
        self._concepts_per_node_max = (
            concepts_per_node_max
            if concepts_per_node_max is not None
            else self.CONCEPTS_PER_NODE_MAX
        )

        self._memory_bank: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._graph: nx.Graph = nx.Graph()
        self._matrix: Optional[np.ndarray] = None
        self._dirty: bool = False
        self._update_counter: int = 0
        self._lock = threading.RLock()

        # Lazy-initialized API clients
        self._embed_client = None
        self._ark_client = None
        self._clients_initialized = False

        if clear:
            self._bank_path.write_text("")
            self._embeddings_path.write_text("")
            if self._graph_path.exists():
                self._graph_path.unlink()

        self._load_bank()

    # ── client initialization ──────────────────────────────────────────────────

    def _ensure_clients(self) -> None:
        if self._clients_initialized:
            return
        with self._lock:
            if self._clients_initialized:
                return

            import openai
            self._embed_client = openai.OpenAI(
                api_key=self._embed_api_key,
                base_url=self._embed_base_url,
            )

            if self._ark_model and self._ark_api_key:
                try:
                    from volcenginesdkarkruntime import Ark
                    self._ark_client = Ark(api_key=self._ark_api_key)
                except ImportError:
                    print(
                        "[GraphRAG] WARNING: volcenginesdkarkruntime not installed — "
                        "concept extraction disabled. "
                        "Install: pip install 'volcengine-python-sdk[ark]'"
                    )
            else:
                if not self._ark_model:
                    print(
                        "[GraphRAG] WARNING: BATCH_MODEL not set — concept extraction "
                        "disabled (graph edges will be cosine-only)."
                    )

            self._clients_initialized = True

    # ── embedding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _embed_batch(self, texts: list[str], keep_tail: bool = False) -> list[np.ndarray]:
        """Embed texts in ≤embed_batch_size sub-batches with exponential-backoff retry.

        Truncates each text to _EMBED_BUDGET_TOKENS before sending to avoid the
        embedding API's 8192-token hard limit (mirrors mem0_backend's pattern).
        keep_tail=True preserves the tail (for query embeddings in utilize()).
        """
        self._ensure_clients()
        texts = [_truncate_for_embed(t, keep_tail=keep_tail) for t in texts]
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._embed_batch_size):
            sub = texts[i : i + self._embed_batch_size]
            last_exc: Optional[Exception] = None
            for attempt in range(self._embed_max_retries):
                try:
                    resp = self._embed_client.embeddings.create(
                        model=self._embed_model, input=sub
                    )
                    for d in resp.data:
                        out.append(
                            self._l2_normalize(np.asarray(d.embedding, dtype=np.float32))
                        )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    # 400 (InvalidParameter / input too long) is deterministic — retrying
                    # with the same payload will always fail.  Fail fast instead of burning
                    # through 10 attempts of exponential back-off.
                    err_str = str(e)
                    if "400" in err_str or "InvalidParameter" in err_str or "input length" in err_str.lower():
                        break
                    if attempt < self._embed_max_retries - 1:
                        delay = self._embed_retry_delay * (2 ** attempt)
                        print(
                            f"[GraphRAG] embed retry "
                            f"{attempt + 1}/{self._embed_max_retries} in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
            if last_exc is not None:
                raise last_exc
        return out

    # ── concept extraction ─────────────────────────────────────────────────────

    def _extract_concepts(self, text: str) -> tuple[list[str], int, int]:
        """Extract concepts via one Ark batch call.

        Returns (concepts, prompt_tokens, completion_tokens).
        Falls back to ([], 0, 0) when the Ark client is unavailable or retries fail.
        """
        self._ensure_clients()
        if not self._ark_client or not self._ark_model:
            return [], 0, 0

        system_msg = _CONCEPT_EXTRACT_SYSTEM.format(
            max_concepts=self._concepts_per_node_max
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text[:4000]},
        ]

        for attempt in range(self._extract_max_retries):
            try:
                response = self._ark_client.batch.chat.completions.create(
                    model=self._ark_model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content or ""
                parsed = json.loads(response_text)
                concepts_raw = parsed.get("concepts", [])
                if not isinstance(concepts_raw, list):
                    concepts_raw = []
                concepts = [str(c).lower().strip() for c in concepts_raw if c]

                p_tok, c_tok = 0, 0
                if response.usage:
                    p_tok = response.usage.prompt_tokens or 0
                    c_tok = response.usage.completion_tokens or 0
                return concepts, p_tok, c_tok

            except Exception as e:
                if attempt < self._extract_max_retries - 1:
                    delay = self._extract_retry_delay * (2 ** min(attempt, 4))
                    print(
                        f"[GraphRAG] concept extraction retry "
                        f"{attempt + 1}/{self._extract_max_retries} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[GraphRAG] concept extraction failed after "
                        f"{self._extract_max_retries} attempts — using empty concepts."
                    )
        return [], 0, 0

    # ── graph helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    def _add_node_and_edges(
        self, i: int, vec: np.ndarray, concepts: list[str]
    ) -> None:
        """Add weighted edges from node i to all j < i where cosine > edges_threshold.

        Must be called inside self._lock (reads self._embeddings and self._memory_bank).
        """
        concepts_set = set(concepts)
        for j in range(i):
            sim = float(vec @ self._embeddings[j])
            if sim > self._edges_threshold:
                concepts_j = set(self._memory_bank[j].get("concepts") or [])
                jacc = self._jaccard(concepts_set, concepts_j)
                w = (1.0 - self._concepts_weight) * sim + self._concepts_weight * jacc
                self._graph.add_edge(i, j, weight=w)

    def _build_matrix(self) -> None:
        if not self._embeddings:
            self._matrix = None
        else:
            self._matrix = np.stack(self._embeddings, axis=0)
        self._dirty = False

    def _graph_walk(
        self, seed_scores: dict[int, float], graph: nx.Graph
    ) -> list[int]:
        """Dijkstra-style heap expansion from cosine top-k seeds via graph edges.

        Accepts graph as an explicit parameter so callers can safely pass a
        captured reference and call this outside the lock.
        """
        visited: set[int] = set()
        ordered: list[int] = []
        pq: list[tuple[float, int]] = []

        for idx, score in seed_scores.items():
            heapq.heappush(pq, (1.0 / max(score, 1e-9), idx))

        while pq and len(ordered) < self._top_k and len(visited) < self._walk_max_nodes:
            prio, idx = heapq.heappop(pq)
            if idx in visited:
                continue
            visited.add(idx)
            ordered.append(idx)
            for nbr, data in graph[idx].items():
                if nbr in visited:
                    continue
                edge_w = data.get("weight", 1e-9)
                heapq.heappush(pq, (prio + 1.0 / max(edge_w, 1e-9), nbr))

        return ordered

    def _save_graph(self) -> None:
        with open(self._graph_path, "wb") as f:
            pickle.dump(self._graph, f)

    def _rebuild_graph(self) -> None:
        self._graph = nx.Graph()
        for i in range(len(self._memory_bank)):
            self._graph.add_node(i)
            vec_i = self._embeddings[i]
            concepts_i = self._memory_bank[i].get("concepts") or []
            self._add_node_and_edges(i, vec_i, concepts_i)
        self._save_graph()

    # ── persistence ────────────────────────────────────────────────────────────

    def _load_bank(self) -> None:
        bank_entries: list[dict] = []
        embed_entries: list[np.ndarray] = []

        if self._bank_path.exists():
            try:
                with open(self._bank_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            bank_entries.append(json.loads(line))
            except Exception as e:
                print(f"[GraphRAG] failed to load bank: {e}")
                return

        if not bank_entries:
            return

        if "chunk_text" not in bank_entries[0]:
            print(f"[GraphRAG] stale schema in {self._bank_path}; discarding.")
            self._bank_path.write_text("")
            self._embeddings_path.write_text("")
            if self._graph_path.exists():
                self._graph_path.unlink()
            return

        if self._embeddings_path.exists():
            try:
                with open(self._embeddings_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            vec = np.asarray(obj["embedding"], dtype=np.float32)
                            embed_entries.append(self._l2_normalize(vec))
            except Exception as e:
                print(f"[GraphRAG] failed to load embeddings: {e}")
                embed_entries = []

        if len(embed_entries) != len(bank_entries):
            print(
                f"[GraphRAG] bank/embedding count mismatch "
                f"({len(bank_entries)} vs {len(embed_entries)}); "
                f"embeddings cleared — will re-accumulate on next update."
            )
            embed_entries = []

        self._memory_bank = bank_entries
        self._embeddings = embed_entries

        if bank_entries and embed_entries:
            self._dirty = True

        if embed_entries:
            if self._graph_path.exists():
                try:
                    with open(self._graph_path, "rb") as f:
                        g = pickle.load(f)
                    if g.number_of_nodes() != len(bank_entries):
                        raise ValueError(
                            f"node count mismatch: {g.number_of_nodes()} vs {len(bank_entries)}"
                        )
                    self._graph = g
                except Exception as e:
                    print(f"[GraphRAG] graph load failed ({e}) — rebuilding...")
                    self._rebuild_graph()
            else:
                self._rebuild_graph()

    # ── Memory interface ───────────────────────────────────────────────────────

    def utilize(self, system_prompt: str) -> str:
        t0 = time.monotonic()

        with self._lock:
            if not self._memory_bank or not self._embeddings:
                return ""
            if self._dirty or self._matrix is None:
                self._build_matrix()
            if self._matrix is None:
                return ""
            matrix = self._matrix
            bank_snapshot = list(self._memory_bank)
            graph_snapshot = self._graph

        if not system_prompt:
            return ""

        try:
            query_vec = self._embed_batch([system_prompt], keep_tail=True)[0]
        except Exception as e:
            print(f"[GraphRAG] utilize embed error: {e}")
            return ""

        scores = matrix @ query_vec
        k = min(self._top_k, len(bank_snapshot))
        top_indices = scores.argsort()[::-1][:k]
        seed_scores = {int(i): float(scores[i]) for i in top_indices}

        if graph_snapshot.number_of_edges() > 0:
            walk_order = self._graph_walk(seed_scores, graph_snapshot)
        else:
            walk_order = list(seed_scores.keys())

        hits = [
            f"- {bank_snapshot[int(i)]['chunk_text']}"
            for i in walk_order[: self._top_k]
            if bank_snapshot[int(i)].get("chunk_text")
        ]

        elapsed = time.monotonic() - t0
        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1
            u.n_embed_calls += 1
            u.embedding_tokens += _token_count(system_prompt)
            u.latency_s += elapsed

        if not hits:
            return ""
        return _UTILIZE_PREFIX + "\n".join(hits)

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return

        t0 = time.monotonic()
        text = _format_trajectory(trajectory)
        if not text.strip():
            return

        chunks = _chunk_text(text, self._chunk_size)
        total_embed_tokens = sum(_token_count(c) for c in chunks)

        # Slow API calls outside lock
        concepts, p_tok, c_tok = self._extract_concepts(text)
        try:
            vectors = self._embed_batch(chunks)
        except Exception as e:
            print(f"[GraphRAG] update embed error: {e}")
            return

        with self._lock:
            task_id = str(self._update_counter)
            self._update_counter += 1
            try:
                with (
                    open(self._bank_path, "a", encoding="utf-8") as bf,
                    open(self._embeddings_path, "a", encoding="utf-8") as ef,
                ):
                    for i_chunk, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
                        bank_entry = {
                            "task_id": task_id,
                            "chunk_index": i_chunk,
                            "chunk_text": chunk_text,
                            "context_category": "",
                            "sub_category": "",
                            "concepts": concepts,
                        }
                        embed_entry = {
                            "id": f"{task_id}_chunk{i_chunk}",
                            "text": chunk_text,
                            "embedding": vec.tolist(),
                        }
                        new_node = len(self._memory_bank)
                        self._memory_bank.append(bank_entry)
                        self._embeddings.append(vec)
                        self._graph.add_node(new_node)
                        self._add_node_and_edges(new_node, vec, concepts)
                        bf.write(json.dumps(bank_entry, ensure_ascii=False) + "\n")
                        ef.write(json.dumps(embed_entry, ensure_ascii=False) + "\n")
                self._save_graph()
                self._dirty = True
            except Exception as e:
                print(f"[GraphRAG] update write error: {e}")
                return

        elapsed = time.monotonic() - t0
        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1
            u.n_embed_calls += 1
            u.embedding_tokens += total_embed_tokens
            if p_tok or c_tok:
                u.n_llm_calls += 1
                u.input_tokens += p_tok
                u.output_tokens += c_tok
            u.latency_s += elapsed

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "GraphRagMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
