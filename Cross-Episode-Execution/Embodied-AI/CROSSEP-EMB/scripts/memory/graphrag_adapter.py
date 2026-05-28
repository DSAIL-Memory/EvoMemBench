"""GraphRAG memory backend for ALFWorld evaluation.

Concept-graph augmented dense retrieval over chunked agent trajectories.
Mirrors Flash-Searcher's graphrag_provider.py but adapted to the BaseMemory
inject+update API used in this fork.

Algorithm:
  update() — format trajectory → chunk → extract concepts (1 Ark LLM call) →
             embed chunks (DashScope text-embedding-v4) → add nodes+edges to
             networkx.Graph (weight = 0.7·cosine + 0.3·Jaccard(concepts)) →
             persist memory_bank.jsonl + embeddings.jsonl + graph.gpickle
  inject() — embed query → cosine top-k seeds → Dijkstra-style graph walk →
             inject results into system prompt (no LLM in retrieval path)

Persistence (per task, under memory_dir/):
  memory_bank.jsonl  — {task_id, chunk_index, chunk_text, context_category,
                         sub_category, concepts}
  embeddings.jsonl   — {id, text, embedding}
  graph.gpickle      — raw pickle of networkx.Graph

Chunking and trajectory formatting are reused from bm25_adapter to guarantee
identical text pre-processing across all baselines.
"""

from __future__ import annotations

import heapq
import json
import os
import pickle
import sys
import threading
import time
from copy import deepcopy

import networkx as nx
import numpy as np

from .base import BaseMemory, MemoryCallStats
from .bm25_adapter import (
    _MEMORY_SKIP_MARKER,  # noqa: F401  (re-exported for callers)
    _chunk_text,
    _format_trajectory,
    _token_count,
)

# DashScope hard limit: ≤10 texts per embedding API call
_EMBED_BATCH_SIZE = 10

_CONCEPT_EXTRACT_SYSTEM = (
    "You are a concept-extraction assistant. "
    "Given a text, extract up to {max_concepts} key concepts: topics, entities, "
    "strategies, domain terms, or task types. "
    "Respond ONLY with a JSON object with a single key \"concepts\" whose value "
    "is a list of short strings (1-5 words each). No prose, no extra keys."
)


class GraphRagMemory(BaseMemory):
    """Concept-graph augmented dense retrieval memory backend.

    Each update() call chunks the trajectory, extracts concepts via one Ark LLM
    call, embeds chunks with DashScope, builds networkx edges (cosine+Jaccard),
    and persists three files under memory_dir.

    inject() embeds the query, picks cosine top-k as seeds, then runs a
    Dijkstra-style graph walk — zero LLM calls in the retrieval path.

    Thread-safe via threading.RLock. Slow I/O (LLM + embedding API calls) is
    performed outside the lock; lock is held only for state mutation and file
    appends.
    """

    # Algorithm constants (match Flash-Searcher's graphrag_provider.py)
    EDGES_THRESHOLD: float = 0.8
    CONCEPTS_WEIGHT: float = 0.3       # Jaccard weight; semantic weight = 1 - 0.3
    WALK_MAX_NODES: int = 20
    CONCEPTS_PER_NODE_MAX: int = 10

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        memory_dir: str | None = None,
        top_k: int = 10,
        chunk_size: int = 1024,
        # ── embedding (DashScope text-embedding-v4) ──────────────────────────
        embed_model: str = "text-embedding-v4",
        embed_api_key: str | None = None,       # fallback → DASHSCOPE_API_KEY env
        embed_base_url: str | None = None,      # fallback → DASHSCOPE_BASE_URL env
        embed_batch_size: int = _EMBED_BATCH_SIZE,
        embed_max_retries: int = 10,
        embed_retry_delay: float = 3.0,
        # ── concept extraction LLM (Volcengine Ark / BATCH_MODEL) ────────────
        extract_model: str | None = None,       # fallback → BATCH_MODEL env
        ark_api_key: str | None = None,         # fallback → DEEPSEEK_API_KEY env
        extract_max_retries: int = 10,
        extract_retry_delay: float = 5.0,
        # ── graph hyperparameters ────────────────────────────────────────────
        edges_threshold: float | None = None,   # None → class constant 0.8
        concepts_weight: float | None = None,   # None → class constant 0.3
        walk_max_nodes: int | None = None,      # None → class constant 20
        concepts_per_node_max: int | None = None,  # None → class constant 10
        return_on_in: bool = False,
    ):
        super().__init__(memory_type="graphrag", read_only=read_only)
        if memory_dir is None:
            raise ValueError("GraphRagMemory requires 'memory_dir' to be set.")

        self._agent_id = agent_id
        self._top_k = top_k
        self._chunk_size = chunk_size

        self._embed_model = embed_model
        self._embed_api_key = embed_api_key
        self._embed_base_url = embed_base_url
        self._embed_batch_size = embed_batch_size
        self._embed_max_retries = embed_max_retries
        self._embed_retry_delay = embed_retry_delay

        self._extract_model_arg = extract_model  # may be resolved later from env
        self._ark_api_key = ark_api_key
        self._extract_max_retries = extract_max_retries
        self._extract_retry_delay = extract_retry_delay

        self._edges_threshold = edges_threshold if edges_threshold is not None else self.EDGES_THRESHOLD
        self._concepts_weight = concepts_weight if concepts_weight is not None else self.CONCEPTS_WEIGHT
        self._walk_max_nodes = walk_max_nodes if walk_max_nodes is not None else self.WALK_MAX_NODES
        self._concepts_per_node_max = (
            concepts_per_node_max if concepts_per_node_max is not None
            else self.CONCEPTS_PER_NODE_MAX
        )
        self._return_on_in = return_on_in

        self._memory_dir = memory_dir
        self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self._embeddings_path = os.path.join(memory_dir, "embeddings.jsonl")
        self._graph_path = os.path.join(memory_dir, "graph.gpickle")

        self._memory_bank: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._graph: nx.Graph = nx.Graph()
        self._matrix: np.ndarray | None = None
        self._dirty: bool = False
        self._lock = threading.RLock()

        # Lazy-initialized API clients
        self._embed_client = None
        self._ark_client = None
        self._extract_model: str | None = None  # resolved in _ensure_clients
        self._clients_initialized = False

        os.makedirs(memory_dir, exist_ok=True)
        self._load_bank()

    # ── client initialization ──────────────────────────────────────────────────

    def _ensure_clients(self) -> None:
        if self._clients_initialized:
            return
        with self._lock:
            if self._clients_initialized:
                return

            # Embedding client (DashScope / OpenAI-compatible)
            api_key = self._embed_api_key or os.environ.get("DASHSCOPE_API_KEY")
            base_url = self._embed_base_url or os.environ.get("DASHSCOPE_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError(
                    "GraphRagMemory requires DASHSCOPE_API_KEY + DASHSCOPE_BASE_URL "
                    "(set in environment or pass embed_api_key / embed_base_url in config)."
                )
            import openai
            self._embed_client = openai.OpenAI(api_key=api_key, base_url=base_url)

            # Ark LLM client for concept extraction (optional)
            self._extract_model = (
                self._extract_model_arg or os.environ.get("BATCH_MODEL") or ""
            )
            ark_key = self._ark_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            if self._extract_model and ark_key:
                try:
                    from volcenginesdkarkruntime import Ark
                    self._ark_client = Ark(api_key=ark_key)
                except ImportError:
                    print(
                        f"[GraphRAG:{self._agent_id}] WARNING: volcenginesdkarkruntime "
                        "not installed — concept extraction disabled. "
                        "Install: pip install 'volcengine-python-sdk[ark]'"
                    )
            else:
                if not self._extract_model:
                    print(
                        f"[GraphRAG:{self._agent_id}] WARNING: BATCH_MODEL not set — "
                        "concept extraction disabled (graph edges will be cosine-only)."
                    )

            self._clients_initialized = True

    # ── embedding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts in ≤embed_batch_size sub-batches with exponential-backoff retry."""
        self._ensure_clients()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._embed_batch_size):
            sub = texts[i : i + self._embed_batch_size]
            last_exc: Exception | None = None
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
                    if attempt < self._embed_max_retries - 1:
                        delay = self._embed_retry_delay * (2 ** attempt)
                        print(
                            f"[GraphRAG:{self._agent_id}] embed retry "
                            f"{attempt + 1}/{self._embed_max_retries} in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
            if last_exc is not None:
                raise last_exc
        return out

    # ── concept extraction ─────────────────────────────────────────────────────

    def _extract_concepts(
        self, text: str
    ) -> tuple[list[str], int, int, int]:
        """Extract concepts via one Ark batch call. Returns (concepts, p_tok, c_tok, cached).

        Falls back to ([], 0, 0, 0) if Ark client is unavailable or all retries fail.
        """
        self._ensure_clients()
        if not self._ark_client or not self._extract_model:
            return [], 0, 0, 0

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
                    model=self._extract_model,
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

                p_tok, c_tok, cached = 0, 0, 0
                if response.usage:
                    p_tok = response.usage.prompt_tokens or 0
                    c_tok = response.usage.completion_tokens or 0
                    if (
                        hasattr(response.usage, "prompt_tokens_details")
                        and response.usage.prompt_tokens_details
                    ):
                        cached = (
                            getattr(
                                response.usage.prompt_tokens_details,
                                "cached_tokens",
                                0,
                            )
                            or 0
                        )
                return concepts, p_tok, c_tok, cached

            except Exception as e:
                if attempt < self._extract_max_retries - 1:
                    delay = self._extract_retry_delay * (2 ** min(attempt, 4))
                    print(
                        f"[GraphRAG:{self._agent_id}] concept extraction retry "
                        f"{attempt + 1}/{self._extract_max_retries} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[GraphRAG:{self._agent_id}] concept extraction failed after "
                        f"{self._extract_max_retries} attempts — using empty concepts."
                    )
        return [], 0, 0, 0

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

        Accepts graph as an explicit parameter so callers outside the lock can
        safely pass a captured reference.
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

        # 1. Load memory bank
        if os.path.exists(self._bank_path):
            try:
                with open(self._bank_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            bank_entries.append(json.loads(line))
            except Exception as e:
                print(f"[GraphRAG:{self._agent_id}] failed to load bank: {e}")
                return

        if not bank_entries:
            return

        if "chunk_text" not in bank_entries[0]:
            print(
                f"[GraphRAG:{self._agent_id}] stale schema in {self._bank_path}; discarding."
            )
            for path in (self._bank_path, self._embeddings_path, self._graph_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        # 2. Load embeddings
        if os.path.exists(self._embeddings_path):
            try:
                with open(self._embeddings_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            vec = np.asarray(obj["embedding"], dtype=np.float32)
                            embed_entries.append(self._l2_normalize(vec))
            except Exception as e:
                print(f"[GraphRAG:{self._agent_id}] failed to load embeddings: {e}")
                embed_entries = []

        if len(embed_entries) != len(bank_entries):
            print(
                f"[GraphRAG:{self._agent_id}] bank/embedding count mismatch "
                f"({len(bank_entries)} vs {len(embed_entries)}); "
                f"embeddings cleared — will re-accumulate on next update."
            )
            embed_entries = []

        self._memory_bank = bank_entries
        self._embeddings = embed_entries

        if bank_entries and embed_entries:
            self._dirty = True

        # 3. Load or rebuild graph
        if embed_entries:
            if os.path.exists(self._graph_path):
                try:
                    with open(self._graph_path, "rb") as f:
                        g = pickle.load(f)
                    if g.number_of_nodes() != len(bank_entries):
                        raise ValueError(
                            f"node count mismatch: {g.number_of_nodes()} vs {len(bank_entries)}"
                        )
                    self._graph = g
                except Exception as e:
                    print(
                        f"[GraphRAG:{self._agent_id}] graph load failed ({e}) — rebuilding..."
                    )
                    self._rebuild_graph()
            else:
                self._rebuild_graph()

        print(
            f"[GraphRAG:{self._agent_id}] loaded {len(bank_entries)} chunks, "
            f"{len(embed_entries)} embeddings, "
            f"{self._graph.number_of_nodes()} nodes / "
            f"{self._graph.number_of_edges()} edges."
        )

    # ── public interface ───────────────────────────────────────────────────────

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]
        t0 = time.time()

        # Acquire lock only to check/build matrix and capture snapshots
        with self._lock:
            if not self._memory_bank or not self._embeddings:
                return conversation, MemoryCallStats(latency=time.time() - t0)
            if self._dirty or self._matrix is None:
                self._build_matrix()
            if self._matrix is None:
                return conversation, MemoryCallStats(latency=time.time() - t0)
            # Capture immutable references; matrix/graph are replaced (not mutated) on update
            matrix = self._matrix
            bank_snapshot = list(self._memory_bank)
            graph_snapshot = self._graph   # nx.Graph reference; safe while we hold snapshot

        # Embed query and walk graph outside the lock (slow HTTP call)
        try:
            query_vec = self._embed_batch([query])[0]
        except Exception as e:
            print(f"[GraphRAG:{self._agent_id}] inject embed error: {e}")
            return conversation, MemoryCallStats(latency=time.time() - t0)

        scores = matrix @ query_vec
        k = min(self._top_k, len(bank_snapshot))
        top_indices = np.argsort(scores)[::-1][:k]
        seed_scores = {int(i): float(scores[i]) for i in top_indices}

        if graph_snapshot.number_of_edges() > 0:
            walk_order = self._graph_walk(seed_scores, graph_snapshot)
        else:
            walk_order = list(seed_scores.keys())

        hits = []
        for idx in walk_order[: self._top_k]:
            chunk_text = bank_snapshot[int(idx)].get("chunk_text", "")
            if chunk_text:
                hits.append(f"- {chunk_text}")

        elapsed = time.time() - t0
        stats = MemoryCallStats(
            latency=elapsed,
            embedding_tokens=_token_count(query),
        )

        if not hits:
            return conversation, stats

        memory_lines = "\n".join(hits)
        injection = f"\n\n--- Retrieved Memories ---\n{memory_lines}\n---"
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = new_conv[0]["content"] + injection
        return new_conv, stats

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        t0 = time.time()
        text = _format_trajectory(conversation)
        if not text.strip():
            return MemoryCallStats(latency=time.time() - t0)

        chunks = _chunk_text(text, self._chunk_size)
        task_id = str(data_idx) if data_idx is not None else ""
        total_embed_tokens = sum(_token_count(c) for c in chunks)

        # Concept extraction + embedding outside the lock (slow API calls)
        concepts, p_tok, c_tok, cached = self._extract_concepts(text)
        try:
            vectors = self._embed_batch(chunks)
        except Exception as e:
            print(f"[GraphRAG:{self._agent_id}] update embed error for idx={task_id}: {e}")
            return MemoryCallStats(
                latency=time.time() - t0,
                input_tokens=p_tok,
                output_tokens=c_tok,
                cached_tokens=cached,
            )

        # Mutate state and write files inside the lock
        with self._lock:
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
                print(
                    f"[GraphRAG:{self._agent_id}] idx={task_id} added {len(chunks)} chunk(s) "
                    f"(total={len(self._memory_bank)}, "
                    f"concepts={len(concepts)}, "
                    f"edges={self._graph.number_of_edges()})"
                )
            except Exception as e:
                print(f"[GraphRAG:{self._agent_id}] update write error: {e}")

        return MemoryCallStats(
            latency=time.time() - t0,
            embedding_tokens=total_embed_tokens,
            input_tokens=p_tok,
            output_tokens=c_tok,
            cached_tokens=cached,
        )

    def load_from_disk(self, memory_dir: str | None = None, **kwargs) -> None:
        with self._lock:
            if memory_dir is not None:
                self._memory_dir = memory_dir
                self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
                self._embeddings_path = os.path.join(memory_dir, "embeddings.jsonl")
                self._graph_path = os.path.join(memory_dir, "graph.gpickle")
                os.makedirs(memory_dir, exist_ok=True)
            self._memory_bank = []
            self._embeddings = []
            self._graph = nx.Graph()
            self._matrix = None
            self._dirty = False
            self._load_bank()


if __name__ == "__main__":
    import shutil

    if not (os.environ.get("DASHSCOPE_API_KEY") and os.environ.get("DASHSCOPE_BASE_URL")):
        print("[SKIP] DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL not set; skipping live API tests.")
        sys.exit(0)

    TEST_DIR = "/tmp/graphrag_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("=== GraphRAG Adapter Smoke Test ===")
    print(f"memory_dir: {TEST_DIR}")

    m = GraphRagMemory(memory_dir=TEST_DIR, top_k=3)

    conv = [
        {"role": "user",      "content": "You are an agent in an ALFWorld household environment. Complete the task by issuing actions."},
        {"role": "assistant", "content": "I will complete the task step by step."},
        {"role": "user",      "content": "You are in the middle of a room. Looking quickly around you, you see a fridge 1, a microwave 1, a countertop 1.\nYour task is to: put a cool potato in the microwave."},
        {"role": "assistant", "content": "think: I need to find a potato first.\ngo to fridge 1"},
        {"role": "user",      "content": "You open the fridge. Inside you see a potato 1."},
        {"role": "assistant", "content": "take potato 1 from fridge 1"},
        {"role": "user",      "content": "You pick up the potato 1."},
        {"role": "assistant", "content": "go to microwave 1"},
        {"role": "user",      "content": "You arrive at microwave 1."},
        {"role": "assistant", "content": "put potato 1 in/on microwave 1"},
        {"role": "user",      "content": "Task completed! Reward: 1"},
    ]

    # 1. inject on empty bank — should return conversation unchanged
    new_conv, s = m.inject(conv)
    assert new_conv[0]["content"] == conv[0]["content"], "FAIL: first inject should not modify conv"
    print(f"[1] inject (empty bank): latency={s.latency * 1000:.1f}ms  ✓")

    # 2. update — writes JSONL files and graph
    s2 = m.update(conv, data_idx=2420)
    assert len(m._memory_bank) > 0, "FAIL: bank should have entries after update"
    assert len(m._memory_bank) == len(m._embeddings), "FAIL: bank/embeddings count mismatch"
    assert m._graph.number_of_nodes() == len(m._memory_bank), "FAIL: graph node count mismatch"
    bank_lines = len(open(m._bank_path).readlines())
    emb_lines = len(open(m._embeddings_path).readlines())
    assert bank_lines == emb_lines, f"FAIL: file line count mismatch {bank_lines} vs {emb_lines}"
    assert os.path.exists(m._graph_path), "FAIL: graph.gpickle not written"
    print(
        f"[2] update: chunks={len(m._memory_bank)}, "
        f"edges={m._graph.number_of_edges()}, "
        f"embedding_tokens={s2.embedding_tokens}  ✓"
    )

    # 3. inject with memory — should find the marker
    new_conv2, s3 = m.inject(conv)
    assert "--- Retrieved Memories ---" in new_conv2[0]["content"], "FAIL: no memories injected"
    print(f"[3] inject (with memory): embedding_tokens={s3.embedding_tokens}  latency={s3.latency * 1000:.1f}ms  ✓")
    print(f"    system prompt tail: ...{new_conv2[0]['content'][-120:]!r}")

    # 4. read_only — update is a no-op
    m_ro = GraphRagMemory(memory_dir=TEST_DIR, top_k=3, read_only=True)
    lines_before = len(open(m_ro._bank_path).readlines())
    s4 = m_ro.update(conv, data_idx=9999)
    lines_after = len(open(m_ro._bank_path).readlines())
    assert lines_before == lines_after, "FAIL: readonly should not write"
    assert s4.embedding_tokens == 0 and s4.input_tokens == 0, "FAIL: readonly stats should be zero"
    print(f"[4] readonly update: bank lines unchanged ({lines_before})  ✓")

    # 5. reload from disk — bank + embeddings + graph all consistent
    m2 = GraphRagMemory(memory_dir=TEST_DIR, top_k=3)
    assert len(m2._memory_bank) == len(m._memory_bank), "FAIL: reload bank count mismatch"
    assert len(m2._embeddings) == len(m._embeddings), "FAIL: reload embedding count mismatch"
    assert m2._graph.number_of_nodes() == m._graph.number_of_nodes(), "FAIL: reload graph node mismatch"
    print(
        f"[5] reload from disk: {len(m2._memory_bank)} chunks, "
        f"{m2._graph.number_of_nodes()} nodes, "
        f"{m2._graph.number_of_edges()} edges  ✓"
    )

    # 6. update without BATCH_MODEL — concepts degrade to [] but update still succeeds
    saved_model = os.environ.pop("BATCH_MODEL", None)
    m3 = GraphRagMemory(memory_dir=TEST_DIR, top_k=3, extract_model=None)
    s6 = m3.update(conv, data_idx=2421)
    if saved_model:
        os.environ["BATCH_MODEL"] = saved_model
    bank_entry_last = m3._memory_bank[-1]
    assert isinstance(bank_entry_last.get("concepts"), list), "FAIL: concepts should be list"
    print(
        f"[6] update without BATCH_MODEL: concepts={bank_entry_last['concepts']}, "
        f"chunks_added={len(m3._memory_bank) - len(m._memory_bank)}  ✓"
    )

    print("\nAll smoke tests passed.")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
