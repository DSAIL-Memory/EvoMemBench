"""GraphRAG memory provider for EvoMemBench.

Ported from CL-Bench's cl_bench_memory/graphrag_memory.py.
Algorithm: DashScope text-embedding-v4 embeddings + networkx concept graph.
Retrieval: cosine top-k seeds + Dijkstra-style graph walk (no LLM in retrieval path).
Concept extraction: one Ark (BATCH_MODEL) call per episode at ingest time.

Differences from CL-Bench:
- Implements BaseMemoryProvider (initialize / provide_memory / take_in_memory)
  instead of CL-Bench's Memory ABC (retrieve / extract).
- Embedding and LLM clients are constructed in initialize() from env vars so
  .env loading happens before any API calls.
- Thread safety via RLock (Phase 1 runs serially, but be safe).
- Retry logic on embedding calls (10 retries, 3s delay).
- Hybrid jieba/whitespace tokenizer carried over for _format_trajectory (same as BM25).
"""

import heapq
import json
import os
import pickle
import sys
import threading
import time
from typing import List

import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest, MemoryResponse, MemoryItem, MemoryItemType,
    MemoryStatus, MemoryType, TrajectoryData,
)
from .bm25_provider import _chunk_text, _format_trajectory  # reuse chunker + formatter


CONCEPT_EXTRACT_SYSTEM = (
    "You are a concept-extraction assistant. "
    "Given a text, extract up to {max_concepts} key concepts: topics, entities, "
    "strategies, domain terms, or task types. "
    "Respond ONLY with a JSON object with a single key \"concepts\" whose value "
    "is a list of short strings (1-5 words each). No prose, no extra keys."
)

_INJECTION_PREFIX = (
    "\n\nBelow are prior task trajectory chunks retrieved by GraphRAG "
    "(embedding similarity + concept-graph walk) from my memory of past "
    "interactions in this context. They may contain useful examples, "
    "facts, or reasoning patterns. Consider each item and decide whether "
    "it is relevant before using it.\n\n"
)


class GraphRAGProvider(BaseMemoryProvider):
    """Concept-graph-augmented dense retrieval over chunked agent trajectories.

    Each take_in_memory() call:
      1. Chunks (query + trajectory) into ≤chunk_size-token sentence-boundary segments
      2. Extracts concepts once from full episode text via LLM (Ark / BATCH_MODEL)
      3. For each chunk: embeds with DashScope text-embedding-v4, adds node to
         networkx.Graph, adds weighted edges to prior nodes:
         weight = 0.7 * cosine + 0.3 * Jaccard(episode_concepts)
      4. Persists memory_bank.jsonl, embeddings.jsonl, graph.gpickle

    provide_memory() embeds the query, picks cosine top-k as seeds, then runs a
    Dijkstra-style graph walk from those seeds (no LLM calls in retrieval).

    Storage layout:
        memory_bank.jsonl   — one JSON record per chunk (includes "concepts")
        embeddings.jsonl    — {id, text, embedding} per chunk (same order)
        graph.gpickle       — networkx.Graph snapshot (overwritten each episode)
    """

    DEFAULT_EMBED_MODEL = "text-embedding-v4"
    _EMBED_BATCH_SIZE = 10
    _EMBED_MAX_RETRIES = 10
    _EMBED_RETRY_DELAY = 3.0
    _CONCEPT_MAX_RETRIES = 10
    _CONCEPT_RETRY_DELAY = 5.0

    CHUNK_SIZE = 1024
    EDGES_THRESHOLD = 0.8
    CONCEPTS_WEIGHT = 0.3       # Jaccard weight; semantic weight = 1 - 0.3 = 0.7
    WALK_MAX_NODES = 20
    CONCEPTS_PER_NODE_MAX = 10

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.GRAPHRAG, config or {})
        self._top_k         = self.config.get("top_k", 10)
        self._return_on_in  = self.config.get("return_on_in", False)
        self._memory_dir    = self.config.get("memory_dir", "./storage/graphrag")
        self._chunk_size    = self.config.get("chunk_size", self.CHUNK_SIZE)
        self._embed_model   = self.config.get("embed_model", self.DEFAULT_EMBED_MODEL)
        self._extract_model = self.config.get("extract_model")  # None → read BATCH_MODEL at init

        self._bank_path  = os.path.join(self._memory_dir, "memory_bank.jsonl")
        self._embed_path = os.path.join(self._memory_dir, "embeddings.jsonl")
        self._graph_path = os.path.join(self._memory_dir, "graph.gpickle")

        self._memory_bank: list = []
        self._embeddings: List[np.ndarray] = []   # L2-normalised, same order as bank
        self._graph: nx.Graph = nx.Graph()
        self._embed_client = None
        self._llm_client = None
        self._lock = threading.RLock()

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            os.makedirs(self._memory_dir, exist_ok=True)

            # Embedding client — DashScope text-embedding-v4
            api_key  = self.config.get("embed_api_key")  or os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = self.config.get("embed_base_url") or os.environ.get("DASHSCOPE_BASE_URL", "")
            if not api_key:
                print("[GraphRAG] WARNING: DASHSCOPE_API_KEY not set — embedding calls will fail")
            import openai
            self._embed_client = openai.OpenAI(api_key=api_key, base_url=base_url)

            # LLM client — Ark (BATCH_MODEL) for concept extraction
            self._extract_model = self._extract_model or os.environ.get("BATCH_MODEL", "")
            if not self._extract_model:
                print("[GraphRAG] WARNING: BATCH_MODEL not set — concept extraction disabled")
            try:
                from volcenginesdkarkruntime import Ark
                self._llm_client = Ark(api_key=os.environ.get("DEEPSEEK_API_KEY", ""))
            except ImportError:
                print("[GraphRAG] WARNING: volcenginesdkarkruntime not installed — "
                      "concept extraction disabled; install 'volcengine-python-sdk[ark]'")
                self._llm_client = None

            self._load_from_disk()

            print(f"[GraphRAG] initialized (top_k={self._top_k}, "
                  f"return_on_in={self._return_on_in}, "
                  f"loaded={len(self._memory_bank)} chunks, "
                  f"graph={self._graph.number_of_nodes()} nodes / "
                  f"{self._graph.number_of_edges()} edges, "
                  f"embed_model={self._embed_model}, "
                  f"extract_model={self._extract_model!r})")
            return True
        except Exception as e:
            print(f"[GraphRAG] initialize failed: {e}")
            return False

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        raw_entries = []
        if os.path.exists(self._bank_path):
            with open(self._bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_entries.append(json.loads(line))

        if raw_entries and "chunk_text" not in raw_entries[0]:
            print(f"[GraphRAG] stale schema at {self._bank_path}; discarding "
                  f"{len(raw_entries)} entries")
            for path in (self._bank_path, self._embed_path, self._graph_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        self._memory_bank = raw_entries

        if os.path.exists(self._embed_path):
            with open(self._embed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        vec = np.array(obj["embedding"], dtype=np.float32)
                        self._embeddings.append(self._l2_normalize(vec))

        if len(self._memory_bank) != len(self._embeddings):
            print(f"[GraphRAG] bank/embedding count mismatch "
                  f"({len(self._memory_bank)} vs {len(self._embeddings)}), re-embedding...")
            docs = [e.get("chunk_text", "") for e in self._memory_bank]
            raw = self._embed_batch(docs)
            self._embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

        # Load or rebuild graph
        if os.path.exists(self._graph_path):
            try:
                with open(self._graph_path, "rb") as f:
                    self._graph = pickle.load(f)
                if self._graph.number_of_nodes() != len(self._memory_bank):
                    raise ValueError("node count mismatch")
            except Exception as e:
                print(f"[GraphRAG] graph load failed ({e}) — rebuilding...")
                self._rebuild_graph()
        elif self._memory_bank:
            self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        self._graph = nx.Graph()
        for i in range(len(self._memory_bank)):
            self._graph.add_node(i)
            concepts_i = self._memory_bank[i].get("concepts") or []
            vec_i = self._embeddings[i]
            for j in range(i):
                vec_j = self._embeddings[j]
                sim = float(vec_i @ vec_j)
                if sim > self.EDGES_THRESHOLD:
                    concepts_j = self._memory_bank[j].get("concepts") or []
                    jacc = self._jaccard(set(concepts_i), set(concepts_j))
                    w = (1.0 - self.CONCEPTS_WEIGHT) * sim + self.CONCEPTS_WEIGHT * jacc
                    self._graph.add_edge(i, j, weight=w)
        self._save_graph()

    def _save_graph(self) -> None:
        with open(self._graph_path, "wb") as f:
            pickle.dump(self._graph, f)

    def _rewrite_embeddings(self) -> None:
        with open(self._embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self._embeddings):
                entry = self._memory_bank[i]
                record = {
                    "id": f"{entry.get('task_id', str(i))}_chunk{entry.get('chunk_index', i)}",
                    "text": entry.get("chunk_text", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── embedding helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    def _embed(self, text: str) -> np.ndarray:
        for attempt in range(self._EMBED_MAX_RETRIES):
            try:
                response = self._embed_client.embeddings.create(
                    model=self._embed_model,
                    input=text,
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt < self._EMBED_MAX_RETRIES - 1:
                    print(f"[GraphRAG] embed failed (attempt {attempt + 1}): {str(e)[:100]}")
                    time.sleep(self._EMBED_RETRY_DELAY)
                else:
                    raise

    def _embed_batch(self, texts: list) -> list:
        if not texts:
            return []
        results = []
        for i in range(0, len(texts), self._EMBED_BATCH_SIZE):
            batch = texts[i: i + self._EMBED_BATCH_SIZE]
            for attempt in range(self._EMBED_MAX_RETRIES):
                try:
                    response = self._embed_client.embeddings.create(
                        model=self._embed_model,
                        input=batch,
                    )
                    for item in response.data:
                        results.append(np.array(item.embedding, dtype=np.float32))
                    break
                except Exception as e:
                    if attempt < self._EMBED_MAX_RETRIES - 1:
                        print(f"[GraphRAG] embed_batch failed (attempt {attempt + 1}): "
                              f"{str(e)[:100]}")
                        time.sleep(self._EMBED_RETRY_DELAY)
                    else:
                        raise
        return results

    # ── graph helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _add_node_and_edges(self, i: int, vec: np.ndarray, concepts: list) -> None:
        for j in range(i):
            sim = float(vec @ self._embeddings[j])
            if sim > self.EDGES_THRESHOLD:
                concepts_j = self._memory_bank[j].get("concepts") or []
                jacc = self._jaccard(set(concepts), set(concepts_j))
                w = (1.0 - self.CONCEPTS_WEIGHT) * sim + self.CONCEPTS_WEIGHT * jacc
                self._graph.add_edge(i, j, weight=w)

    # ── concept extraction ────────────────────────────────────────────────────

    def _extract_concepts(self, text: str) -> list:
        """One Ark call per episode with retry; fallback to [] on final failure."""
        if not self._llm_client or not self._extract_model:
            return []
        system_msg = CONCEPT_EXTRACT_SYSTEM.format(
            max_concepts=self.CONCEPTS_PER_NODE_MAX
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text[:4000]},
        ]
        for attempt in range(self._CONCEPT_MAX_RETRIES):
            try:
                response = self._llm_client.batch.chat.completions.create(
                    model=self._extract_model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content or ""
                parsed = json.loads(response_text)
                concepts = parsed.get("concepts", [])
                if not isinstance(concepts, list):
                    return []
                return [str(c).lower().strip() for c in concepts if c]
            except Exception as e:
                if attempt < self._CONCEPT_MAX_RETRIES - 1:
                    print(f"[GraphRAG] concept extraction failed "
                          f"(attempt {attempt + 1}): {str(e)[:120]}")
                    time.sleep(self._CONCEPT_RETRY_DELAY)
                else:
                    print(f"[GraphRAG] concept extraction failed after "
                          f"{self._CONCEPT_MAX_RETRIES} attempts, using empty concepts")
                    return []

    # ── graph walk ────────────────────────────────────────────────────────────

    def _graph_walk(self, seed_scores: dict) -> list:
        """Dijkstra-style expansion from cosine top-k seeds via graph edges."""
        visited: set = set()
        ordered: list = []
        pq: list = []

        for idx, score in seed_scores.items():
            heapq.heappush(pq, (1.0 / max(score, 1e-9), idx))

        while pq and len(ordered) < self._top_k and len(visited) < self.WALK_MAX_NODES:
            prio, idx = heapq.heappop(pq)
            if idx in visited:
                continue
            visited.add(idx)
            ordered.append(idx)
            for nbr, data in self._graph[idx].items():
                if nbr in visited:
                    continue
                heapq.heappush(pq, (prio + 1.0 / max(data.get("weight", 1e-9), 1e-9), nbr))

        return ordered

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(
                    memories=[], memory_type=self.memory_type,
                    total_count=len(self._memory_bank),
                )

            if not self._memory_bank or not self._embeddings:
                return MemoryResponse(memories=[], memory_type=self.memory_type,
                                      total_count=0)

            query_vec = self._l2_normalize(self._embed(request.query or ""))
            bank_matrix = np.stack(self._embeddings)
            scores = bank_matrix @ query_vec

            k = min(self._top_k, len(self._memory_bank))
            top_indices = np.argsort(scores)[::-1][:k]
            seed_scores = {int(i): float(scores[i]) for i in top_indices}

            if self._graph.number_of_edges() > 0:
                walk_order = self._graph_walk(seed_scores)
            else:
                walk_order = list(seed_scores.keys())

            items: List[MemoryItem] = []
            for rank, idx in enumerate(walk_order[:self._top_k], start=1):
                entry = self._memory_bank[int(idx)]
                chunk_text = entry.get("chunk_text", "")
                if not chunk_text:
                    continue
                items.append(MemoryItem(
                    id=str(int(idx)),
                    content=f"Memory {rank}:\n{chunk_text}",
                    metadata={
                        "task_id": entry.get("task_id", ""),
                        "chunk_index": entry.get("chunk_index", 0),
                        "concepts": entry.get("concepts", []),
                    },
                    score=float(scores[int(idx)] * 100.0),
                    type=MemoryItemType.TEXT,
                ))

            return MemoryResponse(
                memories=items, memory_type=self.memory_type,
                total_count=len(self._memory_bank),
            )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            try:
                content = _format_trajectory(trajectory_data)
                full_text = f"{trajectory_data.query}\n\n{content}"
                chunks = _chunk_text(full_text, self._chunk_size)
                task_id = (trajectory_data.metadata or {}).get("task_id", "")

                concepts = self._extract_concepts(full_text)
                print(f"[GraphRAG] task={task_id!r}: {len(chunks)} chunks, "
                      f"{len(concepts)} concepts")

                embeddings_raw = self._embed_batch(chunks)

                with open(self._bank_path, "a", encoding="utf-8") as fb, \
                     open(self._embed_path, "a", encoding="utf-8") as fe:
                    for chunk_index, (chunk_txt, raw_vec) in enumerate(
                            zip(chunks, embeddings_raw)):
                        entry = {
                            "task_id": task_id,
                            "chunk_index": chunk_index,
                            "chunk_text": chunk_txt,
                            "context_category": "",
                            "sub_category": "",
                            "concepts": concepts,
                        }
                        embedding = self._l2_normalize(raw_vec)
                        i = len(self._memory_bank)
                        self._memory_bank.append(entry)
                        self._embeddings.append(embedding)
                        self._graph.add_node(i)
                        self._add_node_and_edges(i, embedding, concepts)

                        fb.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fe.write(json.dumps({
                            "id": f"{task_id}_chunk{chunk_index}",
                            "text": chunk_txt,
                            "embedding": embedding.tolist(),
                        }, ensure_ascii=False) + "\n")

                self._save_graph()

                return True, (f"added {len(chunks)} chunks "
                              f"(total={len(self._memory_bank)}, "
                              f"concepts={len(concepts)}, "
                              f"edges={self._graph.number_of_edges()})")
            except Exception as e:
                return False, f"take_in_memory error: {e}"


if __name__ == "__main__":
    """Standalone smoke test — requires DASHSCOPE_API_KEY and BATCH_MODEL/DEEPSEEK_API_KEY in env."""
    import shutil

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

    TEST_DIR = "/tmp/graphrag_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    provider = GraphRAGProvider({"memory_dir": TEST_DIR, "top_k": 3})
    assert provider.initialize(), "initialize() failed"

    td = TrajectoryData(
        query="What is the capital of France?",
        trajectory=[
            {"role": "assistant", "content": "I will search for this."},
            {"role": "tool", "content": "Paris is the capital of France."},
            {"role": "assistant", "content": "Based on my search, Paris is the capital."},
        ],
        result="Paris",
        metadata={"task_id": "smoke_001", "is_correct": True},
    )
    ok, msg = provider.take_in_memory(td)
    print(f"take_in_memory: {ok} — {msg}")
    assert ok, f"take_in_memory failed: {msg}"

    td2 = TrajectoryData(
        query="What is the capital of Germany?",
        trajectory=[
            {"role": "assistant", "content": "I will look up Germany's capital."},
            {"role": "tool", "content": "Berlin is the capital of Germany."},
        ],
        result="Berlin",
        metadata={"task_id": "smoke_002", "is_correct": True},
    )
    ok2, msg2 = provider.take_in_memory(td2)
    print(f"take_in_memory 2: {ok2} — {msg2}")

    print(f"Graph: {provider._graph.number_of_nodes()} nodes, "
          f"{provider._graph.number_of_edges()} edges")
    assert provider._graph.number_of_nodes() > 0, "graph has no nodes"

    req = MemoryRequest(query="capital France Paris", context="", status=MemoryStatus.BEGIN)
    resp = provider.provide_memory(req)
    print(f"retrieve: {len(resp.memories)} hits")
    for m in resp.memories:
        print(f"  score={m.score:.2f}  {m.content[:80]!r}")
    assert len(resp.memories) > 0, "no hits on non-empty bank"

    req_in = MemoryRequest(query="capital", context="", status=MemoryStatus.IN)
    resp_in = provider.provide_memory(req_in)
    assert len(resp_in.memories) == 0, "IN status should return empty"
    print("[OK] MemoryStatus.IN short-circuit works")

    provider2 = GraphRAGProvider({"memory_dir": TEST_DIR, "top_k": 3})
    assert provider2.initialize()
    assert len(provider2._memory_bank) == len(provider._memory_bank)
    assert provider2._graph.number_of_nodes() == provider._graph.number_of_nodes()
    print(f"[OK] Persistence reload: {len(provider2._memory_bank)} chunks, "
          f"{provider2._graph.number_of_nodes()} nodes")

    print("\nAll smoke tests passed.")
