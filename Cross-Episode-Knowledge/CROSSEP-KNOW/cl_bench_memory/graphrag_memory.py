"""GraphRAG memory backend: concept-graph-enhanced retrieval over trajectories.

Ported from MemoryAgentBench/methods/graph_rag.py (979 lines) into an
incremental, numpy+networkx implementation that fits CL-bench's streaming
Memory.extract/retrieve interface.

Key differences from MAB:
  - Incremental graph build per episode (MAB rebuilds the whole graph once per context_id)
  - Node content = sentence-boundary chunks (CHUNK_SIZE=1024 tokens, same as BM25/Embedding)
  - Concepts extracted once per episode, shared across all its chunks
  - No LangChain / FAISS / spaCy / NLTK: openai SDK + numpy + networkx only
  - LLM-only concept extraction (MAB also uses spaCy NER)
  - Zero-LLM retrieve: embedding cosine seeds + graph walk (MAB calls LLM per walk step)

Storage layout (per context_id directory):
    memory_bank.jsonl  — one entry per chunk (includes "concepts" list)
    embeddings.jsonl   — one {id, text, embedding} per chunk (same order)
    graph.gpickle      — networkx.Graph snapshot (overwritten each extract)
"""

import heapq
import json
import os
import pickle
import time

import networkx as nx
import numpy as np

from .api_client import call_openai_api
from .base import Memory
from .chunking import chunk_text
from .io import append_jsonl
from .logging_utils import log

# ── Prompts ───────────────────────────────────────────────────────────────────

CONCEPT_EXTRACT_SYSTEM = (
    "You are a concept-extraction assistant. "
    "Given a text, extract up to {max_concepts} key concepts: topics, entities, "
    "strategies, domain terms, or task types. "
    "Respond ONLY with a JSON object with a single key \"concepts\" whose value "
    "is a list of short strings (1-5 words each). No prose, no extra keys."
)

INJECTION_PREFIX = (
    "\n\nBelow are prior task trajectory chunks retrieved by GraphRAG "
    "(embedding similarity + concept-graph walk) from my memory of past "
    "interactions in this context. They may contain useful examples, "
    "facts, or reasoning patterns. Consider each item and decide whether "
    "it is relevant before using it.\n\n"
)


class GraphRAGMemory(Memory):
    """Concept-graph-augmented dense retrieval over chunked trajectories.

    Each extract() call:
      1. Splits (query + trajectory) into sentence-boundary chunks (<=CHUNK_SIZE tokens)
      2. Extracts concepts once from the full episode text via LLM
      3. For each chunk: embed with text-embedding-v4, add node to networkx graph,
         add weighted edges to prior nodes (0.7*cosine + 0.3*Jaccard(concepts))
      4. Persists memory_bank.jsonl, embeddings.jsonl, graph.gpickle

    retrieve() per episode:
      1. Embeds query
      2. Cosine scores against all chunk nodes → top-k seeds
      3. Dijkstra-style graph walk from seeds (no LLM calls)
      4. Returns top-k chunk texts under INJECTION_PREFIX
    """

    CHUNK_SIZE = 1024              # tokens, same as BM25Memory / Qwen3EmbeddingMemory
    EDGES_THRESHOLD = 0.8
    CONCEPTS_WEIGHT = 0.3          # weight on Jaccard; semantic weight = 1 - CONCEPTS_WEIGHT
    WALK_MAX_NODES = 20
    CONCEPTS_PER_NODE_MAX = 10

    def __init__(
        self,
        client=None,
        model=None,
        memory_dir=None,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=10,
        extract_model=None,
    ):
        self.client = client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.embed_provider = embed_provider
        self.top_k = top_k
        self.memory_dir = memory_dir
        self._last_embed_usage: dict = {}

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "embeddings.jsonl")
        self.graph_path = os.path.join(memory_dir, "graph.gpickle")
        self.meta_path = os.path.join(memory_dir, "embed_meta.json")

        self.memory_bank: list = []
        self.embeddings: list = []   # list[np.ndarray], L2-normalized, same order as bank
        self.graph: nx.Graph = nx.Graph()

        self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        raw_entries = []
        if os.path.exists(self.bank_path):
            with open(self.bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_entries.append(json.loads(line))

        if raw_entries and "chunk_text" not in raw_entries[0]:
            log(
                f"   ⚠️ Stale GraphRAG memory at {self.bank_path} "
                f"(old schema, no 'chunk_text'); discarding {len(raw_entries)} entries"
            )
            for path in (self.bank_path, self.embed_path, self.graph_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        self.memory_bank = raw_entries

        if os.path.exists(self.embed_path):
            with open(self.embed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        vec = np.array(obj["embedding"], dtype=np.float32)
                        self.embeddings.append(self._l2_normalize(vec))

        if self.memory_bank:
            log(f"   Loaded {len(self.memory_bank)} GraphRAG chunks from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"   ⚠️ bank({len(self.memory_bank)}) vs embeddings"
                f"({len(self.embeddings)}) mismatch — re-embedding..."
            )
            docs = [e.get("chunk_text", "") for e in self.memory_bank]
            raw = self._embed_batch(docs)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

        # Detect embed-model change or cross-backend contamination
        # Re-embed when: meta absent (unknown source) OR meta present but model mismatches.
        if self.embeddings:
            stored_model = None
            if os.path.exists(self.meta_path):
                with open(self.meta_path) as _mf:
                    stored_model = json.load(_mf).get("embed_model")
            if stored_model != self.embed_model:
                log(
                    f"   ⚠️ Embed model mismatch: stored={stored_model!r}, "
                    f"current={self.embed_model!r} — re-embedding "
                    f"{len(self.memory_bank)} chunks..."
                )
                docs = [e.get("chunk_text", "") for e in self.memory_bank]
                raw = self._embed_batch(docs)
                self.embeddings = [self._l2_normalize(v) for v in raw]
                self._rewrite_embeddings()
                self._rebuild_graph()

        # Load or rebuild graph
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                if self.graph.number_of_nodes() != len(self.memory_bank):
                    raise ValueError("node count mismatch")
            except Exception as e:
                log(f"   ⚠️ Graph load failed ({e}) — rebuilding...")
                self._rebuild_graph()
        elif self.memory_bank:
            self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        self.graph = nx.Graph()
        for i in range(len(self.memory_bank)):
            self.graph.add_node(i)
            concepts_i = self.memory_bank[i].get("concepts") or []
            vec_i = self.embeddings[i]
            for j in range(i):
                vec_j = self.embeddings[j]
                sim = float(vec_i @ vec_j)
                if sim > self.EDGES_THRESHOLD:
                    concepts_j = self.memory_bank[j].get("concepts") or []
                    jacc = self._jaccard(set(concepts_i), set(concepts_j))
                    w = (1.0 - self.CONCEPTS_WEIGHT) * sim + self.CONCEPTS_WEIGHT * jacc
                    self.graph.add_edge(i, j, weight=w)
        self._save_graph()

    def _save_graph(self) -> None:
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def _rewrite_embeddings(self) -> None:
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                entry = self.memory_bank[i]
                record = {
                    "id": f"{entry.get('task_id', str(i))}_chunk{entry.get('chunk_index', i)}",
                    "text": entry.get("chunk_text", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"embed_model": self.embed_model}, f)

    # ── Embedding helpers ────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    def _embed(self, text: str) -> np.ndarray:
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        if response.usage:
            self._last_embed_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        else:
            self._last_embed_usage = {}
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_batch(self, texts: list) -> list:
        if not texts:
            return []
        results = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.embed_client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            if response.usage:
                self._last_embed_usage["prompt_tokens"] = (
                    self._last_embed_usage.get("prompt_tokens", 0)
                    + (response.usage.prompt_tokens or 0)
                )
                self._last_embed_usage["total_tokens"] = (
                    self._last_embed_usage.get("total_tokens", 0)
                    + (response.usage.total_tokens or 0)
                )
            for item in response.data:
                results.append(np.array(item.embedding, dtype=np.float32))
        return results

    # ── Graph helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _add_node_and_edges(self, i: int, vec: np.ndarray, concepts: list) -> None:
        """Add node i and edges to all prior nodes whose cosine similarity exceeds threshold."""
        for j in range(i):
            sim = float(vec @ self.embeddings[j])
            if sim > self.EDGES_THRESHOLD:
                concepts_j = self.memory_bank[j].get("concepts") or []
                jacc = self._jaccard(set(concepts), set(concepts_j))
                w = (1.0 - self.CONCEPTS_WEIGHT) * sim + self.CONCEPTS_WEIGHT * jacc
                self.graph.add_edge(i, j, weight=w)

    # ── LLM concept extraction ───────────────────────────────────────────────

    def _extract_concepts(self, text: str) -> tuple:
        """Call LLM to extract concepts from full episode text; return (list[str], llm_usage).

        Concepts are extracted once per episode and shared across all its chunks.
        Truncate to 4000 chars to fit within LLM context budgets.
        """
        system_msg = CONCEPT_EXTRACT_SYSTEM.format(max_concepts=self.CONCEPTS_PER_NODE_MAX)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text[:4000]},
        ]
        response_text, error, llm_usage = call_openai_api(
            self.client,
            messages,
            self.extract_model,
            temperature=0,
            response_format={"type": "json_object"},
        )
        if error or not response_text:
            log(f"   ⚠️ Concept extraction failed: {error}")
            return [], llm_usage

        try:
            parsed = json.loads(response_text)
            concepts = parsed.get("concepts", [])
            if not isinstance(concepts, list):
                return [], llm_usage
            return [str(c).lower().strip() for c in concepts if c], llm_usage
        except (json.JSONDecodeError, Exception) as e:
            log(f"   ⚠️ Concept JSON parse error: {e}")
            return [], llm_usage

    # ── Graph walk (Dijkstra-style) ───────────────────────────────────────────

    def _graph_walk(self, seed_scores: dict) -> list:
        """Expand from seeds following graph edges by weight; return node indices in visit order."""
        visited: set = set()
        ordered: list = []
        pq: list = []

        for idx, score in seed_scores.items():
            heapq.heappush(pq, (1.0 / max(score, 1e-9), idx))

        while pq and len(ordered) < self.top_k and len(visited) < self.WALK_MAX_NODES:
            prio, idx = heapq.heappop(pq)
            if idx in visited:
                continue
            visited.add(idx)
            ordered.append(idx)
            for nbr, data in self.graph[idx].items():
                if nbr in visited:
                    continue
                heapq.heappush(pq, (prio + 1.0 / max(data.get("weight", 1e-9), 1e-9), nbr))

        return ordered

    # ── Memory ABC ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        if not self.memory_bank or not self.embeddings:
            return "", {"latency_s": 0.0, "embed_usage": {}}

        t0 = time.time()

        self._last_embed_usage = {}
        q_vec = self._l2_normalize(self._embed(query or ""))
        embed_usage = self._last_embed_usage.copy()

        M = np.stack(self.embeddings)
        scores = M @ q_vec

        k = min(self.top_k, len(self.memory_bank))
        seed_idx = np.argsort(scores)[::-1][:k]
        seed_scores = {int(i): float(scores[i]) for i in seed_idx}

        if self.graph.number_of_edges() > 0:
            walk_order = self._graph_walk(seed_scores)
        else:
            walk_order = list(seed_scores.keys())

        items = []
        for rank, idx in enumerate(walk_order[: self.top_k], start=1):
            txt = self.memory_bank[int(idx)].get("chunk_text", "")
            if txt:
                items.append(f"Memory {rank}:\n{txt}")

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
        if not items:
            return "", stats
        return INJECTION_PREFIX + "\n\n".join(items), stats

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        t0 = time.time()

        full_text = f"{query}\n\n{content}" if query else (content or "")
        chunks = chunk_text(full_text, chunk_size=self.CHUNK_SIZE)

        # Extract concepts once from full episode text, shared across all chunks
        concepts, llm_usage = self._extract_concepts(full_text)
        log(f"   🔗 GraphRAG concepts ({len(concepts)}): {concepts[:5]}")

        self._last_embed_usage = {}
        embeddings_raw = self._embed_batch(chunks)
        embed_usage = self._last_embed_usage.copy()

        for chunk_index, (chunk_txt, raw_vec) in enumerate(zip(chunks, embeddings_raw)):
            entry = {
                "task_id": task_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_txt,
                "context_category": context_category or "",
                "sub_category": sub_category or "",
                "concepts": concepts,
            }
            vec = self._l2_normalize(raw_vec)
            i = len(self.memory_bank)
            self.memory_bank.append(entry)
            self.embeddings.append(vec)
            self.graph.add_node(i)
            self._add_node_and_edges(i, vec, concepts)

            append_jsonl(entry, self.bank_path)
            append_jsonl(
                {
                    "id": f"{task_id}_chunk{chunk_index}",
                    "text": chunk_txt,
                    "embedding": vec.tolist(),
                },
                self.embed_path,
            )

        self._save_graph()
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"embed_model": self.embed_model}, f)

        total_nodes = len(self.memory_bank)
        log(f"   🔗 GraphRAG added {len(chunks)} chunks (total nodes: {total_nodes})")

        return {
            "latency_s": time.time() - t0,
            "llm_usage": llm_usage,
            "embed_usage": embed_usage,
            "verdict": f"added {len(chunks)} chunks",
        }
