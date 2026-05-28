"""Dense retrieval memory backend (LLM-free baseline).

Uses an OpenAI-compatible embedding API (default: DashScope text-embedding-v4).
Retrieval is done with numpy cosine similarity.

Follows MAB's chunking approach: each trajectory is split into sentence-boundary-
aligned chunks of <= CHUNK_SIZE tokens (tiktoken), each chunk stored and indexed
separately. retrieve() returns top-k matching chunks directly for injection.
"""

import json
import os
import time

import numpy as np

from .base import Memory
from .chunking import chunk_text
from .io import append_jsonl
from .logging_utils import log


class Qwen3EmbeddingMemory(Memory):
    """Dense retrieval over chunked trajectories using dense embeddings.

    Each extract() splits the trajectory into sentence-boundary chunks (<= CHUNK_SIZE
    tokens, following MAB's approach), embeds each chunk, and stores them separately.
    retrieve() embeds the query and returns top-k matching chunks directly for injection.

    Storage layout (per context_id directory):
        memory_bank.jsonl  — one JSON entry per chunk
        embeddings.jsonl   — one {id, text, embedding} per chunk (same order)
    """

    DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
    CHUNK_SIZE = 1024  # tokens; finer granularity than MAB's 4096 for better retrieval

    INJECTION_PREFIX = (
        "\n\nBelow are prior task trajectory chunks retrieved by dense embedding "
        "semantic similarity from my memory of past interactions in this context. "
        "They may contain useful examples, facts, or reasoning patterns. "
        "Consider each item and decide whether it is relevant before using it.\n\n"
    )

    def __init__(
        self,
        client=None,
        model=None,
        memory_dir=None,
        embed_provider=None,
        embed_model=None,
        embed_client=None,
        top_k=10,
        extract_model=None,
    ):
        self.top_k = top_k
        self.memory_dir = memory_dir
        self._last_embed_usage = {}
        self.EMBED_MODEL = embed_model or self.DEFAULT_EMBED_MODEL

        if embed_client is None:
            raise ValueError(
                "Qwen3EmbeddingMemory requires an embed_client (pass --embed-provider dashscope)"
            )
        self._embed_client = embed_client

        self.memory_bank: list = []  # list of chunk entries
        self.embeddings: list = []   # list[np.ndarray], L2-normalized

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "embeddings.jsonl")

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
                f"   ⚠️ Stale Qwen3Embedding memory at {self.bank_path} "
                f"(old schema, no 'chunk_text'); discarding {len(raw_entries)} entries"
            )
            for path in (self.bank_path, self.embed_path):
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
            log(f"   Loaded {len(self.memory_bank)} Qwen3-Embedding-4B chunks from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"   ⚠️ Bank({len(self.memory_bank)}) vs embeddings"
                f"({len(self.embeddings)}) mismatch, re-embedding chunks..."
            )
            docs = [e.get("chunk_text", "") for e in self.memory_bank]
            self._last_embed_usage = {}
            raw = self._embed_batch(docs)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

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

    # ── Embedding helpers ────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    def _embed(self, text: str, max_retries: int = 3, retry_delay: float = 3.0) -> np.ndarray:
        for attempt in range(max_retries):
            try:
                response = self._embed_client.embeddings.create(
                    model=self.EMBED_MODEL,
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
            except Exception as e:
                if attempt < max_retries - 1:
                    log(f"   ⚠️ Embed failed (attempt {attempt + 1}): {str(e)[:100]}")
                    time.sleep(retry_delay)
                else:
                    raise

    def _embed_batch(self, texts: list, max_retries: int = 3, retry_delay: float = 3.0) -> list:
        if not texts:
            return []
        results = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    response = self._embed_client.embeddings.create(
                        model=self.EMBED_MODEL,
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
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        log(f"   ⚠️ Embed batch failed (attempt {attempt + 1}): {str(e)[:100]}")
                        time.sleep(retry_delay)
                    else:
                        raise
        return results

    # ── Memory ABC ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        if not self.memory_bank or not self.embeddings:
            return "", {"latency_s": 0.0, "embed_usage": {}}

        t0 = time.time()

        self._last_embed_usage = {}
        query_vec = self._l2_normalize(self._embed(query or ""))
        embed_usage = self._last_embed_usage.copy()

        bank_matrix = np.stack(self.embeddings)
        if bank_matrix.shape[1] != query_vec.shape[0]:
            log(
                f"   ⚠️ Embedding dim mismatch (stored {bank_matrix.shape[1]}d vs query {query_vec.shape[0]}d); "
                f"re-embedding {len(self.memory_bank)} chunks..."
            )
            docs = [e.get("chunk_text", "") for e in self.memory_bank]
            raw = self._embed_batch(docs)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()
            bank_matrix = np.stack(self.embeddings)
        scores = (bank_matrix @ query_vec) * 100.0

        k = min(self.top_k, len(self.memory_bank))
        top_indices = np.argsort(scores)[::-1][:k]

        items = []
        for rank, idx in enumerate(top_indices, start=1):
            text = self.memory_bank[int(idx)].get("chunk_text", "")
            if text:
                items.append(f"Memory {rank}:\n{text}")

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
        if not items:
            return "", stats
        return self.INJECTION_PREFIX + "\n\n".join(items), stats

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        t0 = time.time()

        # Follow MAB: combine query + trajectory, then split into sentence-boundary chunks
        full_text = f"{query}\n\n{content}" if query else (content or "")
        chunks = chunk_text(full_text, chunk_size=self.CHUNK_SIZE)

        self._last_embed_usage = {}
        embeddings_raw = self._embed_batch(chunks)
        embed_usage = self._last_embed_usage.copy()

        for chunk_index, (chunk, raw_vec) in enumerate(zip(chunks, embeddings_raw)):
            entry = {
                "task_id": task_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "context_category": context_category or "",
                "sub_category": sub_category or "",
            }
            embedding = self._l2_normalize(raw_vec)
            self.memory_bank.append(entry)
            self.embeddings.append(embedding)

            append_jsonl(entry, self.bank_path)
            append_jsonl(
                {
                    "id": f"{task_id}_chunk{chunk_index}",
                    "text": chunk,
                    "embedding": embedding.tolist(),
                },
                self.embed_path,
            )

        return {
            "latency_s": time.time() - t0,
            "llm_usage": {},
            "embed_usage": embed_usage,
            "verdict": f"added {len(chunks)} chunks",
        }
