"""Dense retrieval memory provider for EvoMemBench.

Ported from CL-Bench's cl_bench_memory/qwen3_embedding_memory.py.
Algorithm: numpy cosine similarity over DashScope text-embedding-v4 vectors.

Differences from CL-Bench:
- Implements BaseMemoryProvider (initialize / provide_memory / take_in_memory)
  instead of CL-Bench's Memory ABC (retrieve / extract).
- Embedding client is constructed in initialize() from env vars so .env loading
  happens before any API calls.
- Thread safety via RLock (Phase 1 runs serially, but be safe).
- Hybrid jieba/whitespace tokenizer carried over for _format_trajectory (same as BM25).
"""

import json
import os
import sys
import threading
import time
from typing import List

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest, MemoryResponse, MemoryItem, MemoryItemType,
    MemoryStatus, MemoryType, TrajectoryData,
)
from .bm25_provider import _chunk_text, _format_trajectory  # reuse chunker + formatter


_INJECTION_PREFIX = (
    "\n\nBelow are prior task trajectory chunks retrieved by dense embedding "
    "semantic similarity from my memory of past interactions in this context. "
    "They may contain useful examples, facts, or reasoning patterns. "
    "Consider each item and decide whether it is relevant before using it.\n\n"
)


class Qwen3EmbeddingProvider(BaseMemoryProvider):
    """Dense retrieval over chunked agent trajectories using DashScope embeddings.

    Each take_in_memory() call chunks the trajectory into ≤1024-token
    sentence-boundary segments, embeds each chunk, and appends both the text
    and embedding vector to disk.

    provide_memory() embeds the query and returns top-k chunks by cosine
    similarity (numpy dot product on L2-normalised vectors).

    Storage layout:
        memory_bank.jsonl   — one JSON record per chunk (same schema as BM25)
        embeddings.jsonl    — {id, text, embedding} per chunk (same order)
    """

    DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
    _EMBED_BATCH_SIZE = 10
    _EMBED_MAX_RETRIES = 10
    _EMBED_RETRY_DELAY = 3.0

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.QWEN3_EMBEDDING, config or {})
        self._top_k        = self.config.get("top_k", 10)
        self._return_on_in = self.config.get("return_on_in", False)
        self._memory_dir   = self.config.get("memory_dir", "./storage/qwen3_embedding")
        self._chunk_size   = self.config.get("chunk_size", 1024)
        self._embed_model  = self.config.get("embed_model", self.DEFAULT_EMBED_MODEL)

        self._bank_path  = os.path.join(self._memory_dir, "memory_bank.jsonl")
        self._embed_path = os.path.join(self._memory_dir, "embeddings.jsonl")

        self._memory_bank: list = []
        self._embeddings: List[np.ndarray] = []  # L2-normalised
        self._embed_client = None
        self._lock = threading.RLock()

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            os.makedirs(self._memory_dir, exist_ok=True)

            api_key  = self.config.get("embed_api_key") or os.environ.get("QWEN3_EMBED_API_KEY", "")
            base_url = self.config.get("embed_base_url") or os.environ.get("QWEN3_EMBED_BASE_URL", "")

            if not api_key:
                print("[Qwen3Emb] WARNING: QWEN3_EMBED_API_KEY not set — embedding calls will fail")

            import openai
            self._embed_client = openai.OpenAI(api_key=api_key, base_url=base_url)

            self._load_from_disk()

            print(f"[Qwen3Emb] initialized (top_k={self._top_k}, "
                  f"return_on_in={self._return_on_in}, "
                  f"loaded={len(self._memory_bank)} chunks, "
                  f"embed_model={self._embed_model})")
            return True
        except Exception as e:
            print(f"[Qwen3Emb] initialize failed: {e}")
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
            print(f"[Qwen3Emb] stale schema at {self._bank_path}; discarding "
                  f"{len(raw_entries)} entries")
            for path in (self._bank_path, self._embed_path):
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
            print(f"[Qwen3Emb] bank/embedding count mismatch "
                  f"({len(self._memory_bank)} vs {len(self._embeddings)}), re-embedding...")
            docs = [e.get("chunk_text", "") for e in self._memory_bank]
            raw = self._embed_batch(docs)
            self._embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

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
                    print(f"[Qwen3Emb] embed failed (attempt {attempt + 1}): {str(e)[:100]}")
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
                        print(f"[Qwen3Emb] embed_batch failed (attempt {attempt + 1}): "
                              f"{str(e)[:100]}")
                        time.sleep(self._EMBED_RETRY_DELAY)
                    else:
                        raise
        return results

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
            scores = (bank_matrix @ query_vec) * 100.0

            k = min(self._top_k, len(self._memory_bank))
            top_indices = np.argsort(scores)[::-1][:k]

            items: List[MemoryItem] = []
            for rank, idx in enumerate(top_indices, start=1):
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
                    },
                    score=float(scores[int(idx)]),
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
                        }
                        embedding = self._l2_normalize(raw_vec)
                        self._memory_bank.append(entry)
                        self._embeddings.append(embedding)

                        fb.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fe.write(json.dumps({
                            "id": f"{task_id}_chunk{chunk_index}",
                            "text": chunk_txt,
                            "embedding": embedding.tolist(),
                        }, ensure_ascii=False) + "\n")

                return True, (f"added {len(chunks)} chunks "
                              f"(total={len(self._memory_bank)})")
            except Exception as e:
                return False, f"take_in_memory error: {e}"


if __name__ == "__main__":
    """Standalone smoke test — requires DASHSCOPE_API_KEY in env."""
    import shutil

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

    TEST_DIR = "/tmp/qwen3_emb_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    provider = Qwen3EmbeddingProvider({"memory_dir": TEST_DIR, "top_k": 3})
    assert provider.initialize(), "initialize() failed"

    td = TrajectoryData(
        query="What is the capital of France?",
        trajectory=[
            {"role": "assistant", "content": "I will search for this."},
            {"role": "tool", "content": "Paris is the capital of France."},
        ],
        result="Paris",
        metadata={"task_id": "smoke_001", "is_correct": True},
    )
    ok, msg = provider.take_in_memory(td)
    print(f"take_in_memory: {ok} — {msg}")

    req = MemoryRequest(query="capital France", context="", status=MemoryStatus.BEGIN)
    resp = provider.provide_memory(req)
    print(f"retrieve: {len(resp.memories)} hits")
    for m in resp.memories:
        print(f"  score={m.score:.2f}  {m.content[:80]!r}")

    req_in = MemoryRequest(query="capital", context="", status=MemoryStatus.IN)
    resp_in = provider.provide_memory(req_in)
    assert len(resp_in.memories) == 0, "IN status should return empty"
    print("[OK] MemoryStatus.IN short-circuit works")

    # Persistence reload
    provider2 = Qwen3EmbeddingProvider({"memory_dir": TEST_DIR, "top_k": 3})
    assert provider2.initialize()
    assert len(provider2._memory_bank) == len(provider._memory_bank)
    print(f"[OK] Persistence reload: {len(provider2._memory_bank)} chunks")

    print("\nAll smoke tests passed.")
