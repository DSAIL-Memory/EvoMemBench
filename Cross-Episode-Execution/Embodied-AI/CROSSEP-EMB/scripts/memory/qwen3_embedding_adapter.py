"""Dense embedding memory backend for ALFWorld evaluation (Qwen3-Embedding compatible).

Uses an OpenAI-compatible embedding API (DashScope text-embedding-v4 by default) for
chunk indexing and cosine-similarity retrieval. No LLM extraction step — this is the
pure dense-retrieval ablation of mem0, mirroring Flash-Searcher's qwen3_embedding_provider.

Persistence: two flat JSONL files per task —
  memory_bank.jsonl  : chunk records  {task_id, chunk_index, chunk_text, ...}
  embeddings.jsonl   : vector records  {id, text, embedding}

Chunking and trajectory formatting are reused directly from bm25_adapter to guarantee
identical text pre-processing across baselines.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from copy import deepcopy

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


class Qwen3EmbeddingMemory(BaseMemory):
    """Dense embedding memory backend using an OpenAI-compatible embedding API.

    Stores trajectories as chunked JSONL (memory_bank.jsonl) with their L2-normalized
    embeddings (embeddings.jsonl). Retrieves top-k matching chunks by cosine similarity
    and injects them into the system prompt under '--- Retrieved Memories ---'.

    Thread-safe via threading.RLock. Embedding HTTP calls are performed outside the lock
    to avoid blocking concurrent workers.
    """

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        memory_dir: str | None = None,
        top_k: int = 10,
        chunk_size: int = 1024,
        embed_model: str = "text-embedding-v4",
        embed_api_key: str | None = None,
        embed_base_url: str | None = None,
        embed_batch_size: int = _EMBED_BATCH_SIZE,
        embed_max_retries: int = 10,
        embed_retry_delay: float = 3.0,
        return_on_in: bool = False,
    ):
        super().__init__(memory_type="qwen3_embedding", read_only=read_only)
        if memory_dir is None:
            raise ValueError("Qwen3EmbeddingMemory requires 'memory_dir' to be set.")

        self._agent_id = agent_id
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._embed_model = embed_model
        self._embed_api_key = embed_api_key
        self._embed_base_url = embed_base_url
        self._embed_batch_size = embed_batch_size
        self._embed_max_retries = embed_max_retries
        self._embed_retry_delay = embed_retry_delay
        self._return_on_in = return_on_in

        self._memory_dir = memory_dir
        self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self._embeddings_path = os.path.join(memory_dir, "embeddings.jsonl")

        self._memory_bank: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._dirty: bool = False
        self._lock = threading.RLock()
        self._client = None

        os.makedirs(memory_dir, exist_ok=True)
        self._load_bank()

    # ── private helpers ────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            api_key = self._embed_api_key or os.environ.get("DASHSCOPE_API_KEY")
            base_url = self._embed_base_url or os.environ.get("DASHSCOPE_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError(
                    "Qwen3EmbeddingMemory requires DASHSCOPE_API_KEY + DASHSCOPE_BASE_URL "
                    "(set in environment or pass embed_api_key / embed_base_url in config)."
                )
            import openai
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts in ≤embed_batch_size sub-batches with exponential-backoff retry."""
        self._ensure_client()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._embed_batch_size):
            sub = texts[i : i + self._embed_batch_size]
            last_exc: Exception | None = None
            for attempt in range(self._embed_max_retries):
                try:
                    resp = self._client.embeddings.create(model=self._embed_model, input=sub)
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
                            f"[Qwen3:{self._agent_id}] embed retry "
                            f"{attempt + 1}/{self._embed_max_retries} in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
            if last_exc is not None:
                raise last_exc
        return out

    def _build_matrix(self) -> None:
        if not self._embeddings:
            self._matrix = None
        else:
            self._matrix = np.stack(self._embeddings, axis=0)
        self._dirty = False

    def _load_bank(self) -> None:
        bank_entries: list[dict] = []
        embed_entries: list[np.ndarray] = []

        if os.path.exists(self._bank_path):
            try:
                with open(self._bank_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            bank_entries.append(json.loads(line))
            except Exception as e:
                print(f"[Qwen3:{self._agent_id}] failed to load bank: {e}")
                return

        if not bank_entries:
            return

        if "chunk_text" not in bank_entries[0]:
            print(f"[Qwen3:{self._agent_id}] stale schema in {self._bank_path}; discarding.")
            os.remove(self._bank_path)
            if os.path.exists(self._embeddings_path):
                os.remove(self._embeddings_path)
            return

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
                print(f"[Qwen3:{self._agent_id}] failed to load embeddings: {e}")
                embed_entries = []

        if len(embed_entries) != len(bank_entries):
            print(
                f"[Qwen3:{self._agent_id}] bank/embedding count mismatch "
                f"({len(bank_entries)} vs {len(embed_entries)}); "
                f"inject will return empty until next update re-syncs them."
            )
            embed_entries = []

        self._memory_bank = bank_entries
        self._embeddings = embed_entries
        if bank_entries and embed_entries:
            self._dirty = True
        print(
            f"[Qwen3:{self._agent_id}] loaded {len(bank_entries)} chunks, "
            f"{len(embed_entries)} embeddings from disk."
        )

    # ── public interface ───────────────────────────────────────────────────────

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]
        t0 = time.time()

        # Acquire lock only to check/build the matrix and capture a snapshot
        with self._lock:
            if not self._memory_bank or not self._embeddings:
                return conversation, MemoryCallStats(latency=time.time() - t0)
            if self._dirty or self._matrix is None:
                self._build_matrix()
            if self._matrix is None:
                return conversation, MemoryCallStats(latency=time.time() - t0)
            # Capture immutable references; matrix is replaced (not mutated) on update
            matrix = self._matrix
            bank_snapshot = list(self._memory_bank)

        # Embed outside lock to avoid blocking other workers during the HTTP call
        try:
            query_vec = self._embed_batch([query])[0]
        except Exception as e:
            print(f"[Qwen3:{self._agent_id}] inject embed error: {e}")
            return conversation, MemoryCallStats(latency=time.time() - t0)

        scores = matrix @ query_vec
        k = min(self._top_k, len(bank_snapshot))
        top_indices = scores.argsort()[::-1][:k]

        hits = []
        for idx in top_indices:
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

        # Embed outside lock (slow HTTP call)
        try:
            vectors = self._embed_batch(chunks)
        except Exception as e:
            print(f"[Qwen3:{self._agent_id}] update embed error for idx={task_id}: {e}")
            return MemoryCallStats(latency=time.time() - t0)

        with self._lock:
            try:
                with (
                    open(self._bank_path, "a", encoding="utf-8") as bf,
                    open(self._embeddings_path, "a", encoding="utf-8") as ef,
                ):
                    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
                        bank_entry = {
                            "task_id": task_id,
                            "chunk_index": i,
                            "chunk_text": chunk_text,
                            "context_category": "",
                            "sub_category": "",
                        }
                        embed_entry = {
                            "id": f"{task_id}_{i}",
                            "text": chunk_text,
                            "embedding": vec.tolist(),
                        }
                        self._memory_bank.append(bank_entry)
                        self._embeddings.append(vec)
                        bf.write(json.dumps(bank_entry, ensure_ascii=False) + "\n")
                        ef.write(json.dumps(embed_entry, ensure_ascii=False) + "\n")
                self._dirty = True
                print(
                    f"[Qwen3:{self._agent_id}] idx={task_id} added {len(chunks)} chunk(s) "
                    f"(total={len(self._memory_bank)})"
                )
            except Exception as e:
                print(f"[Qwen3:{self._agent_id}] update write error: {e}")

        return MemoryCallStats(
            latency=time.time() - t0,
            embedding_tokens=total_embed_tokens,
        )

    def load_from_disk(self, memory_dir: str | None = None, **kwargs) -> None:
        with self._lock:
            if memory_dir is not None:
                self._memory_dir = memory_dir
                self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
                self._embeddings_path = os.path.join(memory_dir, "embeddings.jsonl")
                os.makedirs(memory_dir, exist_ok=True)
            self._memory_bank = []
            self._embeddings = []
            self._matrix = None
            self._dirty = False
            self._load_bank()


if __name__ == "__main__":
    import shutil

    if not (os.environ.get("DASHSCOPE_API_KEY") and os.environ.get("DASHSCOPE_BASE_URL")):
        print("[SKIP] DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL not set; skipping live API tests.")
        sys.exit(0)

    TEST_DIR = "/tmp/qwen3_emb_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("=== Qwen3-Embedding Adapter Smoke Test ===")
    print(f"memory_dir: {TEST_DIR}")

    m = Qwen3EmbeddingMemory(memory_dir=TEST_DIR, top_k=3)

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

    # 2. update — writes both JSONL files
    s2 = m.update(conv, data_idx=2420)
    assert len(m._memory_bank) > 0, "FAIL: bank should have entries after update"
    assert len(m._memory_bank) == len(m._embeddings), "FAIL: bank/embeddings count mismatch"
    bank_lines = len(open(m._bank_path).readlines())
    emb_lines = len(open(m._embeddings_path).readlines())
    assert bank_lines == emb_lines, f"FAIL: file line count mismatch {bank_lines} vs {emb_lines}"
    print(f"[2] update: chunks={len(m._memory_bank)} embedding_tokens={s2.embedding_tokens}  ✓")

    # 3. inject with memory — should find the marker
    new_conv2, s3 = m.inject(conv)
    assert "--- Retrieved Memories ---" in new_conv2[0]["content"], "FAIL: no memories injected"
    print(f"[3] inject (with memory): embedding_tokens={s3.embedding_tokens}  latency={s3.latency * 1000:.1f}ms  ✓")
    print(f"    system prompt tail: ...{new_conv2[0]['content'][-120:]!r}")

    # 4. read_only — update is a no-op
    m_ro = Qwen3EmbeddingMemory(memory_dir=TEST_DIR, top_k=3, read_only=True)
    lines_before = len(open(m_ro._bank_path).readlines())
    s4 = m_ro.update(conv, data_idx=9999)
    lines_after = len(open(m_ro._bank_path).readlines())
    assert lines_before == lines_after, "FAIL: readonly should not write"
    assert s4.embedding_tokens == 0, "FAIL: readonly stats should be zero"
    print(f"[4] readonly update: bank lines unchanged ({lines_before})  ✓")

    # 5. reload from disk
    m2 = Qwen3EmbeddingMemory(memory_dir=TEST_DIR, top_k=3)
    assert len(m2._memory_bank) == len(m._memory_bank), "FAIL: reload count mismatch"
    assert len(m2._embeddings) == len(m._embeddings), "FAIL: embedding reload count mismatch"
    print(f"[5] reload from disk: {len(m2._memory_bank)} chunk(s), {len(m2._embeddings)} embedding(s)  ✓")

    print("\nAll smoke tests passed.")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
