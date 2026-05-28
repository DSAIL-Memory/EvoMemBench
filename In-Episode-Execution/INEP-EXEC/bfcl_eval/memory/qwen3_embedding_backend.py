"""Qwen3-Embedding dense-retrieval memory backend for the BFCL cross-episode pipeline.

Uses an OpenAI-compatible embedding endpoint (DashScope by default) for chunk indexing
and cosine-similarity retrieval. No LLM extraction — this is the pure dense-retrieval
ablation of mem0, mirroring AgentGym's qwen3_embedding_adapter.py.

Persistence: two flat JSONL files per storage_dir:
  memory_bank.jsonl  : chunk records  {task_id, chunk_index, chunk_text, ...}
  embeddings.jsonl   : vector records  {id, text, embedding}

Chunking and trajectory formatting are imported directly from bm25_backend to guarantee
identical text pre-processing across baselines.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

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


class Qwen3EmbeddingMemory(Memory):
    """Dense embedding memory backend using an OpenAI-compatible embedding API.

    Stores trajectories as chunked JSONL (memory_bank.jsonl) with their L2-normalized
    embeddings (embeddings.jsonl). Retrieves top-k matching chunks by cosine similarity
    and injects them into the system prompt under the standard _UTILIZE_PREFIX marker.

    Thread-safe via threading.RLock. Embedding HTTP calls are performed outside the lock
    to avoid blocking concurrent workers during parallel inference.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        dashscope_api_key: str = "",
        dashscope_base_url: str = "",
        embed_model: str | None = None,
        top_k: int = 3,
        chunk_size: int = 1024,
        embed_batch_size: int = _EMBED_BATCH_SIZE,
        embed_max_retries: int = 10,
        embed_retry_delay: float = 3.0,
        clear: bool = False,
        readonly: bool = False,
        **unused,  # absorb ark_api_key, ark_model, mem_model, etc. from the driver
    ):
        super().__init__(clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._bank_path = self._storage_dir / "memory_bank.jsonl"
        self._embeddings_path = self._storage_dir / "embeddings.jsonl"

        # Credential priority: constructor args → DASHSCOPE_* → OPENAI_*
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
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._embed_batch_size = embed_batch_size
        self._embed_max_retries = embed_max_retries
        self._embed_retry_delay = embed_retry_delay

        self._memory_bank: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._dirty: bool = False
        self._lock = threading.RLock()
        self._client = None
        self._update_counter = 0

        if not self._embed_api_key or not self._embed_base_url:
            raise RuntimeError(
                "Qwen3EmbeddingMemory requires an embedding API key + base URL. "
                "Set OPENAI_API_KEY + OPENAI_BASE_URL (or DASHSCOPE_API_KEY + "
                "DASHSCOPE_BASE_URL) in the environment, or pass dashscope_api_key / "
                "dashscope_base_url to the constructor."
            )

        if clear:
            self._bank_path.write_text("")
            self._embeddings_path.write_text("")

        self._load_bank()

    # ── private helpers ────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            import openai
            print(
                f"[Qwen3Emb][debug] init client model={self._embed_model!r} "
                f"base_url={self._embed_base_url!r} api_key={self._embed_api_key[:8]}…{self._embed_api_key[-4:]}",
                flush=True,
            )
            self._client = openai.OpenAI(
                api_key=self._embed_api_key,
                base_url=self._embed_base_url,
                timeout=120.0,
            )

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts in ≤embed_batch_size sub-batches with retry.

        404 (model_not_found) from a relay is usually a transient upstream outage, not a
        permanent config error.  We still retry, but only twice with a short fixed delay so
        we don't waste 25+ minutes on a sustained outage.  Other errors use exponential
        backoff up to embed_max_retries.
        """
        import openai as _openai

        self._ensure_client()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._embed_batch_size):
            sub = texts[i : i + self._embed_batch_size]
            last_exc: Exception | None = None
            for attempt in range(self._embed_max_retries):
                try:
                    resp = self._client.embeddings.create(
                        model=self._embed_model, input=sub
                    )
                    for d in resp.data:
                        out.append(
                            self._l2_normalize(
                                np.asarray(d.embedding, dtype=np.float32)
                            )
                        )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    is_404 = isinstance(e, _openai.NotFoundError)
                    max_retries_for_exc = 2 if is_404 else self._embed_max_retries
                    if attempt < max_retries_for_exc - 1:
                        delay = 8.0 if is_404 else min(self._embed_retry_delay * (2 ** attempt), 30.0)
                        print(
                            f"[Qwen3Emb] embed retry "
                            f"{attempt + 1}/{max_retries_for_exc} in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        break
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

        if self._bank_path.exists():
            try:
                with open(self._bank_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            bank_entries.append(json.loads(line))
            except Exception as e:
                print(f"[Qwen3Emb] failed to load bank: {e}")
                return

        if not bank_entries:
            return

        if "chunk_text" not in bank_entries[0]:
            print(f"[Qwen3Emb] stale schema in {self._bank_path}; discarding.")
            self._bank_path.write_text("")
            self._embeddings_path.write_text("")
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
                print(f"[Qwen3Emb] failed to load embeddings: {e}")
                embed_entries = []

        if len(embed_entries) != len(bank_entries):
            print(
                f"[Qwen3Emb] bank/embedding count mismatch "
                f"({len(bank_entries)} vs {len(embed_entries)}); "
                f"retrieve will return empty until next update re-syncs."
            )
            embed_entries = []

        self._memory_bank = bank_entries
        self._embeddings = embed_entries
        if bank_entries and embed_entries:
            self._dirty = True

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
            # Capture immutable references; matrix is replaced (not mutated) on update
            matrix = self._matrix
            bank_snapshot = list(self._memory_bank)

        if not system_prompt:
            return ""

        # Embed outside lock to avoid blocking other workers during the HTTP call
        try:
            query_vec = self._embed_batch([system_prompt])[0]
        except Exception as e:
            print(f"[Qwen3Emb] utilize embed error: {e}")
            return ""

        scores = matrix @ query_vec
        k = min(self._top_k, len(bank_snapshot))
        top_indices = scores.argsort()[::-1][:k]
        hits = [
            f"- {bank_snapshot[int(i)]['chunk_text']}"
            for i in top_indices
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

        # Embed outside lock (slow HTTP call)
        try:
            vectors = self._embed_batch(chunks)
        except Exception as e:
            print(f"[Qwen3Emb] update embed error: {e}")
            raise

        with self._lock:
            task_id = str(self._update_counter)
            self._update_counter += 1
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
            except Exception as e:
                print(f"[Qwen3Emb] update write error: {e}")
                return

        elapsed = time.monotonic() - t0
        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1
            u.n_embed_calls += 1
            u.embedding_tokens += total_embed_tokens
            u.latency_s += elapsed

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "Qwen3EmbeddingMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
