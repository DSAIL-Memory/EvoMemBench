"""AgentWorkflowMemory — BFCL adapter for AgentWorkflowMemoryProvider.

Process-scoped: learned workflow patterns accumulate across samples.
Each sample's trajectory is buffered per-thread; at drain_usage() time the
trajectory is abstracted into a reusable workflow via LLM and stored.

Embeddings use DashScope text-embedding-v4 via the OpenAI-compatible endpoint
(same as mem0).

Based on EvolveLab/providers/agent_workflow_memory_provider.py.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.metrics import MemoryUsageLog

_UTILIZE_PREFIX = "\n\nRelevant Workflow Patterns:\n"
_EMBED_MODEL = "text-embedding-v4"
_EMBED_DIMS = 1024
_EMBED_BUDGET_CHARS = 8000 * 4


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(t in name for t in ("ratelimit", "timeout", "apiconnection", "apierror")):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in ("rate limit", "timeout", "timed out", "connection", "503", "overloaded"))


_embed_retry = retry(
    wait=wait_fixed(2) + wait_random(0, 1),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


class _DashscopeEmbedClient:
    """Standalone DashScope text-embedding-v4 client (OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str, base_url: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @_embed_retry
    def _create(self, texts: List[str]) -> tuple:
        t0 = time.time()
        resp = self._client.embeddings.create(
            model=_EMBED_MODEL,
            input=texts,
            encoding_format="float",
            dimensions=_EMBED_DIMS,
        )
        return resp, time.time() - t0

    def embed(self, text: str) -> np.ndarray:
        text = text.replace("\n", " ")
        if len(text) > _EMBED_BUDGET_CHARS:
            text = text[-_EMBED_BUDGET_CHARS:]
        resp, dt = self._create([text])
        metrics.record_embed_call(
            total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
            latency_s=dt,
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return (N, dims) float32 array."""
        MAX_BATCH = 100
        cleaned = [t.replace("\n", " ")[-_EMBED_BUDGET_CHARS:] for t in texts]
        all_embs: List[List[float]] = []
        for i in range(0, len(cleaned), MAX_BATCH):
            chunk = cleaned[i : i + MAX_BATCH]
            resp, dt = self._create(chunk)
            metrics.record_embed_call(
                total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
                latency_s=dt,
            )
            all_embs.extend(
                item.embedding for item in sorted(resp.data, key=lambda x: x.index)
            )
        return np.array(all_embs, dtype=np.float32)


class _ArkLLMCallable:
    def __init__(self, api_key: str, model: str):
        from bfcl_eval.memory.ark_client import ArkBatchClient

        self._client = ArkBatchClient(api_key)
        self._model = model

    def __call__(self, messages: list) -> object:
        normalized = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            normalized.append({"role": msg.get("role", "user"), "content": content})
        resp, dt = self._client.chat_completions(self._model, normalized)
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        content = getattr(resp.choices[0].message, "content", "") or ""
        return type("_Resp", (), {"content": content})()


def _cosine_sim(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    nq = np.linalg.norm(query) or 1e-10
    nd = np.linalg.norm(docs, axis=1)
    nd[nd == 0] = 1e-10
    return np.dot(docs, query) / (nd * nq)


class AgentWorkflowMemory(Memory):
    """Semantic-search memory that stores generalized workflows from past executions.

    Embeddings: DashScope text-embedding-v4 (same API as mem0).

    Storage layout under storage_dir/:
        agent_workflow.json  — memory items with cached embedding vectors
    """

    def __init__(
        self,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        top_k: int = 3,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ):
        super().__init__(memory_type="agent_workflow", clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self._storage_dir / "agent_workflow.json"
        self._top_k = top_k
        self._lock = threading.RLock()
        self._tls = threading.local()

        self._llm = _ArkLLMCallable(ark_api_key, ark_model)
        self._embedder = _DashscopeEmbedClient(dashscope_api_key, dashscope_base_url)

        if clear and self._store_path.exists():
            self._store_path.unlink()

        self._items: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._load_store()

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #

    def _load_store(self) -> None:
        if not self._store_path.exists():
            return
        try:
            with open(self._store_path, encoding="utf-8") as f:
                data = json.load(f)
            self._items = data.get("items", [])
            embs = data.get("embeddings", [])
            if embs and len(embs) == len(self._items):
                self._embeddings = np.array(embs, dtype=np.float32)
        except Exception:
            self._items = []
            self._embeddings = None

    def _save_store(self) -> None:
        data = {
            "items": self._items,
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
        }
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def _search(self, query: str) -> List[str]:
        if not self._items or self._embeddings is None:
            return []
        qemb = self._embedder.embed(query)
        scores = _cosine_sim(qemb, self._embeddings)
        if scores.size == 0:
            return []
        top_k = min(self._top_k, len(self._items))
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [self._items[i] for i in top_ids if scores[i] > 0]

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def _induce_workflow(self, query: str, trajectory: list) -> Optional[str]:
        """Use LLM to generalize a specific execution into a reusable workflow."""
        traj_text = "\n".join(
            f"[{m.get('role', '')}]: {str(m.get('content', ''))[:300]}"
            for m in trajectory
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        )[:2500]

        prompt = (
            f"Extract a GENERIC, REUSABLE workflow from this task execution.\n\n"
            f"Task: {query}\n\nExecution:\n{traj_text}\n\n"
            f"Guidelines:\n"
            f"1. Replace specific values with descriptive variable names.\n"
            f"2. Keep logical steps and tool names invariant.\n"
            f"3. Output strictly valid JSON.\n\n"
            f'{{"workflow": "concise reusable workflow summary (under 200 words)"}}'
        )
        try:
            resp = self._llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            text = getattr(resp, "content", "").strip()
            cleaned = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            return result.get("workflow")
        except Exception:
            return None

    def _add_item(self, text: str) -> None:
        if text in self._items:
            return
        self._items.append(text)
        new_emb = self._embedder.embed(text)
        if self._embeddings is None:
            self._embeddings = new_emb.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        self._save_store()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def begin_sample(self) -> None:
        super().begin_sample()
        self._tls.trajectory = []
        self._tls.query = ""

    def utilize(self, user_text: str) -> str:
        try:
            with self._lock:
                results = self._search(user_text)

            u = getattr(metrics._TLS, "usage", None)
            if u is not None:
                u.n_utilize += 1

            if not results:
                return ""
            lines = "\n".join(f"- {r}" for r in results)
            return _UTILIZE_PREFIX + lines
        except Exception:
            return ""

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        traj = getattr(self._tls, "trajectory", [])
        traj.extend(trajectory)
        self._tls.trajectory = traj

        if not getattr(self._tls, "query", ""):
            for msg in trajectory:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    self._tls.query = str(content)[:500]
                    break

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    def drain_usage(self) -> MemoryUsageLog:
        trajectory = getattr(self._tls, "trajectory", [])
        query = getattr(self._tls, "query", "")

        if trajectory and query and not self.readonly:
            try:
                workflow = self._induce_workflow(query, trajectory)
                text = f"Query: {query}\nWorkflow: {workflow}" if workflow else f"Query: {query}"
                with self._lock:
                    self._add_item(text)
            except Exception:
                pass

        self._tls.trajectory = []
        self._tls.query = ""
        return metrics.drain()

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "AgentWorkflowMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
