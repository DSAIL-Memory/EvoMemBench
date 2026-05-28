"""MemoryosMemory — MemoryOS-backed Memory interface.

All LLM calls inside MemoryOS are monkey-patched to route through the Ark
batch endpoint via ArkBatchOpenAIShim (which implements the same 4-method
surface that MemoryOS's OpenAIClient exposes).

Embeddings use DashScope text-embedding-v4; get_embedding in memoryos.utils
is monkey-patched to record token usage in metrics._TLS.

Patch functions are idempotent (guarded by a _done sentinel).
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.mem0_backend import _truncate_query_to_embed_limit

_UTILIZE_PREFIX = "\n\nRelevant memories from prior episodes:\n"


# ─────────────────────────────────────────────────────────────────────────────
# 1. ArkBatchOpenAIShim — drop-in replacement for memoryos.utils.OpenAIClient
# ─────────────────────────────────────────────────────────────────────────────

class ArkBatchOpenAIShim:
    """Implements the four methods memoryos calls on its OpenAIClient.

    chat_completion / chat_completion_async / batch_chat_completion / shutdown

    Token metrics are accumulated on _origin_usage (captured from the parent
    worker thread's TLS at construction time) so that LLM calls spawned by
    MemoryOS's internal ThreadPoolExecutor still land in the right counter.
    """

    _class_api_key: Optional[str] = None

    @classmethod
    def _set_api_key(cls, key: str) -> None:
        cls._class_api_key = key

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, max_workers: int = 5):
        from bfcl_eval.memory.ark_client import ArkBatchClient
        resolved_key = api_key or self._class_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._ark = ArkBatchClient(api_key=resolved_key)
        # Capture parent-thread usage so child threads can still record metrics.
        self._origin_usage = getattr(metrics._TLS, "usage", None)

    def chat_completion(self, model: str, messages, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        resp, dt = self._ark.chat_completions(model=model, messages=list(messages))
        u = getattr(metrics._TLS, "usage", None) or self._origin_usage
        if u is not None:
            u.latency_s += dt
            u.input_tokens += int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            u.output_tokens += int(getattr(resp.usage, "completion_tokens", 0) or 0)
            u.n_llm_calls += 1
        return (resp.choices[0].message.content or "").strip()

    def chat_completion_async(self, model: str, messages, temperature: float = 0.7, max_tokens: int = 2000):
        from concurrent.futures import Future
        f: Future = Future()
        try:
            f.set_result(self.chat_completion(model, messages, temperature, max_tokens))
        except Exception as exc:
            f.set_exception(exc)
        return f

    def batch_chat_completion(self, requests) -> list[str]:
        return [
            self.chat_completion(
                r.get("model", ""),
                r["messages"],
                r.get("temperature", 0.7),
                r.get("max_tokens", 2000),
            )
            for r in requests
        ]

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monkey-patches (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_memoryos_openai_client() -> None:
    """Replace OpenAIClient with ArkBatchOpenAIShim in all memoryos sub-modules."""
    if getattr(_patch_memoryos_openai_client, "_done", False):
        return
    import importlib
    import memoryos.utils as _u
    _u.OpenAIClient = ArkBatchOpenAIShim
    for name in ("memoryos.memoryos", "memoryos.retriever", "memoryos.mid_term", "memoryos.updater"):
        try:
            m = importlib.import_module(name)
            if hasattr(m, "OpenAIClient"):
                m.OpenAIClient = ArkBatchOpenAIShim
        except ImportError:
            pass
    _patch_memoryos_openai_client._done = True  # type: ignore[attr-defined]


def _patch_memoryos_get_embedding() -> None:
    """Wrap get_embedding for text-embedding-v4 to record metrics."""
    if getattr(_patch_memoryos_get_embedding, "_done", False):
        return
    import importlib
    import memoryos.utils as _u
    _orig = _u.get_embedding

    def _wrapped(text, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
        if model_name == "text-embedding-v4":
            import numpy as _np
            from openai import OpenAI as _OpenAI
            _api_key = kwargs.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "")
            _base_url = kwargs.get("base_url") or os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            t0 = time.time()
            _resp = _OpenAI(api_key=_api_key, base_url=_base_url).embeddings.create(
                input=[text], model=model_name
            )
            elapsed = time.time() - t0
            metrics.record_embed_call(
                total_tokens=int(getattr(_resp.usage, "total_tokens", 0) or 0),
                latency_s=elapsed,
            )
            return _np.array(_resp.data[0].embedding, dtype=_np.float32)
        return _orig(text, model_name=model_name, use_cache=use_cache, **kwargs)

    _u.get_embedding = _wrapped
    for name in (
        "memoryos.memoryos", "memoryos.retriever", "memoryos.mid_term",
        "memoryos.updater", "memoryos.short_term", "memoryos.long_term",
    ):
        try:
            m = importlib.import_module(name)
            if hasattr(m, "get_embedding"):
                m.get_embedding = _wrapped
        except ImportError:
            pass
    _patch_memoryos_get_embedding._done = True  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trajectory → user/agent pairs helper
# ─────────────────────────────────────────────────────────────────────────────

def _pairs_from_trajectory(traj: list[dict]) -> list[tuple[str, str]]:
    """Extract (user_input, agent_response) pairs from a multi-turn trajectory.

    Groups each top-level user message with the tool calls, tool results,
    and final assistant text that follow it before the next user message.
    """
    pairs: list[tuple[str, str]] = []
    cur_user: Optional[str] = None
    cur_chunks: list[str] = []

    def flush() -> None:
        if cur_user is not None:
            pairs.append((cur_user, "\n".join(cur_chunks).strip()))

    for m in traj:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "user":
            flush()
            cur_user = content
            cur_chunks = []
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name", "")
                args = (tc.get("function") or {}).get("arguments", "")
                cur_chunks.append(f"[call] {fn}({args})")
            if content:
                cur_chunks.append(f"[say] {content}")
        elif role == "tool":
            cur_chunks.append(f"[result] {content}")
        # system messages are skipped

    flush()
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4. MemoryosMemory — the Memory ABC implementation
# ─────────────────────────────────────────────────────────────────────────────

class MemoryosMemory(Memory):
    """Cross-episode memory backed by the MemoryOS library.

    Storage layout under storage_dir/:
        users/<user_id>/{short_term,mid_term,long_term_user}.json
        assistants/<assistant_id>/long_term_assistant.json

    LLM calls are routed through ArkBatchOpenAIShim (monkey-patched into
    all memoryos sub-modules before the Memoryos instance is constructed).
    Embeddings use DashScope text-embedding-v4 with usage recorded in metrics.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        embed_model: str = "text-embedding-v4",
        user_id: str = "bfcl_run",
        assistant_id: str = "bfcl_assistant",
        top_k: int = 5,
        short_term_capacity: int = 10,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        try:
            from memoryos import Memoryos
        except ImportError as exc:
            raise ImportError(
                "memoryos is required for --memory-type memoryos. "
                "Install it from EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi."
            ) from exc

        # Apply patches before constructing Memoryos (which calls OpenAIClient).
        ArkBatchOpenAIShim._set_api_key(ark_api_key)
        _patch_memoryos_openai_client()
        _patch_memoryos_get_embedding()

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        if clear:
            shutil.rmtree(storage_dir / "users", ignore_errors=True)
            shutil.rmtree(storage_dir / "assistants", ignore_errors=True)

        self._user_id = user_id
        self._assistant_id = assistant_id
        self._ark_model = ark_model
        self._dashscope_api_key = dashscope_api_key
        self._dashscope_base_url = dashscope_base_url
        self._embed_model = embed_model
        self._storage_dir = storage_dir
        self._short_term_capacity = short_term_capacity
        self._top_k = top_k
        self._lock = threading.RLock()

        self._memoryos: Memoryos = Memoryos(
            user_id=user_id,
            openai_api_key=ark_api_key,
            openai_base_url="unused",  # shim ignores base_url
            data_storage_path=str(storage_dir),
            assistant_id=assistant_id,
            short_term_capacity=short_term_capacity,
            llm_model=ark_model,
            embedding_model_name=embed_model,
            embedding_model_kwargs={
                "api_key": dashscope_api_key,
                "base_url": dashscope_base_url,
            },
        )

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        query = _truncate_query_to_embed_limit(system_prompt)
        items: list[tuple[str, str, str]] = []  # (tag, user, agent)

        # Short-term
        try:
            for qa in self._memoryos.short_term_memory.get_all()[: self._top_k]:
                items.append(("short", qa.get("user_input", ""), qa.get("agent_response", "")))
        except Exception:
            pass

        # Mid-term
        try:
            sessions = self._memoryos.mid_term_memory.search_sessions(
                query_text=query,
                segment_similarity_threshold=0.1,
                page_similarity_threshold=0.1,
                top_k_sessions=5,
            )
            for s in sessions or []:
                for pm in s.get("matched_pages", [])[: self._top_k]:
                    pd = pm.get("page_data", {})
                    items.append(("mid", pd.get("user_input", ""), pd.get("agent_response", "")))
        except Exception:
            pass

        # Long-term user knowledge
        try:
            for k in (
                self._memoryos.user_long_term_memory.search_user_knowledge(
                    query, threshold=0.01, top_k=max(1, self._top_k // 2)
                )
                or []
            ):
                items.append(("ltm", k.get("text", ""), ""))
        except Exception:
            pass

        # Deduplicate, keep first top_k
        seen: set[tuple[str, str]] = set()
        dedup: list[tuple[str, str, str]] = []
        for tag, u, a in items:
            key = (u, a)
            if key in seen:
                continue
            seen.add(key)
            dedup.append((tag, u, a))
            if len(dedup) >= self._top_k:
                break

        u_obj = getattr(metrics._TLS, "usage", None)
        if u_obj is not None:
            u_obj.n_utilize += 1

        if not dedup:
            return ""
        lines = "\n".join(
            f"- [{t}] User: {u} | Assistant: {a}" if a else f"- [{t}] {u}"
            for t, u, a in dedup
        )
        return _UTILIZE_PREFIX + lines

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        pairs = _pairs_from_trajectory(trajectory)
        with self._lock:
            for u, a in pairs:
                try:
                    self._memoryos.add_memory(
                        user_input=u,
                        agent_response=a,
                        timestamp=None,
                        meta_data=None,
                    )
                except Exception:
                    pass

        u_obj = getattr(metrics._TLS, "usage", None)
        if u_obj is not None:
            u_obj.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "MemoryosMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
