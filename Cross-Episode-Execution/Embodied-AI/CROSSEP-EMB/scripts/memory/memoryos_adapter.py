"""MemoryOS memory adapter for AgentGym ALFWorld evaluation.

Code structure mirrors memos_adapter.py exactly.
MemoryOS is a three-tier hierarchical memory system (short-term / mid-term / long-term + FAISS).
Reference: ../../../EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi/

Install: pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi
"""
from __future__ import annotations

import logging
import os
import time
import threading
from copy import deepcopy
from typing import Any

from .base import BaseMemory, MemoryCallStats

logger = logging.getLogger(__name__)

_tls = threading.local()


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _coerce_content(content: Any) -> str:
    """Coerce OpenAI message content to str (handles list-of-parts for multi-modal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _tls_stats() -> MemoryCallStats | None:
    return getattr(_tls, "stats", None)


def _make_counting_client(raw_client):
    """Return a proxy around an OpenAI client that counts tokens into _tls.stats."""

    class _CompletionsProxy:
        def __init__(self, inner):
            self._inner = inner

        def create(self, *args, **kwargs):
            result = self._inner.create(*args, **kwargs)
            stats = _tls_stats()
            if stats is not None and hasattr(result, "usage") and result.usage:
                stats.input_tokens += result.usage.prompt_tokens or 0
                stats.output_tokens += result.usage.completion_tokens or 0
                if (hasattr(result.usage, "prompt_tokens_details")
                        and result.usage.prompt_tokens_details):
                    stats.cached_tokens += (
                        getattr(result.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    )
            return result

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _ChatProxy:
        def __init__(self, inner):
            self._inner = inner
            self.completions = _CompletionsProxy(inner.completions)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _ClientProxy:
        def __init__(self, inner):
            self._inner = inner
            self.chat = _ChatProxy(inner.chat)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _ClientProxy(raw_client)


class MemoryOSMemory(BaseMemory):
    """Memory backend backed by MemoryOS three-tier hierarchical memory system."""

    def __init__(
        self,
        read_only: bool = False,
        user_id: str = "alfworld",
        storage_path: str | None = None,
        short_term_capacity: int = 10,
        llm_model: str = "deepseek-v3-2-251201",
        llm: dict | None = None,
        embed: dict | None = None,
        search_limit: int = 3,
        **kwargs,
    ):
        super().__init__(memory_type="memoryos", read_only=read_only)
        self._user_id = user_id
        self._storage_path = storage_path or f"./storage/memoryos/{user_id}"
        self._short_term_capacity = short_term_capacity
        self._llm_model = llm_model
        self._llm_cfg = llm or {}
        self._embed_cfg = embed or {}
        self._search_limit = search_limit
        self._memoryos = None
        self._initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Initialize MemoryOS Memoryos instance."""
        try:
            from memoryos import Memoryos
        except ImportError as e:
            raise ImportError(
                f"memoryos is not installed: {e}\n"
                "Run: pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi"
            ) from e

        os.makedirs(self._storage_path, exist_ok=True)

        self._memoryos = Memoryos(
            user_id=self._user_id,
            openai_api_key=self._llm_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
            openai_base_url=self._llm_cfg.get("base_url") or os.getenv("DEEPSEEK_BASE_URL"),
            data_storage_path=self._storage_path,
            llm_model=self._llm_model,
            embedding_model_name=self._embed_cfg.get("model", "text-embedding-v4"),
            embedding_model_kwargs={
                "api_key": self._embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                "base_url": self._embed_cfg.get("base_url") or os.getenv("DASHSCOPE_BASE_URL"),
            },
            short_term_capacity=self._short_term_capacity,
        )
        self._install_token_hooks()

        try:
            st = self._memoryos.short_term_memory
            count = len(st.get_all()) if hasattr(st, "get_all") else "?"
        except Exception:
            count = "?"
        logger.info(
            "[memoryos] initialized (user_id=%s, search_limit=%d, storage=%s, short_term_entries=%s)",
            self._user_id, self._search_limit, self._storage_path, count,
        )

    def _install_token_hooks(self) -> None:
        """Wrap MemoryOS's OpenAIClient._create_client() to count LLM tokens into _tls.stats.

        OpenAIClient creates a fresh openai.OpenAI per call (thread-safe design).
        We intercept _create_client() to return a counting proxy each time.
        LLM is only called during add_memory() when short-term fills up; retrieve_context() is FAISS-only.
        """
        def _patch_oc(oc, label: str) -> bool:
            orig_create = getattr(oc, "_create_client", None)
            if orig_create is None:
                return False

            def _create_counting():
                return _make_counting_client(orig_create())

            oc._create_client = _create_counting
            logger.info("[memoryos] %s._create_client wrapped for token counting", label)
            return True

        patched = 0
        for owner, label in [
            (getattr(self._memoryos, "updater", None), "updater.client"),
            (getattr(self._memoryos, "mid_term_memory", None), "mid_term_memory.client"),
            (self._memoryos, "memoryos.client"),
        ]:
            oc = getattr(owner, "client", None) if owner is not None else None
            if oc is not None and _patch_oc(oc, label):
                patched += 1

        if patched == 0:
            logger.warning("[memoryos] Could not patch any OpenAIClient; LLM tokens will not be counted")

    def _run_with_stats(self, fn):
        """Run fn() while accumulating token stats. Returns (result, MemoryCallStats)."""
        stats = MemoryCallStats()
        _tls.stats = stats
        t0 = time.time()
        try:
            result = fn()
        finally:
            stats.latency = time.time() - t0
            _tls.stats = None
        return result, stats

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = _coerce_content(conversation[2]["content"])

        # retrieve_context() is pure FAISS; tiktoken estimation for embedding_tokens
        stats = MemoryCallStats()
        stats.embedding_tokens = _count_tokens(query)

        ctx, _s = self._run_with_stats(
            lambda: self._memoryos.retriever.retrieve_context(query, self._user_id)
        )
        stats.latency = _s.latency

        memory_lines = []
        for page in (ctx.get("retrieved_pages") or [])[:self._search_limit]:
            u = (page.get("user_input") or "").strip()
            a = (page.get("agent_response") or "").strip()
            if u or a:
                memory_lines.append(f"{u}\n{a}" if u and a else u or a)
        for k in (ctx.get("retrieved_user_knowledge") or [])[:self._search_limit]:
            text = (k.get("knowledge") or "").strip()
            if text:
                memory_lines.append(text)

        memory_lines = [m for m in memory_lines if m][:self._search_limit]
        if not memory_lines:
            return conversation, stats

        injection = (
            "\n\n--- Retrieved Memories ---\n"
            + "\n".join(f"- {m}" for m in memory_lines)
            + "\n---"
        )
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = _coerce_content(new_conv[0]["content"]) + injection
        return new_conv, stats

    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        # Extract alternating user/assistant obs-action pairs from conv[2:]
        pairs: list[tuple[str, str]] = []
        msgs = conversation[2:]
        for i in range(0, len(msgs) - 1, 2):
            if (msgs[i].get("role") == "user"
                    and i + 1 < len(msgs)
                    and msgs[i + 1].get("role") == "assistant"):
                u = _coerce_content(msgs[i].get("content", ""))
                a = _coerce_content(msgs[i + 1].get("content", ""))
                if u and a:
                    pairs.append((u, a))

        if not pairs:
            return MemoryCallStats()

        def _do():
            for user_input, agent_response in pairs:
                self._memoryos.add_memory(
                    user_input=user_input,
                    agent_response=agent_response,
                )

        _, stats = self._run_with_stats(_do)
        return stats

    def load_from_disk(self, **kwargs) -> None:
        # JSON files persist on disk; re-initializing reconnects to existing data.
        self._initialize()
