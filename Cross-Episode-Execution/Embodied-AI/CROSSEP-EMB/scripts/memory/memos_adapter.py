"""MemOS (memos) memory adapter for AgentGym ALFWorld evaluation.

Code structure mirrors mem0_adapter.py exactly.
Initialization logic ported from EvoMemBench-XbenchWebWalkQA/memos_provider.py.

Install: pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
"""
from __future__ import annotations

import json
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


class MemosMemory(BaseMemory):
    """Memory backend backed by MemOS GeneralTextMemory with Qdrant vector store."""

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        search_limit: int = 3,
        qdrant_path: str | None = None,
        collection_name: str | None = None,
        vector_dimension: int = 1024,
        llm: dict | None = None,
        embed: dict | None = None,
        **kwargs,
    ):
        super().__init__(memory_type="memos", read_only=read_only)
        self._agent_id = agent_id
        self._search_limit = search_limit
        self._qdrant_path = qdrant_path or f"./storage/memos/qdrant/{agent_id}"
        self._collection_name = collection_name or agent_id
        self._vector_dim = vector_dimension
        self._llm_cfg = llm or {}
        self._embed_cfg = embed or {}
        self._memory = None
        self._initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Initialize GeneralTextMemory. Ported from XbenchWebWalkQA/memos_provider.py."""
        # memos/api/config.py calls load_dotenv(override=True) on import, which can
        # overwrite OPENAI_API_KEY etc. Save and restore to prevent env pollution.
        _preserved = {
            k: os.environ.get(k)
            for k in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
                      "MEMRADER_API_KEY", "MEMRADER_API_BASE")
        }
        try:
            from memos.configs.memory import GeneralTextMemoryConfig
            from memos.memories.textual.general import GeneralTextMemory
        except ImportError as e:
            raise ImportError(
                f"memos is not installed: {e}\n"
                "Run: pip install -e ../../../EvoMemBench-Memory-Systems/MemOS"
            ) from e
        finally:
            for k, v in _preserved.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        os.makedirs(self._qdrant_path, exist_ok=True)

        mem_cfg = GeneralTextMemoryConfig.model_validate({
            "extractor_llm": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": (
                        self._llm_cfg.get("model")
                        or os.getenv("ANALYSIS_MODEL", "deepseek-v3-2-251201")
                    ),
                    "api_key": self._llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
                    "api_base": self._llm_cfg.get("base_url") or os.getenv(
                        "OPENAI_BASE_URL", "https://api.openai.com/v1"
                    ),
                },
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "collection_name": self._collection_name,
                    "path": self._qdrant_path,
                    "vector_dimension": self._vector_dim,
                    "distance_metric": "cosine",
                },
            },
            "embedder": {
                "backend": "universal_api",
                "config": {
                    "provider": "openai",
                    "model_name_or_path": self._embed_cfg.get("model", "text-embedding-v4"),
                    "api_key": self._embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                    "base_url": self._embed_cfg.get("base_url") or os.getenv(
                        "DASHSCOPE_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ),
                    "embedding_dims": self._vector_dim,
                },
            },
        })
        self._memory = GeneralTextMemory(mem_cfg)
        self._install_token_hooks()

        try:
            count = len(self._memory.get_all())
        except Exception:
            count = "?"
        logger.info(
            "[memos] initialized (agent_id=%s, search_limit=%d, qdrant=%s, memories=%s)",
            self._agent_id, self._search_limit, self._qdrant_path, count,
        )

    def _install_token_hooks(self) -> None:
        """Monkey-patch memos internals to count tokens into _tls.stats."""
        if self._memory is None:
            return

        # --- LLM tokens: wrap extractor_llm.client (or ._client) ---
        llm = getattr(self._memory, "extractor_llm", None)
        if llm is not None:
            wrapped = False
            for attr in ("client", "_client"):
                raw = getattr(llm, attr, None)
                if raw is not None:
                    setattr(llm, attr, _make_counting_client(raw))
                    logger.info("[memos] extractor_llm.%s wrapped for token counting", attr)
                    wrapped = True
                    break

            if not wrapped:
                # Fallback: wrap generate_response / generate with tiktoken estimation
                for method_name in ("generate_response", "generate"):
                    orig = getattr(llm, method_name, None)
                    if orig is not None:
                        def _tracked(messages, *a, _orig=orig, **kw):
                            stats = _tls_stats()
                            if stats is not None:
                                try:
                                    stats.input_tokens += _count_tokens(
                                        json.dumps(messages, ensure_ascii=False)
                                    )
                                except Exception:
                                    pass
                            result = _orig(messages, *a, **kw)
                            if stats is not None:
                                try:
                                    out = result if isinstance(result, str) else json.dumps(
                                        result, ensure_ascii=False
                                    )
                                    stats.output_tokens += _count_tokens(out)
                                except Exception:
                                    pass
                            return result
                        setattr(llm, method_name, _tracked)
                        logger.info("[memos] extractor_llm.%s wrapped (tiktoken fallback)", method_name)
                        wrapped = True
                        break

            if not wrapped:
                logger.warning(
                    "[memos] Could not find extractor_llm client or generate method; "
                    "LLM tokens will not be counted"
                )
        else:
            logger.warning("[memos] extractor_llm not found on _memory object")

        # --- Embedding tokens: wrap embedder.embed() with tiktoken estimation ---
        emb = getattr(self._memory, "embedding_model", None)
        if emb is None:
            emb = getattr(self._memory, "embedder", None)
        if emb is not None:
            orig_embed = emb.embed

            def tracked_embed(text, *args, **kwargs):
                stats = _tls_stats()
                if stats is not None:
                    try:
                        stats.embedding_tokens += _count_tokens(
                            text if isinstance(text, str) else str(text)
                        )
                    except Exception:
                        pass
                return orig_embed(text, *args, **kwargs)

            emb.embed = tracked_embed
            logger.info("[memos] embedder.embed wrapped for token counting")
        else:
            logger.warning("[memos] Could not find embedder; embedding tokens will not be counted")

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

        query = conversation[2]["content"]

        results, stats = self._run_with_stats(
            lambda: self._memory.search(query, self._search_limit)
        )

        if not results:
            return conversation, stats

        memory_lines = "\n".join(f"- {r.memory}" for r in results if r.memory)
        injection = f"\n\n--- Retrieved Memories ---\n{memory_lines}\n---"

        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = new_conv[0]["content"] + injection
        return new_conv, stats

    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        messages = [
            {
                "role": m["role"],
                "content": _coerce_content(m.get("content", "")),
            }
            for m in conversation
            if m.get("role") in ("user", "assistant", "system") and m.get("content")
        ]
        if not messages:
            return MemoryCallStats()

        def _do():
            extracted = self._memory.extract(messages)
            if extracted:
                self._memory.add(extracted)
            return extracted

        _, stats = self._run_with_stats(_do)
        return stats

    def load_from_disk(self, **kwargs) -> None:
        # Qdrant is file-persistent; re-initializing connects to existing data.
        self._initialize()
