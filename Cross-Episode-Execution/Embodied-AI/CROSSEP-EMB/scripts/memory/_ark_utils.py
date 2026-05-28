"""Shared helpers used by AgentKB and AgentWorkflowMemory adapters:

- ArkBatchLLM:      Callable backed by volcengine ARK *batch* API (cost-efficient but slow under load).
- OpenAICompatLLM:  Callable backed by any OpenAI-compatible endpoint (faster for interactive use).
- Thread-local stat accumulation so token usage is attributed per-call.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import List

from .base import MemoryCallStats

# Thread-local stat accumulator. Tracker functions read/write _tls.stats.
_tls = threading.local()


def _tls_stats() -> MemoryCallStats | None:
    return getattr(_tls, "stats", None)


def run_with_stats(fn):
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


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class ArkBatchLLM:
    """Callable LLM client backed by volcengine ARK batch API.

    Matches the signature expected by the vendored EvolveLab providers:
    `model(messages) -> obj` where `obj.content` is the assistant text.
    Messages may be in the EvolveLab content-array form
    `[{"role": ..., "content": [{"type": "text", "text": "..."}]}]` or
    OpenAI-flat. We normalize to flat strings before sending.

    Token usage is recorded into the active thread-local MemoryCallStats.
    """

    def __init__(self, api_key: str, model: str, max_retries: int = 8, retry_delay: float = 5.0):
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as e:
            raise ImportError(
                "ArkBatchLLM 需要 volcengine SDK，请运行：\n"
                "  pip install 'volcengine-python-sdk[ark]'"
            ) from e
        self.client = Ark(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @staticmethod
    def _normalize_messages(messages: List[dict]) -> List[dict]:
        normalized = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        parts.append(c)
                content = "\n".join(p for p in parts if p)
            normalized.append({"role": m.get("role", "user"), "content": content})
        return normalized

    def __call__(self, messages):
        norm_msgs = self._normalize_messages(messages)
        stats = _tls_stats()

        for attempt in range(self.max_retries):
            try:
                response = self.client.batch.chat.completions.create(
                    model=self.model,
                    messages=norm_msgs,
                )
                if stats is not None and response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    cached = 0
                    if (hasattr(response.usage, "prompt_tokens_details")
                            and response.usage.prompt_tokens_details):
                        cached = getattr(
                            response.usage.prompt_tokens_details, "cached_tokens", 0
                        ) or 0
                    stats.cached_tokens += cached
                content = response.choices[0].message.content
                return SimpleNamespace(content=content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(
                    f"[ArkBatchLLM retry {attempt+1}/{self.max_retries}] {e}. "
                    f"Retrying in {self.retry_delay:.1f}s..."
                )
                time.sleep(self.retry_delay)


def patch_embedding_model(embedding_model) -> None:
    """Wrap an embedding model's `encode` method so embedding tokens flow into _tls.stats."""
    orig_encode = embedding_model.encode

    def tracked_encode(texts, *args, **kwargs):
        stats = _tls_stats()
        if stats is not None:
            try:
                if isinstance(texts, str):
                    stats.embedding_tokens += count_tokens(texts)
                else:
                    for t in texts:
                        stats.embedding_tokens += count_tokens(str(t))
            except Exception:
                pass
        return orig_encode(texts, *args, **kwargs)

    embedding_model.encode = tracked_encode
