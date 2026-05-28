"""mem0 EmbeddingBase subclass for DashScope text-embedding-v4.

Uses the OpenAI-compatible DashScope endpoint and records token usage
in the per-thread metrics accumulator.

Registered as provider "openai_dashscope" in mem0's EmbedderFactory at
Mem0Memory construction time. Only loaded when mem0 is installed.
"""
from __future__ import annotations

import time
from typing import Optional

from mem0.embeddings.openai import OpenAIEmbedding
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

from bfcl_eval.memory import metrics


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(t in name for t in ("ratelimit", "timeout", "apiconnection", "apierror")):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in ("rate limit", "timeout", "timed out", "connection", "503", "overloaded"))


_EMBED_RETRY = retry(
    wait=wait_fixed(2) + wait_random(0, 1),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


class DashscopeEmbedder(OpenAIEmbedding):
    """Wraps OpenAIEmbedding with DashScope base_url, retry, and usage stats."""

    @_EMBED_RETRY
    def _create(self, kwargs: dict) -> tuple:
        # Latency measured inside the retried call so metrics.record_embed_call
        # reflects the successful attempt's wall time only (not retry waits).
        t0 = time.time()
        resp = self.client.embeddings.create(**kwargs)
        return resp, time.time() - t0

    def embed(self, text: str, memory_action: Optional[str] = None) -> list[float]:
        text = text.replace("\n", " ")
        kwargs: dict = {
            "input": [text],
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        resp, elapsed = self._create(kwargs)
        metrics.record_embed_call(
            total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
            latency_s=elapsed,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]:
        MAX_BATCH = 100
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i : i + MAX_BATCH]
            kwargs: dict = {
                "input": chunk,
                "model": self.config.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response, elapsed = self._create(kwargs)
            metrics.record_embed_call(
                total_tokens=int(getattr(response.usage, "total_tokens", 0) or 0),
                latency_s=elapsed,
            )
            all_embeddings.extend(
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            )
        return all_embeddings
