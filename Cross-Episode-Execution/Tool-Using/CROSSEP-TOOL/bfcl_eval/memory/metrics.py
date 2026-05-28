"""Per-thread memory usage accumulator.

Usage pattern in run_sample:
    memory.begin_sample()       # reset for this thread
    snippet = memory.utilize(…) # embed call → record_embed_call
    …inference loop…
    memory.update(trajectory)   # llm call  → record_llm_call
    usage = memory.drain_usage()
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

_TLS = threading.local()


@dataclass
class MemoryUsageLog:
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    n_llm_calls: int = 0
    n_embed_calls: int = 0
    n_utilize: int = 0
    n_update: int = 0


def begin_sample() -> None:
    _TLS.usage = MemoryUsageLog()


def record_llm_call(latency_s: float, input_tokens: int, output_tokens: int) -> None:
    u: MemoryUsageLog | None = getattr(_TLS, "usage", None)
    if u is None:
        return
    u.latency_s += latency_s
    u.input_tokens += input_tokens
    u.output_tokens += output_tokens
    u.n_llm_calls += 1


def record_embed_call(total_tokens: int, latency_s: float = 0.0) -> None:
    u: MemoryUsageLog | None = getattr(_TLS, "usage", None)
    if u is None:
        return
    u.embedding_tokens += total_tokens
    u.latency_s += latency_s
    u.n_embed_calls += 1


def drain() -> MemoryUsageLog:
    u = getattr(_TLS, "usage", MemoryUsageLog())
    _TLS.usage = MemoryUsageLog()
    return u
