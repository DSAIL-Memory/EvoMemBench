"""Shared per-thread Volcengine Ark batch client with tenacity retry.

Shared by the ArkBatchPromptingHandler (main inference) and the mem0 LLM
backend so they reuse the same retry policy and per-thread client-construction
logic.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(tag in name for tag in ("ratelimit", "timeout", "apiconnection", "apierror")):
        return True
    msg = str(exc).lower()
    return any(tag in msg for tag in ("rate limit", "timeout", "timed out", "connection", "503", "overloaded"))


class ArkBatchClient:
    """Synchronous Ark batch chat-completions client.

    One instance per inference runner / LLM backend; per-thread Ark SDK
    instances are created lazily so that httpx connections are not shared
    across threads.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 1200,
        retry_attempts: int = 30,
    ):
        self._api_key = api_key
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._tls = threading.local()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _sdk_client(self):
        client = getattr(self._tls, "client", None)
        if client is None:
            from volcenginesdkarkruntime import Ark  # optional dependency
            client = Ark(api_key=self._api_key, timeout=self._timeout)
            self._tls.client = client
        return client

    def _call_impl(self, model: str, messages: list[dict], tools: list | None = None) -> tuple[Any, float]:
        client = self._sdk_client()
        t0 = time.time()
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = client.batch.chat.completions.create(**kwargs)
        return resp, time.time() - t0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chat_completions(self, model: str, messages: list[dict], tools: list | None = None) -> tuple[Any, float]:
        """Call the Ark batch endpoint and return (response, elapsed_seconds).

        Retries on transient errors with fixed-interval back-off (2–3 s, jitter).
        Pass ``tools`` to use the endpoint's native function-calling mode.
        """
        call = retry(
            wait=wait_fixed(2) + wait_random(0, 1),
            stop=stop_after_attempt(self._retry_attempts),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )(self._call_impl)
        return call(model, messages, tools)
