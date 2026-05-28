"""mem0 LLMBase subclass that routes calls through the Ark batch endpoint.

Registered as provider "ark_batch" in mem0's LlmFactory at Mem0Memory construction
time. Only loaded when mem0 is installed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from mem0.llms.base import LLMBase

from bfcl_eval.memory import metrics
from bfcl_eval.memory.ark_client import ArkBatchClient


class ArkBatchLLM(LLMBase):
    """Routes mem0's internal LLM calls (fact extraction, scoring) through the
    same Volcengine Ark batch endpoint used for main inference."""

    def __init__(self, config=None):
        super().__init__(config)
        api_key = getattr(self.config, "api_key", None) or os.getenv("DEEPSEEK_API_KEY", "")
        self._ark_client = ArkBatchClient(api_key=api_key)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        resp, dt = self._ark_client.chat_completions(
            model=self.config.model,
            messages=messages,
        )
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        return resp.choices[0].message.content or ""
