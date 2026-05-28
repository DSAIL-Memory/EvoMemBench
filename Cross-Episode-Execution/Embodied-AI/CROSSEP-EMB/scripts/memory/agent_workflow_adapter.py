"""Adapter that exposes the vendored AgentWorkflowMemoryProvider as a BaseMemory backend.

`inject` retrieves the top-K nearest workflow memories by DashScope-embedding
similarity against `conversation[2]["content"]`, formats them as a list of
`Query: ...\\nWorkflow: ...` blocks, and appends to the system prompt.

`update` runs LLM-based "workflow induction" on the trajectory (when
reward >= 1 and `enable_induction` is set), producing an abstracted
reusable workflow text that's embedded and appended to the JSON store.
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Optional

from .base import BaseMemory, MemoryCallStats
from ._ark_utils import ArkBatchLLM, patch_embedding_model, run_with_stats
from ._evolvelab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData
from ._evolvelab.providers.agent_workflow_memory_provider import (
    AgentWorkflowMemoryProvider,
)


class AgentWorkflowMemory(BaseMemory):
    """`agent_workflow_memory` BaseMemory backend (a.k.a. AWM)."""

    def __init__(
        self,
        read_only: bool = False,
        store_path: str = "./storage/awm/awm_store.json",
        top_k: int = 3,
        enable_induction: bool = True,
        ark_batch_config: Optional[dict] = None,
        injection_header: str = "Retrieved Workflows (AWM)",
    ):
        super().__init__(memory_type="agent_workflow_memory", read_only=read_only)
        if ark_batch_config is None:
            raise ValueError(
                "AgentWorkflowMemory requires ark_batch_config={api_key, model}."
            )

        self._injection_header = injection_header
        self._lock = threading.RLock()

        llm = ArkBatchLLM(
            api_key=ark_batch_config["api_key"],
            model=ark_batch_config["model"],
        )

        provider_config = {
            "store_path": store_path,
            "top_k": top_k,
            "enable_induction": enable_induction,
            "model": llm,
        }
        self._provider = AgentWorkflowMemoryProvider(provider_config)
        if not self._provider.initialize():
            raise RuntimeError("AgentWorkflowMemoryProvider failed to initialize.")
        patch_embedding_model(self._provider._embedding_model)

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]
        request = MemoryRequest(query=query, context="", status=MemoryStatus.BEGIN)

        with self._lock:
            response, stats = run_with_stats(
                lambda: self._provider.provide_memory(request)
            )

        if not response.memories:
            return conversation, stats

        chunks = []
        for m in response.memories:
            if m.content:
                chunks.append(str(m.content))
        if not chunks:
            return conversation, stats

        injection = (
            f"\n\n--- {self._injection_header} ---\n"
            + "\n\n".join(f"- {c}" for c in chunks)
            + "\n---"
        )

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

        if reward is None or reward < 1:
            return MemoryCallStats()

        if len(conversation) < 3:
            return MemoryCallStats()

        query = conversation[2]["content"]
        trajectory = []
        for msg in conversation[3:]:
            if not msg.get("content"):
                continue
            step_type = "action" if msg.get("role") == "assistant" else "observation"
            trajectory.append({"type": step_type, "content": msg["content"]})

        td = TrajectoryData(
            query=query,
            trajectory=trajectory,
            result=f"reward={reward}",
            metadata={
                "is_correct": True,
                "data_idx": data_idx,
                "reward": float(reward),
            },
        )

        with self._lock:
            (_success, _msg), stats = run_with_stats(
                lambda: self._provider.take_in_memory(td)
            )
        return stats

    def load_from_disk(
        self,
        store_path: str | None = None,
        **_kwargs,
    ) -> None:
        with self._lock:
            if store_path is not None:
                self._provider.config["store_path"] = store_path
            self._provider.initialize()
