"""Adapter that exposes the vendored LightweightMemoryProvider as a BaseMemory backend.

`inject` uses the BEGIN phase of LightweightMemoryProvider: retrieves top-K long-term
memories (strategic/operational) and synthesizes them into concise guidance appended
to the system prompt.

`update` packages the trajectory and calls `take_in_memory` on success (reward >= 1),
which uses the LLM to extract strategic/operational learnings and saves them to the
JSON long-term memory store.

Note: The IN-phase short-term memory (intra-episode key-fact accumulation) is not used
here because the AgentGym framework calls inject only once per episode.
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Optional

from .base import BaseMemory, MemoryCallStats
from ._ark_utils import ArkBatchLLM, run_with_stats
from ._evolvelab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData
from ._evolvelab.providers.lightweight_memory_provider import LightweightMemoryProvider


class LightweightMemory(BaseMemory):
    """`lightweight_memory` BaseMemory backend.

    Storage: JSON file at `store_path` with strategic (up to 30) and operational
    (up to 30) memories; cold-start memories are injected automatically on first
    run so there is something to retrieve even before any successful episodes.

    Retrieval: LLM selects and synthesizes top-K most relevant memories from the
    long-term database into concise actionable guidance injected at the start of
    each episode.

    Update: LLM extracts up to 2 strategic + 2 operational reusable learnings from
    successful trajectories; pruning happens via LLM when the buffer exceeds
    (max + buffer_size).
    """

    def __init__(
        self,
        read_only: bool = False,
        store_path: str = "./storage/lightweight_memory/longterm_memory.json",
        max_strategic: int = 30,
        max_operational: int = 30,
        top_k: int = 5,
        enable_longterm_provision: bool = True,
        ark_batch_config: Optional[dict] = None,
        injection_header: str = "Retrieved Memories (LightweightMemory)",
    ):
        super().__init__(memory_type="lightweight_memory", read_only=read_only)
        if ark_batch_config is None:
            raise ValueError("LightweightMemory requires ark_batch_config={api_key, model}.")

        self._injection_header = injection_header
        self._lock = threading.RLock()

        llm = ArkBatchLLM(
            api_key=ark_batch_config["api_key"],
            model=ark_batch_config["model"],
        )

        import os
        storage_dir = os.path.dirname(store_path) or "."
        provider_config = {
            "storage_dir": storage_dir,
            "longterm_memory_path": store_path,
            "max_strategic_memories": max_strategic,
            "max_operational_memories": max_operational,
            "top_k_longterm": top_k,
            "enable_longterm_provision": enable_longterm_provision,
            "model": llm,
        }
        self._provider = LightweightMemoryProvider(provider_config)
        if not self._provider.initialize():
            raise RuntimeError("LightweightMemoryProvider failed to initialize.")

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]

        request = MemoryRequest(
            query=query,
            context="",
            status=MemoryStatus.BEGIN,
        )

        with self._lock:
            response, stats = run_with_stats(
                lambda: self._provider.provide_memory(request)
            )

        if not response.memories:
            return conversation, stats

        chunks = [str(m.content) for m in response.memories if m.content]
        if not chunks:
            return conversation, stats

        injection = (
            f"\n\n--- {self._injection_header} ---\n"
            + "\n\n".join(chunks)
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
                "task_success": True,
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
                import os
                self._provider.storage_dir = os.path.dirname(store_path) or "."
                self._provider.longterm_memory_path = store_path
            self._provider.initialize()
