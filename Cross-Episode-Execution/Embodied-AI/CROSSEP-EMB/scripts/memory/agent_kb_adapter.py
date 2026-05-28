"""Adapter that exposes the vendored AgentKBProvider as a BaseMemory backend.

`inject` translates the conversation's first env observation into a
MemoryRequest, runs the provider's hybrid retrieval + LLM synthesis, and
appends the synthesized "AGENT-KB Student/Teacher Guidance" block to the
system prompt.

`update` packages the trajectory as a TrajectoryData with `is_correct=True`
when reward >= 1, then calls the provider's `take_in_memory` (which uses an
LLM to summarize the trajectory into 4 reusable fields and appends it to
the JSON knowledge base on disk).
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Optional

from .base import BaseMemory, MemoryCallStats
from ._ark_utils import ArkBatchLLM, patch_embedding_model, run_with_stats
from ._evolvelab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData
from ._evolvelab.providers.agent_kb_provider import AgentKBProvider


class AgentKBMemory(BaseMemory):
    """`agent_kb` BaseMemory backend.

    Storage: JSON file at `kb_database_path`. Each successful trajectory is
    LLM-summarized into 4 fields (agent_planning, search_agent_planning,
    agent_experience, search_agent_experience) and appended.

    Retrieval: TF-IDF + DashScope-semantic hybrid search with weighted
    aggregation. Top-K results are LLM-synthesized into student + teacher
    guidance blocks injected into the system prompt.
    """

    def __init__(
        self,
        read_only: bool = False,
        kb_database_path: str = "./storage/agent_kb/agent_kb_database.json",
        top_k: int = 3,
        search_weights: Optional[dict] = None,
        ark_batch_config: Optional[dict] = None,
        injection_header: str = "Retrieved Memories (AgentKB)",
    ):
        super().__init__(memory_type="agent_kb", read_only=read_only)
        if ark_batch_config is None:
            raise ValueError("AgentKBMemory requires ark_batch_config={api_key, model}.")

        self._injection_header = injection_header
        self._lock = threading.RLock()

        llm = ArkBatchLLM(
            api_key=ark_batch_config["api_key"],
            model=ark_batch_config["model"],
        )

        provider_config = {
            "kb_database_path": kb_database_path,
            "top_k": top_k,
            "search_weights": search_weights or {"text": 0.5, "semantic": 0.5},
            "model": llm,
        }
        self._provider = AgentKBProvider(provider_config)
        if not self._provider.initialize():
            raise RuntimeError("AgentKBProvider failed to initialize.")
        self._patch_embedding_model()

    def _patch_embedding_model(self) -> None:
        if self._provider.akb_manager is not None:
            patch_embedding_model(self._provider.akb_manager.knowledge_base.embedding_model)

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

        chunks = []
        for m in response.memories:
            if m.content:
                chunks.append(str(m.content))
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
                "data_idx": data_idx,
                "reward": float(reward),
            },
        )

        with self._lock:
            (success, _msg), stats = run_with_stats(
                lambda: self._provider.take_in_memory(td)
            )
            if success:
                self._patch_embedding_model()
        return stats

    def load_from_disk(
        self,
        kb_database_path: str | None = None,
        **_kwargs,
    ) -> None:
        with self._lock:
            if kb_database_path is not None:
                self._provider.kb_database_path = kb_database_path
            self._provider.initialize()
            self._patch_embedding_model()
