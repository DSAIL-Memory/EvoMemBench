"""Adapter that exposes the vendored SkillWeaverProvider as a BaseMemory backend.

`inject` retrieves the top-3 most relevant skills (by keyword matching against the
query) and formats their docstrings as text guidance appended to the system prompt.

`update` generates a reusable parameterized Python function from the successful
trajectory using the LLM and appends it to the skills .py file on disk.

In the ALFWorld context, skills provide strategy hints (as text) rather than
executable callables — the agent takes text-based actions, not programmatic tool
calls.
"""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Optional

from .base import BaseMemory, MemoryCallStats
from ._ark_utils import ArkBatchLLM, run_with_stats
from ._evolvelab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData
from ._evolvelab.providers.skillweaver_provider import SkillWeaverProvider


class SkillWeaverMemory(BaseMemory):
    """`skillweaver` BaseMemory backend.

    Storage: Python source files in `skills_dir`. Each successful trajectory
    generates a new generic, parameterised Python function via LLM and appends it
    to the aggregator file. Skills are loaded into an in-memory registry on init
    and after each update.

    Retrieval: Keyword-based scoring against skill name + docstring; top-3 skills
    are formatted as text guidance (function name + docstring) injected into the
    system prompt.
    """

    def __init__(
        self,
        read_only: bool = False,
        skills_dir: str = "./storage/skillweaver",
        ark_batch_config: Optional[dict] = None,
        injection_header: str = "Retrieved Skills (SkillWeaver)",
    ):
        super().__init__(memory_type="skillweaver", read_only=read_only)
        if ark_batch_config is None:
            raise ValueError("SkillWeaverMemory requires ark_batch_config={api_key, model}.")

        self._injection_header = injection_header
        self._lock = threading.RLock()

        llm = ArkBatchLLM(
            api_key=ark_batch_config["api_key"],
            model=ark_batch_config["model"],
        )

        import os
        skills_file = os.path.join(skills_dir, "skillweaver_generated_skills.py")
        provider_config = {
            "skills_dir": skills_dir,
            "skills_file_path": skills_file,
            "model": llm,
        }
        self._provider = SkillWeaverProvider(provider_config)
        if not self._provider.initialize():
            raise RuntimeError("SkillWeaverProvider failed to initialize.")

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
        skills_dir: str | None = None,
        **_kwargs,
    ) -> None:
        with self._lock:
            if skills_dir is not None:
                import os
                self._provider.skills_dir = skills_dir
                self._provider.skills_file_path = os.path.join(
                    skills_dir, "skillweaver_generated_skills.py"
                )
            self._provider.initialize()
