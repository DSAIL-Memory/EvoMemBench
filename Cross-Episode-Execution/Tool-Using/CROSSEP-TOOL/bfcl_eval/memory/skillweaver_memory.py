"""SkillWeaverMemory — bfcl_eval Memory adapter for EvolveLab's SkillWeaverProvider.

SkillWeaver generates reusable Python skill functions from successful
trajectories and stores them as .py files. At retrieval it keyword-matches
the query against skill names and docstrings, returning the most relevant
skill descriptions as system-prompt guidance.

Storage layout under storage_dir/:
    skillweaver_generated_skills.py  — single aggregated skills file
    skills/                          — optional per-skill .py files

Note: the ToolWrapper used by SkillWeaverProvider to wrap skills as
callable Tool objects is stubbed out (bfcl_eval only needs the textual
skill descriptions injected into the prompt).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

# evolvlab_helpers registers the ToolWrapper stub BEFORE SkillWeaverProvider
# is imported, so the `from storage.tools.tool_wrapper import ToolWrapper`
# line inside skillweaver_provider.py resolves cleanly.
from bfcl_eval.memory.evolvlab_helpers import (
    ArkLLMCallable,
    extract_query_from_trajectory,
    extract_result_from_trajectory,
    format_trajectory_as_steps,
)
from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory

from EvolveLab.providers.skillweaver_provider import SkillWeaverProvider
from EvolveLab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData

_UTILIZE_PREFIX = "\n\nSkillWeaver reusable skills from prior episodes:\n"


class SkillWeaverMemory(Memory):
    """Cross-episode memory backed by SkillWeaverProvider.

    On each successful trajectory an LLM synthesises a generic, parameterised
    Python function that captures the task's methodology. Skill functions are
    persisted as .py source files and their textual descriptions are injected
    into the system prompt on retrieval.

    Constructor kwargs:
        storage_dir     Path  — root directory for skill files
        ark_api_key     str   — Ark API key for skill generation LLM
        ark_model       str   — Ark model endpoint
        clear           bool  — delete all skill files on construction
        readonly        bool  — skip update() calls
    """

    def __init__(
        self,
        *,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        clear: bool = False,
        readonly: bool = False,
        **_: object,
    ) -> None:
        super().__init__(clear=clear, readonly=readonly)

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        skills_file = storage_dir / "skillweaver_generated_skills.py"
        skills_dir = storage_dir / "skills"

        if clear:
            if skills_file.exists():
                skills_file.unlink()
            if skills_dir.exists():
                shutil.rmtree(skills_dir)

        llm = ArkLLMCallable(api_key=ark_api_key, model_name=ark_model)
        config = {
            "skills_file_path": str(skills_file),
            # Pass skills_dir="" so the provider never creates an empty directory
            # that would shadow skills_file_path in _reload_skills().
            "skills_dir": "",
            "model": llm,
        }
        self._provider = SkillWeaverProvider(config=config)
        self._provider.initialize()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        request = MemoryRequest(
            query=system_prompt,
            context="",
            status=MemoryStatus.BEGIN,
        )
        response = self._provider.provide_memory(request)

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        if not response.memories:
            return ""
        lines = [str(m.content) for m in response.memories if m.content]
        return (_UTILIZE_PREFIX + "\n\n".join(lines)) if lines else ""

    def update(
        self,
        trajectory: list[dict],
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        if self.readonly:
            return
        # Both is_correct and task_success must be True for SkillWeaver to
        # extract a skill (it skips trajectories that fail either check).
        merged_meta = {"is_correct": True, "task_success": True}
        if metadata:
            merged_meta.update(metadata)
        tdata = TrajectoryData(
            query=extract_query_from_trajectory(trajectory),
            trajectory=format_trajectory_as_steps(trajectory),
            result=extract_result_from_trajectory(trajectory),
            metadata=merged_meta,
        )
        self._provider.take_in_memory(tdata)

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs: object) -> "SkillWeaverMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)  # type: ignore[arg-type]
