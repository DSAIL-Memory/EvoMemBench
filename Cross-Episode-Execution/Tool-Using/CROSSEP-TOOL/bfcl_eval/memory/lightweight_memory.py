"""LightweightMemory — bfcl_eval Memory adapter for LightweightMemoryProvider.

LightweightMemory is a dual-layer memory system:
  - Long-term memory: strategic + operational memories stored in a JSON file,
    retrieved at the BEGIN phase via LLM selection / synthesis.
  - Short-term memory: key facts / constraints accumulated during task
    execution and refreshed every N steps (IN phase).

Mapping to the bfcl_eval Memory interface:
  utilize(system_prompt) → BEGIN phase — retrieves long-term guidance.
    The short-term IN phase is not wired here because bfcl_eval calls
    `utilize` only once per sample (before multi-turn inference starts).
  update(trajectory)     → ingests strategic + operational memories from
    completed successful trajectories.

Storage layout under storage_dir/:
    longterm_memory.json  — JSON dict with 'strategic' and 'operational' arrays
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from bfcl_eval.memory.evolvlab_helpers import (
    ArkLLMCallable,
    extract_query_from_trajectory,
    extract_result_from_trajectory,
    format_trajectory_as_steps,
)
from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory

from EvolveLab.providers.lightweight_memory_provider import LightweightMemoryProvider
from EvolveLab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData

_UTILIZE_PREFIX = "\n\nLightweight memory guidance from prior episodes:\n"


class LightweightMemory(Memory):
    """Cross-episode memory backed by LightweightMemoryProvider.

    Retrieval uses an LLM to select and synthesise the most relevant
    strategic / operational memories from the long-term store.
    Ingestion extracts new strategic and operational insights from
    successful trajectories and prunes the store intelligently when
    capacity limits are reached.

    Constructor kwargs:
        storage_dir                 Path  — root directory
        ark_api_key                 str   — Ark API key
        ark_model                   str   — Ark model endpoint
        top_k                       int   — long-term memories to retrieve (default 5)
        max_strategic_memories      int   — max strategic entries (default 30)
        max_operational_memories    int   — max operational entries (default 30)
        enable_longterm_provision   bool  — enable long-term retrieval (default True)
        clear                       bool  — delete the memory file on construction
        readonly                    bool  — skip update() calls
    """

    def __init__(
        self,
        *,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        top_k: int = 5,
        max_strategic_memories: int = 30,
        max_operational_memories: int = 30,
        enable_longterm_provision: bool = True,
        clear: bool = False,
        readonly: bool = False,
        **_: object,
    ) -> None:
        super().__init__(clear=clear, readonly=readonly)

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        ltm_path = str(storage_dir / "longterm_memory.json")

        if clear and os.path.exists(ltm_path):
            os.remove(ltm_path)

        llm = ArkLLMCallable(api_key=ark_api_key, model_name=ark_model)
        config = {
            "storage_dir": str(storage_dir),
            "longterm_memory_path": ltm_path,
            "model": llm,
            "top_k_longterm": top_k,
            "max_strategic_memories": max_strategic_memories,
            "max_operational_memories": max_operational_memories,
            "enable_longterm_provision": enable_longterm_provision,
        }
        self._provider = LightweightMemoryProvider(config=config)
        self._provider.initialize()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        # Reset per-task state so the provider treats this as a fresh BEGIN.
        self._provider.reset_task_context(query=system_prompt)

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
        tdata = TrajectoryData(
            query=extract_query_from_trajectory(trajectory),
            trajectory=format_trajectory_as_steps(trajectory),
            result=extract_result_from_trajectory(trajectory),
            metadata=metadata or {"is_correct": True},
        )
        self._provider.take_in_memory(tdata)

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs: object) -> "LightweightMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)  # type: ignore[arg-type]
