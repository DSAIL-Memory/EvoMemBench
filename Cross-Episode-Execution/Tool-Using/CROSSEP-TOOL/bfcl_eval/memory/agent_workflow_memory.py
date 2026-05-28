"""AgentWorkflowMemory — bfcl_eval Memory adapter for AgentWorkflowMemoryProvider.

AgentWorkflowMemory stores abstracted workflow descriptions extracted from
past successful trajectories and retrieves the top-K most similar ones via
cosine similarity over sentence-embedding vectors.

Storage layout under storage_dir/:
    workflow_memory.json    — JSON with 'memories' + 'embeddings' arrays
    models/                 — cached sentence-transformer weights
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

from EvolveLab.providers.agent_workflow_memory_provider import AgentWorkflowMemoryProvider
from EvolveLab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData

_UTILIZE_PREFIX = "\n\nWorkflow memories from prior episodes:\n"


class AgentWorkflowMemory(Memory):
    """Cross-episode memory backed by AgentWorkflowMemoryProvider.

    Each completed (correct) trajectory is abstracted into a generic,
    parameterised workflow description by an LLM and stored with its
    sentence embedding. At retrieval, the top-K most query-similar
    workflows are returned as guidance.

    Constructor kwargs:
        storage_dir         Path  — root directory for memory store and model cache
        ark_api_key         str   — Ark API key for the workflow induction LLM
        ark_model           str   — Ark model endpoint
        top_k               int   — number of workflows to retrieve (default 3)
        embed_model_name    str   — sentence-transformer model name
        enable_induction    bool  — run LLM abstraction on ingestion (default True)
        clear               bool  — delete the store file on construction
        readonly            bool  — skip update() calls
    """

    def __init__(
        self,
        *,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        top_k: int = 3,
        enable_induction: bool = True,
        clear: bool = False,
        readonly: bool = False,
        **_: object,
    ) -> None:
        super().__init__(clear=clear, readonly=readonly)

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        store_path = str(storage_dir / "workflow_memory.json")

        if clear and os.path.exists(store_path):
            os.remove(store_path)

        llm = ArkLLMCallable(api_key=ark_api_key, model_name=ark_model)
        config = {
            "store_path": store_path,
            "top_k": top_k,
            "enable_induction": enable_induction,
            "model": llm,
        }
        self._provider = AgentWorkflowMemoryProvider(config=config)
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
    def load_from_disk(cls, **backend_kwargs: object) -> "AgentWorkflowMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)  # type: ignore[arg-type]
