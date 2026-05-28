"""AgentKBMemory — bfcl_eval Memory adapter for EvolveLab's AgentKBProvider.

AgentKB maintains a JSON knowledge-base of past task trajectories and
retrieves the most relevant ones via hybrid TF-IDF + sentence-embedding
search. On each sample it injects synthesised planning and experience
guidance into the system prompt.

Storage layout under storage_dir/:
    agent_kb_database.json  — JSON array of workflow records
    models/                 — cached sentence-transformer weights
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# helpers inject EvolveLab into sys.path and register the ToolWrapper stub
from bfcl_eval.memory.evolvlab_helpers import (
    ArkLLMCallable,
    extract_query_from_trajectory,
    extract_result_from_trajectory,
    format_trajectory_as_steps,
)
from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory

from EvolveLab.providers.agent_kb_provider import AgentKBProvider
from EvolveLab.memory_types import MemoryRequest, MemoryStatus, TrajectoryData

_UTILIZE_PREFIX = "\n\nAgentKB guidance from prior episodes:\n"


class AgentKBMemory(Memory):
    """Cross-episode memory backed by AgentKBProvider.

    Retrieval uses hybrid TF-IDF + semantic search over a growing JSON
    knowledge-base; ingestion extracts structured planning / experience
    summaries via LLM.

    Constructor kwargs (passed by build_memory / load_from_disk):
        storage_dir     Path  — root directory for the knowledge-base and model cache
        ark_api_key     str   — Ark API key for the summarisation LLM
        ark_model       str   — Ark model endpoint for the summarisation LLM
        top_k           int   — number of memories to retrieve (default 3)
        search_weights  dict  — {'text': w1, 'semantic': w2} (default 0.5/0.5)
        clear           bool  — wipe the knowledge-base on construction
        readonly        bool  — skip update() calls
    """

    def __init__(
        self,
        *,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        top_k: int = 3,
        search_weights: Optional[dict] = None,
        clear: bool = False,
        readonly: bool = False,
        **_: object,
    ) -> None:
        super().__init__(clear=clear, readonly=readonly)

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        db_path = str(storage_dir / "agent_kb_database.json")

        if clear and os.path.exists(db_path):
            with open(db_path, "w") as f:
                json.dump([], f)

        llm = ArkLLMCallable(api_key=ark_api_key, model_name=ark_model)
        config = {
            "kb_database_path": db_path,
            "top_k": top_k,
            "search_weights": search_weights or {"text": 0.5, "semantic": 0.5},
            "model": llm,
        }
        self._provider = AgentKBProvider(config=config)
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
    def load_from_disk(cls, **backend_kwargs: object) -> "AgentKBMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)  # type: ignore[arg-type]
