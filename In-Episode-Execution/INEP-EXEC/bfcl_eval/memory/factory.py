"""Memory backend factory."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

from bfcl_eval.memory.base import Memory


def build_memory(memory_type: str, **kwargs) -> Optional[Memory]:
    """Construct a Memory instance for the given memory_type.

    Returns None when memory_type is "none" (memory disabled).
    """
    if memory_type in ("none", None, ""):
        return None
    if memory_type == "mem0":
        from bfcl_eval.memory.mem0_backend import Mem0Memory
        return Mem0Memory(**kwargs)
    if memory_type == "memagent":
        from bfcl_eval.memory.memagent import MemAgentMemory
        return MemAgentMemory(memory_type=memory_type, **kwargs)
    if memory_type == "memobrain":
        from bfcl_eval.memory.memobrain import MemoBrainMemory
        return MemoBrainMemory(memory_type=memory_type, **kwargs)
    if memory_type == "memos":
        from bfcl_eval.memory.memos_backend import MemosMemory
        return MemosMemory(**kwargs)
    if memory_type == "memoryos":
        from bfcl_eval.memory.memoryos_backend import MemoryosMemory
        return MemoryosMemory(**kwargs)
    if memory_type == "amem":
        from bfcl_eval.memory.amem_backend import AmemMemory
        return AmemMemory(**kwargs)
    if memory_type == "agent_kb":
        from bfcl_eval.memory.agent_kb_backend import AgentKBMemory
        return AgentKBMemory(**kwargs)
    if memory_type == "agent_workflow":
        from bfcl_eval.memory.agent_workflow_backend import AgentWorkflowMemory
        return AgentWorkflowMemory(**kwargs)
    if memory_type == "skillweaver":
        from bfcl_eval.memory.skillweaver_backend import SkillWeaverMemory
        return SkillWeaverMemory(**kwargs)
    if memory_type == "lightweight":
        from bfcl_eval.memory.lightweight_backend import LightweightMemory
        return LightweightMemory(**kwargs)
    if memory_type == "bm25":
        from bfcl_eval.memory.bm25_backend import Bm25Memory
        return Bm25Memory(**kwargs)
    if memory_type == "qwen3_embedding":
        from bfcl_eval.memory.qwen3_embedding_backend import Qwen3EmbeddingMemory
        return Qwen3EmbeddingMemory(**kwargs)
    if memory_type == "graphrag":
        from bfcl_eval.memory.graphrag_backend import GraphRagMemory
        return GraphRagMemory(**kwargs)
    if memory_type == "ace":
        from bfcl_eval.memory.ace_backend import ACEMemory
        return ACEMemory(**kwargs)
    if memory_type == "reasoning_bank":
        from bfcl_eval.memory.reasoning_bank_backend import ReasoningBankMemory
        return ReasoningBankMemory(**kwargs)
    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. Supported values: "
        f"none, mem0, memagent, memobrain, memos, memoryos, amem, "
        f"agent_kb, agent_workflow, skillweaver, lightweight, "
        f"bm25, qwen3_embedding, graphrag, ace, reasoning_bank"
    )


def build_memory_for_sample(
    memory_type: str,
    base_dir: Path,
    sample_id: str,
    **kwargs,
) -> Tuple[Optional[Memory], Optional[Path]]:
    """Build a fresh isolated Memory instance for a single sample.

    Returns (memory, storage_dir). Both are None when memory is disabled.
    The storage_dir is always freshly created and empty (clear=True).
    """
    if memory_type in ("none", None, ""):
        return None, None
    safe_id = re.sub(r"[^\w\-.]", "_", sample_id)
    storage_dir = Path(base_dir) / "memory_per_sample" / safe_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    mem = build_memory(
        memory_type,
        storage_dir=storage_dir,
        clear=True,
        readonly=False,
        **kwargs,
    )
    return mem, storage_dir
