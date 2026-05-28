"""Factory for memory backends. Add new types here."""

from .mem0_memory import Mem0Memory
from .bm25_memory import BM25Memory
from .qwen3_embedding_memory import Qwen3EmbeddingMemory
from .graphrag_memory import GraphRAGMemory
from .reasoning_bank import ReasoningBankMemory
from .ace_memory import ACEMemory
from .agent_kb_memory import AgentKBMemory
from .agent_workflow_memory import AgentWorkflowMemory
from .lightweight_memory import LightweightMemory
from .skillweaver_memory import SkillWeaverMemory

try:
    from .a_mem_memory import AMemMemory
    _AMEM_AVAILABLE = True
except ImportError:
    _AMEM_AVAILABLE = False

try:
    from .memos_memory import MemOSMemory
    _MEMOS_AVAILABLE = True
except ImportError:
    _MEMOS_AVAILABLE = False

try:
    from .memoryos_memory import MemoryOSMemory
    _MEMORYOS_AVAILABLE = True
except ImportError:
    _MEMORYOS_AVAILABLE = False


def build_memory(memory_type: str, **kwargs):
    """Instantiate the memory backend for ``memory_type``."""
    if memory_type == "mem0":
        return Mem0Memory(**kwargs)
    if memory_type == "bm25":
        return BM25Memory(**kwargs)
    if memory_type == "qwen3_embedding_4b":
        return Qwen3EmbeddingMemory(**kwargs)
    if memory_type == "graphrag":
        return GraphRAGMemory(**kwargs)
    if memory_type == "reasoning_bank":
        return ReasoningBankMemory(**kwargs)
    if memory_type == "ace":
        return ACEMemory(**kwargs)
    if memory_type == "agent_kb":
        return AgentKBMemory(**kwargs)
    if memory_type == "agent_workflow":
        return AgentWorkflowMemory(**kwargs)
    if memory_type == "lightweight":
        return LightweightMemory(**kwargs)
    if memory_type == "skillweaver":
        return SkillWeaverMemory(**kwargs)
    if memory_type == "a_mem":
        if not _AMEM_AVAILABLE:
            raise ImportError(
                "a_mem requires chromadb and A-mem to be installed. "
                "Run: pip install chromadb && pip install -e <path-to-A-mem>"
            )
        return AMemMemory(**kwargs)
    if memory_type == "memos":
        if not _MEMOS_AVAILABLE:
            raise ImportError(
                "memos requires MemOS to be installed. "
                "Run: pip install -e <path-to-MemOS>"
            )
        return MemOSMemory(**kwargs)
    if memory_type == "memoryos":
        if not _MEMORYOS_AVAILABLE:
            raise ImportError(
                "memoryos requires MemoryOS to be installed. "
                "Run: pip install -e <path-to-memoryos-pypi>"
            )
        return MemoryOSMemory(**kwargs)
    raise ValueError(
        f"Unknown --memory-type {memory_type!r}. "
        f"Supported: mem0, bm25, qwen3_embedding_4b, graphrag, reasoning_bank, ace, "
        f"agent_kb, agent_workflow, lightweight, skillweaver, a_mem, memos, memoryos"
    )
