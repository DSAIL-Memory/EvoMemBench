"""
Pluggable memory backends for CL-bench inference-with-memory.

Add a new implementation:
  1. Subclass ``Memory`` in e.g. ``my_memory.py``
  2. Register it in ``registry.build_memory``
  3. Extend ``--memory-type`` choices in ``infer_with_memory.py``
"""

from .base import Memory
from .registry import build_memory
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
except ImportError:
    AMemMemory = None  # type: ignore

try:
    from .memos_memory import MemOSMemory
except ImportError:
    MemOSMemory = None  # type: ignore

try:
    from .memoryos_memory import MemoryOSMemory
except ImportError:
    MemoryOSMemory = None  # type: ignore

from .trajectory import (
    format_trajectory,
    get_last_user_message,
    inject_memory_into_messages,
)
from .api_client import call_openai_api

__all__ = [
    "Memory",
    "build_memory",
    "Mem0Memory",
    "BM25Memory",
    "Qwen3EmbeddingMemory",
    "GraphRAGMemory",
    "ReasoningBankMemory",
    "ACEMemory",
    "AgentKBMemory",
    "AgentWorkflowMemory",
    "LightweightMemory",
    "SkillWeaverMemory",
    "AMemMemory",
    "MemOSMemory",
    "MemoryOSMemory",
    "format_trajectory",
    "get_last_user_message",
    "inject_memory_into_messages",
    "call_openai_api",
]
