from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MemoryCallStats:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    cached_tokens: int = 0
    latency: float = 0.0

    def __add__(self, other: "MemoryCallStats") -> "MemoryCallStats":
        return MemoryCallStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            latency=self.latency + other.latency,
        )

    def to_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}input_tokens": self.input_tokens,
            f"{prefix}output_tokens": self.output_tokens,
            f"{prefix}embedding_tokens": self.embedding_tokens,
            f"{prefix}cached_tokens": self.cached_tokens,
            f"{prefix}latency": self.latency,
        }


class BaseMemory(ABC):
    def __init__(self, memory_type: str, read_only: bool = False):
        self.memory_type = memory_type
        self.read_only = read_only

    @abstractmethod
    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        """Retrieve memories using conversation[2] (first env observation) as query.

        Returns (modified_conversation, stats). Injects retrieved memories into
        conversation[0]["content"] (system prompt). Returns conversation unchanged
        if no memories are found.
        """

    @abstractmethod
    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        """Extract and store memories from the completed trajectory.

        Backends that only ingest successful trajectories (e.g. AgentKB, AWM)
        treat any non-None ``reward`` as the success signal and skip the
        ingestion when ``reward < 1``. Backends that store every trajectory
        ignore ``reward``.

        No-op returning zero stats when self.read_only is True.
        """

    @abstractmethod
    def load_from_disk(self, **kwargs) -> None:
        """Load or reload the memory store from local storage.

        kwargs are backend-specific (e.g. vector_store_path, db_path).
        """


def create_memory(memory_type: str, read_only: bool = False, **kwargs) -> BaseMemory:
    """Factory function for constructing a memory backend by name."""
    if memory_type == "mem0":
        from .mem0_adapter import Mem0Memory
        return Mem0Memory(read_only=read_only, **kwargs)
    if memory_type == "ace":
        from .ace_adapter import AceMemory
        return AceMemory(read_only=read_only, **kwargs)
    if memory_type == "reasoning_bank":
        from .reasoning_bank_adapter import ReasoningBankMemory
        return ReasoningBankMemory(read_only=read_only, **kwargs)
    if memory_type == "bm25":
        from .bm25_adapter import Bm25Memory
        return Bm25Memory(read_only=read_only, **kwargs)
    if memory_type == "qwen3_embedding":
        from .qwen3_embedding_adapter import Qwen3EmbeddingMemory
        return Qwen3EmbeddingMemory(read_only=read_only, **kwargs)
    if memory_type == "graphrag":
        from .graphrag_adapter import GraphRagMemory
        return GraphRagMemory(read_only=read_only, **kwargs)
    if memory_type == "agent_kb":
        from .agent_kb_adapter import AgentKBMemory
        return AgentKBMemory(read_only=read_only, **kwargs)
    if memory_type in ("agent_workflow_memory", "awm"):
        from .agent_workflow_adapter import AgentWorkflowMemory
        return AgentWorkflowMemory(read_only=read_only, **kwargs)
    if memory_type in ("lightweight_memory", "lwm"):
        from .lightweight_memory_adapter import LightweightMemory
        return LightweightMemory(read_only=read_only, **kwargs)
    if memory_type in ("skillweaver", "sw"):
        from .skillweaver_adapter import SkillWeaverMemory
        return SkillWeaverMemory(read_only=read_only, **kwargs)
    if memory_type == "amem":
        from .amem_adapter import AMemMemory
        return AMemMemory(read_only=read_only, **kwargs)
    if memory_type == "memoryos":
        from .memoryos_adapter import MemoryOSMemory
        return MemoryOSMemory(read_only=read_only, **kwargs)
    if memory_type == "memos":
        from .memos_adapter import MemosMemory
        return MemosMemory(read_only=read_only, **kwargs)
    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. "
        "Supported: ['mem0', 'ace', 'reasoning_bank', 'bm25', 'qwen3_embedding', 'graphrag', "
        "'agent_kb', 'agent_workflow_memory', 'lightweight_memory', 'skillweaver', "
        "'amem', 'memoryos', 'memos']"
    )
