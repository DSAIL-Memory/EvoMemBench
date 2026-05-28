"""Core types for the unified memory system (vendored from Flash-Searcher EvolveLab)."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryStatus(Enum):
    BEGIN = "begin"
    IN = "in"


class MemoryType(Enum):
    AGENT_KB = "agent_kb"
    AGENT_WORKFLOW_MEMORY = "agent_workflow_memory"
    LIGHTWEIGHT_MEMORY = "lightweight_memory"
    SKILLWEAVER = "skillweaver"


class MemoryItemType(Enum):
    TEXT = "text"
    API = "api"


@dataclass
class MemoryRequest:
    query: str
    context: str
    status: MemoryStatus
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class MemoryItem:
    id: str
    content: Any
    metadata: Dict[str, Any]
    score: Optional[float] = None
    type: MemoryItemType = MemoryItemType.TEXT


@dataclass
class MemoryResponse:
    memories: List[MemoryItem]
    memory_type: MemoryType
    total_count: int
    request_id: Optional[str] = None


@dataclass
class TrajectoryData:
    query: str
    trajectory: List[Dict[str, Any]]
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
