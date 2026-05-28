"""
Core types for the unified memory system
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod


class MemoryStatus(Enum):
    """Status of memory operation"""
    BEGIN = "begin"
    IN = "in"


class MemoryType(Enum):
    """Type of memory framework"""
    AGENT_KB = "agent_kb"
    SKILLWEAVER = "skillweaver"
    LIGHTWEIGHT_MEMORY = "lightweight_memory"
    AGENT_WORKFLOW_MEMORY = "agent_workflow_memory"


PROVIDER_MAPPING = {
    MemoryType.AGENT_KB: ("AgentKBProvider", "agent_kb_provider"),
    MemoryType.SKILLWEAVER: ("SkillWeaverProvider", "skillweaver_provider"),
    MemoryType.LIGHTWEIGHT_MEMORY: ("LightweightMemoryProvider", "lightweight_memory_provider"),
    MemoryType.AGENT_WORKFLOW_MEMORY: ("AgentWorkflowMemoryProvider", "agent_workflow_memory_provider"),
}


class MemoryItemType(Enum):
    """Type of memory item content"""
    TEXT = "text"
    API = "api"


@dataclass
class MemoryRequest:
    """Request for memory retrieval"""
    query: str
    context: str
    status: MemoryStatus
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class MemoryItem:
    """Base memory item structure"""
    id: str
    content: Any
    metadata: Dict[str, Any]
    score: Optional[float] = None
    type: MemoryItemType = MemoryItemType.TEXT


@dataclass
class MemoryResponse:
    """Response containing retrieved memories"""
    memories: List[MemoryItem]
    memory_type: MemoryType
    total_count: int
    request_id: Optional[str] = None


@dataclass
class TrajectoryData:
    """Data structure for memory ingestion"""
    query: str
    trajectory: List[Dict[str, Any]]
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
