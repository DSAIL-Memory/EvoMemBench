"""
Core types for the unified memory system
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod


class MemoryStatus(Enum):
    """Status of memory operation"""
    # only these two status
    BEGIN = "begin"
    IN = "in"


class MemoryType(Enum):
    """
    Type of memory framework
    
    IMPORTANT: This enum contains ALL existing memory systems in the codebase.
    When generating new memory systems:
    - DO NOT duplicate any existing memory type names listed below
    """
    AGENT_KB = "agent_kb"
    SKILLWEAVER = "skillweaver"
    MOBILEE = "mobilee"
    EXPEL = "expel"
    LIGHTWEIGHT_MEMORY = "lightweight_memory"
    CEREBRA_FUSION_MEMORY = "cerebra_fusion_memory"
    VOYAGER = "voyager"
    DILU = "dilu"
    GENERATIVE = "generative"
    MEMP = "memp"
    DYNAMIC_CHEATSHEET = "dynamic_cheatsheet"
    AGENT_WORKFLOW_MEMORY = "agent_workflow_memory"
    EVOLVER = "evolver"
    BM25 = "bm25"
    QWEN3_EMBEDDING = "qwen3_embedding_4b"
    GRAPHRAG = "graphrag"
    MEM0 = "mem0"
    ACE = "ace"
    REASONING_BANK = "reasoning_bank"
    A_MEM = "a_mem"
    MEMOS = "memos"
    MEMORY_OS = "memory_os"
# add new memory type upside this line(Enum)

# Provider mapping for dynamic loading
# Format: MemoryType -> (ClassName, ModuleName)
# 
# CRITICAL: When adding new memory systems, ensure:
# 1. The names are NOT duplicates of existing systems listed below
# 2. The naming is innovative, distinctive, creative, and captivating
# 3. The system has a completely unique identity
PROVIDER_MAPPING = {
    MemoryType.AGENT_KB: ("AgentKBProvider", "agent_kb_provider"),
    MemoryType.SKILLWEAVER: ("SkillWeaverProvider", "skillweaver_provider"),
    MemoryType.MOBILEE: ("MobileEProvider", "mobilee_provider"),
    MemoryType.EXPEL: ("ExpeLProvider", "expel_provider"),
    MemoryType.LIGHTWEIGHT_MEMORY: ("LightweightMemoryProvider", "lightweight_memory_provider"),
    MemoryType.CEREBRA_FUSION_MEMORY: ("CerebraFusionMemoryProvider", "cerebra_fusion_memory_provider"),
    MemoryType.DILU: ("DiluMemoryProvider","dilu_memory_provider"),
    MemoryType.GENERATIVE: ("GenerativeMemoryProvider","generative_memory_provider"),
    MemoryType.VOYAGER: ("VoyagerMemoryProvider", "voyager_memory_provider"),
    MemoryType.MEMP:("MempMemoryProvider","memp_memory_provider"),
    MemoryType.DYNAMIC_CHEATSHEET:("DynamicCheatsheetProvider","dynamic_cheatsheet_provider"),
    MemoryType.AGENT_WORKFLOW_MEMORY: ("AgentWorkflowMemoryProvider", "agent_workflow_memory_provider"),
    MemoryType.EVOLVER: ("EvolverMemoryProvider", "evolver_memory_provider"),
    MemoryType.BM25: ("BM25MemoryProvider", "bm25_provider"),
    MemoryType.QWEN3_EMBEDDING: ("Qwen3EmbeddingProvider", "qwen3_embedding_provider"),
    MemoryType.GRAPHRAG: ("GraphRAGProvider", "graphrag_provider"),
    MemoryType.MEM0: ("Mem0Provider", "mem0_provider"),
    MemoryType.ACE: ("ACEProvider", "ace_provider"),
    MemoryType.REASONING_BANK: ("ReasoningBankProvider", "reasoning_bank_provider"),
    MemoryType.A_MEM:     ("AMemMemoryProvider",  "a_mem_provider"),
    MemoryType.MEMOS:     ("MemOSMemoryProvider", "memos_provider"),
    MemoryType.MEMORY_OS: ("MemoryOSProvider",    "memory_os_provider"),
# add new memory type upside this line(PROVIDER_MAPPING)
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