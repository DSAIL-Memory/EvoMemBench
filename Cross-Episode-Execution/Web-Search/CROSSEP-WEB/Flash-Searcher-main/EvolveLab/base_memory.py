"""
Base memory provider interface
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .memory_types import MemoryRequest, MemoryResponse, TrajectoryData, MemoryType


class BaseMemoryProvider(ABC):
    """Abstract base class for memory providers"""
    
    def __init__(self, memory_type: MemoryType, config: Optional[dict] = None):
        self.memory_type = memory_type
        self.config = config or {}
    
    @abstractmethod
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Retrieve relevant memories based on query, context and status
        
        Args:
            request: MemoryRequest containing query, context, status and optional params
            
        Returns:
            MemoryResponse containing relevant memories
        """
        pass
    
    @abstractmethod
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """
        Store/ingest new memory from trajectory data

        Args:
            trajectory_data: TrajectoryData containing query, trajectory and metadata

        Returns:
            tuple[bool, str]: (Success status of memory ingestion, Description of absorbed memory)
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the memory provider (load existing data, setup indices, etc.)
        
        Returns:
            bool: Success status of initialization
        """
        pass
    
    def get_memory_type(self) -> MemoryType:
        """Get the type of this memory provider"""
        return self.memory_type
    
    def get_config(self) -> dict:
        """Get the configuration of this memory provider"""
        return self.config.copy()