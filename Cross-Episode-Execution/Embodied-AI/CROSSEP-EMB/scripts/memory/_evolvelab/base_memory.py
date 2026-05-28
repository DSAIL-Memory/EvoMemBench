"""Base memory provider interface (vendored from Flash-Searcher EvolveLab)."""

from abc import ABC, abstractmethod
from typing import Optional

from .memory_types import MemoryRequest, MemoryResponse, TrajectoryData, MemoryType


class BaseMemoryProvider(ABC):

    def __init__(self, memory_type: MemoryType, config: Optional[dict] = None):
        self.memory_type = memory_type
        self.config = config or {}

    @abstractmethod
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        ...

    @abstractmethod
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        ...

    @abstractmethod
    def initialize(self) -> bool:
        ...

    def get_memory_type(self) -> MemoryType:
        return self.memory_type

    def get_config(self) -> dict:
        return self.config.copy()
