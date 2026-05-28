"""Abstract memory backend interface."""

from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> tuple:
        """Return (memory_text, stats); stats includes latency_s and embed_usage where applicable."""
        pass

    @abstractmethod
    def extract(self, content: str, **kwargs) -> dict:
        """Persist new memory from trajectory text; return stats (latency_s, llm_usage, embed_usage)."""
        pass
