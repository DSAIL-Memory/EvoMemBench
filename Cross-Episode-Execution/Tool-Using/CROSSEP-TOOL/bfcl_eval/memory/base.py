"""Abstract Memory interface and MemoryUsageLog."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from bfcl_eval.memory import metrics
from bfcl_eval.memory.metrics import MemoryUsageLog


class Memory(ABC):
    """Pluggable memory layer for cross-episode learning.

    Lifecycle per sample (called from BatchArkInferenceRunner.run_sample):
        memory.begin_sample()                      # reset per-thread metrics
        snippet = memory.utilize(system_prompt)    # retrieve → append to prompt
        … inference …
        memory.update(full_message_history)        # ingest trajectory (skipped if readonly)
        usage = memory.drain_usage()               # collect metrics
    """

    def __init__(self, *, clear: bool = False, readonly: bool = False, **backend_kwargs):
        self.readonly = readonly

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def utilize(self, system_prompt: str) -> str:
        """Retrieve relevant memories and return a text snippet to append to system_prompt.

        Returns '' when there are no relevant memories yet (e.g. empty store).
        """

    @abstractmethod
    def update(self, trajectory: list[dict]) -> None:
        """Ingest a completed sample's full message history into the memory store.

        No-op when self.readonly is True.
        trajectory: list of {"role": ..., "content": ...} dicts (full_message_history).
        """

    @classmethod
    @abstractmethod
    def load_from_disk(cls, **backend_kwargs) -> "Memory":
        """Construct a Memory instance seeded from an existing on-disk store.

        backend_kwargs: storage-location params specific to each backend
        (e.g. storage_dir= for mem0, db_path= for a hypothetical future backend).
        """

    # ------------------------------------------------------------------ #
    # Concrete helpers (metrics lifecycle)                                 #
    # ------------------------------------------------------------------ #

    def begin_sample(self) -> None:
        metrics.begin_sample()

    def drain_usage(self) -> MemoryUsageLog:
        return metrics.drain()
