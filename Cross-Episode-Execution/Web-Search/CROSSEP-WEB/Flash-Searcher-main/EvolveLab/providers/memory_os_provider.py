"""MemoryOS memory provider for EvoMemBench-XbenchWebWalkQA.

Uses MemoryOS's hierarchical three-tier memory (short→mid→long term):
  - take_in_memory: add_memory(query, result) stores a QA pair — the natural
    unit MemoryOS is designed for. Full trajectory is NOT stored to avoid
    the snowball overflow problem seen with A-mem.
  - provide_memory: retriever.retrieve_context() fetches pages + knowledge
    without triggering LLM response generation (unlike get_response()).

Token counting:
  - Memoryos.client and Memoryos.updater.client are replaced with
    _TaskModelClientAdapter, routing all internal LLM calls through
    the benchmark task_model so tokens are counted by TokenCounter.

Embedding:
  - Uses text-embedding-v4 via DashScope API (consistent with a_mem/memos).
    DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL are read from os.environ.
"""

import os
import sys
import threading
import uuid
from typing import Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)


class _TaskModelClientAdapter:
    """Replaces MemoryOS's OpenAIClient to route LLM calls through benchmark task_model."""

    def __init__(self, task_model):
        self._task_model = task_model

    def chat_completion(self, model: str, messages: list,
                        temperature: float = 0.7, max_tokens: int = 32768) -> str:
        chat_msg = self._task_model(messages, max_tokens=max_tokens)
        return chat_msg.content or ""


class MemoryOSProvider(BaseMemoryProvider):
    """MemoryOS memory provider: hierarchical short/mid/long-term memory via QA pairs."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.MEMORY_OS, config or {})
        self._top_k = self.config.get("top_k", 3)
        self._user_id = self.config.get("user_id", "benchmark_user")
        self._storage_path = self.config.get("storage_path", "./storage/memory_os")
        self._return_on_in = self.config.get("return_on_in", False)
        self._lock = threading.RLock()
        self._memory = None

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            from memoryos import Memoryos
        except ImportError as e:
            print(f"[MemoryOS] import failed: {e}")
            return False

        try:
            os.makedirs(self._storage_path, exist_ok=True)

            embed_cfg = self.config.get("embed", {}) or {}
            self._memory = Memoryos(
                user_id=self._user_id,
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
                data_storage_path=self._storage_path,
                llm_model=self.config.get("llm_model") or os.getenv("ANALYSIS_MODEL", "deepseek-v3-2-251201"),
                embedding_model_name=embed_cfg.get("model") or self.config.get("embedding_model", "text-embedding-v4"),
                embedding_model_kwargs={
                    "api_key": embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                    "base_url": embed_cfg.get("base_url") or os.getenv(
                        "DASHSCOPE_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ),
                },
                short_term_capacity=self.config.get("short_term_capacity", 10),
            )

            task_model = self.config.get("model")
            if task_model is not None:
                adapter = _TaskModelClientAdapter(task_model)
                self._memory.client = adapter
                self._memory.updater.client = adapter
                print("[MemoryOS] client routed through benchmark task_model (tokens tracked)")
            else:
                print("[MemoryOS] WARNING: no task_model in config; LLM tokens will not be counted")

            st_count = len(self._memory.short_term_memory.get_all())
            print(
                f"[MemoryOS] initialized (top_k={self._top_k}, "
                f"return_on_in={self._return_on_in}, "
                f"short_term_loaded={st_count})"
            )
            return True
        except Exception as e:
            print(f"[MemoryOS] initialize failed: {e}")
            return False

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if self._memory is None:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            try:
                ctx = self._memory.retriever.retrieve_context(
                    request.query, self._user_id
                )
            except Exception as e:
                print(f"[MemoryOS] retrieve_context failed: {e}")
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            items: List[MemoryItem] = []

            for page in ctx.get("retrieved_pages", []):
                u = page.get("user_input", "")
                a = page.get("agent_response", "")
                content = f"Q: {u}\nA: {a}" if u or a else str(page)
                items.append(MemoryItem(
                    id=page.get("page_id", str(uuid.uuid4())),
                    content=content,
                    metadata={"source": "mid_term", "timestamp": page.get("timestamp", "")},
                    type=MemoryItemType.TEXT,
                ))

            for k in ctx.get("retrieved_user_knowledge", []):
                items.append(MemoryItem(
                    id=str(uuid.uuid4()),
                    content=k.get("knowledge", ""),
                    metadata={"source": "long_term_user", "timestamp": k.get("timestamp", "")},
                    type=MemoryItemType.TEXT,
                ))

            for k in ctx.get("retrieved_assistant_knowledge", []):
                items.append(MemoryItem(
                    id=str(uuid.uuid4()),
                    content=k.get("knowledge", ""),
                    metadata={"source": "long_term_assistant", "timestamp": k.get("timestamp", "")},
                    type=MemoryItemType.TEXT,
                ))

            return MemoryResponse(
                memories=items[: self._top_k],
                memory_type=self.memory_type,
                total_count=len(items),
            )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._memory is None:
                return False, "MemoryOS not initialized"

            metadata = trajectory_data.metadata or {}
            if not metadata.get("is_correct", False):
                return False, "skipping incorrect trajectory"

            try:
                user_input = trajectory_data.query or ""
                agent_response = str(trajectory_data.result or "")

                if not user_input:
                    return False, "empty query"

                self._memory.add_memory(
                    user_input=user_input,
                    agent_response=agent_response,
                )
                st_count = len(self._memory.short_term_memory.get_all())
                return True, f"memory added (short_term_size={st_count})"

            except Exception as e:
                return False, f"take_in_memory error: {e}"
