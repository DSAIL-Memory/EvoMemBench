"""MemOS memory provider for EvoMemBench-XbenchWebWalkQA.

Uses MemOS's GeneralTextMemory with Fine mode (LLM extraction):
  - take_in_memory: extract() extracts structured facts from trajectory messages,
    then add() stores them in Qdrant (persistent local path).
  - provide_memory: search() retrieves top-k TextualMemoryItem objects,
    returns item.memory (the extracted fact text).

Design principles:
  - Call GeneralTextMemory's standard API without modification or extra safeguards.
    If extract() fails (e.g. LLM overflow), nothing is stored — that is MemOS's behavior.
  - Qdrant with a local path provides persistence; no separate JSON sidecar needed.
  - LLM and embedder credentials are read from os.environ at init time.
"""

import os
import sys
import threading
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


class _TaskModelLLMAdapter:
    """Routes MemOS extractor_llm.generate() through the benchmark task_model for token counting."""

    def __init__(self, task_model):
        self._task_model = task_model

    def generate(self, messages, **kwargs) -> str:
        chat_msg = self._task_model(messages, max_tokens=32768)
        content = chat_msg.content or ""
        if not content:
            # Recover JSON from raw response when think-tag stripping emptied content.
            try:
                raw_content = chat_msg.raw.choices[0].message.content or ""
                if raw_content:
                    j_start = raw_content.find('{')
                    j_end = raw_content.rfind('}')
                    if j_start != -1 and j_end != -1:
                        content = raw_content[j_start:j_end + 1]
            except Exception:
                pass
        return content


def _coerce_content(content: Any) -> str:
    """Coerce OpenAI message content to str (handles list-of-parts for multi-modal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


class MemOSMemoryProvider(BaseMemoryProvider):
    """MemOS memory provider: LLM-extracted structured facts via GeneralTextMemory."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.MEMOS, config or {})
        self._top_k = self.config.get("top_k", 3)
        self._return_on_in = self.config.get("return_on_in", False)
        self._qdrant_path = self.config.get(
            "qdrant_path", "./storage/memos/qdrant"
        )
        self._collection_name = self.config.get("collection_name", "memos_xbench")
        self._vector_dim = self.config.get("vector_dimension", 1024)
        self._lock = threading.RLock()
        self._memory = None

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        # MemOS 的 memos/api/config.py:26 在 import 链路上会执行
        # load_dotenv(override=True)，从 /data/chimo/MemOS/.env 把 OPENAI_API_KEY /
        # OPENAI_API_BASE 等变量覆盖成 aigcbest 的值，污染后续 judge_model 创建。
        # 这里在 import 前后保存/恢复 env 来阻止污染。
        _preserved = {
            k: os.environ.get(k)
            for k in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
                      "MEMRADER_API_KEY", "MEMRADER_API_BASE")
        }
        try:
            from memos.configs.memory import GeneralTextMemoryConfig
            from memos.memories.textual.general import GeneralTextMemory
        except ImportError as e:
            print(f"[MemOS] import failed: {e}")
            return False
        finally:
            for k, v in _preserved.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        llm_cfg = self.config.get("llm", {}) or {}
        embed_cfg = self.config.get("embed", {}) or {}

        try:
            os.makedirs(self._qdrant_path, exist_ok=True)

            # Use model_validate to build configs from dicts — avoids Pydantic
            # re-validation bug where passing pre-created Factory instances causes
            # their model_validators to re-run with already-converted config objects.
            mem_cfg = GeneralTextMemoryConfig.model_validate({
                "extractor_llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": llm_cfg.get("model")
                            or os.getenv("ANALYSIS_MODEL", "deepseek-v3-2-251201"),
                        "api_key": llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", ""),
                        "api_base": llm_cfg.get("base_url") or os.getenv(
                            "OPENAI_BASE_URL", "https://api.openai.com/v1"
                        ),
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": self._collection_name,
                        "path": self._qdrant_path,
                        "vector_dimension": self._vector_dim,
                        "distance_metric": "cosine",
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": embed_cfg.get("model", "text-embedding-v4"),
                        "api_key": embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                        "base_url": embed_cfg.get("base_url") or os.getenv(
                            "DASHSCOPE_BASE_URL",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        ),
                        "embedding_dims": self._vector_dim,
                    },
                },
            })
            self._memory = GeneralTextMemory(mem_cfg)

            task_model = self.config.get("model")
            if task_model is not None:
                self._memory.extractor_llm = _TaskModelLLMAdapter(task_model)
                print("[MemOS] extractor_llm routed through benchmark task_model (tokens tracked)")
            else:
                print("[MemOS] WARNING: no task_model in config; extraction tokens will not be counted")

            count = len(self._memory.get_all())
            print(
                f"[MemOS] initialized (top_k={self._top_k}, "
                f"return_on_in={self._return_on_in}, "
                f"loaded={count} memories from Qdrant)"
            )
            return True
        except Exception as e:
            print(f"[MemOS] initialize failed: {e}")
            return False

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if self._memory is None:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(
                    memories=[],
                    memory_type=self.memory_type,
                    total_count=len(self._memory.get_all()),
                )

            try:
                results = self._memory.search(request.query, self._top_k)
            except Exception as e:
                print(f"[MemOS] search failed: {e}")
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            items: List[MemoryItem] = []
            for r in results:
                items.append(
                    MemoryItem(
                        id=r.id,
                        content=r.memory,
                        metadata=r.metadata.model_dump() if hasattr(r.metadata, "model_dump")
                            else (r.metadata if isinstance(r.metadata, dict) else {}),
                        type=MemoryItemType.TEXT,
                    )
                )

            return MemoryResponse(
                memories=items,
                memory_type=self.memory_type,
                total_count=len(self._memory.get_all()),
            )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._memory is None:
                return False, "MemOS not initialized"

            metadata = trajectory_data.metadata or {}
            if not metadata.get("is_correct", False):
                return False, "skipping incorrect trajectory"

            try:
                # Coerce trajectory messages to plain-text content
                messages = [
                    {
                        "role": msg.get("role", "user"),
                        "content": _coerce_content(msg.get("content", "")),
                    }
                    for msg in (trajectory_data.trajectory or [])
                    if msg.get("role") in ("user", "assistant", "system")
                ]

                if not messages:
                    return False, "empty trajectory"

                # Truncate conversation before passing to extract() to keep the total
                # prompt (SIMPLE_STRUCT_MEM_READER_PROMPT ~2K chars + conversation)
                # within a safe input budget and leave enough room for JSON output.
                # Keep the most recent messages (most relevant for memory extraction).
                _MAX_CONV_CHARS = 15_000
                total = sum(len(m["content"]) for m in messages)
                if total > _MAX_CONV_CHARS:
                    kept, budget = [], _MAX_CONV_CHARS
                    for m in reversed(messages):
                        c = m["content"]
                        if budget <= 0:
                            break
                        if len(c) > budget:
                            m = {"role": m["role"], "content": c[-budget:]}
                        kept.append(m)
                        budget -= len(m["content"])
                    messages = list(reversed(kept))

                # Fine mode: LLM extracts structured facts from trajectory
                extracted = self._memory.extract(messages)

                if not extracted:
                    return False, "extract() returned no items"

                self._memory.add(extracted)
                return True, f"added {len(extracted)} memory items (total={len(self._memory.get_all())})"

            except Exception as e:
                return False, f"take_in_memory error: {e}"
