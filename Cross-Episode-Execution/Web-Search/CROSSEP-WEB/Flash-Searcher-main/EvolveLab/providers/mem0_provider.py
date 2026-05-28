"""mem0 memory provider for EvoMemBench-XbenchWebWalkQA.

Uses the official mem0 SDK with Volcengine Ark batch-inference LLM and DashScope
text-embedding-v4 via OpenAI-compatible API:
  - take_in_memory: add() passes full trajectory messages to mem0; the LLM
    extracts key facts and stores them in a local Qdrant vector DB.
  - provide_memory: search() retrieves top-k relevant memories.

Implementation notes:
  - mem0 must be installed via `pip install -e <path-to-EvoMemBench-Memory-Systems/mem0>`
    so that both `import mem0` and `importlib.metadata.version("mem0ai")` resolve
    from the same local source — no sys.path injection required.
  - mem0's LlmConfig validator only whitelists a fixed set of providers. We construct
    Memory with provider="openai" + a dummy api_key purely to pass that validator,
    then immediately replace self._memory.llm with ArkBatchLLM so all extraction
    calls route through the Ark batch endpoint. The dummy OpenAI client is never
    invoked.
  - mem0 v2.0.0: search() uses filters={"user_id": ...}, add() still uses
    user_id= top-level kwarg (inconsistent but intentional).
  - DashScope text-embedding-v4 does NOT accept a dimensions parameter via
    OpenAI-compatible API — do not set embedding_model_dims in embedder config.
  - text-embedding-v4 produces 1024-dim vectors; Qdrant must be told 1024.
  - Qdrant embedded mode can lock ~/.mem0/migrations_qdrant/.lock from a
    prior run; we remove it before creating Memory to avoid startup hang.
"""

import json
import os
import sys
import threading
import traceback
import uuid
from typing import Any, List

try:
    from mem0.configs.llms.base import BaseLlmConfig as _BaseLlmConfig
    from mem0.llms.base import LLMBase as _LLMBase
except ImportError:
    _BaseLlmConfig = object  # type: ignore[assignment,misc]
    _LLMBase = object  # type: ignore[assignment,misc]

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


class ArkBatchLLM(_LLMBase):
    """LLMBase adapter that routes mem0 LLM extraction through Volcengine Ark's batch endpoint.

    mem0's Memory calls self.llm.generate_response(...) for fact extraction.
    We inject an instance of this class as Memory.llm immediately after Memory()
    construction, so the placeholder OpenAILLM created by mem0's factory is never used.

    Requires:
      - volcenginesdkarkruntime installed (already a transitive dep of CROSSEP-WEB)
      - DEEPSEEK_API_KEY or OPENAI_API_KEY env var  (Ark API key)
      - BATCH_MODEL env var (Ark batch-inference endpoint ID, e.g. ep-bi-…)
    """

    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.1):
        # Bypass _LLMBase.__init__ — it tries to re-construct a typed config and
        # calls _validate_config; we already have all values we need.
        self.config = _BaseLlmConfig(model=model, temperature=temperature)
        from volcenginesdkarkruntime import Ark
        self._ark_client = Ark(api_key=api_key, base_url=base_url)

    def generate_response(self, messages, response_format=None,
                          tools=None, tool_choice="auto", **kwargs):
        params: dict = {"model": self.config.model, "messages": messages}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self._ark_client.batch.chat.completions.create(**params)

        # Mirror mem0.llms.openai.OpenAILLM._parse_response return shape:
        #   no tools  → str
        #   has tools → {"content": str, "tool_calls": [{"name": str, "arguments": dict}]}
        if not tools:
            return response.choices[0].message.content
        tool_calls = []
        for tc in (response.choices[0].message.tool_calls or []):
            tool_calls.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })
        return {"content": response.choices[0].message.content, "tool_calls": tool_calls}


class Mem0Provider(BaseMemoryProvider):
    """mem0 memory provider: LLM-extracted facts via the official mem0 SDK."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.MEM0, config or {})
        self._top_k = self.config.get("top_k", 3)
        self._return_on_in = self.config.get("return_on_in", False)
        self._qdrant_path = self.config.get("qdrant_path", "./storage/mem0/qdrant")
        self._collection_name = self.config.get("collection_name", "mem0_xbench")
        self._user_id = self.config.get("user_id", "benchmark_user")
        self._lock = threading.RLock()
        self._memory = None

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        # Remove stale Qdrant embedded-mode lock to avoid startup hang
        lock_path = os.path.expanduser("~/.mem0/migrations_qdrant/.lock")
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                print(f"[mem0] removed stale Qdrant lock: {lock_path}")
            except OSError as e:
                print(f"[mem0] WARNING: could not remove Qdrant lock {lock_path}: {e}")

        try:
            from mem0.configs.base import MemoryConfig
            from mem0.llms.configs import LlmConfig
            from mem0.embeddings.configs import EmbedderConfig
            from mem0.vector_stores.configs import VectorStoreConfig
            from mem0.memory.main import Memory
        except ImportError as e:
            print(f"[mem0] import failed: {e}")
            return False

        llm_cfg = self.config.get("llm", {}) or {}
        embed_cfg = self.config.get("embed", {}) or {}

        try:
            os.makedirs(self._qdrant_path, exist_ok=True)

            # provider="openai" + dummy api_key: used solely to pass mem0's
            # LlmConfig pydantic whitelist validator. The resulting OpenAILLM
            # instance is replaced with ArkBatchLLM immediately after Memory().
            mem_config = MemoryConfig(
                llm=LlmConfig(
                    provider="openai",
                    config={"model": "placeholder", "api_key": "dummy"},
                ),
                embedder=EmbedderConfig(
                    provider="openai",
                    config={
                        "model": embed_cfg.get("model", "text-embedding-v4"),
                        "api_key": embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                        "openai_base_url": embed_cfg.get("base_url") or os.getenv(
                            "DASHSCOPE_BASE_URL",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        ),
                        # NOTE: do NOT set embedding_model_dims here — DashScope
                        # text-embedding-v4 rejects the 'dimensions' parameter via the
                        # OpenAI-compatible API.
                    },
                ),
                vector_store=VectorStoreConfig(
                    provider="qdrant",
                    config={
                        "collection_name": self._collection_name,
                        "embedding_model_dims": 1024,  # text-embedding-v4 outputs 1024-dim, not 1536
                        "path": self._qdrant_path,
                        "on_disk": True,
                    },
                ),
                history_db_path=os.path.join(self._qdrant_path, "history.db"),
            )

            self._memory = Memory(config=mem_config)

            # Inject Ark batch LLM — replaces the dummy OpenAILLM just constructed.
            self._memory.llm = ArkBatchLLM(
                model=(
                    llm_cfg.get("model")
                    or os.getenv("BATCH_MODEL")
                    or os.getenv("DEFAULT_MODEL", "deepseek-chat")
                ),
                api_key=(
                    llm_cfg.get("api_key")
                    or os.getenv("DEEPSEEK_API_KEY")
                    or os.getenv("OPENAI_API_KEY", "")
                ),
                base_url=(
                    llm_cfg.get("base_url")
                    or os.getenv("DEEPSEEK_BASE_URL")
                    or os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
                ),
            )
            all_mems = self._memory.get_all(filters={"user_id": self._user_id})
            count = len(all_mems.get("results", []))
            print(
                f"[mem0] initialized (top_k={self._top_k}, "
                f"return_on_in={self._return_on_in}, "
                f"loaded={count} memories)"
            )
            return True

        except Exception as e:
            print(f"[mem0] initialize failed: {e}")
            traceback.print_exc()
            return False

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if self._memory is None:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            try:
                # v2.0.0: search() requires entity id via filters={} dict, not top-level kwarg
                results = self._memory.search(
                    request.query,
                    top_k=self._top_k,
                    filters={"user_id": self._user_id},
                )
            except Exception as e:
                print(f"[mem0] search failed: {e}")
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            items: List[MemoryItem] = []
            for r in results.get("results", []):
                meta = {k: v for k, v in r.items() if k not in ("id", "memory", "score")}
                items.append(
                    MemoryItem(
                        id=r.get("id", str(uuid.uuid4())),
                        content=r.get("memory", ""),
                        metadata=meta,
                        score=r.get("score"),
                        type=MemoryItemType.TEXT,
                    )
                )

            return MemoryResponse(
                memories=items,
                memory_type=self.memory_type,
                total_count=len(items),
            )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._memory is None:
                return False, "mem0 not initialized"

            try:
                messages = [
                    {
                        "role": msg.get("role", "user"),
                        "content": _coerce_content(msg.get("content", "")),
                    }
                    for msg in (trajectory_data.trajectory or [])
                    if msg.get("role") in ("user", "assistant", "system")
                ]

                if not messages:
                    # Fallback: encode query + result as a single user message
                    text = trajectory_data.query or ""
                    if trajectory_data.result:
                        text += f"\nResult: {trajectory_data.result}"
                    if not text.strip():
                        return False, "empty trajectory"
                    messages = [{"role": "user", "content": text}]

                meta = trajectory_data.metadata or {}
                is_correct = meta.get("is_correct", True)
                messages = [
                    {"role": "system", "content": f"Trajectory outcome — Correct: {bool(is_correct)}"},
                    *messages,
                ]

                # v2.0.0: add() still uses user_id= as top-level keyword arg (not filters={})
                result = self._memory.add(messages, user_id=self._user_id)
                stored = len(result.get("results", []))
                return True, f"stored {stored} memory items"

            except Exception as e:
                return False, f"take_in_memory error: {e}"
