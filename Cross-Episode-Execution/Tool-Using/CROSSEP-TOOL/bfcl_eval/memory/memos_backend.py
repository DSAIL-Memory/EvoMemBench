"""MemosMemory — memos (MemOS GeneralTextMemory) backed Memory interface.

LLM calls are routed through the Ark batch endpoint (reusing ArkBatchClient).
Embeddings use DashScope text-embedding-v4 via OpenAI-compatible endpoint.
Token usage is recorded in metrics._TLS (same as Mem0Memory).

Registers "ark_batch" provider into MemOS's LLMFactory / LLMConfigFactory
and "universal_api_metered" into EmbedderFactory / EmbedderConfigFactory
at import time (idempotent via setdefault / hasattr guards).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.mem0_backend import _truncate_query_to_embed_limit

_UTILIZE_PREFIX = "\n\nRelevant memories from prior episodes:\n"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ark-batch LLM provider for MemOS
# ─────────────────────────────────────────────────────────────────────────────

def _register_ark_llm_provider() -> None:
    """Register 'ark_batch' backend into MemOS LLMFactory / LLMConfigFactory (idempotent)."""
    from memos.configs.llm import BaseLLMConfig, LLMConfigFactory
    from memos.llms.base import BaseLLM
    from memos.llms.factory import LLMFactory
    from pydantic import Field

    if "ark_batch" in LLMConfigFactory.backend_to_class:
        return

    from bfcl_eval.memory.ark_client import ArkBatchClient

    class _ArkBatchMemosLLMConfig(BaseLLMConfig):
        api_key: str = Field(..., description="Ark API key")
        api_base: str = Field(default="", description="Unused; Ark SDK ignores base_url")

    class _ArkBatchMemosLLM(BaseLLM):
        def __init__(self, config: _ArkBatchMemosLLMConfig):
            self.config = config
            self._ark = ArkBatchClient(api_key=config.api_key)

        def generate(self, messages, **kw) -> str:
            resp, dt = self._ark.chat_completions(
                model=self.config.model_name_or_path,
                messages=list(messages),
            )
            metrics.record_llm_call(
                latency_s=dt,
                input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
            )
            return resp.choices[0].message.content or ""

        def generate_stream(self, messages, **kw) -> Generator[str, None, None]:
            raise NotImplementedError("Streaming not supported for ArkBatch")

    LLMConfigFactory.backend_to_class["ark_batch"] = _ArkBatchMemosLLMConfig
    LLMFactory.backend_to_class["ark_batch"] = _ArkBatchMemosLLM


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metered universal-API embedder (records metrics.record_embed_call)
# ─────────────────────────────────────────────────────────────────────────────

def _register_metered_embedder() -> None:
    """Register 'universal_api_metered' backend (idempotent)."""
    from memos.configs.embedder import EmbedderConfigFactory, UniversalAPIEmbedderConfig
    from memos.embedders.factory import EmbedderFactory
    from memos.embedders.universal_api import UniversalAPIEmbedder

    if "universal_api_metered" in EmbedderFactory.backend_to_class:
        return

    class _MeteredUniversalAPIEmbedder(UniversalAPIEmbedder):
        """UniversalAPIEmbedder that records embedding token usage in metrics."""

        def _embed_batch(self, texts: list[str], client, model: str) -> list[list[float]]:
            t0 = time.time()
            response = client.embeddings.create(model=model, input=texts)
            elapsed = time.time() - t0
            metrics.record_embed_call(
                total_tokens=int(getattr(response.usage, "total_tokens", 0) or 0),
                latency_s=elapsed,
            )
            return [r.embedding for r in response.data]

    EmbedderFactory.backend_to_class["universal_api_metered"] = _MeteredUniversalAPIEmbedder
    EmbedderConfigFactory.backend_to_class["universal_api_metered"] = UniversalAPIEmbedderConfig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trajectory normalizer for memos extract
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_for_memos(trajectory: list[dict]) -> list[dict]:
    """Return a cleaned trajectory for GeneralTextMemory.extract.

    - Drops system messages (not useful for memory extraction).
    - Converts assistant messages with only tool_calls to a text representation.
    - Drops messages whose content is empty after normalization.
    """
    out = []
    for m in trajectory:
        role = m.get("role", "")
        if role == "system":
            continue
        content = m.get("content") or ""
        if role == "assistant" and not content:
            tool_calls = m.get("tool_calls") or []
            parts = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name", "")
                args = (tc.get("function") or {}).get("arguments", "")
                parts.append(f"[call {fn}({args})]")
            content = " ".join(parts)
        if content:
            out.append({"role": role, "content": content})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. MemosMemory — the Memory ABC implementation
# ─────────────────────────────────────────────────────────────────────────────

class MemosMemory(Memory):
    """Cross-episode memory backed by MemOS GeneralTextMemory.

    Storage layout under storage_dir/:
        qdrant/    — Qdrant local vector store (SQLite-backed)

    All LLM calls for fact extraction route through the Ark batch endpoint.
    Embeddings use DashScope text-embedding-v4 via OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        embed_model: str = "text-embedding-v4",
        embed_dims: int = 1024,
        collection_name: str = "bfcl_memos",
        top_k: int = 5,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        # Register custom providers before building MemoryConfigFactory
        # (field_validator reads backend_to_class at validation time).
        _register_ark_llm_provider()
        _register_metered_embedder()

        try:
            from memos.configs.memory import MemoryConfigFactory
            from memos.memories.factory import MemoryFactory
        except ImportError as exc:
            raise ImportError(
                "memos (MemOS) is required for --memory-type memos. "
                "Install it from EvoMemBench-Memory-Systems/MemOS."
            ) from exc

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        qdrant_path = storage_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)

        cfg = MemoryConfigFactory(
            backend="general_text",
            config={
                "extractor_llm": {
                    "backend": "ark_batch",
                    "config": {
                        "model_name_or_path": ark_model,
                        "api_key": ark_api_key,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": collection_name,
                        "distance_metric": "cosine",
                        "vector_dimension": embed_dims,
                        "path": str(qdrant_path),
                    },
                },
                "embedder": {
                    "backend": "universal_api_metered",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": embed_model,
                        "api_key": dashscope_api_key,
                        "base_url": dashscope_base_url,
                        "embedding_dims": embed_dims,
                    },
                },
            },
        )

        self._mem = MemoryFactory.from_config(cfg)
        self._top_k = top_k
        self._lock = threading.RLock()

        if clear:
            try:
                self._mem.delete_all()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        query = _truncate_query_to_embed_limit(system_prompt)
        try:
            items = self._mem.search(query=query, top_k=self._top_k)
        except Exception:
            items = []

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        if not items:
            return ""
        lines = "\n".join(f"- {item.memory}" for item in items if item.memory)
        return (_UTILIZE_PREFIX + lines) if lines else ""

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        norm = _normalize_for_memos(trajectory)
        if norm:
            with self._lock:
                try:
                    extracted = self._mem.extract(norm)
                    if extracted:
                        self._mem.add(extracted)
                except Exception:
                    pass

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "MemosMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
