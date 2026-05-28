"""Mem0Memory — the mem0-backed implementation of the Memory interface.

All mem0 imports are deferred to __init__ so that importing this module
does not fail when mem0 is not installed.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.metrics import MemoryUsageLog

_UTILIZE_PREFIX = "\n\nRelevant memories from prior episodes:\n"

# text-embedding-v4 documents an 8192-token input limit.
# We stay 192 tokens below that to absorb differences between cl100k_base
# (used here for counting) and DashScope's internal tokenizer.
_EMBED_BUDGET_TOKENS = 8000

try:
    import tiktoken as _tiktoken
    _ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    import warnings
    warnings.warn(
        "tiktoken not found; falling back to char-based embedding truncation "
        f"({_EMBED_BUDGET_TOKENS * 4} chars).",
        RuntimeWarning,
        stacklevel=1,
    )
    _ENC = None


def _truncate_query_to_embed_limit(text: str) -> str:
    """Return text truncated to fit text-embedding-v4's documented API limit.

    Preserves the tail so the task-specific framing (appended last by
    system_prompt_pre_processing_chat_model) remains in the query.
    No-op when the text is already within budget.
    """
    if _ENC is not None:
        tokens = _ENC.encode(text)
        if len(tokens) > _EMBED_BUDGET_TOKENS:
            return _ENC.decode(tokens[-_EMBED_BUDGET_TOKENS:])
        return text
    # char-based fallback when tiktoken is unavailable
    char_limit = _EMBED_BUDGET_TOKENS * 4
    return text[-char_limit:] if len(text) > char_limit else text


class Mem0Memory(Memory):
    """Persistent memory backed by the mem0 library.

    Storage layout under storage_dir/:
        history.db     — SQLite fact history (SQLiteManager)
        qdrant/        — Qdrant local vector store

    Both the LLM (fact extraction) and the embedder (search/add) are wired
    through providers registered at construction time:
        LLM:     "ark_batch"       → ArkBatchLLM
        Embedder:"openai_dashscope"→ DashscopeEmbedder
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
        user_id: str = "bfcl_run",
        top_k: int = 5,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        try:
            import mem0 as _mem0_module
            from mem0.configs.base import MemoryConfig
            from mem0.embeddings.configs import EmbedderConfig
            from mem0.llms.configs import LlmConfig
            from mem0.utils.factory import EmbedderFactory, LlmFactory
            from mem0.vector_stores.configs import VectorStoreConfig
        except ImportError as exc:
            raise ImportError(
                "mem0 is required for --memory-type mem0. "
                "Install it: pip install -e bfcl_eval/official_implentations/mem0"
            ) from exc

        # Register custom providers (idempotent across calls).
        LlmFactory.register_provider(
            "ark_batch",
            "bfcl_eval.memory.ark_llm.ArkBatchLLM",
        )
        EmbedderFactory.provider_to_class[
            "openai_dashscope"
        ] = "bfcl_eval.memory.dashscope_embedder.DashscopeEmbedder"

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        (storage_dir / "qdrant").mkdir(parents=True, exist_ok=True)

        # Bypass provider whitelist validators in LlmConfig / EmbedderConfig by
        # using model_construct (Pydantic v2 skips field validators).
        llm_cfg = LlmConfig.model_construct(
            provider="ark_batch",
            config={"model": ark_model, "api_key": ark_api_key},
        )
        embed_cfg = EmbedderConfig.model_construct(
            provider="openai_dashscope",
            config={
                "openai_base_url": dashscope_base_url,
                "api_key": dashscope_api_key,
                "model": embed_model,
                "embedding_dims": embed_dims,
            },
        )
        # VectorStoreConfig uses normal init so its model_validator runs and
        # converts the dict into a typed QdrantConfig (needed for .collection_name).
        vec_cfg = VectorStoreConfig(
            provider="qdrant",
            config={
                "path": str(storage_dir / "qdrant"),
                "collection_name": "bfcl_memories",
                "embedding_model_dims": embed_dims,
                "on_disk": True,
            },
        )
        # Use model_construct on MemoryConfig to avoid re-validating sub-models.
        mem_cfg = MemoryConfig.model_construct(
            llm=llm_cfg,
            embedder=embed_cfg,
            vector_store=vec_cfg,
            history_db_path=str(storage_dir / "history.db"),
            reranker=None,
            version="v1.1",
            custom_instructions=None,
        )

        self._mem = _mem0_module.Memory(mem_cfg)
        self._user_id = user_id
        self._top_k = top_k
        self._lock = threading.RLock()

        if clear:
            self._mem.reset()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        """Search for relevant memories and return a snippet to append to system_prompt."""
        # Use the full system prompt as the retrieval query. Truncate from the
        # head only when the prompt exceeds the embedder's documented 8192-token
        # API limit (DashScope text-embedding-v4), preserving the task-specific
        # tail (the section appended last by system_prompt_pre_processing_chat_model).
        query = _truncate_query_to_embed_limit(system_prompt)
        try:
            results = self._mem.search(
                query=query,
                filters={"user_id": self._user_id},
                top_k=self._top_k,
            )
            hits = results.get("results", [])
        except Exception:
            hits = []

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        if not hits:
            return ""
        lines = "\n".join(f"- {h['memory']}" for h in hits if h.get("memory"))
        return (_UTILIZE_PREFIX + lines) if lines else ""

    def update(self, trajectory: list[dict]) -> None:
        """Ingest a completed trajectory into the memory store."""
        if self.readonly:
            return
        with self._lock:
            try:
                self._mem.add(
                    messages=trajectory,
                    user_id=self._user_id,
                    infer=True,
                )
            except Exception:
                pass  # errors are captured by the caller's try/except

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "Mem0Memory":
        """Load from an existing on-disk store (clear=False by default)."""
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
