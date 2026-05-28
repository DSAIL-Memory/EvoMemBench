"""A-mem (Agentic Memory) adapter for AgentGym ALFWorld evaluation.

Code structure mirrors memos_adapter.py.
A-mem is a Zettelkasten-style semantic memory evolution system backed by ChromaDB.

Key design decisions:
- DashScope text-embedding-v4 injected via custom ChromaDB EmbeddingFunction
  (avoids loading SentenceTransformer; all embeddings go through DashScope API)
- Persistence: JSON sidecar file + fast reindex via retriever.add_document()
  (PersistentChromaRetriever hardcodes SentenceTransformer, cannot be used)
- LLM token counting: tiktoken estimation wrapping llm_controller.llm.get_completion()
- Embedding token counting: real DashScope API usage via _DashScopeEmbeddingFunction

Install: pip install -e ../../../EvoMemBench-Memory-Systems/A-mem
"""
from __future__ import annotations

import json
import logging
import os
import time
import threading
from copy import deepcopy
from typing import Any

from .base import BaseMemory, MemoryCallStats

logger = logging.getLogger(__name__)

_tls = threading.local()

_EMBED_MAX_CHARS = 24000   # conservative safety cut for DashScope 8192-token limit


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _coerce_content(content: Any) -> str:
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


def _tls_stats() -> MemoryCallStats | None:
    return getattr(_tls, "stats", None)


def _truncate_embed(text: str) -> str:
    return text[:_EMBED_MAX_CHARS] if len(text) > _EMBED_MAX_CHARS else text


class _DashScopeEmbeddingFunction:
    """ChromaDB EmbeddingFunction that calls DashScope and counts tokens to _tls.stats."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @staticmethod
    def name() -> str:
        return "dashscope_text_embedding_v4"

    def __call__(self, input):
        truncated = [_truncate_embed(t) for t in input]
        response = self._client.embeddings.create(model=self._model, input=truncated)
        stats = _tls_stats()
        if stats is not None and response.usage:
            stats.embedding_tokens += response.usage.total_tokens or 0
        return [item.embedding for item in response.data]

    def embed_query(self, input):
        # ChromaDB 1.5.8 calls embed_query for query operations (is_query=True)
        return self.__call__(input)


class AMemMemory(BaseMemory):
    """Memory backend backed by A-mem (Agentic Memory) with Zettelkasten evolution."""

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        storage_path: str | None = None,
        llm_model: str = "deepseek-v3-2-251201",
        llm: dict | None = None,
        embed: dict | None = None,
        search_limit: int = 3,
        evo_threshold: int = 99999,
        **kwargs,
    ):
        super().__init__(memory_type="amem", read_only=read_only)
        self._agent_id = agent_id
        self._storage_path = storage_path or f"./storage/amem/{agent_id}"
        self._llm_model = llm_model
        self._llm_cfg = llm or {}
        self._embed_cfg = embed or {}
        self._search_limit = search_limit
        self._evo_threshold = evo_threshold
        self._sidecar_path = os.path.join(self._storage_path, "sidecar.jsonl")
        self._a_mem = None
        self._initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as e:
            raise ImportError(
                f"agentic_memory is not installed: {e}\n"
                "Run: pip install -e ../../../EvoMemBench-Memory-Systems/A-mem"
            ) from e

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package is required") from e

        os.makedirs(self._storage_path, exist_ok=True)

        embed_client = OpenAI(
            api_key=self._embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=self._embed_cfg.get("base_url") or os.getenv(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )
        self._embed_fn = _DashScopeEmbeddingFunction(
            embed_client,
            self._embed_cfg.get("model", "text-embedding-v4"),
        )

        self._a_mem = AgenticMemorySystem(
            llm_backend="openai",
            llm_model=self._llm_model,
            api_key=self._llm_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=self._llm_cfg.get("base_url") or os.getenv("DEEPSEEK_BASE_URL"),
            embedding_function=self._embed_fn,
            evo_threshold=self._evo_threshold,
        )
        self._install_token_hooks()

        n_sidecar = self._reindex_from_sidecar()
        logger.info(
            "[amem] initialized (agent_id=%s, search_limit=%d, sidecar=%s, reindexed=%d)",
            self._agent_id, self._search_limit, self._sidecar_path, n_sidecar,
        )

    def _install_token_hooks(self) -> None:
        """Wrap llm_controller.llm.get_completion() to count LLM tokens via tiktoken."""
        llm_obj = getattr(self._a_mem, "llm_controller", None)
        inner = getattr(llm_obj, "llm", None) if llm_obj is not None else None
        if inner is None:
            logger.warning("[amem] llm_controller.llm not found; LLM tokens will not be counted")
            return

        orig = inner.get_completion

        def _tracked(prompt, response_format=None, temperature=0.7):
            stats = _tls_stats()
            if stats is not None:
                stats.input_tokens += _count_tokens(str(prompt))
            result = orig(prompt, response_format, temperature)
            if stats is not None:
                stats.output_tokens += _count_tokens(str(result) if result else "")
            return result

        inner.get_completion = _tracked
        logger.info("[amem] llm_controller.llm.get_completion wrapped for token counting")

    def _reindex_from_sidecar(self) -> int:
        """Rebuild ChromaDB index from sidecar (no LLM calls). Returns count of notes loaded."""
        if not os.path.exists(self._sidecar_path):
            return 0
        items = []
        with open(self._sidecar_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        for item in items:
            try:
                self._a_mem.retriever.add_document(
                    document=item["content"],
                    metadata=item["metadata"],
                    doc_id=item["id"],
                )
            except Exception as e:
                logger.warning("[amem] reindex failed for id=%s: %s", item.get("id"), e)
        return len(items)

    def _run_with_stats(self, fn):
        stats = MemoryCallStats()
        _tls.stats = stats
        t0 = time.time()
        try:
            result = fn()
        finally:
            stats.latency = time.time() - t0
            _tls.stats = None
        return result, stats

    def _search_retriever(self, query: str, k: int) -> list[str]:
        """Search ChromaDB directly; works whether memories were added via add_note or reindex."""
        try:
            results = self._a_mem.retriever.search(query, k)
        except Exception:
            return []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        contents = []
        for i in range(min(k, len(results["ids"][0]))):
            meta = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
            content = meta.get("content", "")
            if content:
                contents.append(content)
        return contents

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = _coerce_content(conversation[2]["content"])

        memory_lines, stats = self._run_with_stats(
            lambda: self._search_retriever(query, self._search_limit)
        )

        if not memory_lines:
            return conversation, stats

        injection = (
            "\n\n--- Retrieved Memories ---\n"
            + "\n".join(f"- {m}" for m in memory_lines)
            + "\n---"
        )
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = _coerce_content(new_conv[0]["content"]) + injection
        return new_conv, stats

    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        # Build one note per episode: task context + obs-action sequence
        task_desc = _coerce_content(conversation[0].get("content", ""))[:300]
        first_obs = (
            _coerce_content(conversation[2].get("content", ""))[:300]
            if len(conversation) > 2 else ""
        )
        actions = [
            _coerce_content(m.get("content", ""))
            for m in conversation[3:]
            if m.get("role") == "assistant"
        ]
        content = (
            f"Task: {task_desc}\n"
            f"Observation: {first_obs}\n"
            f"Actions: {' → '.join(actions[:15])}"
        )

        def _do():
            note_id = self._a_mem.add_note(content)

            # Build sidecar metadata from stored MemoryNote
            note = self._a_mem.memories.get(note_id)
            if note is not None:
                meta = {
                    "id": note.id,
                    "content": note.content,
                    "keywords": note.keywords,
                    "context": note.context,
                    "tags": note.tags,
                    "links": note.links,
                    "timestamp": note.timestamp,
                    "category": note.category,
                }
            else:
                meta = {"content": content}

            with open(self._sidecar_path, "a") as f:
                f.write(json.dumps({"id": note_id, "content": content, "metadata": meta}) + "\n")

            return note_id

        _, stats = self._run_with_stats(_do)
        return stats

    def load_from_disk(self, **kwargs) -> None:
        # Sidecar persists on disk; re-initializing recreates ChromaDB and reindexes.
        self._initialize()
