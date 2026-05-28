from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy

from .base import BaseMemory, MemoryCallStats

try:
    import mem0  # noqa: F401
except ImportError as e:
    raise ImportError(
        "mem0 is not installed. Install it in editable mode from the vendored "
        "source:\n"
        "  pip install -e ../../../EvoMemBench-Memory-Systems/mem0\n"
        "(path is relative to the CROSSEP-EMB project root)"
    ) from e

# Thread-local storage for per-call stat accumulation (thread-safe).
_tls = threading.local()


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _tls_stats() -> MemoryCallStats | None:
    return getattr(_tls, "stats", None)


class Mem0Memory(BaseMemory):
    """Memory backend backed by the mem0 library."""

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        llm_config: dict | None = None,
        embedder_config: dict | None = None,
        vector_store_config: dict | None = None,
        search_limit: int = 5,
        history_db_path: str | None = None,
        ark_batch_config: dict | None = None,
    ):
        super().__init__(memory_type="mem0", read_only=read_only)
        self._agent_id = agent_id
        self._search_limit = search_limit

        # Ark Batch API client for memory LLM calls (optional).
        self._ark_client = None
        self._batch_model = None
        if ark_batch_config:
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError as e:
                raise ImportError(
                    "ark_batch_config 需要 volcengine SDK，请运行：\n"
                    "  pip install 'volcengine-python-sdk[ark]'"
                ) from e
            self._ark_client = Ark(api_key=ark_batch_config["api_key"])
            self._batch_model = ark_batch_config["model"]

        self._memory = self._build_memory(llm_config, embedder_config, vector_store_config, history_db_path)
        self._install_token_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_memory(
        self,
        llm_config: dict | None,
        embedder_config: dict | None,
        vector_store_config: dict | None,
        history_db_path: str | None = None,
    ):
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig

        config_dict: dict = {}
        if llm_config:
            config_dict["llm"] = llm_config
        if embedder_config:
            config_dict["embedder"] = embedder_config
        if vector_store_config:
            config_dict["vector_store"] = vector_store_config
        if history_db_path:
            config_dict["history_db_path"] = history_db_path

        if config_dict:
            config = MemoryConfig(**config_dict)
            return Memory(config=config)
        return Memory()

    def _install_token_hooks(self):
        """Monkey-patch mem0's LLM and embedding model to count tokens per call."""
        memory = self._memory

        # Patch LLM
        if hasattr(memory, "llm") and memory.llm is not None:
            orig_generate = memory.llm.generate_response
            ark_client = self._ark_client
            batch_model = self._batch_model

            if ark_client is not None:
                # Batch API mode: call Ark Batch API, track real token counts.
                def tracked_generate(messages, tools=None, **kwargs):
                    stats = _tls_stats()
                    max_retries, retry_delay = 8, 5.0
                    for attempt in range(max_retries):
                        try:
                            response = ark_client.batch.chat.completions.create(
                                model=batch_model,
                                messages=messages,
                            )
                            if stats is not None and response.usage:
                                stats.input_tokens += response.usage.prompt_tokens or 0
                                stats.output_tokens += response.usage.completion_tokens or 0
                                cached = 0
                                if (hasattr(response.usage, "prompt_tokens_details")
                                        and response.usage.prompt_tokens_details):
                                    cached = getattr(
                                        response.usage.prompt_tokens_details, "cached_tokens", 0
                                    ) or 0
                                stats.cached_tokens += cached
                            return response.choices[0].message.content
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            print(
                                f"[Mem0 BatchAPI retry {attempt+1}/{max_retries}] {e}. "
                                f"Retrying in {retry_delay:.1f}s..."
                            )
                            time.sleep(retry_delay)
            else:
                # Standard mode: tiktoken estimation, no cached_tokens.
                def tracked_generate(messages, tools=None, **kwargs):
                    stats = _tls_stats()
                    if stats is not None:
                        try:
                            stats.input_tokens += _count_tokens(json.dumps(messages, ensure_ascii=False))
                        except Exception:
                            pass
                    result = orig_generate(messages, tools=tools, **kwargs)
                    if stats is not None:
                        try:
                            out_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                            stats.output_tokens += _count_tokens(out_text)
                        except Exception:
                            pass
                    return result

            memory.llm.generate_response = tracked_generate

        # Patch embedding model
        if hasattr(memory, "embedding_model") and memory.embedding_model is not None:
            orig_embed = memory.embedding_model.embed

            def tracked_embed(text, *args, **kwargs):
                stats = _tls_stats()
                if stats is not None:
                    try:
                        stats.embedding_tokens += _count_tokens(text if isinstance(text, str) else str(text))
                    except Exception:
                        pass
                return orig_embed(text, *args, **kwargs)

            memory.embedding_model.embed = tracked_embed

    def _run_with_stats(self, fn):
        """Run fn() while accumulating token stats. Returns (result, MemoryCallStats)."""
        stats = MemoryCallStats()
        _tls.stats = stats
        t0 = time.time()
        try:
            result = fn()
        finally:
            stats.latency = time.time() - t0
            _tls.stats = None
        return result, stats

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]

        results, stats = self._run_with_stats(
            lambda: self._memory.search(
                query,
                filters={"agent_id": self._agent_id},
                top_k=self._search_limit,
            )
        )

        memories = []
        if isinstance(results, dict):
            memories = results.get("results", [])
        elif isinstance(results, list):
            memories = results

        if not memories:
            return conversation, stats

        memory_lines = "\n".join(f"- {m['memory']}" for m in memories if m.get("memory"))
        injection = f"\n\n--- Retrieved Memories ---\n{memory_lines}\n---"

        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = new_conv[0]["content"] + injection
        return new_conv, stats

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
            if msg.get("content")
        ]

        _, stats = self._run_with_stats(
            lambda: self._memory.add(messages, agent_id=self._agent_id)
        )
        return stats

    def load_from_disk(
        self,
        vector_store_type: str = "chroma",
        vector_store_path: str | None = None,
        collection_name: str = "alfworld_mem0",
        db_path: str | None = None,
        **kwargs,
    ) -> None:
        """Reload mem0 pointing at a persistent vector store and/or SQLite history db.

        For Chroma: provide vector_store_path (persist directory).
        For Qdrant: provide host/port via kwargs.
        db_path: path for mem0's SQLite fact store (overrides MEM0_DB_PATH env var).
        """
        if db_path:
            os.environ["MEM0_DB_PATH"] = db_path

        vector_store_config: dict | None = None
        if vector_store_path is not None:
            vector_store_config = {
                "provider": vector_store_type,
                "config": {
                    "collection_name": collection_name,
                    "path": vector_store_path,
                    **kwargs,
                },
            }

        self._memory = self._build_memory(None, None, vector_store_config)
        self._install_token_hooks()
