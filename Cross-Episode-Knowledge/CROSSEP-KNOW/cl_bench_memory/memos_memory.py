"""MemOS: GeneralTextMemory adapter for CL-bench.

Wraps MemOS's GeneralTextMemory with the CL-bench Memory interface.
Key adaptations vs. the BFCL adaptation:
  - Uses MemOS's universal_api embedder (DashScope text-embedding-v4 via
    OpenAI-compatible API), bypassing SentenceTransformer entirely
    (no tqdm thread-safety issues).
  - Uses in-memory Qdrant (path=":memory:") for ephemeral per-context
    storage, replacing BFCL's thread-specific local paths.
  - Implements retrieve() / extract() as required by infer_context_memory.py.

Threading notes:
  - CL-bench creates one MemOSMemory per context_id inside the worker thread.
    No cross-thread shared mutable state at the application level.
  - EmbedderFactory and LLMFactory are singletons (keyed by config). Both are
    stateless for per-call API invocations, so sharing across threads is safe.
  - VecDBFactory and MemoryFactory are NOT singletons: each context gets its
    own QdrantVecDB backed by a separate QdrantClient(path=":memory:") instance.

Requirements:
  - MemOS installed as a package (pip install -e <path-to-MemOS>)
"""

import os
import threading
import time
import uuid

# Raise the embedding API timeout before MemOS is imported so the env var is
# read at UniversalAPIEmbedder init time.  Default is 5 s; DashScope can be slower.
os.environ.setdefault("MOS_EMBEDDER_TIMEOUT", "30")

from .base import Memory
from .logging_utils import log

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some memory notes from prior interactions in this context. "
    "Use them if relevant to the current task.\n\n"
)

# DashScope text-embedding-v4 default output dimension.
_DASHSCOPE_EMBED_DIM = 1024


class TrackingOpenAILLM:
    """Wraps MemOS extractor_llm to use call_openai_api for retry + token tracking."""

    def __init__(self, memos_llm, client, tokens_dict):
        self._memos_llm = memos_llm    # original OpenAILLM, kept for config
        self._client = client          # CL-bench OpenAI client
        self._tokens_dict = tokens_dict  # shared dict reference (MemOSMemory._llm_tokens)
        self.config = memos_llm.config  # MemOS internals read extractor_llm.config

    def generate(self, messages, **kwargs):
        from .api_client import call_openai_api
        model = kwargs.get("model_name_or_path", self.config.model_name_or_path)
        temperature = kwargs.get("temperature", self.config.temperature)
        response_text, error, usage = call_openai_api(
            self._client, messages, model, temperature=temperature
        )
        if usage:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._tokens_dict[k] += usage.get(k, 0)
        return response_text or ""


class MemOSMemory(Memory):
    """CL-bench wrapper around MemOS's GeneralTextMemory.

    Uses MemOS's LLM-based extraction pipeline to distil conversation
    trajectories into concise key-value memory notes, then stores and
    retrieves them via Qdrant in-memory vector search.
    """

    # Class-level semaphore: EmbedderFactory is a singleton shared across all
    # worker threads. Concurrent calls into the same embedder object cause
    # threading race conditions (ValueError in threading.py). Limit to 4
    # concurrent embed operations to prevent this.
    _embed_semaphore = threading.Semaphore(4)

    def __init__(
        self,
        client,
        model,
        memory_dir,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=10,
        extract_model=None,
    ):
        self.top_k = top_k
        # memory_bank is read by infer_context_memory.py: len(memory.memory_bank)
        self.memory_bank = {}  # id -> TextualMemoryItem

        # ── Extract API credentials from the OpenAI client objects ────────────
        api_key = client.api_key
        raw_base_url = getattr(client, 'base_url', None) or getattr(client, '_base_url', None)
        base_url = str(raw_base_url).rstrip("/") if raw_base_url else None

        _embed_client = embed_client or client
        embed_api_key = _embed_client.api_key
        embed_raw_url = _embed_client.base_url
        embed_base_url = str(embed_raw_url).rstrip("/") if embed_raw_url else None

        _extract_model = extract_model or model

        # ── MemOS imports (guarded so ImportError surfaces cleanly) ───────────
        try:
            from memos.configs.memory import MemoryConfigFactory
            from memos.memories.factory import MemoryFactory
        except ImportError as e:
            raise ImportError(
                "memos_memory requires MemOS to be installed. "
                "Run: pip install -e <path-to-MemOS>"
            ) from e

        # ── Unique collection name isolates each context's Qdrant store ───────
        collection_name = f"memos_{uuid.uuid4().hex}"

        config = MemoryConfigFactory(
            backend="general_text",
            config={
                "extractor_llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": _extract_model,
                        "api_key": api_key,
                        "api_base": base_url,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": collection_name,
                        "distance_metric": "cosine",
                        "vector_dimension": _DASHSCOPE_EMBED_DIM,
                        "path": ":memory:",  # ephemeral; no disk I/O, no path conflicts
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": embed_model,
                        "api_key": embed_api_key,
                        "base_url": embed_base_url,
                    },
                },
            },
        )

        self._memos = MemoryFactory.from_config(config)
        self._llm_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(self._memos, "extractor_llm") and self._memos.extractor_llm is not None:
            self._memos.extractor_llm = TrackingOpenAILLM(
                self._memos.extractor_llm, client, self._llm_tokens
            )
        log(f"🧠 MemOS: initialized (embed={embed_model}, top_k={top_k})")

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Search memories semantically; return formatted text + stats."""
        empty_stats = {"latency_s": 0.0, "embed_usage": {}}
        if not self.memory_bank:
            return "", empty_stats

        t0 = time.time()
        try:
            with MemOSMemory._embed_semaphore:
                results = self._memos.search(query, top_k=self.top_k)
        except Exception as e:
            log(f"   ⚠️ MemOS retrieve error: {e}")
            return "", {"latency_s": time.time() - t0, "embed_usage": {}}

        mem_parts = []
        for i, item in enumerate(results):
            meta = item.metadata
            # metadata may be a Pydantic model or a plain dict
            if hasattr(meta, "key"):
                key = meta.key or ""
                tags = meta.tags or []
            else:
                key = meta.get("key", "") if meta else ""
                tags = meta.get("tags", []) if meta else []
            tags_str = ", ".join(tags) if tags else ""
            mem_parts.append(
                f"# Memory {i + 1}\n"
                f"**Key**: {key}\n"
                f"**Tags**: {tags_str}\n"
                f"**Content**: {item.memory}"
            )

        stats = {"latency_s": time.time() - t0, "embed_usage": {}}
        if not mem_parts:
            return "", stats
        return MEMORY_INJECTION_PREFIX + "\n\n".join(mem_parts), stats

    # ── extract ───────────────────────────────────────────────────────────────

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        """Extract memory notes from conversation trajectory and store them.

        Flow:
          1. MemOS extractor_llm analyses content and returns key-value facts
          2. Extracted TextualMemoryItems are embedded and added to Qdrant
          3. memory_bank is updated for len() counting
        """
        t0 = time.time()
        self._llm_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # MemOS extract() expects a MessageList; wrap trajectory in a user turn.
        messages = [{"role": "user", "content": content}]

        extracted = []
        try:
            extracted = self._memos.extract(messages)
        except Exception as e:
            log(f"   ⚠️ MemOS extract error: {e}")
            return {"latency_s": time.time() - t0, "llm_usage": dict(self._llm_tokens), "embed_usage": {}}

        # DashScope text-embedding-v4 batch limit is 10 items per call.
        _EMBED_BATCH = 10
        for _b in range(0, len(extracted), _EMBED_BATCH):
            batch = extracted[_b : _b + _EMBED_BATCH]
            try:
                with MemOSMemory._embed_semaphore:
                    self._memos.add(batch)
                for item in batch:
                    self.memory_bank[item.id] = item
            except Exception as e:
                log(f"   ⚠️ MemOS add error: {e}")

        log(
            f"   🧠 MemOS: {len(self.memory_bank)} notes | "
            f"extracted={len(extracted)}"
        )

        return {"latency_s": time.time() - t0, "llm_usage": dict(self._llm_tokens), "embed_usage": {}}
