"""Mem0Memory: adapter wrapping the official mem0 library (vendored at memory_official_implentations/mem0/).

LLM calls are routed through cl_bench_memory.api_client.call_openai_api so retry logic and
token-usage capture are consistent with the rest of the pipeline.  Embeddings use the
DashScope text-embedding-v4 model via the embed_client passed by infer_context_memory.py.
The vector store is Qdrant in embedded (local-disk) mode.
"""

import os
import re
import time

os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory as _Mem0Memory  # noqa: E402
from mem0.configs.llms.base import BaseLlmConfig  # noqa: E402
from mem0.configs.embeddings.base import BaseEmbedderConfig  # noqa: E402
from mem0.llms.base import LLMBase  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402

from .api_client import call_openai_api
from .base import Memory
from .logging_utils import log

MEM0_MEMORY_INJECTION_PREFIX = "\n\n[Memory from previous episodes]\n"
_EMBED_BATCH_SIZE = 25
_MAX_TEXT_CHARS = 8000


# ── LLM adapter ──────────────────────────────────────────────────────────────

class CLBenchAdapterLLM(LLMBase):
    """Wraps call_openai_api so mem0's extraction pipeline uses our client."""

    def __init__(self, client, model, usage_sink, temperature=0.1):
        # Bypass LLMBase.__init__ — it would create a duplicate config object
        # and call _validate_config which only checks attribute existence.
        self.config = BaseLlmConfig(model=model, temperature=temperature)
        self._client = client
        self._usage_sink = usage_sink

    def generate_response(self, messages, tools=None, tool_choice="auto",
                          response_format=None, **kwargs):
        text, error, usage = call_openai_api(
            self._client,
            messages,
            self.config.model,
            temperature=self.config.temperature,
            response_format=response_format,
        )
        if error:
            # Raise so mem0's try/except at main.py:746-748 returns [] gracefully.
            raise RuntimeError(f"LLM call failed: {error}")
        if usage:
            self._usage_sink.append(usage)
        return text


# ── Embedding adapter ─────────────────────────────────────────────────────────

class CLBenchDashScopeEmbedding(EmbeddingBase):
    """Wraps the DashScope embed_client for text-embedding-v4."""

    def __init__(self, client, model, usage_sink):
        # Bypass EmbeddingBase.__init__ to avoid re-constructing config.
        self.config = BaseEmbedderConfig(model=model, embedding_dims=1024)
        self._client = client
        self._model = model
        self._usage_sink = usage_sink

    def _record_usage(self, response):
        if hasattr(response, "usage") and response.usage:
            self._usage_sink.append({
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": 0,
                "total_tokens": response.usage.total_tokens or 0,
            })

    def embed(self, text, memory_action=None):
        text = text[:_MAX_TEXT_CHARS]
        response = self._client.embeddings.create(model=self._model, input=text)
        self._record_usage(response)
        return response.data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        texts = [t[:_MAX_TEXT_CHARS] for t in texts]
        results = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            self._record_usage(response)
            ordered = sorted(response.data, key=lambda x: x.index)
            results.extend(item.embedding for item in ordered)
        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sum_usage(usage_list):
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage_list:
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["total_tokens"] += u.get("total_tokens", 0)
    return total


def _parse_trajectory(content, query=""):
    """Reconstruct a full messages list from a format_trajectory string.

    trajectory.py writes blocks in document order:
        **[System Context]**     -> role=system  (at most one)
        **[User Question]**      -> role=user    (one per round)
        **[Assistant Response]** -> role=assistant (known correct answers, interleaved)
        **[Model Output]**       -> role=assistant (replaces the last turn's assistant reply)

    Intermediate **[Assistant Response]** blocks are ground-truth demonstrations of
    how the domain rules are correctly applied — they are the primary signal for
    cross-episode memory accumulation and must not be discarded.

    Returns a list of {"role": ..., "content": ...} dicts ready for mem0.add().
    Falls back to a minimal [user, assistant] pair if no markers are found.
    """
    # Split the content on any block marker, keeping the markers as delimiters.
    # Each token is either a marker tag or the text following it.
    tokens = re.split(r"(\*\*\[(?:System Context|User Question|Assistant Response|Model Output)\]\*\*)\n", content)

    messages = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("**[") and token.endswith("]**"):
            block_text = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
            if token == "**[System Context]**":
                messages.append({"role": "system", "content": block_text})
            elif token == "**[User Question]**":
                messages.append({"role": "user", "content": block_text})
            elif token == "**[Assistant Response]**":
                messages.append({"role": "assistant", "content": block_text})
            elif token == "**[Model Output]**":
                # The model's actual output for the final query — treat as assistant.
                messages.append({"role": "assistant", "content": block_text})
            i += 2
        else:
            i += 1

    if not messages:
        # Fallback: no markers found — build a minimal pair from raw content.
        messages = [
            {"role": "user", "content": query or content[:500]},
            {"role": "assistant", "content": ""},
        ]

    return messages


# ── Main class ────────────────────────────────────────────────────────────────

class Mem0Memory(Memory):
    """Memory backend backed by the official mem0 library.

    Each instance is per-context (infer_context_memory.py gives each context
    its own memory_dir), so self._user_id is a fixed partition key.
    """

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
        **kwargs,
    ):
        if embed_provider != "dashscope":
            raise ValueError(
                f"Mem0Memory only supports embed_provider='dashscope', got {embed_provider!r}"
            )

        self.top_k = top_k
        self._llm_usage = []
        self._embed_usage = []
        self.memory_bank = []

        extract_model = extract_model or model
        embed_client = embed_client or client

        os.makedirs(memory_dir, exist_ok=True)
        qdrant_path = os.path.join(memory_dir, "qdrant")
        history_db_path = os.path.join(memory_dir, "history.db")

        # Build MemoryConfig.  We use provider="openai" (passes the field_validator
        # whitelist in llms/configs.py and embeddings/configs.py) with a dummy api_key;
        # we replace .llm and .embedding_model immediately after construction so the
        # dummy OpenAI instances are never called.
        from mem0.configs.base import MemoryConfig
        from mem0.llms.configs import LlmConfig
        from mem0.embeddings.configs import EmbedderConfig
        from mem0.vector_stores.configs import VectorStoreConfig

        mem0_cfg = MemoryConfig(
            llm=LlmConfig(
                provider="openai",
                config={"model": extract_model, "api_key": "dummy"},
            ),
            embedder=EmbedderConfig(
                provider="openai",
                config={
                    "model": embed_model,
                    "api_key": "dummy",
                    "embedding_dims": 1024,
                },
            ),
            vector_store=VectorStoreConfig(
                provider="qdrant",
                config={
                    "collection_name": "cl_bench",
                    "path": qdrant_path,
                    "embedding_model_dims": 1024,
                    "on_disk": True,
                },
            ),
            history_db_path=history_db_path,
        )

        self._mem0 = _Mem0Memory(mem0_cfg)

        # Replace dummy OpenAI adapters with our CL-bench-wired ones.
        self._mem0.llm = CLBenchAdapterLLM(
            client=client,
            model=extract_model,
            usage_sink=self._llm_usage,
        )
        self._mem0.embedding_model = CLBenchDashScopeEmbedding(
            client=embed_client,
            model=embed_model,
            usage_sink=self._embed_usage,
        )

        self._user_id = "cl_bench"

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        self._embed_usage.clear()
        t0 = time.time()
        try:
            results = self._mem0.search(
                query=query,
                filters={"user_id": self._user_id},
                top_k=self.top_k,
            ).get("results", [])
        except Exception as e:
            log(f"   Mem0Memory.retrieve error: {e}")
            results = []
        latency = time.time() - t0

        if not results:
            text = ""
        else:
            lines = [f"{i + 1}. {r['memory']}" for i, r in enumerate(results)]
            text = MEM0_MEMORY_INJECTION_PREFIX + "\n".join(lines)

        return text, {"latency_s": latency, "embed_usage": _sum_usage(self._embed_usage)}

    # ── extract ───────────────────────────────────────────────────────────────

    def extract(self, content: str, **kwargs) -> dict:
        task_id = kwargs.get("task_id", "")
        query = kwargs.get("query", "")
        context_category = kwargs.get("context_category", "")
        sub_category = kwargs.get("sub_category", "")

        self._llm_usage.clear()
        self._embed_usage.clear()
        t0 = time.time()

        messages = _parse_trajectory(content, query=query)

        try:
            result = self._mem0.add(
                messages,
                user_id=self._user_id,
                metadata={
                    "task_id": task_id,
                    "context_category": context_category,
                    "sub_category": sub_category,
                },
            )
            for ev in result.get("results", []):
                if ev.get("event") == "ADD":
                    self.memory_bank.append({
                        "id": ev["id"],
                        "memory": ev["memory"],
                        "task_id": task_id,
                    })
        except Exception as e:
            log(f"   Mem0Memory.extract error: {e}")

        return {
            "latency_s": time.time() - t0,
            "llm_usage": _sum_usage(self._llm_usage),
            "embed_usage": _sum_usage(self._embed_usage),
        }
