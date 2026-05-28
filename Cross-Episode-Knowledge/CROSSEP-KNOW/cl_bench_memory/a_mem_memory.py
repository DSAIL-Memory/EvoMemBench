"""A-mem: Agentic memory with semantic evolution for CL-bench.

Wraps A-mem's AgenticMemorySystem with the CL-bench Memory interface.
Key adaptations vs. the original A-mem / BFCL adaptation:
  - Injects DashScope text-embedding-v4 via a custom ChromaDB EmbeddingFunction,
    bypassing SentenceTransformer entirely (avoids threading/tqdm issues).
  - Tracks LLM and embedding token usage via thin controller wrappers.
  - Implements retrieve() / extract() as required by infer_context_memory.py.

Requirements:
  - A-mem installed as a package (pip install -e <path-to-A-mem>)
  - chromadb, openai packages must be installed
"""

import time
from typing import List

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("r50k_base")  # loads offline, no network needed
except Exception:
    _TIKTOKEN_ENC = None

_EMBED_MAX_TOKENS = 8192


def _truncate_to_embed_limit(text: str) -> str:
    """Truncate text to fit within DashScope's 8192-token embedding limit.

    Uses tiktoken r50k_base which overestimates CJK token count (~2x),
    so the result is always within the actual limit. Falls back to a
    conservative character cut if tiktoken is unavailable.
    """
    if _TIKTOKEN_ENC is None:
        return text[:24000]
    tokens = _TIKTOKEN_ENC.encode(text)
    if len(tokens) <= _EMBED_MAX_TOKENS:
        return text
    return _TIKTOKEN_ENC.decode(tokens[:_EMBED_MAX_TOKENS])

import chromadb  # type: ignore  # required; install via: pip install chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings  # type: ignore
from agentic_memory.llm_controller import OpenAIController       # type: ignore
from agentic_memory.memory_system import AgenticMemorySystem     # type: ignore

from .base import Memory
from .logging_utils import log

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some memory notes from prior interactions in this context. "
    "Use them if relevant to the current task.\n\n"
)


# ─── DashScope embedding function (ChromaDB compatible) ───────────────────────

class DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB EmbeddingFunction that calls DashScope via the OpenAI-compatible API.

    Tracks cumulative token usage across calls so AMemMemory can report stats.
    Not thread-safe on its own — each AMemMemory instance owns one, and each
    AMemMemory is created inside a single worker thread (CL-bench design).
    """

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self._cumulative_tokens = 0

    @staticmethod
    def name() -> str:
        return "dashscope_text_embedding_v4"

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of texts; ChromaDB calls this with a list of strings."""
        truncated = [_truncate_to_embed_limit(t) for t in input]
        response = self.client.embeddings.create(model=self.model, input=truncated)
        if response.usage:
            self._cumulative_tokens += response.usage.total_tokens or 0
        return [item.embedding for item in response.data]

    def reset_usage(self) -> None:
        self._cumulative_tokens = 0

    def get_usage(self) -> dict:
        t = self._cumulative_tokens
        return {"prompt_tokens": t, "total_tokens": t}


# ─── Tracking LLM controller ──────────────────────────────────────────────────

class TrackingOpenAIController(OpenAIController):
    """Extends A-mem's OpenAIController to accumulate LLM token usage.

    Replaces a_mem.llm_controller.llm so that every get_completion() call
    (analyze_content + process_memory) is counted. AMemMemory resets the counter
    before each add_note() call and reads it after.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        from .api_client import call_openai_api
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt},
        ]
        response_text, error, usage = call_openai_api(
            self.client, messages, self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        if usage:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._cumulative_usage[k] += usage.get(k, 0)
        return response_text or ""

    def reset_usage(self) -> None:
        self._cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_usage(self) -> dict:
        return dict(self._cumulative_usage)


# ─── AMemMemory ───────────────────────────────────────────────────────────────

class AMemMemory(Memory):
    """CL-bench wrapper around A-mem's AgenticMemorySystem.

    Uses DashScope text-embedding-v4 (injected as a custom ChromaDB embedding
    function) instead of SentenceTransformer, which:
      - Avoids loading the SentenceTransformer model in every worker thread
      - Eliminates ChromaRetriever._init_lock serialization bottleneck
      - Prevents tqdm thread-safety crashes documented in MEMORY_ADAPTATION_GUIDE

    Each context_id gets its own AMemMemory instance (created inside the worker
    thread by process_context_group), so there is no shared mutable state between
    threads at the application level.
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
    ):
        self.top_k = top_k
        self.memory_dir = memory_dir

        # ── Embedding setup ───────────────────────────────────────────────────
        _embed_client = embed_client or client
        self._dashscope_ef = DashScopeEmbeddingFunction(_embed_client, embed_model)

        # ── Extract api_key / base_url from the OpenAI client object ──────────
        api_key = client.api_key
        raw_base_url = getattr(client, 'base_url', None) or getattr(client, '_base_url', None)
        base_url = str(raw_base_url).rstrip("/") if raw_base_url else None

        _extract_model = extract_model or model

        # ── AgenticMemorySystem (A-mem core) ──────────────────────────────────
        # Inject DashScope EF directly so SentenceTransformer is never loaded.
        # evo_threshold=99999 ensures consolidate_memories() never triggers
        # (extra safety on top of the consolidate_memories() fix in memory_system.py).
        self._a_mem = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model=_extract_model,
            evo_threshold=99999,
            api_key=api_key,
            base_url=base_url,
            embedding_function=self._dashscope_ef,
        )

        # ── Replace LLM controller with token-tracking version ─────────────────
        self._tracking_llm = TrackingOpenAIController(
            model=_extract_model,
            api_key=api_key,
            base_url=base_url,
        )
        self._tracking_llm.client = client  # reuse upstream client (Ark/OpenAI) so batch routing survives
        self._a_mem.llm_controller.llm = self._tracking_llm

        # ── memory_bank attribute required by infer_context_memory.py:215 ──────
        # len(memory.memory_bank) is called to count final memory entries.
        # a_mem.memories is a dict {id: MemoryNote}.
        self.memory_bank = self._a_mem.memories

        log(f"🧠 A-mem: initialized (embed={embed_model}, top_k={top_k})")

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Search memories semantically; return formatted text + stats.

        Returns:
            (memory_text, {"latency_s": float, "embed_usage": dict})
        """
        empty_stats = {"latency_s": 0.0, "embed_usage": {}}
        if not self._a_mem.memories:
            return "", empty_stats

        t0 = time.time()
        self._dashscope_ef.reset_usage()

        try:
            results = self._a_mem.search_agentic(query, k=self.top_k)
        except Exception as e:
            log(f"   ⚠️ A-mem retrieve error: {e}")
            return "", {"latency_s": time.time() - t0, "embed_usage": {}}

        embed_usage = self._dashscope_ef.get_usage()

        # Truncate each note's content before injection to prevent inference context overflow.
        # A-mem stores the full content internally; we only limit what is injected into the prompt.
        # With top_k=10, each note gets up to ~1200 tokens (~4800 chars) of content budget.
        _CONTENT_INJECT_MAX_CHARS = 4800

        mem_parts = []
        for i, r in enumerate(results):
            label = " (linked)" if r.get("is_neighbor") else ""
            keywords = ", ".join(r.get("keywords") or [])
            tags = ", ".join(r.get("tags") or [])
            ctx = r.get("context", "")
            content = r.get("content", "")[:_CONTENT_INJECT_MAX_CHARS]
            mem_parts.append(
                f"# Memory Note {i + 1}{label}\n"
                f"**Context**: {ctx}\n"
                f"**Keywords**: {keywords}\n"
                f"**Tags**: {tags}\n"
                f"**Content**: {content}"
            )

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
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
        """Add a new memory note from the conversation trajectory.

        Full A-mem flow:
          1. analyze_content() — LLM extracts keywords / context / tags
          2. add_note(content, keywords=..., context=..., tags=...)
             which internally calls process_memory() — LLM decides evolution

        Returns:
            {"latency_s": float, "llm_usage": dict, "embed_usage": dict}
        """
        t0 = time.time()
        self._tracking_llm.reset_usage()
        self._dashscope_ef.reset_usage()

        try:
            # Step 1: analyze_content (LLM call 1) — extracts keywords/context/tags
            analysis = self._a_mem.analyze_content(content)
            keywords = analysis.get("keywords", [])
            context = analysis.get("context", "General")
            tags = analysis.get("tags", [])

            # Step 2: add_note with metadata (LLM call 2 inside process_memory)
            self._a_mem.add_note(
                content,
                keywords=keywords,
                context=context,
                tags=tags,
            )
        except Exception as e:
            log(f"   ⚠️ A-mem extract error: {e}")

        llm_usage = self._tracking_llm.get_usage()
        embed_usage = self._dashscope_ef.get_usage()

        log(
            f"   🧠 A-mem: {len(self._a_mem.memories)} notes | "
            f"llm_tokens={llm_usage.get('total_tokens', 0)} "
            f"embed_tokens={embed_usage.get('total_tokens', 0)}"
        )

        return {
            "latency_s": time.time() - t0,
            "llm_usage": llm_usage,
            "embed_usage": embed_usage,
        }
