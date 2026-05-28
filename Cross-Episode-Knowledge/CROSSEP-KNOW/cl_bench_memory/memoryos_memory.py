"""MemoryOS (three-tier hierarchical memory) adapter for CL-bench.

Wraps the original MemoryOS (short/mid/long-term memory) with the CL-bench
Memory interface.

Key design decisions vs. BFCL adaptation:
  - No clear() needed: CL-bench creates one MemoryOSMemory per context_id, so
    each instance starts fresh.
  - Storage isolation: uses memory_dir as data_storage_path (already unique per
    context in CL-bench).
  - Three-tier retrieval: explicitly retrieves from all three tiers (short-term
    has highest priority), replicating the fix from the BFCL adaptation.
  - Embedding: uses DashScope text-embedding-v4 (1024-dim) via API, consistent
    with other CL-bench memory systems.  No local model loading, no init lock needed.

BFCL lessons applied:
  - Short-term memory must be retrieved directly (not through retriever).

Requirements:
  - MemoryOS installed as a package (pip install -e <path-to-memoryos-pypi>)
  - DASHSCOPE_API_KEY env var (or embed_client passed to __init__)
"""

import os
import threading
import time

from .base import Memory
from .logging_utils import log

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some memory notes from prior interactions in this context. "
    "Use them if relevant to the current task.\n\n"
)


def _try_import_memoryos():
    """Import Memoryos (install via: pip install -e <path-to-memoryos-pypi>)."""
    try:
        from memoryos import Memoryos
        return Memoryos
    except ImportError as e:
        raise ImportError(
            "MemoryOS not installed. Install via: "
            "pip install -e <path-to-memoryos-pypi>"
        ) from e


class TrackingOpenAIClient:
    """Replaces MemoryOS's OpenAIClient to use call_openai_api for retry + token tracking."""

    def __init__(self, api_key, base_url, inference_client):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self._inference_client = inference_client
        self._llm_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._lock = threading.Lock()

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        from .api_client import call_openai_api
        response_text, error, usage = call_openai_api(
            self._inference_client, messages, model, temperature=temperature
        )
        if usage:
            with self._lock:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self._llm_tokens[k] += usage.get(k, 0)
        return response_text or "Error: Could not get response from LLM."

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=1)
        return ex.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def reset_usage(self):
        with self._lock:
            self._llm_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_usage(self):
        with self._lock:
            return dict(self._llm_tokens)


class MemoryOSMemory(Memory):
    """CL-bench wrapper around the three-tier MemoryOS.

    Flow:
      extract(trajectory) → parse user_input / agent_response
                          → memoryos.add_memory(user_input, agent_response)
      retrieve(query)     → short_term.get_all()
                          + mid_term.search_sessions(query)
                          + long_term.search_user_knowledge(query)
                          → formatted Markdown text
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
        self.memory_bank = {}  # episode_index → True; len() used for counting
        self._extract_count = 0

        # Resolve LLM credentials from the inference client.
        api_key = client.api_key
        raw_base_url = getattr(client, 'base_url', None) or getattr(client, '_base_url', None)
        base_url = str(raw_base_url).rstrip("/") if raw_base_url else None

        _llm_model = extract_model or model

        # Resolve DashScope embedding credentials.
        if embed_client is not None:
            embed_api_key = embed_client.api_key
            raw_embed_base_url = embed_client.base_url
            embed_base_url = str(raw_embed_base_url).rstrip("/") if raw_embed_base_url else None
        else:
            embed_api_key = os.environ.get("DASHSCOPE_API_KEY")
            embed_base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

        data_path = os.path.abspath(memory_dir)
        os.makedirs(data_path, exist_ok=True)

        Memoryos = _try_import_memoryos()

        # DashScope embedding uses API calls (no local model loading), so no
        # serialization lock is needed.  Each context gets its own Memoryos instance
        # with an isolated data_storage_path.
        self._memoryos = Memoryos(
            user_id="cl_bench_user",
            openai_api_key=api_key,
            data_storage_path=data_path,
            openai_base_url=base_url,
            llm_model=_llm_model,
            embedding_model_name=embed_model,
            embedding_model_kwargs={"api_key": embed_api_key, "base_url": embed_base_url},
            short_term_capacity=10,
        )

        # Replace MemoryOS's internal OpenAIClient with our tracking wrapper so all
        # LLM calls go through call_openai_api (retry + token accounting).
        self._tracking_client = TrackingOpenAIClient(
            api_key=api_key, base_url=base_url, inference_client=client
        )
        self._memoryos.client = self._tracking_client
        self._memoryos.mid_term_memory.client = self._tracking_client
        self._memoryos.updater.client = self._tracking_client

        log(f"🧠 MemoryOS: initialized (llm={_llm_model}, embed={embed_model}, top_k={top_k})")

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Retrieve memories from all three tiers and format as Markdown."""
        empty_stats = {"latency_s": 0.0, "embed_usage": {}}
        if not self.memory_bank:
            return "", empty_stats

        t0 = time.time()
        mem_parts = []
        idx = 1

        # ── Short-term (highest priority, direct access) ──────────────────────
        try:
            short_term = self._memoryos.short_term_memory.get_all()
            for qa in short_term:
                if idx > self.top_k:
                    break
                user_txt = qa.get("user_input", "").strip()
                asst_txt = qa.get("agent_response", "").strip()
                if user_txt or asst_txt:
                    mem_parts.append(
                        f"# Memory {idx} [Short-term]\n"
                        f"**User**: {user_txt}\n"
                        f"**Assistant**: {asst_txt}"
                    )
                    idx += 1
        except Exception as e:
            log(f"   ⚠️ MemoryOS short-term retrieve error: {e}")

        # ── Mid-term (semantic search) ────────────────────────────────────────
        if idx <= self.top_k:
            try:
                sessions = self._memoryos.mid_term_memory.search_sessions(
                    query_text=query,
                    segment_similarity_threshold=0.1,
                    page_similarity_threshold=0.1,
                    top_k_sessions=5,
                )
                for session in sessions:
                    for page_match in session.get("matched_pages", []):
                        if idx > self.top_k:
                            break
                        page = page_match.get("page_data", {})
                        user_txt = page.get("user_input", "").strip()
                        asst_txt = page.get("agent_response", "").strip()
                        if user_txt or asst_txt:
                            mem_parts.append(
                                f"# Memory {idx} [Mid-term]\n"
                                f"**User**: {user_txt}\n"
                                f"**Assistant**: {asst_txt}"
                            )
                            idx += 1
            except Exception as e:
                log(f"   ⚠️ MemoryOS mid-term retrieve error: {e}")

        # ── Long-term (user knowledge) ────────────────────────────────────────
        if idx <= self.top_k:
            try:
                knowledge = self._memoryos.user_long_term_memory.search_user_knowledge(
                    query, threshold=0.01, top_k=self.top_k // 2 + 1
                )
                for item in knowledge:
                    if idx > self.top_k:
                        break
                    text = item.get("text", "").strip()
                    if text:
                        mem_parts.append(
                            f"# Memory {idx} [Long-term]\n"
                            f"**Knowledge**: {text}"
                        )
                        idx += 1
            except Exception as e:
                log(f"   ⚠️ MemoryOS long-term retrieve error: {e}")

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
        """Parse trajectory and add to MemoryOS short-term memory.

        MemoryOS expects separate user_input and agent_response.
        format_trajectory() produces sections delimited by:
          **[User Question]**   → user_input
          **[Model Output]**    → agent_response
        """
        t0 = time.time()

        user_input, agent_response = _parse_trajectory(content)

        self._tracking_client.reset_usage()
        try:
            self._memoryos.add_memory(
                user_input=user_input,
                agent_response=agent_response,
            )
            self._extract_count += 1
            self.memory_bank[self._extract_count] = True
        except Exception as e:
            log(f"   ⚠️ MemoryOS add_memory error: {e}")

        llm_usage = self._tracking_client.get_usage()
        log(
            f"   🧠 MemoryOS: {len(self.memory_bank)} notes | "
            f"short_term={len(self._memoryos.short_term_memory.get_all())} | "
            f"llm_tokens={llm_usage.get('total_tokens', 0)}"
        )

        return {"latency_s": time.time() - t0, "llm_usage": llm_usage, "embed_usage": {}}


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_trajectory(content: str) -> tuple:
    """Extract (user_input, agent_response) from a format_trajectory() string.

    Expected markers (from cl_bench_memory/trajectory.py):
        **[User Question]**\n{user_msg}
        **[Model Output]**\n{model_output}

    Falls back to (full content, "") if markers are not found.
    """
    user_input = ""
    agent_response = ""

    # Find the last **[User Question]** section
    uq_marker = "**[User Question]**"
    mo_marker = "**[Model Output]**"
    sc_marker = "**[System Context]**"

    uq_pos = content.rfind(uq_marker)
    mo_pos = content.rfind(mo_marker)

    if uq_pos != -1:
        # Extract text between **[User Question]** and the next section marker
        uq_start = uq_pos + len(uq_marker)
        uq_end = mo_pos if mo_pos > uq_pos else len(content)
        user_input = content[uq_start:uq_end].strip()

    if mo_pos != -1:
        mo_start = mo_pos + len(mo_marker)
        agent_response = content[mo_start:].strip()

    # Fallback: use full content as user_input
    if not user_input and not agent_response:
        user_input = content.strip()

    return user_input, agent_response
