"""Agent Knowledge Base (AgentKB) memory — hybrid retrieval + LLM-guided synthesis.

Migrated from EvolveLab's AgentKBProvider (agent_kb_provider.py).

Core logic:
  - retrieve: optional LLM query refinement → DashScope embedding → hybrid
              TF-IDF + semantic search → formatted planning + experience guidance
  - extract:  LLM summarises trajectory into structured JSON → embed & persist
  - update:   incremental append + TF-IDF rebuild after each extract

Embedding backend: DashScope text-embedding-v4 (default), also supports
openai / gemini / qwen (same interface as ReasoningBankMemory).
"""

import json
import os
import re
import time
from typing import Optional

import numpy as np

from .api_client import call_openai_api
from .base import Memory
from .io import append_jsonl
from .logging_utils import log


# ─── Trajectory Summarisation Prompt ──────────────────────────────────────────

TRAJECTORY_SUMMARIZE_PROMPT = """\
You are an AI agent trainer. Extract concise memory patterns from this successful task execution.

TASK: {query}

EXECUTION TRAJECTORY:
{trajectory_text}

Return ONLY a JSON object with exactly these four fields (1-2 sentences each, be concise):

{{
    "agent_planning": "Key planning steps and decision criteria used",
    "search_agent_planning": "Search strategy and query formulation approach used",
    "agent_experience": "Main lesson or best practice from this execution",
    "search_agent_experience": "Effective search patterns or source selection insight"
}}

Rules: output valid JSON only, no markdown, no extra text, keep each value under 200 characters."""

# ─── Query Refinement Prompt ───────────────────────────────────────────────────

QUERY_REFINEMENT_PROMPT = """\
Extract key concepts from the user query to construct efficient search terms for \
retrieving the most relevant past experiences.

Requirements:
1. Identify core concepts, terminology, and keywords from the question.
2. Extract contextual constraints that may affect retrieval quality.
3. Break down complex questions into searchable components.
4. Identify domain, subject matter, and specific needs.

Output: a concise reformulation capturing the core intent of the query.

User query:
{user_query}"""

# ─── Memory injection header ───────────────────────────────────────────────────

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are planning and experience insights retrieved from similar past tasks. "
    "Consider these when forming your approach.\n\n"
)

# ─── Required JSON fields produced by the summarisation prompt ─────────────────

_REQUIRED_FIELDS = [
    "agent_planning",
    "search_agent_planning",
    "agent_experience",
    "search_agent_experience",
]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _merge_usage(base: dict, new: dict) -> None:
    """Accumulate token-usage dicts in-place."""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base[key] = base.get(key, 0) + new.get(key, 0)


# ══════════════════════════════════════════════════════════════════════════════

class AgentKBMemory(Memory):
    """Agent Knowledge Base: hybrid TF-IDF + semantic retrieval with structured
    planning / experience synthesis.

    Constructor signature matches other cl_bench_memory backends so that
    ``build_memory("agent_kb", ...)`` works transparently from
    ``infer_context_memory.py``.
    """

    def __init__(
        self,
        client,
        model: str,
        memory_dir: str,
        embed_provider: str = "dashscope",
        embed_model: str = "text-embedding-v4",
        embed_client=None,
        top_k: int = 3,
        extract_model: Optional[str] = None,
        refine_query: bool = True,
    ):
        """
        Args:
            client:         OpenAI-compatible client used for LLM calls.
            model:          Model name used for inference / query refinement.
            memory_dir:     Directory for persisting memory_bank.jsonl and
                            embeddings.jsonl.
            embed_provider: "dashscope" (default) | "openai" | "gemini" | "qwen".
            embed_model:    Embedding model name (used for dashscope/openai).
            embed_client:   Separate OpenAI-compatible client for embeddings;
                            falls back to ``client`` if None.
            top_k:          Number of entries to retrieve per query.
            extract_model:  Model for extraction / judging; defaults to ``model``.
            refine_query:   Whether to call LLM for query refinement before
                            embedding (adds one LLM call per retrieve).
        """
        self.client = client
        self.embed_client = embed_client or client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.refine_query = refine_query
        self.memory_dir = memory_dir

        # In-memory state
        self.memory_bank: list[dict] = []      # workflow entry dicts
        self.embeddings: list[np.ndarray] = [] # L2-normalised vectors
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._last_embed_usage: dict = {}

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "embeddings.jsonl")

        self._init_embed_provider()
        self._load_from_disk()

    # ── Embedding provider setup ───────────────────────────────────────────────

    def _init_embed_provider(self) -> None:
        if self.embed_provider in ("openai", "dashscope"):
            pass  # OpenAI-compatible interface via self.embed_client
        elif self.embed_provider == "gemini":
            from vertexai.language_models import TextEmbeddingModel
            self._gemini_embed_model = TextEmbeddingModel.from_pretrained(
                "gemini-embedding-001"
            )
        elif self.embed_provider == "qwen":
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", padding_side="left"
            )
            self._qwen_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B")
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._qwen_model = self._qwen_model.to(self._device)
        else:
            raise ValueError(f"Unknown embed_provider: {self.embed_provider!r}")

    # ── Disk persistence ───────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if os.path.exists(self.bank_path):
            with open(self.bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.memory_bank.append(json.loads(line))

        if os.path.exists(self.embed_path):
            with open(self.embed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        vec = np.array(obj["embedding"], dtype=np.float32)
                        self.embeddings.append(self._l2_normalize(vec))

        if self.memory_bank:
            log(f"📚 AgentKB: loaded {len(self.memory_bank)} entries from disk")

        # Repair mismatch (e.g. interrupted write)
        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"⚠️  AgentKB: bank({len(self.memory_bank)}) vs "
                f"embeddings({len(self.embeddings)}) mismatch — re-embedding"
            )
            queries = [e.get("query", "") for e in self.memory_bank]
            raw = self._embed_batch(queries)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

        if self.memory_bank:
            self._build_tfidf()

    def _rewrite_embeddings(self) -> None:
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                record = {
                    "id": self.memory_bank[i].get("task_id", str(i)),
                    "text": self.memory_bank[i].get("query", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── TF-IDF index ──────────────────────────────────────────────────────────

    def _build_tfidf(self) -> None:
        """Fit TF-IDF vectoriser on current corpus of query strings."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            log("⚠️  AgentKB: sklearn not found; falling back to semantic-only retrieval")
            return

        queries = [e.get("query", "") for e in self.memory_bank]
        if not queries:
            return
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(queries)
        except ValueError:
            # All tokens were stop words (e.g. Chinese-only queries) — fall back
            # to semantic-only retrieval by leaving the matrix unset.
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    @staticmethod
    def _truncate_head_tail(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        text = self._truncate_head_tail(text, 8000)
        if self.embed_provider == "gemini":
            return self._embed_with_gemini(text)
        if self.embed_provider == "qwen":
            return self._embed_with_qwen(text)
        return self._embed_with_openai(text)  # covers dashscope + openai

    def _embed_with_openai(self, text: str) -> np.ndarray:
        response = self.embed_client.embeddings.create(
            model=self.embed_model, input=text
        )
        if response.usage:
            self._last_embed_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_with_gemini(self, text: str) -> np.ndarray:
        from vertexai.language_models import TextEmbeddingInput
        inp = TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")
        resp = self._gemini_embed_model.get_embeddings(
            [inp], output_dimensionality=3072
        )
        return np.array(resp[0].values, dtype=np.float32)

    def _embed_with_qwen(self, text: str) -> np.ndarray:
        import torch
        batch = self._qwen_tokenizer(
            [text],
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._qwen_model(**batch)
            hidden = out.last_hidden_state
            masked = hidden.masked_fill(
                ~batch["attention_mask"][..., None].bool(), 0.0
            )
            pooled = masked.sum(1) / batch["attention_mask"].sum(1)[..., None]
        return pooled.squeeze(0).cpu().numpy()

    def _embed_batch(self, texts: list) -> list:
        if not texts:
            return []
        if self.embed_provider in ("openai", "dashscope"):
            results = []
            for i in range(0, len(texts), 100):
                batch = [self._truncate_head_tail(t, 8000) for t in texts[i : i + 100]]
                response = self.embed_client.embeddings.create(
                    model=self.embed_model, input=batch
                )
                for item in response.data:
                    results.append(np.array(item.embedding, dtype=np.float32))
            return results
        return [self._embed(t) for t in texts]

    # ── Query refinement ───────────────────────────────────────────────────────

    def _refine_query(self, query: str) -> tuple[str, dict]:
        """LLM-based query reformulation; returns (refined_query, llm_usage)."""
        prompt = QUERY_REFINEMENT_PROMPT.format(user_query=query)
        response_text, error, llm_usage = call_openai_api(
            self.client,
            [{"role": "user", "content": prompt}],
            self.model,
            temperature=0,
        )
        if error or not response_text:
            return query, llm_usage
        refined = response_text.strip()
        return refined if refined else query, llm_usage

    # ── Hybrid search ──────────────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query: str,
        query_vec: np.ndarray,
        weights: Optional[dict] = None,
    ) -> list[tuple[int, float]]:
        """Return list of (index, combined_score) sorted descending."""
        weights = weights or {"text": 0.5, "semantic": 0.5}
        n = len(self.memory_bank)
        if n == 0:
            return []

        scores = np.zeros(n, dtype=np.float64)

        # Semantic similarity (cosine via dot product on L2-normalised vectors)
        if self.embeddings:
            bank_matrix = np.stack(self.embeddings)
            sem_scores = bank_matrix @ query_vec  # cosine in [0, 1]
            scores += weights["semantic"] * sem_scores

        # TF-IDF similarity (optional; gracefully skipped when sklearn absent)
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity as sk_cos
                q_tfidf = self._tfidf_vectorizer.transform([query])
                tfidf_scores = sk_cos(q_tfidf, self._tfidf_matrix).flatten()
                scores += weights["text"] * tfidf_scores
            except Exception as exc:
                log(f"⚠️  AgentKB: TF-IDF scoring skipped: {exc}")

        k = min(self.top_k, n)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    # ── Guidance synthesis (no extra LLM call — formats raw stored content) ───

    def _format_guidance(self, entries: list[dict]) -> str:
        planning_parts: list[str] = []
        experience_parts: list[str] = []

        for i, entry in enumerate(entries, 1):
            label = f"[Similar task {i}: {entry.get('query', '')[:80]}]"

            plans = []
            if entry.get("agent_planning"):
                plans.append(entry["agent_planning"])
            if entry.get("search_agent_planning"):
                plans.append(entry["search_agent_planning"])
            if plans:
                planning_parts.append(label + "\n" + " ".join(plans))

            exps = []
            if entry.get("agent_experience"):
                exps.append(entry["agent_experience"])
            if entry.get("search_agent_experience"):
                exps.append(entry["search_agent_experience"])
            if exps:
                experience_parts.append(label + "\n" + " ".join(exps))

        sections: list[str] = []
        if planning_parts:
            sections.append(
                "=== Planning Guidance (from similar tasks) ===\n"
                + "\n\n".join(planning_parts)
            )
        if experience_parts:
            sections.append(
                "=== Experience Insights (from similar tasks) ===\n"
                + "\n\n".join(experience_parts)
            )
        return "\n\n".join(sections)

    # ── Public API: retrieve ───────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Retrieve relevant planning/experience guidance for *query*.

        Returns:
            (memory_text, stats) where stats contains:
                latency_s, embed_usage (token counts), llm_usage (token counts
                for optional query refinement — 0 when refine_query=False).
        """
        empty_stats = {"latency_s": 0.0, "embed_usage": {}, "llm_usage": {}}
        if not self.memory_bank or not self.embeddings:
            return "", empty_stats

        t0 = time.time()
        total_llm_usage: dict = {}

        # Step 1: Optional LLM query refinement
        if self.refine_query:
            refined_query, refine_usage = self._refine_query(query)
            _merge_usage(total_llm_usage, refine_usage)
        else:
            refined_query = query

        # Step 2: Embed refined query with DashScope (or configured provider)
        self._last_embed_usage = {}
        query_vec = self._l2_normalize(self._embed(refined_query))
        embed_usage = self._last_embed_usage.copy()

        # Step 3: Hybrid search
        top_results = self._hybrid_search(refined_query, query_vec)
        if not top_results:
            return "", {
                "latency_s": time.time() - t0,
                "embed_usage": embed_usage,
                "llm_usage": total_llm_usage,
            }

        # Step 4: Format guidance from top entries
        entries = [self.memory_bank[idx] for idx, _ in top_results]
        guidance = self._format_guidance(entries)

        stats = {
            "latency_s": time.time() - t0,
            "embed_usage": embed_usage,
            "llm_usage": total_llm_usage,
        }
        return (MEMORY_INJECTION_PREFIX + guidance) if guidance else "", stats

    # ── Trajectory summarisation ───────────────────────────────────────────────

    def _summarize_trajectory(
        self, query: str, trajectory: str
    ) -> tuple[Optional[dict], dict]:
        """Use LLM to extract structured memory from a successful trajectory.

        Returns (summary_dict | None, llm_usage).
        """
        # Keep trajectory short so the 4-field JSON output stays well within token budget
        prompt = TRAJECTORY_SUMMARIZE_PROMPT.format(
            query=query,
            trajectory_text=self._truncate_head_tail(trajectory, 3000),
        )
        response_text, error, llm_usage = call_openai_api(
            self.client,
            [{"role": "user", "content": prompt}],
            self.extract_model,
            temperature=0.3,
        )
        if error or not response_text:
            log(f"   ⚠️  AgentKB summarisation failed: {error}")
            return None, llm_usage

        # Parse JSON — try full response, then first {...} block, then per-field regex
        summary = None

        try:
            summary = json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        if summary is None:
            # Try extracting the first complete {...} block
            match = re.search(r"\{.*?\}", response_text, re.DOTALL)
            if match:
                try:
                    summary = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if summary is None:
            # Last resort: extract each field value via regex even from truncated JSON.
            # Handles: "field": "value with escaped \"quotes\"" or truncated last field.
            recovered: dict = {}
            for field in _REQUIRED_FIELDS:
                # Match opening quote of value; accept escaped chars; stop at unescaped "
                m = re.search(
                    rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)',
                    response_text,
                )
                if m:
                    recovered[field] = m.group(1).strip()
            if len(recovered) == len(_REQUIRED_FIELDS):
                log("   ℹ️  AgentKB: recovered fields from partial JSON via regex")
                summary = recovered

        if summary is None:
            log("   ⚠️  AgentKB: no valid JSON in summarisation response")
            return None, llm_usage

        # Quality gate: all required fields present and non-trivially long
        if all(f in summary and len(summary[f].strip()) >= 20 for f in _REQUIRED_FIELDS):
            return summary, llm_usage

        log("   ⚠️  AgentKB: summarisation content too brief or missing fields")
        return None, llm_usage

    # ── Public API: extract ────────────────────────────────────────────────────

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        """Persist new memory from *content* (formatted trajectory).

        Pipeline:
          1. Summarise trajectory with LLM → structured JSON.
          2. Embed query with DashScope and store entry on disk.

        Returns stats dict with latency_s, llm_usage, embed_usage.
        """
        t0 = time.time()
        total_llm_usage: dict = {}

        # ── Step 1: Summarise trajectory ───────────────────────────────────────
        summary, sum_usage = self._summarize_trajectory(
            query or content[:500], content
        )
        _merge_usage(total_llm_usage, sum_usage)

        if summary is None:
            return {
                "latency_s": time.time() - t0,
                "llm_usage": total_llm_usage,
                "embed_usage": {},
            }

        # ── Step 2: Embed query and persist ────────────────────────────────────
        embed_text = query if query else content[:500]
        self._last_embed_usage = {}
        embedding = self._l2_normalize(self._embed(embed_text))
        embed_usage = self._last_embed_usage.copy()

        entry = {
            "task_id": task_id,
            "query": self._truncate_head_tail(query, 8000),
            "agent_planning": summary["agent_planning"],
            "search_agent_planning": summary["search_agent_planning"],
            "agent_experience": summary["agent_experience"],
            "search_agent_experience": summary["search_agent_experience"],
        }

        self.memory_bank.append(entry)
        self.embeddings.append(embedding)

        # Rebuild TF-IDF with the new entry included
        self._build_tfidf()

        # Append to JSONL files (atomic-ish: bank first, then embedding)
        append_jsonl(entry, self.bank_path)
        append_jsonl(
            {
                "id": task_id,
                "text": self._truncate_head_tail(query, 8000),
                "embedding": embedding.tolist(),
            },
            self.embed_path,
        )

        log(
            f"   ✅ AgentKB: stored entry (bank_size={len(self.memory_bank)})"
        )

        return {
            "latency_s": time.time() - t0,
            "llm_usage": total_llm_usage,
            "embed_usage": embed_usage,
        }
