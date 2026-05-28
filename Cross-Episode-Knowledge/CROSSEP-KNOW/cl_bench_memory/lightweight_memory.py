"""Lightweight dual-memory system: strategic + operational long-term memory.

Adapted from LightweightMemoryProvider (EvolveLab) for cross-episode CL-bench evaluation.

Core logic:
  - retrieve: embed query with DashScope text-embedding-v4 → cosine search over
              memory content embeddings → formatted strategic + operational guidance
  - extract:  LLM extracts strategic/operational insights from trajectory
              → embed content & persist to disk
  - pruning:  score-based (success_count * 2 + usage_count) when buffer overflows;
              newer entries preferred on tie

Embedding backend: DashScope text-embedding-v4 (default), also supports
openai / gemini / qwen.
"""

import hashlib
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

# ─── Cold-Start Memories ──────────────────────────────────────────────────────

COLDSTART_STRATEGIC = [
    {
        "content": "Execute directly rather than over-planning. Prioritize immediate action and concrete steps over extensive theoretical analysis.",
        "tags": ["execution", "planning"],
    },
    {
        "content": "Interpret task instructions literally and precisely. Pay close attention to output format requirements (units, separators, capitalization).",
        "tags": ["instruction", "format"],
    },
    {
        "content": "For numerical tasks, identify all units explicitly and track unit conversions through each step of calculation.",
        "tags": ["computation", "units"],
    },
    {
        "content": "When tasks present contradictory instructions, prioritize direct compliance with the explicit instruction over meta-analysis of the contradiction.",
        "tags": ["instruction", "compliance"],
    },
    {
        "content": "If no direct authoritative evidence is available, clearly label alternative sources or most-likely answers when reporting.",
        "tags": ["evidence", "reporting"],
    },
]

COLDSTART_OPERATIONAL = [
    {
        "content": "Use specific queries for exact information; extract numbers with full context (units, date, conditions).",
        "tags": ["extraction", "precision"],
    },
    {
        "content": "For multi-source tasks: navigate to the exact source, extract verbatim values, write formula with units, apply formatting only at end.",
        "tags": ["procedure", "information_retrieval"],
    },
]


# ─── Extraction Prompt ────────────────────────────────────────────────────────

EXTRACT_PROMPT_TEMPLATE = """\
Extract reusable learnings from this successful context-learning task execution.

**Task:**
{query}

**Trajectory:**
{trajectory}

**Extract TWO types (max 2 each, 1-2 sentences per item):**

**1. STRATEGIC (Task Planning & Reading Strategy):**
- How to approach this type of context-learning task
- When/why to choose specific reasoning approaches
- How to read or parse a knowledge document for this task type
Example: "For rule-based tasks, first enumerate all applicable rules before attempting to answer."

**2. OPERATIONAL (Execution & Format Handling):**
- Specific techniques that worked for extracting information
- How to handle edge cases or format requirements
- Concrete execution patterns
Example: "When a context document defines exceptions, scan for all exception clauses before applying the base rule."

**Output (JSON only):**
{{
  "strategic": ["insight 1", "insight 2"],
  "operational": ["technique 1", "technique 2"]
}}

**Rules:**
- Focus on REUSABLE patterns, not task-specific facts
- Strategic = planning/reasoning approach; Operational = execution/handling
- Only valuable insights; skip obvious points
- Keep each item under 200 characters"""

# ─── Memory injection header ───────────────────────────────────────────────────

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are strategic and operational insights from similar past context-learning tasks. "
    "Consider these when forming your response.\n\n"
)


# ─── Module-level helpers ──────────────────────────────────────────────────────

def _merge_usage(base: dict, new: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base[key] = base.get(key, 0) + new.get(key, 0)


def _compute_signature(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_tags(text: str) -> list:
    tags = []
    t = text.lower()
    if any(k in t for k in ["search", "retrieve", "query", "find"]):
        tags.append("search")
    if any(k in t for k in ["calculate", "compute", "count", "sum"]):
        tags.append("computation")
    if any(k in t for k in ["validate", "verify", "check"]):
        tags.append("validation")
    if any(k in t for k in ["error", "fallback", "handle"]):
        tags.append("error_handling")
    if any(k in t for k in ["strategy", "plan", "approach"]):
        tags.append("strategy")
    if any(k in t for k in ["format", "output", "unit"]):
        tags.append("format")
    if any(k in t for k in ["rule", "exception", "case"]):
        tags.append("rules")
    return list(set(tags))


# ══════════════════════════════════════════════════════════════════════════════

class LightweightMemory(Memory):
    """Lightweight dual-memory: strategic + operational long-term memory.

    Each entry in memory_bank is a flat dict with a ``type`` field
    ("strategic" or "operational").  Retrieval uses cosine similarity over
    per-content embeddings (DashScope text-embedding-v4 by default).
    Extraction uses rubric-free LLM judge + structured insight extraction;
    All trajectories produce new memories unconditionally.

    Constructor signature matches other cl_bench_memory backends so
    ``build_memory("lightweight", ...)`` works from infer_context_memory.py.
    """

    def __init__(
        self,
        client,
        model: str,
        memory_dir: str,
        embed_provider: str = "dashscope",
        embed_model: str = "text-embedding-v4",
        embed_client=None,
        top_k: int = 5,
        extract_model: Optional[str] = None,
        max_strategic: int = 30,
        max_operational: int = 30,
        memory_buffer_size: int = 20,
    ):
        self.client = client
        self.embed_client = embed_client or client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_strategic = max_strategic
        self.max_operational = max_operational
        self.memory_buffer_size = memory_buffer_size
        self.memory_dir = memory_dir

        self.memory_bank: list = []
        self.embeddings: list = []
        self._last_embed_usage: dict = {}

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "embeddings.jsonl")

        self._init_embed_provider()
        self._load_from_disk()

    # ── Embedding provider setup ───────────────────────────────────────────────

    def _init_embed_provider(self) -> None:
        if self.embed_provider in ("openai", "dashscope"):
            return
        if self.embed_provider == "gemini":
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

        if not self.memory_bank:
            log("Lightweight: empty bank — injecting cold-start memories")
            self._inject_coldstart_memories()
            return

        log(f"📚 Lightweight: loaded {len(self.memory_bank)} memories from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"⚠️  Lightweight: bank({len(self.memory_bank)}) vs "
                f"embeddings({len(self.embeddings)}) mismatch — re-embedding"
            )
            texts = [e.get("content", "") for e in self.memory_bank]
            raw = self._embed_batch(texts)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

    def _inject_coldstart_memories(self) -> None:
        coldstart = []
        for mem in COLDSTART_STRATEGIC:
            coldstart.append(("strategic", mem))
        for mem in COLDSTART_OPERATIONAL:
            coldstart.append(("operational", mem))

        texts = [mem["content"] for _, mem in coldstart]
        raw_vecs = self._embed_batch(texts)

        for (mem_type, mem), vec in zip(coldstart, raw_vecs):
            sig = _compute_signature(mem["content"].lower())
            entry = {
                "content": mem["content"],
                "type": mem_type,
                "tags": mem.get("tags", []),
                "usage_count": 0,
                "success_count": 0,
                "signature": sig,
                "task_id": "coldstart",
                "query": "",
            }
            normalized = self._l2_normalize(vec)
            self.memory_bank.append(entry)
            self.embeddings.append(normalized)
            append_jsonl(entry, self.bank_path)
            append_jsonl(
                {
                    "id": "coldstart",
                    "text": entry["content"],
                    "embedding": normalized.tolist(),
                },
                self.embed_path,
            )

        log(f"Lightweight: injected {len(coldstart)} cold-start memories")

    def _rewrite_embeddings(self) -> None:
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                record = {
                    "id": self.memory_bank[i].get("task_id", str(i)),
                    "text": self.memory_bank[i].get("content", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rewrite_bank(self) -> None:
        with open(self.bank_path, "w", encoding="utf-8") as f:
            for entry in self.memory_bank:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Embedding ──────────────────────────────────────────────────────────────

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

    def _embed(self, text: str) -> np.ndarray:
        text = self._truncate_head_tail(text, 8000)
        if self.embed_provider == "gemini":
            return self._embed_with_gemini(text)
        if self.embed_provider == "qwen":
            return self._embed_with_qwen(text)
        return self._embed_with_openai(text)

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
            [text], max_length=1024, padding=True, truncation=True, return_tensors="pt"
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
                batch = [self._truncate_head_tail(t, 8000) for t in texts[i: i + 100]]
                response = self.embed_client.embeddings.create(
                    model=self.embed_model, input=batch
                )
                for item in response.data:
                    results.append(np.array(item.embedding, dtype=np.float32))
            return results
        return [self._embed(t) for t in texts]

    # ── JSON parsing ───────────────────────────────────────────────────────────

    def _parse_json_response(self, text: str):
        if not text or not text.strip():
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pattern in [r'```json\s*\n?(.*?)```', r'```[^\n]*\n?(.*?)```']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass
        for pattern in [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',
        ]:
            for m in re.findall(pattern, text, re.DOTALL):
                try:
                    return json.loads(m)
                except json.JSONDecodeError:
                    continue
        return None

    # ── Extraction ─────────────────────────────────────────────────────────────

    def _extract_from_trajectory(
        self, content: str, query: str
    ) -> tuple:
        prompt = EXTRACT_PROMPT_TEMPLATE.format(
            query=query or content[:300],
            trajectory=self._truncate_head_tail(content, 3000),
        )
        response_text, error, llm_usage = call_openai_api(
            self.client,
            [{"role": "user", "content": prompt}],
            self.extract_model,
            temperature=0.7,
        )
        if error or not response_text:
            log(f"   ⚠️  Lightweight extraction failed: {error}")
            return None, llm_usage

        result = self._parse_json_response(response_text)
        if not result or not isinstance(result, dict):
            log("   ⚠️  Lightweight: failed to parse extraction JSON")
            return None, llm_usage

        return result, llm_usage

    # ── Pruning ────────────────────────────────────────────────────────────────

    def _count_by_type(self, mem_type: str) -> int:
        return sum(1 for e in self.memory_bank if e.get("type") == mem_type)

    def _check_and_prune(self) -> None:
        for mem_type, max_count in [
            ("strategic", self.max_strategic),
            ("operational", self.max_operational),
        ]:
            threshold = max_count + self.memory_buffer_size
            if self._count_by_type(mem_type) > threshold:
                log(
                    f"   Lightweight: pruning {mem_type} "
                    f"({self._count_by_type(mem_type)} > {threshold})"
                )
                self._prune_by_type(mem_type, max_count)

    def _prune_by_type(self, mem_type: str, max_count: int) -> None:
        type_indices = [
            i for i, e in enumerate(self.memory_bank) if e.get("type") == mem_type
        ]
        if len(type_indices) <= max_count:
            return

        # Score: success_count * 2 + usage_count; ties broken by position (newer = higher idx = preferred)
        scored = [
            (
                i,
                self.memory_bank[i].get("success_count", 0) * 2
                + self.memory_bank[i].get("usage_count", 0),
                i,  # tiebreaker: prefer newer (higher index)
            )
            for i in type_indices
        ]
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

        keep_set = {i for i, _, _ in scored[:max_count]}
        remove_indices = sorted(
            [i for i, _, _ in scored[max_count:]], reverse=True
        )

        for idx in remove_indices:
            self.memory_bank.pop(idx)
            self.embeddings.pop(idx)

        self._rewrite_bank()
        self._rewrite_embeddings()

    # ── Public API: retrieve ───────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Embed query, find top-K memories by cosine similarity, return formatted guidance."""
        empty_stats = {"latency_s": 0.0, "embed_usage": {}}
        if not self.memory_bank or not self.embeddings:
            return "", empty_stats

        t0 = time.time()
        self._last_embed_usage = {}
        query_vec = self._l2_normalize(self._embed(query))
        embed_usage = self._last_embed_usage.copy()

        bank_matrix = np.stack(self.embeddings)
        scores = bank_matrix @ query_vec

        k = min(self.top_k, len(self.memory_bank))
        top_indices = np.argsort(scores)[::-1][:k]

        strategic_parts = []
        operational_parts = []

        for idx in top_indices:
            entry = self.memory_bank[idx]
            content = entry.get("content", "").strip()
            if not content:
                continue
            if entry.get("type") == "strategic":
                strategic_parts.append(f"- {content}")
            else:
                operational_parts.append(f"- {content}")

        sections = []
        if strategic_parts:
            sections.append(
                "**Strategic Insights (task planning & reasoning approach):**\n"
                + "\n".join(strategic_parts)
            )
        if operational_parts:
            sections.append(
                "**Operational Tips (execution & format handling):**\n"
                + "\n".join(operational_parts)
            )

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
        if not sections:
            return "", stats
        return MEMORY_INJECTION_PREFIX + "\n\n".join(sections), stats

    # ── Public API: extract ────────────────────────────────────────────────────

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        """Extract and persist strategic/operational memories from trajectory.

        Returns stats dict with latency_s, llm_usage, embed_usage.
        """
        t0 = time.time()
        total_llm_usage: dict = {}

        # Step 1: Extract strategic/operational insights via LLM
        extraction, extract_usage = self._extract_from_trajectory(content, query)
        _merge_usage(total_llm_usage, extract_usage)

        if not extraction:
            return {
                "latency_s": time.time() - t0,
                "llm_usage": total_llm_usage,
                "embed_usage": {},
            }

        # Step 2: Collect deduplicated new entries
        new_entries = []  # list of (entry_dict, embed_text)
        for mem_type, items in [
            ("strategic", extraction.get("strategic", [])),
            ("operational", extraction.get("operational", [])),
        ]:
            for raw in items:
                raw = str(raw).strip()
                if len(raw) < 20:
                    continue
                sig = _compute_signature(raw.lower())
                if any(e.get("signature") == sig for e in self.memory_bank):
                    continue
                entry = {
                    "content": raw,
                    "type": mem_type,
                    "tags": _extract_tags(raw),
                    "usage_count": 0,
                    "success_count": 0,
                    "signature": sig,
                    "task_id": task_id,
                    "query": self._truncate_head_tail(query, 8000),
                }
                new_entries.append((entry, raw))

        if not new_entries:
            return {
                "latency_s": time.time() - t0,
                "llm_usage": total_llm_usage,
                "embed_usage": {},
            }

        # Step 3: Embed all new entries in one batch call
        self._last_embed_usage = {}
        raw_vecs = self._embed_batch([text for _, text in new_entries])
        embed_usage = self._last_embed_usage.copy()

        for (entry, _), vec in zip(new_entries, raw_vecs):
            normalized = self._l2_normalize(vec)
            self.memory_bank.append(entry)
            self.embeddings.append(normalized)
            append_jsonl(entry, self.bank_path)
            append_jsonl(
                {
                    "id": entry["task_id"],
                    "text": entry["content"],
                    "embedding": normalized.tolist(),
                },
                self.embed_path,
            )

        log(
            f"   ✅ Lightweight: stored {len(new_entries)} entries "
            f"(bank_size={len(self.memory_bank)})"
        )

        # Step 4: Prune if any type exceeds (max + buffer)
        self._check_and_prune()

        return {
            "latency_s": time.time() - t0,
            "llm_usage": total_llm_usage,
            "embed_usage": embed_usage,
        }
