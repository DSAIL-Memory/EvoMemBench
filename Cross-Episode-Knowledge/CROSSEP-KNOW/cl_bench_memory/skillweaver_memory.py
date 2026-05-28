"""SkillWeaver Memory: LLM-driven skill extraction with embedding retrieval.

Migrated from EvolveLab/providers/skillweaver_provider.py and adapted
to the CL-bench Memory interface with DashScope text-embedding-v4.

Core workflow:
  Extract  — generate generic skill function from trajectory
             → validate code safety → embed skill description → store
  Retrieve — embed query → cosine similarity search → return formatted skills
  Update   — dedup by function name; scores tracked for future pruning
"""

import ast
import json
import os
import re
import time
import uuid

import numpy as np

from .api_client import call_openai_api
from .base import Memory
from .io import append_jsonl
from .logging_utils import log


# ── Skill generation prompt (from original SkillWeaver) ─────────────────────

SKILL_GENERATION_PROMPT = """\
You are an expert Python programmer. Extract a GENERIC, PARAMETERIZED skill function \
from this successful task execution. Keep the function concise (under 30 lines).

Requirements:
- Single function, generic parameters (no hardcoded values from this execution)
- Short docstring (1 line) describing what the skill does
- No error handling boilerplate needed

BAD: def get_population(): return 1234567
GOOD: def get_population_from_source(source: str, location: str) -> int: ...

Output ONLY the Python code inside a single markdown code block:"""

# ── Failed skill extraction prompt ──────────────────────────────────────────

FAILED_SKILL_PROMPT = """\
You are an expert in analyzing failed task executions. Extract a concise cautionary \
guard function (under 20 lines) that checks for the pitfall encountered.

Requirements:
- Single validation/guard function
- 1-line docstring describing the failure pattern it guards against
- Generic parameters, no hardcoded values

Output ONLY the Python code inside a single markdown code block:"""

# ── Memory injection prefix ────────────────────────────────────────────────

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some reusable skill patterns extracted from past interactions "
    "that may help solve the current task. Review each skill's description and "
    "consider if the methodology applies.\n\n"
)


class SkillWeaverMemory(Memory):
    """SkillWeaver Memory: stores LLM-generated skill functions;
    retrieves via embedding similarity.

    Migrated from EvolveLab/providers/skillweaver_provider.py.
    """

    def __init__(
        self,
        client,
        model,
        memory_dir,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=3,
        extract_model=None,
    ):
        self.client = client
        self.embed_client = embed_client or client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.memory_dir = memory_dir

        self.memory_bank: list[dict] = []
        self.embeddings: list[np.ndarray] = []
        self._last_embed_usage: dict = {}

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "skill_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "skill_embeddings.jsonl")

        self._load_from_disk()

    # ─── Persistence ─────────────────────────────────────────────────────

    def _load_from_disk(self):
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
            log(f"Loaded {len(self.memory_bank)} SkillWeaver skills from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"Bank({len(self.memory_bank)}) vs embeddings"
                f"({len(self.embeddings)}) mismatch, re-embedding..."
            )
            descriptions = [r.get("description", "") for r in self.memory_bank]
            raw = self._embed_batch(descriptions)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_all()

    def _rewrite_all(self):
        """Rewrite both bank and embedding files from in-memory state."""
        with open(self.bank_path, "w", encoding="utf-8") as f:
            for rec in self.memory_bank:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                record = {
                    "id": self.memory_bank[i].get("id", str(i)),
                    "text": self.memory_bank[i].get("description", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ─── Embedding helpers ───────────────────────────────────────────────

    @staticmethod
    def _truncate_head_tail(text: str, max_chars: int = 8000) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text via the OpenAI-compatible API (dashscope text-embedding-v4)."""
        text = self._truncate_head_tail(text, 8000)
        response = self.embed_client.embeddings.create(
            model=self.embed_model, input=text,
        )
        if response.usage:
            self._last_embed_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        results: list[np.ndarray] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [self._truncate_head_tail(t, 8000) for t in texts[i : i + batch_size]]
            response = self.embed_client.embeddings.create(
                model=self.embed_model, input=batch,
            )
            for item in response.data:
                results.append(np.array(item.embedding, dtype=np.float32))
        return results

    # ─── LLM helpers ─────────────────────────────────────────────────────

    def _llm_call(
        self, system_prompt: str, user_content: str, temperature: float = 0.7
    ) -> tuple[str, dict]:
        """Single LLM round-trip. Returns ``(response_text, usage_dict)``."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_text, error, usage = call_openai_api(
            self.client, messages, self.extract_model, temperature=temperature,
        )
        if error:
            log(f"   SkillWeaver LLM call failed: {error}")
            return "", usage
        return response_text or "", usage

    @staticmethod
    def _merge_usage(*usages: dict) -> dict:
        """Sum token counts across multiple usage dicts."""
        merged: dict = {}
        for u in usages:
            if not u:
                continue
            for k, v in u.items():
                merged[k] = merged.get(k, 0) + (v or 0)
        return merged

    # ─── Code validation (from original SkillWeaver) ────────────────────

    @staticmethod
    def _extract_function_from_code(code: str) -> dict | None:
        """Extract the first function definition from Python code via AST."""
        try:
            tree = ast.parse(code)
            func_defs = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            if not func_defs:
                return None
            func = func_defs[0]
            return {"name": func.name, "code": code}
        except Exception:
            return None

    @staticmethod
    def _is_dangerous_code(code: str) -> bool:
        """Basic static checks to avoid dangerous operations in generated skills."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in {
                        "exec", "eval", "compile", "__import__",
                    }:
                        return True
                    if isinstance(node.func, ast.Name) and node.func.id == "open":
                        return True
                if isinstance(node, ast.Attribute):
                    if node.attr in {"system", "popen", "spawn", "remove", "rmdir"}:
                        return True
            return False
        except Exception:
            return True

    # ─── Skill generation (from original SkillWeaver) ────────────────────

    def _generate_skill(
        self, trajectory: str, is_success: bool
    ) -> tuple[dict | None, dict]:
        """Generate a skill function from trajectory.

        Returns ``(skill_dict_or_None, llm_usage)``.
        """
        prompt = SKILL_GENERATION_PROMPT if is_success else FAILED_SKILL_PROMPT
        response_text, usage = self._llm_call(prompt, trajectory)
        if not response_text:
            return None, usage

        # Extract python code block
        m = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
        code = m.group(1).strip() if m else response_text.strip()

        # Validate safety
        if self._is_dangerous_code(code):
            log("   SkillWeaver: generated code failed safety check")
            return None, usage

        func_info = self._extract_function_from_code(code)
        if not func_info:
            log("   SkillWeaver: no valid function found in generated code")
            return None, usage

        # Extract docstring first line as description
        try:
            tree = ast.parse(code)
            func_node = next(
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            )
            docstring = ast.get_docstring(func_node) or func_info["name"]
            description = docstring.split("\n")[0]
        except Exception:
            description = func_info["name"]

        return {
            "name": func_info["name"],
            "code": code,
            "description": description,
            "type": "guiding" if is_success else "cautionary",
        }, usage

    # ─── Retrieve ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        """Retrieve relevant skills for *query* via embedding cosine similarity.

        Returns:
            ``(memory_text, stats)`` where *stats* contains
            ``latency_s`` and ``embed_usage``.
        """
        empty_stats: dict = {"latency_s": 0.0, "embed_usage": {}}
        if not self.memory_bank or not self.embeddings:
            return "", empty_stats

        t0 = time.time()
        self._last_embed_usage = {}
        query_vec = self._l2_normalize(self._embed(query))
        embed_usage = self._last_embed_usage.copy()

        bank_matrix = np.stack(self.embeddings)
        scores = (bank_matrix @ query_vec) * 100.0

        k = min(self.top_k, len(self.memory_bank))
        top_indices = np.argsort(scores)[::-1][:k]

        parts: list[str] = []
        for idx in top_indices:
            if idx >= len(self.memory_bank):
                continue
            rec = self.memory_bank[idx]
            sim_score = float(scores[idx])
            skill_name = rec.get("name", "unnamed_skill")
            description = rec.get("description", "")
            code = rec.get("code", "")
            s_type = rec.get("type", "skill")

            lines = [
                f"--- {s_type.capitalize()} Skill: {skill_name} "
                f"(Sim: {sim_score:.1f}) ---",
                f"**[Description]** {description}",
            ]
            if code:
                lines.append(f"**[Code]**\n```python\n{code}\n```")
            lines.append("---")
            parts.append("\n".join(lines))

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
        if not parts:
            return "", stats
        return MEMORY_INJECTION_PREFIX + "\n\n".join(parts), stats

    # ─── Extract ─────────────────────────────────────────────────────────

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        """Extract a skill from the trajectory and persist it.

        Pipeline:
        1. Generate a generic guiding skill function from trajectory
        2. Validate generated code (AST parse + safety check)
        3. Deduplicate by function name
        4. Embed skill description and store to disk

        Returns:
            ``dict`` with ``latency_s``, ``llm_usage``, ``embed_usage``.
        """
        t0 = time.time()
        total_llm_usage: dict = {}
        total_embed_usage: dict = {}

        # Step 1 -- Generate skill from trajectory
        skill, gen_usage = self._generate_skill(content, is_success=True)
        total_llm_usage = self._merge_usage(total_llm_usage, gen_usage)

        if skill is None:
            return {
                "latency_s": time.time() - t0,
                "llm_usage": total_llm_usage,
                "embed_usage": total_embed_usage,
            }

        # Step 2 -- Deduplicate by function name
        existing_names = {rec.get("name") for rec in self.memory_bank}
        if skill["name"] in existing_names:
            log(f"   SkillWeaver: skill '{skill['name']}' already exists, skipping")
            return {
                "latency_s": time.time() - t0,
                "llm_usage": total_llm_usage,
                "embed_usage": total_embed_usage,
            }

        # Step 3 -- Embed skill description and persist
        self._last_embed_usage = {}
        skill_vec = self._l2_normalize(self._embed(skill["description"]))
        total_embed_usage = self._merge_usage(
            total_embed_usage, self._last_embed_usage
        )

        rid = str(uuid.uuid4())
        rec = {
            "id": rid,
            "name": skill["name"],
            "description": skill["description"],
            "code": skill["code"],
            "type": skill["type"],
            "task_id": task_id,
            "context_category": context_category,
            "sub_category": sub_category,
        }
        self.memory_bank.append(rec)
        self.embeddings.append(skill_vec)

        append_jsonl(rec, self.bank_path)
        append_jsonl(
            {
                "id": rid,
                "text": skill["description"],
                "embedding": skill_vec.tolist(),
            },
            self.embed_path,
        )
        log(
            f"   SkillWeaver: new {skill['type']} skill "
            f"'{skill['name']}' ({rid[:8]})"
        )

        return {
            "latency_s": time.time() - t0,
            "llm_usage": total_llm_usage,
            "embed_usage": total_embed_usage,
        }
