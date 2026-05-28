"""LightweightMemory — BFCL adapter for LightweightMemoryProvider.

Dual-memory system:
- Long-term: strategic + operational memories shared across samples (JSON store).
  Provided once at the first turn (BEGIN phase) via LLM selection and synthesis.
- Short-term: task-level key facts extracted per turn, stored thread-locally.
  Provided every N turns (IN phase) until the sample ends.

Process-scoped (long-term persists across samples); short-term resets per sample.

Based on EvolveLab/providers/lightweight_memory_provider.py.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.metrics import MemoryUsageLog

_UTILIZE_PREFIX_LT = "\n\nMemory Guidance (from past experiences):\n"
_UTILIZE_PREFIX_ST = "\n\nCurrent Task Working Memory:\n"

_COLDSTART_STRATEGIC = [
    {
        "content": "Execute directly rather than over-planning. Prioritize immediate action and concrete steps.",
        "tags": ["execution", "planning"],
    },
    {
        "content": "Interpret task instructions literally and precisely. Pay close attention to output format requirements.",
        "tags": ["instruction", "format"],
    },
    {
        "content": "For numerical tasks, identify all units explicitly and track unit conversions through each step.",
        "tags": ["computation", "units"],
    },
]

_COLDSTART_OPERATIONAL = [
    {
        "content": "When a tool call fails, try alternative parameter formats or query phrasings before giving up.",
        "tags": ["error_handling", "tools"],
    },
    {
        "content": "Extract numbers with full context (units, date, conditions) to avoid ambiguity in final answers.",
        "tags": ["extraction", "precision"],
    },
]


class _ArkLLMCallable:
    def __init__(self, api_key: str, model: str):
        from bfcl_eval.memory.ark_client import ArkBatchClient

        self._client = ArkBatchClient(api_key)
        self._model = model

    def __call__(self, messages: list) -> object:
        normalized = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            normalized.append({"role": msg.get("role", "user"), "content": content})
        resp, dt = self._client.chat_completions(self._model, normalized)
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        content = getattr(resp.choices[0].message, "content", "") or ""
        return type("_Resp", (), {"content": content})()


class LightweightMemory(Memory):
    """Dual short-term/long-term memory with LLM-driven selection.

    Storage layout under storage_dir/:
        lightweight_longterm.json  — strategic + operational long-term memories
    """

    def __init__(
        self,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        top_k: int = 5,
        clear: bool = False,
        readonly: bool = False,
        max_strategic: int = 30,
        max_operational: int = 30,
        max_shortterm_items: int = 10,
        shortterm_provision_interval: int = 3,
        **kwargs,
    ):
        super().__init__(memory_type="lightweight", clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._lt_path = self._storage_dir / "lightweight_longterm.json"
        self._top_k = top_k
        self._max_strategic = max_strategic
        self._max_operational = max_operational
        self._max_st = max_shortterm_items
        self._st_interval = shortterm_provision_interval
        self._lock = threading.RLock()
        self._tls = threading.local()

        self._llm = _ArkLLMCallable(ark_api_key, ark_model)

        if clear and self._lt_path.exists():
            self._lt_path.unlink()

        self._lt_db: dict = self._load_lt_db()

    # ------------------------------------------------------------------ #
    # Long-term DB helpers                                                 #
    # ------------------------------------------------------------------ #

    def _load_lt_db(self) -> dict:
        if self._lt_path.exists():
            try:
                with open(self._lt_path, encoding="utf-8") as f:
                    db = json.load(f)
                    # Inject cold-start if empty
                    if not db.get("strategic") and not db.get("operational"):
                        db = self._inject_coldstart(db)
                    return db
            except Exception:
                pass
        db: dict = {"strategic": [], "operational": [], "meta": {"version": 1}}
        return self._inject_coldstart(db)

    def _inject_coldstart(self, db: dict) -> dict:
        for mem in _COLDSTART_STRATEGIC:
            sig = self._sig(mem["content"].lower())
            db["strategic"].append(
                {
                    "content": mem["content"],
                    "tags": mem.get("tags", []),
                    "usage_count": 0,
                    "success_count": 0,
                    "signature": sig,
                }
            )
        for mem in _COLDSTART_OPERATIONAL:
            sig = self._sig(mem["content"].lower())
            db["operational"].append(
                {
                    "content": mem["content"],
                    "tags": mem.get("tags", []),
                    "usage_count": 0,
                    "success_count": 0,
                    "signature": sig,
                }
            )
        self._save_lt_db_unsafe(db)
        return db

    def _save_lt_db(self) -> None:
        # Caller must hold self._lock
        self._save_lt_db_unsafe(self._lt_db)

    def _save_lt_db_unsafe(self, db: dict) -> None:
        try:
            with open(self._lt_path, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _sig(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Thread-local helpers                                                 #
    # ------------------------------------------------------------------ #

    def _turn(self) -> int:
        return getattr(self._tls, "turn", 0)

    def _set_turn(self, n: int) -> None:
        self._tls.turn = n

    def _st_items(self) -> List[str]:
        return getattr(self._tls, "st_items", [])

    def _set_st_items(self, items: List[str]) -> None:
        self._tls.st_items = items

    def _last_st_turn(self) -> int:
        return getattr(self._tls, "last_st_turn", -999)

    def _set_last_st_turn(self, n: int) -> None:
        self._tls.last_st_turn = n

    def _lt_provided(self) -> bool:
        return getattr(self._tls, "lt_provided", False)

    def _set_lt_provided(self, v: bool) -> None:
        self._tls.lt_provided = v

    def _context_hist(self) -> str:
        return getattr(self._tls, "context_hist", "")

    def _set_context_hist(self, v: str) -> None:
        self._tls.context_hist = v

    def _traj(self) -> List[dict]:
        return getattr(self._tls, "trajectory", [])

    def _set_traj(self, v: List[dict]) -> None:
        self._tls.trajectory = v

    def _query(self) -> str:
        return getattr(self._tls, "query", "")

    def _set_query(self, v: str) -> None:
        self._tls.query = v

    # ------------------------------------------------------------------ #
    # LLM helpers                                                          #
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str) -> str:
        try:
            resp = self._llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            return getattr(resp, "content", "").strip()
        except Exception:
            return ""

    def _parse_json(self, text: str) -> Optional[Any]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pat in [r"```json\s*\n?(.*?)```", r"```[^\n]*\n?(.*?)```"]:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        for pat in [r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"]:
            matches = re.findall(pat, text, re.DOTALL)
            for match_text in matches:
                try:
                    return json.loads(match_text)
                except json.JSONDecodeError:
                    continue
        return None

    # ------------------------------------------------------------------ #
    # Long-term memory retrieval (BEGIN phase)                             #
    # ------------------------------------------------------------------ #

    def _retrieve_longterm(self, query: str) -> str:
        with self._lock:
            strategic = list(self._lt_db.get("strategic", []))
            operational = list(self._lt_db.get("operational", []))

        if not strategic and not operational:
            return ""

        candidates = []
        for idx, mem in enumerate(strategic):
            sr = mem.get("success_count", 0) / max(mem.get("usage_count", 1), 1)
            candidates.append({"id": f"strategic_{idx}", "type": "strategic",
                                "content": mem["content"], "success_rate": sr})
        for idx, mem in enumerate(operational):
            sr = mem.get("success_count", 0) / max(mem.get("usage_count", 1), 1)
            candidates.append({"id": f"operational_{idx}", "type": "operational",
                                "content": mem["content"], "success_rate": sr})

        candidate_lines = "\n".join(
            f"{i+1}. [{c['type'].upper()}] (SR={c['success_rate']:.0%}) {c['content']}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"Select the top {self._top_k} most relevant memories and synthesize guidance.\n\n"
            f"Task: {query}\n\nMemories:\n{candidate_lines}\n\n"
            f"Output JSON: {{\"selected_indices\": [1, 2, ...], \"guidance\": \"...\"}}\n"
            f"Keep guidance to 4-5 sentences, use bullet points, frame as suggestions."
        )
        response = self._call_llm(prompt)
        result = self._parse_json(response)

        if result and isinstance(result, dict):
            selected = result.get("selected_indices", [])
            # Update usage_count for selected memories
            with self._lock:
                for idx in selected:
                    if 1 <= idx <= len(candidates):
                        cand = candidates[idx - 1]
                        mtype = cand["type"]
                        midx = int(cand["id"].split("_")[1])
                        mem_list = self._lt_db[mtype]
                        if 0 <= midx < len(mem_list):
                            mem_list[midx]["usage_count"] = mem_list[midx].get("usage_count", 0) + 1
                self._save_lt_db()
            return result.get("guidance", "").strip()

        return response.strip()

    # ------------------------------------------------------------------ #
    # Short-term memory auto-extraction                                    #
    # ------------------------------------------------------------------ #

    def _auto_extract_st(self, user_text: str) -> None:
        last = self._context_hist()
        # compute delta (new content since last call)
        if user_text.startswith(last):
            delta = user_text[len(last):].strip()
        else:
            delta = user_text.strip()
        self._set_context_hist(user_text)

        if len(delta) < 50:
            return

        current_st = self._st_items()
        current_mem_str = (
            "\n".join(f"- {item}" for item in current_st) if current_st else "(none yet)"
        )

        prompt = (
            f"Extract key task-related information from this new step.\n\n"
            f"Task: {self._query()}\n"
            f"Current working memory:\n{current_mem_str}\n\n"
            f"New information:\n{delta[:2000]}\n\n"
            f"Output JSON: {{\"key_extracts\": [\"item1\", \"item2\"]}}\n"
            f"Rules:\n"
            f"- Max 3 items, each under 20 words.\n"
            f"- Only extract task content (data, constraints, results).\n"
            f"- Skip system instructions, format guidance, or tool API details.\n"
            f"- Skip items already in working memory."
        )
        response = self._call_llm(prompt)
        result = self._parse_json(response)
        if result and isinstance(result, dict):
            for item in result.get("key_extracts", []):
                item = str(item).strip()
                if item and len(item) > 10 and item not in current_st:
                    current_st.append(item)
                    if len(current_st) > self._max_st:
                        current_st.pop(0)  # FIFO fallback
            self._set_st_items(current_st)

    # ------------------------------------------------------------------ #
    # Memory ingestion                                                     #
    # ------------------------------------------------------------------ #

    def _extract_and_store_memories(self, query: str, trajectory: list) -> None:
        traj_str = json.dumps(
            [
                {"role": m.get("role"), "content": str(m.get("content", ""))[:300]}
                for m in trajectory
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ][:30],
            ensure_ascii=False,
        )

        prompt = (
            f"Extract reusable learnings from this task execution.\n\n"
            f"Task: {query}\nTrajectory: {traj_str}\n\n"
            f"Extract max 2 items of each type (1-2 sentences each):\n"
            f"- STRATEGIC: when/why to choose specific approaches, decision criteria.\n"
            f"- OPERATIONAL: how to use tools effectively, edge case handling.\n\n"
            f"Output JSON: {{\"strategic\": [\"...\"], \"operational\": [\"...\"]}}"
        )
        response = self._call_llm(prompt)
        extracted = self._parse_json(response)
        if not extracted or not isinstance(extracted, dict):
            return

        with self._lock:
            for content in extracted.get("strategic", []):
                content = str(content).strip()
                if len(content) < 20:
                    continue
                sig = self._sig(content.lower())
                if any(m.get("signature") == sig for m in self._lt_db["strategic"]):
                    continue
                self._lt_db["strategic"].append(
                    {"content": content, "tags": [], "usage_count": 0,
                     "success_count": 0, "signature": sig}
                )
            for content in extracted.get("operational", []):
                content = str(content).strip()
                if len(content) < 20:
                    continue
                sig = self._sig(content.lower())
                if any(m.get("signature") == sig for m in self._lt_db["operational"]):
                    continue
                self._lt_db["operational"].append(
                    {"content": content, "tags": [], "usage_count": 0,
                     "success_count": 0, "signature": sig}
                )
            # Prune if over capacity
            if len(self._lt_db["strategic"]) > self._max_strategic:
                self._lt_db["strategic"] = sorted(
                    self._lt_db["strategic"],
                    key=lambda m: m.get("success_count", 0) * 2 + m.get("usage_count", 0),
                    reverse=True,
                )[: self._max_strategic]
            if len(self._lt_db["operational"]) > self._max_operational:
                self._lt_db["operational"] = sorted(
                    self._lt_db["operational"],
                    key=lambda m: m.get("success_count", 0) * 2 + m.get("usage_count", 0),
                    reverse=True,
                )[: self._max_operational]
            self._save_lt_db()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def begin_sample(self) -> None:
        super().begin_sample()
        # Reset all thread-local state for this sample
        self._tls.turn = 0
        self._tls.st_items = []
        self._tls.last_st_turn = -999
        self._tls.lt_provided = False
        self._tls.context_hist = ""
        self._tls.trajectory = []
        self._tls.query = ""

    def utilize(self, user_text: str) -> str:
        turn = self._turn() + 1
        self._set_turn(turn)

        # Record query from first user text
        if not self._query():
            self._set_query(user_text[:500])

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        result_parts = []

        # BEGIN phase: long-term guidance (once per sample)
        if turn == 1 and not self._lt_provided():
            try:
                lt_guidance = self._retrieve_longterm(user_text)
                if lt_guidance:
                    result_parts.append(_UTILIZE_PREFIX_LT + lt_guidance)
                self._set_lt_provided(True)
            except Exception:
                pass

        # IN phase: short-term memory (every N turns after first)
        if turn > 1:
            try:
                self._auto_extract_st(user_text)
                steps_since = turn - self._last_st_turn()
                if steps_since >= self._st_interval:
                    st_items = self._st_items()
                    if st_items:
                        lines = "\n".join(f"{i+1}. {item}" for i, item in enumerate(st_items))
                        result_parts.append(_UTILIZE_PREFIX_ST + lines)
                        self._set_last_st_turn(turn)
            except Exception:
                pass

        return "".join(result_parts)

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        traj = self._traj()
        traj.extend(trajectory)
        self._set_traj(traj)

        if not self._query():
            for msg in trajectory:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    self._set_query(str(content)[:500])
                    break

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    def drain_usage(self) -> MemoryUsageLog:
        trajectory = self._traj()
        query = self._query()

        if trajectory and query and not self.readonly:
            try:
                self._extract_and_store_memories(query, trajectory)
            except Exception:
                pass

        # Reset thread-local
        self._tls.trajectory = []
        self._tls.query = ""
        self._tls.turn = 0
        self._tls.st_items = []
        self._tls.last_st_turn = -999
        self._tls.lt_provided = False
        self._tls.context_hist = ""

        return metrics.drain()

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "LightweightMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
