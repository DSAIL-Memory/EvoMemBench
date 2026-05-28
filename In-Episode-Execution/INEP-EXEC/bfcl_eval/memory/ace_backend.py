"""ACEMemory — ACE (Agentic Context Engineering) backend for the BFCL Memory interface.

Implements the no-ground-truth adaptation mode: after each episode trajectory the
Reflector analyses what went wrong, the Curator proposes ADD operations, and the
accumulated Playbook is injected verbatim into the next sample's system prompt.

ACE has no top-k retrieval — the full playbook text is always injected.
All LLM calls go through ArkBatchClient (Volcengine batch endpoint).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

from bfcl_eval.memory import metrics
from bfcl_eval.memory.ark_client import ArkBatchClient
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.ace_playbook import (
    ALLOWED_SECTIONS,
    EMPTY_PLAYBOOK_SKELETON,
    apply_curator_operations,
    extract_json_from_text,
    get_next_global_id,
)

logger = logging.getLogger(__name__)

_PLAYBOOK_FILENAME = "playbook.txt"
_UTILIZE_PREFIX = "\n\nLearned playbook from prior episodes:\n"
_MAX_PARSE_RETRIES = 3
_MAX_TOOL_ARG_CHARS = 2000  # truncate individual tool_call arguments to keep prompts sane

_REFLECTOR_REQUIRED_KEYS = frozenset({
    "reasoning", "error_identification", "root_cause_analysis",
    "correct_approach", "key_insight",
})
_CURATOR_REQUIRED_KEYS = frozenset({"reasoning", "operations"})

_PROMPTS_DIR = Path(__file__).parent / "ace_prompts"


@lru_cache(maxsize=2)
def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _extract_task_text(trajectory: list[dict]) -> str:
    for msg in trajectory:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content — join text parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts)
            if content:
                return str(content)
    # Fallback to system message
    for msg in trajectory:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                return str(content)
    return "[task description not available]"


def _serialize_trajectory(trajectory: list[dict]) -> str:
    """Produce a human-readable flat representation of the conversation."""
    lines: list[str] = []
    for i, msg in enumerate(trajectory):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(parts)
        content = str(content)

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_parts: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "")
                if isinstance(args_raw, dict):
                    args_raw = json.dumps(args_raw)
                if len(args_raw) > _MAX_TOOL_ARG_CHARS:
                    args_raw = args_raw[:_MAX_TOOL_ARG_CHARS] + "…[truncated]"
                tc_parts.append(f"{name}({args_raw})")
            tool_str = " | ".join(tc_parts)
            lines.append(f"[{i}] {role}: {content} TOOL_CALLS: {tool_str}")
        else:
            lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


class ACEMemory(Memory):
    """Cross-episode memory backend using ACE playbook adaptation (no-GT mode).

    Storage layout under storage_dir/:
        playbook.txt   — accumulated bullet-based knowledge store

    All LLM calls (Reflector, Curator) go through ArkBatchClient.
    No embeddings; no top-k retrieval; the entire playbook is injected on utilize().
    """

    def __init__(
        self,
        *,
        storage_dir: Path | str,
        ark_api_key: str,
        ark_model: str,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._playbook_path = self._storage_dir / _PLAYBOOK_FILENAME

        self._model = ark_model
        self._ark_client = ArkBatchClient(api_key=ark_api_key)
        self._lock = threading.RLock()

        if clear:
            self._playbook = EMPTY_PLAYBOOK_SKELETON
            self._next_id = 1
            self._write_playbook()
        elif self._playbook_path.exists():
            self._playbook = self._playbook_path.read_text(encoding="utf-8")
            self._next_id = get_next_global_id(self._playbook)
        else:
            self._playbook = EMPTY_PLAYBOOK_SKELETON
            self._next_id = 1

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        """Return the full playbook text to be appended to the system prompt.

        system_prompt is ignored — ACE performs no retrieval.
        Returns '' when the playbook contains no bullet lines yet.
        """
        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        playbook = self._playbook
        # Only inject if there are actual bullet lines (not just skeleton headers)
        if not any(line.strip().startswith("[") for line in playbook.splitlines()):
            return ""
        return _UTILIZE_PREFIX + playbook

    def update(self, trajectory: list[dict]) -> None:
        """Run Reflector → Curator and update the playbook from this trajectory."""
        if self.readonly:
            return

        with self._lock:
            try:
                self._run_adaptation(trajectory)
            except Exception:
                logger.exception(
                    "ACEMemory.update failed for trajectory of length %d; playbook unchanged",
                    len(trajectory),
                )

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "ACEMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)

    # ------------------------------------------------------------------ #
    # Adaptation pipeline                                                  #
    # ------------------------------------------------------------------ #

    def _run_adaptation(self, trajectory: list[dict]) -> None:
        task_text = _extract_task_text(trajectory)
        trajectory_text = _serialize_trajectory(trajectory)

        reflection = self._reflector_call(task_text, trajectory_text)
        curator_result = self._curator_call(task_text, trajectory_text, reflection)

        if curator_result is None:
            return

        raw_ops: list = curator_result.get("operations") or []
        # Filter: only ADD, only allowed sections
        ops: list[dict] = []
        for op in raw_ops:
            if not isinstance(op, dict):
                continue
            if op.get("type") != "ADD":
                continue
            section_raw = op.get("section", "")
            section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            if section not in ALLOWED_SECTIONS:
                logger.debug("ACE curator: op references unknown section '%s', discarding", section_raw)
                continue
            ops.append(op)

        if not ops:
            return

        bullet_count = sum(1 for line in self._playbook.splitlines() if line.strip().startswith("["))
        if bullet_count >= 200:
            logger.warning(
                "ACE playbook has reached %d bullets; consider pruning. Proceeding anyway.", bullet_count
            )

        new_playbook, new_next_id = apply_curator_operations(self._playbook, ops, self._next_id)
        self._playbook = new_playbook
        self._next_id = new_next_id
        self._write_playbook()

    def _reflector_call(self, task_text: str, trajectory_text: str) -> dict | None:
        template = _load_prompt("bfcl_reflector_no_gt_prompt.txt")
        # The template uses {{double-brace}} for literal braces and {{var}} for substitution
        filled = (
            template
            .replace("{{generated_code}}", trajectory_text)
            .replace("{{generated_rationale}}", "See full conversation history below")
            .replace("{{playbook}}", self._playbook or "N/A")
            .replace("{{previous_reflection}}", "N/A")
        )
        filled += f"\n\n=== TASK ===\n{task_text}\n"

        messages = [{"role": "user", "content": filled}]
        for attempt in range(_MAX_PARSE_RETRIES):
            try:
                resp, dt = self._ark_client.chat_completions(self._model, messages)
                metrics.record_llm_call(
                    latency_s=dt,
                    input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
                content = resp.choices[0].message.content or ""
                parsed = extract_json_from_text(content)
                if parsed and _REFLECTOR_REQUIRED_KEYS.issubset(parsed.keys()):
                    return parsed
                logger.debug(
                    "ACE reflector: parse attempt %d/%d failed (keys=%s)",
                    attempt + 1, _MAX_PARSE_RETRIES,
                    set(parsed.keys()) if parsed else "None",
                )
            except Exception:
                logger.exception("ACE reflector: LLM call failed on attempt %d", attempt + 1)
        return None

    def _curator_call(
        self,
        task_text: str,
        trajectory_text: str,
        reflection: dict | None,
    ) -> dict | None:
        template = _load_prompt("bfcl_curator_prompt.txt")
        reasoning_text = json.dumps(reflection, ensure_ascii=False) if reflection else "N/A"
        # Tail of trajectory for "final_generated_code" slot (last 4000 chars)
        final_code_excerpt = trajectory_text[-4000:] if len(trajectory_text) > 4000 else trajectory_text

        filled = (
            template
            .replace("{question_context}", task_text)
            .replace("{current_playbook}", self._playbook or "N/A")
            .replace("{final_generated_code}", final_code_excerpt)
            .replace("{guidebook}", reasoning_text)
        )

        messages = [{"role": "user", "content": filled}]
        for attempt in range(_MAX_PARSE_RETRIES):
            try:
                resp, dt = self._ark_client.chat_completions(self._model, messages)
                metrics.record_llm_call(
                    latency_s=dt,
                    input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
                content = resp.choices[0].message.content or ""
                parsed = extract_json_from_text(content)
                if parsed and _CURATOR_REQUIRED_KEYS.issubset(parsed.keys()):
                    if isinstance(parsed.get("operations"), list):
                        return parsed
                logger.debug(
                    "ACE curator: parse attempt %d/%d failed (keys=%s)",
                    attempt + 1, _MAX_PARSE_RETRIES,
                    set(parsed.keys()) if parsed else "None",
                )
            except Exception:
                logger.exception("ACE curator: LLM call failed on attempt %d", attempt + 1)
        return None

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _write_playbook(self) -> None:
        """Atomically write the playbook to disk."""
        tmp = self._playbook_path.with_suffix(".txt.tmp")
        tmp.write_text(self._playbook, encoding="utf-8")
        os.replace(tmp, self._playbook_path)
