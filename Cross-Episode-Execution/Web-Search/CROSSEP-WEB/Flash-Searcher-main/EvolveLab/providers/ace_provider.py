"""ACE (Agentic Context Engineering) memory provider for EvoMemBench.

Ports the Generator/Reflector/Curator pipeline from
  ace-appworld/experiments/code/ace/adaptation_react.py

Key differences from the original:
  - No AppWorld dependency; no generator prompt replacement.
    ACE's playbook is injected as a MemoryItem(type=TEXT) via provide_memory(),
    letting the FlashOAgent ReAct loop handle execution.
  - No top-k retrieval: the entire playbook is returned on BEGIN.
  - Uses volcenginesdkarkruntime.Ark + batch.chat.completions (same as reasoning_bank / graphrag).
  - Uses the no-GT reflector prompt (EvoMemBench has no ground-truth code).
  - Domain-specific examples and initial bullets removed from prompts/playbook.

Pipeline (take_in_memory):
  1. Reflector  → 5-field JSON reasoning_text (error analysis)
  2. Curator    → ADD-operation JSON  (new playbook bullets)
  3. apply_curator_operations → updated playbook text → written to disk
"""

import os
import sys
import threading
import traceback
import uuid
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ace_assets")
_PROMPTS_DIR = os.path.join(_ASSETS_DIR, "prompts")

# Allowed sections — must match curator_prompt.txt exactly
_ALLOWED_SECTIONS = {
    "strategies_and_hard_rules",
    "information_sources_to_consult",
    "useful_query_patterns_and_templates",
    "common_mistakes_and_correct_strategies",
    "problem_solving_heuristics_and_workflows",
    "verification_checklist",
    "troubleshooting_and_pitfalls",
    "others",
}


def _coerce_content(content: Any) -> str:
    """Coerce OpenAI message content (str or multimodal list) to plain str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _resolve_prompt_path(config_path: str | None, default_filename: str) -> str:
    """Resolve a prompt file path.

    Priority:
      1. Absolute path from config.
      2. cwd-relative path from config (Flash-Searcher-main when running pipeline).
      3. Bundled asset path relative to this file.
    """
    if config_path:
        if os.path.isabs(config_path) and os.path.exists(config_path):
            return config_path
        rel = os.path.join(os.getcwd(), config_path)
        if os.path.exists(rel):
            return rel
        # Try relative to the directory two levels up from this file (Flash-Searcher-main)
        base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
        from_base = os.path.join(base, config_path)
        if os.path.exists(from_base):
            return from_base
    bundled = os.path.join(_PROMPTS_DIR, default_filename)
    if os.path.exists(bundled):
        return bundled
    raise FileNotFoundError(
        f"[ace] Cannot find prompt file '{config_path}' or bundled fallback '{bundled}'"
    )


class ACEProvider(BaseMemoryProvider):
    """ACE memory provider: grow a text playbook via Reflector→Curator per task."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.ACE, config or {})
        cfg = self.config
        self._playbook_path: str = cfg.get(
            "playbook_path", "./storage/ace/playbook.txt"
        )
        self._initial_playbook_path: str | None = cfg.get("initial_playbook_path")
        self._reflector_prompt_path: str | None = cfg.get("reflector_prompt_path")
        self._curator_prompt_path: str | None = cfg.get("curator_prompt_path")
        self._use_reflector: bool = cfg.get("use_reflector", True)
        self._return_on_in: bool = cfg.get("return_on_in", False)
        self._lock = threading.RLock()
        self._playbook: str = "(empty)"
        self._next_global_id: int = 1
        self._reflector_template: str = ""
        self._curator_template: str = ""
        self._client = None
        self._model: str = "deepseek-chat"

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            from .ace_assets.playbook_utils import get_next_global_id
        except ImportError as e:
            print(f"[ace] import playbook_utils failed: {e}")
            return False

        # ── Resolve and load prompts ──────────────────────────────────────
        try:
            reflector_path = _resolve_prompt_path(
                self._reflector_prompt_path, "reflector_prompt.txt"
            )
            self._reflector_template = _load_text(reflector_path)

            curator_path = _resolve_prompt_path(
                self._curator_prompt_path, "curator_prompt.txt"
            )
            self._curator_template = _load_text(curator_path)
        except FileNotFoundError as e:
            print(f"[ace] initialize failed: {e}")
            return False

        # ── Resolve and load / seed playbook ─────────────────────────────
        playbook_dir = os.path.dirname(self._playbook_path)
        if playbook_dir:
            os.makedirs(playbook_dir, exist_ok=True)

        if not os.path.exists(self._playbook_path):
            # Seed from initial playbook or empty
            seed = "(empty)"
            if self._initial_playbook_path:
                try:
                    ip_path = _resolve_prompt_path(
                        self._initial_playbook_path, "initial_playbook.txt"
                    )
                    seed = _load_text(ip_path)
                except FileNotFoundError:
                    pass
            _write_text(self._playbook_path, seed)

        self._playbook = _load_text(self._playbook_path)
        self._next_global_id = get_next_global_id(self._playbook)

        # ── LLM client — Ark batch inference ─────────────────────────────
        llm_cfg = self.config.get("llm", {}) or {}
        self._model = (
            llm_cfg.get("model")
            or os.getenv("BATCH_MODEL")
            or os.getenv("DEFAULT_MODEL", "deepseek-chat")
        )
        try:
            from volcenginesdkarkruntime import Ark
            self._client = Ark(
                api_key=llm_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
            )
        except ImportError:
            print("[ace] volcenginesdkarkruntime not installed — "
                  "install 'volcengine-python-sdk[ark]'")
            return False

        bullet_count = self._playbook.count("\n[") + (1 if self._playbook.startswith("[") else 0)
        print(
            f"[ace] initialized (use_reflector={self._use_reflector}, "
            f"return_on_in={self._return_on_in}, "
            f"model={self._model}, "
            f"next_id={self._next_global_id}, "
            f"playbook={self._playbook_path})"
        )
        return True

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if request.status == MemoryStatus.IN and not self._return_on_in:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

        with self._lock:
            playbook = self._playbook

        if not playbook or playbook.strip() in ("", "(empty)"):
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

        return MemoryResponse(
            memories=[
                MemoryItem(
                    id=str(uuid.uuid4()),
                    content=playbook,
                    metadata={"kind": "ace_playbook"},
                    score=1.0,
                    type=MemoryItemType.TEXT,
                )
            ],
            memory_type=self.memory_type,
            total_count=1,
        )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._client is None:
                return False, "[ace] not initialized"

            try:
                return self._run_pipeline(trajectory_data)
            except Exception as e:
                traceback.print_exc()
                return False, f"[ace] take_in_memory error: {e}"

    def _run_pipeline(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        from .ace_assets.playbook_utils import apply_curator_operations, extract_json_from_text

        # ── Build conversation transcript ─────────────────────────────────
        messages = trajectory_data.trajectory or []
        conv_lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").upper()
            content = _coerce_content(msg.get("content", ""))
            conv_lines.append(f"[{i}] {role}: {content}")
        conversation_history = "\n\n".join(conv_lines)

        query = trajectory_data.query or ""
        result = str(trajectory_data.result or "")
        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)

        # Append outcome summary to conversation history so reflector has context
        outcome_note = (
            f"\n\n[OUTCOME] Task: {query}\n"
            f"Agent answer: {result}\n"
            f"Correct: {is_correct}"
        )
        conversation_history += outcome_note

        # ── Reflector ─────────────────────────────────────────────────────
        reasoning_text = ""
        if self._use_reflector:
            reflector_prompt = (
                self._reflector_template
                .replace("{{playbook}}", self._playbook or "N/A")
                .replace("{{previous_reflection}}", "N/A")
                .replace("{{conversation_history}}", conversation_history)
            )
            reasoning_text = self._chat_complete(reflector_prompt)
            if not reasoning_text:
                print("[ace] WARNING: reflector returned empty response")

        # ── Curator ───────────────────────────────────────────────────────
        try:
            curator_prompt = self._curator_template.format(
                question_context=query,
                current_playbook=self._playbook,
                final_generated_code="See conversation history below",
                guidebook=reasoning_text or "N/A",
            )
        except KeyError as e:
            return False, f"[ace] curator template formatting error: {e}"

        curator_prompt += f"\n\n=== FULL CONVERSATION HISTORY ===\n{conversation_history}"

        curator_raw = self._chat_complete(curator_prompt)
        if not curator_raw:
            print("[ace] WARNING: curator returned empty response")
            return False, "[ace] curator returned empty response"

        # ── Parse and validate curator JSON ──────────────────────────────
        ops_info = extract_json_from_text(curator_raw, "operations")
        if not ops_info:
            print(f"[ace] curator JSON parse failed; skipping update")
            return False, "[ace] curator JSON parse failed"

        try:
            if "reasoning" not in ops_info or "operations" not in ops_info:
                raise ValueError("curator JSON missing 'reasoning' or 'operations'")
            if not isinstance(ops_info["operations"], list):
                raise ValueError("'operations' must be a list")

            filtered_ops = []
            for i, op in enumerate(ops_info["operations"]):
                if not isinstance(op, dict):
                    raise ValueError(f"operation {i} is not a dict")
                if op.get("type") != "ADD":
                    print(f"[ace] skipping op {i}: only ADD supported (got {op.get('type')!r})")
                    continue
                section_raw = str(op.get("section", "")).strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")
                if section_raw not in _ALLOWED_SECTIONS:
                    print(f"[ace] skipping op {i}: disallowed section '{op.get('section')}' (normalized: '{section_raw}')")
                    continue
                if not {"type", "section", "content"} <= set(op.keys()):
                    raise ValueError(f"ADD op {i} missing required fields")
                filtered_ops.append(op)

            if not filtered_ops:
                _write_text(self._playbook_path, self._playbook)
                return True, "[ace] curator: 0 valid operations (playbook unchanged)"

            print(f"[ace] applying {len(filtered_ops)} ADD operation(s)")
            self._playbook, self._next_global_id = apply_curator_operations(
                self._playbook, filtered_ops, self._next_global_id
            )

        except (ValueError, KeyError, TypeError) as e:
            print(f"[ace] curator validation error: {e}")
            print(f"[ace] raw curator preview: {curator_raw[:300]}...")
            _write_text(self._playbook_path, self._playbook)
            return False, f"[ace] curator validation error: {e}"

        _write_text(self._playbook_path, self._playbook)
        return True, f"[ace] added {len(filtered_ops)} bullet(s)"

    # ── LLM helper ────────────────────────────────────────────────────────────

    def _chat_complete(self, prompt: str) -> str:
        try:
            resp = self._client.batch.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[ace] LLM call failed: {e}")
            return ""
