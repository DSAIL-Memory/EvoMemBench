from __future__ import annotations

import json
import os
import re
import shutil
import time
from copy import deepcopy

from .base import BaseMemory, MemoryCallStats

# ---------------------------------------------------------------------------
# Playbook manipulation utilities (inlined from ace-appworld/experiments/code/ace/playbook.py
# and utils.py to avoid requiring the appworld package to be installed)
# ---------------------------------------------------------------------------

_SECTION_SLUG_MAP = {
    # Original ACE sections
    "strategies_and_hard_rules": "shr",
    "hard_rules": "hr",
    "strategies_and_insights": "si",
    "apis_to_use_for_specific_information": "api",
    "useful_code_snippets_and_templates": "code",
    "code_snippets_and_templates": "code",
    "common_mistakes_and_correct_strategies": "cms",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics_and_workflows": "psw",
    "verification_checklist": "vc",
    "troubleshooting_and_pitfalls": "pit",
    "others": "misc",
    "meta_strategies": "meta",
    # ALFWorld-specific sections
    "object_interaction_patterns": "oip",
    "navigation_heuristics": "nav",
    "task_specific_workflows": "tsw",
}


def _get_section_slug(section_name: str) -> str:
    clean = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    if clean in _SECTION_SLUG_MAP:
        return _SECTION_SLUG_MAP[clean]
    words = clean.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])


def _parse_playbook_line(line: str) -> dict | None:
    text = line.strip()
    m = re.match(r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)', text)
    if m:
        return {"id": m.group(1), "helpful": int(m.group(2)), "harmful": int(m.group(3)),
                "content": m.group(4), "raw_line": line}
    m2 = re.match(r'\[([^\]]+)\]\s*(.*)', text)
    if m2:
        return {"id": m2.group(1), "helpful": 0, "harmful": 0,
                "content": m2.group(2).strip(), "raw_line": line}
    return None


def _get_next_global_id(playbook_text: str) -> int:
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = _parse_playbook_line(line)
        if parsed:
            m = re.search(r"-(\d+)$", parsed["id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def _apply_curator_operations(playbook_text: str, operations: list, next_id: int) -> tuple[str, int]:
    lines = playbook_text.strip().split("\n")
    sections: dict[str, list] = {}
    current_section = "general"
    for i, line in enumerate(lines):
        if line.strip().startswith("##"):
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections[current_section].append((i, line))

    bullets_to_add: list[tuple[str, str]] = []
    for op in operations:
        if op.get("type") != "ADD":
            continue
        section_raw = op.get("section", "general")
        section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        if section not in sections and section != "general":
            print(f"  [ACE] Warning: section '{section_raw}' not found, adding to OTHERS")
            section = "others"
        slug = _get_section_slug(section)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "")
        bullets_to_add.append((section, f"[{new_id}] {content}"))
        print(f"  [ACE] Added bullet {new_id} to section '{section}'")

    final_lines: list[str] = []
    current_section = None
    for line in lines:
        if line.strip().startswith("##"):
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            section_header = line.strip()[2:].strip()
            current_section = section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        final_lines.append(line)

    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    if bullets_to_add:
        print(f"  [ACE] Warning: {len(bullets_to_add)} bullets unmatched, appending to end")
        final_lines.extend(b for _, b in bullets_to_add)

    return "\n".join(final_lines), next_id


def _extract_json_from_text(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try ```json blocks
    for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
    # Balanced-brace scan
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth, start = 1, i
            i += 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                elif text[i] == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if depth == 0:
                try:
                    return json.loads(text[start:i])
                except json.JSONDecodeError:
                    pass
        else:
            i += 1
    return None


# ---------------------------------------------------------------------------
# AceMemory
# ---------------------------------------------------------------------------

class AceMemory(BaseMemory):
    """Memory backend using ACE-style playbook: inject full playbook into system prompt,
    update via Reflector → Curator LLM calls after each episode."""

    _DEFAULT_REFLECTOR_PROMPT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts", "alfworld_ace_reflector_prompt.txt",
    )
    _DEFAULT_CURATOR_PROMPT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts", "alfworld_ace_curator_prompt.txt",
    )
    _DEFAULT_INITIAL_PLAYBOOK_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts", "alfworld_initial_playbook.txt",
    )

    def __init__(
        self,
        read_only: bool = False,
        playbook_path: str = "playbook.txt",
        initial_playbook_path: str | None = None,
        ark_batch_config: dict | None = None,
        reflector_prompt_path: str | None = None,
        curator_prompt_path: str | None = None,
    ):
        super().__init__(memory_type="ace", read_only=read_only)

        self._playbook_path = playbook_path
        self._initial_playbook_path = initial_playbook_path or self._DEFAULT_INITIAL_PLAYBOOK_PATH

        # Ensure playbook file exists (copy from initial if not)
        if not os.path.exists(self._playbook_path):
            os.makedirs(os.path.dirname(self._playbook_path) or ".", exist_ok=True)
            if os.path.exists(self._initial_playbook_path):
                shutil.copy(self._initial_playbook_path, self._playbook_path)
                print(f"[ACE] Initialized playbook from {self._initial_playbook_path}")
            else:
                with open(self._playbook_path, "w") as f:
                    f.write("## OTHERS\n")
                print(f"[ACE] Created empty playbook at {self._playbook_path}")

        # Load prompt templates
        rp = reflector_prompt_path or self._DEFAULT_REFLECTOR_PROMPT_PATH
        cp = curator_prompt_path or self._DEFAULT_CURATOR_PROMPT_PATH
        with open(rp) as f:
            self._reflector_prompt_template = f.read()
        with open(cp) as f:
            self._curator_prompt_template = f.read()

        # ARK batch API client
        self._ark_client = None
        self._batch_model = None
        if ark_batch_config:
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError as e:
                raise ImportError(
                    "ark_batch_config requires volcengine SDK. Run:\n"
                    "  pip install 'volcengine-python-sdk[ark]'"
                ) from e
            self._ark_client = Ark(api_key=ark_batch_config["api_key"])
            self._batch_model = ark_batch_config["model"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_playbook(self) -> str:
        with open(self._playbook_path) as f:
            return f.read()

    def _write_playbook(self, text: str) -> None:
        with open(self._playbook_path, "w") as f:
            f.write(text)

    def _call_llm(self, messages: list[dict], stats: MemoryCallStats) -> str:
        """Call LLM via ARK batch API and accumulate token stats."""
        if self._ark_client is None:
            raise RuntimeError("ark_batch_config not provided; cannot call LLM")
        max_retries, retry_delay = 8, 5.0
        for attempt in range(max_retries):
            try:
                response = self._ark_client.batch.chat.completions.create(
                    model=self._batch_model,
                    messages=messages,
                )
                if response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    if (hasattr(response.usage, "prompt_tokens_details")
                            and response.usage.prompt_tokens_details):
                        stats.cached_tokens += getattr(
                            response.usage.prompt_tokens_details, "cached_tokens", 0
                        ) or 0
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[ACE LLM retry {attempt + 1}/{max_retries}] {e}. "
                      f"Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
        return ""

    def _format_trajectory(self, conversation: list[dict]) -> str:
        lines = []
        for i, msg in enumerate(conversation):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{i}] {role}: {content}")
        return "\n\n".join(lines)

    def _extract_task_instruction(self, conversation: list[dict]) -> str:
        """Best-effort extraction of the task instruction from conversation."""
        for msg in conversation[:4]:
            content = msg.get("content", "")
            if not content:
                continue
            # Try to find task description lines
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith("task:"):
                    return line[5:].strip()
                if line.lower().startswith("your task is to"):
                    return line
            # Fallback: return first non-empty content
            if content.strip():
                return content.strip()[:500]
        return ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        playbook = self._read_playbook()
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = (
            new_conv[0]["content"]
            + "\n\n--- Playbook (accumulated task knowledge) ---\n"
            + playbook
            + "\n---"
        )
        return new_conv, MemoryCallStats()

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        stats = MemoryCallStats()
        t0 = time.time()

        playbook = self._read_playbook()
        trajectory_text = self._format_trajectory(conversation)
        task_instruction = self._extract_task_instruction(conversation)

        # ── Reflector call ──────────────────────────────────────────────
        reflector_prompt = (
            self._reflector_prompt_template
            .replace("{{task_instruction}}", task_instruction)
            .replace("{{playbook}}", playbook)
        )
        reflector_messages = [
            {"role": "user", "content": reflector_prompt + "\n\n=== FULL TRAJECTORY ===\n" + trajectory_text}
        ]
        reflector_raw = self._call_llm(reflector_messages, stats)
        reflector_result = _extract_json_from_text(reflector_raw)
        if reflector_result is None:
            print("[ACE] Warning: reflector returned invalid JSON; skipping playbook update")
            stats.latency = time.time() - t0
            return stats

        # ── Curator call ─────────────────────────────────────────────────
        guidebook = reflector_result.get("key_insight", reflector_raw)
        curator_prompt = (
            self._curator_prompt_template
            .replace("{question_context}", task_instruction)
            .replace("{current_playbook}", playbook)
            .replace("{guidebook}", guidebook)
        )
        curator_messages = [
            {"role": "user", "content": curator_prompt + "\n\n=== FULL TRAJECTORY ===\n" + trajectory_text}
        ]
        curator_raw = self._call_llm(curator_messages, stats)
        curator_result = _extract_json_from_text(curator_raw)

        if curator_result is None or "operations" not in curator_result:
            print("[ACE] Warning: curator returned invalid JSON; skipping playbook update")
            stats.latency = time.time() - t0
            return stats

        operations = curator_result.get("operations", [])
        if not isinstance(operations, list) or len(operations) == 0:
            print("[ACE] Curator returned no operations; playbook unchanged")
            stats.latency = time.time() - t0
            return stats

        # ── Apply operations and save ────────────────────────────────────
        next_id = _get_next_global_id(playbook)
        updated_playbook, _ = _apply_curator_operations(playbook, operations, next_id)
        self._write_playbook(updated_playbook)
        print(f"[ACE] Playbook updated ({len(operations)} operations applied)")

        stats.latency = time.time() - t0
        return stats

    def load_from_disk(self, **kwargs) -> None:
        # Playbook is a plain text file; it is read fresh on each inject/update call.
        pass
