"""Minimal playbook helpers ported from ACE (ace-appworld/experiments/code/ace/).

Retains only what is needed for the BFCL ACE memory backend:
  - Playbook text parsing / serialisation
  - ADD-only curator operation application
  - JSON extraction from LLM responses

UPDATE / MERGE / DELETE / CREATE_META were TODO stubs in the original and are
deliberately omitted here.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)
_VERBOSE = os.environ.get("BFCL_ACE_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _warn(msg: str) -> None:
    if _VERBOSE:
        logger.warning(msg)


# ── Section metadata ──────────────────────────────────────────────────────────

SECTION_SLUG_MAP: dict[str, str] = {
    "strategies_and_hard_rules": "shr",
    "hard_rules": "hr",
    "strategies_and_insights": "si",
    "apis_to_use_for_specific_information": "api",
    "useful_code_snippets_and_templates": "code",
    "code_snippets_and_templates": "code",
    "common_mistakes_and_correct_strategies": "cms",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics_and_workflows": "psw",
    "problem_solving_heuristics": "prob",
    "verification_checklist": "vc",
    "troubleshooting_and_pitfalls": "ts",
    "others": "misc",
    "meta_strategies": "meta",
}

ALLOWED_SECTIONS: frozenset[str] = frozenset({
    "strategies_and_hard_rules",
    "apis_to_use_for_specific_information",
    "useful_code_snippets_and_templates",
    "common_mistakes_and_correct_strategies",
    "problem_solving_heuristics_and_workflows",
    "verification_checklist",
    "troubleshooting_and_pitfalls",
    "others",
})

# Playbook skeleton with all eight sections present so curator ADD ops always
# have a target section available.
EMPTY_PLAYBOOK_SKELETON: str = (
    "## strategies_and_hard_rules\n\n"
    "## apis_to_use_for_specific_information\n\n"
    "## useful_code_snippets_and_templates\n\n"
    "## common_mistakes_and_correct_strategies\n\n"
    "## problem_solving_heuristics_and_workflows\n\n"
    "## verification_checklist\n\n"
    "## troubleshooting_and_pitfalls\n\n"
    "## others\n"
)


# ── Core helpers ──────────────────────────────────────────────────────────────

def get_section_slug(section_name: str) -> str:
    """Convert a normalised section name to its 2-4 char slug."""
    clean = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    if clean in SECTION_SLUG_MAP:
        return SECTION_SLUG_MAP[clean]
    words = clean.split("_")
    return words[0][:4] if len(words) == 1 else "".join(w[0] for w in words[:5])


def parse_playbook_line(line: str) -> dict | None:
    """Parse a single playbook bullet line.

    Supports:
      ``[id] helpful=X harmful=Y :: content``   (legacy with counts)
      ``[id] content``                            (primary / simplified)
    Returns a dict with keys id, helpful, harmful, content, raw_line, or None.
    """
    text = line.strip()
    m = re.match(r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)', text)
    if m:
        return {"id": m.group(1), "helpful": int(m.group(2)),
                "harmful": int(m.group(3)), "content": m.group(4), "raw_line": line}
    m2 = re.match(r'\[([^\]]+)\]\s*(.*)', text)
    if m2:
        return {"id": m2.group(1), "helpful": 0, "harmful": 0,
                "content": m2.group(2).strip(), "raw_line": line}
    return None


def format_playbook_line(bullet_id: str, content: str) -> str:
    """Serialise a bullet as ``[id] content`` (helpful/harmful counts dropped)."""
    return f"[{bullet_id}] {content}"


def get_next_global_id(playbook_text: str) -> int:
    """Return one past the highest numeric suffix seen in any bullet id."""
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            m = re.search(r"-(\d+)$", parsed["id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def apply_curator_operations(
    playbook_text: str,
    operations: list[dict],
    next_id: int,
) -> tuple[str, int]:
    """Apply a list of ADD operations to playbook_text.

    Non-ADD types are silently ignored (they remain unimplemented stubs).
    Returns the updated playbook text and the updated next_id counter.
    """
    lines = playbook_text.strip().split("\n")

    # Build section map: normalised_section_name → exists
    sections: set[str] = set()
    for line in lines:
        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            norm = header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            sections.add(norm)

    # Collect bullets to inject per section
    bullets_to_add: list[tuple[str, str]] = []
    for op in operations:
        if op.get("type") != "ADD":
            continue
        section_raw = op.get("section", "others")
        section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        if section not in sections:
            _warn(f"ACE curator: section '{section_raw}' not found, adding to 'others'")
            section = "others"
        slug = get_section_slug(section)
        bullet_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "")
        bullets_to_add.append((section, format_playbook_line(bullet_id, content)))
        _warn(f"ACE curator: added bullet {bullet_id} to section {section}")

    if not bullets_to_add:
        return playbook_text, next_id

    # Rebuild playbook, inserting bullets after their target section header
    final_lines: list[str] = []
    current_section: str | None = None

    for line in lines:
        if line.strip().startswith("##"):
            if current_section is not None:
                for sec, bl in bullets_to_add:
                    if sec == current_section:
                        final_lines.append(bl)
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        final_lines.append(line)

    # Flush bullets for the last section
    if current_section is not None:
        for sec, bl in bullets_to_add:
            if sec == current_section:
                final_lines.append(bl)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    # Remaining bullets had no matching section — append at end
    if bullets_to_add:
        _warn(f"ACE curator: {len(bullets_to_add)} bullet(s) had no matching section; appended at end")
        for _, bl in bullets_to_add:
            final_lines.append(bl)

    return "\n".join(final_lines), next_id


def extract_json_from_text(text: str, json_key: str | None = None) -> dict | None:
    """Extract a JSON object from an LLM response, tolerating various wrapping styles."""
    if text is None:
        return None
    try:
        # Direct parse (JSON mode)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # ```json ... ``` blocks
        for m in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue

        # Balanced-brace scan
        i = 0
        while i < len(text):
            if text[i] == "{":
                depth, start, j = 1, i, i + 1
                while j < len(text) and depth > 0:
                    c = text[j]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                    elif c == '"':
                        j += 1
                        while j < len(text) and text[j] != '"':
                            if text[j] == "\\":
                                j += 1
                            j += 1
                    j += 1
                if depth == 0:
                    try:
                        return json.loads(text[start:j])
                    except json.JSONDecodeError:
                        pass
            i += 1

    except Exception as exc:
        _warn(f"ACE extract_json_from_text failed: {exc}")

    return None
