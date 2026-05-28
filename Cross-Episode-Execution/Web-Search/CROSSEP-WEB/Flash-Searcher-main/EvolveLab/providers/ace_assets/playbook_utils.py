"""
Vendored port of ACE playbook utilities.

Sources:
  ace-appworld/experiments/code/ace/playbook.py
  ace-appworld/experiments/code/ace/utils.py (get_section_slug only)

No dependencies on appworld or litellm.
"""

import json
import re


# ---------------------------------------------------------------------------
# Section slug map
# ---------------------------------------------------------------------------

def get_section_slug(section_name: str) -> str:
    """Convert a section name to a short slug prefix for bullet IDs."""
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        # original ACE names (kept for backwards compat if they appear in old playbooks)
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        # genericized names used in EvoMemBench adaptation
        "information_sources_to_consult": "info",
        "useful_query_patterns_and_templates": "qpt",
        # shared across both
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta",
    }
    clean = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    if clean in slug_map:
        return slug_map[clean]
    words = clean.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])


# ---------------------------------------------------------------------------
# Playbook parsing
# ---------------------------------------------------------------------------

def parse_playbook_line(line: str) -> dict | None:
    """Parse a single playbook line.

    Supports:
      "[id] helpful=X harmful=Y :: content"
      "[id] content"
    Returns None for section headers and blank lines.
    """
    text = line.strip()
    pattern_full = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    m = re.match(pattern_full, text)
    if m:
        return {
            "id": m.group(1),
            "helpful": int(m.group(2)),
            "harmful": int(m.group(3)),
            "content": m.group(4),
            "raw_line": line,
        }
    pattern_simple = r'\[([^\]]+)\]\s*(.*)'
    m2 = re.match(pattern_simple, text)
    if m2:
        return {
            "id": m2.group(1),
            "helpful": 0,
            "harmful": 0,
            "content": m2.group(2).strip(),
            "raw_line": line,
        }
    return None


def get_next_global_id(playbook_text: str) -> int:
    """Return the next free global bullet counter (max existing + 1)."""
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            m = re.search(r"-(\d+)$", parsed["id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def format_playbook_line(bullet_id: str, helpful: int, harmful: int, content: str) -> str:
    """Format a bullet into the canonical simple form (counts omitted)."""
    return f"[{bullet_id}] {content}"


# ---------------------------------------------------------------------------
# Playbook mutation
# ---------------------------------------------------------------------------

def apply_curator_operations(playbook_text: str, operations: list[dict], next_id: int) -> tuple[str, int]:
    """Apply a list of curator ADD operations to the playbook text.

    Returns the updated playbook string and the new next_id counter.
    Only ADD operations are supported; others are silently skipped.
    """
    lines = playbook_text.strip().split("\n")

    # Build section map
    sections: dict[str, list] = {}
    current_section = "general"
    for i, line in enumerate(lines):
        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            if current_section not in sections:
                sections[current_section] = []
        elif line.strip():
            sections.setdefault(current_section, []).append((i, line))

    bullets_to_add: list[tuple[str, str]] = []

    for op in operations:
        if op.get("type") != "ADD":
            continue
        section_raw = op.get("section", "general")
        section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        if section not in sections and section != "general":
            print(f"[ace] Warning: section '{section_raw}' not found, adding to OTHERS")
            section = "others"
        slug = get_section_slug(section)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "")
        new_line = format_playbook_line(new_id, 0, 0, content)
        bullets_to_add.append((section, new_line))
        print(f"[ace]   Added bullet {new_id} to section '{section}'")

    # Rebuild playbook preserving all existing lines, appending new bullets after each section
    final_lines: list[str] = []
    current_section = None
    remaining = list(bullets_to_add)

    for line in lines:
        if line.strip().startswith("##"):
            if current_section:
                adds = [b for s, b in remaining if s == current_section]
                final_lines.extend(adds)
                remaining = [(s, b) for s, b in remaining if s != current_section]
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        final_lines.append(line)

    # Flush bullets for the last section
    if current_section:
        adds = [b for s, b in remaining if s == current_section]
        final_lines.extend(adds)
        remaining = [(s, b) for s, b in remaining if s != current_section]

    # Orphan bullets (section missing) → append to OTHERS
    if remaining:
        print(f"[ace] Warning: {len(remaining)} bullets had no matching section, appending to OTHERS")
        others_lines = [b for _, b in remaining]
        others_idx = next(
            (i for i, l in enumerate(final_lines) if l.strip().upper() == "## OTHERS"),
            -1,
        )
        if others_idx >= 0:
            for i, b in enumerate(others_lines):
                final_lines.insert(others_idx + 1 + i, b)
        else:
            final_lines.extend(others_lines)

    return "\n".join(final_lines), next_id


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_from_text(text: str, json_key: str | None = None) -> dict | None:
    """Extract the first valid JSON object from LLM output.

    Tries in order:
      1. Parse entire response as JSON (JSON-mode output).
      2. Look for ```json … ``` fences.
      3. Balanced-brace scan.
    """
    if text is None:
        print("[ace] WARNING: extract_json_from_text received None")
        return None
    try:
        # 1. Direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 2. ```json fences
        for m in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue

        # 3. Balanced-brace scan
        i = 0
        while i < len(text):
            if text[i] != "{":
                i += 1
                continue
            depth = 1
            start = i
            i += 1
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif c == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if depth == 0:
                candidate = text[start:i]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[ace] extract_json_from_text error: {e}")
        if len(text) > 500:
            print(f"[ace] raw preview: {text[:500]}...")
        else:
            print(f"[ace] raw: {text}")
    return None
