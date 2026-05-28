"""ACE (Agentic Context Engineering) memory — no-ground-truth path.

Ports the reflector→curator→playbook update loop from:
  memory_official_implentations/ace-appworld/experiments/code/ace/
following the no-GT path (solve_task_wo_gt + reflector_no_gt_prompt).

Key differences from the original:
  - No ReAct/generator loop: CL-bench already has the model's response.
  - No AppWorld evaluator, no test_report, no GT code.
  - No per-bullet helpful/harmful counters or reflector-side tagging.
  - No disk LLM-call cache; no snapshot-every-N-tasks.
  - Sections adapted to CL-bench's QA (knowledge/rule/procedural) tasks.
  - All state is instance-local; one instance per context, thread-safe.
"""

import json
import os
import re
import time

from .api_client import call_openai_api
from .base import Memory
from .logging_utils import log

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TEXT_CHARS = 8000

# CL-bench-friendly section taxonomy (6 sections + slug prefixes)
_SECTION_SLUGS = {
    "key_facts_and_definitions": "kfd",
    "rules_and_constraints": "rule",
    "strategies_and_heuristics": "strat",
    "common_mistakes_and_pitfalls": "cmp",
    "verification_checklist": "vc",
    "others": "misc",
}

_INITIAL_PLAYBOOK = """\
## KEY FACTS AND DEFINITIONS

## RULES AND CONSTRAINTS

## STRATEGIES AND HEURISTICS

## COMMON MISTAKES AND PITFALLS

## VERIFICATION CHECKLIST

## OTHERS
"""

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow is a curated playbook of insights accumulated from prior tasks "
    "in this same knowledge context. Use it when relevant to the current question.\n\n"
)

# ── Prompts ───────────────────────────────────────────────────────────────────
# Both prompts use str.replace for substitution (avoids brace-escaping pain
# in the JSON-example blocks). Slot names: {generated_response}, {playbook},
# {conversation_history}, {question_context}, {current_playbook},
# {final_generated_response}, {guidebook}.

REFLECTOR_PROMPT = """\
You are an expert educator and analyst. Your job is to diagnose the current trajectory: \
identify what went wrong (or could be better) based on the model's response and the available context.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong or could be improved
- Identify specific conceptual errors, overlooked information, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in future similar tasks
- Identify root causes: wrong interpretation, overlooked evidence, incomplete reasoning, or missing attention
- Provide concrete corrections the model should take in this type of task
- Be specific about what the model should have done differently
- You will receive a playbook that is used by the model to answer questions
- You need to analyze this playbook and reason whether the current bullets helped or hurt

**Inputs:**

- Generated response (candidate to critique):
<<<GENERATED_RESPONSE_START>>>
{generated_response}
<<<GENERATED_RESPONSE_END>>>

- (Optional) Playbook (used by the model for answering):
<<<PLAYBOOK_GUIDE>>>
{playbook}
<<<PLAYBOOK_GUIDE>>>

- Full conversation history:
<<<CONVERSATION_HISTORY>>>
{conversation_history}
<<<CONVERSATION_HISTORY>>>

**Examples:**

**Example 1:**
Generated Response: [Response that incorrectly identified a game rule by focusing on surface wording rather than the underlying exception clause.]

Response:
{{
  "reasoning": "The model fixated on the main rule statement while overlooking an exception clause that qualified it. In knowledge-intensive QA, exception clauses typically override the general rule and must be checked first.",
  "error_identification": "The model applied the general rule without checking whether an exception clause applied to the specific scenario.",
  "root_cause_analysis": "The agent treated the first matching rule as sufficient, ignoring subsequent qualifications.",
  "correct_approach": "Always scan the full relevant section for exception clauses or conditional overrides before committing to an answer.",
  "key_insight": "Exception clauses take precedence over general rules; scan the full context before answering."
}}

**Example 2:**
Generated Response: [Response that gave the wrong numeric result because it used a default formula instead of reading the domain-specific formula provided in the context.]

Response:
{{
  "reasoning": "The model defaulted to a common real-world formula rather than using the formula explicitly stated in the provided domain documentation. Context-provided definitions always override prior knowledge.",
  "error_identification": "The model used an assumed formula instead of the one defined in the context document.",
  "root_cause_analysis": "The agent relied on parametric knowledge rather than attending to the context-specific definition.",
  "correct_approach": "Explicitly locate and use any formula, definition, or procedure given in the context document rather than substituting a default.",
  "key_insight": "Context-provided formulas and definitions take priority over general real-world knowledge."
}}

**Outputs:**
Your output should be a JSON object with the following fields:
  - reasoning: your chain of thought / reasoning / thinking process
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy or principle should be remembered to avoid this error?

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought]",
  "error_identification": "[What specifically went wrong]",
  "root_cause_analysis": "[Why did this error occur]",
  "correct_approach": "[What should the model have done instead]",
  "key_insight": "[What strategy or principle to remember]"
}}
"""

CURATOR_PROMPT = """\
You are a master curator of knowledge. Your job is to identify what new insights should be \
added to an existing playbook based on a reflection from a previous task attempt.

**Context:**
- The playbook you curate will be used to help answer similar questions in the future.
- The reflection was generated without access to ground truth answers, so focus on \
meta-level strategies and attention points that generalize to future tasks.

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy — if similar advice already exists, only add content that is a perfect complement
- Do NOT regenerate the entire playbook — only provide the additions needed
- Focus on quality over quantity — a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- If no new content to add, return an empty list for the operations field
- Be concise and specific — each addition should be actionable

- **Task Context (the actual question/task):**
  `{question_context}`

- **Current Playbook:**
  `{current_playbook}`

- **Generated Response (latest attempt):**
  `{final_generated_response}`

- **Current Reflections (principles and strategies from analyzing the current attempt):**
  `{guidebook}`

- **Full Conversation History:**
  `{conversation_history}`

**Examples:**

**Example 1:**
Task Context: "Under the game rules provided, what happens when a player rolls a 6 twice in a row?"
Current Playbook: [Basic strategy guidelines]
Generated Response: [Response that missed the double-6 exception clause]
Reflections: "The agent failed to notice an exception clause that overrides the default rule for consecutive rolls."

Response:
{{
  "reasoning": "The reflection identifies a critical oversight: the model applied a general rule without checking the exception clause for consecutive identical rolls. This generalizable pattern should be captured.",
  "operations": [
    {{
      "type": "ADD",
      "section": "rules_and_constraints",
      "content": "Always check for exception clauses or conditional overrides before applying any general rule. Exception clauses typically appear after the main rule statement and take precedence."
    }}
  ]
}}

**Example 2:**
Task Context: "Calculate the compound interest using the formula defined in the document."
Current Playbook: [Basic guidelines]
Generated Response: [Response using a standard financial formula instead of the context-defined one]
Reflections: "The agent substituted a real-world formula for the one explicitly defined in the context."

Response:
{{
  "reasoning": "The reflection shows a common failure mode: substituting parametric knowledge for context-provided definitions. This pattern generalizes across many domain-knowledge tasks.",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_heuristics",
      "content": "Context-provided formulas, definitions, and procedures always override real-world defaults. Locate and use the context-specific version before applying any prior knowledge."
    }}
  ]
}}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process
- operations: a list of ADD operations to perform on the playbook
  - type: must be "ADD"
  - section: one of: key_facts_and_definitions, rules_and_constraints, strategies_and_heuristics, common_mistakes_and_pitfalls, verification_checklist, others
  - content: the new content of the bullet point

**RESPONSE FORMAT — Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_heuristics",
      "content": "[New insight or strategy here...]"
    }}
  ]
}}
"""

# ── Playbook helpers (ported from ace/playbook.py) ────────────────────────────

def _normalize_section(name: str) -> str:
    return name.lower().replace(" ", "_").replace("&", "and").replace("-", "_").rstrip(":")


def _get_section_slug(section: str) -> str:
    return _SECTION_SLUGS.get(section, "misc")


def _parse_playbook_line(line: str):
    """Parse '[id] content' line. Returns dict or None."""
    m = re.match(r'\[([^\]]+)\]\s*(.*)', line.strip())
    if m:
        return {"id": m.group(1), "content": m.group(2).strip()}
    return None


def _format_playbook_line(bullet_id: str, content: str) -> str:
    return f"[{bullet_id}] {content}"


def _get_next_global_id(playbook_text: str) -> int:
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = _parse_playbook_line(line)
        if parsed:
            m = re.search(r"-(\d+)$", parsed["id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def _has_bullets(playbook_text: str) -> bool:
    return any(_parse_playbook_line(l) for l in playbook_text.strip().split("\n"))


def _apply_add_operations(playbook_text: str, operations: list, next_id: int):
    """Apply ADD operations to playbook. Returns (new_playbook_text, new_next_id)."""
    lines = playbook_text.strip().split("\n")

    # Build set of known normalized section names
    known_sections = set()
    for line in lines:
        if line.strip().startswith("##"):
            known_sections.add(_normalize_section(line.strip()[2:].strip()))

    bullets_to_add = []  # list of (section, line_str)
    for op in operations:
        if op.get("type") != "ADD":
            continue
        section = _normalize_section(op.get("section", "others"))
        if section not in known_sections:
            section = "others"
        slug = _get_section_slug(section)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "")
        bullets_to_add.append((section, _format_playbook_line(new_id, content)))
        log(f"   ACE: added bullet {new_id} to section '{section}'")

    if not bullets_to_add:
        return playbook_text, next_id

    # Rebuild: after each section header, flush bullets queued for that section
    final_lines = []
    current_section = None
    for line in lines:
        if line.strip().startswith("##"):
            if current_section is not None:
                for sec, bline in bullets_to_add:
                    if sec == current_section:
                        final_lines.append(bline)
                bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]
            current_section = _normalize_section(line.strip()[2:].strip())
        final_lines.append(line)

    # Flush the last section
    if current_section:
        for sec, bline in bullets_to_add:
            if sec == current_section:
                final_lines.append(bline)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    # Any remaining bullets that couldn't find a section → append to OTHERS header
    if bullets_to_add:
        others_idx = next(
            (i for i, l in enumerate(final_lines) if l.strip() == "## OTHERS"), -1
        )
        remaining = [b for _, b in bullets_to_add]
        if others_idx >= 0:
            for i, bline in enumerate(remaining):
                final_lines.insert(others_idx + 1 + i, bline)
        else:
            final_lines.extend(remaining)

    return "\n".join(final_lines), next_id


def _extract_json_from_text(text: str):
    """Robustly extract a JSON object from LLM output (ported from ace/playbook.py)."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Balanced-brace heuristic
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth, start = 1, i
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
                try:
                    return json.loads(text[start:i])
                except json.JSONDecodeError:
                    pass
        else:
            i += 1
    return None

# ── Internal helpers ──────────────────────────────────────────────────────────

def _truncate_head_tail(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _sum_usage(usages: list) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["total_tokens"] += u.get("total_tokens", 0)
    return total


def _extract_model_output(content: str) -> str:
    """Extract the **[Model Output]** block from a format_trajectory string."""
    marker = "**[Model Output]**"
    idx = content.rfind(marker)
    if idx >= 0:
        return content[idx + len(marker):].strip()
    return content.strip()


# ── Main class ────────────────────────────────────────────────────────────────

class ACEMemory(Memory):
    """ACE memory — no-GT path.

    Maintains a structured playbook (plain text with section headers and
    ID-prefixed bullets). Retrieval injects the whole playbook into the
    system prompt (no embeddings). Extraction runs reflector → curator →
    appends new bullets to the playbook in-memory and on disk.
    """

    def __init__(
        self,
        client,
        model,
        memory_dir,
        embed_provider="dashscope",     # accepted, unused — ACE has no embeddings
        embed_model="text-embedding-v4", # accepted, unused
        embed_client=None,               # accepted, unused
        top_k=10,                        # accepted, unused — whole-playbook injection
        extract_model=None,
        use_reflector=True,
        max_text_chars=_MAX_TEXT_CHARS,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.extract_model = extract_model or model
        self.memory_dir = memory_dir
        self.use_reflector = use_reflector
        self.max_text_chars = max_text_chars

        self._llm_usage: list = []

        os.makedirs(memory_dir, exist_ok=True)
        self.playbook_path = os.path.join(memory_dir, "playbook.txt")
        self._load_playbook()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_playbook(self) -> None:
        if os.path.exists(self.playbook_path):
            with open(self.playbook_path, "r", encoding="utf-8") as f:
                self.playbook = f.read()
            self._next_id = _get_next_global_id(self.playbook)
            bullet_count = sum(1 for l in self.playbook.splitlines() if _parse_playbook_line(l))
            if bullet_count > 0:
                log(f"   ACE: loaded playbook ({bullet_count} bullets) from disk")
        else:
            self.playbook = _INITIAL_PLAYBOOK
            self._next_id = 1

    def _save_playbook(self) -> None:
        with open(self.playbook_path, "w", encoding="utf-8") as f:
            f.write(self.playbook)

    # ── Internal: reflector ──────────────────────────────────────────────────

    def _call_reflector(self, generated_response: str, conversation_history: str) -> str:
        prompt = (
            REFLECTOR_PROMPT
            .replace("{generated_response}", generated_response)
            .replace("{playbook}", self.playbook)
            .replace("{conversation_history}", conversation_history)
        )
        text, error, usage = call_openai_api(
            self.client,
            [{"role": "user", "content": prompt}],
            self.extract_model,
            temperature=0.7,
        )
        if usage:
            self._llm_usage.append(usage)
        if error or not text:
            log(f"   ACE: reflector call failed: {error}")
            return ""
        return text

    # ── Internal: curator ────────────────────────────────────────────────────

    def _call_curator(
        self,
        question_context: str,
        final_generated_response: str,
        guidebook: str,
        conversation_history: str,
    ) -> None:
        prompt = (
            CURATOR_PROMPT
            .replace("{question_context}", question_context)
            .replace("{current_playbook}", self.playbook)
            .replace("{final_generated_response}", final_generated_response)
            .replace("{guidebook}", guidebook)
            .replace("{conversation_history}", conversation_history)
        )
        text, error, usage = call_openai_api(
            self.client,
            [{"role": "user", "content": prompt}],
            self.extract_model,
            temperature=0.7,
        )
        if usage:
            self._llm_usage.append(usage)
        if error or not text:
            log(f"   ACE: curator call failed: {error}")
            return

        parsed = _extract_json_from_text(text)
        if not parsed:
            log(f"   ACE: curator output not parseable — playbook unchanged")
            log(f"   ACE: raw curator output preview: {text[:200]!r}")
            return

        operations = parsed.get("operations", [])
        if not isinstance(operations, list):
            log(f"   ACE: 'operations' field is not a list — skipping")
            return

        valid_ops = [op for op in operations if isinstance(op, dict) and op.get("type") == "ADD"]
        if not valid_ops:
            log(f"   ACE: no valid ADD operations in curator output")
            return

        new_playbook, new_next_id = _apply_add_operations(self.playbook, valid_ops, self._next_id)
        self.playbook = new_playbook
        self._next_id = new_next_id
        self._save_playbook()

    # ── Compatibility ─────────────────────────────────────────────────────────

    @property
    def memory_bank(self) -> list:
        """Return the list of bullet lines in the current playbook (for stats reporting)."""
        return [l for l in self.playbook.splitlines() if _parse_playbook_line(l)]

    # ── Memory ABC ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        t0 = time.time()
        if not _has_bullets(self.playbook):
            return "", {"latency_s": time.time() - t0, "embed_usage": {}}
        memory_text = MEMORY_INJECTION_PREFIX + self.playbook
        return memory_text, {"latency_s": time.time() - t0, "embed_usage": {}}

    def extract(self, content: str, **kwargs) -> dict:
        query = kwargs.get("query", "")

        self._llm_usage.clear()
        t0 = time.time()

        generated_response = _extract_model_output(content)
        conversation_history = _truncate_head_tail(content, self.max_text_chars)

        guidebook = ""
        if self.use_reflector:
            guidebook = self._call_reflector(generated_response, conversation_history)
            log(f"   ACE: reflector done ({len(guidebook)} chars)")

        self._call_curator(
            question_context=query,
            final_generated_response=generated_response,
            guidebook=guidebook,
            conversation_history=conversation_history,
        )

        return {
            "latency_s": time.time() - t0,
            "llm_usage": _sum_usage(self._llm_usage),
            "embed_usage": {},
        }
