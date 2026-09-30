#!/usr/bin/env python3
"""Does the model track SERIAL recency or TEXT POSITION?  [T14, offline re-analysis]

Answers a question put by a thesis advisor: *if you swap the positions of the
contradictory old and new facts, does the response change?*

No new model calls. This re-reads the RAW OUTPUTS already committed by the
oracle probe (``stale_suppression_probe.py``, calibration) and the confirmatory
run (``detector_gap.py --confirmatory``, held out) and re-classifies each answer
by WHICH VALUE the model actually produced, instead of by right/wrong:

    NEW    the value carried by the highest-serial fact of the queried key
    OLD    the value carried by a superseded fact of the same key
    OTHER  neither (refusal, off-list, malformed)

Right/wrong collapses OLD and OTHER together, which is exactly the distinction
the question turns on, so the accuracy tables in ``STAGE0_REPORT.md`` cannot
answer it and this module can.

Why the existing arms answer the question
-----------------------------------------
The arena supplies TWO recency cues that the native layout CONFOUNDS, because
facts are listed in ascending serial order:

    symbolic   the serial number, plus a prompt that states "the newer fact has
               larger serial number"
    positional raw text order

``anti`` dissociates them: the highest-serial fact is moved to the FRONT and the
most recent stale fact to the END, so the two ends literally swap. The symbolic
cue still says the front fact is newer; the positional cue now says the back one
is. Whichever the model follows, it must follow it here.

``oracle_recency`` / ``detector_demote_late`` move the newest fact to the END,
intensifying the positional cue while leaving the symbolic one aligned.

``oracle_suppress`` / ``detector_suppress`` delete the stale fact, giving the
no-competitor reference against which both placement arms are read.

The A/A arm (``native_repeat``) is what makes the readout interpretable: it is
the same prompt called twice, so its change count is the noise floor for
"answer changed vs native". It is 0 on every subset, so every change counted
below is caused by the surgery.

Offline tier: reads ``answers`` to name the gold value, so it lives under
``hnav/stage1/`` alongside the other oracles (brief §1 rule 1). Nothing here
ships and nothing online imports it.

Usage
-----
    python3 hnav/stage1/position_taxonomy.py
    python3 hnav/stage1/position_taxonomy.py --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hnav.labeling.conflict_analysis import parse  # noqa: E402  (validated parser; do not rewrite)

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
RESULTS = REPO / "stage0_results/stage1"
FACT_RE_LINE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)

# (probe file, subset, split, arm -> human label). Arm names differ between the
# oracle probe and the confirmatory run; the ROLE of each arm is the same.
SOURCES = [
    (RESULTS / "stale_suppression_probe_sh6k.json", "factconsolidation_sh_6k", "calibration",
     {"native": "native (serial order)", "native_repeat": "native repeat (A/A floor)",
      "oracle_recency": "NEW -> END", "anti": "NEW -> FRONT, OLD -> END",
      "oracle_suppress": "OLD deleted"}),
    (RESULTS / "stale_suppression_probe_sh32k.json", "factconsolidation_sh_32k", "calibration",
     {"native": "native (serial order)", "native_repeat": "native repeat (A/A floor)",
      "oracle_recency": "NEW -> END", "anti": "NEW -> FRONT, OLD -> END",
      "oracle_suppress": "OLD deleted"}),
    (RESULTS / "detector_gap_confirmatory_sh64k.json", "factconsolidation_sh_64k", "held out",
     {"native": "native (serial order)", "native_repeat": "native repeat (A/A floor)",
      "detector_demote_late": "NEW -> END", "detector_anti": "NEW -> FRONT",
      "detector_suppress": "OLD deleted"}),
]


def norm(text: str | None) -> str:
    """Casefold, collapse whitespace, drop a trailing period.

    The benchmark's own grader is ``substring_exact_match`` on the raw string;
    this is the same comparison with the formatting noise removed, applied
    symmetrically to gold, stale and output so no side is favoured.
    """
    return re.sub(r"\s+", " ", (text or "").strip().lower().rstrip(" .")).strip()


def load_facts(subset: str) -> dict[int, str]:
    data = json.load(open(DATA, encoding="utf-8"))
    for item in data:
        if item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] == subset:
            return {int(n): t.strip() for n, t in FACT_RE_LINE.findall(item["context"])}
    raise KeyError(subset)


def stale_values(record: dict, facts: dict[int, str],
                 by_key: dict[tuple[str, str], list[tuple[int, str]]]) -> list[str]:
    """The superseded value(s) of THIS question's key.

    The oracle probe records ``plan.stale_serials`` for the queried key directly.
    The confirmatory run's plan spans every verified group on the page, so the
    queried key's rivals are recovered from the parsed index instead — any value
    of the key that is not the gold value.
    """
    gold = [norm(t) for t in record["truths"]]
    plan = record.get("plan") or {}
    serials = plan.get("stale_serials")
    if serials is not None:
        out = []
        for s in serials:
            p = parse(facts.get(s, ""))
            if p:
                out.append(norm(p[2]))
        return [v for v in out if v]
    return [norm(v) for _, v in by_key.get(tuple(record["key"]), [])
            if v and not any(g and g in norm(v) for g in gold)]


def analyse(path: pathlib.Path, subset: str, split: str, arms: dict[str, str]) -> dict:
    facts = load_facts(subset)
    by_key: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    for serial, text in facts.items():
        p = parse(text)
        if p:
            by_key[(p[0], p[1])].append((serial, p[2]))

    payload = json.load(open(path, encoding="utf-8"))
    result = payload["results"][0]
    rows = [q for q in result["per_question"] if q["stratum"] == "conflicted"]

    tally = {a: {"NEW": 0, "OLD": 0, "OTHER": 0} for a in arms}
    changed = {a: 0 for a in arms}
    skipped = 0

    for q in rows:
        gold = [g for g in (norm(t) for t in q["truths"]) if g]
        stale = stale_values(q, facts, by_key)
        if not gold or not stale:
            skipped += 1
            continue
        native_out = norm(q["arms"]["native"]["output"])
        for arm in arms:
            out = norm(q["arms"][arm]["output"])
            if any(g in out for g in gold):
                tally[arm]["NEW"] += 1
            elif any(v in out for v in stale):
                tally[arm]["OLD"] += 1
            else:
                tally[arm]["OTHER"] += 1
            if arm != "native" and out != native_out:
                changed[arm] += 1

    return {"subset": subset, "split": split, "source": str(path.relative_to(REPO)),
            "n_conflicted": len(rows), "n_scored": len(rows) - skipped,
            "n_unresolvable_stale_value": skipped,
            "arms": {arms[a]: {**tally[a],
                               "n_changed_vs_native": (None if a == "native" else changed[a])}
                     for a in arms}}


def render(block: dict) -> str:
    n = block["n_scored"]
    head = (f"\n{'=' * 78}\n{block['subset']}  [{block['split']}]  conflicted n={n}"
            f"  (unresolvable stale value: {block['n_unresolvable_stale_value']})\n"
            f"source: {block['source']}\n{'=' * 78}\n"
            f"{'arm':<30}{'NEW':>6}{'OLD':>6}{'OTHER':>7}{'changed vs native':>21}\n")
    body = ""
    for label, t in block["arms"].items():
        ch = "-" if t["n_changed_vs_native"] is None else f"{t['n_changed_vs_native']} / {n}"
        body += f"{label:<30}{t['NEW']:>6}{t['OLD']:>6}{t['OTHER']:>7}{ch:>21}\n"
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=pathlib.Path, help="also write the blocks as JSON")
    args = ap.parse_args()

    blocks = []
    for path, subset, split, arms in SOURCES:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        block = analyse(path, subset, split, arms)
        blocks.append(block)
        print(render(block))

    if not blocks:
        print("no probe outputs found under stage0_results/stage1/", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps({"blocks": blocks}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
