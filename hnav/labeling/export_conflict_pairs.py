#!/usr/bin/env python3
"""Export the parser-tagged conflict pairs as a paired list.  [offline]

Purpose: the conflict labels every measurement rests on come from the template
parser (``conflict_analysis.parse``). This exports each tagged pair — serials,
both fact texts, the shared ``(relation, subject)`` key and the two objects —
so an INDEPENDENT judge (e.g. an LLM asked "do these two sentences state
conflicting facts about the same thing?") can audit the labels one pair at a
time. The export is deliberately self-contained per pair: the judge needs no
access to the benchmark, the parser, or any other file.

Ground rules inherited from the parser:
  * a pair = one ``(relation, subject)`` key with two DISTINCT objects;
  * orientation is earlier-serial -> later-serial (the later fact supersedes);
  * every conflicted key in these subsets has exactly two members — asserted,
    not assumed, so a future dataset change cannot silently truncate groups.

Usage:
    python hnav/labeling/export_conflict_pairs.py            # sh_6k sh_32k sh_64k
    python hnav/labeling/export_conflict_pairs.py --subsets sh_6k

Writes ``stage0_results/conflict_pairs/conflict_pairs.json``.
Stdlib only, no gold answers read: only the ``context`` field is touched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.conflict_analysis import parse  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
OUT = REPO / "stage0_results" / "conflict_pairs" / "conflict_pairs.json"
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)   # the raw context is line-anchored
EXPECTED = {"sh_6k": 160, "sh_32k": 835, "sh_64k": 1687}


def pairs_for(context: str) -> tuple[list[dict], int, int]:
    """All parser-tagged conflict pairs of one subset's raw context."""
    groups: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    n_facts = n_unparsed = 0
    for num, text in FACT_RE.findall(context):
        n_facts += 1
        p = parse(text)
        if p is None:
            n_unparsed += 1
            continue
        rel, subj, obj = p
        groups[(rel, subj)].append((int(num), text.strip(), obj))
    out = []
    for (rel, subj), members in groups.items():
        if len({obj for _, _, obj in members}) < 2:
            continue                       # unique key: no conflict
        assert len(members) == 2, (
            f"conflict group size {len(members)} for key ({rel!r}, {subj!r}) — "
            f"the exporter assumes pairs; extend it before trusting the output")
        (s_a, t_a, o_a), (s_b, t_b, o_b) = sorted(members)
        out.append({
            "serial_earlier": s_a, "serial_later": s_b,
            "id_earlier": f"fact:{s_a}", "id_later": f"fact:{s_b}",
            "text_earlier": t_a, "text_later": t_b,
            "relation": rel, "subject": subj,
            "object_earlier": o_a, "object_later": o_b,
        })
    out.sort(key=lambda r: (r["serial_earlier"], r["serial_later"]))
    return out, n_facts, n_unparsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["sh_6k", "sh_32k", "sh_64k"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {it["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
             .replace("factconsolidation_", ""): it for it in data}

    payload = {
        "what": "parser-tagged conflict pairs, for independent (LLM) label audit",
        "pair_definition": "same (relation, subject) key, two distinct objects; "
                           "earlier serial -> later serial; the later fact "
                           "supersedes the earlier",
        "producer": "hnav/labeling/export_conflict_pairs.py, delegating to the "
                    "validated conflict_analysis.parse (99.5%+ template coverage)",
        "note": "unparsed facts (~0.5%) carry no label and so appear in no pair; "
                "an audit of these pairs bounds the parser's PRECISION, not its "
                "recall on unparsed text",
        "subsets": {},
    }
    for s in args.subsets:
        if s not in items:
            print(f"unknown subset {s}", file=sys.stderr)
            return 1
        pairs, n_facts, n_unparsed = pairs_for(items[s]["context"])
        payload["subsets"][s] = {
            "n_facts": n_facts, "n_unparsed": n_unparsed,
            "n_conflict_pairs": len(pairs), "pairs": pairs,
        }
        exp = EXPECTED.get(s)
        flag = "" if exp is None else (" == committed" if len(pairs) == exp
                                       else f" != COMMITTED {exp} — STOP")
        print(f"{s:7s} facts={n_facts:5d} unparsed={n_unparsed:3d} "
              f"conflict_pairs={len(pairs):5d}{flag}")
        if exp is not None and len(pairs) != exp:
            return 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
