#!/usr/bin/env python3
"""The five-model comparison table, generated from the artifacts.  [E2E-5]

Written because a hand-transcribed table already went wrong once. The committed
reference numbers for sh_6k/sh_32k (`hnav_raw` 94 and 86) came from runs with
``page_source=None`` - the prepass page - while every ``pipelines/`` arm and
every new model uses ``page_source=benchmark``. The native rows are the tell:
28 and 48 there, against 30 and 53 for the same model at benchmark page. Each
delta was internally valid; the TABLE was not, because its columns came from
two different configurations.

So this reads every artifact, reports what configuration each cell was measured
under, and REFUSES to put cells of different configurations in the same table
without saying so. A summary you cannot audit is a summary you cannot defend.

    python -m hnav.geometry_filter.multimodel_summary --subset sh_64k
    python -m hnav.geometry_filter.multimodel_summary --all --out summary.md
"""
from __future__ import annotations

import argparse
import json
import pathlib

from hnav.stage1.stale_suppression_probe import mcnemar_exact_p

REPO = pathlib.Path(__file__).resolve().parents[2]
MECHANISM = "detector_suppress"
SUBSETS = ("sh_6k", "sh_32k", "sh_64k")
ARM_ORDER = ("hnav_raw", "hnav_idonly", "hnav_geo", "hnav_ces",
             "hnav_abtt_noparser", "hnav_abtt")
VOID_MARK = "_VOID"


def cells(include_void: bool = False) -> list[dict]:
    """One record per (model, arm, subset) measured cell."""
    out = []
    for arm_dir in sorted((REPO / "pipelines").iterdir()):
        res = arm_dir / "results"
        if not res.is_dir():
            continue
        for tag in sorted(res.iterdir()):
            if not tag.is_dir():
                continue
            voided = VOID_MARK in tag.name.upper()
            if voided and not include_void:
                continue
            for art in sorted(tag.glob("detector_gap_*.json")):
                try:
                    out.append(_read(art, arm_dir.name, tag.name, voided))
                except Exception as exc:                      # noqa: BLE001
                    out.append({"arm": arm_dir.name, "tag": tag.name,
                                "error": str(exc)[:160]})
    return [c for c in out if "error" not in c]


def _read(path: pathlib.Path, arm: str, tag: str, voided: bool) -> dict:
    a = json.loads(path.read_text(encoding="utf-8"))
    r = a["results"][0]
    q = r["per_question"]
    nat = [bool(x["arms"]["native"]["correct"]) for x in q]
    det = [bool(x["arms"][MECHANISM]["correct"]) for x in q]
    strata = [x["stratum"] for x in q]

    def acc(sel):
        n = [v for v, s in zip(nat, sel) if s]
        d = [v for v, s in zip(det, sel) if s]
        b = sum(1 for x, y in zip(n, d) if x and not y)
        c = sum(1 for x, y in zip(n, d) if y and not x)
        return {"n": len(n), "native": sum(n), "arm": sum(d),
                "net": c - b, "p": mcnemar_exact_p(b, c)}

    aa = r["aa_floor"]
    voids = [k for k, v in (r.get("void_conditions") or {}).items()
             if isinstance(v, dict) and v.get("voids") == "run"
             and v.get("status") == "fail" and not k.startswith("2_")]
    h4 = (r.get("void_conditions") or {}).get("4_no_harmful_suppression", {})
    return {
        "arm": arm, "tag": tag, "voided_run": voided,
        "model": a.get("harness", {}).get("llm_model"),
        "subset": r["subset"],
        "page_source": a.get("page_source"),
        "harness_kind": a.get("harness_kind") or a.get("harness_name") or "retrieval",
        "all": acc([True] * len(q)),
        "conflicted": acc([s == "conflicted" for s in strata]),
        "unique": acc([s == "unique" for s in strata]),
        "aa_discordant": aa["b_native_only"] + aa["c_arm_only"],
        "void_fails": voids,
        "n_suppressed_harmful": (h4.get("observed") or {}).get("n_suppressed_harmful"),
        "n_suppressed_superseded": (h4.get("observed") or {}).get("n_suppressed_superseded"),
        "path": str(path.relative_to(REPO).as_posix()),
    }


def audit(rows: list[dict]) -> list[str]:
    """Every reason these cells might not belong in one table."""
    notes = []
    ps = sorted({str(r["page_source"]) for r in rows})
    if len(ps) > 1:
        notes.append(
            f"MIXED page_source across cells: {ps}. Cells measured under "
            f"different page sources are NOT comparable - the retrieved page "
            f"differs, so native itself differs. Split the table.")
    bad_aa = [f"{r['model']}/{r['arm']}/{r['subset']}={r['aa_discordant']}"
              for r in rows if r["aa_discordant"]]
    if bad_aa:
        notes.append(f"NON-ZERO A/A floor (run is void): {bad_aa}")
    harms = {r["n_suppressed_harmful"] for r in rows
             if r["arm"] == "hnav_geo" and r["subset"] == "sh_64k"
             and r["n_suppressed_harmful"] is not None}
    if len(harms) == 1 and harms != {0}:
        notes.append(
            f"hnav_geo sh_64k n_suppressed_harmful is {harms.pop()} on EVERY "
            f"model - suppression plans contain no LLM, so harm cannot vary "
            f"with the answering model. Void by condition 4 throughout.")
    return notes


def table(rows: list[dict], subset: str, stratum: str) -> list[str]:
    sel = [r for r in rows if r["subset"] == subset]
    models = sorted({r["model"] for r in sel}, key=lambda m: str(m))
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in sel)]
    L = [f"### {subset} · {stratum} stratum", "",
         "| model | native | " + " | ".join(arms) + " |",
         "|---" * (len(arms) + 2) + "|"]
    for m in models:
        nat = next((r[stratum]["native"] for r in sel if r["model"] == m), None)
        cellstr = []
        for a in arms:
            r = next((x for x in sel if x["model"] == m and x["arm"] == a), None)
            if r is None:
                cellstr.append("—")
                continue
            s = f"{r[stratum]['arm']} ({r[stratum]['net']:+d})"
            if r["void_fails"]:
                s += " **VOID**"
            cellstr.append(s)
        L.append(f"| {m} | {nat} | " + " | ".join(cellstr) + " |")
    return L + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, choices=SUBSETS)
    ap.add_argument("--stratum", default="all",
                    choices=("all", "conflicted", "unique"))
    ap.add_argument("--all", action="store_true", help="every subset")
    ap.add_argument("--include-void", action="store_true",
                    help="include runs whose directory is marked _VOID")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = cells(include_void=args.include_void)
    if not rows:
        raise SystemExit("no artifacts found under pipelines/*/results/")

    L = ["# Multi-model comparison (generated from artifacts)", "",
         f"{len(rows)} measured cells · mechanism `{MECHANISM}` · "
         f"page_source `{sorted({str(r['page_source']) for r in rows})}`", ""]
    for note in audit(rows):
        L += [f"> **{note}**", ""]
    subsets = SUBSETS if (args.all or not args.subset) else (args.subset,)
    strata = ("all", "conflicted", "unique") if args.all else (args.stratum,)
    for s in subsets:
        for st in strata:
            L += table(rows, s, st)

    L += ["## Provenance", "",
          "| model | arm | subset | page_source | A/A | void | harmful |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (str(x["model"]), x["arm"], x["subset"])):
        L.append(f"| {r['model']} | {r['arm']} | {r['subset']} | "
                 f"{r['page_source']} | {r['aa_discordant']} | "
                 f"{';'.join(r['void_fails']) or '-'} | "
                 f"{r['n_suppressed_harmful'] if r['n_suppressed_harmful'] is not None else '-'} |")

    text = "\n".join(L)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8",
                                          newline="\n")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
