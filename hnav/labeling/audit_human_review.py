#!/usr/bin/env python3
"""Blind human review of Laya–judge disagreements: sheets and ingest.  [offline]

Two annotators label the same pairs independently, blind to both machine
verdicts, so the paper can report human–human agreement and the judge's
accuracy against human labels with confidence intervals.

    python hnav/labeling/audit_human_review.py build      # -> sheets A and B + key
    python hnav/labeling/audit_human_review.py ingest \\
        --sheet-a human_review_sheet_A.filled.csv --sheet-b human_review_sheet_B.filled.csv \\
        [--adjudicated human_review_adjudication.filled.csv]

Review set (seed 20260824; user decision 2026-09-30, "targeted, two annotators"):
  positive_disagreement   every update_conflict disagreement with tier core,
                          update_only_fork or rejected
  discovered              all discovered_unverified pairs (judge-only positives),
                          agreement or not
  negative_disagreement   100 negative-tier disagreements, stratified by Laya
                          confidence tertile
  agreement_control       100 pairs the two readers agree on, tier-proportional

The sheets carry review_id, subset, the two facts and empty label columns —
no pair_id, tier, cosine or machine verdict. ``human_review_key.json`` maps
review_id back to the pair and stratum; annotators do not open it.

``ingest`` validates both sheets, computes human–human agreement, writes the
pairs the annotators differ on to ``human_review_adjudication.csv`` for a
joint pass, and — once that file is filled and passed back — scores the judge
and Laya against the final human labels per stratum with Wilson intervals.
It never modifies the gold dataset; re-tiering is a separate decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.agreement_stats import (  # noqa: E402
    agreement, cohen_kappa, largest_remainder, table2x2, wilson)
from hnav.labeling.audit_compare_laya import aggregate, read_jsonl  # noqa: E402

DIR = REPO / "stage0_results" / "conflict_pairs"
LAYA = DIR / "audit_results_laya.jsonl"
GOLD = DIR / "gold_conflict_dataset.jsonl.gz"
DIS = DIR / "audit_laya_disagreements.jsonl"
SHEET_A = DIR / "human_review_sheet_A.csv"
SHEET_B = DIR / "human_review_sheet_B.csv"
KEY = DIR / "human_review_key.json"
README = DIR / "human_review_README.md"
ADJ = DIR / "human_review_adjudication.csv"
OUT_JSON = DIR / "HUMAN_REVIEW_SUMMARY.json"
OUT_MD = DIR / "HUMAN_REVIEW_SUMMARY.md"

SEED = 20260824
LABEL_COLS = ["same_referent", "same_relation", "context_overlap",
              "values_incompatible", "update_conflict", "strict_conflict"]
SHEET_COLS = ["review_id", "subset", "fact_a", "fact_b"] + LABEL_COLS + ["notes"]
FORBIDDEN_COLS = {"pair_id", "tier", "cosine_similarity", "judge", "laya", "stratum"}
POSITIVE_TIERS = {"core", "update_only_fork", "rejected"}
N_NEGATIVE_DISAGREEMENT = 100
N_AGREEMENT_CONTROL = 100
YES = {"yes", "y", "true", "1", "t"}
NO = {"no", "n", "false", "0", "f"}

RUBRIC = """# Human review — annotator instructions

You are labelling pairs of sentences from a fact store. Do **not** check whether
either sentence is true in the real world; decide whether the two sentences
express conflicting propositions. Work alone, do not consult the other
annotator, and do not open `human_review_key.json`.

For each pair, first identify what each sentence asserts — subject, attribute
(relation), value, and any validity context (time, place, population,
organization, condition) — then answer yes/no in each column:

- `same_referent`: both sentences are about the same real-world entity
  (aliases, abbreviations and different surface strings can name one entity).
- `same_relation`: both assert the same attribute or relation, even if phrased
  differently.
- `context_overlap`: the propositions apply under compatible or overlapping
  validity conditions.
- `values_incompatible`: the two values cannot both occupy the same slot under
  the same context. Paraphrases, aliases, one value containing the other, and
  values that can coexist are **compatible** (answer no). Do not answer yes
  merely because the strings differ.
- `update_conflict`: under a memory-store convention with one current value
  per (entity, attribute) slot, one sentence replaces the other's value. This
  is yes exactly when the four checks above are all yes, even if the relation
  is naturally multi-valued (citizenships, children, languages, ...).
- `strict_conflict`: the two sentences cannot both be true at the same time in
  the real world. This is yes only when `update_conflict` is yes **and** the
  relation admits a single value at a time.

Write `yes` or `no` in every label cell (no blanks, no "unsure"); use `notes`
for anything you want to flag. Save the file with the same columns.
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_yn(v: str, where: str) -> bool:
    s = (v or "").strip().lower()
    if s in YES:
        return True
    if s in NO:
        return False
    raise SystemExit(f"{where}: expected yes/no, got {v!r}")


def read_sheet(path: pathlib.Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in SHEET_COLS if c not in cols]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        rows = {}
        for i, r in enumerate(reader, start=2):
            rid = (r.get("review_id") or "").strip()
            if not rid:
                raise SystemExit(f"{path}:{i}: empty review_id")
            if rid in rows:
                raise SystemExit(f"{path}:{i}: duplicate review_id {rid}")
            rows[rid] = {c: _parse_yn(r.get(c, ""), f"{path}:{i}:{c}") for c in LABEL_COLS}
            rows[rid]["notes"] = (r.get("notes") or "").strip()
    return rows


def write_sheet(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SHEET_COLS})


# ── selection ────────────────────────────────────────────────────────────────

def select_review_set(gold: list[dict], laya_agg: dict[str, dict], disagreements: list[dict],
                      seed: int = SEED, n_negative: int = N_NEGATIVE_DISAGREEMENT,
                      n_control: int = N_AGREEMENT_CONTROL) -> list[dict]:
    """Return review rows: {pair_id, subset, tier, stratum, fact_a, fact_b}."""
    rng = random.Random(seed)
    gold_by_id = {g["pair_id"]: g for g in gold}
    dis_ids = {d["pair_id"] for d in disagreements}
    chosen: dict[str, str] = {}                     # pair_id -> stratum

    # 1. every positive-tier disagreement
    for d in sorted(disagreements, key=lambda d: d["pair_id"]):
        if d["tier"] in POSITIVE_TIERS:
            chosen[d["pair_id"]] = "positive_disagreement"
    # 2. every discovered pair
    for g in sorted(gold, key=lambda g: g["pair_id"]):
        if g["tier"] == "discovered_unverified" and g["pair_id"] not in chosen:
            chosen[g["pair_id"]] = "discovered"
    # 3. negative-tier disagreements, stratified by Laya confidence tertile
    neg_dis = sorted([d for d in disagreements if d["tier"] == "negative" and d["pair_id"] not in chosen],
                     key=lambda d: d["pair_id"])
    if neg_dis:
        confs = sorted(d["laya"]["update_conflict"]["conf"] for d in neg_dis)
        c1, c2 = confs[len(confs) // 3], confs[(2 * len(confs)) // 3]
        tertiles: dict[int, list[dict]] = defaultdict(list)
        for d in neg_dis:
            c = d["laya"]["update_conflict"]["conf"]
            tertiles[0 if c < c1 else (1 if c < c2 else 2)].append(d)
        seats = largest_remainder([len(tertiles[i]) for i in range(3)], min(n_negative, len(neg_dis)))
        for i in range(3):
            pool = tertiles[i]
            rng.shuffle(pool)
            for d in pool[:seats[i]]:
                chosen[d["pair_id"]] = "negative_disagreement"
    # 4. agreement controls, tier-proportional among agreeing audited pairs
    agree = [g for g in sorted(gold, key=lambda g: g["pair_id"])
             if g["pair_id"] in laya_agg and g["pair_id"] not in dis_ids
             and g["pair_id"] not in chosen and g["tier"] != "discovered_unverified"]
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for g in agree:
        by_tier[g["tier"]].append(g)
    tiers = sorted(by_tier)
    seats = largest_remainder([len(by_tier[t]) for t in tiers], min(n_control, len(agree)))
    for t, k in zip(tiers, seats):
        pool = by_tier[t]
        rng.shuffle(pool)
        for g in pool[:k]:
            chosen[g["pair_id"]] = "agreement_control"

    rows = []
    for pid, stratum in chosen.items():
        g = gold_by_id[pid]
        rows.append({"pair_id": pid, "subset": g["subset"], "tier": g["tier"], "stratum": stratum,
                     "fact_a": g["fact_a"], "fact_b": g["fact_b"]})
    rows.sort(key=lambda r: r["pair_id"])
    rng.shuffle(rows)                               # review_id order carries no tier information
    for i, r in enumerate(rows):
        r["review_id"] = f"R{i:04d}"
    return rows


def build(args) -> int:
    gold = list(read_jsonl(pathlib.Path(args.gold)))
    laya_agg = aggregate(read_jsonl(pathlib.Path(args.laya)))
    disagreements = list(read_jsonl(pathlib.Path(args.disagreements)))
    rows = select_review_set(gold, laya_agg, disagreements, seed=args.seed,
                             n_negative=args.n_negative, n_control=args.n_control)

    key = {r["review_id"]: {"pair_id": r["pair_id"], "tier": r["tier"], "stratum": r["stratum"]}
           for r in rows}
    pathlib.Path(args.key).write_text(json.dumps({
        "what": "review_id -> pair; annotators do not open this file",
        "seed": args.seed, "n": len(rows),
        "by_stratum": Counter(r["stratum"] for r in rows),
        "by_stratum_tier": Counter(f"{r['stratum']}/{r['tier']}" for r in rows),
        "rows": key}, indent=1, ensure_ascii=False, default=dict), encoding="utf-8")

    blind = [{c: r[c] for c in ("review_id", "subset", "fact_a", "fact_b")} for r in rows]
    a = list(blind)
    b = list(blind)
    random.Random(args.seed + 1).shuffle(a)
    random.Random(args.seed + 2).shuffle(b)
    write_sheet(pathlib.Path(args.sheet_a), a)
    write_sheet(pathlib.Path(args.sheet_b), b)
    pathlib.Path(args.readme).write_text(RUBRIC, encoding="utf-8")
    print(f"review set: {len(rows)} pairs; by stratum {dict(Counter(r['stratum'] for r in rows))}")
    print(f"wrote {args.sheet_a}, {args.sheet_b}, {args.key}, {args.readme}")
    return 0


# ── ingest ───────────────────────────────────────────────────────────────────

def merge_labels(sheet_a: dict[str, dict], sheet_b: dict[str, dict],
                 adjudicated: dict[str, dict] | None) -> tuple[dict[str, dict], list[str]]:
    """Final label per review_id: agreement, else the adjudicated value.

    Returns (final, unresolved_ids). ``final[rid][col]`` is None while
    unresolved.
    """
    final: dict[str, dict] = {}
    unresolved: list[str] = []
    for rid in sheet_a:
        a, b = sheet_a[rid], sheet_b[rid]
        out = {}
        open_cols = []
        for c in LABEL_COLS:
            if a[c] == b[c]:
                out[c] = a[c]
            elif adjudicated and rid in adjudicated:
                out[c] = adjudicated[rid][c]
            else:
                out[c] = None
                open_cols.append(c)
        if open_cols:
            unresolved.append(rid)
        final[rid] = out
    return final, unresolved


def _acc_block(truth: list[bool], pred: list[bool]) -> dict:
    n = len(truth)
    k = sum(t == p for t, p in zip(truth, pred))
    return {"n": n, "correct": k, "accuracy": (k / n) if n else None,
            "wilson95": wilson(k, n) if n else None, "table_truth_pred": table2x2(truth, pred)}


def ingest(args) -> int:
    key = json.loads(pathlib.Path(args.key).read_text(encoding="utf-8"))["rows"]
    sheet_a = read_sheet(pathlib.Path(args.sheet_a))
    sheet_b = read_sheet(pathlib.Path(args.sheet_b))
    for name, sh in (("A", sheet_a), ("B", sheet_b)):
        if set(sh) != set(key):
            extra, missing = sorted(set(sh) - set(key)), sorted(set(key) - set(sh))
            raise SystemExit(f"sheet {name}: review_ids differ from the key "
                             f"(extra {extra[:5]}, missing {missing[:5]})")
    adjudicated = None
    if args.adjudicated:
        adjudicated = read_sheet(pathlib.Path(args.adjudicated))

    ids = sorted(key)
    hh = {}
    for c in LABEL_COLS:
        x = [sheet_a[r][c] for r in ids]
        y = [sheet_b[r][c] for r in ids]
        hh[c] = {"n": len(ids), "agreement": agreement(x, y), "kappa": cohen_kappa(x, y),
                 "table_A_B": table2x2(x, y)}

    final, unresolved = merge_labels(sheet_a, sheet_b, adjudicated)
    report: dict = {"n_reviewed": len(ids), "human_human": hh,
                    "n_unresolved": len(unresolved), "adjudicated_file": args.adjudicated}

    if unresolved:
        gold = {g["pair_id"]: g for g in read_jsonl(pathlib.Path(args.gold))}
        with open(args.adjudication_out, "w", newline="", encoding="utf-8") as f:
            cols = ["review_id", "subset", "fact_a", "fact_b"] + \
                   [f"{c}_A" for c in LABEL_COLS] + [f"{c}_B" for c in LABEL_COLS] + LABEL_COLS + ["notes"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for rid in unresolved:
                g = gold[key[rid]["pair_id"]]
                row = {"review_id": rid, "subset": g["subset"], "fact_a": g["fact_a"], "fact_b": g["fact_b"]}
                for c in LABEL_COLS:
                    row[f"{c}_A"] = "yes" if sheet_a[rid][c] else "no"
                    row[f"{c}_B"] = "yes" if sheet_b[rid][c] else "no"
                    row[c] = "yes" if final[rid][c] is True else ("no" if final[rid][c] is False else "")
                row["notes"] = " | ".join(x for x in (sheet_a[rid]["notes"], sheet_b[rid]["notes"]) if x)
                w.writerow(row)
        print(f"{len(unresolved)} pairs need joint adjudication -> {args.adjudication_out}; "
              f"fill every empty label cell and re-run ingest with --adjudicated")
        pathlib.Path(args.out_json).write_text(json.dumps(report, indent=1, default=dict), encoding="utf-8")
        pathlib.Path(args.out_md).write_text(render_md(report), encoding="utf-8")
        return 0

    # scoring against the final human labels
    gold = {g["pair_id"]: g for g in read_jsonl(pathlib.Path(args.gold))}
    laya_agg = aggregate(read_jsonl(pathlib.Path(args.laya)))
    strata = sorted({key[r]["stratum"] for r in ids})
    report["by_stratum_n"] = Counter(key[r]["stratum"] for r in ids)
    report["judge_vs_human"] = {}
    report["laya_vs_human"] = {}
    for lab in ("update_conflict", "strict_conflict"):
        jb, lb = {}, {}
        for name in ["all"] + strata:
            sel = [r for r in ids if name == "all" or key[r]["stratum"] == name]
            truth = [final[r][lab] for r in sel]
            jpred = [gold[key[r]["pair_id"]]["judge"][lab] for r in sel]
            lpred = [laya_agg[key[r]["pair_id"]]["q"][lab]["flag"] for r in sel]
            jb[name] = _acc_block(truth, jpred)
            lb[name] = _acc_block(truth, lpred)
        report["judge_vs_human"][lab] = jb
        report["laya_vs_human"][lab] = lb

    # tier-level judge accuracy, combining fully reviewed disagreements with sampled agreements
    all_gold = list(gold.values())
    dis_ids = {key[r]["pair_id"] for r in ids if key[r]["stratum"].endswith("disagreement")}
    tier_est = {}
    for tier in sorted({g["tier"] for g in all_gold}):
        members = [g for g in all_gold if g["tier"] == tier and g["pair_id"] in laya_agg]
        n_dis = sum(1 for g in members
                    if laya_agg[g["pair_id"]]["q"]["update_conflict"]["flag"] != g["judge"]["update_conflict"])
        n_agr = len(members) - n_dis
        rev_dis = [r for r in ids if key[r]["tier"] == tier and key[r]["pair_id"] in dis_ids]
        rev_agr = [r for r in ids if key[r]["tier"] == tier and key[r]["stratum"] == "agreement_control"]
        acc = lambda rs: (sum(final[r]["update_conflict"] == gold[key[r]["pair_id"]]["judge"]["update_conflict"]
                              for r in rs) / len(rs)) if rs else None
        a_dis, a_agr = acc(rev_dis), acc(rev_agr)
        est = None
        if n_dis + n_agr:
            if n_dis and a_dis is None:
                est = None
            elif n_agr and a_agr is None:
                est = None
            else:
                est = ((n_dis * (a_dis or 0.0)) + (n_agr * (a_agr or 0.0))) / (n_dis + n_agr)
        tier_est[tier] = {"n_tier": len(members), "n_disagree": n_dis, "n_agree": n_agr,
                          "reviewed_disagree": len(rev_dis), "reviewed_agree": len(rev_agr),
                          "judge_acc_on_reviewed_disagree": a_dis, "judge_acc_on_reviewed_agree": a_agr,
                          "estimated_judge_accuracy_on_tier": est}
    report["judge_update_accuracy_by_tier_estimate"] = tier_est
    report["discovered_confirmed_by_humans"] = {
        "n": len([r for r in ids if key[r]["stratum"] == "discovered"]),
        "human_update_conflict_yes": sum(final[r]["update_conflict"] for r in ids if key[r]["stratum"] == "discovered"),
        "note": "re-tiering is a separate decision; the gold dataset is unchanged"}

    pathlib.Path(args.out_json).write_text(json.dumps(report, indent=1, default=dict), encoding="utf-8")
    pathlib.Path(args.out_md).write_text(render_md(report), encoding="utf-8")
    print(f"reviewed {len(ids)} pairs; wrote {args.out_md}, {args.out_json}")
    return 0


def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    return str(x)


def render_md(rep: dict) -> str:
    L = ["# Human review of Laya–judge disagreements — summary", ""]
    L.append(f"- pairs reviewed: {rep['n_reviewed']}; unresolved after both sheets: {rep['n_unresolved']}"
             + (f"; adjudicated file: `{rep['adjudicated_file']}`" if rep.get("adjudicated_file") else ""))
    if "by_stratum_n" in rep:
        L.append(f"- by stratum: {dict(rep['by_stratum_n'])}")
    L.append("")
    L.append("## Human–human agreement (independent, blind)")
    L.append("")
    L.append("| field | n | agreement | κ |")
    L.append("| --- | ---: | ---: | ---: |")
    for c, s in rep["human_human"].items():
        L.append(f"| {c} | {s['n']} | {_f(s['agreement'])} | {_f(s['kappa'])} |")
    L.append("")
    if rep["n_unresolved"]:
        L.append(f"{rep['n_unresolved']} pairs await joint adjudication (`human_review_adjudication.csv`).")
        return "\n".join(L) + "\n"
    for who in ("judge_vs_human", "laya_vs_human"):
        L.append(f"## {who.replace('_', ' ')}")
        L.append("")
        for lab, blocks in rep[who].items():
            L.append(f"### {lab}")
            L.append("")
            L.append("| stratum | n | correct | accuracy | Wilson 95 % |")
            L.append("| --- | ---: | ---: | ---: | --- |")
            for name, b in blocks.items():
                L.append(f"| {name} | {b['n']} | {b['correct']} | {_f(b['accuracy'])} | {_f(b['wilson95'])} |")
            L.append("")
    L.append("## Judge update_conflict accuracy by tier (disagreements fully reviewed, agreements sampled)")
    L.append("")
    L.append("| tier | n | disagree | agree | reviewed dis | reviewed agr | acc on dis | acc on agr | estimated |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t, e in rep["judge_update_accuracy_by_tier_estimate"].items():
        L.append(f"| {t} | {e['n_tier']:,} | {e['n_disagree']:,} | {e['n_agree']:,} | {e['reviewed_disagree']} | "
                 f"{e['reviewed_agree']} | {_f(e['judge_acc_on_reviewed_disagree'])} | "
                 f"{_f(e['judge_acc_on_reviewed_agree'])} | {_f(e['estimated_judge_accuracy_on_tier'])} |")
    L.append("")
    d = rep["discovered_confirmed_by_humans"]
    L.append(f"- discovered pairs reviewed: {d['n']}; humans say update conflict on {d['human_update_conflict_yes']} — {d['note']}")
    L.append("")
    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gold", default=str(GOLD))
    b.add_argument("--laya", default=str(LAYA))
    b.add_argument("--disagreements", default=str(DIS))
    b.add_argument("--sheet-a", default=str(SHEET_A))
    b.add_argument("--sheet-b", default=str(SHEET_B))
    b.add_argument("--key", default=str(KEY))
    b.add_argument("--readme", default=str(README))
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--n-negative", type=int, default=N_NEGATIVE_DISAGREEMENT)
    b.add_argument("--n-control", type=int, default=N_AGREEMENT_CONTROL)
    b.set_defaults(func=build)
    g = sub.add_parser("ingest")
    g.add_argument("--gold", default=str(GOLD))
    g.add_argument("--laya", default=str(LAYA))
    g.add_argument("--key", default=str(KEY))
    g.add_argument("--sheet-a", required=True)
    g.add_argument("--sheet-b", required=True)
    g.add_argument("--adjudicated", default=None)
    g.add_argument("--adjudication-out", default=str(ADJ))
    g.add_argument("--out-json", default=str(OUT_JSON))
    g.add_argument("--out-md", default=str(OUT_MD))
    g.set_defaults(func=ingest)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
