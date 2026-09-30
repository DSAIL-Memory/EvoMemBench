#!/usr/bin/env python3
"""Laya vs GPT-5-mini: agreement report over the audited candidates.  [offline]

Reads the second reader's rows (``audit_laya.py``), the gold dataset (judge
verdicts and tiers) and the candidate universe, and writes

  stage0_results/conflict_pairs/LAYA_AGREEMENT.md / .json   the report
  stage0_results/conflict_pairs/audit_laya_disagreements.jsonl
        every audited pair where Laya's update_conflict differs from the
        judge's, with both verdicts and a review priority
  stage0_results/conflict_pairs/laya_tail_screen.jsonl
        the never-judged candidates ranked by Laya's P(update conflict)

Nothing here relabels anything. Agreement between two machine readers is a
description of the labels, not a correction; the correction step is the
human review (``audit_human_review.py``).

Per pair, the two input orders are averaged: ``p`` is the mean probability
mass on the True option, the flag is ``p >= 0.5``, ``flip`` records whether
the two orders' argmax disagreed, and ``conf = |2p - 1|``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.agreement_stats import (  # noqa: E402
    agreement, cohen_kappa, rank_auc, rogan_gladen, table2x2, wilson)
from hnav.labeling.audit_laya import FLAGS  # noqa: E402

DIR = REPO / "stage0_results" / "conflict_pairs"
LAYA = DIR / "audit_results_laya.jsonl"
GOLD = DIR / "gold_conflict_dataset.jsonl.gz"
GOLD_SUMMARY = DIR / "gold_conflict_dataset.summary.json"
CANDIDATES = DIR / "audit_candidates_cos080.jsonl.gz"
OUT_JSON = DIR / "LAYA_AGREEMENT.json"
OUT_MD = DIR / "LAYA_AGREEMENT.md"
OUT_DIS = DIR / "audit_laya_disagreements.jsonl"
OUT_TAIL = DIR / "laya_tail_screen.jsonl"

ALIGNMENT_FLAGS = ["same_referent", "same_relation", "context_overlap",
                   "values_incompatible", "relation_allows_multiple_values"]
LABELS = ["update_conflict", "strict_conflict"]
TIERS = ["core", "update_only_fork", "rejected", "discovered_unverified", "negative"]
TIER_PRIORITY = {"core": 0, "update_only_fork": 0, "rejected": 0,
                 "discovered_unverified": 1, "negative": 2}
GALILEO = "sh_32k:1477-1870"      # the known judge error, quarantined in discovered


# ── I/O ──────────────────────────────────────────────────────────────────────

def _open(path: pathlib.Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else open(path, "rt", encoding="utf-8")


def resolve(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    raise SystemExit(f"missing: {path} (or .gz)")


def read_jsonl(path: pathlib.Path):
    with _open(resolve(path)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ── aggregation over orders ──────────────────────────────────────────────────

def aggregate(rows) -> dict[str, dict]:
    """Laya rows (one per pair × order) -> one record per pair."""
    by_pair: dict[str, dict] = {}
    for r in rows:
        rec = by_pair.setdefault(r["pair_id"], {
            "pair_id": r["pair_id"], "subset": r["subset"],
            "parser_tagged_conflict": r.get("parser_tagged_conflict"),
            "orders": {}, "meta": r.get("meta")})
        rec["orders"][r["order"]] = r
    out: dict[str, dict] = {}
    for pid, rec in by_pair.items():
        agg = {"pair_id": pid, "subset": rec["subset"],
               "parser_tagged_conflict": rec["parser_tagged_conflict"],
               "n_orders": len(rec["orders"]), "meta": rec["meta"], "q": {}}
        for q in FLAGS:
            ps, choices = [], []
            for o in rec["orders"].values():
                a = o["answers"][q]
                p = a.get("p_true")
                if p is None:                       # no probabilities recorded
                    p = 1.0 if o["verdict"][q] else 0.0
                ps.append(float(p))
                choices.append(bool(o["verdict"][q]))
            p_mean = sum(ps) / len(ps)
            agg["q"][q] = {
                "p": p_mean,
                "flag": p_mean >= 0.5,
                "conf": abs(2 * p_mean - 1),
                "flip": (len(set(choices)) > 1) if len(choices) > 1 else None,
            }
        out[pid] = agg
    return out


# ── metrics ──────────────────────────────────────────────────────────────────

def _pair_stats(judge: list[bool], laya: list[bool]) -> dict:
    return {"n": len(judge), "agreement": agreement(judge, laya),
            "kappa": cohen_kappa(judge, laya), "table_judge_laya": table2x2(judge, laya)}


def compute(laya_agg: dict[str, dict], gold: list[dict], candidates: list[dict],
            cos_only_auc: dict[str, float] | None = None) -> tuple[dict, list[dict], list[dict]]:
    gold_by_id = {g["pair_id"]: g for g in gold}
    audited = [g for g in gold if g["pair_id"] in laya_agg]
    missing = [g["pair_id"] for g in gold if g["pair_id"] not in laya_agg]

    report: dict = {
        "n_gold_records": len(gold), "n_laya_pairs": len(laya_agg),
        "n_audited_with_laya": len(audited), "n_gold_missing_laya": len(missing),
        "n_orders_per_pair": Counter(a["n_orders"] for a in laya_agg.values()),
        "meta": next(iter(laya_agg.values()))["meta"] if laya_agg else None,
    }

    def flags_of(g):
        return g["judge"], laya_agg[g["pair_id"]]["q"]

    # agreement on the two label fields, overall / per subset / per tier
    for lab in LABELS:
        block = {"overall": None, "by_subset": {}, "by_tier": {}}
        j = [g["judge"][lab] for g in audited]
        l = [laya_agg[g["pair_id"]]["q"][lab]["flag"] for g in audited]
        block["overall"] = _pair_stats(j, l)
        for key, sel in (("by_subset", lambda g: g["subset"]), ("by_tier", lambda g: g["tier"])):
            groups: dict[str, list[dict]] = defaultdict(list)
            for g in audited:
                groups[sel(g)].append(g)
            for name in sorted(groups):
                gs = groups[name]
                block[key][name] = _pair_stats(
                    [g["judge"][lab] for g in gs],
                    [laya_agg[g["pair_id"]]["q"][lab]["flag"] for g in gs])
        report[lab] = block

    # per-flag agreement for the alignment checks
    report["alignment_flags"] = {}
    for q in ALIGNMENT_FLAGS:
        j = [g["judge"][q] for g in audited]
        l = [laya_agg[g["pair_id"]]["q"][q]["flag"] for g in audited]
        report["alignment_flags"][q] = _pair_stats(j, l)

    # agreement by Laya confidence decile (label-free strata)
    dec: dict[int, list[tuple[bool, bool]]] = defaultdict(list)
    for g in audited:
        a = laya_agg[g["pair_id"]]["q"]["update_conflict"]
        d = min(9, int(a["conf"] * 10))
        dec[d].append((g["judge"]["update_conflict"], a["flag"]))
    report["update_conflict_by_confidence_decile"] = {
        f"{d/10:.1f}-{(d+1)/10:.1f}": {"n": len(v), "agreement": agreement([x for x, _ in v], [y for _, y in v])}
        for d, v in sorted(dec.items())}

    # order sensitivity
    report["order_flip_rate"] = {}
    for q in FLAGS:
        vals = [a["q"][q]["flip"] for a in laya_agg.values() if a["q"][q]["flip"] is not None]
        report["order_flip_rate"][q] = (sum(vals) / len(vals)) if vals else None
    report["order_flip_rate_n"] = len([a for a in laya_agg.values() if a["n_orders"] > 1])

    # label-free sanity checks
    tagged = [a for a in laya_agg.values() if a["parser_tagged_conflict"]]
    report["sanity"] = {
        "parser_tagged_same_relation_rate": {
            "n": len(tagged),
            "rate": (sum(a["q"]["same_relation"]["flag"] for a in tagged) / len(tagged)) if tagged else None,
            "note": "parser-tagged pairs share the relation template by construction"},
        "parser_tagged_same_referent_rate": {
            "n": len(tagged),
            "rate": (sum(a["q"]["same_referent"]["flag"] for a in tagged) / len(tagged)) if tagged else None,
            "note": "parser-tagged pairs share the subject string by construction"},
    }
    cand_by_id = {c["pair_id"]: c for c in candidates}
    diff_subj = [a for a in laya_agg.values()
                 if a["pair_id"] in cand_by_id
                 and cand_by_id[a["pair_id"]]["parser_metadata"].get("both_parse")
                 and not cand_by_id[a["pair_id"]]["parser_metadata"].get("same_subject")
                 and not cand_by_id[a["pair_id"]]["parser_metadata"].get("same_object")
                 and cand_by_id[a["pair_id"]]["cosine_similarity"] < 0.85]
    report["sanity"]["different_subject_and_object_low_cosine_same_referent_false_rate"] = {
        "n": len(diff_subj),
        "rate": (sum(not a["q"]["same_referent"]["flag"] for a in diff_subj) / len(diff_subj)) if diff_subj else None,
        "note": "parser sees different subject and object strings, cosine < 0.85; a descriptive rate, not a truth"}
    if GALILEO in laya_agg:
        g = gold_by_id.get(GALILEO)
        report["sanity"]["galileo_pair"] = {
            "pair_id": GALILEO, "tier": g["tier"] if g else None,
            "judge_update_conflict": g["judge"]["update_conflict"] if g else None,
            "laya": {q: laya_agg[GALILEO]["q"][q] for q in FLAGS}}

    # competence reference on the balanced eval set
    report["eval_set_auroc"] = {}
    for subset in sorted({g["subset"] for g in audited}):
        ev = [g for g in audited if g["in_eval_set"] and g["subset"] == subset]
        pos = [laya_agg[g["pair_id"]]["q"]["update_conflict"]["p"] for g in ev if g["gold_update"]]
        neg = [laya_agg[g["pair_id"]]["q"]["update_conflict"]["p"] for g in ev if not g["gold_update"]]
        report["eval_set_auroc"][subset] = {
            "n_pos": len(pos), "n_neg": len(neg),
            "laya_p_update_auroc": rank_auc(pos, neg),
            "cosine_only_auc": (cos_only_auc or {}).get(subset),
            "note": "labels are judge-derived; a competence reference, not a detector result"}

    # disagreements
    disagreements: list[dict] = []
    for g in audited:
        a = laya_agg[g["pair_id"]]
        if a["q"]["update_conflict"]["flag"] == g["judge"]["update_conflict"]:
            continue
        disagreements.append({
            "pair_id": g["pair_id"], "subset": g["subset"], "tier": g["tier"],
            "fact_a": g["fact_a"], "fact_b": g["fact_b"],
            "cosine_similarity": g["cosine_similarity"],
            "judge": {k: g["judge"][k] for k in FLAGS + ["reason_code", "explanation"]},
            "laya": {q: a["q"][q] for q in FLAGS},
            "laya_says_conflict": a["q"]["update_conflict"]["flag"],
            "priority_tier": TIER_PRIORITY[g["tier"]],
        })
    disagreements.sort(key=lambda d: (d["priority_tier"], -d["laya"]["update_conflict"]["conf"], d["pair_id"]))
    for i, d in enumerate(disagreements):
        d["priority"] = i
    report["disagreements"] = {
        "n": len(disagreements),
        "by_tier": Counter(d["tier"] for d in disagreements),
        "by_direction": Counter("laya_conflict_judge_not" if d["laya_says_conflict"] else "judge_conflict_laya_not"
                                for d in disagreements),
    }

    # tail screen: candidates never judged
    tail = [c for c in candidates if c["pair_id"] not in gold_by_id and c["pair_id"] in laya_agg]
    tail_rows = []
    for c in tail:
        a = laya_agg[c["pair_id"]]
        tail_rows.append({
            "pair_id": c["pair_id"], "subset": c["subset"],
            "fact_a": c["fact_a"], "fact_b": c["fact_b"],
            "cosine_similarity": c["cosine_similarity"],
            "parser": {k: c["parser_metadata"].get(k) for k in
                       ("both_parse", "same_key", "same_subject", "same_relation", "same_object")},
            "p_update": a["q"]["update_conflict"]["p"], "p_strict": a["q"]["strict_conflict"]["p"],
            "laya_update_conflict": a["q"]["update_conflict"]["flag"],
            "laya_strict_conflict": a["q"]["strict_conflict"]["flag"],
        })
    tail_rows.sort(key=lambda r: (-r["p_update"], r["pair_id"]))
    for i, r in enumerate(tail_rows):
        r["rank"] = i
    bulk = [g for g in audited if g["provenance"]["audit_slice"] == "bulk"]
    jp = [g for g in bulk if g["judge"]["update_conflict"]]
    jn = [g for g in bulk if not g["judge"]["update_conflict"]]
    sens = (sum(laya_agg[g["pair_id"]]["q"]["update_conflict"]["flag"] for g in jp) / len(jp)) if jp else None
    fpr = (sum(laya_agg[g["pair_id"]]["q"]["update_conflict"]["flag"] for g in jn) / len(jn)) if jn else None
    n_tail = len(tail_rows)
    laya_pos_tail = sum(r["laya_update_conflict"] for r in tail_rows)
    r_tail = (laya_pos_tail / n_tail) if n_tail else None
    judge_bulk_rate = (len(jp) / len(bulk)) if bulk else None
    report["tail"] = {
        "n_unjudged_candidates": len([c for c in candidates if c["pair_id"] not in gold_by_id]),
        "n_unjudged_with_laya": n_tail,
        "laya_positive_count": laya_pos_tail, "laya_positive_rate": r_tail,
        "bulk_reference": {"n_bulk_audited": len(bulk), "n_judge_positive": len(jp),
                           "judge_positive_rate": judge_bulk_rate,
                           "laya_sensitivity_vs_judge": sens, "laya_fpr_vs_judge": fpr,
                           "wilson_judge_rate": wilson(len(jp), len(bulk)) if bulk else None},
        "expected_conflicts_in_tail_from_judge_bulk_rate":
            (judge_bulk_rate * n_tail) if (judge_bulk_rate is not None and n_tail) else None,
        "rogan_gladen_prevalence": rogan_gladen(r_tail, sens, fpr) if r_tail is not None else None,
        "note": ("the tail is the same shuffled bulk population the judge sampled; the judge's bulk "
                 "rate is the direct estimate, the Laya screen ranks the tail for optional judging"),
    }
    return report, disagreements, tail_rows


# ── rendering ────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_md(rep: dict) -> str:
    L = ["# Laya as a second reader — agreement with the GPT-5-mini judge", ""]
    m = rep.get("meta") or {}
    L.append(f"- Laya {m.get('laya_version')} · checkpoint `{m.get('checkpoint')}` · snapshot "
             f"`{m.get('snapshot_sha')}` · device {m.get('device')} · questions sha256 "
             f"`{(m.get('questions_sha256') or '')[:12]}`")
    L.append(f"- gold records {rep['n_gold_records']:,}; Laya pairs {rep['n_laya_pairs']:,}; audited pairs with "
             f"both readers {rep['n_audited_with_laya']:,}; gold pairs missing Laya {rep['n_gold_missing_laya']:,}")
    L.append(f"- orders per pair: {dict(rep['n_orders_per_pair'])}")
    L.append("")
    L.append("Agreement between two machine readers describes the labels; it corrects nothing. "
             "Disagreements go to the human review (`audit_human_review.py`).")
    L.append("")
    for lab in LABELS:
        b = rep[lab]
        o = b["overall"]
        L.append(f"## {lab}")
        L.append("")
        t = o["table_judge_laya"]
        L.append(f"- overall: n={o['n']:,}, agreement {_f(o['agreement'])}, κ {_f(o['kappa'])}; "
                 f"judge+/Laya+ {t['TT']:,}, judge+/Laya− {t['TF']:,}, judge−/Laya+ {t['FT']:,}, "
                 f"judge−/Laya− {t['FF']:,}")
        L.append("")
        L.append("| group | n | agreement | κ | J+L+ | J+L− | J−L+ | J−L− |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in ("by_subset", "by_tier"):
            for name, s in b[key].items():
                t = s["table_judge_laya"]
                L.append(f"| {name} | {s['n']:,} | {_f(s['agreement'])} | {_f(s['kappa'])} | "
                         f"{t['TT']:,} | {t['TF']:,} | {t['FT']:,} | {t['FF']:,} |")
        L.append("")
    L.append("## Alignment flags")
    L.append("")
    L.append("| flag | n | agreement | κ |")
    L.append("| --- | ---: | ---: | ---: |")
    for q, s in rep["alignment_flags"].items():
        L.append(f"| {q} | {s['n']:,} | {_f(s['agreement'])} | {_f(s['kappa'])} |")
    L.append("")
    L.append("## update_conflict agreement by Laya confidence (|2p−1|)")
    L.append("")
    L.append("| confidence | n | agreement |")
    L.append("| --- | ---: | ---: |")
    for k, s in rep["update_conflict_by_confidence_decile"].items():
        L.append(f"| {k} | {s['n']:,} | {_f(s['agreement'])} |")
    L.append("")
    L.append(f"## Order sensitivity (n={rep['order_flip_rate_n']:,} pairs scored in both orders)")
    L.append("")
    L.append("| question | argmax flip rate |")
    L.append("| --- | ---: |")
    for q, v in rep["order_flip_rate"].items():
        L.append(f"| {q} | {_f(v, 4)} |")
    L.append("")
    L.append("## Label-free sanity checks")
    L.append("")
    for k, s in rep["sanity"].items():
        if k == "galileo_pair":
            lq = s["laya"]
            L.append(f"- Galileo pair `{s['pair_id']}` (tier {s['tier']}, judge update_conflict="
                     f"{s['judge_update_conflict']}): Laya update_conflict={lq['update_conflict']['flag']} "
                     f"(p={_f(lq['update_conflict']['p'])}), same_referent={lq['same_referent']['flag']} "
                     f"(p={_f(lq['same_referent']['p'])})")
        else:
            L.append(f"- {k}: n={s['n']:,}, rate {_f(s['rate'])} — {s['note']}")
    L.append("")
    L.append("## Competence reference on the balanced eval set (judge-derived labels)")
    L.append("")
    L.append("| subset | pos | neg | Laya P(update) AUROC | cosine-only AUC |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for s, v in rep["eval_set_auroc"].items():
        L.append(f"| {s} | {v['n_pos']:,} | {v['n_neg']:,} | {_f(v['laya_p_update_auroc'])} | {_f(v['cosine_only_auc'])} |")
    L.append("")
    d = rep["disagreements"]
    L.append("## Disagreements on update_conflict")
    L.append("")
    L.append(f"- n={d['n']:,}; by tier {dict(d['by_tier'])}; by direction {dict(d['by_direction'])}")
    L.append("- full list with both verdicts and review priority: `audit_laya_disagreements.jsonl`")
    L.append("")
    t = rep["tail"]
    br = t["bulk_reference"]
    L.append("## Unjudged tail")
    L.append("")
    L.append(f"- candidates never judged: {t['n_unjudged_candidates']:,}; scored by Laya: {t['n_unjudged_with_laya']:,}")
    L.append(f"- Laya positives in the tail: {t['laya_positive_count']:,} ({_f(t['laya_positive_rate'], 4)})")
    L.append(f"- judge's rate on the audited bulk: {br['n_judge_positive']:,}/{br['n_bulk_audited']:,} = "
             f"{_f(br['judge_positive_rate'], 4)} (Wilson {tuple(round(x, 4) for x in br['wilson_judge_rate']) if br['wilson_judge_rate'] else '—'}); "
             f"expected conflicts in the tail ≈ {_f(t['expected_conflicts_in_tail_from_judge_bulk_rate'], 1)}")
    L.append(f"- Laya vs judge on the audited bulk: sensitivity {_f(br['laya_sensitivity_vs_judge'])}, "
             f"false-positive rate {_f(br['laya_fpr_vs_judge'], 4)}; Rogan–Gladen corrected tail prevalence "
             f"{_f(t['rogan_gladen_prevalence'], 4)}")
    L.append(f"- ranked tail: `laya_tail_screen.jsonl` — {t['note']}")
    L.append("")
    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────────────

def load_cos_only_auc(path: pathlib.Path = GOLD_SUMMARY) -> dict[str, float]:
    if not path.exists():
        return {}
    s = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for subset, e in (s.get("per_subset") or {}).items():       # builder layout
        if isinstance(e, dict) and isinstance(e.get("eval"), dict) and "cosine_only_auc" in e["eval"]:
            out[subset] = e["eval"]["cosine_only_auc"]
    if not out:                                     # tolerate other layouts
        def walk(o, sub=None):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "cosine_only_auc" and sub:
                        out[sub] = v
                    walk(v, k if k.startswith("sh_") else sub)
        walk(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--laya", default=str(LAYA))
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--gold-summary", default=str(GOLD_SUMMARY))
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    ap.add_argument("--out-disagreements", default=str(OUT_DIS))
    ap.add_argument("--out-tail", default=str(OUT_TAIL))
    args = ap.parse_args(argv)

    laya_agg = aggregate(read_jsonl(pathlib.Path(args.laya)))
    gold = list(read_jsonl(pathlib.Path(args.gold)))
    candidates = list(read_jsonl(pathlib.Path(args.candidates)))
    cos = load_cos_only_auc(pathlib.Path(args.gold_summary))
    rep, dis, tail = compute(laya_agg, gold, candidates, cos)

    pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=dict), encoding="utf-8")
    pathlib.Path(args.out_md).write_text(render_md(rep), encoding="utf-8")
    with open(args.out_disagreements, "wt", encoding="utf-8") as f:
        for d in dis:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(args.out_tail, "wt", encoding="utf-8") as f:
        for r in tail:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"audited with both readers: {rep['n_audited_with_laya']:,}; disagreements: {rep['disagreements']['n']:,}; "
          f"tail rows: {len(tail):,}")
    print(f"wrote {args.out_md}, {args.out_json}, {args.out_disagreements}, {args.out_tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
