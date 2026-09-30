#!/usr/bin/env python3
"""Build the gold conflict dataset for the geometry filter.  [offline]

Joins the committed audit artifacts in ``stage0_results/conflict_pairs/`` —
the cos>=0.80 candidate set, the GPT-5-mini verdicts, the parser-tagged pair
export and the discovered-conflicts export — into one tiered, dual-labeled
dataset, plus a balanced cosine-matched eval set for the geometry filter.

Tier rules (user decisions 2026-08-26: dual-label with update as default;
rule-based triage, no new LLM calls; balanced eval set):

  core                  parser-tagged AND judge update_conflict
  update_only_fork      parser-tagged, judge says values compatible but the
                        slot alignment HELD (same_referent AND same_relation
                        AND context_overlap). This is the documented
                        definitional fork: under the benchmark convention the
                        later serial supersedes the same key, so the pair is
                        an update conflict by construction; under strict
                        semantics the values can coexist. gold_update=True,
                        gold_strict=False, disputed_by_judge=True.
  rejected              parser-tagged and the judge rejected the slot
                        alignment itself (different referent / relation /
                        context) — the genuine parser-FP candidates. Not gold.
  discovered_unverified untagged, judge update_conflict — single-model labels,
                        quarantined until adjudicated. Not gold, and NEVER in
                        the negative pool.
  negative              audited, untagged, judge negative — the verified
                        non-conflict pool the filter operates over.

Triage keys on the judge's recorded per-pair FLAGS, not on ``reason_code``:
the reason taxonomy is noisy (e.g. ``relation_paraphrase`` used for value
paraphrases like "association football" vs "football" on pairs whose relation
template is identical by construction), while the four alignment flags encode
the fork directly. AUDIT_SUMMARY.md already notes the fork is "re-derivable
from the recorded per-pair flags".

Balanced eval set: per subset, every gold positive plus an equal number of
``negative``-tier pairs sampled cosine-matched (0.01 bins, nearest-non-empty-
bin fallback recorded per record, seed 20260824). Cosine matching is
load-bearing — positives sit near cos 0.96 while the negative pool skews
lower, and an unmatched sample would let a geometry filter score by cosine
alone instead of conflict structure.

Every fact string is re-parsed with the validated ``conflict_analysis.parse``
and checked against the committed ``parser_metadata``; a mismatch is a build
error, not a silent overwrite.

Unaudited candidates (~32.5k, budget-stopped tail) carry no verdict and are
excluded entirely. The selection frame is cos>=0.80 with the campaign
embeddings (Qwen3-Embedding-4B float32 L8192); the dataset says nothing about
conflicts below that similarity.

Usage:
    python hnav/labeling/build_gold_conflict_dataset.py

Writes ``gold_conflict_dataset.jsonl.gz``, ``gold_conflict_dataset.summary.json``
and ``gold_conflict_dataset.summary.md`` next to its inputs. Stdlib only.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.audit_runner import _slice_of  # noqa: E402
from hnav.labeling.conflict_analysis import parse  # noqa: E402

DIR = REPO / "stage0_results" / "conflict_pairs"
CANDIDATES_GZ = DIR / "audit_candidates_cos080.jsonl.gz"
RESULTS = DIR / "audit_results_gpt5mini.jsonl"
RESULTS_GZ = DIR / "audit_results_gpt5mini.jsonl.gz"
TAGGED_PAIRS = DIR / "conflict_pairs.json"
DISCOVERED = DIR / "llm_discovered_conflicts.json"

OUT_JSONL = DIR / "gold_conflict_dataset.jsonl.gz"
OUT_SUMMARY = DIR / "gold_conflict_dataset.summary.json"
OUT_MD = DIR / "gold_conflict_dataset.summary.md"

SEED = 20260824          # the audit campaign's seed convention
BIN = 0.01               # cosine bin width for matching
SUBSETS = ("sh_6k", "sh_32k", "sh_64k")
SPLIT = {"sh_6k": "calibration", "sh_32k": "calibration", "sh_64k": "confirmatory"}
SLICE_NAME = {1: "tagged", 2: "same_key_or_unparsed", 3: "structural", 4: "bulk"}

# Every count below was derived from the committed audit files in-session
# (2026-08-26) and is asserted at build time: a drifting input must fail
# loudly, never re-baseline silently.
EXPECTED = {
    "core": 2388, "core_strict": 1966, "update_only_fork": 282,
    "rejected": 12, "discovered_unverified": 105, "negative": 51782,
    "gold_by_subset": {"sh_6k": 160, "sh_32k": 829, "sh_64k": 1681},
}


def _read_jsonl(path: pathlib.Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tier_of(tagged: bool, verdict: dict) -> str:
    if tagged and verdict["update_conflict"]:
        return "core"
    if tagged:
        aligned = (verdict["same_referent"] and verdict["same_relation"]
                   and verdict["context_overlap"])
        return "update_only_fork" if aligned else "rejected"
    return "discovered_unverified" if verdict["update_conflict"] else "negative"


def check_parse(fact: str, stored: dict | None) -> None:
    """The committed parser_metadata must equal a fresh conflict_analysis.parse."""
    p = parse(fact)
    if p is None:
        if stored is not None:
            raise SystemExit(f"parse mismatch: parser now fails on {fact!r} "
                             f"but metadata has {stored!r}")
        return
    rel, subj, obj = p
    if stored is None:
        raise SystemExit(f"parse mismatch: parser now parses {fact!r} "
                         f"but metadata recorded a parse failure")
    got = {"relation": rel, "subject": subj, "object": obj}
    if got != stored:
        raise SystemExit(f"parse mismatch on {fact!r}: fresh {got} != stored {stored}")


def cos_bin(c: float) -> int:
    return min(int(c / BIN), int(1.0 / BIN) - 1)   # cos 1.0 joins the top bin


def rank_auc(pos: list[float], neg: list[float]) -> float:
    """P(random positive > random negative), ties counted half. O(n log n)."""
    both = sorted((v, 1) for v in pos) + sorted((v, 0) for v in neg)
    both.sort(key=lambda t: t[0])
    wins = i = 0.0
    seen_neg = tie_neg = 0
    while i < len(both):
        j = int(i)
        k = j
        while k < len(both) and both[k][0] == both[j][0]:
            k += 1
        tie_neg = sum(1 for t in both[j:k] if t[1] == 0)
        tie_pos = (k - j) - tie_neg
        wins += tie_pos * (seen_neg + tie_neg / 2.0)
        seen_neg += tie_neg
        i = k
    return wins / (len(pos) * len(neg))


def main() -> int:
    candidates = {r["pair_id"]: r for r in _read_jsonl(CANDIDATES_GZ)}
    results_path = RESULTS if RESULTS.exists() else RESULTS_GZ
    results = {r["pair_id"]: r for r in _read_jsonl(results_path)
               if r.get("status") == "ok"}
    tagged_export = json.loads(TAGGED_PAIRS.read_text(encoding="utf-8"))
    discovered = json.loads(DISCOVERED.read_text(encoding="utf-8"))
    channel_by_id = {d["pair_id"]: d["parser_miss_channel"]
                     for d in discovered["discovered_conflicts"]}

    tagged_serials = {}   # pair_id -> (serial_earlier, serial_later)
    for s, blob in tagged_export["subsets"].items():
        for p in blob["pairs"]:
            pid = f"{s}:{p['serial_earlier']}-{p['serial_later']}"
            tagged_serials[pid] = (p["serial_earlier"], p["serial_later"])

    records = []
    tiers = Counter()
    by_subset: dict[str, Counter] = defaultdict(Counter)
    for pid, res in sorted(results.items()):
        cand = candidates.get(pid)
        if cand is None:
            raise SystemExit(f"result {pid} has no candidate record")
        v = res["verdict"]
        tagged = bool(cand["parser_tagged_conflict"])
        tier = tiers_key = tier_of(tagged, v)
        m = cand["parser_metadata"]
        check_parse(cand["fact_a"], m["fact_a_parsed"])
        check_parse(cand["fact_b"], m["fact_b_parsed"])
        if tagged and pid not in tagged_serials:
            raise SystemExit(f"tagged pair {pid} missing from conflict_pairs.json")
        if tier == "discovered_unverified" and pid not in channel_by_id:
            raise SystemExit(f"discovery {pid} missing from llm_discovered_conflicts.json")
        rec = {
            "pair_id": pid,
            "subset": cand["subset"],
            "split": SPLIT[cand["subset"]],
            "tier": tier,
            "in_eval_set": False,          # filled below
            "gold_update": tier in ("core", "update_only_fork"),
            "gold_strict": tier == "core" and bool(v["strict_conflict"]),
            "disputed_by_judge": tier == "update_only_fork",
            "fact_a": cand["fact_a"], "fact_b": cand["fact_b"],
            "fact_a_id": cand["fact_a_id"], "fact_b_id": cand["fact_b_id"],
            "cosine_similarity": cand["cosine_similarity"],
            "parser": m,
            "judge": v,
            "provenance": {
                "audit_slice": SLICE_NAME[min(_slice_of(cand), 4)],
                "parser_miss_channel": channel_by_id.get(pid),
                "cosine_bin_fallback": None,   # set on matched negatives
                "judge_model": res["model"],
            },
        }
        if tagged:
            rec["serial_earlier"], rec["serial_later"] = tagged_serials[pid]
        records.append(rec)
        tiers[tiers_key] += 1
        by_subset[cand["subset"]][tiers_key] += 1

    n_core_strict = sum(1 for r in records if r["gold_strict"])
    got = {"core": tiers["core"], "core_strict": n_core_strict,
           "update_only_fork": tiers["update_only_fork"],
           "rejected": tiers["rejected"],
           "discovered_unverified": tiers["discovered_unverified"],
           "negative": tiers["negative"],
           "gold_by_subset": {s: by_subset[s]["core"] + by_subset[s]["update_only_fork"]
                              for s in SUBSETS}}
    if got != EXPECTED:
        raise SystemExit(f"tier counts drifted from the committed audit:\n"
                         f"  expected {EXPECTED}\n  got      {got}")

    # ── balanced, cosine-matched eval set ────────────────────────────────────
    rng = random.Random(SEED)
    by_id = {r["pair_id"]: r for r in records}
    match_diag = {}
    for s in SUBSETS:
        positives = [r for r in records if r["subset"] == s and r["gold_update"]]
        pool: dict[int, list[str]] = defaultdict(list)
        for r in records:
            if r["subset"] == s and r["tier"] == "negative":
                pool[cos_bin(r["cosine_similarity"])].append(r["pair_id"])
        for b in pool:
            pool[b].sort()                 # determinism before sampling
        want = Counter(cos_bin(r["cosine_similarity"]) for r in positives)
        fallbacks = Counter()

        def take(bb: int, k: int, dist: int) -> int:
            taken = 0
            while taken < k and pool.get(bb):
                i = rng.randrange(len(pool[bb]))
                pid = pool[bb].pop(i)
                by_id[pid]["in_eval_set"] = True
                by_id[pid]["provenance"]["cosine_bin_fallback"] = dist or None
                fallbacks[dist] += 1
                taken += 1
            return taken

        # phase 1: exact-bin matches everywhere, so no bin's shortfall can
        # steal a negative that another bin would have matched exactly
        short = {}
        for b in sorted(want):
            short[b] = want[b] - take(b, want[b], 0)
        # phase 2: nearest-bin fallback for the remainder, high bins first
        # (their pool is the thinnest)
        for b in sorted((b for b, n in short.items() if n), reverse=True):
            need = short[b]
            for dist in range(1, int(1.0 / BIN)):
                need -= take(b - dist, need, dist) + take(b + dist, need, dist)
                if not need:
                    break
            if need:
                raise SystemExit(f"{s}: negative pool exhausted at bin {b}")
        for r in positives:
            r["in_eval_set"] = True
        eval_neg = [r for r in records if r["subset"] == s
                    and r["tier"] == "negative" and r["in_eval_set"]]
        pcs = sorted(r["cosine_similarity"] for r in positives)
        ncs = sorted(r["cosine_similarity"] for r in eval_neg)
        pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]
        match_diag[s] = {
            "n_pos": len(positives), "n_neg": len(eval_neg),
            "exact_bin": fallbacks[0],
            "fallback": {str(d): c for d, c in sorted(fallbacks.items()) if d},
            "pos_cos_p10_p50_p90": [round(pct(pcs, q), 4) for q in (.1, .5, .9)],
            "neg_cos_p10_p50_p90": [round(pct(ncs, q), 4) for q in (.1, .5, .9)],
            # the baseline a geometry filter must beat on this eval set: the
            # negative pool is thin above cos~0.95, so matching cannot fully
            # remove the cosine signal — report it instead of hiding it
            "cosine_only_auc": round(rank_auc(pcs, ncs), 4),
        }

    # ── outputs ──────────────────────────────────────────────────────────────
    with gzip.open(OUT_JSONL, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "what": "gold conflict dataset for the geometry filter — tiered, "
                "dual-labeled, with a balanced cosine-matched eval set",
        "producer": "hnav/labeling/build_gold_conflict_dataset.py",
        "conventions": {
            "gold_update": "benchmark memory-store convention: later serial "
                           "supersedes the same (relation, subject) key; the "
                           "DEFAULT gold label (user decision 2026-08-26)",
            "gold_strict": "values logically cannot coexist (judge strict_conflict); "
                           "gold_strict implies gold_update",
            "triage": "recorded judge flags, not reason codes: fork = slot "
                      "alignment held but values judged compatible",
            "eval_set": f"per subset, all gold positives + equal negatives "
                        f"cosine-matched in {BIN} bins, nearest-bin fallback, "
                        f"seed {SEED}",
            "splits": SPLIT,
        },
        "caveats": [
            "selection frame is cos>=0.80 under the campaign embeddings "
            "(Qwen3-Embedding-4B float32 L8192); nothing below that is covered",
            "judge is a single model (openai/gpt-5-mini, minimal reasoning); "
            "discovered_unverified stays quarantined until adjudicated",
            "the shuffled bulk tail was 34.5% audited (budget stop) — "
            "prevalence claims from the negative pool need that weighting",
            "unaudited candidates (32,533) are excluded entirely",
        ],
        "tiers_total": {k: tiers[k] for k in sorted(tiers)},
        "core_strict": n_core_strict,
        "per_subset": {s: {"tiers": dict(by_subset[s]), "eval": match_diag[s]}
                       for s in SUBSETS},
        "n_records": len(records),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                           encoding="utf-8")

    md = ["# Gold conflict dataset — summary", "",
          f"- records: {len(records):,} (audited pairs only); eval set: "
          f"{sum(1 for r in records if r['in_eval_set']):,} "
          f"(balanced 1:1 per subset, cosine-matched, seed {SEED})",
          f"- gold positives: {tiers['core'] + tiers['update_only_fork']:,} "
          f"(core {tiers['core']:,}, of which strict {n_core_strict:,}; "
          f"update-only fork {tiers['update_only_fork']:,}) — "
          f"rejected {tiers['rejected']}, discovered-unverified "
          f"{tiers['discovered_unverified']} (quarantined)", "",
          "| subset | split | core | fork | rejected | discovered | negatives "
          "| eval pos | eval neg | exact-bin | pos cos p50 | neg cos p50 | cos-only AUC |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in SUBSETS:
        t, e = by_subset[s], match_diag[s]
        md.append(f"| {s} | {SPLIT[s]} | {t['core']} | {t['update_only_fork']} "
                  f"| {t.get('rejected', 0)} | {t.get('discovered_unverified', 0)} "
                  f"| {t['negative']:,} | {e['n_pos']} | {e['n_neg']} "
                  f"| {e['exact_bin']} | {e['pos_cos_p10_p50_p90'][1]} "
                  f"| {e['neg_cos_p10_p50_p90'][1]} | {e['cosine_only_auc']} |")
    md += ["", "Conventions and caveats: see `gold_conflict_dataset.summary.json`.",
           "Dual labels: `gold_update` (benchmark update convention, default) and "
           "`gold_strict` (logical incompatibility). The `update_only_fork` tier is "
           "gold under update semantics only and carries `disputed_by_judge`."]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"records={len(records):,}  tiers={dict(tiers)}  strict={n_core_strict}")
    for s in SUBSETS:
        e = match_diag[s]
        print(f"{s:7s} eval {e['n_pos']}+{e['n_neg']}  exact-bin {e['exact_bin']}"
              f"  fallback {e['fallback'] or '{}'}"
              f"  pos p50 {e['pos_cos_p10_p50_p90'][1]}"
              f"  neg p50 {e['neg_cos_p10_p50_p90'][1]}"
              f"  cos-only AUC {e['cosine_only_auc']}")
    print(f"wrote {OUT_JSONL}\nwrote {OUT_SUMMARY}\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
