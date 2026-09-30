#!/usr/bin/env python3
"""Re-derive every number that the revised manuscript adds on top of the
committed reports.  Stdlib only; reads committed artifacts and the benchmark
data file, writes nothing.

    python3 paper/derive_revision_numbers.py

Offline analysis: it reads benchmark answers (through the question strata),
which is allowed here and nowhere under hnav/core or hnav/adapters.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hnav.labeling.question_strata import (  # noqa: E402  (offline tier)
    SINGLE_HOP_PREFIX, classify_questions, error_class, key_members)

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
RESULTS = REPO / "pipelines/hnav_geo/results"
GOLD = REPO / "stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz"
SUBSETS = ("sh_6k", "sh_32k", "sh_64k")
MODELS = {  # paper order; the VOID gemma-3 run (fp8 KV cache) is deliberately absent
    "gemma-3-4b-it": "google_gemma-3-4b-it_2026-08-31",
    "gemma-4-E2B-it": "google_gemma-4-E2B-it_2026-08-31",
    "Phi-4-mini-instruct": "microsoft_Phi-4-mini-instruct_2026-08-30",
    "Qwen3-4B-Instruct-2507": "Qwen_Qwen3-4B-Instruct-2507_2026-08-29",
    "Qwen3.5-9B": "Qwen_Qwen3.5-9B_2026-08-31",
}
FACT_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$", re.M)


def load_items() -> dict:
    items = {}
    for it in json.loads(DATA.read_text(encoding="utf-8")):
        full = it["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        if full.startswith(SINGLE_HOP_PREFIX):
            items[full.replace("factconsolidation_", "")] = it
    return items


def load(model: str, subset: str) -> dict:
    path = RESULTS / MODELS[model] / f"detector_gap_{subset}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"][0]


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    def cdf(x, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def bisect(f):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return lo
    lower = 0.0 if k == 0 else bisect(lambda p: 1 - cdf(k - 1, p) < a / 2)
    upper = bisect(lambda p: cdf(k, p) > a / 2)
    return lower, upper


def _parsed(rec: dict):
    """(fact_a_parsed, fact_b_parsed) or None; the parser field is a dict or its repr."""
    import ast
    p = rec["parser"] if isinstance(rec["parser"], dict) else ast.literal_eval(rec["parser"])
    a, b = p.get("fact_a_parsed"), p.get("fact_b_parsed")
    return (a, b) if a and b else None


def correct(q: dict, arm: str) -> bool:
    return bool(q["arms"][arm]["correct"])


def main() -> int:
    items = load_items()
    texts = {s: {int(n): t for n, t in FACT_RE.findall(items[s]["context"])} for s in SUBSETS}

    print("## Store overlap (Section 3.2, 4.3, 5.1)")
    sets = {s: set(texts[s].values()) for s in SUBSETS}
    calib = sets["sh_6k"] | sets["sh_32k"]
    print(f"facts: {[len(texts[s]) for s in SUBSETS]}; sh_6k within sh_32k: "
          f"{len(sets['sh_6k'] & sets['sh_32k'])}/{len(sets['sh_6k'])}; "
          f"sh_32k in sh_64k: {len(sets['sh_32k'] & sets['sh_64k'])}; "
          f"calibration distinct: {len(calib)} (ABTT fit rows {len(texts['sh_6k']) + len(texts['sh_32k'])}); "
          f"sh_64k facts in calibration: {len(sets['sh_64k'] & calib)}")

    print("\n## Table 2: native error composition at sh_6k and the clean-page control")
    recs6 = {r["index"]: r for r in classify_questions(items["sh_6k"])}
    mem6 = key_members(items["sh_6k"])
    tot_wrong = tot_stale = 0
    for m in MODELS:
        r = load(m, "sh_6k")
        err, clean_nat, clean_del = Counter(), Counter(), Counter()
        n_ok = 0
        for q in r["per_question"]:
            if q["stratum"] != "conflicted":
                continue
            if correct(q, "native"):
                n_ok += 1
            else:
                err[error_class(q["arms"]["native"]["output"], recs6[q["index"]])] += 1
            rows = sorted(mem6[tuple(q["key"])])
            latest, cut = rows[-1][0], set(q["plan"]["suppress_serials"])
            pool = {int(x.split(":")[1]) for x in q["pool"]}
            stale_left = [s for s, _, o in rows if s != latest and s not in cut and o != rows[-1][2]]
            if stale_left or latest in cut or q["target_serial"] != latest or latest not in pool:
                continue
            for arm, cnt in (("native", clean_nat), ("detector_suppress", clean_del)):
                cnt["correct" if correct(q, arm) else
                    error_class(q["arms"][arm]["output"], recs6[q["index"]])] += 1
        wrong = sum(err.values())
        tot_wrong, tot_stale = tot_wrong + wrong, tot_stale + err["stale_value"]
        print(f"{m:24s} correct {n_ok:2d} wrong {wrong:2d} stale {err['stale_value']:2d} "
              f"other {err['off_list']:2d} empty {err['empty']} | clean n={sum(clean_nat.values())} "
              f"native stale {clean_nat['stale_value']} -> deleted stale {clean_del['stale_value']}, "
              f"deleted correct {clean_del['correct']}")
    print(f"superseded share of conflicted errors: {tot_stale}/{tot_wrong} = {tot_stale / tot_wrong:.3f}")

    print("\n## Clean-page superseded answers after deletion at sh_32k")
    recs32 = {r["index"]: r for r in classify_questions(items["sh_32k"])}
    mem32 = key_members(items["sh_32k"])
    for m in MODELS:
        n = stale = 0
        for q in load(m, "sh_32k")["per_question"]:
            if q["stratum"] != "conflicted":
                continue
            rows = sorted(mem32[tuple(q["key"])])
            latest, cut = rows[-1][0], set(q["plan"]["suppress_serials"])
            if any(s != latest and s not in cut and o != rows[-1][2] for s, _, o in rows) \
                    or latest in cut or q["target_serial"] != latest:
                continue
            n += 1
            if not correct(q, "detector_suppress") and error_class(
                    q["arms"]["detector_suppress"]["output"], recs32[q["index"]]) == "stale_value":
                stale += 1
        print(f"{m:24s} clean n={n} superseded after deletion {stale}")

    print("\n## McNemar, Holm (Section 6.3) and unique-stratum flips (Appendix B)")
    cells = []
    for s in SUBSETS:
        for m in MODELS:
            b = c = ub = uc = 0
            for q in load(m, s)["per_question"]:
                n_, g_ = correct(q, "native"), correct(q, "detector_suppress")
                b += n_ and not g_
                c += g_ and not n_
                if q["stratum"] == "unique":
                    ub += n_ and not g_
                    uc += g_ and not n_
            cells.append((mcnemar_exact(b, c), s, m, b, c, ub, uc))
    rejecting = True                      # Holm is step-down: stop at the first failure
    for i, (p, s, m, b, c, ub, uc) in enumerate(sorted(cells)):
        rejecting = rejecting and p <= 0.05 / (len(cells) - i)
        print(f"{s:7s} {m:24s} b/c {b:2d}/{c:2d} p={p:.2e} Holm {'pass' if rejecting else 'fail'} "
              f"| unique b/c {ub}/{uc}")

    print("\n## sh_64k retrieval misses and novel-fact slice (Section 6.3)")
    for m in MODELS:
        conf = [q for q in load(m, "sh_64k")["per_question"] if q["stratum"] == "conflicted"]
        miss = [q for q in conf if f"fact:{q['target_serial']}" not in q["pool"]]
        errs = sum(not correct(q, "detector_suppress") for q in conf)
        novel = [q for q in conf if texts["sh_64k"][q["target_serial"]] not in calib]
        seen = [q for q in conf if texts["sh_64k"][q["target_serial"]] in calib]

        def acc(qs, arm):
            return sum(correct(q, arm) for q in qs)
        print(f"{m:24s} misses {len(miss)} (native ok {acc(miss, 'native')}, GEO ok "
              f"{acc(miss, 'detector_suppress')}) = {len(miss)}/{errs} GEO errors | "
              f"seen n={len(seen)} {acc(seen, 'native')}->{acc(seen, 'detector_suppress')} | "
              f"novel n={len(novel)} {acc(novel, 'native')}->{acc(novel, 'detector_suppress')}")

    print("\n## Expected answer deleted: all cases (Section 7.3)")
    for s in ("sh_32k", "sh_64k"):
        mem = key_members(items[s])
        for q in load("Qwen3-4B-Instruct-2507", s)["per_question"]:
            if q["target_serial"] in set(q["plan"]["suppress_serials"]):
                rows = sorted(mem[tuple(q["key"])])
                print(f"{s} q{q['index']}: expected-answer serial {q['target_serial']}, "
                      f"latest serial of key {rows[-1][0]} -> {[(a, o) for a, _, o in rows]}")

    print("\n## Placement beats deletion: per-question diagnosis (Section 6.5)")
    for m, s in (("gemma-3-4b-it", "sh_32k"), ("Phi-4-mini-instruct", "sh_32k")):
        mem = key_members(items[s])
        recs = {r["index"]: r for r in classify_questions(items[s])}
        tally = Counter()
        for q in load(m, s)["per_question"]:
            if q["stratum"] != "conflicted":
                continue
            if not (correct(q, "detector_demote_late") and not correct(q, "detector_suppress")):
                continue
            rows = sorted(mem[tuple(q["key"])])
            latest, cut = rows[-1][0], set(q["plan"]["suppress_serials"])
            clean = not any(x != latest and x not in cut and o != rows[-1][2] for x, _, o in rows)
            tally[("clean" if clean else "stale left",
                   error_class(q["arms"]["detector_suppress"]["output"], recs[q["index"]]))] += 1
        print(f"{m} {s}: end right, delete wrong -> {dict(tally)}")

    print("\n## Governed small model vs native Qwen3.5-9B (Section 7.1)")
    for small in ("Qwen3-4B-Instruct-2507", "Phi-4-mini-instruct"):
        for s in SUBSETS:
            a = {q["index"]: correct(q, "detector_suppress") for q in load(small, s)["per_question"]}
            n = {q["index"]: correct(q, "native") for q in load("Qwen3.5-9B", s)["per_question"]}
            b = sum(n[i] and not a[i] for i in a)
            c = sum(a[i] and not n[i] for i in a)
            print(f"{small:24s} {s:7s} {sum(a.values())} vs {sum(n.values())} "
                  f"b/c {b}/{c} p={mcnemar_exact(b, c):.2g}")

    print("\n## Recall ceiling of the 0.94 candidate threshold (Section 4.5)")
    recs = [json.loads(line) for line in gzip.open(GOLD, "rt", encoding="utf-8")]
    for s in SUBSETS:
        gold = [r for r in recs if r["subset"] == s and r["gold_update"] in (True, "True")]
        below = sum(float(r["cosine_similarity"]) < 0.94 for r in gold)
        print(f"{s}: {below}/{len(gold)} = {below / len(gold):.3f} of version pairs below raw cosine 0.94")

    print("\n## Probe negatives (Section 4.4)")
    hard = [r for r in recs if r["split"] == "calibration" and r["tier"] == "negative"
            and _parsed(r) and _parsed(r)[0]["relation"] == _parsed(r)[1]["relation"]
            and _parsed(r)[0]["subject"] != _parsed(r)[1]["subject"]]
    same_obj = sum(_parsed(r)[0].get("object") == _parsed(r)[1].get("object") for r in hard)
    print(f"calibration same-relation different-subject negatives: {len(hard)}; sharing the object: {same_obj}")

    print("\n## Tail and harm intervals (Sections 6.1, 6.2)")
    print(f"TPR seen 752/1045 Wilson {wilson(752, 1045)}; unseen 306/636 Wilson {wilson(306, 636)}")
    print(f"harmful 8/532 Clopper-Pearson {clopper_pearson(8, 532)}; "
          f"0/2157 upper {clopper_pearson(0, 2157)[1]:.5f}, rule of three {3 / 2157:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
