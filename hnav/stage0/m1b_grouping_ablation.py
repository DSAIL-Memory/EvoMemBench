#!/usr/bin/env python3
"""M1b — regex-vs-geometry grouping ablation.  [T2]

Runs off T1's embedding cache; no new embedding compute if M1 has been run with
the same model and dtype.

**Why this is not optional.** The benchmark hands H-Nav two shortcuts: facts are
templated, so a regex groups them, and each carries a serial number, so
supersession is explicit. A critic will say any downstream gain comes from that
metadata rather than from geometry. This measures it directly, by treating
"which facts are competing versions of the same fact?" as a retrieval problem
and scoring the geometry grouper against the regex grouper.

  Regex grouper    ``parse()`` -> exact ``(relation, subject)`` match. The oracle.
  Geometry grouper nearest neighbours above a cosine threshold. No parsing.

Candidate generation is top-N nearest neighbours per fact, so recall is bounded
by whether a fact's true partner is within its top-N. That ceiling is computed
and reported rather than left implicit.

    python hnav/stage0/m1b_grouping_ablation.py
    python hnav/stage0/m1b_grouping_ablation.py --subsets sh_6k --smoke-embedder

Leakage: reads ONLY ``context``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import explode_facts, fact_key  # noqa: E402
from hnav.core.embedding import build_embedder  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"

INTERPRETATION = (
    "Geometry that recovers the regex grouping *without parsing* is what licenses "
    "applying H-Nav to CrossEp-Know, where no templates and no serial numbers exist. "
    "High F1 -> the detector is validated. Low F1 -> any downstream gain is "
    "attributable to the metadata, not to geometry, and must be reported as such."
)


def truth_pairs(facts: list[tuple[int, str]]) -> tuple[set, dict, int]:
    """Ground truth from the oracle grouper.

    Returns ``(pairs, key_of_index, n_parsed)`` where ``pairs`` holds ``(i, j)``
    index pairs with ``i < j`` that share a ``(relation, subject)`` key.
    """
    by_key: dict[tuple, list[int]] = defaultdict(list)
    key_of: dict[int, tuple] = {}
    n_parsed = 0
    for i, (_, text) in enumerate(facts):
        key, _ = fact_key(text)
        if key is None:
            continue
        n_parsed += 1
        by_key[key].append(i)
        key_of[i] = key

    pairs = set()
    for members in by_key.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.add((members[a], members[b]))
    return pairs, key_of, n_parsed


def knn_candidates(mat: np.ndarray, top_n: int, block: int = 512
                   ) -> list[tuple[int, int, float]]:
    """Top-N neighbours per row, by cosine, excluding self.

    Blocked so that a 18,332 x 2560 bank never materializes an N x N matrix.
    """
    n = mat.shape[0]
    top_n = min(top_n, max(n - 1, 1))
    out: dict[tuple[int, int], float] = {}
    for s in range(0, n, block):
        e = min(s + block, n)
        sims = mat[s:e] @ mat.T                      # (b, n) cosines
        for r in range(e - s):
            i = s + r
            sims[r, i] = -np.inf                     # never pair a fact with itself
            idx = np.argpartition(sims[r], -top_n)[-top_n:]
            for j in idx:
                j = int(j)
                key = (i, j) if i < j else (j, i)
                out[key] = max(out.get(key, -np.inf), float(sims[r, j]))
    return [(i, j, s) for (i, j), s in out.items()]


def sweep(cands: list[tuple[int, int, float]], truth: set,
          thresholds: np.ndarray) -> list[dict]:
    rows = []
    scores = np.array([s for _, _, s in cands]) if cands else np.zeros(0)
    is_true = np.array([(i, j) in truth for i, j, _ in cands], dtype=bool) \
        if cands else np.zeros(0, dtype=bool)
    n_truth = len(truth)
    for tau in thresholds:
        sel = scores >= tau
        n_pred = int(sel.sum())
        tp = int((sel & is_true).sum())
        precision = tp / n_pred if n_pred else float("nan")
        recall = tp / n_truth if n_truth else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if n_pred and tp and n_truth else 0.0)
        rows.append({"tau": float(tau), "n_predicted": n_pred, "tp": tp,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def equal_coverage_point(cands, truth) -> dict:
    """The operating point where the geometry grouper predicts as many pairs as
    the oracle has — so precision and recall are directly comparable and neither
    grouper is advantaged by simply predicting more."""
    if not cands or not truth:
        return {}
    ordered = sorted(cands, key=lambda x: -x[2])
    n = min(len(truth), len(ordered))
    chosen = ordered[:n]
    tp = sum(1 for i, j, _ in chosen if (i, j) in truth)
    precision = tp / n
    recall = tp / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"tau": float(chosen[-1][2]), "n_predicted": n, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1}


def run_subset(item, name, emb, args) -> dict:
    facts = explode_facts(item["context"])
    if args.max_facts:
        facts = facts[: args.max_facts]
    truth, key_of, n_parsed = truth_pairs(facts)

    print(f"\n── {name}: {len(facts)} facts, {n_parsed} parsed, "
          f"{len(truth)} same-key pairs (oracle)", flush=True)

    mat = emb.encode([t for _, t in facts]).astype(np.float32)
    cands = knn_candidates(mat, args.top_n)

    # Ceiling: a truth pair the candidate generator never proposed can never be
    # recovered at any threshold. Reported, not hidden.
    proposed = {(i, j) for i, j, _ in cands}
    reachable = len(truth & proposed)
    ceiling = reachable / len(truth) if truth else float("nan")

    thresholds = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    curve = sweep(cands, truth, thresholds)
    best = max(curve, key=lambda r: (r["f1"] if np.isfinite(r["f1"]) else -1))
    eq = equal_coverage_point(cands, truth)

    res = {
        "subset": name,
        "n_facts": len(facts), "n_parsed": n_parsed,
        "parse_coverage_pct": round(100 * n_parsed / len(facts), 2) if facts else 0.0,
        "n_truth_pairs": len(truth),
        "top_n_candidates": args.top_n,
        "n_candidate_pairs": len(cands),
        "recall_ceiling_from_knn": ceiling,
        "best_f1": best, "equal_coverage": eq,
        "pr_curve": curve,
        "interpretation": INTERPRETATION,
    }

    print(f"   knn ceiling on recall : {ceiling:.3f}  ({reachable}/{len(truth)} "
          f"truth pairs proposed at top-{args.top_n})")
    print(f"   best F1               : {best['f1']:.3f} at tau={best['tau']:.2f} "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")
    if eq:
        print(f"   equal-coverage point  : F1={eq['f1']:.3f} at tau={eq['tau']:.3f} "
              f"(P={eq['precision']:.3f} R={eq['recall']:.3f})")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--include-mh", action="store_true")
    ap.add_argument("--max-facts", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=10,
                    help="candidate neighbours per fact (bounds achievable recall)")
    ap.add_argument("--tau-min", type=float, default=0.50)
    ap.add_argument("--tau-max", type=float, default=0.99)
    ap.add_argument("--tau-step", type=float, default=0.01)
    ap.add_argument("--smoke-embedder", action="store_true",
                    help="HashEmbedder instead of the real model: logic smoke test "
                         "only, numbers are MEANINGLESS, written to a _SMOKE file")
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(" M1b — regex-vs-geometry grouping ablation")
    if args.smoke_embedder:
        print("   *** SMOKE RUN — HashEmbedder. No result here is interpretable. ***")
        from hnav.core.embedding import HashEmbedder
        emb = HashEmbedder(dim=64)
    else:
        print(f"   model={cfg.embed_model}  dtype={cfg.embed_dtype}  "
              f"(reuses T1's cache)")
        emb = build_embedder(cfg)
    print("=" * 74)

    data = json.load(open(DATA, encoding="utf-8"))
    results = []
    for item in data:
        name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
            .replace("factconsolidation_", "")
        if not args.include_mh and name.startswith("mh_"):
            continue
        if args.subsets and name not in args.subsets:
            continue
        res = run_subset(item, name, emb, args)
        if args.smoke_embedder:
            res["SMOKE_HASH_EMBEDDER"] = True
        results.append(res)

    out = cfg.out_dir / ("m1b_grouping_ablation_SMOKE.json" if args.smoke_embedder
                         else "m1b_grouping_ablation.json")
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print(" M1b SUMMARY")
    print("=" * 74)
    print(f" {'subset':<9} {'facts':>7} {'truth':>7} {'ceil':>6} {'bestF1':>7} "
          f"{'tau*':>6} {'eqF1':>6}")
    for r in results:
        print(f" {r['subset']:<9} {r['n_facts']:>7} {r['n_truth_pairs']:>7} "
              f"{r['recall_ceiling_from_knn']:>6.3f} {r['best_f1']['f1']:>7.3f} "
              f"{r['best_f1']['tau']:>6.2f} "
              f"{(r['equal_coverage'].get('f1', float('nan'))):>6.3f}")
    print(f"\n wrote {out}")
    print("\n INTERPRETATION (to be recorded verbatim in the Stage-0 report):")
    print(" " + INTERPRETATION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
