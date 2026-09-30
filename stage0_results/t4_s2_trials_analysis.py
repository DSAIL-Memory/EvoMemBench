#!/usr/bin/env python3
"""T4/S2 pre-registered trials analysis — see t4_s2_protocol.md Part 1.

Run from the repo root on the box AFTER the driver finishes:

    python stage0_results/t4_s2_trials_analysis.py

Reads MAB/outputs/hnav_s2trial_{off_i,sha_j}/... results JSONs (the discarded
warmup is not read), runs EXACTLY the pre-registered analyses, and writes
stage0_results/t4_s2_trials_summary.json (per-run outputs digest + statistics).

Pre-registered constants (do not edit after registration):
  N_OFF=10, N_SHA=5, TOST margin delta=2.0 points, alpha=0.05,
  permutation reps=10000, seed=20260814.
"""
from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAB = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"

N_OFF, N_SHA = 10, 5
DELTA, ALPHA = 2.0, 0.05
PERM_REPS, SEED = 10_000, 20260814


def load_run(tag: str) -> dict:
    pat = str(MAB / f"outputs/hnav_s2trial_{tag}/*/*.json")
    files = glob.glob(pat)
    if len(files) != 1:
        sys.exit(f"expected exactly 1 results JSON for {tag}, found {len(files)} ({pat})")
    d = json.load(open(files[0], encoding="utf-8"))
    outs = [r["output"] for r in d["data"]]
    sem = float(d["averaged_metrics"]["substring_exact_match"])
    if len(outs) != 100:
        sys.exit(f"{tag}: expected 100 results, got {len(outs)}")
    return {"tag": tag, "outputs": outs, "sem": sem}


def mismatch(a: dict, b: dict) -> float:
    return sum(x != y for x, y in zip(a["outputs"], b["outputs"])) / 100.0


def welch_tost(x: np.ndarray, y: np.ndarray, delta: float) -> dict:
    """Two one-sided Welch t-tests for equivalence of means within ±delta."""
    from scipy import stats
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        eq = abs(mx - my) < delta
        return {"diff": float(mx - my), "se": 0.0, "df": float("nan"),
                "p_lower": 0.0 if eq else 1.0, "p_upper": 0.0 if eq else 1.0,
                "equivalent": bool(eq), "note": "zero variance; direct comparison"}
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t_lower = ((mx - my) + delta) / se   # H0: diff <= -delta
    t_upper = ((mx - my) - delta) / se   # H0: diff >= +delta
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    return {"diff": float(mx - my), "se": float(se), "df": float(df),
            "p_lower": float(p_lower), "p_upper": float(p_upper),
            "equivalent": bool(p_lower < ALPHA and p_upper < ALPHA)}


def main() -> int:
    offs = [load_run(f"off_{i}") for i in range(1, N_OFF + 1)]
    shas = [load_run(f"sha_{j}") for j in range(1, N_SHA + 1)]

    # 1. Noise floor
    within_off = [mismatch(a, b) for a, b in itertools.combinations(offs, 2)]
    within_sha = [mismatch(a, b) for a, b in itertools.combinations(shas, 2)]
    cross = [mismatch(a, b) for a in offs for b in shas]
    off_sems = np.array([r["sem"] for r in offs])
    sha_sems = np.array([r["sem"] for r in shas])

    # 2. TOST on substring_exact_match
    tost = welch_tost(off_sems, sha_sems, DELTA)

    # 3. Permutation test on Δ = mean(cross) − mean(within)
    runs = offs + shas
    labels = np.array([0] * N_OFF + [1] * N_SHA)
    mm = np.zeros((15, 15))
    for i, j in itertools.combinations(range(15), 2):
        mm[i, j] = mm[j, i] = mismatch(runs[i], runs[j])

    def stat(lbl: np.ndarray) -> float:
        c, w = [], []
        for i, j in itertools.combinations(range(15), 2):
            (c if lbl[i] != lbl[j] else w).append(mm[i, j])
        return float(np.mean(c) - np.mean(w))

    observed = stat(labels)
    rng = np.random.default_rng(SEED)
    perm = np.array([stat(rng.permutation(labels)) for _ in range(PERM_REPS)])
    p_perm = float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (PERM_REPS + 1))

    supports = bool(tost["equivalent"] and p_perm >= ALPHA
                    and abs(observed) <= float(np.mean(within_off + within_sha)))
    against = bool(p_perm < ALPHA
                   or (abs(tost["diff"]) > DELTA and not tost["equivalent"]))

    out = {
        "protocol": "stage0_results/t4_s2_protocol.md Part 1 (pre-registered)",
        "n_off": N_OFF, "n_shadow": N_SHA, "delta_points": DELTA, "alpha": ALPHA,
        "off_sem_per_run": off_sems.tolist(), "shadow_sem_per_run": sha_sems.tolist(),
        "off_sem_mean": float(off_sems.mean()), "off_sem_sd": float(off_sems.std(ddof=1)),
        "shadow_sem_mean": float(sha_sems.mean()), "shadow_sem_sd": float(sha_sems.std(ddof=1)),
        "noise_floor": {
            "within_off_mismatch_mean": float(np.mean(within_off)),
            "within_off_mismatch_max": float(np.max(within_off)),
            "within_shadow_mismatch_mean": float(np.mean(within_sha)),
            "cross_mismatch_mean": float(np.mean(cross)),
            "n_pairs": {"within_off": len(within_off), "within_shadow": len(within_sha),
                        "cross": len(cross)},
        },
        "tost": tost,
        "permutation": {"observed_delta": observed, "p_two_sided": p_perm,
                        "reps": PERM_REPS, "seed": SEED},
        "decision_rule_result": {
            "supports_substrate_noise_attribution": supports,
            "evidence_against_neutrality": against,
            "note": "SUPPORTING EVIDENCE ONLY — the definitive verdict is the "
                    ":8002 deterministic rerun (protocol Part 2).",
        },
    }
    dest = REPO / "stage0_results/t4_s2_trials_summary.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:3000])
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
