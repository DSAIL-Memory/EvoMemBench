#!/usr/bin/env python3
"""Stage-1 A/A analysis — the noise floor of the FROZEN substrate.  [T11]

Run from the repo root on the box AFTER stage1_aa_driver.sh finishes:

    python stage0_results/stage1_aa_analysis.py

Reads MAB/outputs/hnav_stage1aa_off_{1..5} (warmup ignored), computes the
run-level accuracy spread and pairwise output-mismatch rates, and writes
stage0_results/stage1_aa_summary.json. No decision rule lives here — the
numbers feed the pre-registration (stage0_results/stage1_preregistration.md),
which is committed BEFORE the campaign and states how they are used.
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
N_RUNS = 5
SUBSET = "factconsolidation_sh_32k"


def load_run(tag: str) -> dict:
    pat = str(MAB / f"outputs/hnav_stage1aa_{tag}/*/*.json")
    files = glob.glob(pat)
    if len(files) != 1:
        sys.exit(f"expected exactly 1 results JSON for {tag}, found {len(files)} ({pat})")
    d = json.load(open(files[0], encoding="utf-8"))
    outs = [r["output"] for r in d["data"]]
    sem = float(d["averaged_metrics"]["substring_exact_match"])
    if len(outs) != 100:
        sys.exit(f"{tag}: expected 100 results, got {len(outs)}")
    return {"tag": tag, "outputs": outs, "sem": sem}


def main() -> int:
    runs = [load_run(f"off_{i}") for i in range(1, N_RUNS + 1)]
    sems = np.array([r["sem"] for r in runs])
    mism = [sum(x != y for x, y in zip(a["outputs"], b["outputs"])) / 100.0
            for a, b in itertools.combinations(runs, 2)]

    out = {
        "protocol": "stage0_results/stage1_preregistration.md (A/A section)",
        "subset": SUBSET,
        "substrate": "frozen Stage-1: :8003 chat (serve_stage1_chat.sh) + "
                     ":8001 bf16 embed (serve_stage1_embed.sh)",
        "n_runs": N_RUNS,
        "sem_per_run": sems.tolist(),
        "sem_mean": float(sems.mean()),
        "sem_sd": float(sems.std(ddof=1)),
        "sem_min": float(sems.min()),
        "sem_max": float(sems.max()),
        "pairwise_mismatch": {
            "n_pairs": len(mism),
            "mean": float(np.mean(mism)),
            "max": float(np.max(mism)),
            "values": mism,
        },
        "reference": {
            "t4_s2_off_sd_sh6k_:8000": 1.523883926754995,
            "note": "the prior noise floor was measured on the user's :8000 "
                    "(prefix caching ON); the frozen substrate disables it and "
                    "serializes requests, so a smaller SD is expected but must "
                    "be MEASURED, not assumed.",
        },
    }
    dst = REPO / "stage0_results/stage1_aa_summary.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
