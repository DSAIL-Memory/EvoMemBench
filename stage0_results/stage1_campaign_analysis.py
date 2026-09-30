#!/usr/bin/env python3
"""Stage-1 campaign analysis — PRE-REGISTERED, exact.  [T11]

Committed BEFORE the sh_64k campaign (see stage0_results/
stage1_preregistration.md, which this file implements verbatim). Run from the
repo root on the box AFTER stage1_campaign_driver.sh finishes:

    python stage0_results/stage1_campaign_analysis.py

No optional stopping: this script is run ONCE, on the full 7+7, and reports
whatever comes out. Grading is the benchmark's own deterministic
``substring_exact_match`` (transcribed in hnav/labeling/counterfactual.py and
pinned by test_counterfactual.py), applied offline to the recorded outputs.

Pre-registered success criterion (ALL must hold):
  1. paired mean Δaccuracy = mean_i[acc(live_i) − acc(off_i)] ≥ +3.0 points;
  2. per-pair harm never exceeds help: harmed_i ≤ helped_i for every i;
  3. pooled harm rate Σ_i harmed_i / (7×100) ≤ 0.02;
  4. token neutral-or-better: mean live (input_len+output_len) ≤
     1.001 × mean off (input_len+output_len).
Supporting (reported, NOT part of the decision): paired t statistic, run-level
accuracies, per-question pooled help/harm counts, A/A context from
stage1_aa_summary.json, and the live-arm audit sanity block (RERANK counts,
embed-cache misses — misses > 0 or missing/short audit files make the run
VOID, criterion unevaluable, report and stop).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAB = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
DATA = MAB / "data/Conflict_Resolution.json"
AUDIT_DIR = REPO / "hnav/_out/audit"
N_PAIRS = 7
SUBSET_PREFIX = "factconsolidation_sh_64k_"

sys.path.insert(0, str(REPO))
from hnav.labeling.counterfactual import substring_exact_match  # noqa: E402


def load_truths() -> list:
    data = json.load(open(DATA, encoding="utf-8"))
    for item in data:
        if item["metadata"]["qa_pair_ids"][0].startswith(SUBSET_PREFIX):
            return list(item["answers"])
    sys.exit("sh_64k subset not found in the dataset")


def load_run(tag: str) -> dict:
    pat = str(MAB / f"outputs/hnav_stage1c_{tag}/*/*.json")
    files = glob.glob(pat)
    if len(files) != 1:
        sys.exit(f"expected exactly 1 results JSON for {tag}, found {len(files)} ({pat})")
    d = json.load(open(files[0], encoding="utf-8"))
    rows = d["data"]
    if len(rows) != 100:
        sys.exit(f"{tag}: expected 100 results, got {len(rows)}")
    return {"tag": tag,
            "outputs": [r["output"] for r in rows],
            "input_len": [int(r.get("input_len", 0)) for r in rows],
            "output_len": [int(r.get("output_len", 0)) for r in rows]}


def grade(run: dict, truths: list) -> np.ndarray:
    return np.array([bool(substring_exact_match(o, t))
                     for o, t in zip(run["outputs"], truths)])


def audit_sanity(tag: str) -> dict:
    """Live-arm audit trail: present, complete, armed, cache-clean."""
    path = AUDIT_DIR / f"stage1c_{tag}.read.jsonl"
    out = {"path": str(path), "present": path.exists(), "n_read": 0,
           "n_rerank": 0, "n_pass": 0, "embed_misses": None, "void": True}
    if not path.exists():
        return out
    n_rerank = n_pass = n_read = 0
    misses = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("schema") != "hnav.read.v1":
            continue
        n_read += 1
        if rec.get("hnav_action") == "RERANK":
            n_rerank += 1
        else:
            n_pass += 1
        ec = (rec.get("hnav_reasons") or {}).get("embed_cache") or {}
        if ec.get("misses"):
            misses = max(misses, int(ec["misses"]))
    out.update({"n_read": n_read, "n_rerank": n_rerank, "n_pass": n_pass,
                "embed_misses": misses,
                "void": not (n_read == 100 and misses == 0)})
    return out


def main() -> int:
    truths = load_truths()
    offs = [load_run(f"off_{i}") for i in range(1, N_PAIRS + 1)]
    lives = [load_run(f"live_{i}") for i in range(1, N_PAIRS + 1)]

    off_acc = np.array([grade(r, truths).mean() * 100 for r in offs])
    live_correct = [grade(r, truths) for r in lives]
    off_correct = [grade(r, truths) for r in offs]
    live_acc = np.array([c.mean() * 100 for c in live_correct])

    deltas = live_acc - off_acc
    helped = [int(np.sum(l & ~o)) for l, o in zip(live_correct, off_correct)]
    harmed = [int(np.sum(o & ~l)) for l, o in zip(live_correct, off_correct)]
    pooled_harm = sum(harmed) / (N_PAIRS * 100)

    tok_off = np.mean([np.sum(r["input_len"]) + np.sum(r["output_len"]) for r in offs])
    tok_live = np.mean([np.sum(r["input_len"]) + np.sum(r["output_len"]) for r in lives])
    tok_ratio = float(tok_live / tok_off) if tok_off else float("nan")

    audits = [audit_sanity(f"live_{i}") for i in range(1, N_PAIRS + 1)]
    any_void = any(a["void"] for a in audits)

    sd = float(deltas.std(ddof=1))
    t_stat = float(deltas.mean() / (sd / np.sqrt(N_PAIRS))) if sd > 0 else float("inf")

    crit = {
        "c1_delta_ge_3": bool(deltas.mean() >= 3.0),
        "c2_per_pair_harm_le_help": bool(all(h <= p for h, p in zip(harmed, helped))),
        "c3_pooled_harm_le_2pct": bool(pooled_harm <= 0.02),
        "c4_token_neutral": bool(tok_ratio <= 1.001),
    }
    verdict = ("VOID (audit sanity failed — report and stop)" if any_void
               else ("SUCCESS" if all(crit.values()) else "NOT SUPPORTED"))

    aa_path = REPO / "stage0_results/stage1_aa_summary.json"
    aa = json.loads(aa_path.read_text()) if aa_path.exists() else None

    out = {
        "protocol": "stage0_results/stage1_preregistration.md",
        "subset": "factconsolidation_sh_64k",
        "n_pairs": N_PAIRS,
        "off_acc": off_acc.tolist(), "live_acc": live_acc.tolist(),
        "paired_delta": deltas.tolist(),
        "paired_delta_mean": float(deltas.mean()),
        "paired_delta_sd": sd,
        "paired_t": t_stat,
        "helped_per_pair": helped, "harmed_per_pair": harmed,
        "pooled_harm_rate": pooled_harm,
        "token_mean_off": float(tok_off), "token_mean_live": float(tok_live),
        "token_ratio_live_over_off": tok_ratio,
        "criteria": crit,
        "audit_sanity": audits,
        "verdict": verdict,
        "aa_context": None if aa is None else {
            "subset": aa["subset"], "sem_sd": aa["sem_sd"],
            "pairwise_mismatch_mean": aa["pairwise_mismatch"]["mean"]},
    }
    dst = REPO / "stage0_results/stage1_campaign_result.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nverdict: {verdict}\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
