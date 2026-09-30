"""Figure for experiment 4 — dimension structure and detector comparison.

Panels:
  A  sorted per-dimension effect size (|Cohen d| of |d_hat_i|, calibration
     gold vs hard negatives) — shows the signal is distributed, no magic dim
  B  topdim calibration AUROC vs mask size m
  C  CES score distributions on the CONFIRMATORY hard-negative task
  D  held-out comparison bars: hard AUROC / balanced-band AUROC / inverted-win

Writes stage0_results/geometry_filter/dimension_ideas.png.

Usage:  python -m hnav.geometry_filter.plot_dimension_ideas
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .methods import fit_training_edits
from .run_dimension_ideas import ContrastiveSubspace


def run() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blob = json.loads((data.OUT_DIR / "dimension_ideas.json").read_text(encoding="utf-8"))

    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    V = data.build_spaces(V_raw)["raw"]
    pv = data.PairView(records, index)
    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])

    pv_pos, pv_neg = pv.subset(gold & cal), pv.subset(hardneg & cal)
    P = np.abs(pv_pos.diff(V, normalize=True, oriented=False))
    N = np.abs(pv_neg.diff(V, normalize=True, oriented=False))
    d_eff = (P.mean(0) - N.mean(0)) / np.sqrt(
        0.5 * (P.std(0) ** 2 + N.std(0) ** 2) + 1e-12)

    D_pos, rel_pos = fit_training_edits(records, pv, V, gold & cal)
    D_neg = pv_neg.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace().fit(D_pos, rel_pos, D_neg, list(pv_neg.relation))
    conf_mask = (gold | hardneg) & ~cal
    view = pv.subset(conf_mask)
    y = gold[conf_mask]
    ces_scores = ces.score(view, V)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Difference-vector dimension analysis and detector comparison "
                 "(fit: calibration only; panels C/D: held-out sh_64k)", fontsize=11)

    ax = axes[0, 0]
    srt = np.sort(np.abs(d_eff))[::-1]
    ax.plot(np.arange(1, len(srt) + 1), srt, lw=1.2)
    ax.set_xscale("log")
    ax.axhline(0.2, color="gray", ls="--", lw=0.8)
    ax.annotate(f"max |d| = {srt[0]:.2f}\n{int((srt > 0.2).sum())} dims > 0.2",
                xy=(1.5, srt[0]), fontsize=9,
                xytext=(4, srt[0] * 0.85))
    ax.set_xlabel("dimension rank (log)")
    ax.set_ylabel("|Cohen d| of |d_hat_i|")
    ax.set_title("A  no dominant 'conflict dimension' — signal is distributed",
                 fontsize=10, loc="left")

    ax = axes[0, 1]
    grid = blob["dimension_analysis"]["topdim_cal_auroc_by_m"]
    ms = sorted(int(m) for m in grid)
    ax.plot(ms, [grid[str(m)] for m in ms], "o-", label="topdim energy fraction (cal)")
    ax.set_xscale("log")
    ax.set_xlabel("mask size m (top positive-effect dims)")
    ax.set_ylabel("calibration AUROC")
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("B  hard masks saturate ~0.91 — weighting/structure needed",
                 fontsize=10, loc="left")

    ax = axes[1, 0]
    bins = np.linspace(ces_scores.min(), ces_scores.max(), 80)
    ax.hist(ces_scores[~y], bins=bins, density=True, alpha=0.6,
            label=f"hard negatives (n={int((~y).sum()):,})")
    ax.hist(ces_scores[y], bins=bins, density=True, alpha=0.6,
            label=f"gold conflicts (n={int(y.sum()):,})")
    ax.set_xlabel("CES score  =  ||U_obj^T d||^2  -  ||U_subj^T d||^2")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("C  CES on held-out sh_64k (AUROC 0.981)", fontsize=10, loc="left")

    ax = axes[1, 1]
    methods = ["abtt_cos", "ces", "axis_lr", "rcesp", "campaign_cos", "topdim"]
    hardm = blob["hard_negative_confirmatory"]["methods"]
    balm = blob["balanced_sh64k"]["methods"]
    metrics = {
        "hard-task AUROC": [hardm[m]["auroc"] for m in methods],
        "balanced band AUROC": [balm[m]["band_auroc"] for m in methods],
        "inverted-win vs cosine": [hardm[m]["inverted_vs_campaign_cos"]["win_rate"]
                                   for m in methods],
    }
    x = np.arange(len(methods))
    w = 0.27
    for j, (label, vals) in enumerate(metrics.items()):
        ax.bar(x + (j - 1) * w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("D  held-out comparison — CES wins band + inverted, "
                 "ABTT-cos wins aggregate", fontsize=10, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dst = data.OUT_DIR / "dimension_ideas.png"
    fig.savefig(dst, dpi=150)
    print("written:", dst)


if __name__ == "__main__":
    run()
