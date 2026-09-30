# fig16 / fig17: H-Nav detector suppression vs the oracle ceiling.
# Every number is read live from the committed artifacts - nothing is hardcoded.
# Palette: dataviz reference categorical slots, validated with validate_palette.js
# (3 slots, ALL CHECKS PASS; contrast WARN on #1baf7a -> every bar is direct-labelled).
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "presentation_evidence", "figures")

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; SEC = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": SEC,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "svg.fonttype": "none",
})


def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


def conflicted_arms(path):
    return load(path)["results"][0]["by_stratum"]["conflicted"]["arms"]


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ── numbers, read live ───────────────────────────────────────────────────────
P = "stage0_results/stage1/"
O6 = conflicted_arms(P + "stale_suppression_probe_sh6k.json")
D6 = conflicted_arms(P + "detector_gap_sh6k.json")
O32 = conflicted_arms(P + "stale_suppression_probe_sh32k.json")
D32 = conflicted_arms(P + "detector_gap_sh32k.json")
C64 = conflicted_arms(P + "detector_gap_confirmatory_sh64k.json")

valid = [
    ("sh_6k", O6["native"], O6["oracle_suppress"], D6["detector_suppress"]),
    ("sh_32k", O32["native"], O32["oracle_suppress"], D32["detector_suppress"]),
]
held = ("sh_64k", C64["native"], C64["detector_suppress"])

# ── fig16: grouped bars, valid comparison separated from the held-out run ────
fig = plt.figure(figsize=(11.4, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.10,
                      left=0.075, right=0.985, top=0.70, bottom=0.175)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1], sharey=axL)

W = 0.26
for ax in (axL, axR):
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])


def bar_labels(ax, pos, v, c, n):
    ax.text(pos, v + 1.8, f"{v:.1f}%", ha="center", va="bottom",
            fontsize=11, color=INK, fontweight="bold")
    ax.text(pos, v + 8.0, f"{c}/{n}", ha="center", va="bottom",
            fontsize=8.2, color=MUTED)


x = np.arange(len(valid))
series = [
    ("Native (no intervention)", S1, 1, None),
    ("Oracle stale suppression (ceiling)", S2, 2, "///"),
    ("H-Nav detector suppression", S3, 3, None),
]
for (label, color, idx, hatch) in series:
    pos = x + (idx - 2) * (W + 0.02)
    for xi, v4 in zip(pos, valid):
        a = v4[idx]
        v = a["correct"] / a["n"] * 100
        axL.bar([xi], [v], W, label=label if xi == pos[0] else None,
                color=color, hatch=hatch, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        bar_labels(axL, xi, v, a["correct"], a["n"])

axL.set_xticks(x)
axL.set_xticklabels(["sh_6k\n(calibration split)", "sh_32k\n(calibration split)"],
                    fontsize=10.5, color=SEC)
axL.set_ylabel("conflicted-stratum accuracy (%)", fontsize=10)
axL.text(0, 1.035, "VALID ORACLE COMPARISON  —  whole-context harness; oracle and\ndetector arms run on the identical prompt shape and identical questions",
         transform=axL.transAxes, fontsize=10, color=INK, fontweight="bold",
         va="bottom", ha="left", linespacing=1.5)
axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=3,
           fontsize=9.5, labelcolor=SEC, handlelength=1.6, columnspacing=1.6)

# held-out panel
for (label, color, idx, hatch), val in zip([series[0], series[2]], [held[1], held[2]]):
    p = 0.0 if idx == 1 else 0.34
    v = val["correct"] / val["n"] * 100
    axR.bar([p], [v], W, color=color, hatch=hatch, edgecolor=SURFACE,
            linewidth=1.6, zorder=3)
    bar_labels(axR, p, v, val["correct"], val["n"])

axR.set_xticks([0.17])
axR.set_xticklabels(["sh_64k\n(HELD OUT, one shot)"], fontsize=10.5, color=SEC)
axR.set_xlim(-0.30, 0.64)
axR.tick_params(labelleft=False, left=False)
axR.text(0, 1.035, "NOT COMPARABLE TO THE LEFT PANEL —\nretrieved-page harness, and no oracle arm exists",
         transform=axR.transAxes, fontsize=10, color=INK, fontweight="bold",
         va="bottom", ha="left", linespacing=1.5)
axR.text(0.17, 96.0,
         "NO ORACLE ARM WAS EVER RUN HERE.\nA whole-context prompt is 75,886 tokens\nagainst a 65,536 limit, so the ceiling at\nthis scale is unmeasured. The missing bar\nis absent, not zero, and the 100% / 95.7%\nratios may not be carried over.",
         ha="center", va="top", fontsize=8.4, color=SEC, linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.55", facecolor="#f4f2ea",
                   edgecolor=BASELINE, linewidth=0.9))

fig.add_artist(plt.Line2D([0.706, 0.706], [0.10, 0.755], color=BASELINE,
                          linewidth=1.1, linestyle=(0, (4, 3))))

fig.text(0.012, 0.965,
         "H-Nav recovers nearly all of the oracle ceiling — where the ceiling was measured",
         fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.012, 0.905,
         "Oracle = stale facts deleted using gold labels: an upper bound on the intervention, not a deployable system.\n"
         "H-Nav = the real gold-free detector; its thresholds were frozen on detection quality alone before any answer was graded.",
         fontsize=9.2, color=SEC, ha="left", va="top", linespacing=1.6)
fig.text(0.012, 0.035,
         "sources: stage0_results/stage1/stale_suppression_probe_sh{6,32}k.json · detector_gap_sh{6,32}k.json · detector_gap_confirmatory_sh64k.json "
         "→ results[0].by_stratum.conflicted.arms.   sh_64k uses the benchmark's own retrieved top-10 page (retrieval incomplete: 10 of 17 chunks) and its detector_vs_oracle block is empty by design.",
         fontsize=7, color=MUTED, ha="left", va="top")
save(fig, "fig16_oracle_vs_detector")

# ── fig17: fraction of the oracle gain captured ─────────────────────────────
fig2 = plt.figure(figsize=(7.6, 3.4))
ax = fig2.add_axes([0.155, 0.30, 0.815, 0.42])
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
names, caps, detail = [], [], []
for sub, nat, orc, det in valid:
    na = nat["correct"] / nat["n"] * 100
    oa = orc["correct"] / orc["n"] * 100
    da = det["correct"] / det["n"] * 100
    caps.append((da - na) / (oa - na) * 100)
    names.append(sub)
    detail.append(f"({da:.1f} − {na:.1f}) ÷ ({oa:.1f} − {na:.1f})")

y = np.arange(len(names))[::-1]
ax.barh(y, caps, 0.46, color=S3, edgecolor=SURFACE, linewidth=1.6, zorder=3)
ax.axvline(100, color=BASELINE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
for yi, c, d in zip(y, caps, detail):
    ax.text(c - 2.0, yi + 0.10, f"{c:.1f}%", ha="right", va="center",
            fontsize=13, color="#ffffff", fontweight="bold")
    ax.text(2.5, yi - 0.145, d, ha="left", va="center", fontsize=8,
            color="#eaf7f1")
ax.set_yticks(y)
ax.set_yticklabels([f"{n}\n(calibration)" for n in names], fontsize=10.5, color=SEC)
ax.set_ylim(-0.55, 1.55)
ax.set_xlim(0, 116); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of the available oracle gain that H-Nav actually realised (%)",
              fontsize=9.5, labelpad=7)
ax.text(100, 1.62, "oracle ceiling = 100%", fontsize=8.5, color=MUTED,
        ha="right", va="bottom")
fig2.text(0.012, 0.955,
          "How much of the perfect-knowledge opportunity does the real detector recover?",
          fontsize=12, fontweight="bold", color=INK, ha="left", va="top")
fig2.text(0.012, 0.885,
          "Captured gain = (H-Nav − Native) ÷ (Oracle − Native) on the conflicted stratum.",
          fontsize=9, color=SEC, ha="left", va="top")
fig2.text(0.012, 0.085,
          "Computable only where an oracle arm exists. sh_64k has none, so it cannot appear here — "
          "and these two ratios may not be transferred to it.\n"
          "sources: stale_suppression_probe_sh{6,32}k.json · detector_gap_sh{6,32}k.json → results[0].by_stratum.conflicted.arms",
          fontsize=7, color=MUTED, ha="left", va="top", linespacing=1.5)
save(fig2, "fig17_captured_oracle_gain")

# ── printed check ────────────────────────────────────────────────────────────
for sub, nat, orc, det in valid:
    na = nat["correct"] / nat["n"] * 100
    oa = orc["correct"] / orc["n"] * 100
    da = det["correct"] / det["n"] * 100
    print(f"{sub}: native {nat['correct']}/{nat['n']}={na:.2f}%  oracle {orc['correct']}/{orc['n']}={oa:.2f}%  "
          f"detector {det['correct']}/{det['n']}={da:.2f}%  captured={(da-na)/(oa-na)*100:.2f}%")
print(f"sh_64k: native {held[1]['correct']}/{held[1]['n']}  detector {held[2]['correct']}/{held[2]['n']}  (no oracle)")
