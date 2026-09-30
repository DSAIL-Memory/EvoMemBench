# All figures for the evidence pack. PNG (300 dpi) + SVG per chart.
# Palette: dataviz reference categorical slots, validated with validate_palette.js
# (contrast WARN on slots 3-5 -> every chart carries direct labels; CSVs in data/ are the table view).
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
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

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

def style_ax(ax, ygrid_only=True):
    if ygrid_only:
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

def caption(fig, text):
    fig.text(0.01, -0.025, text, fontsize=6.5, color=MUTED, ha="left", va="top")

def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"

qs = load("stage0_results/question_strata.json")
RUNS = qs["runs"]
run_names = [r["run"].replace("sh_6k_", "") for r in RUNS]

# ── fig02a: per-run conflicted-stratum outcome classes (stacked) ─────────────
fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(RUNS))
correct = np.array([r["strata"]["conflicted"]["correct"] for r in RUNS])
stale = np.array([r["strata"]["conflicted"]["errors"]["stale_value"] for r in RUNS])
off_list = np.array([r["strata"]["conflicted"]["errors"]["off_list"] for r in RUNS])
w = 0.62
ax.bar(x, correct, w, color=S1, label="correct", edgecolor=SURFACE, linewidth=1)
ax.bar(x, stale, w, bottom=correct, color=S2, label="stale value (same key, superseded value)",
       edgecolor=SURFACE, linewidth=1)
ax.bar(x, off_list, w, bottom=correct + stale, color=S3, label="off-list", edgecolor=SURFACE, linewidth=1)
for i in range(len(RUNS)):
    ax.text(x[i], correct[i] + stale[i] / 2, str(stale[i]), ha="center", va="center",
            color="#ffffff", fontsize=9, fontweight="bold")
    if correct[i]:
        ax.text(x[i], correct[i] / 2, str(correct[i]), ha="center", va="center",
                color="#ffffff", fontsize=8)
ax.set_xticks(x, run_names)
ax.set_ylabel("conflicted-stratum questions (n = 74 per run)")
ax.set_title("Conflicted stratum, 8 committed sh_6k runs: nearly every error reads the stale value")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=3, fontsize=8)
style_ax(ax)
caption(fig, "source: stage0_results/question_strata.json -> runs[].strata.conflicted; classes defined in .definitions")
save(fig, "fig02a_error_classes_per_run")

# ── fig02b: aggregate error classes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 2.4))
classes = ["stale_value", "off_list", "empty"]
vals = [qs["aggregate"]["errors_total"][c] for c in classes]
colors = [S2, S3, MUTED]
y = np.arange(len(classes))[::-1]
ax.barh(y, vals, 0.55, color=colors, edgecolor=SURFACE, linewidth=1)
for yi, v in zip(y, vals):
    ax.text(v + 6, yi, str(v), va="center", color=INK, fontsize=10, fontweight="bold")
ax.set_yticks(y, ["stale value\n(same key)", "off-list", "empty"])
ax.set_xlim(0, 640)
ax.set_xlabel("errors across all 8 runs (575 total)")
ax.set_title("572 of 575 errors name the superseded value of the correct key")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
caption(fig, "source: stage0_results/question_strata.json -> aggregate.errors_total (recomputed from runs[].strata)")
save(fig, "fig02b_error_classes_total")

# ── fig03: unique vs conflicted accuracy per run ─────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.2))
uacc = [r["strata"]["unique"]["accuracy"] for r in RUNS]
cacc = [r["strata"]["conflicted"]["accuracy"] for r in RUNS]
w = 0.38
ax.bar(x - w / 2, uacc, w, color=S1, label="unique stratum (no conflict), n=26", edgecolor=SURFACE, linewidth=1)
ax.bar(x + w / 2, cacc, w, color=S2, label="conflicted stratum, n=74", edgecolor=SURFACE, linewidth=1)
ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(-0.62, 1.045, "ceiling = 1.0", color=MUTED, fontsize=8, ha="left")
for i in range(len(RUNS)):
    ax.text(x[i] - w / 2, uacc[i] + 0.015, "100%", ha="center", color=SEC, fontsize=7.5)
    ax.text(x[i] + w / 2, cacc[i] + 0.015, pct(cacc[i], 1), ha="center", color=SEC, fontsize=7.5)
ax.set_xticks(x, run_names)
ax.set_ylim(0, 1.09)
ax.set_ylabel("accuracy (substring_exact_match)")
ax.set_title("Same model, same runs: 100% without conflict, near 0% with conflict (sh_6k)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=2, fontsize=8)
style_ax(ax)
caption(fig, "source: stage0_results/question_strata.json -> runs[].strata.{unique,conflicted}; raw runs in stage0_results/t4_s2_evidence/")
save(fig, "fig03_strata_collapse")

# ── fig05a: position arms, conflicted accuracy ───────────────────────────────
item5 = load("presentation_evidence/data/item05_arms.json")
subsets5 = ["sh_6k", "sh_32k", "sh_64k"]
arm_keys = {
    "sh_6k": ("native", "oracle_recency", "anti"),
    "sh_32k": ("native", "oracle_recency", "anti"),
    "sh_64k": ("native", "detector_demote_late", "detector_anti"),
}
def arms_for(tag):
    src = item5[tag] if tag != "sh_64k" else item5["sh_64k_confirmatory_mirror"]
    return src["conflicted_arms"]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
gx = np.arange(len(subsets5))
w = 0.26
labels5 = ["baseline (serial order)", "newest fact moved to END", "newest fact moved to FRONT"]
colors5 = [S1, S2, S3]
for j in range(3):
    vals = [arms_for(t)[arm_keys[t][j]]["accuracy"] for t in subsets5]
    ns = [arms_for(t)[arm_keys[t][j]]["n"] for t in subsets5]
    b = ax.bar(gx + (j - 1) * w, vals, w * 0.92, color=colors5[j], label=labels5[j],
               edgecolor=SURFACE, linewidth=1)
    for xi, v, n in zip(gx + (j - 1) * w, vals, ns):
        ax.text(xi, v + 0.012, pct(v), ha="center", color=SEC, fontsize=7.5)
ax.set_xticks(gx, ["sh_6k\n(calibration, oracle probe)", "sh_32k\n(calibration, oracle probe)",
                   "sh_64k\n(HELD OUT, detector harness)"])
ax.set_ylim(0, 0.62)
ax.set_ylabel("conflicted-stratum accuracy")
ax.set_title("Moving one fact changes the answers — despite an explicit recency instruction")
ax.legend(loc="upper left", fontsize=8)
style_ax(ax)
caption(fig, "sources: stale_suppression_probe_sh{6,32}k.json and detector_gap_confirmatory_sh64k.json -> "
             "results[0].by_stratum.conflicted.arms; sh_64k arms are detector-planned edits (different harness)")
save(fig, "fig05a_position_arms")

# ── fig05b: NEW / OLD / OTHER taxonomy ───────────────────────────────────────
tax = load("presentation_evidence/data/item05_taxonomy.json")
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharex=False)
seg_colors = {"NEW": S1, "OLD": S2, "OTHER": MUTED}
for ax, block in zip(axes, tax["blocks"]):
    arms = list(block["arms"].items())
    y = np.arange(len(arms))[::-1]
    left = np.zeros(len(arms))
    n = block["n_conflicted"]
    for seg in ("NEW", "OLD", "OTHER"):
        vals = np.array([a[1][seg] for a in arms], dtype=float)
        ax.barh(y, vals, 0.6, left=left, color=seg_colors[seg], edgecolor=SURFACE, linewidth=1,
                label=seg if ax is axes[0] else None)
        for yi, v, l in zip(y, vals, left):
            if v >= max(3, 0.06 * n):
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        color="#ffffff", fontsize=8, fontweight="bold")
        left += vals
    ax.set_yticks(y, [a[0] for a in arms], fontsize=7.5)
    ax.set_xlabel(f"conflicted questions (n = {n})", fontsize=8)
    short = block["subset"].replace("factconsolidation_", "")
    ax.set_title(f"{short}  [{block['split']}]", fontsize=10)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
fig.suptitle("Which value does the model actually name? (NEW = highest serial, OLD = superseded, A/A row = noise floor 0)",
             fontsize=11.5, fontweight="bold")
fig.legend(loc="lower right", ncols=3, fontsize=8.5, bbox_to_anchor=(0.99, -0.02))
fig.tight_layout(rect=(0, 0.03, 1, 1))
caption(fig, "source: presentation_evidence/data/item05_taxonomy.json (position_taxonomy.py, commit 50dd955, re-reads committed raw outputs)")
save(fig, "fig05b_taxonomy")

# ── fig06: geometry percentile summary ───────────────────────────────────────
m1 = load("stage0_results/final/m1_geometry_calibration.json")
fig, ax = plt.subplots(figsize=(7.6, 4.4))
series6 = [("whole_blob_sim", "conflict pairs (whole-fact cosine)", S1),
           ("control_whole_blob_sim", "random control pairs", S2),
           ("diff_sim", "changed-span (diff) cosine", S3)]
gx = np.arange(len(m1))
for j, (key, label, col) in enumerate(series6):
    xs = gx + (j - 1) * 0.22
    p10 = [e[key]["p10"] for e in m1]; p50 = [e[key]["p50"] for e in m1]; p90 = [e[key]["p90"] for e in m1]
    ax.vlines(xs, p10, p90, color=col, linewidth=3.2, alpha=0.85, label=label)
    ax.scatter(xs, p50, s=42, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
for i, e in enumerate(m1):
    ax.text(gx[i], 0.145, f"AUC {e['separation_auc_conflict_vs_control']:.4f}",
            ha="center", fontsize=8, color=SEC)
    ax.text(gx[i], 0.085, f"{e['n_conflict_pairs']} / {e['n_control_pairs']} pairs",
            ha="center", fontsize=7, color=MUTED)
ax.set_xticks(gx, [e["subset"] for e in m1])
ax.set_ylim(0.05, 1.02)
ax.set_ylabel("cosine similarity (p10 – p50 – p90)")
ax.set_title("Percentile summary: conflicting facts sit far closer than random pairs")
ax.legend(loc="center left", fontsize=8)
style_ax(ax)
caption(fig, "source: stage0_results/final/m1_geometry_calibration.json — PERCENTILE SUMMARIES ONLY; "
             "raw per-pair arrays are not stored, so no histogram/ROC can be drawn offline")
save(fig, "fig06_geometry_percentiles")

# ── fig07a: precision-recall sweep ───────────────────────────────────────────
m1b = load("stage0_results/final/m1b_grouping_ablation.json")
cols7 = [S1, S2, S3, S4]
fig, ax = plt.subplots(figsize=(6.8, 5.0))
offsets7 = {"sh_6k": (8, 8, "left"), "sh_32k": (8, -20, "left"),
            "sh_64k": (-10, 12, "right"), "sh_262k": (-10, -22, "right")}
for e, col in zip(m1b, cols7):
    rec = [p["recall"] for p in e["pr_curve"]]
    prec = [p["precision"] for p in e["pr_curve"]]
    ax.plot(rec, prec, color=col, linewidth=2, label=e["subset"])
    bf = e["best_f1"]
    ax.scatter([bf["recall"]], [bf["precision"]], s=52, color=col, edgecolor=SURFACE,
               linewidth=1.5, zorder=3)
    dx, dy, ha = offsets7[e["subset"]]
    ax.annotate(f"{e['subset']}: F1 {bf['f1']:.2f} @ tau {bf['tau']:.2f}",
                (bf["recall"], bf["precision"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=7.5, color=SEC, ha=ha)
ax.set_xlabel("recall (vs parser-derived truth pairs)")
ax.set_ylabel("precision")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_title("Geometry-only pairing: precision–recall over the cosine threshold sweep")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(True)
caption(fig, "source: stage0_results/final/m1b_grouping_ablation.json -> pr_curve (50 points per subset), best_f1")
save(fig, "fig07a_pr_curve")

# ── fig07b: F1 vs tau ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for e, col in zip(m1b, cols7):
    tau = [p["tau"] for p in e["pr_curve"]]
    f1 = [p["f1"] for p in e["pr_curve"]]
    ax.plot(tau, f1, color=col, linewidth=2, label=e["subset"])
    bf = e["best_f1"]
    ax.scatter([bf["tau"]], [bf["f1"]], s=48, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
ax.set_xlabel("cosine threshold tau")
ax.set_ylabel("F1")
ax.set_ylim(0, 1.0)
ax.set_title("F1 across the threshold sweep, best point marked per subset")
ax.legend(loc="lower left", fontsize=8.5)
style_ax(ax)
caption(fig, "source: stage0_results/final/m1b_grouping_ablation.json -> pr_curve, best_f1.tau")
save(fig, "fig07b_f1_vs_tau")

# ── fig10: NLI false-verification strip ──────────────────────────────────────
cal = load("stage0_results/stage1/stage1_calibration.json")
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
groups = [(False, 0, S2, "subject screen OFF\n(pair_filter = false)"),
          (True, 1, S1, "subject screen ON\n(pair_filter = true)")]
for pf, gx0, col, label in groups:
    ys = [(c["n_fv_diff_key"] + c["n_fv_same_object"]) / c["n_verified"]
          for c in cal["cells"] if c["pair_filter"] == pf]
    xs = gx0 + rng.uniform(-0.13, 0.13, len(ys))
    ax.scatter(xs, ys, s=26, color=col, alpha=0.75, edgecolor=SURFACE, linewidth=0.8)
ax.text(1, 0.045, "81 cells — all exactly 0.000", ha="center", fontsize=9, color=S1, fontweight="bold")
off_rates = [(c["n_fv_diff_key"] + c["n_fv_same_object"]) / c["n_verified"]
             for c in cal["cells"] if not c["pair_filter"]]
ax.text(0, max(off_rates) + 0.035, f"range {min(off_rates):.2f}–{max(off_rates):.2f}",
        ha="center", fontsize=9, color=S2)
ax.set_xticks([0, 1], [g[3] for g in groups])
ax.set_ylim(-0.04, 1.04)
ax.set_ylabel("false-verification rate per cell\n(n_fv_diff_key + n_fv_same_object) / n_verified")
ax.set_title("Bidirectional NLI alone false-verifies 31–94% of pairs; the parsed subject screen: 0")
style_ax(ax)
caption(fig, "source: stage0_results/stage1/stage1_calibration.json -> cells (162 = 81 + 81); rate recomputed per cell; "
             "NLI model cross-encoder/nli-deberta-v3-large")
save(fig, "fig10_nli_false_verification")

# ── fig11: helped vs harmed scatter ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 5.6))
lim = max(max(c["helped"] for c in cal["cells"]), max(c["harmed"] for c in cal["cells"])) + 3
ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
from collections import Counter
for pf, col, label in [(True, S1, "subject screen ON (81 cells)"),
                       (False, S2, "subject screen OFF (81 cells)")]:
    counts = Counter((c["helped"], c["harmed"]) for c in cal["cells"] if c["pair_filter"] == pf)
    xs = [k[0] for k in counts]; ys = [k[1] for k in counts]
    sizes = [26 * v for v in counts.values()]
    ax.scatter(xs, ys, s=sizes, color=col, alpha=0.7, edgecolor=SURFACE, linewidth=0.8, label=label)
ax.text(lim * 0.79, lim * 0.30, "marker area = number of cells\nat that (helped, harmed) point",
        fontsize=7.5, color=MUTED, ha="center")
ax.text(lim * 0.30, lim * 0.88, "above diagonal:\nchunk moves HARM more than help",
        fontsize=8.5, color=SEC, ha="center")
ax.text(lim * 0.80, lim * 0.10, "below diagonal:\nnet positive", fontsize=8.5, color=SEC, ha="center")
ax.set_xlabel("questions helped (per cell)")
ax.set_ylabel("questions harmed (per cell)")
ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
ax.set_aspect("equal")
ax.set_title("Chunk-level intervention: every screen-on cell is at or above the y = x line")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(True)
caption(fig, "source: stage0_results/stage1/stage1_calibration.json -> cells[].{helped,harmed,pair_filter}; "
             "screen-on: 0 of 81 net-positive (helped 228 / harmed 441)")
save(fig, "fig11_helped_vs_harmed")

# ── fig12: detector vs oracle net improvement ────────────────────────────────
item12 = load("presentation_evidence/data/item12_detector_vs_oracle.json")
fig, ax = plt.subplots(figsize=(6.0, 4.2))
tags = ["sh_6k", "sh_32k"]
gx = np.arange(len(tags))
w = 0.32
oracle = [item12[t]["suppress"]["oracle_net"] for t in tags]
det = [item12[t]["suppress"]["detector_net"] for t in tags]
ax.bar(gx - w / 2, oracle, w, color=S1, label="oracle suppression (ground truth)", edgecolor=SURFACE, linewidth=1)
ax.bar(gx + w / 2, det, w, color=S2, label="detector suppression (gold-free)", edgecolor=SURFACE, linewidth=1)
for i, t in enumerate(tags):
    ax.text(gx[i] - w / 2, oracle[i] + 0.8, f"+{oracle[i]}", ha="center", color=SEC, fontsize=9)
    ax.text(gx[i] + w / 2, det[i] + 0.8, f"+{det[i]}", ha="center", color=SEC, fontsize=9)
    ratio = item12[t]["suppress"]["net_ratio"]
    ax.text(gx[i], max(oracle[i], det[i]) + 5.2, f"ratio {ratio:.3f}", ha="center",
            color=INK, fontsize=10, fontweight="bold")
ax.set_xticks(gx, ["sh_6k (calibration)", "sh_32k (calibration)"])
ax.set_ylim(0, 74)
ax.set_ylabel("net questions gained vs native (McNemar net)")
ax.set_title("The gold-free detector recovers 98.4% / 95.7% of the oracle's net gain")
ax.legend(loc="upper right", fontsize=8.5)
style_ax(ax)
caption(fig, "source: detector_gap_sh{6,32}k.json -> detector_vs_oracle[..].by_mechanism.detector_suppress; "
             "CALIBRATION ONLY — no oracle arm exists at sh_64k (confirmatory corrections[0].items[4])")
save(fig, "fig12_detector_vs_oracle")

# ── fig13: confirmatory arms x strata ────────────────────────────────────────
item13 = load("presentation_evidence/data/item13_summary.json")
arm_order = ["native", "native_repeat", "detector_suppress", "detector_demote_late", "detector_anti"]
arm_labels = ["native", "native repeat\n(A/A)", "detector\nsuppress", "detector\ndemote-late", "detector\nanti"]
arm_cols = [S1, S2, S3, S4, S5]
strata13 = [("overall", "overall (n=100)"), ("unique", "non-conflicted (n=34)"), ("conflicted", "conflicted (n=66)")]
fig, ax = plt.subplots(figsize=(9.6, 4.6))
gx = np.arange(len(strata13))
w = 0.15
for j, (arm, col) in enumerate(zip(arm_order, arm_cols)):
    vals = [item13["arms_by_stratum"][arm][s]["accuracy"] for s, _ in strata13]
    xs = gx + (j - 2) * w
    ax.bar(xs, vals, w * 0.9, color=col, label=arm_labels[j].replace("\n", " "), edgecolor=SURFACE, linewidth=1)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.012, pct(v, 0 if v >= 0.995 else 1), ha="center", color=SEC, fontsize=6.8)
supp_x = gx[2] + 0 * w  # detector_suppress is j=2 -> offset 0
ax.annotate("McNemar: 0 harmed / 20 fixed\np = 1.9e-06", (gx[2] + 0 * w, 0.5606),
            textcoords="offset points", xytext=(10, 26), fontsize=8.5, color=INK,
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
ax.set_xticks(gx, [lbl for _, lbl in strata13])
ax.set_ylim(0, 1.0)
ax.set_ylabel("accuracy (substring_exact_match)")
ax.set_title("Held-out sh_64k, one pre-registered run: suppression lifts conflicted accuracy 25.8% -> 56.1%")
ax.legend(loc="upper right", fontsize=7.6, ncols=2)
style_ax(ax)
caption(fig, "source: stage0_results/stage1/detector_gap_confirmatory_sh64k.json -> results[0].{arms,by_stratum}; "
             "protective criterion VOIDED by q77 (unique stratum, refusal_after_edit) — effective, not yet safe")
save(fig, "fig13_confirmatory_arms")

# ── fig15: serving noise floor vs effect size ────────────────────────────────
t4 = load("stage0_results/t4_s2_trials_summary.json")
off = np.array(t4["off_sem_per_run"]) / 100.0
sha = np.array(t4["shadow_sem_per_run"]) / 100.0
fig, ax = plt.subplots(figsize=(8.6, 3.6))
def beeswarm(vals, y0, col, label):
    seen = {}
    for v in vals:
        k = round(v, 4)
        seen[k] = seen.get(k, 0) + 1
        ax.scatter([v], [y0 + (seen[k] - 1) * 0.13], s=46, color=col,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.text(-0.012, y0, label, ha="right", va="center", fontsize=9, color=INK)
band_lo = off.mean() - t4["noise_floor"]["within_off_mismatch_mean"]
band_hi = off.mean() + t4["noise_floor"]["within_off_mismatch_mean"]
ax.axvspan(band_lo, band_hi, color=GRID, alpha=0.7, zorder=1)
ax.text((band_lo + band_hi) / 2, 2.62, "baseline's own run-to-run band\n(within-off mismatch mean 3.04%)",
        ha="center", fontsize=7.5, color=SEC)
beeswarm(off, 2.0, S1, "HNAV off (10 runs, sh_6k)")
beeswarm(sha, 1.3, S2, "HNAV shadow (5 runs, sh_6k)")
y_eff = 0.45
ax.annotate("", xy=(37 / 66, y_eff), xytext=(17 / 66, y_eff),
            arrowprops=dict(arrowstyle="-|>", color=S3, linewidth=2.2))
ax.scatter([17 / 66], [y_eff], s=46, color=S3, edgecolor=SURFACE, linewidth=1.2, zorder=3)
ax.text(-0.012, y_eff, "held-out effect (sh_64k,\nconflicted stratum)", ha="right", va="center", fontsize=9, color=INK)
ax.text((17 / 66 + 37 / 66) / 2, y_eff + 0.17, "25.8% -> 56.1%  (+30.3 pts)", ha="center",
        fontsize=9, color=S3, fontweight="bold")
ax.set_yticks([])
ax.set_ylim(0, 3.1)
ax.set_xlim(-0.02, 0.72)
ax.set_xlabel("accuracy")
ax.set_title("Serving noise (off-vs-off 3.04%, off-vs-shadow 2.42%) vs the held-out effect, one axis")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
caption(fig, "sources: stage0_results/t4_s2_trials_summary.json (runs, noise floor, TOST, permutation); "
             "detector_gap_confirmatory_sh64k.json (effect; different subset & stratum, drawn for scale)")
save(fig, "fig15_noise_floor_vs_effect")

print("ALL FIGURES DONE")
