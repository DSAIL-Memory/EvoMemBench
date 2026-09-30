#!/usr/bin/env python3
"""M7c - the conflict fingerprint, drawn to be read.  [visualisation only]

Companion to M7b. M7b established THAT conflict difference vectors carry a
per-coordinate sign pattern cosine cannot see; this script draws WHAT that
pattern is - which coordinates, which sign, how concentrated, how reproducible
across subsets and spaces, and how much of it is needed.

The fingerprint of a set of unit deltas is the per-coordinate z-score of the
signed mean: z_i = mean(u_i) / (sd(u_i)/sqrt(m)). Positive z_i means the LATER
fact sits higher than the earlier one in coordinate i, consistently across the
set. Every panel compares conflict pairs with their cosine-matched controls at
the same n, so a coordinate that lights up is lighting up for conflict-ness,
not for similarity.

Nothing here is new measurement; every number is recomputed from the same
vectors and controls M7b used (same seed), then drawn differently.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.config import get_config                                       # noqa: E402
from hnav.core.geometry import ABTTWhitening                             # noqa: E402
from hnav.stage0.m7_delta_geometry import (                              # noqa: E402
    CALIBRATION, DATA, build_controls, deltas, load_store,
)
from hnav.stage0.m7b_dimension_profile import per_dim                    # noqa: E402

OUT = REPO / "stage0_results" / "delta_geometry" / "dims"
RED, BLUE, GREY, CTRL = "#c0392b", "#2471a3", "#b0b7bc", "#2c3e50"


def fingerprint(store, V, rng, caliper):
    gram = (V @ V.T).astype(np.float32)
    np.fill_diagonal(gram, -2.0)
    conflict = store.conflicts
    ctrl, ps = build_controls(store, conflict, gram, rng, caliper)
    partner = np.array(ps["cos_match"]["matched_target_idx"], dtype=int)
    _, _, U_c = deltas(V, conflict)
    _, _, U_k = deltas(V, ctrl["cos_matched"])
    U_cm = U_c[partner]
    return {"U_c": U_c, "U_cm": U_cm, "U_k": U_k, "partner": partner,
            "pd_c": per_dim(U_c), "pd_cm": per_dim(U_cm), "pd_k": per_dim(U_k),
            "rels": [store.rel[i] for i, _ in conflict],
            "frac_pos_cm": (U_cm > 0).mean(axis=0), "frac_pos_k": (U_k > 0).mean(axis=0)}


def heldout_fingerprint_score(U_c, U_k, partner, rng, k, n_rep=20):
    """Score every held-out pair on the fingerprint fitted on the other half."""
    m = U_c.shape[0]
    have = np.full(m, -1); have[partner] = np.arange(len(partner))
    sc_c, sc_k = [], []
    for _ in range(n_rep):
        perm = rng.permutation(m); A, B = perm[: m // 2], perm[m // 2:]
        z = per_dim(U_c[A])["z"]
        dims = np.argsort(-np.abs(z))[:k]
        w = np.sign(z[dims])
        Bm = B[have[B] >= 0]
        sc_c.append(U_c[Bm][:, dims] @ w)
        sc_k.append(U_k[have[Bm]][:, dims] @ w)
    return np.concatenate(sc_c), np.concatenate(sc_k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["sh_6k", "sh_64k"])
    ap.add_argument("--whitening-artifact",
                    default=str(REPO / "stage0_results/abtt/abtt_whitening_D128.json"))
    ap.add_argument("--caliper", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    cfg = get_config()
    cfg.require_not_live()
    outdir = pathlib.Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {it["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
             .replace("factconsolidation_", ""): it for it in data}
    blob = json.loads(pathlib.Path(args.whitening_artifact).read_text(encoding="utf-8"))
    if any(x not in CALIBRATION for x in blob.get("fit_subsets", [])):
        print("REFUSED: whitening fitted on held-out data", file=sys.stderr); return 2
    w = ABTTWhitening.from_dict(blob["whitening"])

    F = {}
    for s in args.subsets:
        st = load_store(s, items[s], cfg, False, None)
        F[s] = {}
        for space in ("raw", "abtt"):
            V = st.V if space == "raw" else np.asarray(w.transform(st.V), dtype=np.float64)
            F[s][space] = fingerprint(st, V, np.random.default_rng(args.seed), args.caliper)
            print(f"{s} {space}: {len(st.conflicts)} conflicts, "
                  f"{len(F[s][space]['partner'])} matched")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.22, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    written = []

    def save(fig, name):
        fig.savefig(outdir / name, bbox_inches="tight"); plt.close(fig); written.append(name)

    main_s = "sh_64k" if "sh_64k" in F else args.subsets[-1]
    d = F[main_s]["raw"]["U_c"].shape[1]

    # ── 1. THE FINGERPRINT: where in the 2560 coordinates, and which sign ──
    for space in ("raw", "abtt"):
        G = F[main_s][space]
        z_c, z_k = G["pd_cm"]["z"], G["pd_k"]["z"]
        top = np.argsort(-np.abs(z_c))[: args.top]
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.1, 1.3], width_ratios=[3, 1],
                              hspace=0.55, wspace=0.18)

        # 1a Manhattan: |z| per coordinate, conflict vs matched control
        a = fig.add_subplot(gs[0, :])
        x = np.arange(d)
        a.vlines(x, 0, np.abs(z_k), color=GREY, lw=0.6, label=f"matched non-conflict control (n={len(G['U_k'])})")
        pos, neg = z_c > 0, z_c < 0
        a.vlines(x[pos], 0, np.abs(z_c[pos]), color=RED, lw=0.6, alpha=0.9,
                 label="conflict, later fact HIGHER in this coordinate")
        a.vlines(x[neg], 0, np.abs(z_c[neg]), color=BLUE, lw=0.6, alpha=0.9,
                 label="conflict, later fact LOWER in this coordinate")
        a.axhline(4, color="k", lw=0.8, ls="--"); a.text(d * 0.995, 4.3, "|z| = 4", ha="right", fontsize=7)
        a.scatter(top, np.abs(z_c[top]), s=14, color="k", zorder=5, label=f"top {args.top}")
        a.set_xlim(0, d); a.set_ylim(0, max(np.abs(z_c).max(), np.abs(z_k).max()) * 1.08)
        a.set_xlabel("embedding coordinate (0 … 2559)")
        a.set_ylabel("|z| of the signed mean Δ")
        n_c, n_k = int((np.abs(z_c) > 4).sum()), int((np.abs(z_k) > 4).sum())
        a.set_title(f"{main_s} · {space} · the conflict fingerprint across all coordinates — "
                    f"{n_c} coordinates above |z|=4 for conflicts, {n_k} for their cosine-matched controls "
                    f"(same n = {len(G['U_k'])})", fontsize=9)
        a.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)

        # 1b the top coordinates, zoomed, with the control on the same axis
        a = fig.add_subplot(gs[1, :])
        xi = np.arange(len(top))
        a.bar(xi - 0.2, z_c[top], width=0.4, color=[RED if v > 0 else BLUE for v in z_c[top]],
              label="conflict")
        a.bar(xi + 0.2, z_k[top], width=0.4, color=CTRL, alpha=0.6, label="matched control, same coordinate")
        a.axhspan(-4, 4, color="k", alpha=0.06)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(xi); a.set_xticklabels([str(t) for t in top], rotation=90, fontsize=7)
        a.set_xlabel("coordinate index"); a.set_ylabel("z of signed mean Δ")
        a.set_title(f"the {args.top} strongest coordinates (ranked by conflict |z|): the control is flat "
                    f"in every one of them", fontsize=9)
        a.legend(fontsize=7, frameon=False, loc="upper right")

        # 1c sign agreement: what fraction of pairs go UP in each top coordinate
        a = fig.add_subplot(gs[2, 0])
        fp_c, fp_k = G["frac_pos_cm"][top], G["frac_pos_k"][top]
        a.bar(xi - 0.2, fp_c, width=0.4, color=[RED if v > 0.5 else BLUE for v in fp_c])
        a.bar(xi + 0.2, fp_k, width=0.4, color=CTRL, alpha=0.6)
        a.axhline(0.5, color="k", lw=0.8, ls="--")
        a.set_ylim(0, 1); a.set_xticks(xi); a.set_xticklabels([str(t) for t in top], rotation=90, fontsize=7)
        a.set_ylabel("fraction of pairs with Δᵢ > 0"); a.set_xlabel("coordinate index")
        a.set_title("read it per pair: share of pairs whose later fact is higher in the coordinate "
                    "(0.5 = coin flip)", fontsize=9)

        # 1d how concentrated is the fingerprint?
        a = fig.add_subplot(gs[2, 1])
        srt_c = np.sort(np.abs(z_c))[::-1]; srt_k = np.sort(np.abs(z_k))[::-1]
        cum_c = np.cumsum(srt_c ** 2) / (srt_c ** 2).sum()
        cum_k = np.cumsum(srt_k ** 2) / (srt_k ** 2).sum()
        a.plot(np.arange(1, d + 1), cum_c, color=RED, lw=1.6, label="conflict")
        a.plot(np.arange(1, d + 1), cum_k, color=CTRL, lw=1.2, label="matched control")
        a.plot(np.arange(1, d + 1), np.arange(1, d + 1) / d, "k:", lw=0.9, label="uniform")
        for k in (16, 64, 256):
            a.axvline(k, color="k", lw=0.5, alpha=0.4)
            a.text(k, 0.02, f"{k}: {cum_c[k - 1]:.0%}", rotation=90, fontsize=7, va="bottom")
        a.set_xscale("log"); a.set_xlabel("coordinates, strongest first"); a.set_ylabel("share of fingerprint energy Σz²")
        a.set_title("how spread out it is", fontsize=9); a.legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle(f"What conflict difference vectors look like, coordinate by coordinate  ·  {main_s}  ·  {space} space",
                     fontsize=12, y=0.95)
        save(fig, f"fp1_fingerprint_{main_s}_{space}.png")

    # ── 2. REPRODUCIBILITY: is it the same fingerprint on sh_6k, sh_64k, raw, ABTT? ──
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    pairs = []
    if len(args.subsets) >= 2:
        s0, s1 = args.subsets[0], args.subsets[-1]
        pairs.append((F[s0]["raw"]["pd_c"]["z"], F[s1]["raw"]["pd_c"]["z"], f"{s0} raw", f"{s1} raw"))
    pairs.append((F[main_s]["raw"]["pd_c"]["z"], F[main_s]["abtt"]["pd_c"]["z"], f"{main_s} raw", f"{main_s} ABTT"))
    pairs.append((F[main_s]["raw"]["pd_cm"]["z"], F[main_s]["raw"]["pd_k"]["z"],
                  f"{main_s} conflict (n-matched)", f"{main_s} matched control"))
    for a, (za, zb, la, lb) in zip(ax, pairs):
        r = np.corrcoef(za, zb)[0, 1]
        lim = max(np.abs(za).max(), np.abs(zb).max()) * 1.05
        a.scatter(za, zb, s=4, alpha=0.35, color=CTRL)
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.plot([-lim, lim], [-lim, lim], "--", color=GREY, lw=0.8)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel(f"z per coordinate · {la}"); a.set_ylabel(f"z per coordinate · {lb}")
        ka, kb = set(np.argsort(-np.abs(za))[:256]), set(np.argsort(-np.abs(zb))[:256])
        jac = len(ka & kb) / len(ka | kb)
        same_sign = float((np.sign(za[list(ka & kb)]) == np.sign(zb[list(ka & kb)])).mean()) if ka & kb else float("nan")
        a.set_title(f"r = {r:+.3f}   top-256 overlap {jac:.0%}, same sign {same_sign:.0%}", fontsize=9)
    fig.suptitle("Is it the same fingerprint? Per-coordinate z compared across subsets, across spaces, "
                 "and against the control (last panel: no relation → no shared pattern)", fontsize=10)
    fig.tight_layout()
    save(fig, "fp2_reproducibility.png")

    # ── 3. WHAT IT DOES: one score per pair, held-out ──
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    for c, space in enumerate(("raw", "abtt")):
        G = F[main_s][space]
        for r_, k in enumerate((16, 256)):
            a = ax[r_][c]
            sc, sk = heldout_fingerprint_score(G["U_c"], G["U_k"], G["partner"],
                                               np.random.default_rng(args.seed), k)
            lo, hi = min(sc.min(), sk.min()), max(sc.max(), sk.max())
            bins = np.linspace(lo, hi, 60)
            a.hist(sk, bins=bins, density=True, color=CTRL, alpha=0.55, label="matched non-conflict control")
            a.hist(sc, bins=bins, density=True, color=RED, alpha=0.55, label="conflict (held out)")
            acc = float((sc > sk).mean())
            a.axvline(np.median(sc), color=RED, lw=1.2); a.axvline(np.median(sk), color=CTRL, lw=1.2)
            a.set_title(f"{main_s} · {space} · fingerprint of {k} coordinates → conflict scores higher "
                        f"than its own matched control in {acc:.0%} of pairs", fontsize=9)
            a.set_xlabel("fingerprint score  Σ sign(zᵢ)·Δᵢ  over the selected coordinates")
            a.set_ylabel("density")
            if r_ == 0 and c == 0:
                a.legend(fontsize=7, frameon=False)
    fig.suptitle("Projecting each pair onto the fingerprint (fitted on the other half of the conflicts). "
                 "Cosine is nearly equal inside every pair by construction.", fontsize=10)
    fig.tight_layout()
    save(fig, "fp3_pair_scores.png")

    # ── 4. BY RELATION: does every template use the same coordinates? ──
    G = F[main_s]["raw"]
    rels = np.array(G["rels"])
    uniq = [r for r, n in sorted(((r, (rels == r).sum()) for r in set(rels)), key=lambda t: -t[1]) if n >= 20][:14]
    top = np.argsort(-np.abs(G["pd_c"]["z"]))[:60]
    M = np.array([per_dim(G["U_c"][rels == r])["z"][top] for r in uniq])
    fig, a = plt.subplots(figsize=(15, 5.2))
    v = np.nanmax(np.abs(M))
    im = a.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
    a.set_yticks(range(len(uniq)))
    a.set_yticklabels([f"{r.replace('|', ' __ ').strip()[:38]}  (n={(rels == r).sum()})" for r in uniq], fontsize=7.5)
    a.set_xticks(range(len(top))); a.set_xticklabels([str(t) for t in top], rotation=90, fontsize=6.5)
    a.set_xlabel("the 60 strongest fingerprint coordinates (all relations pooled)")
    a.grid(False)
    cb = plt.colorbar(im, ax=a, fraction=0.02, pad=0.01); cb.set_label("z within the relation")
    a.set_title(f"{main_s} · raw · the fingerprint split by relation template: columns that are the same colour "
                f"down the whole column are shared across templates; patchy columns belong to one template",
                fontsize=9)
    fig.tight_layout()
    save(fig, f"fp4_by_relation_{main_s}.png")

    print("figures: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
