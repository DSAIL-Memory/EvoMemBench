#!/usr/bin/env python3
"""M7b - per-dimension profile of supersession difference vectors.  [exploratory]

Question
--------
Cosine similarity collapses a pair of 2560-dimensional vectors to ONE number:
the sum over dimensions of ``a_i * b_i``. Two pairs with the same cosine can
differ in every coordinate. Does the conflict / non-conflict distinction live
in *which* dimensions the difference vector ``delta = v_later - v_earlier``
occupies, or in how it is spread across them - information cosine throws away?

Two layers, deliberately kept apart:

1. **The literal request**: ten conflict pairs and ten cosine-matched
   non-conflict pairs, their difference vectors drawn coordinate by coordinate,
   so the eye can see which dimensions are active. This is a *picture*, not
   evidence. Ten vectors in 2560 dimensions will always look patterned.

2. **The population**: the same per-dimension quantities over every conflict
   pair in the subset against every matched control, with split-half held-out
   evaluation so that a dimension picked on one half has to earn its keep on
   the other. Only this layer supports a claim.

What "more information than cosine" has to mean here
----------------------------------------------------
Inside the cosine-matched control set the cosine is nearly uninformative:
each control sits within 0.02 of its conflict partner. Nearly, not exactly -
non-conflict pairs are scarce at high cosine, so the control usually lands a
little BELOW its conflict and cosine still ranks the conflict first ~90% of
the time on a residual gap of ~0.01. The clean test is therefore stratified
by that residual gap: in the bin where the control's cosine is HIGHER than
the conflict's, cosine is wrong by definition, and anything that still picks
the conflict there is information cosine does not have.

Nothing is trained in the classifier sense. The only fitted objects are a
ranking of dimensions and a sign per dimension, fitted on one half of the
conflict pairs and scored on the other, and additionally on a
**relation-disjoint** split so a dimension that merely encodes a relation
template cannot pass.
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

from hnav.adapters.mab_adapter import explode_facts                      # noqa: E402
from hnav.config import get_config                                       # noqa: E402
from hnav.core.geometry import ABTTWhitening                             # noqa: E402
from hnav.stage0.m4_marginal_diff_test import auc                        # noqa: E402
from hnav.stage0.m7_delta_geometry import (                              # noqa: E402
    CALIBRATION, DATA, build_controls, deltas, describe, load_store,
)

OUT = REPO / "stage0_results" / "delta_geometry" / "dims"
SPACES = ("raw", "abtt")
KS = (1, 4, 16, 64, 256, 1024, 2560)
COL = {"conflict": "#c0392b", "control": "#2c3e50", "random": "#95a5a6",
       "same_relation": "#2980b9", "same_subject": "#16a085"}


# ── per-dimension statistics ─────────────────────────────────────────────────
def per_dim(U: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinate-wise summary of a set of unit difference vectors (m x d)."""
    m = U.shape[0]
    mean = U.mean(axis=0)
    sd = U.std(axis=0, ddof=1) if m > 1 else np.ones(U.shape[1])
    return {
        "mean": mean,
        # t-like statistic of the signed mean per coordinate
        "z": mean / np.maximum(sd / np.sqrt(max(m, 1)), 1e-12),
        # share of the (unit) energy each coordinate carries, averaged
        "energy": (U ** 2).mean(axis=0),
        # |mean sign|: 1 when every pair agrees on the sign, ~0.8/sqrt(m) under
        # the null of independent random signs
        "sign_consistency": np.abs(np.sign(U).mean(axis=0)),
    }


def concentration(U: np.ndarray) -> dict[str, np.ndarray]:
    """How each unit delta spreads its energy across coordinates.

    ``effective_dims`` = 1 / sum(u_i^4)  (= d for a perfectly flat vector, 1
    for a one-hot); ``top10_share`` = energy in the ten largest coordinates;
    ``l1_over_l2`` = ||u||_1 (<= sqrt(d), equality when flat). None of these is
    a function of the pair cosine, so at matched cosine they are genuinely
    extra information.
    """
    e = U ** 2
    srt = -np.sort(-e, axis=1)
    return {
        "effective_dims": 1.0 / np.maximum((e ** 2).sum(axis=1), 1e-300),
        "top10_share": srt[:, :10].sum(axis=1),
        "top64_share": srt[:, :64].sum(axis=1),
        "l1_over_l2": np.abs(U).sum(axis=1),
        "max_abs": np.abs(U).max(axis=1),
    }


def paired_signflip(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                    n_perm: int = 4000) -> dict:
    """Paired test of a per-pair scalar: conflict_j vs its own matched
    control_j. The null randomly swaps which member of each pair is called the
    conflict. Returns the observed mean difference, its null z, and the
    two-sided permutation p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d.size
    if m < 2:
        return {}
    obs = float(d.mean())
    null = np.array([(d * rng.choice((-1.0, 1.0), size=m)).mean()
                     for _ in range(n_perm)])
    sd = float(null.std())
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"n": int(m), "mean_conflict": float(np.mean(a)),
            "mean_control": float(np.mean(b)), "diff": obs,
            "z": obs / sd if sd > 0 else float("nan"), "p_perm": p,
            "frac_conflict_higher": float((d > 0).mean())}


# ── where does the cosine itself come from, coordinate by coordinate ─────────
def cosine_anatomy(V: np.ndarray, pairs: list[tuple[int, int]], rng,
                   n_random: int = 20000) -> dict:
    """Per-coordinate contribution ``a_i * b_i`` to the inner product, for the
    given pairs and for random pairs. If a handful of coordinates supply most
    of the 0.60 that unrelated facts share, cosine's 'single perspective' is
    even narrower than one number suggests: it is one number dominated by a
    few coordinates."""
    n = V.shape[0]
    a = rng.integers(0, n, n_random)
    b = rng.integers(0, n, n_random)
    keep = a != b
    rand = (V[a[keep]] * V[b[keep]]).mean(axis=0)
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    conf = (V[pa] * V[pb]).mean(axis=0)
    mu = V.mean(axis=0)

    def cum_share(x):
        s = -np.sort(-x)
        c = np.cumsum(s) / max(float(x.sum()), 1e-12)
        return {f"top{k}": float(c[k - 1]) for k in (1, 4, 16, 64, 256) if k <= len(c)}

    return {
        "random_pair_cos_mean": float(rand.sum()),
        "conflict_pair_cos_mean": float(conf.sum()),
        "random_contrib_share": cum_share(rand),
        "conflict_contrib_share": cum_share(conf),
        "mean_vector_energy_share": cum_share(mu ** 2),
        "mean_vector_norm": float(np.linalg.norm(mu)),
        "_random_contrib": rand, "_conflict_contrib": conf, "_mu": mu,
    }


# ── held-out sparse scores ───────────────────────────────────────────────────
def split_indices(m: int, rels: list[str], rng, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        perm = rng.permutation(m)
        return perm[: m // 2], perm[m // 2:]
    # relation-disjoint: whole relations go to one side or the other
    uniq = list(dict.fromkeys(rels))
    rng.shuffle(uniq)
    side, count = {}, [0, 0]
    for r in uniq:
        s = int(count[1] < count[0])
        side[r] = s
        count[s] += sum(1 for x in rels if x == r)
    A = np.array([i for i, r in enumerate(rels) if side[r] == 0])
    B = np.array([i for i, r in enumerate(rels) if side[r] == 1])
    return A, B


def heldout_scores(U_conf: np.ndarray, U_ctrl: np.ndarray, partner: np.ndarray,
                   cos_conf: np.ndarray, cos_ctrl: np.ndarray, rels: list[str],
                   rng, mode: str, n_rep: int = 20) -> dict:
    """Fit a sparse coordinate pattern on half the conflicts, evaluate on the
    other half against cosine-matched controls.

    ``partner[j]`` is the index into ``U_conf`` of the conflict that
    ``U_ctrl[j]`` was matched to. Two pattern families:

    sign   : keep the k coordinates with the largest |z| of the signed mean on
             the fit half; score = sum over them of consensus_sign_i * u_i.
             Asks: do conflict deltas point the same way in specific
             coordinates?
    energy : keep the k coordinates where the fit-half conflicts carry more
             energy than their matched controls; score = energy in them.
             Asks: are specific coordinates *active* for conflicts, sign aside?

    Two read-outs, both on pairs the fit never saw:
      auc            conflict (held-out half) vs ALL matched controls
      paired_acc     among held-out conflicts that have a matched control,
                     fraction whose score beats their own control's. Cosine
                     is ~0.5 here by construction - that is the comparison.
    Averaged over ``n_rep`` random splits so one lucky split cannot carry it.
    """
    m, d = U_conf.shape
    have = np.full(m, -1)
    have[partner] = np.arange(len(partner))            # conflict -> control row
    out = {k: {"sign": {"auc": [], "paired_acc": []},
               "energy": {"auc": [], "paired_acc": []}} for k in KS}
    cos_paired = []
    # Matching is only good to the caliper, and the residual gap is biased:
    # non-conflict pairs are scarce at high cosine, so the control usually
    # lands BELOW its conflict and cosine still "wins" a paired comparison.
    # So every held-out matched pair is also binned by its residual cosine
    # gap (conflict - control); the bin at gap <= 0 is where cosine is wrong
    # or silent, and it is the only bin that tests "beyond cosine" cleanly.
    GAP_BINS = ((-1.0, 0.0), (0.0, 0.005), (0.005, 0.01), (0.01, 0.021))
    strat = {k: {b: {"n": 0, "sign_correct": 0, "energy_correct": 0}
                 for b in GAP_BINS} for k in KS}
    for _ in range(n_rep):
        A, B = split_indices(m, rels, rng, mode)
        if len(A) < 5 or len(B) < 5:
            continue
        sA = per_dim(U_conf[A])
        consensus = np.sign(sA["mean"])
        order_sign = np.argsort(-np.abs(sA["z"]))
        # energy contrast on the fit half, against that half's own controls
        ctrlA = have[A][have[A] >= 0]
        eA_conf = (U_conf[A] ** 2).mean(axis=0)
        eA_ctrl = (U_ctrl[ctrlA] ** 2).mean(axis=0) if len(ctrlA) else np.zeros(d)
        order_energy = np.argsort(-(eA_conf - eA_ctrl))

        B_m = B[have[B] >= 0]                           # held-out with partner
        ctrlB = have[B_m]
        gap = cos_conf[B_m] - cos_ctrl[ctrlB]
        cos_paired.append(float((gap > 0).mean()))
        y = np.r_[np.ones(len(B)), np.zeros(U_ctrl.shape[0])]
        for k in KS:
            dims = order_sign[:k]
            s = lambda X: X[:, dims] @ consensus[dims]                 # noqa: E731
            out[k]["sign"]["auc"].append(auc(y, np.r_[s(U_conf[B]), s(U_ctrl)]))
            win_s = s(U_conf[B_m]) > s(U_ctrl[ctrlB])
            out[k]["sign"]["paired_acc"].append(float(win_s.mean()))
            dims = order_energy[:k]
            e = lambda X: (X[:, dims] ** 2).sum(axis=1)               # noqa: E731
            if k >= d:
                # every unit vector has total energy 1: the score is a tie
                # for all pairs and the family is undefined here
                out[k]["energy"]["auc"].append(float("nan"))
                out[k]["energy"]["paired_acc"].append(float("nan"))
                win_e = np.zeros(len(B_m), dtype=bool)
            else:
                out[k]["energy"]["auc"].append(auc(y, np.r_[e(U_conf[B]), e(U_ctrl)]))
                win_e = e(U_conf[B_m]) > e(U_ctrl[ctrlB])
                out[k]["energy"]["paired_acc"].append(float(win_e.mean()))
            for b in GAP_BINS:
                sel = (gap > b[0]) & (gap <= b[1])
                strat[k][b]["n"] += int(sel.sum())
                strat[k][b]["sign_correct"] += int(win_s[sel].sum())
                strat[k][b]["energy_correct"] += int(win_e[sel].sum())
    res = {"mode": mode, "n_rep": len(cos_paired),
           "cosine_paired_acc": describe(np.array(cos_paired)), "k": {},
           "by_cos_gap": {}}
    for k in KS:
        res["k"][k] = {fam: {stat: describe(np.array(v))
                             for stat, v in fams.items()}
                       for fam, fams in out[k].items()}
        res["by_cos_gap"][k] = {
            f"({b[0]:g},{b[1]:g}]": {
                "n_pair_evaluations": v["n"],
                "sign_acc": v["sign_correct"] / v["n"] if v["n"] else None,
                "energy_acc": (v["energy_correct"] / v["n"]
                               if v["n"] and k < d else None)}
            for b, v in strat[k].items()}
    return res


# ── one subset × space ───────────────────────────────────────────────────────
def analyse(store, V: np.ndarray, rng, caliper: float, n_show: int) -> tuple[dict, dict]:
    gram = (V @ V.T).astype(np.float32)
    np.fill_diagonal(gram, -2.0)
    conflict = store.conflicts
    ctrl, pair_space = build_controls(store, conflict, gram, rng, caliper)
    partner = np.array(pair_space["cos_match"]["matched_target_idx"], dtype=int)

    cos_c, _, U_c = deltas(V, conflict)
    cos_k, _, U_k = deltas(V, ctrl["cos_matched"])

    def slot_kind(a, b):
        return "|".join(("same_rel" if store.rel[a] == store.rel[b] else "diff_rel",
                         "same_subj" if store.subj[a] == store.subj[b] else "diff_subj",
                         "same_obj" if store.obj[a] == store.obj[b] else "diff_obj"))
    from collections import Counter
    composition = Counter(slot_kind(a, b) for a, b in ctrl["cos_matched"])
    rels = [store.rel[i] for i, _ in conflict]

    pd_c, pd_k = per_dim(U_c), per_dim(U_k)
    pd_cm = per_dim(U_c[partner])                  # same n as the control
    cc_c, cc_k = concentration(U_c), concentration(U_k)
    # matched, paired comparison of every concentration statistic
    paired = {name: paired_signflip(cc_c[name][partner], cc_k[name], rng)
              for name in cc_c}
    others = {}
    for name in ("same_relation", "same_subject", "random"):
        _, _, U_o = deltas(V, ctrl[name])
        if U_o.shape[0]:
            others[name] = {k: describe(v) for k, v in concentration(U_o).items()}

    anatomy = cosine_anatomy(V, conflict, rng)
    held = {mode: heldout_scores(U_c, U_k, partner, cos_c, cos_k, rels, rng, mode)
            for mode in ("random", "relation_disjoint")}

    # the ten pairs for the picture: first ten matched conflicts, deterministic
    show = np.arange(min(n_show, len(partner)))
    ten = {"conflict_idx": [conflict[partner[j]] for j in show],
           "control_idx": [ctrl["cos_matched"][j] for j in show],
           "conflict_text": [(store.text[a], store.text[b]) for a, b in
                             (conflict[partner[j]] for j in show)],
           "control_text": [(store.text[a], store.text[b]) for a, b in
                            (ctrl["cos_matched"][j] for j in show)],
           "conflict_cos": [float(cos_c[partner[j]]) for j in show],
           "control_cos": [float(cos_k[j]) for j in show]}

    m = U_c.shape[0]
    res = {
        "n_facts": store.n, "n_conflict": m, "n_matched": int(len(partner)),
        "dim": int(V.shape[1]),
        "sign_consistency": {
            "conflict": describe(pd_c["sign_consistency"]),
            "control": describe(pd_k["sign_consistency"]),
            "null_expected": float(np.sqrt(2 / np.pi) / np.sqrt(m)),
            "n_dims_conflict_above_3sigma": int(
                (pd_c["sign_consistency"] > 3 / np.sqrt(m)).sum()),
            "n_dims_control_above_3sigma": int(
                (pd_k["sign_consistency"] > 3 / np.sqrt(U_k.shape[0])).sum()),
            "n_dims_conflict_abs_z_above_4": int((np.abs(pd_c["z"]) > 4).sum()),
            "n_dims_control_abs_z_above_4": int((np.abs(pd_k["z"]) > 4).sum()),
            # the n-matched comparison: same number of pairs on both sides
            "n_matched_pairs": int(len(partner)),
            "n_dims_conflict_matched_abs_z_above_4": int((np.abs(pd_cm["z"]) > 4).sum()),
            "conflict_matched": describe(pd_cm["sign_consistency"]),
            "n_dims_conflict_matched_above_3sigma": int(
                (pd_cm["sign_consistency"] > 3 / np.sqrt(len(partner))).sum()),
        },
        "energy_profile": {
            # how flat is the AVERAGE energy profile across coordinates
            "conflict_effective_dims_of_mean_profile": float(
                1 / (pd_cm["energy"] ** 2).sum() * pd_cm["energy"].sum() ** 2),
            "control_effective_dims_of_mean_profile": float(
                1 / (pd_k["energy"] ** 2).sum() * pd_k["energy"].sum() ** 2),
            "top64_share_conflict": float(-np.sort(-pd_cm["energy"])[:64].sum()),
            "top64_share_control": float(-np.sort(-pd_k["energy"])[:64].sum()),
            "pearson_conflict_vs_control_profiles": float(
                np.corrcoef(pd_cm["energy"], pd_k["energy"])[0, 1]),
        },
        "concentration": {
            "conflict": {k: describe(v) for k, v in cc_c.items()},
            "control": {k: describe(v) for k, v in cc_k.items()},
            "paired_conflict_vs_matched_control": paired,
            "other_controls": others,
        },
        "cosine_anatomy": {k: v for k, v in anatomy.items() if not k.startswith("_")},
        "heldout": held,
        "ten_pairs": ten,
        "pair_space": {k: v for k, v in pair_space.items() if k != "cos_match"},
        "matched_control_composition": dict(composition.most_common()),
        "cos_match": {k: v for k, v in pair_space["cos_match"].items()
                      if k != "matched_target_idx"},
    }
    arrays = {"U_c": U_c, "U_k": U_k, "partner": partner, "pd_c": pd_c, "pd_k": pd_k,
              "pd_cm": pd_cm,
              "cc_c": cc_c, "cc_k": cc_k, "anatomy": anatomy, "show": show}
    return res, arrays


# ── figures ──────────────────────────────────────────────────────────────────
def figures(arrays: dict, results: dict, outdir: pathlib.Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    subsets = list(arrays)
    written = []

    def save(fig, name):
        fig.savefig(outdir / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    # A — the ten pairs, coordinate by coordinate ----------------------------
    for s in subsets:
        fig, ax = plt.subplots(5, 2, figsize=(13, 13), squeeze=False,
                               gridspec_kw={"height_ratios": [3, 1, 1, 1, 1.4]})
        for c, space in enumerate(SPACES):
            A = arrays[s][space]
            show, P = A["show"], A["partner"]
            Uc, Uk = A["U_c"][P[show]], A["U_k"][show]
            d = Uc.shape[1]
            v = float(max(np.abs(Uc).max(), np.abs(Uk).max()))
            a = ax[0][c]
            im = a.imshow(np.vstack([Uc, Uk]), aspect="auto", cmap="RdBu_r",
                          vmin=-v, vmax=v, interpolation="nearest")
            a.axhline(len(show) - 0.5, color="k", lw=1.2)
            a.set_yticks(range(2 * len(show)))
            a.set_yticklabels([f"conflict {j}" for j in range(len(show))]
                              + [f"control {j}" for j in range(len(show))], fontsize=6)
            a.set_title(f"{s} · {space} · unit Δ of 10 conflict pairs and their "
                        f"cosine-matched controls", fontsize=8)
            a.grid(False)
            plt.colorbar(im, ax=a, fraction=0.02, pad=0.01)
            x = np.arange(d)
            a = ax[1][c]
            a.bar(x, Uc.mean(axis=0), width=1.0, color=COL["conflict"])
            a.set_ylabel("mean Δ\n(10 conflict)", fontsize=7)
            a = ax[2][c]
            a.bar(x, Uk.mean(axis=0), width=1.0, color=COL["control"])
            a.set_ylabel("mean Δ\n(10 control)", fontsize=7)
            a = ax[3][c]
            a.bar(x, Uc.mean(axis=0) - Uk.mean(axis=0), width=1.0, color="#8e44ad")
            a.set_ylabel("difference\nof means", fontsize=7)
            a.set_xlabel("embedding coordinate")
            for r in (1, 2, 3):
                ax[r][c].set_xlim(0, d)
            a = ax[4][c]
            bins = np.linspace(-v, v, 121)
            a.hist(Uc.ravel(), bins=bins, density=True, histtype="step", lw=1.4,
                   color=COL["conflict"], label="10 conflict Δ (all coordinates)")
            a.hist(Uk.ravel(), bins=bins, density=True, histtype="step", lw=1.4,
                   color=COL["control"], label="10 control Δ (all coordinates)")
            a.set_yscale("log")
            a.set_xlabel("coordinate value of unit Δ")
            a.legend(frameon=False, fontsize=7)
        fig.suptitle(f"{s}: ten pairs, coordinate by coordinate. A PICTURE, not "
                     f"evidence - see fig B for the population", fontsize=9, y=1.0)
        fig.tight_layout()
        save(fig, f"figA_ten_pairs_{s}.png")

    # B — population per-dimension profiles ---------------------------------
    fig, ax = plt.subplots(len(subsets), 4, figsize=(16, 3.0 * len(subsets)),
                           squeeze=False)
    for r, s in enumerate(subsets):
        for space, ls in (("raw", "-"), ("abtt", "--")):
            A = arrays[s][space]
            pc, pk = A["pd_cm"], A["pd_k"]          # n-matched on both sides
            m_c, m_k = A["partner"].size, A["U_k"].shape[0]
            a = ax[r][0]
            a.plot(-np.sort(-pc["energy"]), color=COL["conflict"], ls=ls, lw=1.3,
                   label=f"conflict (n-matched) · {space}")
            a.plot(-np.sort(-pk["energy"]), color=COL["control"], ls=ls, lw=1.3,
                   label=f"matched control · {space}")
            a.set_xscale("log"); a.set_yscale("log")
            a.set_title(f"{s} · mean energy per coordinate, sorted", fontsize=8)
            a.set_xlabel("coordinate rank"); a.set_ylabel("share of unit energy")
            a.axhline(1 / pc["energy"].size, color="k", lw=0.7, ls=":")
            a = ax[r][1]
            a.plot(-np.sort(-pc["sign_consistency"]), color=COL["conflict"], ls=ls, lw=1.3)
            a.plot(-np.sort(-pk["sign_consistency"]), color=COL["control"], ls=ls, lw=1.3)
            if space == "raw":
                a.axhline(3 / np.sqrt(m_c), color=COL["conflict"], lw=0.7, ls=":",
                          label=r"3$\sigma$ null (conflict n)")
                a.axhline(3 / np.sqrt(m_k), color=COL["control"], lw=0.7, ls=":",
                          label=r"3$\sigma$ null (control n)")
            a.set_xscale("log")
            a.set_title(f"{s} · sign consistency per coordinate, sorted", fontsize=8)
            a.set_xlabel("coordinate rank"); a.set_ylabel("|mean sign|")
            a = ax[r][2]
            a.plot(-np.sort(-np.abs(pc["z"])), color=COL["conflict"], ls=ls, lw=1.3)
            a.plot(-np.sort(-np.abs(pk["z"])), color=COL["control"], ls=ls, lw=1.3)
            a.axhline(4, color="k", lw=0.7, ls=":")
            a.set_xscale("log")
            a.set_title(f"{s} · |z| of signed mean per coordinate, sorted", fontsize=8)
            a.set_xlabel("coordinate rank"); a.set_ylabel("|z|")
            a = ax[r][3]
            if space == "raw":
                a.scatter(pk["energy"], pc["energy"], s=3, alpha=0.4, color="#7f8c8d")
                lim = max(pc["energy"].max(), pk["energy"].max())
                a.plot([0, lim], [0, lim], "k--", lw=0.8)
                a.set_xscale("log"); a.set_yscale("log")
                rho = results[s]["raw"]["energy_profile"]["pearson_conflict_vs_control_profiles"]
                a.set_title(f"{s} · raw · per-coordinate energy, conflict vs control "
                            f"(r={rho:.3f})", fontsize=8)
                a.set_xlabel("matched control"); a.set_ylabel("conflict")
    ax[0][0].legend(frameon=False, fontsize=7); ax[0][1].legend(frameon=False, fontsize=7)
    fig.suptitle("Population per-coordinate profiles: every conflict pair vs every "
                 "cosine-matched control (solid raw, dashed ABTT)", fontsize=9, y=1.0)
    fig.tight_layout()
    save(fig, "figB_population_profiles.png")

    # C — concentration, paired ----------------------------------------------
    stats = (("effective_dims", "effective number of coordinates  1/Σu⁴"),
             ("top10_share", "energy in the 10 largest coordinates"),
             ("l1_over_l2", r"$\|u\|_1$  (flat = $\sqrt{d}$ = 50.6)"))
    fig, ax = plt.subplots(len(subsets), 3 * 2, figsize=(17, 2.8 * len(subsets)),
                           squeeze=False)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            for j, (key, lab) in enumerate(stats):
                a = ax[r][c * 3 + j]
                A = arrays[s][space]
                x_c, x_k = A["cc_c"][key][A["partner"]], A["cc_k"][key]
                lo, hi = min(x_c.min(), x_k.min()), max(x_c.max(), x_k.max())
                bins = np.linspace(lo, hi, 50)
                a.hist(x_c, bins=bins, density=True, histtype="step", lw=1.4,
                       color=COL["conflict"], label="conflict (matched)")
                a.hist(x_k, bins=bins, density=True, histtype="step", lw=1.4,
                       color=COL["control"], label="matched control")
                pr = results[s][space]["concentration"]["paired_conflict_vs_matched_control"][key]
                a.set_title(f"{s} · {space}\n{lab}\npaired z={pr['z']:+.1f}  "
                            f"conflict higher in {100 * pr['frac_conflict_higher']:.0f}%",
                            fontsize=7)
    ax[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("How a unit Δ spreads over coordinates - none of these is a function "
                 "of the cosine, and the control is cosine-matched pair by pair",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    save(fig, "figC_concentration.png")

    # D — anatomy of the cosine -----------------------------------------------
    fig, ax = plt.subplots(len(subsets), 3, figsize=(13, 2.9 * len(subsets)), squeeze=False)
    for r, s in enumerate(subsets):
        for space, ls in (("raw", "-"), ("abtt", "--")):
            an = arrays[s][space]["anatomy"]
            rc, cc, mu = an["_random_contrib"], an["_conflict_contrib"], an["_mu"]
            a = ax[r][0]
            a.plot(np.cumsum(-np.sort(-rc)) / rc.sum(), ls=ls, color=COL["random"], lw=1.4,
                   label=f"random pairs (cos={rc.sum():.3f}) · {space}")
            a.plot(np.cumsum(-np.sort(-cc)) / cc.sum(), ls=ls, color=COL["conflict"], lw=1.4,
                   label=f"conflict pairs (cos={cc.sum():.3f}) · {space}")
            a.plot(np.arange(1, rc.size + 1) / rc.size, "k:", lw=0.8)
            a.set_xscale("log"); a.set_ylim(0, 1.02)
            a.set_title(f"{s} · cumulative share of the inner product by coordinate", fontsize=8)
            a.set_xlabel("coordinates, largest contribution first"); a.set_ylabel("share of cos")
            a = ax[r][1]
            a.plot(-np.sort(-np.abs(mu)), ls=ls, lw=1.3, color="#8e44ad",
                   label=f"{space}  ‖μ‖={np.linalg.norm(mu):.3f}")
            a.set_xscale("log"); a.set_yscale("log")
            a.set_title(f"{s} · |mean vector| per coordinate, sorted", fontsize=8)
            a.set_xlabel("coordinate rank"); a.set_ylabel("|μ_i|")
            if space == "raw":
                a = ax[r][2]
                a.scatter(rc, cc, s=3, alpha=0.4, color="#7f8c8d")
                a.set_xlabel("contribution to random-pair cos")
                a.set_ylabel("contribution to conflict-pair cos")
                a.set_title(f"{s} · raw · per coordinate", fontsize=8)
                lim = max(rc.max(), cc.max())
                a.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0][0].legend(frameon=False, fontsize=7); ax[0][1].legend(frameon=False, fontsize=7)
    fig.suptitle("Where the cosine comes from: per-coordinate contribution a_i·b_i", fontsize=9, y=1.0)
    fig.tight_layout()
    save(fig, "figD_cosine_anatomy.png")

    # E — held-out sparse scores ---------------------------------------------
    fig, ax = plt.subplots(2, len(subsets) * 2, figsize=(3.6 * len(subsets) * 2, 6.2), squeeze=False)
    for c0, s in enumerate(subsets):
        for c1, space in enumerate(SPACES):
            c = c0 * 2 + c1
            for r, mode in enumerate(("random", "relation_disjoint")):
                a = ax[r][c]
                h = results[s][space]["heldout"][mode]
                ks = [k for k in KS if k in h["k"]]
                for fam, col in (("sign", "#c0392b"), ("energy", "#2980b9")):
                    kk = [k for k in ks if h["k"][k][fam]["paired_acc"].get("mean") == h["k"][k][fam]["paired_acc"].get("mean")]
                    y = [h["k"][k][fam]["paired_acc"]["mean"] for k in kk]
                    e = [h["k"][k][fam]["paired_acc"]["sd"] for k in kk]
                    a.errorbar(kk, y, yerr=e, marker="o", ms=3, lw=1.3, color=col,
                               label=f"{fam} pattern, paired acc")
                    y2 = [h["k"][k][fam]["auc"]["mean"] for k in kk]
                    a.plot(kk, y2, marker="s", ms=2.5, lw=0.9, ls="--", color=col,
                           alpha=0.7, label=f"{fam} pattern, AUC vs all controls")
                cp = h["cosine_paired_acc"]
                a.axhspan(cp["mean"] - cp["sd"], cp["mean"] + cp["sd"], color="k",
                          alpha=0.12, label="cosine, same pairs")
                a.axhline(0.5, color="k", lw=0.7)
                a.set_xscale("log"); a.set_ylim(0.35, 1.0)
                a.set_title(f"{s} · {space} · split: {mode}", fontsize=8)
                a.set_xlabel("k coordinates kept (fitted on other half)")
                if c == 0:
                    a.set_ylabel("held-out separation")
    ax[0][0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.suptitle("Coordinate pattern fitted on one half, scored on held-out conflicts vs their "
                 "cosine-matched controls. Grey = cosine on the same pairs: still ~0.9, because the "
                 "residual 0.01 gap inside the caliper favours the conflict (see gap-stratified table)",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    save(fig, "figE_heldout_coordinates.png")
    return written


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=["sh_6k", "sh_64k"])
    ap.add_argument("--whitening-artifact",
                    default=str(REPO / "stage0_results/abtt/abtt_whitening_D128.json"))
    ap.add_argument("--caliper", type=float, default=0.02)
    ap.add_argument("--n-show", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    cfg = get_config()
    cfg.require_not_live()
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
             .replace("factconsolidation_", ""): item for item in data}
    blob = json.loads(pathlib.Path(args.whitening_artifact).read_text(encoding="utf-8"))
    held = [x for x in blob.get("fit_subsets", []) if x not in CALIBRATION]
    if held:
        print(f"REFUSED: whitening artifact fitted on held-out {held}", file=sys.stderr)
        return 2
    w = ABTTWhitening.from_dict(blob["whitening"])

    results, arrays = {}, {}
    for s in args.subsets:
        st = load_store(s, items[s], get_config(), args.embed, None)
        print(f"{s}: {st.n} facts, {len(st.conflicts)} conflict pairs, ns={st.namespace}")
        results[s], arrays[s] = {}, {}
        for space in SPACES:
            V = st.V if space == "raw" else np.asarray(w.transform(st.V), dtype=np.float64)
            rng = np.random.default_rng(args.seed)
            r, a = analyse(st, V, rng, args.caliper, args.n_show)
            results[s][space], arrays[s][space] = r, a
            sc, cc = r["sign_consistency"], r["concentration"]["paired_conflict_vs_matched_control"]
            hr = r["heldout"]["random"]["k"]; hd = r["heldout"]["relation_disjoint"]["k"]
            print(f"  {s:7s} {space:4s} matched={r['n_matched']}/{r['n_conflict']}  "
                  f"dims |z|>4: conflict={sc['n_dims_conflict_abs_z_above_4']} "
                  f"control={sc['n_dims_control_abs_z_above_4']}  "
                  f"eff-dims paired z={cc['effective_dims']['z']:+.1f}  "
                  f"top10 z={cc['top10_share']['z']:+.1f}")
            print(f"           held-out paired acc (sign, k=16/256/all): "
                  f"{hr[16]['sign']['paired_acc']['mean']:.3f}/"
                  f"{hr[256]['sign']['paired_acc']['mean']:.3f}/"
                  f"{hr[2560]['sign']['paired_acc']['mean']:.3f}   "
                  f"relation-disjoint: {hd[16]['sign']['paired_acc']['mean']:.3f}/"
                  f"{hd[256]['sign']['paired_acc']['mean']:.3f}/"
                  f"{hd[2560]['sign']['paired_acc']['mean']:.3f}   "
                  f"cosine: {r['heldout']['random']['cosine_paired_acc']['mean']:.3f}")
            g = r["heldout"]["random"]["by_cos_gap"]
            for k in (16, 256, 2560):
                print(f"           gap-stratified sign acc k={k:4d}: " + "  ".join(
                    f"{b}: {v['sign_acc']:.3f} (n={v['n_pair_evaluations']})"
                    if v["sign_acc"] is not None else f"{b}: -" for b, v in g[k].items()))

    payload = {"measurement": "M7b per-dimension profile of supersession deltas (exploratory)",
               "no_classifier": True, "no_threshold_derived": True,
               "whitening": {"source": args.whitening_artifact, "d": w.n_components,
                             "fingerprint": w.fingerprint(), "fit_subsets": blob["fit_subsets"]},
               "config": {"seed": args.seed, "caliper": args.caliper, "ks": list(KS)},
               "subsets": results}
    (outdir / "m7b_dimension_profile.json").write_text(json.dumps(payload, indent=1),
                                                       encoding="utf-8")
    print(f"\nwrote {outdir / 'm7b_dimension_profile.json'}")
    if not args.no_figures:
        print("figures: " + ", ".join(figures(arrays, results, outdir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
