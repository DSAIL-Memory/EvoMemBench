"""Stage 5 — evaluation, all metrics via the geometry_filter harness.  [QDA]

Every headline number is produced by the SAME functions that produced the
committed geometry_filter report (auroc/auprc/bootstrap/inverted-win/
tpr_at_fpr are imported, not re-derived), so rows here are directly
comparable to raw cosine 0.893 / ABTT-cosine 0.965 / CES 0.976 on the
balanced sh_64k set.
"""
from __future__ import annotations

import numpy as np

from hnav.geometry_filter.metrics import (auprc, auroc, bootstrap_ci,
                                          inverted_win_rate,
                                          paired_bootstrap_delta_auc)
from hnav.geometry_filter.run_nuisance_analysis import tpr_at_fpr

SEED = 20260824
N_BOOT = 1000
BAND = (0.87, 0.97)
FPRS = (1e-2, 1e-3, 1e-4)


def _band_mask(cos: np.ndarray) -> np.ndarray:
    return (cos >= BAND[0]) & (cos <= BAND[1])


def balanced_eval(b, scores: dict, baseline: str = "V0") -> dict:
    """Per-subset balanced (cosine-matched) eval + the 0.87–0.97 band, with
    CIs and paired deltas against the baseline method."""
    out = {}
    for sub in ("sh_6k", "sh_32k", "sh_64k"):
        m = b.in_eval & (b.subset == sub)
        y = b.y[m]
        band = _band_mask(b.cos[m])
        row = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
               "band_n_pos": int(y[band].sum()),
               "band_n_neg": int((~y[band]).sum()),
               "in_sample_for_fit": sub in ("sh_6k", "sh_32k"),
               "methods": {}}
        base = scores[baseline][m]
        for name, s_all in scores.items():
            s = s_all[m]
            sb, yb = s[band], y[band]
            e = {"auroc": auroc(s[y], s[~y]),
                 "band_auroc": auroc(sb[yb], sb[~yb]) if band.any() else None}
            if sub == "sh_64k":
                e["auroc_ci95"] = bootstrap_ci(auroc, s[y], s[~y],
                                               N_BOOT, SEED)
                if name != baseline:
                    e[f"delta_vs_{baseline}"] = paired_bootstrap_delta_auc(
                        s[y], s[~y], base[y], base[~y], N_BOOT, SEED)
                    sbb, ybb = s[band], y[band]
                    bb = base[band]
                    e[f"band_delta_vs_{baseline}"] = paired_bootstrap_delta_auc(
                        sbb[ybb], sbb[~ybb], bb[ybb], bb[~ybb], N_BOOT, SEED)
            row["methods"][name] = e
        out[sub] = row
    return out


def hard_task_eval(b, scores: dict, baseline: str = "V0") -> dict:
    """Confirmatory gold vs hard negatives + inverted-win vs campaign cos."""
    m = (b.y | b.hardneg) & ~b.cal
    y = b.y[m]
    cc = b.cos[m]
    out = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "methods": {}}
    base = scores[baseline][m]
    for name, s_all in scores.items():
        s = s_all[m]
        e = {"auroc": auroc(s[y], s[~y]), "auprc": auprc(s[y], s[~y]),
             "auroc_ci95": bootstrap_ci(auroc, s[y], s[~y], N_BOOT, SEED),
             "inverted_vs_campaign_cos": inverted_win_rate(
                 s[y], cc[y], s[~y], cc[~y])}
        if name != baseline:
            e[f"delta_vs_{baseline}"] = paired_bootstrap_delta_auc(
                s[y], s[~y], base[y], base[~y], N_BOOT, SEED)
        out["methods"][name] = e
    return out


def tail_eval(b, scores: dict) -> dict:
    """TPR at fixed FPR, seen vs unseen transitions (nuisance convention:
    the gold slice varies, the negative pool is all confirmatory hard
    negatives). G4: flags extrapolation when n_neg < 1/fpr."""
    conf_gold = np.flatnonzero(b.y & ~b.cal)
    seen = np.array([b.trans_keys[i] in b.cal_transitions for i in conf_gold])
    neg_idx = np.flatnonzero(b.hardneg & ~b.cal)
    out = {"n_neg": int(len(neg_idx)),
           "n_pos_seen": int(seen.sum()), "n_pos_unseen": int((~seen).sum()),
           "methods": {}}
    for name, s_all in scores.items():
        neg = s_all[neg_idx]
        e = {}
        for tag, gidx in (("seen", conf_gold[seen]),
                          ("unseen", conf_gold[~seen])):
            pos = s_all[gidx]
            for f in FPRS:
                e[f"tpr_{tag}_fpr_{f:g}"] = tpr_at_fpr(pos, neg, f)
                e[f"fpr_{f:g}_extrapolated"] = bool(len(neg_idx) < 1.0 / f)
        out["methods"][name] = e
    return out


def cosine_strata(b, scores: dict, a: str = "V2", v0: str = "V0") -> dict:
    """AUROC per 0.01 campaign-cosine bin on the sh_64k eval set. If the V2
    advantage vanishes within bins, the aggregate win was the cosine axis."""
    m = b.in_eval & (b.subset == "sh_64k")
    y = b.y[m]
    cos = b.cos[m]
    sa, s0 = scores[a][m], scores[v0][m]
    bins = np.round(np.arange(0.80, 1.001, 0.01), 2)
    rows = []
    for lo in bins[:-1]:
        sel = (cos >= lo) & (cos < lo + 0.01)
        n1, n0 = int(y[sel].sum()), int((~y[sel]).sum())
        row = {"bin_lo": float(lo), "n_pos": n1, "n_neg": n0,
               "flagged_small": bool(n1 < 5 or n0 < 5)}
        if n1 and n0:
            row[a] = auroc(sa[sel][y[sel]], sa[sel][~y[sel]])
            row[v0] = auroc(s0[sel][y[sel]], s0[sel][~y[sel]])
            row["delta"] = row[a] - row[v0]
        rows.append(row)
    ok = [r for r in rows if not r["flagged_small"] and a in r]
    return {"bins": rows,
            "mean_within_bin_delta_unflagged":
                float(np.mean([r["delta"] for r in ok])) if ok else None,
            "n_unflagged_bins": len(ok)}


def subspace_stability(Z_gold: np.ndarray, U_obj_ref: np.ndarray,
                       k_obj: int, n_boot: int = 50, seed: int = 42) -> dict:
    """Median largest principal angle (degrees) between U_obj refit on
    bootstrap resamples of the fit gold (W0 fixed) and the point estimate."""
    from .spectrum import nontrivial_spectrum

    rng = np.random.default_rng(seed)
    ks = sorted({1, 5, k_obj} & set(range(1, k_obj + 1)))
    angles = {k: [] for k in ks}
    for _ in range(n_boot):
        idx = rng.integers(0, len(Z_gold), len(Z_gold))
        _, vt = nontrivial_spectrum(Z_gold[idx])
        for k in ks:
            s = np.linalg.svd(U_obj_ref[:, :k].T @ vt[:k].T,
                              compute_uv=False)
            angles[k].append(np.degrees(np.arccos(np.clip(s.min(), 0, 1))))
    return {"n_boot": n_boot, "seed": seed,
            "median_largest_principal_angle_deg": {
                str(k): float(np.median(v)) for k, v in angles.items()}}


def relation_disjoint(b, n_perm_inner: int = 50) -> dict:
    """cal 2-fold relation-disjoint transfer for V2/V3: the WHOLE pipeline
    (halves, LW, W0, spectrum, ranks, weights) refit on one fold's
    calibration pairs, scored on the other fold's calibration gold vs hard
    negatives. Uses the committed relation_fold hash split."""
    from hnav.geometry_filter import data as gfdata

    from .fit import QDAModel
    from .spectrum import fit_spectrum, ledoit_wolf_whitener

    rels = sorted({r for r in b.relation if r is not None})
    fold_of = gfdata.relation_fold(rels, n_folds=2)
    rel_fold = np.array([fold_of.get(r, -1) for r in b.relation])
    out = {"folds": []}
    for f in (0, 1):
        fit_m = b.cal & (rel_fold == f)
        ev_m = b.cal & (rel_fold == 1 - f) & (b.y | b.hardneg)
        gold_idx = np.flatnonzero(fit_m & b.y)
        neg_idx = np.flatnonzero(fit_m & b.negative & ~b.conformal_neg)
        rng = np.random.default_rng(0)
        perm = neg_idx[rng.permutation(len(neg_idx))]
        half = len(perm) // 2
        Xa = b.D_t[perm[:half]].astype(np.float64)
        W0, _ = ledoit_wolf_whitener(Xa)
        Zg = b.whiten(W0, gold_idx)
        Zb = b.whiten(W0, perm[half:])
        spec = fit_spectrum(Zg, Zb, n_perm=n_perm_inner, seed=SEED)
        mu1 = Zg.mean(axis=0)
        mu0 = Zb.mean(axis=0)
        model = QDAModel(W0, spec, mu1, mu0, ordered_on=True)
        ev_idx = np.flatnonzero(ev_m)
        Ze = b.whiten(W0, ev_idx)
        y = b.y[ev_idx]
        v2 = model.core(Ze)
        v3 = v2 + model.ordered(Ze)
        out["folds"].append({
            "fit_fold": f, "n_fit_gold": int(len(gold_idx)),
            "n_eval_pos": int(y.sum()), "n_eval_neg": int((~y).sum()),
            "k_obj": spec["k_obj"], "k_subj": spec["k_subj"],
            "V2_auroc": auroc(v2[y], v2[~y]),
            "V3_auroc": auroc(v3[y], v3[~y]),
        })
    return out


def pipeline_permutation_null(b, n_repeats: int, n_perm_inner: int,
                              neg_subsample: int = 4000) -> dict:
    """Label-shuffle null of the WHOLE pipeline (halving, LW, whitening,
    parallel-analysis rank selection, weights, V2 scoring).

    Compute-budget notes (documented deviations, none touch the real run):
    inner parallel-analysis permutations are ``n_perm_inner`` (real run: 200);
    the null hard-task AUROC uses a fixed seed-7 subsample of confirmatory
    hard negatives, and the same subsample is used for the real comparison.
    """
    from .fit import QDAModel
    from .spectrum import fit_spectrum, ledoit_wolf_whitener

    fit_idx = np.flatnonzero(b.cal & (b.y | (b.negative & ~b.conformal_neg)))
    n1 = int(b.y[fit_idx].sum())

    bal_idx = np.flatnonzero(b.in_eval & (b.subset == "sh_64k"))
    y_bal = b.y[bal_idx]
    conf_pos = np.flatnonzero(b.y & ~b.cal)
    conf_neg = np.flatnonzero(b.hardneg & ~b.cal)
    sub_rng = np.random.default_rng(7)
    conf_neg_sub = conf_neg[sub_rng.choice(len(conf_neg),
                                           size=min(neg_subsample,
                                                    len(conf_neg)),
                                           replace=False)]

    def run_once(pseudo_gold_local: np.ndarray, seed: int) -> tuple:
        is_g = np.zeros(len(fit_idx), bool)
        is_g[pseudo_gold_local] = True
        gold_idx = fit_idx[is_g]
        neg_idx = fit_idx[~is_g]
        rng = np.random.default_rng(seed)
        perm = neg_idx[rng.permutation(len(neg_idx))]
        half = len(perm) // 2
        W0, _ = ledoit_wolf_whitener(b.D_t[perm[:half]].astype(np.float64))
        Zg = b.whiten(W0, gold_idx)
        Zb = b.whiten(W0, perm[half:])
        spec = fit_spectrum(Zg, Zb, n_perm=n_perm_inner, seed=seed)
        model = QDAModel(W0, spec, Zg.mean(0), Zb.mean(0), ordered_on=False)
        s_bal = model.core(b.whiten(W0, bal_idx))
        s_p = model.core(b.whiten(W0, conf_pos))
        s_n = model.core(b.whiten(W0, conf_neg_sub))
        return (auroc(s_bal[y_bal], s_bal[~y_bal]), auroc(s_p, s_n),
                spec["k_obj"], spec["k_subj"])

    real_local = np.flatnonzero(b.y[fit_idx])
    real = run_once(real_local, seed=999)

    null_bal, null_hard, null_kobj = [], [], []
    for rep in range(n_repeats):
        rng = np.random.default_rng(1000 + rep)
        pseudo = rng.choice(len(fit_idx), size=n1, replace=False)
        a_bal, a_hard, ko, _ = run_once(pseudo, seed=1000 + rep)
        null_bal.append(a_bal)
        null_hard.append(a_hard)
        null_kobj.append(ko)

    from hnav.geometry_filter.metrics import perm_pvalue
    return {
        "n_repeats": n_repeats, "n_perm_inner": n_perm_inner,
        "neg_subsample": int(len(conf_neg_sub)), "neg_subsample_seed": 7,
        "real_balanced_sh64k_auroc": real[0],
        "real_hard_subsample_auroc": real[1],
        "real_run_note": "recomputed inside this harness (n_perm_inner "
                         "envelopes) so real vs null share one code path; "
                         "the headline uses the full 200-perm envelopes",
        "null_balanced": {"max": float(np.max(null_bal)),
                          "q95": float(np.quantile(null_bal, 0.95)),
                          "mean": float(np.mean(null_bal))},
        "null_hard": {"max": float(np.max(null_hard)),
                      "q95": float(np.quantile(null_hard, 0.95)),
                      "mean": float(np.mean(null_hard))},
        "null_k_obj_mean": float(np.mean(null_kobj)),
        "p_balanced": perm_pvalue(real[0], null_bal),
        "p_hard": perm_pvalue(real[1], null_hard),
    }
