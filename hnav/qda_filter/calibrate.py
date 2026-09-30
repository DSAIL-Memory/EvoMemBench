"""Stage 6 — conformal thresholds and the chi^2 tail check.  [QDA]

The conformal-calibration negatives were carved out of the fit split BEFORE
any fitting (seed 0) and touch nothing fit-side, so their score order
statistics give distribution-free FPR guarantees at any achievable alpha:
P(score_new > threshold_alpha) <= alpha for an exchangeable new negative.
Any alpha below 1/(n0_cal + 1) is EXTRAPOLATION — the order statistic simply
does not exist — and is labeled so, never claimed.

The chi^2 comparison tells the user whether the parametric tail can extend
below the conformal floor: under the Gaussian null the object-subspace
energy ||U_obj^T z||^2 of a whitened negative is chi^2_{k_obj}; the empirical
tail excess at the 1e-2 / 1e-3 quantiles measures how wrong that is where it
matters.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_thresholds(neg_scores: np.ndarray,
                         alphas=(1e-1, 1e-2, 1e-3, None)) -> dict:
    """Threshold_alpha = the ceil((n+1)(1-alpha))-th order statistic.
    ``None`` in alphas means the floor alpha = 1/(n+1)."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))
    n = len(s)
    floor = 1.0 / (n + 1)
    out = {"n0_cal": n, "alpha_floor": floor,
           "extrapolation_note": f"any alpha < {floor:.3e} is extrapolation, "
                                 "not a guarantee", "thresholds": {}}
    for a in alphas:
        alpha = floor if a is None else float(a)
        k = math.ceil((n + 1) * (1.0 - alpha))
        entry = {"alpha": alpha, "order_statistic_k": k,
                 "achievable": bool(k <= n)}
        if k <= n:
            entry["threshold"] = float(s[k - 1])
        else:
            entry["threshold"] = None
            entry["note"] = "k exceeds n0_cal: not achievable at this n"
        out["thresholds"]["alpha_floor" if a is None else f"{alpha:g}"] = entry
    return out


def chi2_tail_check(obj_energy_neg: np.ndarray, k_obj: int) -> dict:
    """Empirical ||U_obj^T z||^2 on calibration negatives vs chi^2_{k_obj}."""
    from scipy import stats

    x = np.sort(np.asarray(obj_energy_neg, dtype=np.float64))
    n = len(x)
    ks = stats.kstest(x, "chi2", args=(k_obj,))
    qs = [0.5, 0.9, 0.99, 0.999]
    qq = []
    for q in qs:
        if q > 1 - 1.0 / n:
            continue
        qq.append({"q": q, "empirical": float(np.quantile(x, q)),
                   "theoretical": float(stats.chi2.ppf(q, k_obj))})
    tail = {}
    for q, tag in ((1 - 1e-2, "1e-2"), (1 - 1e-3, "1e-3")):
        theo = float(stats.chi2.ppf(q, k_obj))
        if q <= 1 - 1.0 / n:
            emp = float(np.quantile(x, q))
            tail[tag] = {"empirical": emp, "theoretical": theo,
                         "excess_ratio": emp / theo}
        else:
            tail[tag] = {"theoretical": theo,
                         "note": f"empirical quantile not resolvable with "
                                 f"n={n}"}
    return {"n": n, "k_obj": k_obj,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "qq": qq, "tail_excess": tail,
            "verdict_hint": "excess_ratio >> 1 means the chi^2 p-value is "
                            "anticonservative at that FPR and must not "
                            "replace the conformal threshold"}
