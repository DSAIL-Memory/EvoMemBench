#!/usr/bin/env python3
"""M4 — the H2 test: does marginal-diff geometry add information?  [T7]

On the **calibration split only** (``sh_6k`` + ``sh_32k``), two nested logistic
models with ``y = "this candidate is must_write"``::

    M_base : y ~ whole_blob_sim + qr_residual
    M_diff : y ~ whole_blob_sim + qr_residual + diff_sim + diff_novelty

Reports the likelihood-ratio test, ΔAUC with a subset-clustered bootstrap CI
(10,000 resamples), and calibration curves.

**H2 passes iff ΔAUC > 0 with a 95% CI excluding 0 AND LRT p < 0.01.** Both
conditions, stated before the run; neither is negotiable afterwards.

Two honesty notes that are printed with every result:

* The calibration split has **two** subsets, so a subset-clustered bootstrap
  resamples from two clusters and its CI is close to meaningless on its own. The
  protocol names subset clustering, so it is reported as primary — and a
  key-clustered bootstrap (thousands of clusters, and the level at which conflict
  pairs are actually dependent) is reported beside it. Disagreement between them
  is a finding, not a nuisance.
* In-sample ΔAUC is optimistic for the larger model by construction. A
  group-k-fold cross-validated ΔAUC is reported alongside it. If the two point in
  different directions, H2 has not passed whatever the CI says.

Input: ``m3_write_rows_<subset>.json`` and ``m3_write_outcomes_<subset>.json``
from M3. Run M3 with counterfactual labeling first.

    python hnav/stage0/m4_marginal_diff_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402

CALIBRATION = ("sh_6k", "sh_32k")
BASE_FEATURES = ("whole_blob_sim", "qr_residual")
DIFF_FEATURES = ("diff_sim", "diff_novelty")
ALPHA_LRT = 0.01


# ── metrics ──────────────────────────────────────────────────────────────────
def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; NaN when a class is empty."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def fit_logistic(X: np.ndarray, y: np.ndarray):
    """Unpenalized logistic regression on standardized features.

    Unpenalized because the likelihood-ratio test requires the maximum-likelihood
    fit; a penalty would shrink the larger model and bias the LRT toward the
    null. Standardization is for conditioning only and does not change the fit.
    """
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    model.fit(Z, y)
    return model, mu, sd


def predict(model, mu, sd, X: np.ndarray) -> np.ndarray:
    return model.predict_proba((X - mu) / sd)[:, 1]


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Observed rate vs mean predicted probability, by quantile bin."""
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, min(n_bins, max(len(order), 1))):
        if len(chunk) == 0:
            continue
        out.append({"n": int(len(chunk)),
                    "mean_predicted": float(p[chunk].mean()),
                    "observed_rate": float(np.asarray(y)[chunk].mean())})
    return out


# ── bootstrap ────────────────────────────────────────────────────────────────
def cluster_bootstrap_delta_auc(y, p_base, p_diff, clusters, n_boot=10000, seed=0):
    """Resample whole clusters with replacement; recompute ΔAUC each time.

    Clusters, not rows: rows inside a subset (or inside a (relation, subject)
    key) are dependent, and a row-level bootstrap would understate the CI.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in uniq}

    deltas = np.empty(n_boot)
    deltas[:] = np.nan
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[c] for c in drawn])
        yb = np.asarray(y)[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        deltas[b] = auc(yb, np.asarray(p_diff)[idx]) - auc(yb, np.asarray(p_base)[idx])

    good = deltas[np.isfinite(deltas)]
    if good.size == 0:
        return {"n_clusters": int(len(uniq)), "n_valid": 0,
                "lo": float("nan"), "hi": float("nan"), "mean": float("nan")}
    return {"n_clusters": int(len(uniq)), "n_valid": int(good.size),
            "lo": float(np.percentile(good, 2.5)),
            "hi": float(np.percentile(good, 97.5)),
            "mean": float(good.mean())}


def cv_delta_auc(X_base, X_diff, y, groups, n_splits=5, seed=0) -> dict:
    """Group-k-fold cross-validated ΔAUC — out-of-sample, so it cannot be
    inflated by the larger model simply having more parameters."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return {"n_splits": 0, "delta_auc": float("nan"), "note": "too few groups"}

    p_base = np.zeros(len(y))
    p_diff = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=n_splits).split(X_base, y, groups):
        if len(np.unique(np.asarray(y)[train])) < 2:
            continue
        m1, mu1, sd1 = fit_logistic(X_base[train], y[train])
        m2, mu2, sd2 = fit_logistic(X_diff[train], y[train])
        p_base[test] = predict(m1, mu1, sd1, X_base[test])
        p_diff[test] = predict(m2, mu2, sd2, X_diff[test])

    a_base, a_diff = auc(y, p_base), auc(y, p_diff)
    return {"n_splits": int(n_splits), "auc_base": a_base, "auc_diff": a_diff,
            "delta_auc": a_diff - a_base}


# ── data ─────────────────────────────────────────────────────────────────────
def load_rows(out_dir: Path, subsets) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for name in subsets:
        f_rows = out_dir / f"m3_write_rows_{name}.json"
        f_out = out_dir / f"m3_write_outcomes_{name}.json"
        if not f_rows.exists() or not f_out.exists():
            missing.append(name)
            continue
        outcomes = json.loads(f_out.read_text())
        for r in json.loads(f_rows.read_text()):
            o = outcomes.get(r["id"])
            if o is None:
                continue
            r = dict(r)
            r["subset"] = name
            r["counterfactual_class"] = o["counterfactual_class"]
            r["y"] = int(o["counterfactual_class"] == "must_write")
            rows.append(r)
    return rows, missing


def build_matrices(rows: list[dict]):
    def col(name):
        return np.array([np.nan if r.get(name) is None else float(r[name])
                         for r in rows])

    feats = {n: col(n) for n in BASE_FEATURES + DIFF_FEATURES}
    ok = np.ones(len(rows), dtype=bool)
    for v in feats.values():
        ok &= np.isfinite(v)

    X_base = np.column_stack([feats[n][ok] for n in BASE_FEATURES])
    X_diff = np.column_stack([feats[n][ok] for n in BASE_FEATURES + DIFF_FEATURES])
    y = np.array([r["y"] for r in rows])[ok]
    subsets = np.array([r["subset"] for r in rows])[ok]
    keys = np.array(["|".join(r["key"]) if r["key"] else r["id"] for r in rows])[ok]
    return X_base, X_diff, y, subsets, keys, int((~ok).sum())


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    from scipy import stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=list(CALIBRATION),
                    help="calibration split only; changing this is a protocol breach")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()

    print("=" * 74)
    print(" M4 — H2: does marginal-diff geometry add information?")
    print(f"   split: {args.subsets}   (nothing may be fit on the confirmatory split)")
    print("=" * 74)

    if set(args.subsets) - set(CALIBRATION):
        print(f"\n REFUSING: {sorted(set(args.subsets) - set(CALIBRATION))} is not in "
              f"the calibration split {CALIBRATION}.")
        return 2

    rows, missing = load_rows(cfg.out_dir, args.subsets)
    if missing:
        print(f"\n missing M3 output for {missing}. Run m3_headroom.py "
              f"(with counterfactual labeling) first.")
        return 1
    if not rows:
        print("\n no labeled candidates found.")
        return 1

    X_base, X_diff, y, subsets, keys, n_dropped = build_matrices(rows)
    print(f"\n n={len(y)} labeled candidates  ({n_dropped} dropped for missing "
          f"features)   positives={int(y.sum())} ({y.mean():.3f})")

    if y.sum() == 0 or y.sum() == len(y):
        print("\n y has no variance: every candidate has the same counterfactual\n"
              " class. H2 is UNTESTABLE on this data — report as NO_GO (benchmark),\n"
              " not as a failure of the signal.")
        (cfg.out_dir / "m4_marginal_diff_test.json").write_text(json.dumps(
            {"verdict": "UNTESTABLE", "reason": "y has no variance",
             "n": int(len(y)), "n_positive": int(y.sum())}, indent=2))
        return 0

    m_base, mu_b, sd_b = fit_logistic(X_base, y)
    m_diff, mu_d, sd_d = fit_logistic(X_diff, y)
    p_base = predict(m_base, mu_b, sd_b, X_base)
    p_diff = predict(m_diff, mu_d, sd_d, X_diff)

    ll_base, ll_diff = log_likelihood(y, p_base), log_likelihood(y, p_diff)
    lr_stat = 2.0 * (ll_diff - ll_base)
    df = len(DIFF_FEATURES)
    p_value = float(stats.chi2.sf(max(lr_stat, 0.0), df))

    auc_base, auc_diff = auc(y, p_base), auc(y, p_diff)
    delta = auc_diff - auc_base

    boot_subset = cluster_bootstrap_delta_auc(y, p_base, p_diff, subsets,
                                              args.n_boot, args.seed)
    boot_key = cluster_bootstrap_delta_auc(y, p_base, p_diff, keys,
                                           args.n_boot, args.seed)
    cv = cv_delta_auc(X_base, X_diff, y, keys, seed=args.seed)

    ci_excludes_zero = bool(np.isfinite(boot_subset["lo"]) and boot_subset["lo"] > 0)
    h2_pass = bool(delta > 0 and ci_excludes_zero and p_value < ALPHA_LRT)

    result = {
        "split": list(args.subsets),
        "n": int(len(y)), "n_positive": int(y.sum()), "base_rate": float(y.mean()),
        "n_dropped_missing_features": n_dropped,
        "features_base": list(BASE_FEATURES),
        "features_diff": list(BASE_FEATURES + DIFF_FEATURES),
        "coef_base": dict(zip(BASE_FEATURES, m_base.coef_[0].tolist())),
        "coef_diff": dict(zip(BASE_FEATURES + DIFF_FEATURES, m_diff.coef_[0].tolist())),
        "log_likelihood_base": ll_base, "log_likelihood_diff": ll_diff,
        "lrt_statistic": lr_stat, "lrt_df": df, "lrt_p": p_value,
        "auc_base": auc_base, "auc_diff": auc_diff, "delta_auc": delta,
        "bootstrap_subset_clustered": boot_subset,
        "bootstrap_key_clustered": boot_key,
        "cross_validated": cv,
        "calibration_base": calibration_curve(y, p_base),
        "calibration_diff": calibration_curve(y, p_diff),
        "criterion": "delta_auc > 0 AND subset-clustered 95% CI excludes 0 AND "
                     f"LRT p < {ALPHA_LRT}",
        "h2_pass": h2_pass,
    }
    out = cfg.out_dir / "m4_marginal_diff_test.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\n LRT  : chi2({df}) = {lr_stat:.3f}   p = {p_value:.3e}"
          f"   {'PASS' if p_value < ALPHA_LRT else 'FAIL'} (alpha={ALPHA_LRT})")
    print(f" AUC  : base={auc_base:.4f}  diff={auc_diff:.4f}  delta={delta:+.4f}")
    print(f" CI   : subset-clustered [{boot_subset['lo']:+.4f}, {boot_subset['hi']:+.4f}] "
          f"over {boot_subset['n_clusters']} cluster(s)")
    print(f"        key-clustered    [{boot_key['lo']:+.4f}, {boot_key['hi']:+.4f}] "
          f"over {boot_key['n_clusters']} cluster(s)")
    print(f" CV   : delta_auc={cv.get('delta_auc', float('nan')):+.4f} "
          f"({cv.get('n_splits', 0)}-fold, grouped by key)")

    if boot_subset["n_clusters"] < 5:
        print(f"\n WARNING: the subset-clustered bootstrap has only "
              f"{boot_subset['n_clusters']} cluster(s). Its CI is not a reliable\n"
              " interval on its own — read it together with the key-clustered CI\n"
              " and the cross-validated delta.")
    if np.isfinite(cv.get("delta_auc", float("nan"))) and \
            np.sign(cv["delta_auc"]) != np.sign(delta):
        print("\n WARNING: in-sample and cross-validated delta_auc disagree in SIGN.\n"
              " Treat H2 as NOT passed regardless of the interval.")

    print("\n" + "=" * 74)
    print(f" H2 {'PASSES' if h2_pass else 'DOES NOT PASS'}: {result['criterion']}")
    print("=" * 74)
    print(f" wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
