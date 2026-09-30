"""M4 / H2 statistics, against constructed answers.  [T7]

Two synthetic regimes are used throughout:

  *informative* — the diff features genuinely carry signal about ``y``, so a
  correct implementation must show ΔAUC > 0 and a small LRT p;
  *null*        — the diff features are pure noise, so a correct implementation
  must NOT declare H2 passed.

A test suite that only checks the informative case would pass with an
implementation that always says yes.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hnav.stage0.m4_marginal_diff_test import (auc, build_matrices,
                                               calibration_curve,
                                               cluster_bootstrap_delta_auc,
                                               cv_delta_auc, fit_logistic,
                                               load_rows, log_likelihood, predict)


# ── metrics ──────────────────────────────────────────────────────────────────
def test_auc_endpoints_and_ties():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    # one class empty -> undefined, not 0.5
    assert np.isnan(auc(np.array([1, 1]), np.array([0.1, 0.9])))


def test_auc_handles_partial_ties_by_averaging_ranks():
    y = np.array([0, 1, 1, 0])
    # positives at 0.5 and 0.9; negatives at 0.5 and 0.1
    assert auc(y, np.array([0.5, 0.5, 0.9, 0.1])) == pytest.approx(0.875)


def test_log_likelihood_matches_hand_computation():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    assert log_likelihood(y, p) == pytest.approx(np.log(0.8) + np.log(0.7))
    # a confident wrong prediction is clipped, not infinite
    assert np.isfinite(log_likelihood(np.array([1]), np.array([0.0])))


def test_calibration_curve_bins_are_complete():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    curve = calibration_curve(y, p, n_bins=10)
    assert len(curve) == 10
    assert sum(b["n"] for b in curve) == 200
    assert all(0.0 <= b["observed_rate"] <= 1.0 for b in curve)


# ── synthetic regimes ────────────────────────────────────────────────────────
def make_data(informative: bool, n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    whole = rng.normal(size=n)
    qr = rng.normal(size=n)
    diff_sim = rng.normal(size=n)
    diff_nov = rng.normal(size=n)

    logit = 0.5 * whole - 0.4 * qr
    if informative:
        logit += 1.8 * diff_sim - 1.5 * diff_nov
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    X_base = np.column_stack([whole, qr])
    X_diff = np.column_stack([whole, qr, diff_sim, diff_nov])
    return X_base, X_diff, y


def fit_pair(X_base, X_diff, y):
    m1, mu1, sd1 = fit_logistic(X_base, y)
    m2, mu2, sd2 = fit_logistic(X_diff, y)
    return predict(m1, mu1, sd1, X_base), predict(m2, mu2, sd2, X_diff)


def test_informative_regime_shows_a_positive_delta_auc():
    X_base, X_diff, y = make_data(informative=True)
    p_base, p_diff = fit_pair(X_base, X_diff, y)
    assert auc(y, p_diff) - auc(y, p_base) > 0.05


def test_null_regime_shows_almost_no_delta_auc():
    X_base, X_diff, y = make_data(informative=False)
    p_base, p_diff = fit_pair(X_base, X_diff, y)
    delta = auc(y, p_diff) - auc(y, p_base)
    assert abs(delta) < 0.03, "noise features must not buy real discrimination"


def test_lrt_separates_the_two_regimes():
    from scipy import stats

    def lrt_p(informative):
        X_base, X_diff, y = make_data(informative)
        p_base, p_diff = fit_pair(X_base, X_diff, y)
        stat = 2 * (log_likelihood(y, p_diff) - log_likelihood(y, p_base))
        return float(stats.chi2.sf(max(stat, 0.0), 2))

    assert lrt_p(True) < 1e-6
    assert lrt_p(False) > 0.01, "the null regime must not clear the LRT bar"


def test_bootstrap_ci_excludes_zero_only_when_the_signal_is_real():
    for informative, expect_positive in ((True, True), (False, False)):
        X_base, X_diff, y = make_data(informative)
        p_base, p_diff = fit_pair(X_base, X_diff, y)
        clusters = np.arange(len(y)) % 40          # 40 clusters
        ci = cluster_bootstrap_delta_auc(y, p_base, p_diff, clusters,
                                         n_boot=400, seed=1)
        assert ci["n_clusters"] == 40
        assert (ci["lo"] > 0) is expect_positive


def test_bootstrap_reports_a_degenerate_cluster_count():
    """Two clusters is what the calibration split actually has — the function
    must still return, and must report how few clusters it had."""
    X_base, X_diff, y = make_data(True, n=200)
    p_base, p_diff = fit_pair(X_base, X_diff, y)
    clusters = np.where(np.arange(len(y)) < 100, "sh_6k", "sh_32k")
    ci = cluster_bootstrap_delta_auc(y, p_base, p_diff, clusters, n_boot=200, seed=2)
    assert ci["n_clusters"] == 2 and ci["n_valid"] > 0


def test_cross_validated_delta_is_out_of_sample():
    X_base, X_diff, y = make_data(informative=False, n=400)
    groups = np.arange(len(y)) % 20
    cv = cv_delta_auc(X_base, X_diff, y, groups, seed=3)
    assert cv["n_splits"] == 5
    # out of sample, four useless parameters should not help and may well hurt
    assert cv["delta_auc"] < 0.05


def test_cv_declines_gracefully_with_too_few_groups():
    X_base, X_diff, y = make_data(True, n=50)
    cv = cv_delta_auc(X_base, X_diff, y, np.ones(50), seed=0)
    assert cv["n_splits"] == 0 and np.isnan(cv["delta_auc"])


# ── data plumbing ────────────────────────────────────────────────────────────
def test_build_matrices_drops_rows_with_missing_features():
    rows = [
        {"id": "fact:1", "key": ["r", "s"], "subset": "sh_6k", "y": 1,
         "whole_blob_sim": 0.9, "qr_residual": 0.1, "diff_sim": 0.2,
         "diff_novelty": 0.3},
        {"id": "fact:2", "key": None, "subset": "sh_6k", "y": 0,
         "whole_blob_sim": None, "qr_residual": 0.1, "diff_sim": 0.2,
         "diff_novelty": 0.3},
    ]
    X_base, X_diff, y, subsets, keys, dropped = build_matrices(rows)
    assert dropped == 1
    assert X_base.shape == (1, 2) and X_diff.shape == (1, 4)
    assert list(y) == [1] and list(subsets) == ["sh_6k"] and list(keys) == ["r|s"]


def test_load_rows_joins_on_candidate_id(tmp_path):
    (tmp_path / "m3_write_rows_sh_6k.json").write_text(json.dumps([
        {"id": "fact:1", "key": ["r", "s"], "whole_blob_sim": 0.9,
         "qr_residual": 0.1, "diff_sim": 0.2, "diff_novelty": 0.3},
        {"id": "fact:9", "key": ["r", "s"], "whole_blob_sim": 0.9,
         "qr_residual": 0.1, "diff_sim": 0.2, "diff_novelty": 0.3},
    ]))
    (tmp_path / "m3_write_outcomes_sh_6k.json").write_text(json.dumps(
        {"fact:1": {"counterfactual_class": "must_write"}}))

    rows, missing = load_rows(tmp_path, ["sh_6k"])
    assert missing == []
    assert [r["id"] for r in rows] == ["fact:1"], "unlabeled candidates are excluded"
    assert rows[0]["y"] == 1

    _, missing2 = load_rows(tmp_path, ["sh_32k"])
    assert missing2 == ["sh_32k"], "a missing subset must be reported, not skipped"


def test_y_is_must_write_and_nothing_else(tmp_path):
    (tmp_path / "m3_write_rows_sh_6k.json").write_text(json.dumps(
        [{"id": f"fact:{i}", "key": None, "whole_blob_sim": 0.5, "qr_residual": 0.5,
          "diff_sim": 0.5, "diff_novelty": 0.5} for i in range(4)]))
    (tmp_path / "m3_write_outcomes_sh_6k.json").write_text(json.dumps({
        "fact:0": {"counterfactual_class": "must_write"},
        "fact:1": {"counterfactual_class": "must_suppress"},
        "fact:2": {"counterfactual_class": "may_suppress"},
        "fact:3": {"counterfactual_class": "uncertain"},
    }))
    rows, _ = load_rows(tmp_path, ["sh_6k"])
    assert [r["y"] for r in rows] == [1, 0, 0, 0]
