"""M6 / ABTT geometry re-measurement.  [ABTT Phase 1]

Every quantity here is checked against a closed form or an independently
computed answer, not against "it ran". The equal-coverage machinery is the part
that decides the campaign's verdict, so it is the part tested hardest: a
comparison that silently admits a different number of pairs in the two spaces
would manufacture a precision difference out of nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.geometry import ABTTWhitening
from hnav.stage0.m6_abtt_geometry import (
    describe,
    equal_coverage_delta,
    fit_whitener,
    rank_agreement,
    recall_at_precision,
)


# -- the transform itself ----------------------------------------------------
def test_d0_is_mean_centering_only_and_removes_no_direction():
    """D=0 must keep the mean subtraction and drop every principal direction.

    This is the variance-cheap baseline the whole D sweep is measured against,
    so if it silently removed a component the sweep would compare D=1 to D=1.
    """
    rng = np.random.default_rng(0)
    m = rng.standard_normal((300, 16))
    m /= np.linalg.norm(m, axis=1, keepdims=True)

    w = fit_whitener("per_store", 0, m, None, min_fit_n=200)
    assert w.fitted is True
    assert w.components.shape[0] == 0, "D=0 must remove no principal direction"

    # closed form: centre, then renormalize. Nothing else.
    expected = m - m.mean(axis=0)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(w.transform(m), expected, atol=1e-12)


def test_d0_on_already_centred_unit_data_is_the_identity():
    """The transform's no-op case, stated as a closed form.

    The fixture is antipodal — every row ``v`` is paired with ``-v`` — which is
    the only cheap way to be exactly mean-zero *and* exactly unit-norm at once.
    Centring a random matrix and then renormalizing does not stay centred, so
    that construction would not test what this claims to test.
    """
    rng = np.random.default_rng(1)
    half = rng.standard_normal((150, 8))
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    m = np.vstack([half, -half])             # mean exactly 0, every row unit
    assert np.allclose(m.mean(axis=0), 0.0, atol=1e-12)

    out = fit_whitener("per_store", 0, m, None, min_fit_n=200).transform(m)
    assert np.allclose(out, m, atol=1e-10)


def test_whitened_rows_stay_unit_norm_for_every_d():
    """``ReadGate.decide`` computes ``mat @ mat.T`` and calls the result a cosine
    matrix. That is only true for unit rows, so this is the invariant that lets
    ABTT be inserted without touching the gate."""
    rng = np.random.default_rng(2)
    m = rng.standard_normal((400, 32))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    for d in (0, 1, 3, 8, 16):
        out = fit_whitener("per_store", d, m, None, min_fit_n=200).transform(m)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), f"D={d} broke unit norm"


def test_abtt_removes_an_injected_common_direction():
    """Independently computed oracle: inject a shared direction, confirm the
    mean random-pair cosine collapses toward the isotropic 0."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((500, 24))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    common = np.zeros(24)
    common[0] = 1.0
    m = base + 3.0 * common                  # a dominant shared direction
    m /= np.linalg.norm(m, axis=1, keepdims=True)

    def mean_pair_cos(x):
        g = x @ x.T
        return float(g[np.triu_indices_from(g, k=1)].mean())

    before = mean_pair_cos(m)
    after = mean_pair_cos(fit_whitener("per_store", 1, m, None, min_fit_n=200).transform(m))
    assert before > 0.8, "the fixture should be strongly anisotropic"
    assert abs(after) < abs(before) / 4, "ABTT failed to remove the common direction"


def test_pool_level_regime_fits_on_the_pool_not_the_store():
    """The refusal this regime exists to demonstrate.

    ``pool_level`` must be handed ``pool_cap`` rows. Handing it the full store
    would make it a duplicate of ``per_store`` and the documented "ABTT cannot
    run at gate time" claim would never actually be exercised.
    """
    rng = np.random.default_rng(4)
    store = rng.standard_normal((455, 16))
    store /= np.linalg.norm(store, axis=1, keepdims=True)

    w = fit_whitener("pool_level", 3, store, None, min_fit_n=200, pool_cap=50)
    assert w.fitted is False and w.refused is True
    assert w.n_fit == 50, "pool_level must fit on the pool, not the 455-row store"

    # and per_store on the same data does fit — the two regimes must differ
    assert fit_whitener("per_store", 3, store, None, min_fit_n=200).fitted is True


def test_frozen_global_uses_the_supplied_matrix_and_refuses_without_one():
    rng = np.random.default_rng(5)
    store = rng.standard_normal((300, 12))
    store /= np.linalg.norm(store, axis=1, keepdims=True)
    glob = rng.standard_normal((900, 12))
    glob /= np.linalg.norm(glob, axis=1, keepdims=True)

    w = fit_whitener("frozen_global", 2, store, glob, min_fit_n=200)
    assert w.fitted is True and w.n_fit == 900, "must fit on the global matrix"
    assert fit_whitener("frozen_global", 2, store, None, min_fit_n=200).refused is True


def test_fit_whitener_is_deterministic():
    rng = np.random.default_rng(6)
    m = rng.standard_normal((300, 16))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    a = fit_whitener("per_store", 3, m, None, min_fit_n=200).transform(m)
    b = fit_whitener("per_store", 3, m, None, min_fit_n=200).transform(m)
    assert np.array_equal(a, b)


# -- equal coverage: the comparison that decides the verdict -----------------
def test_equal_coverage_admits_exactly_the_same_number_of_pairs():
    """The whole point of the helper. If the two spaces admit different counts,
    any precision difference between them is an artifact of coverage."""
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.85, 0.99, size=1000)
    whit = rng.uniform(0.30, 0.97, size=1000)     # deliberately a different scale
    y = rng.random(1000) < 0.3

    for row in equal_coverage_delta(raw, whit, y, [0.90, 0.92, 0.94]):
        n_raw = int((raw >= row["raw_threshold"]).sum())
        n_whit = int((whit >= row["whitened_threshold_equal_coverage"]).sum())
        assert n_raw == row["n_admitted"]
        assert n_whit == n_raw, "coverage was not matched"


def test_equal_coverage_reports_zero_delta_for_a_monotone_rescale():
    """A strictly increasing transform cannot change any thresholded decision.

    This is the null the campaign has to be able to detect: if whitening were
    merely a rescale, every delta must be exactly 0.
    """
    rng = np.random.default_rng(8)
    raw = rng.uniform(0.85, 0.99, size=500)
    y = rng.random(500) < 0.4
    rescaled = (raw - 0.85) * 7.0 - 2.0          # strictly increasing

    for row in equal_coverage_delta(raw, rescaled, y, [0.90, 0.94]):
        assert row["delta_precision"] == pytest.approx(0.0, abs=1e-12)
        assert row["delta_recall"] == pytest.approx(0.0, abs=1e-12)


def test_equal_coverage_refuses_mismatched_supports():
    """Two different candidate sets are not comparable pair-by-pair."""
    assert equal_coverage_delta(np.zeros(10), np.zeros(9),
                                np.ones(10, bool), [0.5]) == []


# -- readouts ----------------------------------------------------------------
def test_recall_at_precision_against_a_hand_computed_case():
    """Six items, ranked. Worked by hand:

        rank 1 2 3 4 5 6
        y    1 1 0 1 0 0     n_true = 3
        P    1 1 .67 .75 .6 .5
        R    .33 .67 .67 1.0 1.0 1.0

    At P>=1.0 the best recall is 0.667 (first two). At P>=0.7 the prefix of
    length 4 qualifies (P=0.75) so recall is 1.0.
    """
    score = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    y = np.array([True, True, False, True, False, False])
    out = recall_at_precision(score, y, targets=(1.0, 0.7))
    assert out["recall_at_precision_1"] == pytest.approx(2 / 3)
    assert out["recall_at_precision_0.7"] == pytest.approx(1.0)


def test_recall_at_precision_is_zero_when_the_target_is_unreachable():
    score = np.array([3.0, 2.0, 1.0])
    y = np.array([False, True, False])
    assert recall_at_precision(score, y, targets=(1.0,))["recall_at_precision_1"] == 0.0


def test_rank_agreement_is_one_for_a_monotone_map_and_minus_one_reversed():
    x = np.arange(50, dtype=float)
    assert rank_agreement(x, 3 * x + 7)["spearman"] == pytest.approx(1.0)
    assert rank_agreement(x, -x)["spearman"] == pytest.approx(-1.0)


def test_describe_band_is_the_p10_p90_width():
    v = np.arange(101, dtype=float)          # 0..100, so p10=10, p90=90
    d = describe(v)
    assert d["p10"] == pytest.approx(10.0)
    assert d["p90"] == pytest.approx(90.0)
    assert d["band_p10_p90"] == pytest.approx(80.0)


def test_describe_is_empty_for_an_empty_array():
    assert describe(np.zeros(0)) == {}


# -- the insertion-point invariant -------------------------------------------
def test_whitened_matrix_is_a_valid_cosine_matrix():
    """``ReadGate.decide`` is handed vectors and computes ``mat @ mat.T``.

    For the gate to need no change, the whitened Gram matrix must still have a
    unit diagonal and lie in [-1, 1] — i.e. still be a cosine matrix.
    """
    rng = np.random.default_rng(9)
    m = rng.standard_normal((250, 20))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    mw = ABTTWhitening(n_components=3, min_fit_n=200).fit(m).transform(m)
    g = mw @ mw.T
    assert np.allclose(np.diag(g), 1.0, atol=1e-10)
    assert g.max() <= 1.0 + 1e-10 and g.min() >= -1.0 - 1e-10
