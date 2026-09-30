"""Numeric core of the retrieval signals, against known answers.  [T5]

Fixtures are built so the right answer is known in closed form — orthonormal
bases, duplicated rows, deliberate ties, uniform scores — rather than asserting
that the code merely runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.retrieval_signals import (DeltaSignalSet, RetrievalSignals,
                                         RetrievalSignalSet, entropy, softmax)
from hnav.core.types import RetrievalView


def view(scores, ids=None, top_k=3, query="q") -> RetrievalView:
    scores = np.asarray(scores, dtype=np.float64)
    ids = ids or [f"id:{i}" for i in range(len(scores))]
    return RetrievalView(query=query, query_vector=None, ids=list(ids),
                         scores=scores, top_k=top_k)


# ── primitives ───────────────────────────────────────────────────────────────
def test_softmax_and_entropy_closed_form():
    p = softmax(np.zeros(4))
    assert np.allclose(p, 0.25)
    assert entropy(p) == pytest.approx(np.log(4))

    # a one-hot distribution has zero entropy; log(0) must not appear
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy(np.array([])) == 0.0

    # softmax is shift-invariant, which is what makes it numerically stable
    assert np.allclose(softmax(np.array([1e4, 1e4 + 1.0])),
                       softmax(np.array([0.0, 1.0])))


def test_von_neumann_entropy_endpoints():
    """ln(m) for an orthonormal neighbourhood, 0 for a collinear one."""
    m = 8
    orthonormal = np.eye(m)
    assert RetrievalSignals.von_neumann(orthonormal) == pytest.approx(np.log(m))

    v = np.zeros((m, m))
    v[:, 0] = 1.0                     # m copies of the same unit vector
    assert RetrievalSignals.von_neumann(v) == pytest.approx(0.0, abs=1e-9)

    # two orthogonal directions, m/2 copies each -> ln 2
    half = np.zeros((m, m))
    half[: m // 2, 0] = 1.0
    half[m // 2:, 1] = 1.0
    assert RetrievalSignals.von_neumann(half) == pytest.approx(np.log(2))

    assert np.isnan(RetrievalSignals.von_neumann(np.zeros((0, 4))))


# ── single ranking ───────────────────────────────────────────────────────────
def test_margins_and_ranks_are_exact():
    sig = RetrievalSignals(top_m=50, k=3).compute(
        view([90.0, 80.0, 70.0, 10.0]), self_id="id:2")
    assert sig.top1 == 90.0 and sig.top2 == 80.0
    assert sig.margin == pytest.approx(10.0)
    assert sig.nmargin == pytest.approx(10.0 / 90.0)
    assert sig.rank_self == 2
    assert sig.n_ranked == 4 and sig.top_m == 4
    assert sig.dispersion == pytest.approx(np.std([90.0, 80.0, 70.0, 10.0]))


def test_uniform_scores_give_maximum_entropy_and_all_ties():
    s = RetrievalSignals(top_m=5, k=3).compute(view([42.0] * 5))
    assert s.H_raw == pytest.approx(np.log(5))
    assert s.H_z == pytest.approx(np.log(5))      # std==0 -> z all zero -> uniform
    assert s.eff_size == pytest.approx(5.0)
    assert s.n_ties_top_m == 4                    # 5 values, 1 distinct
    assert s.margin == pytest.approx(0.0)
    assert s.score_std == pytest.approx(0.0)


def test_raw_entropy_is_degenerate_at_cosine_times_100():
    """The reason H_raw is barred from decisions (hard rule 6).

    On a cosine×100 scale a perfectly ordinary neighbourhood — cosines .90 down
    to .10 — turns into 80 logits of spread, and the raw softmax collapses onto
    the top item. z-scoring restores a usable spread. M2 measures whether this
    holds on the real rankings; here it is demonstrated on a fixture.
    """
    m = 20
    cosines = np.linspace(0.90, 0.10, m)
    s = RetrievalSignals(top_m=m, k=10).compute(view(cosines * 100.0))
    ceiling = np.log(m)                    # entropy of a uniform top-m
    assert s.H_raw < 0.05 * ceiling, "raw softmax was expected to saturate"
    assert s.H_z > 0.75 * ceiling, "z-scored entropy should stay informative"
    assert s.H_z > 30 * s.H_raw
    assert s.eff_size == pytest.approx(np.exp(s.H_z))


def test_empty_ranking_is_not_an_error():
    s = RetrievalSignals().compute(view([]))
    assert s.n_ranked == 0 and s.top_m == 0


def test_h_vn_requires_vectors_and_is_nan_otherwise():
    sig = RetrievalSignals(top_m=4).compute(view([9.0, 8.0, 7.0, 6.0]))
    assert np.isnan(sig.H_vn)

    vecs = np.eye(4)
    sig2 = RetrievalSignals(top_m=4).compute(view([9.0, 8.0, 7.0, 6.0]), vectors=vecs)
    assert sig2.H_vn == pytest.approx(np.log(4))


# ── provisional insert ───────────────────────────────────────────────────────
def test_delta_churn_and_rank_shift_are_exact():
    before = view([90.0, 80.0, 70.0, 60.0], ids=list("abcd"), top_k=3)
    # the new item lands at rank 1: a,X,b,c,d  -> b and c shift by one, d leaves
    after = view([90.0, 85.0, 80.0, 70.0, 60.0], ids=["a", "X", "b", "c", "d"], top_k=3)

    d = RetrievalSignals(top_m=50, k=3).compute_delta(before, after, self_id="X", k=3)
    assert isinstance(d, DeltaSignalSet)
    assert d.rank_self_after == 1
    assert d.displaced_ids == ["c"]        # top-3 was a,b,c; now a,X,b
    assert d.churn_at_k == 1
    assert d.churn_frac == pytest.approx(1 / 3)
    # survivors in the new top-3: a (0->0), b (1->2) -> mean |shift| = 0.5
    assert d.rank_shift == pytest.approx(0.5)


def test_delta_neighbor_is_the_mean_over_probes():
    b1, a1 = view([10.0, 9.0], ids=list("ab")), view([10.0, 9.0, 9.0], ids=list("abX"))
    b2, a2 = view([10.0, 1.0], ids=list("ab")), view([10.0, 1.0, 1.0], ids=list("abX"))
    sigs = RetrievalSignals(top_m=50, k=2)
    d = sigs.compute_delta(b1, a1, self_id="X", neighbors=[(b1, a1), (b2, a2)])

    expected = np.mean([sigs.compute(a1).H_z - sigs.compute(b1).H_z,
                        sigs.compute(a2).H_z - sigs.compute(b2).H_z])
    assert d.dH_neighbor == pytest.approx(expected)
    assert d.dH_self == pytest.approx(sigs.compute(a1).H_z - sigs.compute(b1).H_z)


def test_insert_that_changes_nothing_has_zero_churn():
    before = view([90.0, 80.0, 70.0], ids=list("abc"), top_k=3)
    after = view([90.0, 80.0, 70.0, 1.0], ids=list("abcX"), top_k=3)
    d = RetrievalSignals(top_m=50, k=3).compute_delta(before, after, self_id="X", k=3)
    assert d.churn_at_k == 0 and d.displaced_ids == []
    assert d.rank_shift == pytest.approx(0.0)
    assert d.rank_self_after == 3


def test_signal_sets_are_json_shaped():
    s = RetrievalSignals(top_m=4).compute(view([1.0, 2.0]))
    assert set(RetrievalSignalSet().to_dict()) == set(s.to_dict())
    assert isinstance(s.to_dict()["top1"], float)
