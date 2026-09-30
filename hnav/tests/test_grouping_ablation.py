"""M1b grouping-ablation mechanics, on fixtures with known answers.  [T2]

The ablation's verdict is only as trustworthy as its bookkeeping: if candidate
generation silently drops true pairs, or the sweep miscounts, a real geometry
signal could be reported as absent (or vice versa). So the pieces are checked
against constructed cases where precision, recall and the recall ceiling are
known exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.stage0.m1b_grouping_ablation import (equal_coverage_point,
                                               knn_candidates, sweep, truth_pairs)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def planted_matrix(n_pairs: int = 5, dim: int = 32, eps: float = 0.02) -> np.ndarray:
    """``n_pairs`` planted near-duplicate pairs, otherwise near-orthogonal."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pairs):
        base = unit(rng.standard_normal(dim))
        partner = unit(base + eps * unit(rng.standard_normal(dim)))
        rows += [base, partner]
    return np.stack(rows)


def test_truth_pairs_uses_the_regex_oracle():
    facts = [(0, "Thomas Kyd was born in the city of London."),
             (1, "Thomas Kyd was born in the city of Madrid."),
             (2, "The capital of France is Paris."),
             (3, "not a templated fact at all")]
    pairs, key_of, n_parsed = truth_pairs(facts)
    assert pairs == {(0, 1)}, "only the same (relation, subject) pair is truth"
    assert n_parsed == 3, "the unparseable line must not be counted as parsed"
    assert 3 not in key_of


def test_truth_pairs_counts_all_pairs_in_a_larger_group():
    facts = [(i, f"The capital of France is City{i}.") for i in range(4)]
    pairs, _, _ = truth_pairs(facts)
    assert len(pairs) == 6            # 4 choose 2
    assert all(i < j for i, j in pairs)


def test_knn_never_pairs_a_fact_with_itself_and_is_symmetric():
    mat = planted_matrix(n_pairs=6)
    cands = knn_candidates(mat, top_n=3, block=4)     # block < n forces the loop
    assert all(i < j for i, j, _ in cands)
    assert len({(i, j) for i, j, _ in cands}) == len(cands), "duplicate pairs"
    assert all(i != j for i, j, _ in cands)


def test_knn_recovers_planted_pairs_and_the_ceiling_is_one():
    mat = planted_matrix(n_pairs=5)
    truth = {(2 * i, 2 * i + 1) for i in range(5)}
    cands = knn_candidates(mat, top_n=3)
    proposed = {(i, j) for i, j, _ in cands}
    assert truth <= proposed, "planted near-duplicates must be within top-3"

    rows = sweep(cands, truth, np.array([0.9]))
    assert rows[0]["recall"] == pytest.approx(1.0)
    assert rows[0]["precision"] == pytest.approx(1.0)
    assert rows[0]["f1"] == pytest.approx(1.0)


def test_ceiling_is_reported_when_knn_is_too_narrow():
    """A truth pair the generator never proposes is unrecoverable at any tau."""
    mat = planted_matrix(n_pairs=4)
    truth = {(0, 1), (2, 3), (4, 5), (6, 7), (0, 7)}   # (0,7) is not a near-duplicate
    cands = knn_candidates(mat, top_n=1)
    proposed = {(i, j) for i, j, _ in cands}
    ceiling = len(truth & proposed) / len(truth)
    assert ceiling < 1.0
    best = max(sweep(cands, truth, np.arange(0.0, 1.0, 0.05)),
               key=lambda r: r["f1"])
    assert best["recall"] <= ceiling + 1e-9


def test_sweep_is_monotone_in_recall():
    mat = planted_matrix(n_pairs=6)
    truth = {(2 * i, 2 * i + 1) for i in range(6)}
    curve = sweep(knn_candidates(mat, top_n=4), truth, np.arange(0.0, 1.0, 0.05))
    recalls = [r["recall"] for r in curve]
    assert all(a >= b for a, b in zip(recalls, recalls[1:])), \
        "raising the threshold cannot increase recall"


def test_equal_coverage_point_predicts_exactly_as_many_pairs_as_truth():
    mat = planted_matrix(n_pairs=5)
    truth = {(2 * i, 2 * i + 1) for i in range(5)}
    eq = equal_coverage_point(knn_candidates(mat, top_n=4), truth)
    assert eq["n_predicted"] == len(truth)
    assert eq["precision"] == pytest.approx(eq["recall"])   # equal denominators
    assert eq["f1"] == pytest.approx(eq["precision"])


def test_no_truth_and_no_candidates_do_not_crash():
    assert equal_coverage_point([], set()) == {}
    rows = sweep([], set(), np.array([0.5]))
    assert rows[0]["n_predicted"] == 0 and rows[0]["f1"] == 0.0


def test_real_subset_has_the_expected_grouping_structure(sh_6k):
    """Regression anchor against the committed dataset."""
    from hnav.adapters.mab_adapter import explode_facts

    facts = explode_facts(sh_6k["context"])
    pairs, _, n_parsed = truth_pairs(facts)
    assert len(facts) == 455
    assert n_parsed / len(facts) > 0.99, "template coverage dropped below 99%"
    assert len(pairs) == 160
