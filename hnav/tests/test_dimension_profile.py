"""Tests for M7b per-coordinate profile. Closed forms and planted signals."""
from __future__ import annotations

import numpy as np
import pytest

from hnav.stage0.m7b_dimension_profile import (
    concentration, heldout_scores, paired_signflip, per_dim, split_indices,
)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_concentration_closed_forms():
    d = 64
    flat = np.full((1, d), 1 / np.sqrt(d))
    onehot = np.zeros((1, d)); onehot[0, 5] = 1.0
    cf, co = concentration(flat), concentration(onehot)
    assert cf["effective_dims"][0] == pytest.approx(d)
    assert co["effective_dims"][0] == pytest.approx(1.0)
    assert cf["l1_over_l2"][0] == pytest.approx(np.sqrt(d))
    assert co["top10_share"][0] == pytest.approx(1.0)


def test_sign_consistency_null_and_planted():
    rng = np.random.default_rng(0)
    m, d = 400, 50
    iso = unit(rng.standard_normal((m, d)))
    s = per_dim(iso)["sign_consistency"]
    assert s.mean() == pytest.approx(np.sqrt(2 / np.pi) / np.sqrt(m), rel=0.15)
    planted = iso.copy(); planted[:, 3] = np.abs(planted[:, 3])   # all positive
    assert per_dim(unit(planted))["sign_consistency"][3] == pytest.approx(1.0)


def test_paired_signflip_detects_a_planted_shift_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200)
    null = paired_signflip(a, a + rng.standard_normal(200) * 0.01, rng, n_perm=500)
    assert abs(null["z"]) < 3
    shift = paired_signflip(a + 0.5, a, rng, n_perm=500)
    assert shift["z"] > 5 and shift["frac_conflict_higher"] > 0.9


def test_relation_disjoint_split_never_shares_a_relation():
    rng = np.random.default_rng(2)
    rels = [f"R{i % 7}" for i in range(100)]
    A, B = split_indices(100, rels, rng, "relation_disjoint")
    assert not ({rels[i] for i in A} & {rels[i] for i in B})
    assert len(A) + len(B) == 100


def test_heldout_sign_pattern_finds_a_planted_direction_and_not_noise():
    """Conflicts share a small signed component in 20 coordinates; matched
    controls do not. The held-out sign pattern must separate them, and on pure
    noise it must sit at 0.5."""
    rng = np.random.default_rng(3)
    m, d = 300, 200
    e = np.zeros(d); e[:20] = 1 / np.sqrt(20)
    conf = unit(0.4 * e + unit(rng.standard_normal((m, d))))
    ctrl = unit(rng.standard_normal((m, d)))
    partner = np.arange(m)
    cos = np.full(m, 0.9)
    rels = [f"R{i % 5}" for i in range(m)]
    r = heldout_scores(conf, ctrl, partner, cos, cos, rels, rng, "random", n_rep=5)
    assert r["k"][16]["sign"]["paired_acc"]["mean"] > 0.9
    noise = unit(rng.standard_normal((m, d)))
    r0 = heldout_scores(noise, ctrl, partner, cos, cos, rels, rng, "random", n_rep=5)
    assert abs(r0["k"][2560]["sign"]["paired_acc"]["mean"] - 0.5) < 0.12
    # energy family is undefined when every coordinate is kept
    assert r["k"][2560]["energy"]["paired_acc"]["mean"] != r["k"][2560]["energy"]["paired_acc"]["mean"]
