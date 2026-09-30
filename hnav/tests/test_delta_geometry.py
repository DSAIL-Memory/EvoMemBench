"""Tests for M7 delta-vector geometry.

Every quantity is checked against a closed form or an independently constructed
answer, never against "it ran". The two regimes that matter are a NULL regime
(isotropic directions, where each statistic has a known expectation) and an
INFORMATIVE regime (a planted common direction, where the answer is known by
construction) - a statistic that does not separate those two is decoration.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.stage0.m7_delta_geometry import (
    Store, build_controls, caliper_match, deltas, directional, heldout_energy,
    relation_decomposition,
)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ── the identity the whole analysis rests on ─────────────────────────────────
def test_delta_norm_is_a_deterministic_function_of_the_cosine():
    """||v_l - v_e||^2 == 2(1 - cos) exactly, for unit vectors."""
    rng = np.random.default_rng(0)
    V = unit(rng.standard_normal((40, 16)))
    pairs = [(i, j) for i in range(0, 20) for j in (i + 20,)]
    cos, nrm, U = deltas(V, pairs)
    np.testing.assert_allclose(nrm ** 2, 2 * (1 - cos), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-12)


def test_delta_of_whitened_pair_equals_delta_of_raw_before_renormalisation():
    """Mean subtraction cancels in a difference - the claim the module makes
    about why ABTT cannot help delta geometry much."""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((10, 7))
    mu = V.mean(axis=0)
    np.testing.assert_allclose((V[3] - mu) - (V[8] - mu), V[3] - V[8], atol=1e-12)


# ── directional statistics: null regime and informative regime ───────────────
def test_resultant_of_isotropic_directions_matches_the_analytic_null():
    """E[||mean of m iid isotropic unit vectors||^2] == 1/m exactly."""
    rng = np.random.default_rng(2)
    m, d, reps = 60, 48, 400
    sq = [np.linalg.norm(unit(rng.standard_normal((m, d))).mean(axis=0)) ** 2
          for _ in range(reps)]
    assert abs(np.mean(sq) - 1.0 / m) < 0.25 / m


def test_isotropic_directions_are_indistinguishable_from_the_signflip_null():
    rng = np.random.default_rng(3)
    U = unit(rng.standard_normal((120, 64)))
    st = directional(U, np.random.default_rng(4), n_perm=200)
    assert abs(st["resultant_signflip_z"]) < 3.0
    assert abs(st["align_signflip_z"]) < 3.0
    assert abs(st["align_mean"]) < 4.0 / np.sqrt(64)


def test_planted_common_direction_is_recovered_at_its_planted_strength():
    """Build directions with a known shared component of size ``a``; the mean
    resultant must come back at ``a`` and the mean pairwise alignment at a^2."""
    rng = np.random.default_rng(5)
    m, d, a = 400, 64, 0.5
    e = np.zeros(d)
    e[0] = 1.0
    noise = unit(rng.standard_normal((m, d)))
    U = unit(a * e + np.sqrt(1 - a ** 2) * noise)
    st = directional(U, np.random.default_rng(6), n_perm=200)
    assert st["resultant"] == pytest.approx(a, abs=0.06)
    assert st["align_mean"] == pytest.approx(a ** 2, abs=0.06)
    assert st["resultant_signflip_z"] > 10
    assert st["participation_ratio"] < d          # concentrated, not isotropic


def test_participation_ratio_is_the_dimension_for_an_orthonormal_set():
    U = np.eye(12)
    st = directional(U, np.random.default_rng(7), n_perm=20)
    assert st["participation_ratio"] == pytest.approx(12.0, rel=1e-9)


# ── held-out subspace energy ─────────────────────────────────────────────────
def test_heldout_energy_is_one_for_a_rank_one_set_and_k_over_d_for_noise():
    rng = np.random.default_rng(8)
    d, m = 32, 60
    e = np.zeros(d)
    e[3] = 1.0
    rank1 = np.tile(e, (m, 1))
    iso = unit(rng.standard_normal((m, d)))
    out = heldout_energy(rank1, {"same": rank1, "iso": iso}, (1, 4))
    assert out["curves"]["same"][0] == pytest.approx(1.0, abs=1e-9)
    # an unrelated isotropic set sees exactly the random-subspace baseline
    assert out["curves"]["iso"][0] == pytest.approx(1.0 / d, abs=3.0 / d)
    assert out["baseline"] == [1 / d, 4 / d]


# ── caliper matching ─────────────────────────────────────────────────────────
def test_caliper_match_is_one_to_one_and_respects_the_caliper():
    # 0.20 and 0.21 can each serve one of the two nearby targets; 0.90 has no
    # candidate inside the caliper and must come back unmatched rather than be
    # given a far one.
    pool = np.array([0.10, 0.20, 0.21, 0.50])
    target = np.array([0.205, 0.195, 0.90])
    out = caliper_match(pool, target, caliper=0.02)
    assert out[2] == -1
    used = [int(x) for x in out if x >= 0]
    assert len(used) == len(set(used)) == 2              # no candidate reused
    assert set(np.round(pool[used], 3)) == {0.20, 0.21}
    for t, k in zip(target, out):
        if k >= 0:
            assert abs(pool[k] - t) <= 0.02 + 1e-12


def test_caliper_match_refuses_rather_than_stretching_the_caliper():
    """Scarcity must show up as an unmatched target. Silently returning a
    far-away control would make an unmatched comparison look matched."""
    pool = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    out = caliper_match(pool, np.array([0.205, 0.195]), caliper=0.02)
    assert sorted(int(x) for x in out) == [-1, 1]


def test_caliper_match_serves_the_scarce_high_cosine_targets_first():
    """Two targets compete for one high candidate. The scarcer (higher) target
    must win it; naive left-to-right greedy would give it away."""
    pool = np.array([0.50, 0.99])
    out = caliper_match(pool, np.array([0.52, 0.98]), caliper=0.05)
    assert pool[out[1]] == 0.99 and pool[out[0]] == 0.50


# ── control construction ─────────────────────────────────────────────────────
def _toy_store(rng):
    """8 facts: 2 conflicted keys, plus shared relations and shared subjects."""
    rel = ["R1", "R1", "R2", "R2", "R1", "R3", "R3", "R2"]
    sub = ["A", "A", "B", "B", "C", "A", "D", "C"]
    obj = ["x", "y", "p", "q", "z", "m", "n", "w"]        # keys (R1,A),(R2,B) clash
    txt = [f"{s} {r} {o}" for r, s, o in zip(rel, sub, obj)]
    V = unit(rng.standard_normal((8, 12)))
    return Store("toy", list(range(8)), txt, rel, sub, obj, V, "test", {})


def test_conflicts_are_exactly_the_keys_with_disagreeing_objects():
    st = _toy_store(np.random.default_rng(9))
    assert st.conflicts == [(0, 1), (2, 3)]


def test_no_control_is_a_conflict_and_each_control_matches_its_definition():
    rng = np.random.default_rng(10)
    st = _toy_store(rng)
    gram = st.V @ st.V.T
    np.fill_diagonal(gram, -2.0)
    conflict = st.conflicts
    ctrl, diag = build_controls(st, conflict, gram, rng, caliper=2.0)
    banned = {tuple(sorted(p)) for p in conflict}
    for name, pairs in ctrl.items():
        for a, b in pairs:
            assert tuple(sorted((a, b))) not in banned, name
            # no control may share a key with itself - that would be a
            # supersession or a duplicate wearing a control's label
            assert not (st.rel[a] == st.rel[b] and st.subj[a] == st.subj[b]), name
    for a, b in ctrl["same_relation"]:
        assert st.rel[a] == st.rel[b] and st.subj[a] != st.subj[b]
    for a, b in ctrl["same_subject"]:
        assert st.subj[a] == st.subj[b] and st.rel[a] != st.rel[b]
    assert diag["n_same_key"] == len(conflict)


def test_every_pair_is_oriented_earlier_serial_to_later_serial():
    rng = np.random.default_rng(11)
    st = _toy_store(rng)
    st.serial = [7, 6, 5, 4, 3, 2, 1, 0]                  # reversed on purpose
    gram = st.V @ st.V.T
    np.fill_diagonal(gram, -2.0)
    ctrl, _ = build_controls(st, st.conflicts, gram, rng, caliper=2.0)
    for pairs in ctrl.values():
        for a, b in pairs:
            assert st.serial[a] <= st.serial[b]


def test_cos_matched_control_reproduces_the_target_cosines_within_the_caliper():
    rng = np.random.default_rng(12)
    n, d = 220, 24
    V = unit(rng.standard_normal((n, d)))
    rel = [f"R{i % 7}" for i in range(n)]
    sub = [f"S{i}" for i in range(n)]
    obj = [f"O{i}" for i in range(n)]
    for i in range(0, 40, 2):                             # plant 20 conflicts
        sub[i + 1], rel[i + 1] = sub[i], rel[i]
        V[i + 1] = unit(V[i] + 0.35 * unit(rng.standard_normal(d)))
    st = Store("t", list(range(n)), [f"f{i}" for i in range(n)], rel, sub, obj,
               V, "test", {})
    gram = V @ V.T
    np.fill_diagonal(gram, -2.0)
    conflict = st.conflicts
    ctrl, diag = build_controls(st, conflict, gram, rng, caliper=0.03)
    cm = diag["cos_match"]
    assert cm["n_matched"] + cm["n_unmatched"] == len(conflict)
    if cm["n_matched"]:
        assert cm["max_abs_gap"] <= 0.03 + 1e-12
        # and the matched control really does sit at the conflict cosine
        c_conf, _, _ = deltas(V, [conflict[k] for k in cm["matched_target_idx"]])
        c_ctrl, _, _ = deltas(V, ctrl["cos_matched"])
        assert np.abs(np.sort(c_conf) - np.sort(c_ctrl)).max() <= 0.03 + 1e-12


# ── relation decomposition ───────────────────────────────────────────────────
def test_relation_decomposition_separates_a_planted_per_relation_direction():
    """Give each relation its own delta direction and nothing shared across
    relations: within-relation alignment must be ~1, across-relation ~0."""
    rng = np.random.default_rng(13)
    d = 40
    dirs = unit(rng.standard_normal((4, d)))
    rels, rows = [], []
    for r in range(4):
        for _ in range(6):
            rels.append(f"R{r}")
            rows.append(dirs[r])
    out = relation_decomposition(np.array(rows), rels)
    assert out["within_relation"]["mean"] == pytest.approx(1.0, abs=1e-9)
    assert abs(out["across_relation"]["mean"]) < 0.5
    assert out["n_relations"] == 4
