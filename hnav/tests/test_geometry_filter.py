"""Geometry-filter experiment machinery — closed-form and oracle checks.

Synthetic data only: nothing here needs the embedding cache or the committed
dataset, so these run everywhere the rest of the suite runs. sklearn is used
solely as an independent oracle for AUPRC and is skipped when absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.geometry_filter import data as gfdata
from hnav.geometry_filter.methods import RCED, RCESP
from hnav.geometry_filter.metrics import (auprc, auroc, best_f1_threshold,
                                          inverted_win_rate,
                                          paired_bootstrap_delta_auc,
                                          perm_pvalue, prf_at)
from hnav.geometry_filter.run_nulls import _stats_for

RNG = np.random.default_rng(7)


# ── metrics against closed forms ─────────────────────────────────────────────
def test_auroc_matches_hand_computed_values():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    # one tie: pairs (1,0)win (1,1)tie (2,0)win (2,1)win -> 3.5/4
    assert auroc([1, 2], [0, 1]) == pytest.approx(3.5 / 4)


def test_auprc_matches_sklearn_average_precision():
    sk = pytest.importorskip("sklearn.metrics")
    pos, neg = RNG.normal(1, 1, 300), RNG.normal(0, 1, 700)
    y = np.r_[np.ones(300), np.zeros(700)]
    s = np.r_[pos, neg]
    assert auprc(pos, neg) == pytest.approx(
        sk.average_precision_score(y, s), abs=1e-9)


def test_prf_at_closed_form():
    m = prf_at(0.5, pos=[0.9, 0.6, 0.4], neg=[0.7, 0.2, 0.1, 0.0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 4)


def test_best_f1_threshold_is_a_calibration_quantity_that_separates_cleanly():
    t = best_f1_threshold([0.8, 0.9], [0.1, 0.2])
    assert 0.2 < t <= 0.8
    assert prf_at(t, [0.8, 0.9], [0.1, 0.2])["f1"] == 1.0


def test_inverted_win_rate_counts_only_cosine_wrong_comparisons():
    # neg cos exceeds pos cos only for (pos0, neg0): 1 comparison.
    out = inverted_win_rate(pos_scores=[0.9, 0.9], pos_cos=[0.5, 0.99],
                            neg_scores=[0.1, 0.1], neg_cos=[0.7, 0.4])
    assert out["n_comparisons"] == 1 and out["win_rate"] == 1.0
    # method that IS cosine loses every inverted comparison by construction
    out = inverted_win_rate([0.5, 0.99], [0.5, 0.99], [0.7, 0.4], [0.7, 0.4])
    assert out["win_rate"] == 0.0


def test_perm_pvalue_add_one_formula():
    assert perm_pvalue(5.0, [1, 2, 3]) == pytest.approx(1 / 4)
    assert perm_pvalue(0.0, [1, 2, 3]) == pytest.approx(4 / 4)


def test_paired_bootstrap_flags_a_real_difference_and_not_a_null_one():
    pos, neg = RNG.normal(2, 1, 200), RNG.normal(0, 1, 200)
    noise_p, noise_n = RNG.normal(0, 1, 200), RNG.normal(0, 1, 200)
    real = paired_bootstrap_delta_auc(pos, neg, noise_p, noise_n,
                                      n_boot=200, seed=1)
    assert real["delta"] > 0.3 and real["lo"] > 0.0
    null = paired_bootstrap_delta_auc(pos, neg, pos, neg, n_boot=50, seed=1)
    assert null["delta"] == 0.0


# ── synthetic PairView helpers ───────────────────────────────────────────────
def _mk_records(n, tagged=True):
    recs = []
    for i in range(n):
        r = {"pair_id": f"syn:{i}-x", "fact_a": f"a{i}", "fact_b": f"b{i}",
             "parser": {"fact_a_parsed": {"relation": "R", "subject": f"s{i}",
                                          "object": "o1"},
                        "fact_b_parsed": {"relation": "R", "subject": f"s{i}",
                                          "object": "o2"},
                        "both_parse": True}}
        if tagged:
            r["serial_earlier"], r["serial_later"] = 0, 1
        recs.append(r)
    return recs


def _view_from_vectors(Va, Vb, tagged=True):
    recs = _mk_records(len(Va), tagged)
    index = {}
    rows = []
    for i in range(len(Va)):
        index[f"a{i}"] = len(rows); rows.append(Va[i])
        index[f"b{i}"] = len(rows); rows.append(Vb[i])
    return gfdata.PairView(recs, index), np.asarray(rows, dtype=np.float64)


# ── data-layer invariants ────────────────────────────────────────────────────
def test_mean_centering_without_renormalization_is_a_noop_on_diffs():
    Va, Vb = RNG.normal(size=(5, 8)), RNG.normal(size=(5, 8))
    mu = RNG.normal(size=8)
    assert np.allclose((Vb - mu) - (Va - mu), Vb - Va)


def test_orientation_sign_is_plus_one_for_tagged_and_deterministic_otherwise():
    tagged = {"pair_id": "sh_6k:1-2", "serial_earlier": 1, "serial_later": 2}
    assert gfdata.orientation_sign(tagged) == 1
    untagged = {"pair_id": "sh_6k:9-7"}
    s1, s2 = gfdata.orientation_sign(untagged), gfdata.orientation_sign(untagged)
    assert s1 == s2 and s1 in (-1, 1)


def test_pairview_diff_is_unit_norm_and_respects_orientation():
    Va = np.eye(4)[:3]
    Vb = np.eye(4)[1:]
    pv, V = _view_from_vectors(Va, Vb, tagged=True)
    D = pv.diff(V, normalize=True, oriented=True)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    assert np.allclose(D[0], (V[1] - V[0]) / np.sqrt(2))


# ── RCED / RCESP closed forms ────────────────────────────────────────────────
def test_rced_recovers_a_shared_edit_direction_and_scores_by_alignment():
    dim = 16
    e1 = np.eye(dim)[0]
    D = np.tile(e1, (6, 1)) + RNG.normal(scale=1e-3, size=(6, dim))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    m = RCED(min_pairs=2).fit(D, ["R"] * 6)
    assert abs(float(m.mu["R"] @ e1)) > 0.999
    # pair whose edit is exactly e1 -> score 1; orthogonal edit -> ~0
    Va = np.zeros((2, dim)); Va[0, 2] = 1.0; Va[1, 3] = 1.0
    Vb = Va.copy(); Vb[0] += e1; Vb[1] += np.eye(dim)[4]
    pv, V = _view_from_vectors(Va, Vb)
    s, info = m.score(pv, V)
    assert s[0] > 0.999 and s[1] < 0.01
    assert info["n_relation_fallback"] == 0
    # the score is sign-invariant: reversing the pair cannot change it
    pv2, V2 = _view_from_vectors(Vb, Va)
    s2, _ = m.score(pv2, V2)
    assert np.allclose(s, s2)


def test_rcesp_score_is_one_inside_the_span_and_zero_orthogonal_to_it():
    dim = 12
    basis = np.eye(dim)[:2]  # span{e1, e2}
    coeff = RNG.normal(size=(8, 2))
    D = coeff @ basis
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    m = RCESP(k=2, min_pairs=2).fit(D, ["R"] * 8)
    inside = 0.6 * basis[0] + 0.8 * basis[1]
    Va = np.zeros((2, dim)); Va[:, 5] = 1.0
    Vb = Va.copy(); Vb[0] += inside; Vb[1] += np.eye(dim)[7]
    pv, V = _view_from_vectors(Va, Vb)
    s, _ = m.score(pv, V)
    assert s[0] > 0.99 and s[1] < 1e-6


def test_rcesp_k_truncation_drops_out_of_subspace_energy():
    dim = 10
    # energy overwhelmingly along e1, a whisper along e2
    D = np.array([np.eye(dim)[0]] * 20 + [np.eye(dim)[1]] * 1)
    m = RCESP(k=1, min_pairs=2).fit(D, ["R"] * 21)
    Va = np.zeros((1, dim)); Va[0, 5] = 1.0
    Vb = Va.copy(); Vb[0] += np.eye(dim)[1]  # edit along the weak direction
    pv, V = _view_from_vectors(Va, Vb)
    s, _ = m.score(pv, V)
    assert s[0] < 0.1  # k=1 kept only e1, so an e2 edit is outside


def test_unknown_relation_falls_back_to_global_and_is_counted():
    D = np.tile(np.eye(8)[0], (5, 1))
    m = RCED(min_pairs=2).fit(D, ["R"] * 5)
    pv, V = _view_from_vectors(np.zeros((1, 8)), np.eye(8)[:1])
    pv.relation = ["UNSEEN"]
    _, info = m.score(pv, V)
    assert info["n_relation_fallback"] == 1


# ── contrastive subspace (experiment 4) closed forms ─────────────────────────
def test_contrastive_subspace_is_positive_for_object_edits_and_negative_for_subject_edits():
    from hnav.geometry_filter.run_dimension_ideas import ContrastiveSubspace
    dim = 16
    obj_basis, subj_basis = np.eye(dim)[:2], np.eye(dim)[4:6]
    D_pos = RNG.normal(size=(10, 2)) @ obj_basis
    D_neg = RNG.normal(size=(10, 2)) @ subj_basis
    D_pos /= np.linalg.norm(D_pos, axis=1, keepdims=True)
    D_neg /= np.linalg.norm(D_neg, axis=1, keepdims=True)
    m = ContrastiveSubspace(k=2, min_pairs=2).fit(
        D_pos, ["R"] * 10, D_neg, ["R"] * 10)
    Va = np.zeros((2, dim)); Va[:, 9] = 1.0
    Vb = Va.copy(); Vb[0] += obj_basis[0]; Vb[1] += subj_basis[0]
    pv, V = _view_from_vectors(Va, Vb)
    s = m.score(pv, V)
    assert s[0] > 0.9 and s[1] < -0.9  # +||.||^2 in-object, -||.||^2 in-subject
    # sign-invariant: reversing the pair changes nothing
    pv2, V2 = _view_from_vectors(Vb, Va)
    assert np.allclose(s, m.score(pv2, V2))


def test_topdim_energy_fraction_recovers_planted_discriminative_dimensions():
    # gold edits live on dims {0,1}; negative edits on dims {5,6}
    dim, n = 12, 60
    P = np.zeros((n, dim)); P[:, :2] = np.abs(RNG.normal(1, 0.1, (n, 2)))
    N = np.zeros((n, dim)); N[:, 5:7] = np.abs(RNG.normal(1, 0.1, (n, 2)))
    d_eff = (P.mean(0) - N.mean(0)) / np.sqrt(
        0.5 * (P.std(0) ** 2 + N.std(0) ** 2) + 1e-12)
    order = np.argsort(-np.abs(d_eff))
    top4 = set(int(i) for i in order[:4])
    assert top4 == {0, 1, 5, 6}
    pos_dims = order[:4][d_eff[order[:4]] > 0]
    assert set(int(i) for i in pos_dims) == {0, 1}
    sc_p = (P[:, pos_dims] ** 2).sum(1)
    sc_n = (N[:, pos_dims] ** 2).sum(1)
    assert auroc(sc_p, sc_n) == 1.0


# ── null machinery on informative vs null synthetic regimes ──────────────────
def test_stats_separate_an_informative_regime_from_an_isotropic_null():
    dim, n = 32, 40
    rels = ["R1"] * 20 + ["R2"] * 20
    subjs = [f"s{i}" for i in range(n)]
    trans = [("R1", "a", "b")] * 20 + [("R2", "c", "d")] * 20
    # informative: within a transition, edits share a direction
    D = np.r_[np.tile(np.eye(dim)[0], (20, 1)), np.tile(np.eye(dim)[1], (20, 1))]
    D = D + RNG.normal(scale=0.05, size=D.shape)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    info = _stats_for(D, rels, subjs, trans)
    assert info["same_transition_diff_subject"] > 0.9
    assert abs(info["across_relation"]) < 0.2
    # null: isotropic random unit vectors
    R = RNG.standard_normal((n, dim))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    null = _stats_for(R, rels, subjs, trans)
    assert abs(null["same_transition_diff_subject"]) < 0.3
    assert null["global_mean_direction_norm"] < info["global_mean_direction_norm"]
