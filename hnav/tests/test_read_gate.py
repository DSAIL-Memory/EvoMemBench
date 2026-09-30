"""Read-gate numerics and logic, against constructed answers.  [T9]

Every geometric quantity is checked on embeddings whose conflict structure is
known *by construction* (exact basis vectors, hand-computed cosines and
residuals), and every NLI-stage behaviour is checked in closed form against
:class:`StubNLI` — both-directions-required logic, inclusive threshold
boundary, asymmetric rejection. A null regime (orthonormal candidates) is
asserted to produce nothing, matching the repo's testing philosophy: a gate
that cannot stay silent is decoration.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hnav.core.read_gate import (COS_PAIR_CAL, GateThresholds, NLIScores,
                                 ReadGate, StubNLI)
from hnav.core.retrieval_signals import RetrievalSignalSet
from hnav.core.types import MemoryRecord

# Signal fixtures: frozen Stage-0 thresholds are nmargin<0.0048, H_z>1.9569.
AMBIG = {"nmargin": 0.0, "H_z": 3.0}          # both screens fire
CALM = {"nmargin": 0.5, "H_z": 0.5}           # neither fires

# Same template, different slot — StubNLI's contradiction shape (ratio 0.833).
T_LONDON = "The capital of Xanadu is London."
T_MADRID = "The capital of Xanadu is Madrid."
T_RED = "The colour of the door is red."
T_BLUE = "The colour of the door is blue."
T_GREEN = "The colour of the door is green."
T_NOISE1 = "zebra quantum mist over the harbour"
T_NOISE2 = "seventeen bicycles dreamed of rain"


def _rec(rid: str, text: str, vec, version: int = 0) -> MemoryRecord:
    v = np.asarray(vec, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return MemoryRecord(id=rid, text=text, vector=v.astype(np.float32),
                        version=version)


def _e(i: int, dim: int = 6) -> np.ndarray:
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def _pair_at(cos: float, dim: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors with cosine exactly ``cos``, in the e0/e1 plane."""
    return _e(0, dim), cos * _e(0, dim) + np.sqrt(1 - cos ** 2) * _e(1, dim)


class CountingStub(StubNLI):
    """StubNLI that counts directed pairs scored — for stage-order assertions."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_scored = 0

    def score_pairs(self, pairs):
        self.n_scored += len(pairs)
        return super().score_pairs(pairs)


# ── stage 1: geometry, closed form ───────────────────────────────────────────
def test_null_regime_orthonormal_candidates_produce_nothing():
    recs = [_rec(f"c{i}", f"text {i}", _e(i)) for i in range(5)]
    dec = ReadGate(StubNLI()).decide(recs, AMBIG)
    assert dec.ambiguous is True
    assert dec.n_pairs_screened == 10
    assert dec.n_pairs_cos == 0
    assert dec.n_groups_cos == 0 and dec.n_groups_geometric == 0
    assert dec.groups == [] and dec.pair_checks == []


def test_constructed_conflict_pair_is_found_verified_and_latest_named():
    a_vec, b_vec = _pair_at(0.99)
    recs = [
        _rec("old", T_LONDON, a_vec, version=3),
        _rec("new", T_MADRID, b_vec, version=7),
        _rec("d1", T_NOISE1, _e(2)),
        _rec("d2", T_NOISE2, _e(3)),
    ]
    dec = ReadGate(StubNLI()).decide(recs, AMBIG)

    assert dec.n_pairs_cos == 1
    assert dec.n_groups_geometric == 1
    assert dec.n_pairs_verified == 1
    assert len(dec.groups) == 1

    g = dec.groups[0]
    assert g.member_ids == ["old", "new"]
    assert g.latest_id == "new"                  # version 7 > 3
    assert g.stale_ids == ["old"]
    # Leave-one-out residual of e0 against span{b, e2, e3}: only b has an e0
    # component (0.99), so r = sqrt(1 - 0.99^2) — hand-computed.
    assert g.residuals["old"] == pytest.approx(np.sqrt(1 - 0.99 ** 2), abs=1e-4)

    (pc,) = dec.pair_checks
    assert pc.verified is True
    assert pc.cos == pytest.approx(0.99, abs=1e-6)
    assert pc.contra_ab >= 0.5 and pc.contra_ba >= 0.5


def test_r_screen_rejects_similar_but_not_span_redundant_pair():
    """cos 0.93 passes the cosine screen (0.92) but its leave-one-out residual
    sqrt(1 - 0.93^2) = 0.3676 exceeds the frozen r_min 0.1924 — the group must
    die at the R screen, BEFORE any NLI call is spent on it."""
    assert 0.93 >= COS_PAIR_CAL
    a_vec, b_vec = _pair_at(0.93)
    recs = [
        _rec("a", T_LONDON, a_vec),
        _rec("b", T_MADRID, b_vec),
        _rec("d1", T_NOISE1, _e(2)),
        _rec("d2", T_NOISE2, _e(3)),
    ]
    nli = CountingStub()
    dec = ReadGate(nli).decide(recs, AMBIG)
    assert dec.n_pairs_cos == 1
    assert dec.n_groups_cos == 1
    assert dec.n_groups_geometric == 0
    assert dec.n_pairs_nli == 0 and nli.n_scored == 0
    assert dec.groups == []


def test_member_inside_the_span_passes_r_screen_with_zero_residual():
    a_vec, b_vec = _pair_at(0.99)
    c_vec = a_vec + b_vec                       # exactly in span{a, b}
    recs = [
        _rec("v1", T_RED, a_vec, version=1),
        _rec("v2", T_BLUE, b_vec, version=2),
        _rec("v3", T_GREEN, c_vec, version=3),
    ]
    dec = ReadGate(StubNLI()).decide(recs, AMBIG)

    assert dec.n_groups_geometric == 1
    assert len(dec.groups) == 1
    g = dec.groups[0]
    assert g.member_ids == ["v1", "v2", "v3"]
    assert g.residuals["v3"] == pytest.approx(0.0, abs=1e-5)
    assert g.latest_id == "v3" and set(g.stale_ids) == {"v1", "v2"}
    assert dec.n_pairs_nli == 3                 # all within-group pairs checked


# ── stage 2: bidirectional NLI, closed form ──────────────────────────────────
def test_asymmetric_pair_one_direction_entailment_is_not_verified():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec), _rec("b", T_MADRID, b_vec)]
    nli = StubNLI(overrides={
        (T_LONDON, T_MADRID): (0.03, 0.07, 0.90),   # A=>B: contradiction
        (T_MADRID, T_LONDON): (0.90, 0.05, 0.05),   # B=>A: entailment
    })
    dec = ReadGate(nli).decide(recs, AMBIG)
    assert dec.n_pairs_nli == 1
    assert dec.n_pairs_verified == 0
    assert dec.groups == []
    (pc,) = dec.pair_checks
    assert pc.verified is False
    assert pc.contra_ab == pytest.approx(0.90)
    assert pc.entail_ba == pytest.approx(0.90)


def test_nli_threshold_boundary_is_inclusive_and_both_directions_required():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=1),
            _rec("b", T_MADRID, b_vec, version=2)]
    thr = GateThresholds(nli_contradiction=0.9)

    at_boundary = StubNLI(overrides={
        (T_LONDON, T_MADRID): (0.02, 0.08, 0.90),
        (T_MADRID, T_LONDON): (0.02, 0.08, 0.90),
    })
    dec = ReadGate(at_boundary, thr).decide(recs, AMBIG)
    assert dec.n_pairs_verified == 1, "score == threshold must pass (>=)"

    just_below = StubNLI(overrides={
        (T_LONDON, T_MADRID): (0.02, 0.08, 0.90),
        (T_MADRID, T_LONDON): (0.02, 0.08, 0.9 - 1e-9),
    })
    dec2 = ReadGate(just_below, thr).decide(recs, AMBIG)
    assert dec2.n_pairs_verified == 0, "one direction below threshold must fail"


# ── LATEST identification ────────────────────────────────────────────────────
def test_latest_key_callable_overrides_version_and_ties_refuse_to_guess():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=3),
            _rec("b", T_MADRID, b_vec, version=7)]
    gate = ReadGate(StubNLI())

    flipped = gate.decide(recs, AMBIG, latest_key=lambda r: -r.version)
    assert flipped.groups[0].latest_id == "a"
    assert flipped.groups[0].stale_ids == ["b"]

    tied = [_rec("a", T_LONDON, a_vec, version=5),
            _rec("b", T_MADRID, b_vec, version=5)]
    g = gate.decide(tied, AMBIG).groups[0]
    assert g.latest_id is None and g.stale_ids == []
    assert "tied" in g.note

    missing = gate.decide(recs, AMBIG, latest_key=lambda r: None).groups[0]
    assert missing.latest_id is None and missing.stale_ids == []
    assert "missing" in missing.note


# ── ambiguity precondition ───────────────────────────────────────────────────
def test_calm_ranking_disarms_the_gate_entirely():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec), _rec("b", T_MADRID, b_vec)]
    nli = CountingStub()
    dec = ReadGate(nli).decide(recs, CALM)
    assert dec.ambiguous is False
    assert dec.groups == [] and dec.n_pairs_screened == 0
    assert nli.n_scored == 0


def test_ambiguity_mode_any_fires_on_either_signal_alone():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec), _rec("b", T_MADRID, b_vec)]
    gate = ReadGate(StubNLI())
    assert gate.decide(recs, {"nmargin": 0.0, "H_z": 0.5}).ambiguous is True
    assert gate.decide(recs, {"nmargin": 0.5, "H_z": 3.0}).ambiguous is True


def test_ambiguity_mode_all_requires_both_signals():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec), _rec("b", T_MADRID, b_vec)]
    gate = ReadGate(StubNLI(), GateThresholds(ambiguity_mode="all"))
    assert gate.decide(recs, {"nmargin": 0.0, "H_z": 0.5}).ambiguous is False
    assert gate.decide(recs, {"nmargin": 0.0, "H_z": 3.0}).ambiguous is True


def test_ambiguity_mode_none_needs_no_signals():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=1),
            _rec("b", T_MADRID, b_vec, version=2)]
    dec = ReadGate(StubNLI(), GateThresholds(ambiguity_mode="none")).decide(recs)
    assert dec.ambiguous is True and len(dec.groups) == 1


def test_missing_signals_with_active_precondition_is_an_error():
    recs = [_rec("a", T_LONDON, _e(0))]
    with pytest.raises(ValueError, match="ambiguity"):
        ReadGate(StubNLI()).decide(recs, None)


def test_forbidden_raw_entropy_signal_is_refused():
    recs = [_rec("a", T_LONDON, _e(0))]
    with pytest.raises(ValueError, match="forbidden"):
        ReadGate(StubNLI()).decide(recs, {"nmargin": 0.0, "H_z": 3.0, "H_raw": 1.0})


def test_retrieval_signal_set_is_consumed_via_its_policy_view():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=1),
            _rec("b", T_MADRID, b_vec, version=2)]
    sig = RetrievalSignalSet(nmargin=0.0, H_z=3.0, H_raw=0.123)
    dec = ReadGate(StubNLI()).decide(recs, sig)
    assert dec.ambiguous is True and len(dec.groups) == 1


def test_invalid_ambiguity_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="ambiguity_mode"):
        GateThresholds(ambiguity_mode="sometimes")


# ── neutrality and hygiene ───────────────────────────────────────────────────
def test_gate_mutates_neither_records_nor_signals():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=1),
            _rec("b", T_MADRID, b_vec, version=2),
            _rec("d", T_NOISE1, _e(4))]
    before_vecs = [r.vector.copy() for r in recs]
    before_ids = [r.id for r in recs]
    sig = dict(AMBIG)

    ReadGate(StubNLI()).decide(recs, sig)

    assert [r.id for r in recs] == before_ids
    for r, v in zip(recs, before_vecs):
        assert np.array_equal(r.vector, v)
    assert sig == AMBIG


def test_fewer_than_two_candidates_is_trivially_quiet():
    dec = ReadGate(StubNLI()).decide([_rec("a", T_LONDON, _e(0))], AMBIG)
    assert dec.ambiguous is True
    assert dec.n_pairs_screened == 0 and dec.groups == []


def test_candidates_without_vectors_are_refused():
    recs = [MemoryRecord(id="a", text=T_LONDON, vector=None),
            MemoryRecord(id="b", text=T_MADRID, vector=None)]
    with pytest.raises(ValueError, match="vector"):
        ReadGate(StubNLI()).decide(recs, AMBIG)


def test_disabled_screens_pass_everything_to_nli():
    words = ["alpha bravo charlie", "delta echo foxtrot",
             "golf hotel india", "juliett kilo lima"]     # pairwise-disjoint
    recs = [_rec(f"c{i}", words[i], _e(i)) for i in range(4)]
    thr = GateThresholds(cos_pair=None, r_min=None, ambiguity_mode="none")
    dec = ReadGate(StubNLI(), thr).decide(recs)
    assert dec.n_pairs_cos == dec.n_pairs_screened == 6
    assert dec.n_groups_geometric == 1          # one component, R screen off
    assert dec.n_pairs_nli == 6
    assert dec.groups == []                     # stub: unrelated pairs are neutral


def test_decision_serializes_to_json():
    a_vec, b_vec = _pair_at(0.99)
    recs = [_rec("a", T_LONDON, a_vec, version=1),
            _rec("b", T_MADRID, b_vec, version=2)]
    dec = ReadGate(StubNLI()).decide(recs, AMBIG)
    dumped = json.loads(json.dumps(dec.to_dict()))
    assert dumped["groups"][0]["latest_id"] == "b"
    assert dumped["reasons"]["thresholds"]["nmargin"] == pytest.approx(0.0048, abs=1e-4)


# ── StubNLI itself: closed-form rules ────────────────────────────────────────
def test_stub_nli_rules():
    nli = StubNLI()
    assert nli.score(T_LONDON, T_LONDON).label == "entailment"
    assert nli.score(T_LONDON, T_MADRID).label == "contradiction"
    assert nli.score(T_MADRID, T_LONDON).label == "contradiction"
    assert nli.score(T_LONDON, T_NOISE1).label == "neutral"

    a = nli.score(T_LONDON, T_MADRID)
    b = nli.score(T_LONDON, T_MADRID)
    assert a == b, "StubNLI must be a pure function of its inputs"

    forced = StubNLI(overrides={(T_LONDON, T_MADRID): (0.9, 0.05, 0.05)})
    assert forced.score(T_LONDON, T_MADRID).label == "entailment", "override wins"
    assert forced.score(T_MADRID, T_LONDON).label == "contradiction", \
        "non-overridden direction follows the rules"


def test_nli_scores_label_argmax():
    assert NLIScores(0.7, 0.2, 0.1).label == "entailment"
    assert NLIScores(0.1, 0.7, 0.2).label == "neutral"
    assert NLIScores(0.1, 0.2, 0.7).label == "contradiction"
