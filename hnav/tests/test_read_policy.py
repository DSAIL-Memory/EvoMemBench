"""Rerank policy: every permutation checked against a hand-written answer.  [T11]

The policy's whole job is a permutation, so every test states the expected
output list literally — no "it changed something" assertions. Token
neutrality (same ids, same count) is asserted in every case, and the
non-permutation guard is exercised directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.read_gate import (ConflictGroup, GateDecision, GateThresholds,
                                 ReadGate, StubNLI)
from hnav.core.read_policy import ReadRerankPolicy, rerank_order, stage1_thresholds
from hnav.core.types import MemoryRecord

# ── fixtures ─────────────────────────────────────────────────────────────────
PAGE = ["c0", "c1", "c2", "c3", "c4"]


def _decision(groups: list[ConflictGroup]) -> GateDecision:
    return GateDecision(ambiguous=True, groups=groups)


def _group(latest, stale, note=""):
    members = ([latest] if latest else []) + list(stale)
    return ConflictGroup(member_ids=members, latest_id=latest,
                         stale_ids=list(stale), residuals={}, note=note)


# ── rerank_order: closed form ────────────────────────────────────────────────
def test_no_groups_is_identity_and_a_new_list():
    dec = _decision([])
    out = rerank_order(PAGE, dec)
    assert out == PAGE and out is not PAGE


def test_single_promotion_exact_permutation():
    # latest c3 must land immediately above its first stale rival c1.
    dec = _decision([_group("c3", ["c1"])])
    assert rerank_order(PAGE, dec) == ["c0", "c3", "c1", "c2", "c4"]


def test_latest_already_above_every_rival_is_identity():
    dec = _decision([_group("c0", ["c2", "c4"])])
    assert rerank_order(PAGE, dec) == PAGE


def test_latest_id_none_group_is_untouched():
    dec = _decision([_group(None, [], note="recency key tied at the maximum")])
    assert rerank_order(PAGE, dec) == PAGE


def test_promotion_targets_the_highest_ranked_stale_rival():
    # rivals at positions 0 and 2; latest must go to position 0, not 2.
    dec = _decision([_group("c4", ["c2", "c0"])])
    assert rerank_order(PAGE, dec) == ["c4", "c0", "c1", "c2", "c3"]


def test_two_groups_applied_in_first_stale_position_order():
    # g2 (rival c0 at 0) applies before g1 (rival c1 at 1):
    #   apply g2: [d,a,b,c,e] ; apply g1: e above b -> [d,a,e,b,c]
    page = ["a", "b", "c", "d", "e"]
    dec = _decision([_group("e", ["b"]), _group("d", ["a"])])
    assert rerank_order(page, dec) == ["d", "a", "e", "b", "c"]


def test_id_map_projects_facts_onto_chunks():
    fact_chunk = {"fact:9": "c3", "fact:2": "c1"}
    dec = _decision([_group("fact:9", ["fact:2"])])
    out = rerank_order(PAGE, dec, id_map=fact_chunk.get)
    assert out == ["c0", "c3", "c1", "c2", "c4"]


def test_intra_unit_conflict_is_a_no_op():
    # latest and stale live in the SAME chunk — nothing to reorder.
    fact_chunk = {"fact:9": "c2", "fact:2": "c2"}
    dec = _decision([_group("fact:9", ["fact:2"])])
    assert rerank_order(PAGE, dec, id_map=fact_chunk.get) == PAGE


def test_unmapped_and_off_page_members_are_skipped():
    fact_chunk = {"fact:9": "c9-not-on-page", "fact:2": "c1"}
    dec = _decision([_group("fact:9", ["fact:2"]),          # latest off page
                     _group("fact:7", ["fact:1"])])         # both unmapped
    assert rerank_order(PAGE, dec, id_map=fact_chunk.get) == PAGE


def test_duplicate_page_ids_are_refused():
    dec = _decision([_group("c3", ["c1"])])
    with pytest.raises(ValueError, match="duplicate"):
        rerank_order(["c0", "c1", "c1"], dec)


def test_every_case_is_token_neutral():
    cases = [
        _decision([_group("c3", ["c1"])]),
        _decision([_group("c4", ["c0", "c2"]), _group("c3", ["c1"])]),
        _decision([_group("c2", ["c0"]), _group("c4", ["c1", "c3"])]),
    ]
    for dec in cases:
        out = rerank_order(PAGE, dec)
        assert sorted(out) == sorted(PAGE) and len(out) == len(PAGE)


# ── ReadRerankPolicy end to end (StubNLI, constructed vectors) ───────────────
T_OLD = "The capital of Xanadu is London."
T_NEW = "The capital of Xanadu is Madrid."
T_NOISE = "zebra quantum mist over the harbour"

AMBIG = {"nmargin": 0.0, "H_z": 3.0}
CALM = {"nmargin": 0.5, "H_z": 0.5}


def _rec(rid, text, vec, version=0, key=None):
    v = np.asarray(vec, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return MemoryRecord(id=rid, text=text, vector=v.astype(np.float32),
                        version=version, metadata={"key": key})


def _e(i, dim=6):
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def _pair_at(cos, dim=6):
    return _e(0, dim), cos * _e(0, dim) + np.sqrt(1 - cos ** 2) * _e(1, dim)


def _candidates():
    a, b = _pair_at(0.99)
    return [
        _rec("fact:3", T_OLD, a, version=3, key=("|cap", "Xanadu")),
        _rec("fact:7", T_NEW, b, version=7, key=("|cap", "Xanadu")),
        _rec("fact:1", T_NOISE, _e(2), version=1, key=("|noise", "z")),
    ]


FACT_CHUNK = {"fact:3": "chunkA", "fact:7": "chunkB", "fact:1": "chunkA"}
ORDERING = ["chunkA", "chunkB"]


def test_policy_promotes_the_latest_carrier_chunk():
    policy = ReadRerankPolicy(ReadGate(StubNLI()))
    dec = policy.decide(_candidates(), ORDERING, AMBIG,
                        latest_key=lambda r: r.version, id_map=FACT_CHUNK.get)
    assert dec.action == "RERANK"
    assert dec.payload == {"order": ["chunkB", "chunkA"]}
    assert dec.shadow is True, "the policy itself must never arm a decision"
    assert dec.reasons["gate"]["n_pairs_verified"] == 1
    assert dec.reasons["groups"][0]["latest_id"] == "fact:7"


def test_policy_is_quiet_on_a_calm_ranking():
    policy = ReadRerankPolicy(ReadGate(StubNLI()))
    dec = policy.decide(_candidates(), ORDERING, CALM,
                        latest_key=lambda r: r.version, id_map=FACT_CHUNK.get)
    assert dec.action == "PASS" and dec.payload is None
    assert dec.reasons["gate"]["ambiguous"] is False


def test_policy_pass_when_latest_is_already_first():
    policy = ReadRerankPolicy(ReadGate(StubNLI()))
    dec = policy.decide(_candidates(), ["chunkB", "chunkA"], AMBIG,
                        latest_key=lambda r: r.version, id_map=FACT_CHUNK.get)
    assert dec.action == "PASS" and dec.payload is None


def test_policy_mutates_neither_ordering_nor_candidates():
    cands = _candidates()
    ordering = list(ORDERING)
    before = [(r.id, r.version) for r in cands]
    ReadRerankPolicy(ReadGate(StubNLI())).decide(
        cands, ordering, AMBIG, latest_key=lambda r: r.version,
        id_map=FACT_CHUNK.get)
    assert ordering == ORDERING
    assert [(r.id, r.version) for r in cands] == before


def test_stage1_thresholds_returns_a_gate_thresholds():
    """Value equality with its provenance artifact is enforced by
    test_threshold_provenance.py; here only the type/shape contract.

    The frozen ambiguity CONSTANTS must stay on the record even when the T13
    operating point runs the screen in ``ambiguity_mode="none"`` — dropping them
    would erase the trail of what was disabled. That the departure is argued in
    the committed artifact is asserted by
    ``test_threshold_provenance.py::test_disabling_the_ambiguity_screen_is_declared_in_the_artifact``.
    """
    thr = stage1_thresholds()
    assert isinstance(thr, GateThresholds)
    assert thr.nmargin is not None and thr.H_z is not None, \
        "the frozen ambiguity constants must stay recorded in live wiring"


# ── the gate's pair_filter screen (Note-1 mitigation seam) ───────────────────
def test_pair_filter_drops_cross_key_pairs_before_nli():
    """The audit's measured failure: same-template different-subject pairs are
    NLI-verified bidirectionally. With key-equality evidence supplied by the
    adapter the pair must die at stage 1a — zero NLI spent on it."""
    a, b = _pair_at(0.99)
    recs = [
        _rec("kyd", "Thomas Kyd was born in London.", a, 1, key=("|born", "Thomas Kyd")),
        _rec("marlowe", "Marlowe was born in London.", b, 2, key=("|born", "Marlowe")),
    ]

    class Counting(StubNLI):
        def __init__(self):
            super().__init__()
            self.n = 0

        def score_pairs(self, pairs):
            self.n += len(pairs)
            return super().score_pairs(pairs)

    same_key = lambda x, y: (x.metadata.get("key") is not None  # noqa: E731
                             and x.metadata.get("key") == y.metadata.get("key"))

    nli = Counting()
    dec = ReadGate(nli, pair_filter=same_key).decide(recs, AMBIG)
    assert dec.n_pairs_filter_rejected == 1
    assert dec.n_pairs_cos == 0 and dec.n_pairs_nli == 0 and nli.n == 0
    assert dec.groups == []
    assert dec.reasons["pair_filter_active"] is True

    # Without the filter the stub (overlap rule) rubber-stamps it — the
    # measured Note-1 failure shape, reproduced so the mitigation is testable.
    dec2 = ReadGate(Counting()).decide(recs, AMBIG)
    assert dec2.n_pairs_verified == 1 and len(dec2.groups) == 1
    assert dec2.reasons["pair_filter_active"] is False


def test_pair_filter_keeps_same_key_pairs():
    same_key = lambda x, y: x.metadata.get("key") == y.metadata.get("key")  # noqa: E731
    dec = ReadGate(StubNLI(), pair_filter=same_key).decide(_candidates(), AMBIG)
    assert dec.n_pairs_filter_rejected == 0
    assert dec.n_pairs_verified == 1
    assert dec.groups[0].latest_id == "fact:7"


def test_pair_filter_none_key_is_rejected_by_the_adapter_style_filter():
    a, b = _pair_at(0.99)
    recs = [_rec("x", T_OLD, a, 1, key=None), _rec("y", T_NEW, b, 2, key=None)]
    same_key = lambda p, q: (p.metadata.get("key") is not None  # noqa: E731
                             and p.metadata.get("key") == q.metadata.get("key"))
    dec = ReadGate(StubNLI(), pair_filter=same_key).decide(recs, AMBIG)
    assert dec.n_pairs_filter_rejected == 1 and dec.groups == []
