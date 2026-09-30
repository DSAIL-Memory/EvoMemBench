"""Fact-level SUPPRESS and DEMOTE_LATE, byte for byte.  [T13]

The mechanisms exist because the oracle probe measured the arena's failure to
be positional and per-FACT (sh_6k conflicted stratum: native 4/74, stale fact
removed 66/74, newest fact moved to the END 20/74, moved to the FRONT 1/74)
while the CHUNK-level version of the same idea is harmful. So the thing that
has to be right here is the *granularity and the exactness of the edit*, and
every test below writes the expected page out in full rather than asserting
that something changed:

* which ids each mechanism names, from hand-built gate decisions;
* what the page looks like afterwards, for the newline-joined raw context AND
  for a space-joined memorize chunk (the benchmark's chunker produces the
  second, and a line-anchored parse finds nothing in it);
* that serials are never renumbered and every byte outside the edited span
  survives — checked against the probe's own surgery on the real 455-fact
  ``sh_6k`` context, which is the arm construction the oracle numbers came from;
* the no-op paths (empty decision, ``latest_id=None``) and the fallback paths
  (unknown id, overlapping sets) — including ``HNAV_MODE=off`` neutrality and
  the shadow/live seam.

No socket, no torch, no model.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from hnav import config as _config
from hnav.adapters import mab_adapter as mab
from hnav.adapters.mab_adapter import (MABAdapter, explode_facts, fact_spans,
                                       page_edit)
from hnav.core.read_gate import (ConflictGroup, GateDecision, GateThresholds,
                                 ReadGate, StubNLI)
from hnav.core.read_policy import (FACT_MECHANISMS, ReadFactPolicy, demote_ids,
                                   suppress_ids)
from hnav.core.types import Decision, MemoryRecord
from hnav.stage1.stale_suppression_probe import (move_to_end, render_context,
                                                 split_context, suppress)

OFF = _config.HNavConfig(mode=_config.MODE_OFF)
SHADOW = _config.HNavConfig(mode=_config.MODE_SHADOW)
LIVE = _config.HNavConfig(mode=_config.MODE_LIVE)

# ── page fixtures ────────────────────────────────────────────────────────────
# (a) the raw dataset shape: preamble + one fact per line.
RAW = ("Here is a list of facts:\n"
       "0. The capital of Atlantis is Poseidonis.\n"
       "1. The chairperson of Fatah is Ada Lovelace.\n"
       "2. The capital of Atlantis is Kallipolis.\n")
# (b) what the benchmark's chunker actually hands the model: nltk splits the
#     context into sentences and rejoins them with SPACES, so FACT_RE alone
#     finds nothing (test_chunking_and_facts.py pins that).
JOINED = ("0. The capital of Atlantis is Poseidonis. "
          "1. The chairperson of Fatah is Ada Lovelace. "
          "2. The capital of Atlantis is Kallipolis.")
# (c) the chunker's dangling-serial tail: fact 3's number ends this chunk while
#     its text opens the next one.
DANGLING = JOINED + " 3."


def _groups(*specs) -> GateDecision:
    return GateDecision(ambiguous=True, groups=[
        ConflictGroup(member_ids=([lat] if lat else []) + list(stale),
                      latest_id=lat, stale_ids=list(stale), residuals={},
                      note=note)
        for lat, stale, note in specs])


# ── id selection: closed form ────────────────────────────────────────────────
def test_suppress_names_every_stale_member_and_no_latest():
    dec = _groups(("fact:9", ["fact:2", "fact:5"], ""),
                  ("fact:8", ["fact:1"], ""))
    assert suppress_ids(dec) == ["fact:2", "fact:5", "fact:1"]
    assert demote_ids(dec) == ["fact:9", "fact:8"]


def test_group_without_a_latest_is_skipped_by_both_mechanisms():
    dec = _groups((None, [], "recency key tied at the maximum"),
                  ("fact:9", ["fact:2"], ""))
    assert suppress_ids(dec) == ["fact:2"]
    assert demote_ids(dec) == ["fact:9"]


def test_empty_decision_names_nothing():
    dec = GateDecision(ambiguous=True)
    assert suppress_ids(dec) == [] and demote_ids(dec) == []


def test_duplicate_ids_collapse_to_one_edit():
    dec = _groups(("fact:9", ["fact:2", "fact:2"], ""))
    assert suppress_ids(dec) == ["fact:2"]


def test_overlapping_groups_are_refused_not_guessed_at():
    """fact:2 is a LATEST in one group and stale in another — the gate emits
    disjoint components, so this cannot happen; if it ever does, deleting the
    fact another group wants kept is the worst possible response."""
    dec = _groups(("fact:9", ["fact:2"], ""), ("fact:2", ["fact:1"], ""))
    with pytest.raises(ValueError, match="overlapping"):
        suppress_ids(dec)
    with pytest.raises(ValueError, match="overlapping"):
        demote_ids(dec)


# ── fact_spans ───────────────────────────────────────────────────────────────
def test_spans_cover_the_raw_context_exactly():
    sp = fact_spans(RAW)
    assert [s.serial for s in sp] == [0, 1, 2]
    assert [RAW[s.start:s.own_end] for s in sp] == [
        "0. The capital of Atlantis is Poseidonis.",
        "1. The chairperson of Fatah is Ada Lovelace.",
        "2. The capital of Atlantis is Kallipolis."]
    # a fact owns the separator AFTER it — except the last, which owns the one
    # BEFORE it, so deleting it cannot leave a blank line behind.
    assert RAW[sp[0].del_start:sp[0].del_end] == \
        "0. The capital of Atlantis is Poseidonis.\n"
    assert RAW[sp[2].del_start:sp[2].del_end] == \
        "\n2. The capital of Atlantis is Kallipolis."


def test_spans_work_on_the_space_joined_chunk_the_chunker_produces():
    sp = fact_spans(JOINED)
    assert [s.serial for s in sp] == [0, 1, 2]
    assert JOINED[sp[1].start:sp[1].own_end] == \
        "1. The chairperson of Fatah is Ada Lovelace."
    assert JOINED[sp[2].del_start:sp[2].del_end] == \
        " 2. The capital of Atlantis is Kallipolis."


def test_the_dangling_next_fact_serial_is_outside_every_span():
    sp = fact_spans(DANGLING)
    assert [s.serial for s in sp] == [0, 1, 2], "the ' 3.' tail is not a fact here"
    assert DANGLING[sp[-1].own_end:] == " 3."
    # deleting the last real fact must leave those four bytes alone: they
    # belong to a fact whose text lives in the NEXT chunk.
    out, _ = page_edit([DANGLING], drop_ids=["fact:2"])
    assert out[0] == ("0. The capital of Atlantis is Poseidonis. "
                      "1. The chairperson of Fatah is Ada Lovelace. 3.")


def test_spans_agree_with_explode_facts_on_the_real_chunks(sh_6k):
    from hnav.stage0.m2_retrieval_calibration import build_chunks
    chunks, fallback = build_chunks(sh_6k["context"], 4096)
    assert not fallback, "nltk/punkt missing — this is not the benchmark's chunker"
    for c in chunks:
        sp, ex = fact_spans(c), explode_facts(c)
        assert [s.serial for s in sp] == [n for n, _ in ex]
        assert [c[s.start:s.own_end] for s in sp] == \
            [f"{n}. {t}" for n, t in ex]


# ── page_edit: expected pages written out ────────────────────────────────────
def test_suppress_middle_fact_exact_page():
    out, rep = page_edit([RAW], drop_ids=["fact:1"])
    assert out == ["Here is a list of facts:\n"
                   "0. The capital of Atlantis is Poseidonis.\n"
                   "2. The capital of Atlantis is Kallipolis.\n"]
    assert rep["dropped_serials"] == [1] and rep["delta_chars"] < 0


def test_suppress_keeps_original_serials_with_gaps():
    out, _ = page_edit([RAW], drop_ids=["fact:0", "fact:1"])
    assert out == ["Here is a list of facts:\n"
                   "2. The capital of Atlantis is Kallipolis.\n"]
    assert [s.serial for s in fact_spans(out[0])] == [2], "no renumbering"


def test_suppress_last_fact_leaves_no_blank_line():
    out, _ = page_edit([RAW], drop_ids=["fact:2"])
    assert out == ["Here is a list of facts:\n"
                   "0. The capital of Atlantis is Poseidonis.\n"
                   "1. The chairperson of Fatah is Ada Lovelace.\n"]


def test_suppress_on_the_joined_chunk_rejoins_with_the_chunker_separator():
    out, rep = page_edit([JOINED], drop_ids=["fact:0"])
    assert out == ["1. The chairperson of Fatah is Ada Lovelace. "
                   "2. The capital of Atlantis is Kallipolis."]
    assert rep["separator"] == " "


def test_demote_moves_one_fact_to_the_end_and_is_token_neutral():
    out, rep = page_edit([RAW], move_last_ids=["fact:1"])
    assert out == ["Here is a list of facts:\n"
                   "0. The capital of Atlantis is Poseidonis.\n"
                   "2. The capital of Atlantis is Kallipolis.\n"
                   "1. The chairperson of Fatah is Ada Lovelace.\n"]
    assert rep["delta_chars"] == 0
    assert sorted(explode_facts(out[0])) == sorted(explode_facts(RAW))


def test_demoting_the_already_last_fact_is_byte_identical():
    out, rep = page_edit([RAW], move_last_ids=["fact:2"])
    assert out == [RAW] and rep["delta_chars"] == 0


def test_demote_appends_in_ascending_serial_so_the_newest_ends_last():
    out, rep = page_edit([RAW], move_last_ids=["fact:1", "fact:0"])
    assert out == ["Here is a list of facts:\n"
                   "2. The capital of Atlantis is Kallipolis.\n"
                   "0. The capital of Atlantis is Poseidonis.\n"
                   "1. The chairperson of Fatah is Ada Lovelace.\n"]
    assert rep["moved_serials"] == [0, 1]


def test_empty_edit_returns_the_page_unchanged():
    out, rep = page_edit([RAW, JOINED])
    assert out == [RAW, JOINED] and rep["delta_chars"] == 0


# ── multi-chunk pages ────────────────────────────────────────────────────────
CH_A = "0. Alpha is one. 1. Beta is two."
CH_B = "2. Gamma is three. 3. Delta is four."


def test_suppress_edits_only_the_chunk_that_carries_the_fact():
    out, rep = page_edit([CH_A, CH_B], drop_ids=["fact:1"])
    assert out == ["0. Alpha is one.", CH_B]
    assert rep["n_chunks"] == 2 and len(out) == 2, "page shape is preserved"


def test_demote_moves_a_fact_across_chunks_to_the_page_end():
    out, rep = page_edit([CH_A, CH_B], move_last_ids=["fact:0"])
    assert out == ["1. Beta is two.",
                   "2. Gamma is three. 3. Delta is four. 0. Alpha is one."]
    assert rep["delta_chars"] == 0


def test_unknown_id_is_a_lookup_error_and_overlap_is_a_value_error():
    with pytest.raises(LookupError, match="not on this page"):
        page_edit([CH_A], drop_ids=["fact:99"])
    with pytest.raises(ValueError, match="both drop and move"):
        page_edit([CH_A], drop_ids=["fact:0"], move_last_ids=["fact:0"])


# ── equivalence with the probe's own surgery on the real corpus ──────────────
def test_page_edit_reproduces_the_probe_arms_on_the_real_sh_6k_context(sh_6k):
    """The oracle numbers this whole task is calibrated against came from the
    probe's ``suppress``/``move_to_end`` on the raw context. If the shipped
    page edit is not byte-identical to those arms, the detector/oracle ratio
    compares two different interventions."""
    preamble, facts = split_context(sh_6k["context"])
    serials = [s for s, _ in facts]
    rng = random.Random(20260815)
    for _ in range(60):
        chosen = rng.sample(serials, rng.choice([1, 2, 3]))
        got, _ = page_edit([sh_6k["context"]],
                           drop_ids=[f"fact:{s}" for s in chosen])
        assert got[0] == render_context(preamble, suppress(facts, chosen))
        one = rng.choice(serials)
        got, rep = page_edit([sh_6k["context"]], move_last_ids=[f"fact:{one}"])
        assert got[0] == render_context(preamble, move_to_end(facts, one))
        assert rep["delta_chars"] == 0


# ── ReadFactPolicy end to end (StubNLI, constructed vectors) ─────────────────
T_OLD = "The capital of Xanadu is London."
T_NEW = "The capital of Xanadu is Madrid."
T_NOISE = "zebra quantum mist over the harbour"
AMBIG = {"nmargin": 0.0, "H_z": 3.0}
CALM = {"nmargin": 0.5, "H_z": 0.5}


def _rec(rid, text, vec, version=0, key=None):
    v = np.asarray(vec, dtype=np.float64)
    return MemoryRecord(id=rid, text=text,
                        vector=(v / np.linalg.norm(v)).astype(np.float32),
                        version=version, metadata={"key": key})


def _candidates():
    e0, e1, e2 = np.eye(6)[0], np.eye(6)[1], np.eye(6)[2]
    b = 0.99 * e0 + np.sqrt(1 - 0.99 ** 2) * e1
    return [_rec("fact:3", T_OLD, e0, 3, ("|cap", "Xanadu")),
            _rec("fact:7", T_NEW, b, 7, ("|cap", "Xanadu")),
            _rec("fact:1", T_NOISE, e2, 1, ("|noise", "z"))]


def test_fact_policy_suppress_emits_the_stale_id_only():
    pol = ReadFactPolicy(ReadGate(StubNLI()), mechanism="suppress")
    dec = pol.decide(_candidates(), ordering=["chunkA"], signal_view=AMBIG,
                     latest_key=lambda r: r.version)
    assert dec.action == "SUPPRESS"
    assert dec.payload == {"drop_ids": ["fact:3"]}
    assert dec.shadow is True, "the policy itself must never arm a decision"
    assert dec.reasons["mechanism"] == "suppress" and dec.reasons["n_ids"] == 1


def test_fact_policy_demote_emits_the_latest_id_only():
    pol = ReadFactPolicy(ReadGate(StubNLI()), mechanism="demote_late")
    dec = pol.decide(_candidates(), signal_view=AMBIG,
                     latest_key=lambda r: r.version)
    assert dec.action == "DEMOTE_LATE"
    assert dec.payload == {"move_last_ids": ["fact:7"]}


def test_fact_policy_passes_on_a_calm_ranking():
    for mech in FACT_MECHANISMS:
        dec = ReadFactPolicy(ReadGate(StubNLI()), mech).decide(
            _candidates(), signal_view=CALM, latest_key=lambda r: r.version)
        assert dec.action == "PASS" and dec.payload is None


def test_fact_policy_mutates_nothing():
    cands = _candidates()
    before = [(r.id, r.version, r.text) for r in cands]
    ReadFactPolicy(ReadGate(StubNLI())).decide(
        cands, signal_view=AMBIG, latest_key=lambda r: r.version)
    assert [(r.id, r.version, r.text) for r in cands] == before


def test_unknown_mechanism_is_refused_at_construction():
    with pytest.raises(ValueError, match="mechanism"):
        ReadFactPolicy(ReadGate(StubNLI()), mechanism="inject")


# ── the adapter seam: off / shadow / live and the fallback ───────────────────
def _scored(texts):
    class _Doc:
        def __init__(self, t): self.page_content = t
    return [(_Doc(t), 0.0) for t in texts]


def _adapter(cfg, page):
    a = MABAdapter(cfg=cfg)
    for i, t in enumerate(page):
        a._register_chunk(t, i)
    return a


@pytest.mark.parametrize("action,payload", [
    ("SUPPRESS", {"drop_ids": ["fact:1"]}),
    ("DEMOTE_LATE", {"move_last_ids": ["fact:0"]}),
])
def test_off_and_shadow_modes_return_the_native_page(action, payload):
    page = [CH_A, CH_B]
    for cfg in (OFF, SHADOW):
        a = _adapter(cfg, page)
        d = Decision(action=action, payload=payload, shadow=True)
        assert a.apply_read_decision(d, _scored(page), 2) == page
    # live but still shadow (never armed) is equally a no-op
    a = _adapter(LIVE, page)
    d = Decision(action=action, payload=payload, shadow=True)
    assert a.apply_read_decision(d, _scored(page), 2) == page
    assert a.n_fact_edits_applied == 0


def test_live_armed_suppress_edits_the_page_and_counts_tokens():
    page = [CH_A, CH_B]
    a = _adapter(LIVE, page)
    d = Decision(action="SUPPRESS", payload={"drop_ids": ["fact:1"]}, shadow=False)
    out = a.apply_read_decision(d, _scored(page), 2)
    assert out == ["0. Alpha is one.", CH_B]
    assert a.n_fact_edits_applied == 1 and a.n_facts_suppressed == 1
    assert a.chars_removed == len(" 1. Beta is two.")
    assert d.reasons["page_edit"]["dropped_serials"] == [1]


def test_live_armed_demote_is_token_neutral():
    page = [CH_A, CH_B]
    a = _adapter(LIVE, page)
    d = Decision(action="DEMOTE_LATE", payload={"move_last_ids": ["fact:0"]},
                 shadow=False)
    out = a.apply_read_decision(d, _scored(page), 2)
    assert out == ["1. Beta is two.",
                   "2. Gamma is three. 3. Delta is four. 0. Alpha is one."]
    assert a.n_facts_demoted == 1 and a.chars_removed == 0


def test_an_id_off_the_page_falls_back_to_native_and_is_counted():
    page = [CH_A, CH_B]
    a = _adapter(LIVE, page)
    d = Decision(action="SUPPRESS", payload={"drop_ids": ["fact:404"]}, shadow=False)
    assert a.apply_read_decision(d, _scored(page), 2) == page
    assert a.n_rerank_fallbacks == 1 and a.n_fact_edits_applied == 0
    assert "page_edit_error" in d.reasons


def test_an_empty_payload_is_a_no_op_without_counting_a_fallback():
    page = [CH_A, CH_B]
    a = _adapter(LIVE, page)
    d = Decision(action="SUPPRESS", payload={"drop_ids": []}, shadow=False)
    assert a.apply_read_decision(d, _scored(page), 2) == page
    assert a.n_rerank_fallbacks == 0 and a.n_fact_edits_applied == 0


def test_the_rerank_seam_still_works_unchanged():
    """T11's mechanism is evidence, not legacy — the T13 dispatch must not
    have disturbed it."""
    page = [CH_A, CH_B]
    a = _adapter(LIVE, page)
    ids = [a._chunk_by_text[t].id for t in page]
    d = Decision(action="RERANK", payload={"order": [ids[1], ids[0]]}, shadow=False)
    assert a.apply_read_decision(d, _scored(page), 2) == [CH_B, CH_A]
    assert a.n_reranks_applied == 1


def test_build_components_selects_the_configured_mechanism(monkeypatch):
    """``HNAV_READ_MECHANISM`` picks the live policy class; the default is the
    pre-T13 rerank so that adding these mechanisms changes no existing command."""
    from hnav.core import read_policy as rp

    class _FakeGate:
        def __init__(self, *a, **k): pass

    monkeypatch.setattr(mab, "_build_components", mab._build_components)
    monkeypatch.setattr("hnav.core.read_gate.build_nli", lambda cfg: StubNLI())
    for mech, cls in (("rerank", rp.ReadRerankPolicy),
                      ("suppress", rp.ReadFactPolicy),
                      ("demote_late", rp.ReadFactPolicy)):
        cfg = _config.HNavConfig(mode=_config.MODE_SHADOW, read_mechanism=mech)
        pol = mab._build_components(cfg).get("read_policy")
        assert isinstance(pol, cls), mech
        if mech != "rerank":
            assert pol.mechanism == mech
    bad = _config.HNavConfig(mode=_config.MODE_SHADOW, read_mechanism="inject")
    assert mab._build_components(bad).get("read_policy") is None, \
        "an unknown mechanism must degrade to no policy, never to a guess"


# ── the full live seam: memorize -> on_retrieve -> gate -> policy -> page ────
# Same fixture shape as test_live_read_path.py, which pins the rerank seam:
# chunk A carries the STALE fact (serial 1) plus an unrelated distractor, chunk B
# carries the superseder (serial 3). HashEmbedder vectors are near-orthogonal, so
# the geometric screens are disabled and verification rides on the NLI stage plus
# the adapter's key-equality filter — each closed-form tested elsewhere.
SEAM_A = ("1. Thomas Kyd was born in the city of London. "
          "2. The chairperson of Fatah is Mahmoud Abbas.")
SEAM_B = "3. Thomas Kyd was born in the city of Madrid."
NLI_ONLY = GateThresholds(cos_pair=None, r_min=None, ambiguity_mode="none")


def _seam_adapter(cfg, embedder, mechanism):
    policy = ReadFactPolicy(
        ReadGate(StubNLI(), NLI_ONLY, pair_filter=MABAdapter.same_key_pair),
        mechanism=mechanism)
    a = MABAdapter(cfg=cfg, embedder=embedder, read_policy=policy)
    a.before_memorize(SEAM_A, context_id=0, chunk_index=0)
    a.before_memorize(SEAM_B, context_id=0, chunk_index=1)
    return a


def _seam_retrieve(adapter, page):
    return adapter.on_retrieve(query="Where was Thomas Kyd born?",
                               ranked_texts=list(page), scores=[95.0, 90.0],
                               top_k=2, score_kind="similarity")


def test_live_seam_suppress_removes_the_stale_sentence(embedder):
    page = [SEAM_A, SEAM_B]
    a = _seam_adapter(LIVE, embedder, "suppress")
    dec = _seam_retrieve(a, page)
    assert dec.action == "SUPPRESS" and dec.shadow is False
    assert dec.payload == {"drop_ids": ["fact:1"]}
    g = dec.reasons["groups"][0]
    assert g["latest_id"] == "fact:3" and g["stale_ids"] == ["fact:1"]

    out = a.apply_read_decision(dec, _scored(page), top_k=2)
    assert out == ["2. The chairperson of Fatah is Mahmoud Abbas.", SEAM_B]
    assert a.n_facts_suppressed == 1 and a.n_rerank_fallbacks == 0
    assert a.chars_removed == len("1. Thomas Kyd was born in the city of London. ")


def test_live_seam_demote_moves_the_superseder_to_the_page_end(embedder):
    """Retrieved as [B, A] — the superseder's chunk ranks first, which is
    exactly the case chunk rerank cannot improve and the probe showed to be the
    wrong direction anyway. The fact-level move puts serial 3 at the very end."""
    page = [SEAM_B, SEAM_A]
    a = _seam_adapter(LIVE, embedder, "demote_late")
    dec = _seam_retrieve(a, page)
    assert dec.action == "DEMOTE_LATE" and dec.payload == {"move_last_ids": ["fact:3"]}

    out = a.apply_read_decision(dec, _scored(page), top_k=2)
    assert out == ["", SEAM_A + " " + SEAM_B]
    assert len(out) == 2, "the page keeps its length even when a chunk empties"
    assert a.n_facts_demoted == 1
    # The degenerate corner of token neutrality, stated rather than hidden:
    # fact 3 was the ONLY fact in its chunk, so leaving it removed no separator,
    # while arriving added one. Net +1 character. The fact multiset is still
    # exactly preserved — which is the invariant that matters — and the measured
    # delta is REPORTED, never assumed (page_edit's docstring says the same).
    assert a.chars_removed == -1
    # In the arena a chunk carries 230-260 facts, so neither an emptied chunk nor
    # this +1 can occur; the sh_6k run measured DEMOTE_LATE at exactly 0.00%.
    # The page shape is preserved regardless, because the off/shadow neutrality
    # argument rests on the page count never changing.


def test_live_seam_distractor_key_is_never_suppressed(embedder):
    """The Fatah fact shares chunk A but not the key. With the identity screen
    on it must not join the group, and therefore must survive."""
    page = [SEAM_A, SEAM_B]
    a = _seam_adapter(LIVE, embedder, "suppress")
    dec = _seam_retrieve(a, page)
    members = {m for g in dec.reasons["groups"] for m in g["member_ids"]}
    assert members == {"fact:1", "fact:3"}
    assert "fact:2" not in dec.payload["drop_ids"]


def test_shadow_seam_decides_but_never_edits(embedder):
    page = [SEAM_A, SEAM_B]
    for mech in FACT_MECHANISMS:
        a = _seam_adapter(SHADOW, embedder, mech)
        dec = _seam_retrieve(a, page)
        assert dec.action in ("SUPPRESS", "DEMOTE_LATE")
        assert dec.shadow is True, "only live may arm a decision"
        assert a.apply_read_decision(dec, _scored(page), top_k=2) == page
        assert a.n_fact_edits_applied == 0


def test_off_seam_makes_no_decision_at_all(embedder):
    page = [SEAM_A, SEAM_B]
    for mech in FACT_MECHANISMS:
        a = _seam_adapter(OFF, embedder, mech)
        dec = _seam_retrieve(a, page)
        assert dec.action == "PASS" and dec.shadow is True
        assert a.apply_read_decision(dec, _scored(page), top_k=2) == page


def test_demote_token_neutrality_holds_except_for_separator_bookkeeping():
    """State the exact claim, and pin both halves of it.

    A move is a permutation of the fact multiset in every case. Character count
    is additionally unchanged whenever the fact leaves a chunk that still has
    another fact, because it takes one separator with it and acquires one on
    arrival. The one exception is a fact that was ALONE in its chunk: nothing
    was there to separate it, so the arrival separator is a net addition."""
    # within one chunk: exactly neutral
    _, rep = page_edit([JOINED], move_last_ids=["fact:0"])
    assert rep["delta_chars"] == 0
    # across chunks, source keeps a fact: exactly neutral
    _, rep = page_edit([CH_A, CH_B], move_last_ids=["fact:0"])
    assert rep["delta_chars"] == 0
    # across chunks, source empties: +len(separator), reported not hidden
    out, rep = page_edit(["0. Alpha is one.", CH_B], move_last_ids=["fact:0"])
    assert out == ["", "2. Gamma is three. 3. Delta is four. 0. Alpha is one."]
    assert rep["delta_chars"] == len(rep["separator"]) == 1
    # the invariant that never bends, in all three cases
    for page, ids in (([JOINED], ["fact:0"]), ([CH_A, CH_B], ["fact:0"]),
                      (["0. Alpha is one.", CH_B], ["fact:0"])):
        edited, _ = page_edit(page, move_last_ids=ids)
        before = sorted(f for t in page for f in explode_facts(t))
        after = sorted(f for t in edited for f in explode_facts(t))
        assert after == before, "a move must preserve the fact multiset exactly"


def test_a_moved_fact_is_never_appended_past_a_dangling_serial():
    """Regression, found by the retrieval-harness integrity check on the real
    sh_6k page (T13). The benchmark's chunker ends chunk 0 with
    "...Leeds. 307." — fact 307's serial, whose text opens chunk 1. Appending a
    moved fact after those bytes made the inline parser read the whole appended
    sentence as the TEXT OF FACT 307, so the moved fact silently lost its own
    serial: the supersession key the entire mechanism turns on. The moved fact
    must land before the dangling serial, which stays last."""
    page = ["0. Alpha is one. 1. Beta is two. 2."]
    out, rep = page_edit(page, move_last_ids=["fact:0"])
    assert out == ["1. Beta is two. 0. Alpha is one. 2."]
    assert explode_facts(out[0]) == [(1, "Beta is two."), (0, "Alpha is one.")]
    assert rep["delta_chars"] == 0
    # and the same for trailing whitespace, which is equally not content
    out, _ = page_edit(["0. A is X.\n1. B is Y.\n"], move_last_ids=["fact:0"])
    assert out == ["1. B is Y.\n0. A is X.\n"]


def test_content_tail_splits_whitespace_and_dangling_serial():
    from hnav.adapters.mab_adapter import _content_tail
    assert _content_tail("0. A is X. 1.") == ("0. A is X.", " 1.")
    assert _content_tail("0. A is X.\n") == ("0. A is X.", "\n")
    assert _content_tail("0. A is X. 1. B is Y.") == ("0. A is X. 1. B is Y.", "")
    # a genuine numeric ending is NOT a dangling serial: no period before it
    assert _content_tail("0. The population is 42.") == ("0. The population is 42.", "")
