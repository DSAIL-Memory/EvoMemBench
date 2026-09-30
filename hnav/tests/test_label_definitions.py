"""Label definitions: deterministic, declared, and checked on real data.  [T6]

Acceptance criterion from the plan §13: *every label has a deterministic
function, a declared online/offline status, and ≥50 hand-checkable examples*.

"Hand-checkable" is implemented as: each example context is built from the
committed ``sh_6k`` subset by a replay written **independently of
``labels.py``**, and each label's verdict is compared against an oracle
reimplemented inline here from the protocol's prose. If the two ever disagree,
one of them is wrong and the test says which example.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from hnav.adapters.mab_adapter import explode_facts, fact_key
from hnav.labeling import labels as L

MIN_EXAMPLES = 50
TAU_HIGH, TAU_LOW, TAU_SIM = 0.80, 0.30, 0.85


# ── example construction (independent of labels.py) ──────────────────────────
@pytest.fixture(scope="module")
def write_contexts(sh_6k) -> list[dict]:
    """One context per fact, replayed in serial order with write-time visibility."""
    from hnav.labeling.counterfactual import map_questions_to_keys

    facts = explode_facts(sh_6k["context"])
    parsed = [(s, t, *fact_key(t)) for s, t in facts]
    keys_with_question = {tuple(q["key"]) for q in map_questions_to_keys(sh_6k)
                          if q["key"]}

    latest_after: dict = defaultdict(list)
    for s, _, key, obj in parsed:
        if key:
            latest_after[key].append((s, obj))

    seen_text: set[str] = set()
    prior_by_key: dict = {}
    out = []
    for i, (serial, text, key, obj) in enumerate(parsed):
        prior = prior_by_key.get(key)
        later = [(s, o) for s, o in latest_after.get(key, []) if s > serial and o != obj]
        # Deterministic stand-ins for the geometry: identical objects are
        # "similar", different objects are not. Keeps the label logic under test
        # rather than the embedder.
        same_obj = bool(prior and prior[1] == obj)
        out.append({
            "_serial": serial, "_index": i, "_key": key, "_text": text,
            "has_prior_same_key": prior is not None,
            "prior_object": prior[1] if prior else None,
            "candidate_object": obj,
            "whole_blob_sim": 0.95 if prior else None,
            "diff_sim": 0.95 if same_obj else (0.10 if prior else None),
            "tau_high": TAU_HIGH, "tau_low": TAU_LOW,
            "sim_max": 0.95 if prior else 0.20,
            "tau": TAU_SIM,
            "verbatim_new_values": [] if same_obj else ([obj] if obj else []),
            "is_exact_dup": text in seen_text,
            "later_fact_exists": bool(later),
            "question_maps_to_key": key in keys_with_question,
            # every fact in the source list becomes a candidate on this arena
            "in_source_not_candidate": False,
        })
        seen_text.add(text)
        if key:
            prior_by_key[key] = (serial, obj)
    return out


@pytest.fixture(scope="module")
def read_contexts(sh_6k) -> list[dict]:
    """Read contexts spanning every combination the labels distinguish."""
    out = []
    for i in range(120):
        stale = [{"old_id": f"fact:{i}", "new_id": f"fact:{i + 1}",
                  "superseder_retrieved": (i % 3 == 0)}] if i % 2 == 0 else []
        out.append({
            "_index": i,
            "stale_hits": stale,
            "n_conflicted_keys_in_topk": i % 4,
            "nmargin": (i % 10) / 10.0, "nmargin_threshold": 0.5,
            "H_z": (i % 7) / 2.0, "H_z_threshold": 2.0,
            "n_subject_only_matches": 1 if i % 5 == 0 else 0,
            "key_matched_in_topk": i % 6 != 0,
        })
    return out


# ── registry contract ────────────────────────────────────────────────────────
def test_every_label_declares_side_status_and_definition():
    assert len(L.ALL_LABELS) >= 16
    for name, d in L.ALL_LABELS.items():
        assert d.name == name
        assert d.side in ("write", "read")
        assert isinstance(d.online, bool)
        assert isinstance(d.retained, bool)
        assert len(d.definition) > 20, f"{name} has no usable definition"
        if d.retained and name != "READ_CLEAR":
            assert callable(d.fn), f"{name} is retained but has no function"


def test_dropped_labels_keep_their_reason():
    d = L.WRITE_LABELS["WRITE_DESTRUCTIVE_OVERWRITE"]
    assert d.retained is False
    assert "overwrites" in d.dropped_reason
    assert d not in L.retained("write")
    assert d({"anything": True}) is False


def test_offline_labels_are_excluded_from_the_online_view(write_contexts):
    offline = {d.name for d in L.retained("write") if not d.online}
    assert offline == {"WRITE_STALE_SUPERSEDE", "WRITE_UNNECESSARY",
                       "WRITE_MISSED_UPDATE"}
    for ctx in write_contexts[:200]:
        online_only = set(L.apply_write_labels(ctx, online_only=True))
        assert not (online_only & offline)


# ── determinism and example coverage ─────────────────────────────────────────
def test_labels_are_deterministic(write_contexts, read_contexts):
    for ctx in write_contexts:
        assert L.apply_write_labels(ctx) == L.apply_write_labels(ctx)
    for ctx in read_contexts:
        assert L.apply_read_labels(ctx) == L.apply_read_labels(ctx)


# Labels whose base rate on the primary arena is structurally zero. Each is a
# FINDING — "EvoMemBench does not generate this class" — and is the NO_GO
# (benchmark) verdict of protocol §4, not a coverage gap in the labeler.
STRUCTURALLY_ABSENT = {
    "WRITE_DUPLICATE": "every fact in Conflict_Resolution is a unique string",
    "WRITE_REDUNDANT": "a repeated (relation, subject) always carries a NEW object "
                       "here, so there is never a same-value restatement to call "
                       "redundant; the class lives on CrossEp-Know",
    "WRITE_MISSED_UPDATE": "the memorize path enumerates every source fact, so no "
                           "fact can fail to become a candidate",
}


def test_enough_examples_and_labels_discriminate(write_contexts, read_contexts):
    """A label that always fires is not discriminating; one that never fires on
    real data must be a declared structural absence, and must still fire on a
    constructed context — otherwise the function is simply broken."""
    assert len(write_contexts) >= MIN_EXAMPLES
    assert len(read_contexts) >= MIN_EXAMPLES

    for d in L.retained("write"):
        fired = sum(1 for c in write_contexts if d(c))
        assert fired < len(write_contexts), f"{d.name} fired on every example"
        if fired == 0:
            assert d.name in STRUCTURALLY_ABSENT, (
                f"{d.name} never fires on real data and is not declared absent")
            assert d(_positive_write_context(d.name)), \
                f"{d.name} does not fire even on a constructed positive"

    for d in L.retained("read"):
        if d.name == "READ_CLEAR":
            continue
        fired = sum(1 for c in read_contexts if d(c))
        assert 0 < fired < len(read_contexts), f"{d.name} fired on {fired} examples"


def _positive_write_context(name: str) -> dict:
    base = {"has_prior_same_key": True, "prior_object": "X", "candidate_object": "X",
            "whole_blob_sim": 0.99, "diff_sim": 0.99,
            "tau_high": TAU_HIGH, "tau_low": TAU_LOW,
            "sim_max": 0.99, "tau": TAU_SIM, "verbatim_new_values": [],
            "is_exact_dup": False, "later_fact_exists": False,
            "question_maps_to_key": True, "in_source_not_candidate": False}
    if name == "WRITE_DUPLICATE":
        base["is_exact_dup"] = True
    elif name == "WRITE_MISSED_UPDATE":
        base["in_source_not_candidate"] = True
    return base


def test_structural_absences_are_reported_not_hidden(write_contexts):
    """The absence itself is the measurement, so assert it rather than skip it."""
    for name, reason in STRUCTURALLY_ABSENT.items():
        fired = sum(1 for c in write_contexts if L.WRITE_LABELS[name](c))
        assert fired == 0, (
            f"{name} now fires {fired} times on sh_6k — the declared reason "
            f"({reason}) no longer holds and the Stage-0 report must be updated")


# ── independent oracles ──────────────────────────────────────────────────────
def test_write_labels_match_an_independent_oracle(write_contexts):
    for ctx in write_contexts:
        conflict = (ctx["has_prior_same_key"]
                    and ctx["prior_object"] != ctx["candidate_object"])
        critical = (conflict and ctx["whole_blob_sim"] is not None
                    and ctx["whole_blob_sim"] >= TAU_HIGH
                    and ctx["diff_sim"] <= TAU_LOW)
        redundant = ctx["sim_max"] >= ctx["tau"] and not ctx["verbatim_new_values"]

        got = set(L.apply_write_labels(ctx))
        where = f"fact:{ctx['_serial']}"
        assert ("WRITE_CONFLICT" in got) == conflict, where
        assert ("WRITE_CRITICAL_DELTA" in got) == critical, where
        assert ("WRITE_REDUNDANT" in got) == redundant, where
        assert ("WRITE_DUPLICATE" in got) == ctx["is_exact_dup"], where
        assert ("WRITE_STALE_SUPERSEDE" in got) == ctx["later_fact_exists"], where
        assert ("WRITE_UNNECESSARY" in got) == (not ctx["question_maps_to_key"]), where


def test_read_labels_match_an_independent_oracle(read_contexts):
    for ctx in read_contexts:
        stale = len(ctx["stale_hits"]) > 0
        below_k = any(not h["superseder_retrieved"] for h in ctx["stale_hits"])
        got = set(L.apply_read_labels(ctx))
        where = f"read#{ctx['_index']}"
        assert ("READ_STALE" in got) == stale, where
        assert ("READ_RELEVANT_BELOW_K" in got) == below_k, where
        assert ("READ_CONFLICT" in got) == (ctx["n_conflicted_keys_in_topk"] > 0), where
        assert ("READ_AMBIGUOUS" in got) == (ctx["nmargin"] < 0.5), where
        assert ("READ_HIGH_INTERFERENCE" in got) == (ctx["H_z"] > 2.0), where
        assert ("READ_MISSING" in got) == (not ctx["key_matched_in_topk"]), where


def test_read_clear_is_exactly_the_complement(read_contexts):
    for ctx in read_contexts:
        got = L.apply_read_labels(ctx)
        if got == ["READ_CLEAR"]:
            assert not any(d(ctx) for d in L.retained("read") if d.name != "READ_CLEAR")
        else:
            assert "READ_CLEAR" not in got

    clean = {"stale_hits": [], "n_conflicted_keys_in_topk": 0,
             "nmargin": 0.9, "nmargin_threshold": 0.5,
             "H_z": 0.1, "H_z_threshold": 2.0,
             "n_subject_only_matches": 0, "key_matched_in_topk": True}
    assert L.apply_read_labels(clean) == ["READ_CLEAR"]


def test_labels_do_not_fire_on_absent_evidence():
    """An empty context must produce no write label and READ_CLEAR.

    On this benchmark a spurious write label means suppressing the only fact
    carrying the answer, so "unknown" must never read as "true".
    """
    assert L.apply_write_labels({}) == []
    assert L.apply_read_labels({}) == ["READ_CLEAR"]
