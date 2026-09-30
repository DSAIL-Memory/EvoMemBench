"""Counterfactual replay and grading.  [T6]

The LLM is injected, so every case here runs offline with a stub. The stubs are
written to behave like specific failure modes — a model that answers from
whatever it retrieved, a model that anchors on the first fact it sees — so the
counterfactual classes are exercised as intended rather than asserted into
existence.
"""
from __future__ import annotations

import re

import pytest

from hnav.adapters.mab_adapter import explode_facts, fact_key
from hnav.labeling.counterfactual import (CounterfactualLabeler, build_prompt,
                                          map_questions_to_keys, normalize_answer,
                                          substring_exact_match)
from hnav.core.replica import FaissFlatReplica
from hnav.tests.conftest import build_store


# ── grading, pinned to the benchmark's own evaluator ─────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("The Answer.", "answer"),
    ("A CAT", "cat"),
    ("an apple, the pear", "apple pear"),
    ("  spaced   out  ", "spaced out"),
    ("Washington, D.C.", "washington dc"),
    ("", ""),
])
def test_normalize_answer_matches_the_transcription(raw, expected):
    assert normalize_answer(raw) == expected


def test_substring_exact_match_semantics():
    assert substring_exact_match("The capital is Paris.", ["Paris"])
    assert substring_exact_match("paris", ["Paris"])
    assert not substring_exact_match("The capital is Lyon.", ["Paris"])
    # max over ground truths, including the nested-list form the dataset uses
    assert substring_exact_match("It is Rodez.", [["Belgium"], ["Rodez"]])
    assert substring_exact_match("answer: dc", "Washington, D.C.") is False


def test_grading_is_free_and_deterministic():
    for _ in range(3):
        assert substring_exact_match("Paris is the answer", ["Paris"]) is True


# ── question -> key mapping on real data ─────────────────────────────────────
def test_map_questions_to_keys_on_the_real_subset(sh_6k):
    mapped = map_questions_to_keys(sh_6k)
    assert len(mapped) == len(sh_6k["questions"])

    hit = [m for m in mapped if m["key"]]
    assert len(hit) / len(mapped) > 0.6, "most single-hop questions must map to a key"

    conflicted = [m for m in hit if m["conflicted"]]
    assert conflicted, "sh_6k has questions on conflicted keys"

    # the mapped target is the highest serial whose object matches — the
    # "latest wins" rule the committed gold_rule.py established
    for m in conflicted:
        assert m["target_serial"] == max(m["member_serials"]) or \
            m["target_serial"] in m["member_serials"]


def test_map_questions_never_invents_a_key():
    empty = {"context": "", "questions": ["anything?"], "answers": [["x"]]}
    mapped = map_questions_to_keys(empty)
    assert mapped[0]["key"] is None and mapped[0]["target_serial"] is None


# ── a small real store around one conflicted key ─────────────────────────────
@pytest.fixture(scope="module")
def conflict_case(sh_6k, embedder):
    """The first conflicted key in sh_6k, plus 20 unrelated facts as ballast.

    Records keep the benchmark's own ``"<serial>. <fact>"`` surface form, because
    that is what a memorize chunk actually contains and it is what makes the
    stub models below deterministic regardless of ranking order.
    """
    facts = explode_facts(sh_6k["context"])
    groups: dict = {}
    for serial, text in facts:
        key, obj = fact_key(text)
        if key:
            groups.setdefault(key, []).append((serial, text, obj))

    key, members = next((k, sorted(v)) for k, v in groups.items()
                        if len({o for _, _, o in v}) > 1)
    old, new = members[0], members[-1]
    ballast = [(s, t) for s, t in facts if s not in (old[0], new[0])][:20]

    numbered = [f"{old[0]}. {old[1]}", f"{new[0]}. {new[1]}"] + \
               [f"{s}. {t}" for s, t in ballast]
    store = build_store(numbered, embedder,
                        versions=[old[0], new[0]] + [s for s, _ in ballast])
    return {"key": key, "old": old, "new": new, "store": store,
            "old_text": numbered[0], "new_text": numbered[1],
            "question": {"index": 0,
                         "question": f"What about {key[1]}?",
                         "truths": [new[2].lower()]}}


NUMBERED = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)


def _facts_in_prompt(prompt: str, key) -> list[tuple[int, str]]:
    """``[(serial, object)]`` for the prompt's facts that match ``key``."""
    body = prompt.split("Facts:\n")[1].split("\n\nQuestion")[0]
    out = []
    for serial, text in NUMBERED.findall(body):
        k, obj = fact_key(text)
        if k == key and obj:
            out.append((int(serial), obj))
    return out


def _latest_wins_stub(key, calls: list):
    """A model that uses the highest-serial matching fact it was shown.

    This is the gold rule the benchmark itself follows, so the model is only
    ever wrong when retrieval failed to give it the superseding fact — which is
    exactly the bottleneck the counterfactual is meant to isolate.
    """
    def answer(prompt: str) -> str:
        calls.append(prompt)
        hits = _facts_in_prompt(prompt, key)
        return max(hits)[1] if hits else "no idea"
    return answer


def test_label_write_marks_the_superseding_fact_must_write(conflict_case, embedder):
    """Dropping the newest fact for a key breaks its question -> must_write."""
    case = conflict_case
    calls: list = []
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=25),
                                    _latest_wins_stub(case["key"], calls), top_k=25)

    out = labeler.label_write(case["store"], "rec:1", [case["question"]],
                              is_newest_for_key=True)
    assert out.n_affected == 1
    assert out.correct_with == [True]
    assert out.correct_without == [False]
    assert out.counterfactual_class == "must_write"
    assert labeler.n_calls == 2, "one call per arm, no more"


def test_label_write_marks_a_stale_fact_must_suppress(conflict_case, embedder):
    """A model anchored on the first fact it sees is fixed by removing the stale
    one — the must_suppress case."""
    case = conflict_case

    def anchored(prompt: str) -> str:
        hits = _facts_in_prompt(prompt, case["key"])
        return min(hits)[1] if hits else "no idea"

    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=25),
                                    anchored, top_k=25)
    out = labeler.label_write(case["store"], "rec:0", [case["question"]],
                              is_newest_for_key=False)
    # with the stale fact present the anchored model answers with the stale
    # object and is wrong; without it, it answers with the superseding object
    assert out.correct_with == [False] and out.correct_without == [True]
    assert out.counterfactual_class == "must_suppress"


def test_label_write_with_no_affected_question(conflict_case, embedder):
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=5),
                                    lambda p: "unused")
    superseded = labeler.label_write(conflict_case["store"], "rec:0", [],
                                     is_newest_for_key=False)
    assert superseded.counterfactual_class == "inert_superseded"
    assert labeler.n_calls == 0, "an empty affected set must cost nothing"

    newest = labeler.label_write(conflict_case["store"], "rec:1", [],
                                 is_newest_for_key=True)
    assert newest.counterfactual_class == "uncertain"


def test_may_suppress_when_the_outcome_is_unchanged(conflict_case, embedder):
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=25),
                                    lambda p: conflict_case["new"][2], top_k=25)
    out = labeler.label_write(conflict_case["store"], "rec:0",
                              [conflict_case["question"]], is_newest_for_key=False)
    assert out.correct_with == out.correct_without == [True]
    assert out.counterfactual_class == "may_suppress"


def test_prompt_cache_prevents_duplicate_calls(conflict_case, embedder):
    calls: list = []
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=25),
                                    _latest_wins_stub(conflict_case["key"], calls),
                                    top_k=25)
    q = conflict_case["question"]
    for _ in range(4):
        labeler.answer_and_grade(q["question"],
                                 labeler.retrieve_texts(conflict_case["store"],
                                                        q["question"]),
                                 q["truths"])
    assert labeler.n_calls == 1 and len(calls) == 1


# ── read side ────────────────────────────────────────────────────────────────
def test_label_read_detects_a_repair(conflict_case, embedder):
    case = conflict_case
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=5),
                                    _latest_wins_stub(case["key"], []), top_k=5)

    native = [case["old_text"]]                   # only the stale fact retrieved
    injected = [case["new_text"]]                 # the superseder, injected
    out = labeler.label_read(case["question"], native, injected)

    assert out.correct_native is False
    assert out.correct_repaired is True
    assert out.repair_helped is True and out.repair_harmed is False
    assert out.n_injected == 1


def test_label_read_does_not_double_inject(conflict_case, embedder):
    case = conflict_case
    labeler = CounterfactualLabeler(FaissFlatReplica(embedder, top_k=5),
                                    _latest_wins_stub(case["key"], []), top_k=5)
    out = labeler.label_read(case["question"], [case["new_text"]], [case["new_text"]])
    assert out.n_injected == 0, "a fact already retrieved must not be added twice"


def test_build_prompt_is_stable_and_contains_only_what_it_was_given():
    p = build_prompt("Q?", ["fact one.", "fact two."])
    assert p == build_prompt("Q?", ["fact one.", "fact two."])
    assert "fact one." in p and "Q?" in p
    # the prompt carries the question and the retrieved facts, and nothing else
    assert "fact three" not in p
