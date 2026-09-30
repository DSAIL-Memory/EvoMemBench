"""Context surgery and the arm plumbing of the oracle probe.  [T12]

The probe's whole evidential value rests on the arms differing from ``native``
by *exactly* the intended edit and by nothing else, so the surgery is checked
against hand-built fixtures where the intended edit is written out:

* ``render_context(*split_context(ctx))`` reproduces the real dataset context
  byte for byte — the ``native`` arm is the untouched context, not a
  re-rendering of it;
* ``oracle_suppress`` removes exactly the stale sentences of the queried key and
  leaves every other line's serial, text and order untouched;
* the placement arms are permutations — the multiset of ``(serial, text)`` is
  invariant — and they move exactly the fact they claim to;
* **serials are never renumbered**, in any arm. The benchmark's prompt states
  its rule in terms of serial order ("the newer fact has larger serial
  number"), so renumbering after a deletion would quietly redefine the task.

The statistics are checked against closed forms (an exact two-sided binomial has
one), and the whole pipeline is run end to end against a deterministic stub whose
per-arm score is known by construction. No test here opens a socket.
"""
from __future__ import annotations

import json

import pytest

from hnav import config as _config
from hnav.labeling.question_strata import classify_questions, key_members
from hnav.stage1 import stale_suppression_probe as P

# ── a fixture whose surgery is obvious by inspection ─────────────────────────
# key A = (capital, Atlantis): serial 1 stale, serial 4 gold+latest
# key B = (capital, Numenor): single value  -> unique stratum
ITEM = {
    "context": (
        "Here is a list of facts:\n"
        "0. The capital of Numenor is Armenelos.\n"
        "1. The capital of Atlantis is Poseidonis.\n"
        "2. The chairperson of Fatah is Ada Lovelace.\n"
        "3. The capital of Atlantis is Byzantium.\n"
        "4. The capital of Atlantis is Kallipolis.\n"
        # a trailing unrelated fact, so that moving the queried key's newest
        # fact to the end is a REAL move and not a coincidental no-op
        "5. Thomas Kyd was born in the city of London.\n"
    ),
    "questions": ["What is the capital of Atlantis?",
                  "What is the capital of Numenor?"],
    "answers": [["Kallipolis"], ["Armenelos"]],
    "metadata": {"qa_pair_ids": ["factconsolidation_sh_6k_no0",
                                 "factconsolidation_sh_6k_no1"]},
}
FACTS = [
    (0, "The capital of Numenor is Armenelos."),
    (1, "The capital of Atlantis is Poseidonis."),
    (2, "The chairperson of Fatah is Ada Lovelace."),
    (3, "The capital of Atlantis is Byzantium."),
    (4, "The capital of Atlantis is Kallipolis."),
    (5, "Thomas Kyd was born in the city of London."),
]


@pytest.fixture(scope="module")
def plans():
    conflicted, unique = classify_questions(ITEM)
    members = key_members(ITEM)
    return {
        "conflicted": P.surgery_plan(conflicted, members[tuple(conflicted["key"])]),
        "unique": P.surgery_plan(unique, members[tuple(unique["key"])]),
    }


# ── split / render ───────────────────────────────────────────────────────────
def test_split_context_is_exact():
    preamble, facts = P.split_context(ITEM["context"])
    assert preamble == "Here is a list of facts:"
    assert facts == FACTS


def test_render_reproduces_the_real_context_byte_for_byte(sh_6k):
    """If this drifts, the native arm stops being the benchmark's own context."""
    preamble, facts = P.split_context(sh_6k["context"])
    assert P.render_context(preamble, facts) == sh_6k["context"]
    assert len(facts) == 455


# ── surgery ──────────────────────────────────────────────────────────────────
def test_surgery_plan_names_the_right_serials(plans):
    conf = plans["conflicted"]
    assert conf["gold_serials"] == [4]
    assert conf["stale_serials"] == [1, 3]      # both non-gold values of the key
    assert conf["latest"] == 4
    assert conf["stale_anchor"] == 3            # most recent stale, != latest
    assert conf["gold_is_latest"] is True

    uniq = plans["unique"]
    assert uniq["stale_serials"] == [] and uniq["stale_anchor"] is None
    assert uniq["latest"] == 0


def test_suppress_removes_exactly_the_stale_sentences(plans):
    out = P.apply_arm(FACTS, "oracle_suppress", plans["conflicted"])
    assert out == [(0, "The capital of Numenor is Armenelos."),
                   (2, "The chairperson of Fatah is Ada Lovelace."),
                   (4, "The capital of Atlantis is Kallipolis."),
                   (5, "Thomas Kyd was born in the city of London.")]
    # the surviving facts keep their ORIGINAL serials: 0, 2, 4, 5 with gaps
    assert [s for s, _ in out] == [0, 2, 4, 5]


def test_suppress_is_a_no_op_on_a_unique_key(plans):
    assert P.apply_arm(FACTS, "oracle_suppress", plans["unique"]) == FACTS


def test_recency_moves_only_the_latest_fact(plans):
    out = P.apply_arm(FACTS, "oracle_recency", plans["conflicted"])
    assert out[-1] == (4, "The capital of Atlantis is Kallipolis.")
    assert out[:-1] == [f for f in FACTS if f[0] != 4]      # order otherwise kept
    assert [s for s, _ in out] == [0, 1, 2, 3, 5, 4]
    assert sorted(out) == sorted(FACTS), "a permutation, nothing added or dropped"


def test_anti_puts_the_latest_first_and_the_stale_anchor_last(plans):
    out = P.apply_arm(FACTS, "anti", plans["conflicted"])
    assert out[0] == (4, "The capital of Atlantis is Kallipolis.")
    assert out[-1] == (3, "The capital of Atlantis is Byzantium.")
    assert sorted(out) == sorted(FACTS)
    # the untouched facts keep their relative order between the two moved ones
    assert [s for s, _ in out] == [4, 0, 1, 2, 5, 3]


def test_anti_and_recency_disagree_about_the_last_fact(plans):
    """The direction contrast the arm exists for."""
    rec = P.apply_arm(FACTS, "oracle_recency", plans["conflicted"])
    anti = P.apply_arm(FACTS, "anti", plans["conflicted"])
    assert rec[-1][0] == plans["conflicted"]["latest"]
    assert anti[-1][0] == plans["conflicted"]["stale_anchor"]
    assert rec[-1] != anti[-1]


def test_native_arms_change_nothing(plans):
    for arm in ("native", "native_repeat"):
        assert P.apply_arm(FACTS, arm, plans["conflicted"]) == FACTS


def test_unknown_arm_is_an_error(plans):
    with pytest.raises(ValueError):
        P.apply_arm(FACTS, "wishful", plans["conflicted"])


def test_no_arm_ever_renumbers_a_fact(plans):
    """The prompt's rule is 'larger serial = newer'. If an arm renumbered, the
    task would change under the model's feet and the comparison would be void."""
    original = dict(FACTS)
    for arm in P.ARMS:
        for plan in plans.values():
            for serial, text in P.apply_arm(FACTS, arm, plan):
                assert original[serial] == text, (arm, serial)


def test_move_helpers_are_stable_permutations():
    assert P.move_to_end(FACTS, 0)[-1] == FACTS[0]
    assert P.move_to_front(FACTS, 4)[0] == FACTS[4]
    assert P.move_to_end(FACTS, 999) == FACTS, "an absent serial moves nothing"
    for helper in (P.move_to_end, P.move_to_front):
        assert sorted(helper(FACTS, 2)) == sorted(FACTS)


# ── prompts ──────────────────────────────────────────────────────────────────
def test_prompt_has_the_benchmark_shape_and_carries_the_whole_context():
    prompt = P.build_prompt(ITEM["context"], "What is the capital of Atlantis?")
    assert prompt.startswith("Memory 1:\n")
    assert ITEM["context"] in prompt
    assert "Pretend you are a knowledge management system" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "Now Answer the Question: Based on the provided Knowledge Pool, " \
           "What is the capital of Atlantis?" in prompt


def test_plan_subset_marks_which_arms_are_byte_identical_to_native():
    plan = P.plan_subset(ITEM, "sh_6k")
    conflicted, unique = plan["questions"]
    assert conflicted["stratum"] == "conflicted" and unique["stratum"] == "unique"
    # a conflicted key has stale facts to remove -> suppress really differs
    assert conflicted["identical_to_native"]["oracle_suppress"] is False
    # a unique key has none -> suppress is structurally a no-op, and is counted
    assert unique["identical_to_native"]["oracle_suppress"] is True
    # ...as is `anti` on a key whose only fact is already the first line
    assert unique["identical_to_native"]["anti"] is True
    assert conflicted["arms"]["oracle_suppress"]["n_facts"] == 4
    assert conflicted["arms"]["oracle_recency"]["n_facts"] == 6


def test_plan_subset_respects_max_questions():
    assert len(P.plan_subset(ITEM, "sh_6k", max_questions=1)["questions"]) == 1


# ── statistics against closed forms ──────────────────────────────────────────
@pytest.mark.parametrize("b,c,expected", [
    (0, 0, 1.0),
    (0, 1, 1.0),          # 2 * 1/2
    (0, 2, 0.5),          # 2 * 1/4
    (0, 5, 2 / 32),
    (1, 3, 2 * (1 + 4) / 16),
    (3, 3, 1.0),          # symmetric, clipped at 1
])
def test_mcnemar_exact_p_matches_the_binomial(b, c, expected):
    assert P.mcnemar_exact_p(b, c) == pytest.approx(expected)


def test_mcnemar_is_symmetric():
    for b, c in [(0, 4), (2, 7), (1, 1)]:
        assert P.mcnemar_exact_p(b, c) == P.mcnemar_exact_p(c, b)


def test_paired_cells_counts_the_right_corners():
    native = [True, True, False, False]
    arm = [True, False, True, False]
    cells = P.paired_cells(native, arm)
    assert cells == {"n": 4, "b_native_only": 1, "c_arm_only": 1, "net": 0,
                     "p_exact": 1.0}


# ── end to end with a deterministic stub ─────────────────────────────────────
def test_stub_end_to_end_produces_the_counts_the_stub_implies():
    """The stub answers from the FIRST fact whose subject is in the question.

    On this fixture that makes the outcome deducible without running anything:

      native / native_repeat  q0 -> serial 1 (Poseidonis)  WRONG
                              q1 -> serial 0 (Armenelos)   right
      oracle_suppress         q0 -> serial 4 (Kallipolis)  right   (1 and 3 gone)
                              q1 unchanged                 right
      oracle_recency          q0 -> serial 1 still first   WRONG
                              q1 unchanged                 right
      anti                    q0 -> serial 4 moved first   right
                              q1 -> serial 0 still first   right
    """
    plan = P.plan_subset(ITEM, "sh_6k")
    res = P.run_subset(plan, P.primacy_stub())

    got = {arm: (res["arms"][arm]["correct"], res["arms"][arm]["n"])
           for arm in P.ARMS}
    assert got == {"native": (1, 2), "native_repeat": (1, 2),
                   "oracle_suppress": (2, 2), "oracle_recency": (1, 2),
                   "anti": (2, 2)}

    conf = res["by_stratum"]["conflicted"]["arms"]
    assert conf["native"]["accuracy"] == 0.0
    assert conf["oracle_suppress"]["accuracy"] == 1.0
    assert conf["oracle_recency"]["accuracy"] == 0.0
    assert conf["anti"]["accuracy"] == 1.0
    assert res["by_stratum"]["unique"]["arms"]["native"]["accuracy"] == 1.0

    # the A/A floor is a genuine second call, and a deterministic stub makes it 0
    assert res["aa_floor"] == {"n": 2, "b_native_only": 0, "c_arm_only": 0,
                               "net": 0, "p_exact": 1.0}
    assert res["paired_vs_native"]["oracle_suppress"]["c_arm_only"] == 1


def test_the_stub_output_is_the_stale_value_not_a_refusal():
    """The stub reproduces the failure mode the real runs show (``stale_value``),
    so the smoke path exercises the same branch the real data lands in."""
    plan = P.plan_subset(ITEM, "sh_6k")
    res = P.run_subset(plan, P.primacy_stub())
    q0 = res["per_question"][0]
    assert q0["arms"]["native"]["output"] == "Poseidonis"
    assert q0["arms"]["oracle_suppress"]["output"] == "Kallipolis"


def test_budget_predicts_the_call_count_exactly():
    """The pre-flight budget is a promise about how many completions will be
    issued; here it is checked against the number actually issued."""
    plan = P.plan_subset(ITEM, "sh_6k")
    predicted = P.budget([plan])["per_subset"][0]
    res = P.run_subset(plan, P.primacy_stub())
    assert res["n_llm_calls"] == predicted["n_calls"]
    # enumerated by hand:
    #   q0 (conflicted) native, suppress, recency, anti  -> 4 distinct prompts
    #   q1 (unique)     native, recency                  -> 2 (suppress and anti
    #                                                        are no-ops here)
    #   plus one forced A/A repeat per question          -> +2 calls
    assert predicted["n_distinct_prompts"] == 6
    assert predicted["n_aa_repeats"] == 2
    assert predicted["n_calls"] == 8
    assert predicted["n_identical_to_native"] == {
        "oracle_suppress": 1, "oracle_recency": 0, "anti": 1}


# ── split discipline and mode ────────────────────────────────────────────────
def test_main_refuses_a_confirmatory_subset(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["probe", "--subsets", "sh_64k", "--dry-run"])
    assert P.main() == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "Nothing was sent" in out


def test_main_refuses_live_mode(monkeypatch):
    live = _config.HNavConfig(mode=_config.MODE_LIVE)
    monkeypatch.setattr(P._config, "get_config", lambda *a, **k: live)
    monkeypatch.setattr("sys.argv", ["probe", "--dry-run"])
    with pytest.raises(RuntimeError, match="Stage-0 measurement"):
        P.main()


def test_dry_run_sends_nothing(monkeypatch, capsys):
    """--dry-run must be safe to run on a box with no LLM: if it ever reached
    the client, ``make_answer_fn`` would import openai and connect."""
    def explode(*a, **k):
        raise AssertionError("--dry-run must not build an answer function")
    monkeypatch.setattr(P, "make_answer_fn", explode)
    monkeypatch.setattr("sys.argv", ["probe", "--subsets", "sh_6k",
                                     "--max-questions", "3", "--dry-run"])
    assert P.main() == 0
    assert "nothing sent" in capsys.readouterr().out


def test_calibration_split_is_the_one_imported_from_the_campaign():
    assert P.CALIBRATION == ("sh_6k", "sh_32k")
    assert set(P.CALIBRATION) & set(P.CONFIRMATORY) == set()


def test_real_budget_is_within_the_stated_envelope(sh_6k):
    """~4 arms x 100 questions plus the A/A repeats, on one subset."""
    plan = P.plan_subset(sh_6k, "sh_6k")
    row = P.budget([plan])["per_subset"][0]
    assert row["n_questions"] == 100
    assert row["n_calls"] == row["n_distinct_prompts"] + 100
    assert 400 <= row["n_calls"] <= 500
    # every unique-stratum question makes oracle_suppress a structural no-op
    assert row["n_identical_to_native"]["oracle_suppress"] == 26


def test_campaign_reference_reads_the_committed_strata_record(sh_6k):
    """The probe must place its native arm next to the campaign's own number,
    and it must agree with T12 task 1 about how the questions split."""
    if not P.STRATA_JSON.exists():
        pytest.skip("stage0_results/question_strata.json not generated")
    ref = P.campaign_reference("sh_6k")
    assert ref["n_runs"] == 8
    assert ref["unique_accuracy_range"] == [1.0, 1.0]
    assert ref["conflicted_accuracy_range"][1] <= 5 / 74
    assert ref["strata_counts"] == {"unique": 26, "conflicted": 74,
                                    "ambiguous": 0, "unmatched": 0}
    # the probe's own stratification must match it, not merely resemble it.
    # STRATA is imported from question_strata rather than redeclared, so a new
    # stratum there cannot silently go unreported here.
    assert P.STRATA == ("unique", "conflicted", "ambiguous", "unmatched")
    assert P.plan_subset(sh_6k, "sh_6k")["strata_counts"] == ref["strata_counts"]


def test_campaign_reference_is_absent_where_there_is_no_committed_run():
    assert P.campaign_reference("sh_32k") is None          # never run in t4_s2
    assert P.campaign_reference("sh_6k", path=P.REPO / "nope.json") is None


def test_no_module_level_network_or_torch_import():
    """Guard mirrored from test_no_torch_at_import: the openai client must be
    built lazily so the module imports on a box with no endpoint."""
    src = (P.__file__)
    text = json.dumps(open(src, encoding="utf-8").read())
    assert "from openai import OpenAI" in json.loads(text)
    assert not json.loads(text).startswith("import openai")
