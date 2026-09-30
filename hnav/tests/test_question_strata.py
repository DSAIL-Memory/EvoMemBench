"""Question strata and the error taxonomy.  [T12]

Three levels of check, matching the standard the rest of the suite is held to:

1. **Synthetic, known by construction.** Hand-built contexts where the answer
   is determined by the fixture, not by the classifier: a key with two distinct
   values is conflicted, a key with one is unique, a question whose subject
   matches nothing is unmatched. Includes the case that separates the right
   rule from the plausible-but-wrong one — the same value written twice is a
   *duplicate*, not a conflict.
2. **Independent oracle on the real artifacts.** ``_oracle_counts`` below goes
   from the raw dataset JSON and the raw run JSON to per-run, per-stratum
   accuracy and error counts through a separate implementation — subsumption
   decided by contiguous token sublists rather than by regex containment — and
   the two must agree exactly on all 400 single-hop questions and all 8
   committed runs.
3. **Negative controls.** A deliberately mislabeled fixture must be *rejected*
   by the same assertion helper the positive tests use, and an output carrying a
   different key's value must not be scored as a stale value of this one. A
   classifier that cannot fail proves nothing.

Plus regression pins on the numbers this task exists to establish: sh_6k is
26 unique / 74 conflicted, the unique stratum is 26/26 in every one of the eight
committed runs, and 572 of the 575 conflicted-stratum errors are the stale value
of the correct key.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import pytest

from hnav.labeling.conflict_analysis import parse as parse_fact
from hnav.labeling.counterfactual import (map_questions_to_keys, normalize_answer,
                                          substring_exact_match)
from hnav.labeling.question_strata import (DATA, ERROR_CLASSES, EVIDENCE, STRATA,
                                           accuracy_bounds, build, candidate_keys,
                                           classify_questions,
                                           conflicted_only_accuracy, error_class,
                                           key_members, load_runs, score_run,
                                           stratum_of, subject_matches,
                                           subset_of_run)

# ── synthetic fixtures: the classification is fixed by construction ──────────
SYNTHETIC = {
    "context": (
        "Here is a list of facts:\n"
        "0. The capital of Atlantis is Poseidonis.\n"
        "1. The capital of Numenor is Armenelos.\n"
        "2. The capital of Atlantis is Kallipolis.\n"
        "3. The chairperson of Fatah is Ada Lovelace.\n"
    ),
    # index 0: key has TWO distinct values          -> conflicted
    # index 1: key has ONE value                    -> unique
    # index 2: key has ONE value, different template-> unique
    # index 3: no subject in the context matches    -> unmatched
    "questions": ["What is the capital of Atlantis?",
                  "What is the capital of Numenor?",
                  "Who is the chairperson of Fatah?",
                  "Who won the 1998 world cup?"],
    "answers": [["Kallipolis"], ["Armenelos"], ["Ada Lovelace"], ["France"]],
}
SYNTHETIC_EXPECTED = ["conflicted", "unique", "unique", "unmatched"]

DUPLICATE = {
    # the SAME value twice: two facts, one distinct value. A classifier that
    # counts facts calls this conflicted; the rule counts distinct values.
    "context": ("Here is a list of facts:\n"
                "0. The capital of Utopia is Amaurot.\n"
                "1. The capital of Utopia is Amaurot.\n"),
    "questions": ["What is the capital of Utopia?"],
    "answers": [["Amaurot"]],
}

# The audit case, reduced to its skeleton: a SHORT subject that is a prefix of
# the right one, on a conflicted key, carrying the same expected value. Under a
# bare substring match "Arthur" qualifies and (being earlier) wins, and a unique
# question is scored conflicted. sh_262k q26 in the real data.
ARTHUR_MILLER = {
    "context": ("Here is a list of facts:\n"
                "0. Arthur was created in the country of French Third Republic.\n"
                "1. Arthur was created in the country of United States of America.\n"
                "2. Arthur Miller is a citizen of United States of America.\n"),
    "questions": ["What is the country of citizenship of Arthur Miller?"],
    "answers": [["United States of America"]],
}

# Two DIFFERENT subjects, neither containing the other, both qualifying, and
# they disagree about conflictedness. Nothing can break this tie, so it must be
# named `ambiguous` rather than first-won.
AMBIGUOUS = {
    "context": ("Here is a list of facts:\n"
                "0. The capital of Alpha is Zenith.\n"
                "1. The capital of Alpha is Quux.\n"
                "2. The capital of Beta is Quux.\n"),
    "questions": ["Which is the capital of Alpha and of Beta?"],
    "answers": [["Quux"]],
}


def assert_strata(item: dict, expected: list[str]) -> None:
    """The single assertion helper both the positive tests and the negative
    control go through — so the negative control proves *this* check can fail."""
    got = [r["stratum"] for r in classify_questions(item)]
    assert got == expected, f"strata {got} != expected {expected}"


def test_synthetic_classification_is_known_by_construction():
    assert_strata(SYNTHETIC, SYNTHETIC_EXPECTED)


def test_conflicted_record_carries_the_full_value_inventory():
    rec = classify_questions(SYNTHETIC)[0]
    assert rec["key"][1] == "Atlantis"
    assert rec["n_members"] == 2
    assert rec["other_values"] == ["Poseidonis"]        # gold excluded
    assert rec["target_serial"] == 2 and rec["latest_serial"] == 2
    assert rec["gold_is_latest"] is True


def test_unique_record_has_no_other_values():
    rec = classify_questions(SYNTHETIC)[1]
    assert rec["stratum"] == "unique" and rec["other_values"] == []


def test_unmatched_record_has_no_key_at_all():
    rec = classify_questions(SYNTHETIC)[3]
    assert rec["key"] is None and rec["target_serial"] is None
    assert rec["other_values"] == [] and rec["latest_serial"] is None
    assert stratum_of(rec) == "unmatched"


def test_duplicate_values_are_not_a_conflict():
    """The discrimination the whole analysis rests on: >=2 *facts* is not the
    rule, >=2 *distinct values* is."""
    rec = classify_questions(DUPLICATE)[0]
    assert rec["n_members"] == 2, "the fixture really does contain two facts"
    assert rec["stratum"] == "unique"


# ── the audit fix: word boundaries and no first-win ──────────────────────────
def test_subject_matching_is_on_token_boundaries():
    """Word boundaries stop a subject matching INSIDE a longer word."""
    assert subject_matches("Arthur", "where did Arthur go?")
    assert not subject_matches("Arthur", "the Arthurian legend")
    assert not subject_matches("Miller", "a trip to Millerton")
    assert subject_matches("H. P. Lovecraft", "Who is H. P. Lovecraft?"), \
        "subjects with non-word characters must still anchor"
    assert not subject_matches("", "anything")


def test_word_boundaries_alone_do_not_resolve_the_audit_case():
    """Stated explicitly so the fix is not miscredited: "Arthur" is followed by
    a space in "Arthur Miller", so it is a perfectly good boundary match. What
    disqualifies it is SUBSUMPTION by the longer candidate, below."""
    assert subject_matches("Arthur", "citizenship of Arthur Miller?")
    assert subject_matches("Arthur Miller", "citizenship of Arthur Miller?")


def test_arthur_miller_is_unique_not_conflicted():
    """The one real mis-assignment the supervisor audit found. Regression pin."""
    rec = classify_questions(ARTHUR_MILLER)[0]
    assert rec["stratum"] == "unique"
    assert rec["key"][1] == "Arthur Miller"
    assert rec["candidate_subjects"] == ["Arthur Miller"], \
        "the shorter, subsumed subject must not even be a candidate"
    # and the rule it replaces gets it wrong, which is why the fixture exists
    assert map_questions_to_keys(ARTHUR_MILLER)[0]["conflicted"] is True


def test_a_less_specific_subject_loses_to_a_longer_one():
    groups = key_members(ARTHUR_MILLER)
    cands = candidate_keys(ARTHUR_MILLER["questions"][0],
                           {"united states of america"}, groups)
    assert [k[1] for k in cands] == ["Arthur Miller"]


def test_irreducible_disagreement_is_named_ambiguous_not_first_won():
    """An outcome that can never fire is decoration — this makes it fire."""
    rec = classify_questions(AMBIGUOUS)[0]
    assert rec["stratum"] == "ambiguous"
    assert rec["n_candidates"] == 2
    assert sorted(rec["candidate_subjects"]) == ["Alpha", "Beta"]
    assert rec["key"] is None, "an ambiguous question is not assigned a key"


def test_ambiguous_questions_cannot_contribute_a_stale_value():
    """No settled key means no defensible 'superseded value of the same key';
    the error must fall to off_list, the direction that works AGAINST the
    headline finding."""
    rec = classify_questions(AMBIGUOUS)[0]
    assert rec["other_values"] == []
    assert error_class("Zenith", rec) == "off_list"


def test_key_members_indexes_every_parsable_fact_in_context_order():
    groups = key_members(SYNTHETIC)
    atlantis = [k for k in groups if k[1] == "Atlantis"][0]
    assert [s for s, _, _ in groups[atlantis]] == [0, 2]
    assert len(groups) == 3


# ── NEGATIVE CONTROLS ────────────────────────────────────────────────────────
def test_negative_control_a_mislabeled_fixture_is_rejected():
    """Swap two labels and the same helper the positive test uses must fail.

    Without this, ``assert_strata`` passing would be evidence of nothing.
    """
    mislabeled = ["unique", "conflicted", "unique", "unmatched"]
    assert mislabeled != SYNTHETIC_EXPECTED
    with pytest.raises(AssertionError):
        assert_strata(SYNTHETIC, mislabeled)


def test_negative_control_a_different_keys_value_is_not_a_stale_value():
    """``Armenelos`` is a real value in the context — of a *different* key. The
    taxonomy must call that ``off_list``; calling it ``stale_value`` would
    inflate the headline finding."""
    rec = classify_questions(SYNTHETIC)[0]
    assert error_class("Armenelos", rec) == "off_list"


def test_negative_control_a_corrupted_context_changes_the_verdict():
    """Rewrite the conflicting fact to carry the gold value and the question
    stops being conflicted — the classifier tracks the data, not the fixture."""
    flattened = dict(SYNTHETIC)
    flattened["context"] = SYNTHETIC["context"].replace(
        "0. The capital of Atlantis is Poseidonis.",
        "0. The capital of Atlantis is Kallipolis.")
    assert classify_questions(flattened)[0]["stratum"] == "unique"


# ── error taxonomy, oracle written out by hand ───────────────────────────────
def test_error_class_cases():
    conflicted = classify_questions(SYNTHETIC)[0]
    unique = classify_questions(SYNTHETIC)[1]

    assert error_class("Poseidonis", conflicted) == "stale_value"
    # normalized substring containment, exactly as the evaluator compares golds
    assert error_class("It is Poseidonis, I think.", conflicted) == "stale_value"
    assert error_class("the Poseidonis", conflicted) == "stale_value"
    assert error_class("Byzantium", conflicted) == "off_list"
    assert error_class("", conflicted) == "empty"
    assert error_class("  ...  ", conflicted) == "empty", \
        "punctuation-only survives nothing after normalization"
    # a unique key has no non-gold value, so every miss is off_list
    assert error_class("Byzantium", unique) == "off_list"


def _fake_run(outputs: list[str], sub_dataset: str = "factconsolidation_syn") -> dict:
    """A run artifact in the benchmark's own shape, with chosen outputs."""
    truths = [a for a in SYNTHETIC["answers"]]
    return {
        "dataset_config": {"sub_dataset": sub_dataset},
        "averaged_metrics": {"substring_exact_match": None},
        "data": [{"query_id": i, "qa_pair_id": f"{sub_dataset}_no{i}",
                  "parsed_output": o, "answer": truths[i],
                  "substring_exact_match": substring_exact_match(o, truths[i])}
                 for i, o in enumerate(outputs)],
    }


def test_score_run_against_a_hand_written_expectation():
    records = classify_questions(SYNTHETIC)
    scored = score_run(_fake_run([
        "Poseidonis",     # q0 conflicted -> wrong, stale value of the same key
        "Armenelos",      # q1 unique     -> correct
        "Grace Hopper",   # q2 unique     -> wrong, off_list
        "",               # q3 unmatched  -> wrong, empty
    ]), records)

    # written out by hand from the four lines above, not computed
    assert scored["strata"]["conflicted"] == {
        "n": 1, "correct": 0, "accuracy": 0.0,
        "errors": {"stale_value": 1, "off_list": 0, "empty": 0}}
    assert scored["strata"]["unique"] == {
        "n": 2, "correct": 1, "accuracy": 0.5,
        "errors": {"stale_value": 0, "off_list": 1, "empty": 0}}
    assert scored["strata"]["unmatched"] == {
        "n": 1, "correct": 0, "accuracy": 0.0,
        "errors": {"stale_value": 0, "off_list": 0, "empty": 1}}
    assert scored["accuracy_overall"] == 0.25
    assert scored["grade_check"]["n_disagreements"] == 0
    assert [s["output"] for s in scored["off_list_outputs"]] == ["Grace Hopper"]


def test_score_run_reports_disagreement_with_a_wrong_recorded_grade():
    """The cross-check against the benchmark's own field must be able to fire."""
    payload = _fake_run(["Poseidonis", "Armenelos", "Grace Hopper", ""])
    payload["data"][0]["substring_exact_match"] = True      # a lie
    scored = score_run(payload, classify_questions(SYNTHETIC))
    assert scored["grade_check"]["n_disagreements"] == 1


# ── closed form: bounds, and where a point estimate is allowed at all ────────
def test_accuracy_bounds_are_the_stated_counting_argument():
    # sh_262k after the audit fix: 20 correct of 100, 22 unique, 76 conflicted,
    # 2 unmatched. lo = max(0, 20-22-2)/76 = 0; hi = min(20,76)/76.
    b = accuracy_bounds(20, 22, 76, 2)
    assert b["lo"] == 0.0
    assert b["hi"] == pytest.approx(20 / 76)
    assert b["unique_accuracy_upper_bound"] == pytest.approx(20 / 22)
    assert b["assumption_refuted"] is True, \
        "20 correct cannot fill 22 unique questions - the premise is refuted"

    # sh_6k: 33 correct, 26 unique, 74 conflicted. The premise is NOT refuted.
    b6 = accuracy_bounds(33, 26, 74, 0)
    assert b6["lo"] == pytest.approx(7 / 74)
    assert b6["hi"] == pytest.approx(33 / 74)
    assert b6["unique_accuracy_upper_bound"] == 1.0
    assert b6["assumption_refuted"] is False
    assert accuracy_bounds(33, 26, 0, 0) == {}


def test_a_point_estimate_appears_only_where_the_premise_was_measured():
    counts = {"unique": 26, "conflicted": 74, "ambiguous": 0, "unmatched": 0}
    measured = {"accuracy": 1.0, "n_runs": 8, "source": "t4_s2"}

    got = conflicted_only_accuracy(0.33, counts, 100, measured)
    assert got["kind"] == "estimate"
    assert got["estimate"] == pytest.approx(7 / 74)
    # the estimate is exactly the bound's lower end: assuming the unique stratum
    # is perfect is what leaves the conflicted stratum the fewest correct answers
    assert got["estimate"] == pytest.approx(got["bound"]["lo"])

    unmeasured = conflicted_only_accuracy(0.33, counts, 100, None)
    assert unmeasured["kind"] == "bound"
    assert unmeasured["estimate"] is None
    assert unmeasured["bound"]["lo"] == pytest.approx(7 / 74)


def test_a_subset_whose_premise_is_refuted_never_emits_an_estimate():
    """The reviewer-bait case: the old code printed a NEGATIVE probability."""
    counts = {"unique": 22, "conflicted": 76, "ambiguous": 0, "unmatched": 2}
    got = conflicted_only_accuracy(0.20, counts, 100, None)
    assert got["kind"] == "bound" and got["estimate"] is None
    assert got["bound"]["lo"] == 0.0 and got["bound"]["hi"] == pytest.approx(20 / 76)
    assert got["bound"]["assumption_refuted"] is True
    assert all(v is None or v >= 0 for v in
               (got["estimate"], got["bound"]["lo"], got["bound"]["hi"])), \
        "no probability may be negative"
    assert "m3" in got["harness_caveat"], "the harness caveat must travel with it"


def test_conflicted_only_accuracy_without_a_pooled_number():
    counts = {"unique": 26, "conflicted": 74, "ambiguous": 0, "unmatched": 0}
    got = conflicted_only_accuracy(None, counts, 100, None)
    assert got["kind"] == "unavailable" and got["estimate"] is None
    assert got["bound"] is None


# ── the real thing ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def payload():
    if not DATA.exists():
        pytest.skip(f"missing {DATA}")
    return build()


def test_real_subset_strata_counts_are_pinned(payload):
    """Regression pin. These four rows are the finding.

    sh_262k is 22/76, not the 21/77 the first version produced: see
    ``test_arthur_miller_is_unique_not_conflicted``.
    """
    got = {r["subset"]: r["counts"] for r in payload["subsets"]}
    assert got == {
        "sh_6k":   {"unique": 26, "conflicted": 74, "ambiguous": 0, "unmatched": 0},
        "sh_32k":  {"unique": 35, "conflicted": 65, "ambiguous": 0, "unmatched": 0},
        "sh_64k":  {"unique": 34, "conflicted": 66, "ambiguous": 0, "unmatched": 0},
        "sh_262k": {"unique": 22, "conflicted": 76, "ambiguous": 0, "unmatched": 2},
    }


def test_no_question_anywhere_is_left_ambiguous(payload):
    """The escape hatch exists, and on this corpus it is never needed — which is
    what makes the strata assignment a fact about the data rather than a choice.
    Asserted for the calibration split explicitly, since that is where the probe
    will run."""
    for row in payload["subsets"]:
        assert row["counts"]["ambiguous"] == 0, row["subset"]
    cal = [r for r in payload["subsets"] if r["subset"] in ("sh_6k", "sh_32k")]
    assert len(cal) == 2
    assert all(r["counts"]["ambiguous"] == 0 and r["n_multi_candidate"] == 0
               for r in cal), "calibration split must have no ambiguity at all"


def test_the_only_divergence_from_the_committed_mapper_is_arthur_miller():
    """The refined rule must not quietly re-label the corpus: exactly one
    question of 400 moves, and it is the one the audit identified."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    moved = []
    n = 0
    for item in data:
        name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        if not name.startswith("factconsolidation_sh_"):
            continue
        for mine, theirs in zip(classify_questions(item),
                                map_questions_to_keys(item)):
            n += 1
            old = "unmatched" if theirs["key"] is None else \
                ("conflicted" if theirs["conflicted"] else "unique")
            if mine["stratum"] != old:
                moved.append((name, mine["index"], old, mine["stratum"]))
    assert n == 400
    assert moved == [("factconsolidation_sh_262k", 26, "conflicted", "unique")]


def test_the_conflicted_only_column_is_a_bound_except_where_measured(payload):
    by = {r["subset"]: r["conflicted_only_accuracy"] for r in payload["subsets"]}
    assert by["sh_6k"]["kind"] == "estimate"      # unique stratum measured 26/26
    assert by["sh_6k"]["estimate"] == pytest.approx(7 / 74)
    for name in ("sh_32k", "sh_64k", "sh_262k"):
        assert by[name]["kind"] == "bound", name
        assert by[name]["estimate"] is None, name
    b = by["sh_262k"]["bound"]
    assert (b["lo"], round(b["hi"], 2)) == (0.0, 0.26)
    assert b["assumption_refuted"] is True
    assert by["sh_64k"]["bound"]["assumption_refuted"] is False
    # the harness caveat must travel with every one of them
    assert all(v["harness_caveat"].startswith("m3_headroom.py") for v in by.values())


def test_every_run_answers_the_unique_stratum_perfectly(payload):
    assert len(payload["runs"]) == 8, "eight committed sh_6k runs"
    for run in payload["runs"]:
        u = run["strata"]["unique"]
        assert (u["correct"], u["n"]) == (26, 26), f"{run['run']}: {u}"


def test_the_conflicted_stratum_is_near_zero_in_every_run(payload):
    for run in payload["runs"]:
        c = run["strata"]["conflicted"]
        assert c["n"] == 74
        assert 0 <= c["correct"] <= 5, f"{run['run']}: {c}"


def test_the_error_taxonomy_totals_are_pinned(payload):
    assert payload["aggregate"]["errors_total"] == {
        "stale_value": 572, "off_list": 3, "empty": 0}


def test_recomputed_grades_agree_with_the_benchmarks_own_field(payload):
    assert payload["aggregate"]["n_grade_disagreements"] == 0
    total = sum(r["grade_check"]["n_compared"] for r in payload["runs"])
    assert total == 800, "8 runs x 100 questions"
    assert all(r["grade_check"]["n_id_mismatches"] == 0 for r in payload["runs"])


def test_overall_accuracy_matches_the_runs_own_averaged_metric(payload):
    for run in payload["runs"]:
        recorded = run["recorded_averaged_substring_exact_match"]
        assert recorded == pytest.approx(100 * run["accuracy_overall"], abs=1e-6)


# ── independent oracle: raw files -> counts, via gold_rule's matching form ───
def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _is_sub(short: list[str], long: list[str]) -> bool:
    """Is ``short`` a contiguous sublist of ``long``? (Proper containment.)"""
    if not short or len(short) >= len(long):
        return False
    return any(long[i:i + len(short)] == short
               for i in range(len(long) - len(short) + 1))


def _oracle_records(item: dict) -> list[dict]:
    """The same rule, implemented differently.

    Subject matching and subsumption are decided on TOKEN LISTS here — a
    subject qualifies if its token sequence occurs contiguously in the
    question's, and loses to another candidate whose token sequence properly
    contains it — where the module uses regex lookarounds. Two implementations
    of one rule; they must agree on all 400 questions.
    """
    groups: dict = defaultdict(list)
    for num, txt in re.findall(r"^\s*(\d+)\.\s+(.*)$", item["context"], re.M):
        p = parse_fact(txt)
        if p:
            groups[(p[0], p[1])].append((int(num), p[2]))

    out = []
    for q, ans in zip(item["questions"], item["answers"]):
        gold = {str(a).lower() for a in
                (ans if isinstance(ans, (list, tuple)) else [ans])}
        qt = _tokens(q)
        cands = [k for k, vals in groups.items()
                 if _tokens(k[1]) and _is_sub(_tokens(k[1]), qt + ["\x00"])
                 and any(o.lower() in gold for _, o in vals)]
        kept = [k for k in cands
                if not any(_is_sub(_tokens(k[1]), _tokens(o[1])) for o in cands)]
        if not kept:
            out.append({"stratum": "unmatched", "gold": gold, "others": []})
            continue
        flags = {len({o for _, o in groups[k]}) > 1 for k in kept}
        if len(flags) > 1:
            out.append({"stratum": "ambiguous", "gold": gold, "others": []})
            continue
        best = sorted(kept, key=lambda k: (-len(k[1]), k[0]))[0]
        vals = groups[best]
        out.append({"stratum": "conflicted" if flags.pop() else "unique",
                    "gold": gold,
                    "others": [o for _, o in vals if o.lower() not in gold]})
    return out


def _oracle_counts(item: dict, run_payload: dict) -> dict:
    per = {s: {"n": 0, "correct": 0, "errors": Counter()} for s in STRATA}
    recs = _oracle_records(item)
    for row in run_payload["data"]:
        rec = recs[row["query_id"]]
        out = row["parsed_output"]
        bucket = per[rec["stratum"]]
        bucket["n"] += 1
        if any(normalize_answer(g) in normalize_answer(out) for g in rec["gold"]):
            bucket["correct"] += 1
            continue
        if not normalize_answer(out).strip():
            bucket["errors"]["empty"] += 1
        elif any(normalize_answer(o) and normalize_answer(o) in normalize_answer(out)
                 for o in rec["others"]):
            bucket["errors"]["stale_value"] += 1
        else:
            bucket["errors"]["off_list"] += 1
    return per


def test_module_agrees_with_the_independent_oracle_on_every_committed_run(payload):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {it["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
             .replace("factconsolidation_", ""): it for it in data}

    runs = {name: p for name, _, p in load_runs(EVIDENCE)}
    assert runs, "no committed run artifacts found"
    for run in payload["runs"]:
        item = items[run["subset"]]
        oracle = _oracle_counts(item, runs[run["run"]])
        for s in STRATA:
            assert run["strata"][s]["n"] == oracle[s]["n"], (run["run"], s)
            assert run["strata"][s]["correct"] == oracle[s]["correct"], (run["run"], s)
            for c in ERROR_CLASSES:
                assert run["strata"][s]["errors"][c] == oracle[s]["errors"].get(c, 0), \
                    (run["run"], s, c)


def test_module_agrees_with_the_independent_oracle_on_all_400_questions():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    n = 0
    for item in data:
        name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        if not name.startswith("factconsolidation_sh_"):
            continue
        mine = [r["stratum"] for r in classify_questions(item)]
        theirs = [r["stratum"] for r in _oracle_records(item)]
        assert mine == theirs, name
        n += len(mine)
    assert n == 400


def test_subset_of_run_reads_the_runs_own_config():
    assert subset_of_run({"dataset_config": {"sub_dataset": "factconsolidation_sh_6k"}}) \
        == "sh_6k"
    assert subset_of_run({}) == "unknown"
