"""audit_runner: ordering, pricing, budget, resume, prompt integrity.

No network anywhere — the Judge client is never instantiated.
"""
import hashlib
import json

import pytest

from hnav.labeling.audit_runner import (
    FEWSHOT, REASON_CODES, SCHEMA, SYSTEM_PROMPT, VERDICT_FIELDS,
    BudgetTracker, ab_order, build_messages, is_subsumption_candidate,
    load_done, pilot_selection, price, priority_order,
)

# sha256 of the user's verbatim judge prompt (2026-08-24). If this test fails,
# someone edited SYSTEM_PROMPT — which the audit protocol forbids.
PROMPT_SHA = "a3632d8b7b29e97c103abcf881be192f17c5f1f4de48c62b44101ec23c342a30"


def _rec(pid, tagged=False, both_parse=True, same_key=False, same_rel=False,
         same_subj=False, same_obj=False, obj_a="x", obj_b="y"):
    return {
        "pair_id": pid, "subset": pid.split(":")[0],
        "fact_a": f"fact A of {pid}", "fact_b": f"fact B of {pid}",
        "parser_tagged_conflict": tagged,
        "parser_metadata": {
            "both_parse": both_parse, "same_key": same_key,
            "same_relation": same_rel, "same_subject": same_subj,
            "same_object": same_obj,
            "fact_a_parsed": {"object": obj_a} if both_parse else None,
            "fact_b_parsed": {"object": obj_b} if both_parse else None,
        },
    }


def test_system_prompt_is_the_users_verbatim_text():
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == PROMPT_SHA


def test_fewshot_answers_satisfy_the_schema():
    for _, ans in FEWSHOT:
        assert set(ans) == set(VERDICT_FIELDS)
        assert ans["reason_code"] in REASON_CODES
        assert len(ans["explanation"]) <= 80
        for f in VERDICT_FIELDS[:7]:
            assert isinstance(ans[f], bool)
        # update conflict must equal the conjunction the prompt defines
        expect = (ans["same_referent"] and ans["same_relation"]
                  and ans["context_overlap"] and ans["values_incompatible"])
        assert ans["update_conflict"] == expect
    assert SCHEMA["required"] == VERDICT_FIELDS
    assert SCHEMA["additionalProperties"] is False


def test_ab_order_is_deterministic_and_mixed():
    ids = [f"sh_6k:{i}-{i+1}" for i in range(200)]
    orders = [ab_order(p) for p in ids]
    assert orders == [ab_order(p) for p in ids]
    assert {"ab", "ba"} == set(orders)          # both orders actually occur


def test_build_messages_swaps_facts_for_ba_order():
    rec = _rec("sh_6k:1-2")
    msgs, order = build_messages(rec)
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert len(msgs) == 2 + 2 * len(FEWSHOT)
    last = msgs[-1]["content"]
    if order == "ab":
        assert last == f"A: {rec['fact_a']}\nB: {rec['fact_b']}"
    else:
        assert last == f"A: {rec['fact_b']}\nB: {rec['fact_a']}"


def test_priority_order_tagged_first_then_blindspots_then_bulk():
    bulk = [_rec(f"sh_64k:{i}-{i+1}") for i in range(50)]
    recs = (bulk[:25]
            + [_rec("sh_6k:1-2", tagged=True, same_key=True, same_rel=True,
                    same_subj=True)]
            + [_rec("sh_6k:3-4", both_parse=False)]        # unparsed channel
            + [_rec("sh_6k:5-6", same_subj=True)]          # cross-template
            + bulk[25:])
    out = priority_order(recs, bulk_sample_n=10)
    ids = [r["pair_id"] for r in out]
    assert len(out) == len(recs)
    assert ids[0] == "sh_6k:1-2"                           # tagged first
    assert ids[1] == "sh_6k:3-4"                           # then unparsed
    assert ids[2] == "sh_6k:5-6"                           # then channels
    assert set(ids[3:]) == {r["pair_id"] for r in bulk}
    # deterministic across calls
    assert ids == [r["pair_id"] for r in priority_order(recs, bulk_sample_n=10)]


def test_pilot_selection_stratifies_and_forces_subsumption_pairs():
    tagged = [_rec(f"sh_6k:{i}-{i+1}", tagged=True, same_key=True,
                   same_rel=True, same_subj=True, obj_a=f"a{i}", obj_b=f"b{i}")
              for i in range(0, 40, 2)]
    subs = _rec("sh_6k:99-100", tagged=True, same_key=True, same_rel=True,
                same_subj=True, obj_a="rugby", obj_b="rugby union")
    untag = [_rec(f"sh_64k:{i}-{i+1}") for i in range(40)]
    ordered = priority_order(tagged + [subs] + untag, bulk_sample_n=5)
    pick = pilot_selection(ordered, 20)
    assert len(pick) == 20
    assert sum(r["parser_tagged_conflict"] for r in pick) == 10
    assert is_subsumption_candidate(subs)
    assert "sh_6k:99-100" in {r["pair_id"] for r in pick}   # force-included


def test_price_matches_hand_computation():
    usage = {"prompt_tokens": 1400, "completion_tokens": 90,
             "prompt_tokens_details": {"cached_tokens": 1200}}
    expect = 200 * 0.25e-6 + 1200 * 0.03e-6 + 90 * 2.00e-6
    assert price(usage) == pytest.approx(expect)
    assert price({}) == 0.0


def test_budget_tracker_trips_at_the_cap():
    b = BudgetTracker(1.0)
    assert b.allow()
    b.add(0.5)
    assert b.allow()
    b.add(0.5)
    assert not b.allow()                       # spent == cap -> stop
    assert b.spent == pytest.approx(1.0)


def test_resume_skips_completed_pair_ids(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        json.dumps({"pair_id": "sh_6k:1-2"}) + "\n"
        + json.dumps({"pair_id": "sh_6k:3-4"}) + "\n", encoding="utf-8")
    done = load_done(out)
    assert done == {"sh_6k:1-2", "sh_6k:3-4"}
    todo = [r for r in [_rec("sh_6k:1-2"), _rec("sh_6k:5-6")]
            if r["pair_id"] not in done]
    assert [r["pair_id"] for r in todo] == ["sh_6k:5-6"]
    assert load_done(tmp_path / "missing.jsonl") == set()
