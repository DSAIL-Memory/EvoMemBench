"""The detector-gap measurement, checked against hand-computed answers.  [T13]

The script's job is to make ONE number defensible — detector-achieved over
oracle-achieved — so the parts that could quietly corrupt it are the parts
tested here, each against an answer worked out by hand rather than against
"it ran":

* the parse-derived ground truth (which facts are genuinely superseded, which
  pairs are true supersession pairs), on a fixture where the answer is obvious;
* the arm construction, including the assertion that the probe-style arm and
  the SHIPPED ``page_edit`` path produce the same bytes — if they ever diverge
  the ratio compares two different interventions;
* the detection metrics, from a hand-built ``GateDecision`` whose true/false
  verifications and suppression classes are counted by inspection;
* the pre-registered operating-point rule, including the two cases it exists to
  refuse (a higher-recall cell with the identity screen off, and a cell that
  would delete a fact carrying its key's current value);
* the ratios, and the split refusal.

No socket, no torch, no embedding cache, no model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hnav.adapters.mab_adapter import explode_facts, page_edit
from hnav.core.read_gate import ConflictGroup, GateDecision, PairCheck
from hnav.stage1 import detector_gap as D
from hnav.stage1.stale_suppression_probe import render_context, split_context

# ── a fixture whose ground truth is obvious by inspection ────────────────────
# key A = (capital, Atlantis): 1 Poseidonis -> 3 Byzantium -> 4 Kallipolis
#         so facts 1 and 3 are superseded; 4 is the latest.
# key B = (capital, Numenor): a single value  -> nothing superseded.
# key C = (chairperson, Fatah): a single value.
ITEM = {
    "context": (
        "Here is a list of facts:\n"
        "0. The capital of Numenor is Armenelos.\n"
        "1. The capital of Atlantis is Poseidonis.\n"
        "2. The chairperson of Fatah is Ada Lovelace.\n"
        "3. The capital of Atlantis is Byzantium.\n"
        "4. The capital of Atlantis is Kallipolis.\n"
        "5. Thomas Kyd was born in the city of London.\n"
    ),
    "questions": ["What is the capital of Atlantis?",
                  "What is the capital of Numenor?"],
    "answers": [["Kallipolis"], ["Armenelos"]],
    "metadata": {"qa_pair_ids": ["factconsolidation_sh_6k_no0",
                                 "factconsolidation_sh_6k_no1"]},
}
RAW_CTX = ITEM["context"]
FACTS = [(0, "The capital of Numenor is Armenelos."),
         (1, "The capital of Atlantis is Poseidonis."),
         (2, "The chairperson of Fatah is Ada Lovelace."),
         (3, "The capital of Atlantis is Byzantium."),
         (4, "The capital of Atlantis is Kallipolis."),
         (5, "Thomas Kyd was born in the city of London.")]


@pytest.fixture(scope="module")
def table():
    return D.fact_table(ITEM)


# ── ground truth, decided without reading an answer ──────────────────────────
def test_superseded_is_exactly_the_two_older_atlantis_facts(table):
    assert table["superseded"] == {"fact:1", "fact:3"}


def test_latest_and_latest_object_per_key(table):
    """The relation half of the key is whatever ``conflict_analysis.parse``
    produces, so the fixture selects on the SUBJECT rather than hardcoding a
    template string this test has no business owning."""
    atlantis = [k for k in table["latest"] if k[1] == "Atlantis"]
    assert len(atlantis) == 1
    k = atlantis[0]
    assert table["latest"][k] == 4 and table["latest_obj"][k] == "Kallipolis"


def test_gt_pairs_are_the_three_atlantis_pairs(table):
    ids = [f"fact:{i}" for i in range(6)]
    pairs = D.gt_pairs(ids, table["by_id"])
    assert pairs == {("fact:1", "fact:3"), ("fact:1", "fact:4"),
                     ("fact:3", "fact:4")}
    assert D.count_gt_pairs(ids, table["by_id"]) == 3


def test_count_gt_pairs_matches_the_quadratic_form_and_ignores_duplicates(table):
    """Two facts of one key with the SAME object are not a supersession pair —
    the linear counter must subtract them exactly like the set builder does."""
    item = dict(ITEM)
    item["context"] = ("Here is a list of facts:\n"
                       "0. The capital of Atlantis is Poseidonis.\n"
                       "1. The capital of Atlantis is Poseidonis.\n"
                       "2. The capital of Atlantis is Byzantium.\n")
    t = D.fact_table(item)
    ids = ["fact:0", "fact:1", "fact:2"]
    assert D.count_gt_pairs(ids, t["by_id"]) == len(D.gt_pairs(ids, t["by_id"])) == 2
    assert t["superseded"] == {"fact:0", "fact:1"}


# ── arms ─────────────────────────────────────────────────────────────────────
PLAN = {"suppress_serials": [1, 3], "demote_serials": [4]}


def test_detector_suppress_removes_exactly_the_named_serials():
    out = D.arm_facts("detector_suppress", FACTS, PLAN)
    assert out == [FACTS[0], FACTS[2], FACTS[4], FACTS[5]]
    assert [s for s, _ in out] == [0, 2, 4, 5], "serials are never renumbered"


def test_detector_demote_moves_the_named_serials_to_the_end():
    out = D.arm_facts("detector_demote_late", FACTS, PLAN)
    assert out[-1] == FACTS[4]
    assert sorted(out) == sorted(FACTS), "a permutation: nothing added or dropped"


def test_detector_anti_is_the_exact_mirror():
    out = D.arm_facts("detector_anti", FACTS, PLAN)
    assert out[0] == FACTS[4]
    assert sorted(out) == sorted(FACTS)


def test_multiple_demotions_put_the_highest_serial_last_and_first():
    plan = {"suppress_serials": [], "demote_serials": [1, 4]}
    late = D.arm_facts("detector_demote_late", FACTS, plan)
    assert late[-2:] == [FACTS[1], FACTS[4]]
    anti = D.arm_facts("detector_anti", FACTS, plan)
    assert anti[:2] == [FACTS[4], FACTS[1]]


def test_native_arms_are_the_untouched_fact_list():
    for arm in ("native", "native_repeat"):
        assert D.arm_facts(arm, FACTS, PLAN) == FACTS


def test_the_shipped_page_contract_agrees_with_the_probe_style_arm():
    """The guard the run refuses on. If ``page_edit`` and the probe helpers
    ever disagree, the measured effect is not the mechanism that ships."""
    preamble, facts = split_context(ITEM["context"])
    for arm in ("detector_suppress", "detector_demote_late"):
        want = render_context(preamble, D.arm_facts(arm, facts, PLAN))
        assert D.shipped_page(arm, ITEM["context"], PLAN) == want
    assert D.shipped_page("detector_anti", ITEM["context"], PLAN) is None


# ── detection metrics ────────────────────────────────────────────────────────
def _decision(groups, checks, ambiguous=True):
    return GateDecision(ambiguous=ambiguous, groups=groups, pair_checks=checks)


def _grp(latest, stale):
    return ConflictGroup(member_ids=[latest] + list(stale), latest_id=latest,
                         stale_ids=list(stale), residuals={})


def _chk(a, b, verified=True):
    return PairCheck(id_a=a, id_b=b, cos=0.99, verified=verified)


def test_score_decision_counts_a_perfect_detection(table):
    strat = {"stratum": "conflicted", "key": [k for k in table["latest"]
                                              if k[1] == "Atlantis"][0],
             "member_serials": [1, 3, 4], "target_serial": 4}
    dec = _decision([_grp("fact:4", ["fact:1", "fact:3"])],
                    [_chk("fact:1", "fact:3"), _chk("fact:1", "fact:4"),
                     _chk("fact:3", "fact:4")])
    m = D.blank_metrics()
    D.score_decision(m, dec, table, strat, [f"fact:{i}" for i in range(6)], 3)
    out = D.finish_metrics(m)
    assert (out["tp"], out["fp"]) == (3, 0)
    assert out["pair_precision"] == 1.0
    assert out["pair_recall_pool"] == 1.0 and out["pair_recall_page"] == 1.0
    assert out["n_suppressed"] == 2 and out["n_suppressed_superseded"] == 2
    assert out["n_suppressed_harmful"] == 0
    assert out["n_demoted"] == 1 and out["n_demoted_is_latest"] == 1
    assert out["n_conflicted_hit"] == 1 and out["n_conflicted_gold_cut"] == 0
    assert out["question_recall_conflicted"] == 1.0 and out["coverage"] == 1.0


def test_a_cross_key_verification_is_a_false_positive_and_a_harmful_cut(table):
    """Two facts of DIFFERENT keys verified as a conflict: the pair is a false
    positive, and suppressing the older one deletes a fact that carries its own
    key's current value. Both counters must fire."""
    strat = {"stratum": "unique", "key": None, "member_serials": [],
             "target_serial": None}
    dec = _decision([_grp("fact:5", ["fact:2"])], [_chk("fact:2", "fact:5")])
    m = D.blank_metrics()
    D.score_decision(m, dec, table, strat, [f"fact:{i}" for i in range(6)], 3)
    out = D.finish_metrics(m)
    assert (out["tp"], out["fp"]) == (0, 1) and out["pair_precision"] == 0.0
    assert out["n_suppressed_harmful"] == 1 and out["fact_precision"] == 0.0
    assert out["n_unique_touched"] == 1


def test_suppressing_a_same_valued_duplicate_is_neither_true_nor_harmful():
    item = dict(ITEM)
    item["context"] = ("Here is a list of facts:\n"
                       "0. The capital of Atlantis is Kallipolis.\n"
                       "1. The capital of Atlantis is Kallipolis.\n")
    t = D.fact_table(item)
    strat = {"stratum": "unique", "key": None, "member_serials": [],
             "target_serial": None}
    dec = _decision([_grp("fact:1", ["fact:0"])], [_chk("fact:0", "fact:1")])
    m = D.blank_metrics()
    D.score_decision(m, dec, t, strat, ["fact:0", "fact:1"], 0)
    out = D.finish_metrics(m)
    assert out["n_suppressed_same_value"] == 1
    assert out["n_suppressed_superseded"] == 0 and out["n_suppressed_harmful"] == 0


def test_cutting_the_gold_fact_is_counted_and_kills_the_hit(table):
    key = [k for k in table["latest"] if k[1] == "Atlantis"][0]
    strat = {"stratum": "conflicted", "key": key, "member_serials": [1, 3, 4],
             "target_serial": 4}
    dec = _decision([_grp("fact:3", ["fact:1", "fact:4"])], [])
    m = D.blank_metrics()
    D.score_decision(m, dec, table, strat, [f"fact:{i}" for i in range(6)], 3)
    out = D.finish_metrics(m)
    assert out["n_conflicted_gold_cut"] == 1
    assert out["n_conflicted_hit"] == 0, \
        "the key's newest fact was deleted; that is not a detection"


# ── the pre-registered selection rule ────────────────────────────────────────
def _cell(cos=0.90, rlab="off", amb="none", nli=0.5, filt=True, recall=0.5,
          harmful=0):
    m = D.blank_metrics()
    m.update({"n_suppressed_harmful": harmful})
    m = D.finish_metrics(m)
    m["pair_recall_pool"] = recall
    return {"cos_pair": cos, "r_min_label": rlab, "r_min": D.r_min_of(rlab),
            "ambiguity_mode": amb, "nli_contradiction": nli,
            "pair_filter": filt, "metrics": m}


def test_selection_maximises_recall_among_admissible_cells():
    cells = [_cell(recall=0.4), _cell(cos=0.94, recall=0.9), _cell(recall=0.7)]
    assert D.select(cells)["cos_pair"] == 0.94


def test_selection_refuses_a_higher_recall_cell_with_the_identity_screen_off():
    """The T11 lesson made a rule: with ``pair_filter`` off, ~86% of
    verifications were spurious and the apparent gains were noise."""
    cells = [_cell(recall=0.5), _cell(filt=False, recall=0.99)]
    assert D.select(cells)["pair_filter"] is True


def test_selection_refuses_any_cell_that_would_delete_a_current_value():
    cells = [_cell(recall=0.5), _cell(recall=0.99, harmful=1)]
    assert D.select(cells)["metrics"]["pair_recall_pool"] == 0.5


def test_selection_returns_none_when_nothing_qualifies():
    assert D.select([_cell(filt=False, recall=0.9),
                     _cell(recall=0.9, harmful=3)]) is None


def test_ties_break_towards_the_more_conservative_gate():
    cells = [_cell(cos=0.90, nli=0.5, rlab="off", amb="none", recall=0.8),
             _cell(cos=0.94, nli=0.5, rlab="off", amb="none", recall=0.8),
             _cell(cos=0.94, nli=0.99, rlab="off", amb="none", recall=0.8),
             _cell(cos=0.94, nli=0.99, rlab="frozen", amb="none", recall=0.8),
             _cell(cos=0.94, nli=0.99, rlab="frozen", amb="all", recall=0.8)]
    best = D.select(cells)
    assert (best["cos_pair"], best["nli_contradiction"], best["r_min_label"],
            best["ambiguity_mode"]) == (0.94, 0.99, "frozen", "all")


def test_the_selection_rule_is_declared_not_inferred():
    assert "pair_filter is True" in D.SELECTION_RULE["require"]
    assert "n_suppressed_harmful == 0" in D.SELECTION_RULE["require"]
    assert D.SELECTION_RULE["maximise"] == "pair_recall_pool"


# ── the headline ratio ───────────────────────────────────────────────────────
def _acc_block(correct, n):
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None}


def _fake_result():
    return {
        "subset": "sh_6k",
        "arms": {"native": _acc_block(29, 100),
                 "detector_suppress": _acc_block(80, 100),
                 "detector_demote_late": _acc_block(40, 100),
                 "detector_anti": _acc_block(27, 100)},
        "paired_vs_native": {"detector_suppress": {"net": 51},
                             "detector_demote_late": {"net": 11},
                             "detector_anti": {"net": -2}},
        "by_stratum": {"conflicted": {"arms": {
            "native": _acc_block(4, 74),
            "detector_suppress": _acc_block(55, 74),
            "detector_demote_late": _acc_block(15, 74),
            "detector_anti": _acc_block(2, 74)}}},
    }


def _fake_oracle():
    return {"file": "stage0_results/stage1/x.json", "harness": {},
            "result": {
                "subset": "sh_6k",
                "arms": {"native": _acc_block(29, 100),
                         "oracle_suppress": _acc_block(91, 100),
                         "oracle_recency": _acc_block(46, 100),
                         "anti": _acc_block(26, 100)},
                "paired_vs_native": {"oracle_suppress": {"net": 62},
                                     "oracle_recency": {"net": 17},
                                     "anti": {"net": -3}},
                "by_stratum": {"conflicted": {"arms": {
                    "native": _acc_block(4, 74),
                    "oracle_suppress": _acc_block(66, 74),
                    "oracle_recency": _acc_block(20, 74),
                    "anti": _acc_block(1, 74)}}}}}


def test_ratios_are_computed_exactly():
    cmp_ = D.compare_to_oracle(_fake_result(), _fake_oracle())
    sup = cmp_["by_mechanism"]["detector_suppress"]
    assert sup["oracle_arm"] == "oracle_suppress"
    assert sup["net_ratio"] == pytest.approx(51 / 62)
    assert sup["detector_conflicted_gain"] == 51 and sup["oracle_conflicted_gain"] == 62
    assert sup["conflicted_gain_ratio"] == pytest.approx(51 / 62)
    dem = cmp_["by_mechanism"]["detector_demote_late"]
    assert dem["oracle_arm"] == "oracle_recency"
    assert dem["net_ratio"] == pytest.approx(11 / 17)
    assert cmp_["native_cross_run"]["identical"] is True


def test_a_zero_oracle_effect_yields_no_ratio_rather_than_a_division():
    oracle = _fake_oracle()
    oracle["result"]["paired_vs_native"]["oracle_suppress"]["net"] = 0
    cmp_ = D.compare_to_oracle(_fake_result(), oracle)
    assert cmp_["by_mechanism"]["detector_suppress"]["net_ratio"] is None


def test_a_drifted_native_arm_is_flagged_not_silently_ratioed():
    oracle = _fake_oracle()
    oracle["result"]["arms"]["native"] = _acc_block(31, 100)
    cmp_ = D.compare_to_oracle(_fake_result(), oracle)
    assert cmp_["native_cross_run"]["identical"] is False


def test_no_oracle_file_means_no_comparison_block():
    assert D.compare_to_oracle(_fake_result(), None) is None


# ── guards ───────────────────────────────────────────────────────────────────
def test_confirmatory_subsets_are_refused(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["detector_gap.py", "--subsets", "sh_64k"])
    assert D.main() == 2
    assert "REFUSED" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["detector_gap.py", "--subsets", "sh_262k"])
    assert D.main() == 2


def test_a_run_without_a_frozen_operating_point_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "OPERATING_POINT", tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="--select"):
        D.frozen_cell()


def test_frozen_cell_reads_the_committed_artifact(monkeypatch, tmp_path):
    art = tmp_path / "op.json"
    art.write_text(json.dumps({
        "thresholds": {"cos_pair": 0.9, "r_min": None, "nmargin": 1.0,
                       "H_z": 2.0, "ambiguity_mode": "none",
                       "nli_contradiction": 0.99},
        "r_min_label": "off", "pair_filter": True}), encoding="utf-8")
    monkeypatch.setattr(D, "OPERATING_POINT", art)
    cell = D.frozen_cell()
    assert cell["cos_pair"] == 0.9 and cell["ambiguity_mode"] == "none"
    assert cell["nli_contradiction"] == 0.99 and cell["pair_filter"] is True


def test_cosine_error_is_zero_on_the_vectors_that_produced_the_pairs():
    v = np.eye(4, dtype=np.float32)
    order = ["fact:0", "fact:1", "fact:2", "fact:3"]
    pp = {"questions": [{"pairs": [
        {"a": "fact:0", "b": "fact:1", "cos": 0.0},
        {"a": "fact:2", "b": "fact:2", "cos": 1.0}]}]}
    assert D.cosine_error(v, order, [pp]) == pytest.approx(0.0, abs=1e-12)


def test_cosine_error_catches_a_wrong_namespace():
    v = np.eye(4, dtype=np.float32)
    order = ["fact:0", "fact:1"]
    pp = {"questions": [{"pairs": [{"a": "fact:0", "b": "fact:1", "cos": 0.97}]}]}
    assert D.cosine_error(v, order, [pp]) == pytest.approx(0.97)


def test_no_verifiable_pair_is_infinite_error_not_zero():
    """An unverifiable namespace must never pass the check by default."""
    assert D.cosine_error(np.eye(2, dtype=np.float32), ["a", "b"],
                          [{"questions": []}]) == float("inf")


def test_load_fact_vectors_refuses_when_nothing_reproduces_the_geometry(
        monkeypatch, tmp_path):
    from hnav import config as _config

    cfg = _config.HNavConfig(cache_dir=tmp_path)
    monkeypatch.setattr(
        "hnav.core.embedding.DiskCachedEmbedder.encode",
        lambda self, texts: np.eye(len(texts), dtype=np.float32))
    pp = {"questions": [{"pairs": [{"a": "fact:0", "b": "fact:1", "cos": 0.99}]}]}
    with pytest.raises(SystemExit, match="REFUSED"):
        D.load_fact_vectors(cfg, ["fact:0", "fact:1"], ["a", "b"], [pp])


# ── the retrieval-path harness (pre-registration v2 section 10) ──────────────
# On the calibration split retrieval is COMPLETE, so this harness changes only
# the block structure and order. What has to be right is that the arms go
# through the shipped page contract and that the integrity check can actually
# fail — a check that cannot fail is decoration, and this one is what refuses
# the run before any call is sent.
PAGE_A = "0. Alpha is one. 1. Beta is two."
PAGE_B = "2. Gamma is three. 3. Delta is four."
R_PLAN = {"suppress_serials": [1], "demote_serials": [3]}


def test_retrieval_suppress_goes_through_the_shipped_page_edit():
    out = D.retrieval_arm_page("detector_suppress", [PAGE_A, PAGE_B], R_PLAN)
    assert out == ["0. Alpha is one.", PAGE_B]
    assert out == page_edit([PAGE_A, PAGE_B], drop_ids=["fact:1"])[0]


def test_retrieval_demote_goes_through_the_shipped_page_edit():
    out = D.retrieval_arm_page("detector_demote_late", [PAGE_A, PAGE_B], R_PLAN)
    assert out == page_edit([PAGE_A, PAGE_B], move_last_ids=["fact:3"])[0]
    assert out == [PAGE_A, "2. Gamma is three. 3. Delta is four."]


def test_retrieval_anti_puts_the_newest_fact_first_across_chunks():
    plan = {"suppress_serials": [], "demote_serials": [3]}
    out = D.retrieval_arm_page("detector_anti", [PAGE_A, PAGE_B], plan)
    assert out == ["3. Delta is four. 0. Alpha is one. 1. Beta is two.",
                   "2. Gamma is three."]
    assert D.page_facts(out) == D.page_facts([PAGE_A, PAGE_B]), "a permutation"


def test_retrieval_anti_orders_several_moves_newest_first():
    plan = {"suppress_serials": [], "demote_serials": [1, 3]}
    out = D.retrieval_arm_page("detector_anti", [PAGE_A, PAGE_B], plan)
    assert out[0].startswith("3. Delta is four. 1. Beta is two. 0. Alpha is one.")
    assert D.page_facts(out) == D.page_facts([PAGE_A, PAGE_B])


def test_retrieval_anti_is_the_exact_mirror_of_demote():
    """Same facts, same primitive, opposite end — the direction control has to
    differ from DEMOTE_LATE only in where the fact lands."""
    plan = {"suppress_serials": [], "demote_serials": [0]}
    late = D.retrieval_arm_page("detector_demote_late", [PAGE_A, PAGE_B], plan)
    front = D.retrieval_arm_page("detector_anti", [PAGE_A, PAGE_B], plan)
    assert late[-1].endswith("0. Alpha is one.")
    assert front[0].startswith("0. Alpha is one.")
    assert D.page_facts(late) == D.page_facts(front) == D.page_facts([PAGE_A, PAGE_B])


def test_front_move_preserves_a_preamble():
    page = ["Here is a list of facts:\n0. A is X.\n1. B is Y.\n"]
    out = D.page_move_front(page, ["fact:1"])
    assert out == ["Here is a list of facts:\n1. B is Y.\n0. A is X.\n"]


def test_front_move_refuses_an_id_that_is_not_on_the_page():
    with pytest.raises(LookupError, match="not on this page"):
        D.page_move_front([PAGE_A], ["fact:99"])


def test_front_move_with_no_ids_is_the_identity():
    assert D.page_move_front([PAGE_A, PAGE_B], []) == [PAGE_A, PAGE_B]


def test_integrity_check_passes_on_every_honest_arm():
    page = [PAGE_A, PAGE_B]
    for arm in D.ARMS:
        edited = D.retrieval_arm_page(arm, page, R_PLAN)
        assert not D.retrieval_integrity(arm, page, edited, R_PLAN), arm


@pytest.mark.parametrize("arm,bad", [
    ("detector_suppress", ["0. Alpha is one.", "2. Gamma is three."]),  # dropped too much
    ("detector_suppress", [PAGE_A, PAGE_B]),                            # dropped nothing
    ("detector_demote_late", ["0. Alpha is one.", PAGE_B]),             # lost a fact
    ("detector_anti", [PAGE_A]),                                        # lost a chunk
    ("native", ["0. Alpha is one.", PAGE_B]),                           # native must not move
])
def test_integrity_check_can_actually_fail(arm, bad):
    assert D.retrieval_integrity(arm, [PAGE_A, PAGE_B], bad, R_PLAN)


def test_chunk_rebuild_refuses_a_prepass_it_disagrees_with(sh_6k):
    """The rebuilt chunks must carry the prepass's own fact->chunk assignment;
    otherwise the page is not the one the gate's geometry was measured on."""
    good = {}
    from hnav.stage0.m2_retrieval_calibration import build_chunks
    chunks, fallback = build_chunks(sh_6k["context"], D.CHUNK_SIZE)
    assert not fallback
    for i, c in enumerate(chunks):
        for serial, _ in explode_facts(c):
            good[f"fact:{serial}"] = f"chunk:{i}"
    assert D.chunk_texts_for(sh_6k, {"fact_chunk": good}) == chunks
    drifted = dict(good)
    drifted["fact:0"] = "chunk:99"
    with pytest.raises(SystemExit, match="disagree with the prepass"):
        D.chunk_texts_for(sh_6k, {"fact_chunk": drifted})


def test_plan_subset_rejects_an_unknown_harness():
    with pytest.raises(ValueError, match="harness"):
        D.plan_subset({}, "sh_6k", {}, {}, 0, {}, harness="telepathy")


def test_cross_harness_comparison_is_labelled_not_silently_ratioed():
    res = _fake_result()
    res["harness"] = "retrieval"
    cmp_ = D.compare_to_oracle(res, _fake_oracle())
    assert cmp_["harness_match"] is False
    assert "NOT the detector/oracle ratio" in cmp_["harness_caveat"]
    plain = D.compare_to_oracle(_fake_result(), _fake_oracle())
    assert plain["harness_match"] is True and plain["harness_caveat"] is None


# ── page selection must come from the benchmark, never from us (Amendment 3) ──
def test_benchmark_pages_cover_every_question_and_index_real_chunks():
    """Measured: re-encoding cannot reproduce the benchmark's sh_64k page (26/100
    set agreement even at min cosine 0.99997), so the page is READ from the
    benchmark's own vectors. This pins the shape of what is read."""
    counts = {"sh_6k": 2, "sh_32k": 9, "sh_64k": 17}
    for subset, n_chunks in counts.items():
        pages = D.benchmark_pages(subset)
        if pages is None:
            pytest.skip(f"no refit artifact entry for {subset}")
        assert len(pages) == 100, subset
        for pg in pages:
            assert len(pg) == min(10, n_chunks), subset
            assert len(set(pg)) == len(pg), "a page cannot repeat a chunk"
            assert all(0 <= i < n_chunks for i in pg), subset


def test_a_complete_retrieval_page_holds_every_chunk():
    """sh_6k and sh_32k have 2 and 9 chunks against top_k 10, so the benchmark's
    page must be a permutation of ALL of them — which is exactly why page
    MEMBERSHIP is not at risk on the calibration split and is at sh_64k."""
    for subset, n_chunks in (("sh_6k", 2), ("sh_32k", 9)):
        pages = D.benchmark_pages(subset)
        if pages is None:
            pytest.skip(f"no refit artifact entry for {subset}")
        for pg in pages:
            assert sorted(pg) == list(range(n_chunks)), subset


def test_missing_artifact_yields_none_rather_than_a_guess(tmp_path):
    assert D.benchmark_pages("sh_64k", path=tmp_path / "absent.json") is None
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"subsets": {}}), encoding="utf-8")
    assert D.benchmark_pages("sh_64k", path=empty) is None


def test_the_retrieval_harness_refuses_to_default_its_page_source():
    """No default is the point: the two sources differ on 74% of sh_64k
    questions, so choosing between them must be an explicit act."""
    with pytest.raises(ValueError, match="explicit page_source"):
        D.plan_subset({"context": RAW_CTX, "questions": [], "answers": []},
                      "sh_6k", {"questions": [], "fact_chunk": {}}, {}, 0, {},
                      harness="retrieval")


def test_benchmark_page_source_refuses_when_nothing_is_recorded(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(D, "BENCHMARK_PAGES", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="do NOT fall back"):
        D.plan_subset({"context": RAW_CTX, "questions": [], "answers": []},
                      "sh_6k", {"questions": [], "fact_chunk": {}}, {}, 0, {},
                      harness="retrieval", page_source="benchmark")


def test_whole_context_harness_needs_no_page_source():
    plan = D.plan_subset(ITEM, "sh_6k", {"questions": []}, {}, 0,
                         D.fact_table(ITEM), harness="whole_context")
    assert plan["page_source"] is None and plan["harness"] == "whole_context"


# ── the registered harm taxonomy (pre-registration v2, Amendment 2) ──────────
def test_harm_classes_are_tested_in_the_registered_order():
    # gold_cut wins even when the output would also look like a refusal
    assert D.classify_harm("Rome", "The pool does not contain any fact", True) \
        == "gold_cut"
    # the measured "Shinzo Abe" -> "Sinzo Abe" shape
    assert D.classify_harm("Shinz\u014d Abe", "Sinz\u014d Abe", False) \
        == "malformed_generation"
    # the measured sh_32k retrieval q14 shape
    assert D.classify_harm(
        "Beirut", "The provided knowledge pool does not contain any fact about",
        False) == "refusal_after_edit"
    # a different, confidently wrong fact
    assert D.classify_harm("Rome", "Watertown", False) == "information_loss"
    # an empty answer is a refusal, not information loss
    assert D.classify_harm("Rome", "   ", False) == "refusal_after_edit"


def test_every_harm_class_still_counts_as_harm():
    """Amendment 2 is diagnostic, not permissive: naming a class must not
    remove it from the harm count."""
    per_q = [
        _harm_q(0, "conflicted", "Rome", "Watertown", target=5, cut=True),
        _harm_q(1, "unique", "Beirut", "does not contain any fact", target=9),
        _harm_q(2, "unique", "Shinz\u014d Abe", "Sinz\u014d Abe", target=11),
        _harm_q(3, "conflicted", "Paris", "Lyon", target=13),
    ]
    h = D.harm_report(per_q, "detector_suppress")
    assert h["n_harmed"] == 4, "naming a class must never drop it from the count"
    assert h["counts"] == {"gold_cut": 1, "malformed_generation": 1,
                           "refusal_after_edit": 1, "information_loss": 1}


def test_a_unique_refusal_voids_the_protective_claim():
    """The registered §5b verdict, restated: only a malformed generation is
    survivable on the protected stratum."""
    h = D.harm_report(
        [_harm_q(1, "unique", "Beirut", "does not contain any fact", target=9)],
        "detector_suppress")
    assert h["protective_claim_void"] is True and h["voiding_questions"] == [1]


def test_a_unique_malformed_generation_does_not_void():
    h = D.harm_report(
        [_harm_q(2, "unique", "Shinz\u014d Abe", "Sinz\u014d Abe", target=11)],
        "detector_suppress")
    assert h["protective_claim_void"] is False


def test_conflicted_harms_never_void_the_protective_claim():
    """The protective criterion is about the non-conflicted stratum; a gold_cut
    on a conflicted question is the PREDICTED mode, not a protective failure."""
    h = D.harm_report(
        [_harm_q(0, "conflicted", "Rome", "Watertown", target=5, cut=True)],
        "detector_suppress")
    assert h["counts"]["gold_cut"] == 1 and h["protective_claim_void"] is False


def test_an_arm_that_harms_nothing_reports_zeros():
    q = _harm_q(0, "unique", "Rome", "Rome", target=5)
    q["arms"]["detector_suppress"]["correct"] = True
    h = D.harm_report([q], "detector_suppress")
    assert h["n_harmed"] == 0 and h["protective_claim_void"] is False
    assert set(h["counts"]) == set(D.HARM_CLASSES)


def _harm_q(idx, stratum, native, arm, target=None, cut=False):
    return {"index": idx, "stratum": stratum, "target_serial": target,
            "plan": {"suppress_serials": [target] if cut else [],
                     "demote_serials": []},
            "arms": {"native": {"output": native, "correct": True},
                     "detector_suppress": {"output": arm, "correct": False}}}


# ── the containment guard (the one that stops a wiring bug wearing a null's
#    costume). A pool built from a different page than the model sees lets the
#    policy name an absent fact; page_edit raises, the live adapter fails open,
#    and the artifact records an intervention that never happened.
def test_containment_passes_when_every_named_fact_is_on_the_page():
    plan = {"suppress_serials": [1], "demote_serials": [3]}
    assert D.containment_violations([PAGE_A, PAGE_B], plan) == []


def test_containment_names_exactly_the_absent_ids():
    plan = {"suppress_serials": [1, 42], "demote_serials": [3, 99]}
    assert D.containment_violations([PAGE_A, PAGE_B], plan) == \
        ["fact:42", "fact:99"]


def test_containment_catches_the_real_failure_mode(sh_6k):
    """The exact bug the guard exists for: a pool built from a page that
    includes chunk 1, applied to a page that does not."""
    from hnav.stage0.m2_retrieval_calibration import build_chunks
    chunks, fallback = build_chunks(sh_6k["context"], D.CHUNK_SIZE)
    assert not fallback and len(chunks) == 2
    only_first = [chunks[0]]
    in_second = [s for s, _ in explode_facts(chunks[1])][:3]
    plan = {"suppress_serials": list(in_second), "demote_serials": []}
    missing = D.containment_violations(only_first, plan)
    assert missing == [f"fact:{s}" for s in sorted(in_second)]
    # ... and the same plan is clean against the full page
    assert D.containment_violations(chunks, plan) == []


def test_an_empty_plan_can_never_violate_containment():
    assert D.containment_violations([PAGE_A], {"suppress_serials": [],
                                               "demote_serials": []}) == []


def test_prepass_path_separates_the_two_page_sources():
    from hnav import config as _config
    cfg = _config.HNavConfig()
    a = D.prepass_path(cfg, "sh_64k", "benchmark")
    b = D.prepass_path(cfg, "sh_64k", "prepass")
    assert a != b
    assert a.name == "stage1_prepass_sh_64k_benchmarkpage.json"
    assert b.name == "stage1_prepass_sh_64k.json"
    assert D.prepass_path(cfg, "sh_64k", None) == b, \
        "the whole-context harness keeps the original filename"


# ── G1/G2/G3: the guard package, each able to fail ───────────────────────────
def test_pool_violations_catch_a_pool_that_escaped_the_page():
    """G1. The named ids are a subset of the pool, so a pool that already
    escapes the page is the upstream bug — caught one step earlier."""
    assert D.pool_violations([PAGE_A, PAGE_B], ["fact:0", "fact:3"]) == []
    assert D.pool_violations([PAGE_A], ["fact:0", "fact:2", "fact:3"]) == \
        ["fact:2", "fact:3"]


def test_pool_and_named_checks_are_against_the_same_page_object(sh_6k):
    """Both guards take the page they are handed, so neither can be satisfied
    by a page other than the one page_edit will edit."""
    from hnav.stage0.m2_retrieval_calibration import build_chunks
    chunks, _ = build_chunks(sh_6k["context"], D.CHUNK_SIZE)
    second = [s for s, _ in explode_facts(chunks[1])][:2]
    plan = {"suppress_serials": second, "demote_serials": []}
    pool = [f"fact:{s}" for s in second]
    assert D.containment_violations([chunks[0]], plan)
    assert D.pool_violations([chunks[0]], pool)
    assert not D.containment_violations(chunks, plan)
    assert not D.pool_violations(chunks, pool)


# ── the confirmatory prepass refuses everything except its one job ───────────
def test_confirmatory_prepass_refuses_any_other_subset(monkeypatch, capsys):
    from hnav.stage1 import confirmatory_prepass as CP
    for subset in ("sh_6k", "sh_32k", "sh_262k"):
        monkeypatch.setattr("sys.argv",
                            ["confirmatory_prepass.py", "--subset", subset])
        assert CP.main() == 2, subset
        assert "REFUSED" in capsys.readouterr().out


def test_confirmatory_prepass_forces_benchmark_page_and_cache_only(monkeypatch,
                                                                   capsys):
    """The two settings that must not be overridable from the command line:
    the pool has to come from the page the model sees, and no vector may be
    computed fresh on the confirmatory subset."""
    from hnav.stage1 import confirmatory_prepass as CP
    monkeypatch.setattr("sys.argv", ["confirmatory_prepass.py", "--dry-run"])
    assert CP.main() == 0
    out = capsys.readouterr().out
    assert "page source : benchmark  (forced" in out
    assert "cache-only (forced" in out
    assert "tunables    : none" in out
    import inspect
    src = inspect.getsource(CP.main)
    assert 'page_source="benchmark"' in src
    assert "cache_only=True" in src
    assert "--page-source" not in src, "page source must not be a CLI option"


# ── --confirmatory is the act of firing, and it is narrow ────────────────────
def test_confirmatory_refuses_any_subset_but_sh_64k(monkeypatch, capsys):
    for subset in ("sh_262k", "sh_6k"):
        monkeypatch.setattr("sys.argv", [
            "detector_gap.py", "--confirmatory", "--subsets", subset,
            "--harness", "retrieval", "--page-source", "benchmark"])
        assert D.main() == 2, subset
        assert "REFUSED" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--harness", "whole_context"],
    ["--harness", "retrieval", "--page-source", "prepass"],
    ["--harness", "retrieval"],
])
def test_confirmatory_refuses_any_design_but_the_pre_registered_one(
        monkeypatch, capsys, extra):
    monkeypatch.setattr("sys.argv",
                        ["detector_gap.py", "--confirmatory",
                         "--subsets", "sh_64k"] + extra)
    assert D.main() == 2
    assert "pre-registered confirmatory design" in capsys.readouterr().out


def test_sh_64k_stays_refused_without_the_confirmatory_flag(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "detector_gap.py", "--subsets", "sh_64k",
        "--harness", "retrieval", "--page-source", "benchmark"])
    assert D.main() == 2
    assert "outside the calibration split" in capsys.readouterr().out


def test_confirmatory_and_select_are_mutually_exclusive(monkeypatch, capsys):
    """--select FITS the grid. On the confirmatory subset that would be tuning
    on held-out data, and it would write an operating-point artifact that a
    later reader could mistake for the frozen one. Found in supervisor
    pre-flight; the firing command never used --select, but nothing may invoke
    sh_64k again without this refusal in place."""
    monkeypatch.setattr("sys.argv", [
        "detector_gap.py", "--confirmatory", "--subsets", "sh_64k",
        "--harness", "retrieval", "--page-source", "benchmark", "--select"])
    assert D.main() == 2
    out = capsys.readouterr().out
    assert "mutually exclusive" in out and "held-out" in out.lower()


# ── the void-condition object (supervisor note 1) ────────────────────────────
def _vc_result(**over):
    base = {
        "arms": {"native": _acc_block(45, 100)},
        "by_stratum": {"unique": {"arms": {"native": _acc_block(28, 34)}}},
        "aa_floor": {"b_native_only": 0, "c_arm_only": 0},
        "n_page_edit_mismatch": 0, "n_containment_violations": 0,
        "n_page_edit_errors": 0,
        "positive_control": {"ok": True, "n_questions": 100,
                             "n_questions_policy_fired": 100,
                             "n_fact_edits_applied": 200,
                             "n_facts_suppressed": 735},
        "harm": {"detector_suppress": {"protective_claim_void": False,
                                       "voiding_questions": [],
                                       "counts": {c: 0 for c in D.HARM_CLASSES}}},
        "per_question": [],
    }
    base.update(over)
    return base


def test_void_conditions_all_pass_on_a_clean_result(table):
    vc = D.void_condition_report(_vc_result(), table, "benchmark")
    entries = {k: v for k, v in vc.items() if k != "verdict"}
    assert len(entries) == 7, "conditions 1,2,3,4,5,6/7,8"
    assert all(v["status"] == "pass" for v in entries.values())
    assert vc["verdict"] == {"run_void": False, "run_void_because": [],
                             "protective_claim_void": False, "shot_spent": True}


def test_condition_5_voids_only_the_protective_claim(table):
    res = _vc_result(harm={"detector_suppress": {
        "protective_claim_void": True, "voiding_questions": [77],
        "counts": dict.fromkeys(D.HARM_CLASSES, 0)}})
    vc = D.void_condition_report(res, table, "benchmark")
    assert vc["5_protected_stratum"]["status"] == "fail"
    assert vc["5_protected_stratum"]["voids"] == "protective_claim_only"
    assert vc["verdict"]["run_void"] is False, "the run and the shot still stand"
    assert vc["verdict"]["protective_claim_void"] is True
    assert vc["verdict"]["shot_spent"] is True


@pytest.mark.parametrize("over,cond", [
    ({"n_page_edit_mismatch": 1}, "1_page_edit_mismatch"),
    ({"arms": {"native": _acc_block(80, 100)}}, "2_native_in_band"),
    ({"by_stratum": {"unique": {"arms": {"native": _acc_block(20, 34)}}}},
     "2_native_in_band"),
    ({"aa_floor": {"b_native_only": 1, "c_arm_only": 0}}, "3_aa_floor_zero"),
    ({"n_containment_violations": 1}, "8_guards_and_positive_control"),
    ({"n_page_edit_errors": 1}, "8_guards_and_positive_control"),
])
def test_each_run_voiding_condition_can_actually_fail(table, over, cond):
    vc = D.void_condition_report(_vc_result(**over), table, "benchmark")
    assert vc[cond]["status"] == "fail"
    assert vc["verdict"]["run_void"] is True
    assert vc["verdict"]["shot_spent"] is False, "a void run does not spend it"


def test_a_non_benchmark_page_source_voids_the_run(table):
    vc = D.void_condition_report(_vc_result(), table, "prepass")
    assert vc["6_7_page_source"]["status"] == "fail"
    assert vc["verdict"]["run_void"] is True


def test_the_committed_confirmatory_artifact_carries_its_verdict():
    """The artifact an external reviewer will read must state the verdict, not
    imply it. Also pins the VC8 counter clarification."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / \
        "stage0_results/stage1/detector_gap_confirmatory_sh64k.json"
    if not p.exists():
        pytest.skip("confirmatory artifact not present")
    d = json.loads(p.read_text(encoding="utf-8"))
    r = d["results"][0]
    v = r["void_conditions"]["verdict"]
    assert v["run_void"] is False and v["shot_spent"] is True
    assert v["protective_claim_void"] is True
    pc = r["positive_control"]
    assert pc["n_fact_edits_applied"] == 200 and pc["n_editing_arms"] == 2
    assert pc["n_fact_edits_applied_per_arm"] == 100
    assert "ACCUMULATES" in pc["counter_note"]
    assert d["detector_vs_oracle"] == {}, \
        "no oracle arm exists at sh_64k; no ratio may be quoted for it"
    assert any("BACKWARDS" in i for c in d["corrections"] for i in c["items"])


# ── the pre-registration must be readable standalone, and stay true ──────────
def test_preregistration_banner_reports_the_fired_outcome():
    """A reviewer handed only the pre-registration must be able to tell that the
    campaign ran and how it came out — and the banner's figures must not drift
    from the artifact they describe. Prose provenance drifts; a pinned equality
    cannot."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    prereg = root / "stage0_results/stage1_preregistration_v2.md"
    art = root / "stage0_results/stage1/detector_gap_confirmatory_sh64k.json"
    if not art.exists():
        pytest.skip("confirmatory artifact not present")
    text = prereg.read_text(encoding="utf-8")
    head = text[:4000]

    assert "REGISTERED · FIRED · REPORTED" in head
    assert "THE CAMPAIGN RAN" in head
    assert "detector_gap_confirmatory_sh64k.json" in head
    assert "2026-08-15T22:00:02+03:00" in head
    assert "effective\n> but not yet safe" in head or "effective" in head

    r = json.loads(art.read_text(encoding="utf-8"))["results"][0]
    c = r["by_stratum"]["conflicted"]
    paired = c["paired_vs_native"]["detector_suppress"]
    v = r["void_conditions"]["verdict"]

    # the banner's headline numbers, each checked against the artifact
    assert f"net **+{paired['net']}**" in head
    assert (f"{c['arms']['native']['correct']}/66 → "
            f"**{c['arms']['detector_suppress']['correct']}/66**") in head
    assert "MET" in head and "VOID" in head
    assert v["run_void"] is False and v["shot_spent"] is True
    assert "NOT void" in head and "the shot IS spent" in head
    # the gold-cut miss is reported as a miss, not smoothed over
    assert "A MISS" in head
    # and the prohibition travels with the document
    assert "No oracle-ceiling ratio exists" in head
