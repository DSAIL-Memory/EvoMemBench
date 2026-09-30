"""Second-reader audit (Laya) — oracle checks with no torch and no laya.

Every statistic is compared to a hand-computed value; the runner, the
comparison and the human-review sheets are exercised end to end on a
synthetic candidate universe written into ``tmp_path``.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hnav.labeling import audit_compare_laya as cmp  # noqa: E402
from hnav.labeling import audit_human_review as hr  # noqa: E402
from hnav.labeling import audit_laya as runner  # noqa: E402
from hnav.labeling.agreement_stats import (  # noqa: E402
    agreement, cohen_kappa, largest_remainder, rank_auc, rogan_gladen, wilson)


# ── closed-form statistics ───────────────────────────────────────────────────

def test_kappa_matches_a_hand_computed_table():
    # TT=20, TF=5, FT=10, FF=65: po=0.85, pe=0.25*0.30+0.75*0.70=0.60, kappa=0.625
    x = [True] * 25 + [False] * 75
    y = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 65
    assert math.isclose(cohen_kappa(x, y), 0.625, abs_tol=1e-12)
    assert math.isclose(agreement(x, y), 0.85, abs_tol=1e-12)


def test_kappa_is_zero_under_independence_and_one_under_identity():
    # rows 10/90, cols 60/40: TT=6, TF=4, FT=54, FF=36 -> po = pe = 0.42
    x = [True] * 10 + [False] * 90
    y = [True] * 6 + [False] * 4 + [True] * 54 + [False] * 36
    assert math.isclose(cohen_kappa(x, y), 0.0, abs_tol=1e-12)
    z = [True, False, True, True, False]
    assert math.isclose(cohen_kappa(z, z), 1.0, abs_tol=1e-12)
    assert cohen_kappa([True, True], [True, True]) is None      # undefined


def test_wilson_matches_the_closed_form_endpoints():
    z = 1.959963984540054
    lo, hi = wilson(0, 10)
    assert math.isclose(lo, 0.0, abs_tol=1e-12) and math.isclose(hi, z * z / (10 + z * z), abs_tol=1e-12)
    lo, hi = wilson(10, 10)
    assert math.isclose(hi, 1.0, abs_tol=1e-12) and math.isclose(lo, 10 / (10 + z * z), abs_tol=1e-12)
    assert wilson(0, 0) is None


def test_rank_auc_counts_pairs_and_halves_ties():
    assert math.isclose(rank_auc([0.9, 0.8], [0.7, 0.85]), 0.75)
    assert math.isclose(rank_auc([0.5], [0.5]), 0.5)
    assert rank_auc([], [0.1]) is None


def test_rogan_gladen_and_apportionment():
    assert math.isclose(rogan_gladen(0.3, 0.9, 0.1), 0.25)
    assert rogan_gladen(0.3, 0.1, 0.1) is None
    assert largest_remainder([2, 2, 1], 2) == [1, 1, 0]
    assert largest_remainder([1, 1, 1], 2) == [1, 1, 0]
    assert sum(largest_remainder([51782, 2388, 282, 12], 100)) == 100


# ── the runner's pure parts ──────────────────────────────────────────────────

def test_questions_mirror_the_judge_fields_with_neutral_binary_options():
    assert list(runner.QUESTIONS) == runner.FLAGS
    for q in runner.QUESTIONS.values():
        assert q["type"] == "choice"
        assert set(q["criteria"]) == {"A", "B"}
    assert runner.TRUE_KEY == "A"


def test_state_order_and_answer_parsing():
    rec = {"fact_a": "X is a.", "fact_b": "X is b."}
    assert runner.build_state(rec, "ab") == "Sentence A: X is a.\nSentence B: X is b."
    assert runner.build_state(rec, "ba") == "Sentence A: X is b.\nSentence B: X is a."
    answers = {q: {"choice": "B", "probabilities": {"A": 0.2, "B": 0.8}, "confidence": 0.6}
               for q in runner.FLAGS}
    answers["update_conflict"] = {"choice": "A", "probabilities": {"A": 0.7, "B": 0.3}}
    answers["strict_conflict"] = {"probabilities": {"A": 0.55, "B": 0.45}}   # no choice field
    out, verdict = runner.parse_answers(answers)
    assert verdict["update_conflict"] is True and out["update_conflict"]["p_true"] == 0.7
    assert verdict["strict_conflict"] is True                                 # from p >= 0.5
    assert verdict["same_referent"] is False


def test_work_items_skip_completed_pair_orders():
    recs = [{"pair_id": "p1"}, {"pair_id": "p2"}]
    items = runner.work_items(recs, ("ab", "ba"), done={("p1", "ab")})
    assert [(r["pair_id"], o) for r, o in items] == [("p1", "ba"), ("p2", "ab"), ("p2", "ba")]


# ── synthetic universe ───────────────────────────────────────────────────────

def _cand(pid, subset, a, b, cos, tagged, same_subject=True, same_relation=True,
          same_object=False, same_key=None, both_parse=True):
    if same_key is None:
        same_key = same_subject and same_relation
    return {"pair_id": pid, "subset": subset, "fact_a_id": "fact:0", "fact_b_id": "fact:1",
            "fact_a": a, "fact_b": b, "cosine_similarity": cos, "parser_tagged_conflict": tagged,
            "parser_metadata": {"fact_a_parsed": {}, "fact_b_parsed": {}, "both_parse": both_parse,
                                "same_key": same_key, "same_relation": same_relation,
                                "same_subject": same_subject, "same_object": same_object,
                                "superseding_serial": None},
            "llm_audit_status": "pending"}


def _judge(update, strict=None, same_referent=True, same_relation=True, context=True, reason="direct_replacement"):
    strict = update if strict is None else strict
    return {"same_referent": same_referent, "same_relation": same_relation, "context_overlap": context,
            "values_incompatible": update, "relation_allows_multiple_values": False,
            "strict_conflict": strict, "update_conflict": update, "reason_code": reason, "explanation": ""}


def _gold(c, tier, judge, slice_="tagged", in_eval=True):
    return {"pair_id": c["pair_id"], "subset": c["subset"], "split": "calibration", "tier": tier,
            "in_eval_set": in_eval, "gold_update": tier in ("core", "update_only_fork"),
            "gold_strict": tier == "core" and judge["strict_conflict"],
            "disputed_by_judge": tier == "update_only_fork", "fact_a": c["fact_a"], "fact_b": c["fact_b"],
            "fact_a_id": "fact:0", "fact_b_id": "fact:1", "cosine_similarity": c["cosine_similarity"],
            "parser": c["parser_metadata"], "judge": judge,
            "provenance": {"audit_slice": slice_, "parser_miss_channel": None,
                           "cosine_bin_fallback": None, "judge_model": "openai/gpt-5-mini"}}


def _laya_row(pid, subset, tagged, order, p_update, p_other=None):
    p_other = p_update if p_other is None else p_other
    answers, verdict = {}, {}
    for q in runner.FLAGS:
        p = p_update if q == "update_conflict" else p_other
        answers[q] = {"choice": "A" if p >= 0.5 else "B", "p_true": p, "confidence": abs(2 * p - 1),
                      "answer_confidence": None}
        verdict[q] = p >= 0.5
    return {"pair_id": pid, "subset": subset, "parser_tagged_conflict": tagged, "order": order,
            "answers": answers, "verdict": verdict,
            "meta": {"reader": "laya", "laya_version": "0.0-test", "checkpoint": "test", "snapshot_sha": None,
                     "device": "cpu", "questions_sha256": "x"}, "ts": 0.0}


@pytest.fixture
def universe(tmp_path):
    """10 audited pairs (3 core, 1 fork, 1 rejected, 1 discovered, 4 negative) + 4 unjudged."""
    C = [
        _cand("sh_6k:1-2", "sh_6k", "A born in X.", "A born in Y.", 0.99, True),        # P1 core, agree
        _cand("sh_6k:3-4", "sh_6k", "B born in X.", "B born in Y.", 0.98, True),        # P2 core, agree (flip)
        _cand("sh_6k:5-6", "sh_6k", "C born in X.", "C born in Y.", 0.97, True),        # P3 core, Laya says no
        _cand("sh_6k:7-8", "sh_6k", "D speaks X.", "D speaks Y.", 0.96, True),          # P4 fork, agree (no)
        _cand("sh_6k:9-10", "sh_6k", "E born in X.", "E born in X City.", 0.95, True),  # P5 rejected, Laya yes
        _cand("sh_6k:11-12", "sh_6k", "Author of F is G.", "Author of H is G.", 0.86, False,
              same_subject=False, same_object=True),                                    # P6 discovered, agree
        _cand("sh_6k:13-14", "sh_6k", "I born in X.", "I citizen of Z.", 0.84, False,
              same_relation=False, same_object=False),                                  # P7 negative, agree
        _cand("sh_6k:15-16", "sh_6k", "J born in X.", "K born in X.", 0.83, False,
              same_subject=False, same_object=True),                                    # P8 negative, Laya yes
        _cand("sh_6k:17-18", "sh_6k", "L born in X.", "M born in Y.", 0.82, False,
              same_subject=False, same_object=False),                                   # P9 negative, Laya yes
        _cand("sh_6k:19-20", "sh_6k", "N born in X.", "O born in Y.", 0.81, False,
              same_subject=False, same_object=False),                                   # P10 negative, agree
        _cand("sh_6k:21-22", "sh_6k", "T1 a.", "T1 b.", 0.80, False, same_subject=False),   # tail
        _cand("sh_6k:23-24", "sh_6k", "T2 a.", "T2 b.", 0.80, False, same_subject=False),
        _cand("sh_6k:25-26", "sh_6k", "T3 a.", "T3 b.", 0.80, False, same_subject=False),
        _cand("sh_6k:27-28", "sh_6k", "T4 a.", "T4 b.", 0.80, False, same_subject=False),
    ]
    P = {f"P{i+1}": C[i] for i in range(10)}
    G = [
        _gold(P["P1"], "core", _judge(True)),
        _gold(P["P2"], "core", _judge(True)),
        _gold(P["P3"], "core", _judge(True)),
        _gold(P["P4"], "update_only_fork", _judge(False, reason="multi_valued_relation")),
        _gold(P["P5"], "rejected", _judge(False, same_relation=False, reason="different_relation")),
        _gold(P["P6"], "discovered_unverified", _judge(True), slice_="bulk", in_eval=False),
        _gold(P["P7"], "negative", _judge(False, same_relation=False), slice_="bulk"),
        _gold(P["P8"], "negative", _judge(False, same_referent=False), slice_="bulk"),
        _gold(P["P9"], "negative", _judge(False, same_referent=False), slice_="bulk"),
        _gold(P["P10"], "negative", _judge(False, same_referent=False), slice_="bulk"),
    ]
    p_update = {"P1": 0.95, "P2": None, "P3": 0.3, "P4": 0.2, "P5": 0.8, "P6": 0.9,
                "P7": 0.1, "P8": 0.6, "P9": 0.75, "P10": 0.05}
    rows = []
    for name, c in P.items():
        if name == "P2":                              # orders disagree: 0.9 / 0.4 -> mean 0.65
            rows.append(_laya_row(c["pair_id"], c["subset"], True, "ab", 0.9))
            rows.append(_laya_row(c["pair_id"], c["subset"], True, "ba", 0.4))
        else:
            for o in ("ab", "ba"):
                rows.append(_laya_row(c["pair_id"], c["subset"], c["parser_tagged_conflict"], o, p_update[name]))
    for c, p in zip(C[10:], (0.95, 0.2, 0.7, 0.1)):
        for o in ("ab", "ba"):
            rows.append(_laya_row(c["pair_id"], c["subset"], False, o, p))

    d = tmp_path
    (d / "cand.jsonl").write_text("".join(json.dumps(c) + "\n" for c in C))
    (d / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in G))
    (d / "laya.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps(
        {"per_subset": {"sh_6k": {"eval": {"cosine_only_auc": 0.9598}}}}))
    return d, C, G, rows


def test_runner_dry_run_needs_no_model(universe):
    d, C, _, _ = universe
    proc = subprocess.run([sys.executable, str(REPO / "hnav/labeling/audit_laya.py"), "--dry-run",
                           "--candidates", str(d / "cand.jsonl"), "--out", str(d / "none.jsonl"),
                           "--limit", "5"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "to score=10" in proc.stdout and "Sentence A:" in proc.stdout
    assert "laya" not in proc.stderr.lower()


def test_aggregate_averages_orders_and_flags_flips(universe):
    d, _, _, rows = universe
    agg = cmp.aggregate(rows)
    p2 = agg["sh_6k:3-4"]["q"]["update_conflict"]
    assert math.isclose(p2["p"], 0.65) and p2["flag"] is True and p2["flip"] is True
    assert math.isclose(p2["conf"], 0.3)
    p1 = agg["sh_6k:1-2"]["q"]["update_conflict"]
    assert p1["flip"] is False and p1["flag"] is True


def test_compare_report_matches_hand_values(universe):
    d, C, G, rows = universe
    rep, dis, tail = cmp.compute(cmp.aggregate(rows), G, C, {"sh_6k": 0.9598})
    o = rep["update_conflict"]["overall"]
    # judge T on P1,P2,P3,P6; Laya T on P1,P2,P5,P6,P8,P9 -> TT=3 TF=1 FT=3 FF=3
    assert o["table_judge_laya"] == {"TT": 3, "TF": 1, "FT": 3, "FF": 3}
    assert math.isclose(o["agreement"], 0.6)
    assert math.isclose(o["kappa"], (0.6 - 0.48) / 0.52, abs_tol=1e-12)
    assert rep["update_conflict"]["by_tier"]["core"]["table_judge_laya"] == {"TT": 2, "TF": 1, "FT": 0, "FF": 0}
    assert [x["pair_id"] for x in dis] == ["sh_6k:9-10", "sh_6k:5-6", "sh_6k:17-18", "sh_6k:15-16"]
    assert [x["priority"] for x in dis] == [0, 1, 2, 3]
    assert rep["disagreements"]["by_direction"] == {"laya_conflict_judge_not": 3, "judge_conflict_laya_not": 1}
    # order flips: only P2, out of 14 pairs scored in both orders
    assert math.isclose(rep["order_flip_rate"]["update_conflict"], 1 / 14)
    # tail ranking by p_update
    assert [t["pair_id"] for t in tail] == ["sh_6k:21-22", "sh_6k:25-26", "sh_6k:23-24", "sh_6k:27-28"]
    t = rep["tail"]
    assert t["n_unjudged_candidates"] == 4 and t["laya_positive_count"] == 2
    br = t["bulk_reference"]
    assert br["n_bulk_audited"] == 5 and br["n_judge_positive"] == 1
    assert math.isclose(br["laya_sensitivity_vs_judge"], 1.0) and math.isclose(br["laya_fpr_vs_judge"], 0.5)
    assert math.isclose(t["rogan_gladen_prevalence"], 0.0)
    assert math.isclose(t["expected_conflicts_in_tail_from_judge_bulk_rate"], 0.8)
    # eval-set competence: pos P1..P4 (gold_update) vs neg P5,P7..P10 among in_eval
    ev = rep["eval_set_auroc"]["sh_6k"]
    assert ev["n_pos"] == 4 and ev["n_neg"] == 5 and ev["cosine_only_auc"] == 0.9598
    assert 0.0 <= ev["laya_p_update_auroc"] <= 1.0
    assert rep["sanity"]["parser_tagged_same_relation_rate"]["n"] == 5
    md = cmp.render_md(rep)
    assert "## update_conflict" in md and "Unjudged tail" in md


def test_compare_cli_writes_all_outputs(universe):
    d, *_ = universe
    rc = cmp.main(["--laya", str(d / "laya.jsonl"), "--gold", str(d / "gold.jsonl"),
                   "--gold-summary", str(d / "summary.json"), "--candidates", str(d / "cand.jsonl"),
                   "--out-json", str(d / "rep.json"), "--out-md", str(d / "rep.md"),
                   "--out-disagreements", str(d / "dis.jsonl"), "--out-tail", str(d / "tail.jsonl")])
    assert rc == 0
    assert json.loads((d / "rep.json").read_text())["disagreements"]["n"] == 4
    assert sum(1 for _ in open(d / "dis.jsonl")) == 4 and sum(1 for _ in open(d / "tail.jsonl")) == 4


def _build(d):
    return hr.main(["build", "--gold", str(d / "gold.jsonl"), "--laya", str(d / "laya.jsonl"),
                    "--disagreements", str(d / "dis.jsonl"), "--sheet-a", str(d / "A.csv"),
                    "--sheet-b", str(d / "B.csv"), "--key", str(d / "key.json"),
                    "--readme", str(d / "README.md"), "--n-negative", "1", "--n-control", "2"])


def test_review_sheets_are_blind_identical_in_content_and_differently_ordered(universe):
    d, *_ = universe
    test_compare_cli_writes_all_outputs(universe)
    assert _build(d) == 0
    key = json.loads((d / "key.json").read_text())
    # strata: P3,P5 positive; P6 discovered; 1 of {P8,P9}; 2 controls (core 1, negative 1)
    assert dict(key["by_stratum"]) == {"positive_disagreement": 2, "discovered": 1,
                                       "negative_disagreement": 1, "agreement_control": 2}
    assert dict(key["by_stratum_tier"])["agreement_control/core"] == 1
    assert dict(key["by_stratum_tier"])["agreement_control/negative"] == 1
    with open(d / "A.csv", newline="") as f:
        A = list(csv.DictReader(f))
    with open(d / "B.csv", newline="") as f:
        B = list(csv.DictReader(f))
    assert A[0].keys() == B[0].keys() == set(hr.SHEET_COLS)
    assert not (set(A[0]) & hr.FORBIDDEN_COLS)
    assert {r["review_id"] for r in A} == {r["review_id"] for r in B} == set(key["rows"])
    assert [r["review_id"] for r in A] != [r["review_id"] for r in B]
    assert all(r[c] == "" for r in A for c in hr.LABEL_COLS)
    for r in A:                                       # round trip through the key
        pid = key["rows"][r["review_id"]]["pair_id"]
        assert r["fact_a"] and pid.startswith("sh_6k:")
    assert "yes" in (d / "README.md").read_text()


def _fill(src, dst, labels: dict[str, dict]):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in hr.LABEL_COLS:
            r[c] = "yes" if labels[r["review_id"]][c] else "no"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hr.SHEET_COLS)
        w.writeheader()
        w.writerows(rows)


def test_ingest_rejects_blanks_then_scores_against_human_labels(universe):
    d, C, G, rows = universe
    test_compare_cli_writes_all_outputs(universe)
    assert _build(d) == 0
    key = json.loads((d / "key.json").read_text())["rows"]
    gold = {g["pair_id"]: g for g in G}
    # "truth": the judge is right everywhere except on the rejected pair P5 (humans: conflict)
    truth = {}
    for rid, k in key.items():
        j = gold[k["pair_id"]]["judge"]
        t = {c: j[c] for c in hr.LABEL_COLS}
        if k["pair_id"] == "sh_6k:9-10":
            t = {c: True for c in hr.LABEL_COLS}
        truth[rid] = t
    # annotator B differs from A on update_conflict for one discovered pair
    disc = next(rid for rid, k in key.items() if k["stratum"] == "discovered")
    truth_b = {rid: dict(v) for rid, v in truth.items()}
    truth_b[disc]["update_conflict"] = not truth_b[disc]["update_conflict"]
    _fill(d / "A.csv", d / "A.filled.csv", truth)
    _fill(d / "B.csv", d / "B.filled.csv", truth_b)

    # a blank cell is refused
    bad = (d / "B.filled.csv").read_text().replace("yes", "", 1)
    (d / "B.bad.csv").write_text(bad)
    with pytest.raises(SystemExit):
        hr.main(["ingest", "--gold", str(d / "gold.jsonl"), "--laya", str(d / "laya.jsonl"),
                 "--key", str(d / "key.json"), "--sheet-a", str(d / "A.filled.csv"),
                 "--sheet-b", str(d / "B.bad.csv"), "--adjudication-out", str(d / "adj.csv"),
                 "--out-json", str(d / "h.json"), "--out-md", str(d / "h.md")])

    common = ["ingest", "--gold", str(d / "gold.jsonl"), "--laya", str(d / "laya.jsonl"),
              "--key", str(d / "key.json"), "--sheet-a", str(d / "A.filled.csv"),
              "--sheet-b", str(d / "B.filled.csv"), "--adjudication-out", str(d / "adj.csv"),
              "--out-json", str(d / "h.json"), "--out-md", str(d / "h.md")]
    assert hr.main(common) == 0
    rep = json.loads((d / "h.json").read_text())
    assert rep["n_unresolved"] == 1
    hh = rep["human_human"]["update_conflict"]
    assert hh["n"] == 6 and math.isclose(hh["agreement"], 5 / 6)
    with open(d / "adj.csv", newline="") as f:
        adj = list(csv.DictReader(f))
    assert [r["review_id"] for r in adj] == [disc] and adj[0]["update_conflict"] == ""
    # joint adjudication sides with annotator A
    for r in adj:
        for c in hr.LABEL_COLS:
            r[c] = "yes" if truth[r["review_id"]][c] else "no"
    with open(d / "adj.filled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(adj[0].keys()))
        w.writeheader()
        w.writerows(adj)
    assert hr.main(common + ["--adjudicated", str(d / "adj.filled.csv")]) == 0
    rep = json.loads((d / "h.json").read_text())
    assert rep["n_unresolved"] == 0
    j_all = rep["judge_vs_human"]["update_conflict"]["all"]
    assert j_all["n"] == 6 and j_all["correct"] == 5                 # wrong only on P5
    assert math.isclose(j_all["accuracy"], 5 / 6)
    lo, hi = j_all["wilson95"]
    assert 0 < lo < 5 / 6 < hi <= 1
    assert rep["judge_vs_human"]["update_conflict"]["positive_disagreement"]["correct"] == 1
    est = rep["judge_update_accuracy_by_tier_estimate"]
    assert est["rejected"]["estimated_judge_accuracy_on_tier"] == 0.0
    assert est["core"]["n_disagree"] == 1 and est["core"]["reviewed_disagree"] == 1
    assert "Human–human agreement" in (d / "h.md").read_text()
