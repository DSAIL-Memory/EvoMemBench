"""Stage-0 report assembly.  [T8]

The report is the artefact a human reads to make the gate decision, so the
things that must not go wrong are:

* a measurement that was not run must read as **NOT RUN**, never as a zero and
  never as a pass;
* an unmeasured criterion must never be tagged as a NO_GO (detection) — that
  would report "we did not measure it" as "evidence against H-Nav";
* the three NO_GO verdict types must come out distinct on inputs that are
  unambiguously each one.
"""
from __future__ import annotations

import json

import pytest

from hnav.stage0.report import (GATE, build_report, classify_no_go,
                                component_row, derive_components, verdict_short,
                                wilson)


# ── Wilson interval ──────────────────────────────────────────────────────────
def test_wilson_interval_known_values():
    lo, hi = wilson(0, 0)
    assert lo != lo and hi != hi          # NaN on an empty sample

    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.6 < lo < 1.0

    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    # the interval must be tighter with more data at the same proportion
    lo2, hi2 = wilson(50, 100)
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_is_not_the_normal_approximation_at_zero():
    """0/40 must give a non-zero upper bound; the normal approximation gives 0
    and would let a component pass the harm criterion on no evidence."""
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and hi > 0.0


# ── component gating ─────────────────────────────────────────────────────────
def test_component_passes_only_when_every_criterion_is_met():
    c = component_row("X", positives=200, coverage=0.5, precision=0.95,
                      harm=0.0, delta_pp=5.0)
    assert c["pass"] is True and verdict_short(c) == "GO"
    assert classify_no_go(c, None, None) == "GO"


def test_the_harm_criterion_implies_a_minimum_sample_size():
    """A consequence of the frozen gate worth knowing before the campaign: with
    ZERO observed harm, the 95% Wilson upper bound only drops below 0.03 at
    n ≈ 124. A component with 40 positives cannot clear the harm criterion no
    matter how clean it is — the gate is demanding evidence, not just absence."""
    assert component_row("X", positives=40, coverage=0.5, precision=1.0,
                         harm=0.0, delta_pp=5.0)["checks"]["harm_ub"] is False
    assert component_row("X", positives=130, coverage=0.5, precision=1.0,
                         harm=0.0, delta_pp=5.0)["checks"]["harm_ub"] is True


@pytest.mark.parametrize("field,value", [
    ("positives", GATE["min_positives"] - 1),
    ("coverage", GATE["min_coverage"] - 0.01),
    ("precision", GATE["min_precision"] - 0.01),
    ("delta_pp", GATE["min_delta_acc_pp"] - 0.5),
])
def test_each_criterion_can_fail_the_component(field, value):
    kw = dict(positives=100, coverage=0.5, precision=0.95, harm=0.0, delta_pp=5.0)
    kw[field] = value
    assert component_row("X", **kw)["pass"] is False


def test_verdict_benchmark_when_the_class_is_absent():
    c = component_row("A", positives=3, coverage=0.001, precision=1.0,
                      harm=0.0, delta_pp=10.0)
    assert verdict_short(c) == "NO_GO"
    assert "NO_GO (benchmark)" in classify_no_go(c, None, None)


def test_verdict_detection_when_the_signal_does_not_predict():
    c = component_row("A", positives=500, coverage=0.5, precision=0.40,
                      harm=0.10, delta_pp=0.0)
    assert "NO_GO (detection)" in classify_no_go(c, None, None)


def test_verdict_policy_when_detection_works_but_repair_does_not():
    c = component_row("A", positives=500, coverage=0.5, precision=0.99,
                      harm=0.0, delta_pp=0.5)
    assert "NO_GO (policy)" in classify_no_go(c, None, None)


def test_unmeasured_is_never_reported_as_a_failure():
    c = component_row("A", positives=500, coverage=0.5, precision=None,
                      harm=None, delta_pp=None)
    verdict = classify_no_go(c, None, None)
    assert "NOT ASSESSABLE" in verdict
    assert "detection" not in verdict and "policy" not in verdict
    assert verdict_short(c) == "NOT ASSESSABLE"


def test_a_measured_zero_is_not_an_unmeasured_value():
    """0 positives out of a real denominator is a benchmark finding."""
    c = component_row("A", positives=0, coverage=0.0, precision=None,
                      harm=None, delta_pp=None)
    assert "NO_GO (benchmark)" in classify_no_go(c, None, None)


# ── report assembly ──────────────────────────────────────────────────────────
def test_empty_out_dir_reports_everything_as_not_run(tmp_path):
    text, missing = build_report(tmp_path)
    assert set(missing) >= {"M1", "M1b", "M2", "M3", "M4"}
    assert "**NOT RUN.**" in text
    assert "INCOMPLETE" in text
    assert "0.000" not in text.split("## GO")[0].split("## M1")[1], \
        "an unrun measurement must not print zeros"
    assert "STOP." in text and "write/read policies" in text


def test_report_states_the_hard_stop_and_the_absent_policy_modules(tmp_path):
    text, _ = build_report(tmp_path)
    assert "write_policy.py" in text and "read_policy.py" in text
    assert "do not exist" in text


def test_gate_thresholds_match_the_frozen_protocol(tmp_path):
    text, _ = build_report(tmp_path)
    assert "positives ≥ 40" in text
    assert "coverage ≥ 10%" in text
    assert "precision ≥ 0.90" in text
    assert "Wilson LB ≥ 0.80" in text
    assert "harm 95% UB ≤ 0.03" in text
    assert "ΔAcc ≥ 3 pp" in text


def test_m1_gate_failure_is_rendered_loudly(tmp_path):
    (tmp_path / "m1_geometry_calibration.json").write_text(json.dumps([{
        "subset": "sh_6k", "n_conflict_pairs": 160,
        "whole_blob_sim": {"p50": 0.41}, "diff_sim": {"p50": 0.2},
        "control_whole_blob_sim": {"p50": 0.3},
        "separation_auc_conflict_vs_control": 0.7,
        "critical_delta_pct": 1.0, "gate_pass": False}]))
    text, _ = build_report(tmp_path)
    assert "S3 gate" in text and "**FAILED.**" in text
    assert "geometry half of the thesis is dead" in text


def test_derive_components_reads_only_the_calibration_split():
    m3 = [
        {"subset": "sh_6k", "write": {"n_write_decisions": 100,
                                      "would_intervene_after_vetoes_rate": 0.2},
         "read": {"n_reads": 50, "repair_helped": 5, "repair_harmed": 1,
                  "accuracy_native": 0.5, "accuracy_repaired": 0.6,
                  "strata": {"READ_RELEVANT_BELOW_K": {"n": 20}}}},
        {"subset": "sh_262k", "write": {"n_write_decisions": 9999,
                                        "would_intervene_after_vetoes_rate": 0.99},
         "read": {"n_reads": 100, "repair_helped": 99, "repair_harmed": 0,
                  "accuracy_native": 0.1, "accuracy_repaired": 0.9,
                  "strata": {"READ_RELEVANT_BELOW_K": {"n": 99}}}},
    ]
    comps = {c["component"].split(" —")[0]: c for c in derive_components(m3, None)}
    a3 = comps["A3"]
    assert a3["positives"] == 20, "the confirmatory subset must not enter the gate"
    assert a3["coverage"] == pytest.approx(20 / 50)
    assert a3["precision"] == pytest.approx(5 / 6)
    assert a3["delta_acc_pp"] == pytest.approx(10.0)


def test_h2_failure_blocks_a2():
    m3 = [{"subset": "sh_6k", "write": {"n_write_decisions": 100,
                                        "would_intervene_after_vetoes_rate": 0.2},
           "read": {"n_reads": 50, "strata": {}}}]
    m4 = {"n_positive": 500, "base_rate": 0.5, "h2_pass": False}
    a2 = [c for c in derive_components(m3, m4) if c["component"].startswith("A2")][0]
    assert a2["pass"] is False
    assert "did not pass" in a2["note"]
