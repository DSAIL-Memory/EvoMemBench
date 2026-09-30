"""Frozen constants in code == the committed Stage-0 artifacts. Exactly.  [T11]

Supervisor audit Note 2 (Faz A): a constant whose provenance is a docstring can
drift silently; one whose provenance is an equality test cannot. Every frozen
number the read gate ships is compared here against the committed measurement
JSON it claims to come from — ``==``, not ``approx``, because the constants
were COPIED from these files and a copy is either exact or wrong.

When the Faz B calibration freeze lands, this file also pins the Stage-1
operating point (``stage0_results/stage1_operating_point.json``) to the values
``hnav.core.read_policy.stage1_thresholds()`` actually runs with.
"""
from __future__ import annotations

import json
from pathlib import Path

from hnav.core import read_gate, read_policy

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "stage0_results/final"


def _load(name: str):
    return json.loads((FINAL / name).read_text(encoding="utf-8"))


def test_m3_thresholds_match_the_gate_constants_exactly():
    rows = _load("m3_headroom.json")
    assert rows, "m3_headroom.json is empty"
    for row in rows:
        thr = row["thresholds"]
        # Every subset row carries the SAME frozen threshold block …
        assert thr["fit_subsets"] == ["sh_6k", "sh_32k"]
        assert thr["unfit_for_analysis"] is False
        # … and the gate constants are byte-for-byte those values.
        assert read_gate.NMARGIN_CAL == thr["nmargin"]
        assert read_gate.H_Z_CAL == thr["H_z"]
        assert read_gate.R_MIN_CAL == thr["r_min"]


def test_cos_pair_is_the_rounded_mean_of_m1b_best_f1_taus_on_calibration():
    rows = {r["subset"]: r for r in _load("m1b_grouping_ablation.json")}
    taus = [rows["sh_6k"]["best_f1"]["tau"], rows["sh_32k"]["best_f1"]["tau"]]
    assert taus == [0.91, 0.93], "m1b calibration taus moved — re-derive COS_PAIR_CAL"
    assert read_gate.COS_PAIR_CAL == round(sum(taus) / 2, 2)


def test_gate_defaults_are_the_frozen_constants():
    thr = read_gate.GateThresholds()
    assert thr.nmargin == read_gate.NMARGIN_CAL
    assert thr.H_z == read_gate.H_Z_CAL
    assert thr.r_min == read_gate.R_MIN_CAL
    assert thr.cos_pair == read_gate.COS_PAIR_CAL


def test_stage1_operating_point_artifact_matches_the_live_thresholds():
    """Until the calibration freeze the artifact does not exist and the live
    thresholds must be the Faz A baseline. Once the artifact is committed the
    equality is mandatory — a frozen operating point the live path does not
    actually run at would invalidate the campaign."""
    art = REPO / "stage0_results/stage1_operating_point.json"
    thr = read_policy.stage1_thresholds()
    if not art.exists():
        assert thr == read_gate.GateThresholds(), (
            "stage1_thresholds() departed from the Faz A baseline without a "
            "committed stage0_results/stage1_operating_point.json")
        return
    op = json.loads(art.read_text(encoding="utf-8"))["thresholds"]
    assert thr.cos_pair == op["cos_pair"]
    assert thr.r_min == op["r_min"]
    assert thr.nmargin == op["nmargin"]
    assert thr.H_z == op["H_z"]
    assert thr.nli_contradiction == op["nli_contradiction"]
    assert thr.ambiguity_mode == op["ambiguity_mode"]


def test_stage1_operating_point_was_fit_on_the_calibration_split_only():
    """The one invariant a threshold artifact can violate silently.  [T13]"""
    art = REPO / "stage0_results/stage1_operating_point.json"
    if not art.exists():
        return
    prov = json.loads(art.read_text(encoding="utf-8"))["provenance"]
    assert sorted(prov["fit_subsets"]) == ["sh_32k", "sh_6k"]
    assert sorted(prov["confirmatory_refused"]) == ["sh_262k", "sh_64k"]


def test_disabling_the_ambiguity_screen_is_declared_in_the_artifact():
    """``ambiguity_mode="none"`` switches off the frozen Stage-0 nmargin/H_z
    precondition. That is a legitimate, argued choice (T13) — but it must never
    be a silent one, so the artifact has to carry the argument, and the frozen
    constants have to stay recorded even while the screen is off."""
    art = REPO / "stage0_results/stage1_operating_point.json"
    thr = read_policy.stage1_thresholds()
    assert thr.nmargin is not None and thr.H_z is not None, (
        "the frozen ambiguity constants must stay on the record even when the "
        "screen is off — otherwise the departure stops being auditable")
    if thr.ambiguity_mode != "none":
        return
    assert art.exists(), (
        "the live gate disables the ambiguity screen but no operating-point "
        "artifact explains why")
    payload = json.loads(art.read_text(encoding="utf-8"))
    assert payload.get("ambiguity_note"), "ambiguity_note missing"
    assert payload["pair_filter"] is True, (
        "with the ambiguity screen off, the identity screen is the only thing "
        "bounding intervention volume; the artifact must record it as required")


# ── T12/T13 re-fit: the superseded-but-retained arrangement ──────────────────
# Coordinator decision, 2026-08-16 (option 3 of
# stage0_results/refit_threshold_diff.md §5): the L512-era constants STAY as
# the live values so detector_gap.py's committed, pre-registered confirmatory
# artifacts remain reproducible from the code that produced them; the corrected
# per-subset values are recorded ALONGSIDE them. These tests exist so that
# arrangement cannot decay in either direction — a silent swap of the live
# values, or drift between the recorded per-subset values and the artifact.

REFIT = REPO / "stage0_results/refit_L8192"
GATE_SRC = REPO / "hnav/core/read_gate.py"


def test_live_constants_are_still_the_L512_values():
    """Swapping the live values (e.g. to the new pooled scalars) must fail
    here — it would strand detector_gap.py's committed artifacts."""
    pooled_new = json.loads((REFIT / "per_subset_thresholds.json")
                            .read_text(encoding="utf-8"))["pooled"]
    assert read_gate.NMARGIN_CAL == 0.004764391389658354
    assert read_gate.H_Z_CAL == 1.9569327964981853
    assert read_gate.NMARGIN_CAL != pooled_new["nmargin"], (
        "live NMARGIN_CAL was swapped to the L8192 pooled value; that breaks "
        "reproducibility of the L512-era artifacts (refit_threshold_diff.md §5)")
    assert read_gate.H_Z_CAL != pooled_new["H_z"], "live H_Z_CAL was swapped"


def test_the_superseded_annotation_is_present():
    """The values may only stay because they are LABELLED superseded. If
    someone deletes the annotation, the constants silently become 'current'
    again — so the label is part of the contract, not decoration."""
    src = GATE_SRC.read_text(encoding="utf-8")
    for marker in ("SUPERSEDED FOR ANALYSIS", "RETAINED FOR REPRODUCIBILITY",
                   "refit_threshold_diff.md", "REFIT_PER_SUBSET",
                   "detector_gap.py"):
        assert marker in src, f"the superseded annotation lost its {marker!r} marker"


def test_recorded_per_subset_values_match_the_refit_artifact_exactly():
    """REFIT_PER_SUBSET in code == stage0_results/refit_L8192/. Exact equality,
    same reasoning as the original Note-2 test: these were copied."""
    art = json.loads((REFIT / "per_subset_thresholds.json").read_text(encoding="utf-8"))
    per = art["per_subset"]
    assert set(read_gate.REFIT_PER_SUBSET) == {"sh_6k", "sh_32k"} == set(per)
    for subset, rec in read_gate.REFIT_PER_SUBSET.items():
        for key in ("nmargin_p25", "H_z_p75", "r_min_p10"):
            assert rec[key] == per[subset][key], (
                f"{subset}.{key} drifted from the re-fit artifact: "
                f"code {rec[key]!r} vs artifact {per[subset][key]!r}")
    assert art["percentiles"] == {"nmargin": 25, "H_z": 75, "r_min": 10}
    assert art["namespace"].endswith("|L8192"), "artifact is not the corrected namespace"


def test_the_two_eras_are_distinct_records_not_copies():
    """A guard against someone 'tidying' by pointing both at one number."""
    assert read_gate.REFIT_PER_SUBSET["sh_32k"]["nmargin_p25"] != read_gate.NMARGIN_CAL
    assert read_gate.REFIT_PER_SUBSET["sh_32k"]["H_z_p75"] != read_gate.H_Z_CAL


def test_r_min_is_the_uncontaminated_threshold():
    """R_MIN_CAL is fit from write rows (facts, far below 512 tokens), so the
    truncation never touched it. The re-fit measured 1.1e-06 RELATIVE movement.
    This stability is what licenses treating fact-level Stage-0 quantities as
    unaffected, so it is pinned rather than remembered."""
    art = json.loads((REFIT / "per_subset_thresholds.json").read_text(encoding="utf-8"))
    rel = abs(art["pooled"]["r_min"] - read_gate.R_MIN_CAL) / read_gate.R_MIN_CAL
    assert rel < 1e-5, (
        f"R_MIN_CAL moved by {rel:.2e} relative under the corrected embedder. "
        f"That falsifies the 'fact-level signals are safe' premise the whole "
        f"T12 triage rests on — ESCALATE before re-deriving anything else.")


def test_H_z_is_structurally_unreachable_on_sh_6k():
    """A property of the statistic, not a tuning accident: n_chunks = 2 makes
    z-scoring yield {+1, -1} for ANY two scores, so H_z is a constant and the
    ln(2) ceiling sits far below every threshold ever used."""
    import math

    art = json.loads((REFIT / "per_subset_thresholds.json").read_text(encoding="utf-8"))
    sh6k = art["per_subset"]["sh_6k"]
    c = read_gate.H_Z_SH6K_DEGENERATE_VALUE
    # the artifact confirms it is a point mass: min == max == p75 == the constant
    assert sh6k["H_z_min"] == sh6k["H_z_max"] == sh6k["H_z_p75"] == c
    # and no threshold ever applied can select from a constant
    assert c < math.log(2), "the ln(2) entropy ceiling argument no longer holds"
    for threshold in (read_gate.H_Z_CAL, art["pooled"]["H_z"], sh6k["H_z_p75"]):
        assert not (c > threshold), (
            f"H_z {c!r} would fire against threshold {threshold!r}; the "
            f"vacuity finding needs revisiting")
