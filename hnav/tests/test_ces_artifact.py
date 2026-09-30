"""CES artifact + the E2E identity-screen plumbing — oracle checks.

Synthetic artifacts throughout; the two tests that touch the committed
artifact/pipeline specs skip when those files are absent.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import hnav.stage1.detector_gap as D
from hnav.core.read_gate import StubNLI
from hnav.core.types import MemoryRecord
from hnav.geometry_filter.ces_artifact import CESArtifact

REPO = pathlib.Path(__file__).resolve().parents[2]
DIM = 16


def _artifact() -> CESArtifact:
    obj = np.zeros((DIM, 2)); obj[0, 0] = obj[1, 1] = 1.0     # span{e0,e1}
    subj = np.zeros((DIM, 2)); subj[4, 0] = subj[5, 1] = 1.0  # span{e4,e5}
    rel_obj = np.zeros((DIM, 1)); rel_obj[2, 0] = 1.0          # span{e2}
    rel_subj = np.zeros((DIM, 1)); rel_subj[6, 0] = 1.0        # span{e6}
    return CESArtifact(2, 5, obj, subj,
                       {"R": {"U_obj": rel_obj, "U_subj": rel_subj}})


def _rec(fid, vec, key):
    return MemoryRecord(id=fid, text=fid, vector=np.asarray(vec, float),
                        version=0, metadata={"key": key, "object": "x"})


# ── artifact semantics ───────────────────────────────────────────────────────
def test_score_pair_is_positive_for_object_edits_negative_for_subject_edits():
    art = _artifact()
    base = np.zeros(DIM); base[9] = 1.0
    e = np.eye(DIM)
    assert art.score_pair(base, base + e[0], None) > 0.9       # global obj
    assert art.score_pair(base, base + e[4], None) < -0.9      # global subj
    assert art.score_pair(base, base + e[2], "R") > 0.9        # relation obj
    assert art.score_pair(base, base + e[6], "R") < -0.9       # relation subj
    # unknown relation falls back to the GLOBAL subspaces
    assert art.score_pair(base, base + e[2], "UNSEEN") == pytest.approx(0.0, abs=1e-9)
    # sign-invariance and the no-edit case
    assert art.score_pair(base + e[0], base, None) == \
        pytest.approx(art.score_pair(base, base + e[0], None))
    assert art.score_pair(base, base, "R") == 0.0


def test_pair_filter_uses_relation_only_and_ignores_subject_identity():
    art = _artifact()
    base = np.zeros(DIM); base[9] = 1.0
    e = np.eye(DIM)
    f = art.pair_filter(0.0)
    # same relation, DIFFERENT subject in the key: relation half is still used
    a = _rec("a", base, ("R", "alice"))
    b_obj = _rec("b", base + e[2], ("R", "bob"))
    b_subj = _rec("c", base + e[6], ("R", "bob"))
    assert f(a, b_obj) is True
    assert f(a, b_subj) is False
    # unparsed key -> global subspaces
    assert f(_rec("d", base, None), _rec("e", base + e[0], None)) is True


def test_round_trip_is_bit_identical_and_a_tampered_manifest_refuses(tmp_path):
    art = _artifact()
    p = tmp_path / "ces.json"
    art.save(p, {"note": "synthetic"})
    art2, man = CESArtifact.load(p)
    assert art2.fingerprint() == art.fingerprint() == man["fingerprint"]
    assert np.array_equal(art2.relations["R"]["U_subj"],
                          art.relations["R"]["U_subj"])
    man["fingerprint"] = "0" * 64
    p.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CESArtifact.load(p)


# ── gate plumbing ────────────────────────────────────────────────────────────
def _cell(pair_filter, **kw):
    c = {"cos_pair": 0.0, "r_min": None, "ambiguity_mode": "none",
         "nli_contradiction": 0.9, "pair_filter": pair_filter}
    c.update(kw)
    return c


def test_make_gate_builds_the_right_identity_screen_per_cell():
    from hnav.adapters.mab_adapter import MABAdapter
    art = _artifact()
    assert D.make_gate(_cell(True), StubNLI()).pair_filter \
        is MABAdapter.same_key_pair
    assert D.make_gate(_cell(False), StubNLI()).pair_filter is None
    gate = D.make_gate(_cell("ces", ces_tau=0.0), StubNLI(), art)
    base = np.zeros(DIM); base[9] = 1.0
    e = np.eye(DIM)
    assert gate.pair_filter(_rec("a", base, ("R", "s1")),
                            _rec("b", base + e[2], ("R", "s2"))) is True
    with pytest.raises(SystemExit, match="'ces' screen"):
        D.make_gate(_cell("ces", ces_tau=0.0), StubNLI(), None)


def test_grid_cells_per_screen_and_the_backward_compatible_default():
    base = D.grid_cells()
    assert {c["pair_filter"] for c in base} == {True, False}
    assert all("ces_tau" not in c for c in base)
    none = D.grid_cells(pair_screen="none")
    assert {c["pair_filter"] for c in none} == {False}
    assert len(none) == len(base) // 2
    ces = D.grid_cells(cos_grid=[0.80], pair_screen="ces", ces_grid=[0.0, 0.1])
    assert {c["pair_filter"] for c in ces} == {"ces"}
    assert {c["ces_tau"] for c in ces} == {0.0, 0.1}
    assert all(c["cos_pair"] == 0.80 for c in ces)
    with pytest.raises(ValueError):
        D.grid_cells(pair_screen="bogus")


def test_select_respects_each_screen_and_keeps_the_harm_veto():
    def cell(pf, harmful, recall, tau=None):
        c = _cell(pf, r_min_label="loose", ces_tau=tau)
        c["metrics"] = {"n_suppressed_harmful": harmful,
                        "pair_recall_pool": recall}
        c["cos_pair"] = 0.8
        return c
    cells = [cell("ces", 0, 0.5, 0.0), cell("ces", 0, 0.9, 0.1),
             cell("ces", 1, 1.0, 0.05)]
    chosen = D.select(cells, "ces")
    assert chosen["ces_tau"] == 0.1, "harmful cell must lose despite recall 1.0"
    assert D.select([cell(True, 0, 0.9)], "ces") is None, \
        "a parser cell may never satisfy the ces screen"
    assert D.select([cell(False, 0, 0.9)], "none")["pair_filter"] is False
    # rule text records the arm-specific requirement
    assert any("n_suppressed_harmful == 0" in r
               for r in D.selection_rule_for("ces")["require"])
    assert D.selection_rule_for("parser") is D.SELECTION_RULE


def test_operating_point_paths_are_arm_specific_and_undefined_combos_refuse():
    assert D.operating_point_path("raw", "parser") == D.OPERATING_POINT \
        or D.operating_point_path("raw", "parser").name == "stage1_operating_point.json"
    assert D.operating_point_path("abtt", "parser").name == "abtt_operating_point.json"
    assert D.operating_point_path("raw", "ces").name == "ces_operating_point.json"
    assert D.operating_point_path("abtt", "none").name == \
        "abtt_noparser_operating_point.json"
    with pytest.raises(SystemExit):
        D.operating_point_path("raw", "none")
    with pytest.raises(SystemExit):
        D.operating_point_path("abtt", "ces")


def test_prepass_path_carries_the_tag():
    class Cfg:
        out_dir = pathlib.Path("x")
    a = D.prepass_path(Cfg, "sh_6k", "benchmark", "raw", "_ces")
    assert a.name == "stage1_prepass_sh_6k_benchmarkpage_ces.json"
    b = D.prepass_path(Cfg, "sh_6k", "benchmark", "raw")
    assert b.name == "stage1_prepass_sh_6k_benchmarkpage.json"


# ── fusion screen ────────────────────────────────────────────────────────────
def test_fusion_screen_score_is_the_documented_logistic_and_filter_thresholds():
    from hnav.geometry_filter.fusion_screen import FusionScreen
    ces = _artifact()
    wh_mean = np.zeros(DIM)
    wh_comp = np.eye(DIM)[10:11]          # remove e10 only
    mu, sd = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    w, b = np.array([2.0, 1.0]), -1.0
    f = FusionScreen(ces, wh_mean, wh_comp, mu, sd, w, b)
    base = np.zeros(DIM); base[9] = 1.0
    e = np.eye(DIM)
    va, vb = base, base + e[0]            # object edit in ces terms
    expect = 2.0 * ces.score_pair(va, vb, None) + 1.0 * float(
        f._whiten(va) @ f._whiten(vb)) - 1.0
    assert f.score_pair(va, vb, None) == pytest.approx(expect)
    # filter thresholds on the logit
    s = f.score_pair(va, vb, None)
    ra, rb = _rec("a", va, None), _rec("b", vb, None)
    assert f.pair_filter(s - 0.1)(ra, rb) is True
    assert f.pair_filter(s + 0.1)(ra, rb) is False
    # whitening inside the screen: a shared e10 component is removed
    va2, vb2 = va + 5 * e[10], vb + 5 * e[10]
    assert f.score_pair(va2, vb2, None) == pytest.approx(
        f.score_pair(va, vb, None), abs=1e-9)


def test_committed_fusion_screen_loads_and_matches_its_pins():
    from hnav.geometry_filter.fusion_screen import FUSION_JSON, FusionScreen
    if not FUSION_JSON.exists():
        pytest.skip("committed fusion screen not present")
    f, blob = FusionScreen.load(FUSION_JSON)
    assert blob["fingerprint"] == f.fingerprint()
    spec = json.loads((REPO / "pipelines" / "hnav_fusion" / "pipeline.json")
                      .read_text(encoding="utf-8"))
    assert spec["fusion_fingerprint"] == blob["fingerprint"]
    assert blob["provenance"]["fit_subsets"] == ["sh_6k", "sh_32k"]


def test_fusion_grid_and_operating_point_path():
    cells = D.grid_cells(cos_grid=[0.80], pair_screen="fusion",
                         ces_grid=[0.0, 4.0])
    assert {c["pair_filter"] for c in cells} == {"fusion"}
    assert {c["ces_tau"] for c in cells} == {0.0, 4.0}
    assert D.operating_point_path("raw", "fusion").name == \
        "fusion_operating_point.json"
    with pytest.raises(SystemExit):
        D.operating_point_path("abtt", "fusion")


# ── runner spec plumbing ─────────────────────────────────────────────────────
def test_new_pipeline_specs_wire_the_screen_through_the_runner():
    from pipelines._shared import runner
    ces = runner.load_pipeline(REPO / "pipelines" / "hnav_ces")
    nop = runner.load_pipeline(REPO / "pipelines" / "hnav_abtt_noparser")

    class Cfg:
        out_dir = pathlib.Path("o")
    assert runner.prepass_file(Cfg, "sh_6k", ces).name == \
        "stage1_prepass_sh_6k_benchmarkpage_ces.json"
    assert runner.prepass_file(Cfg, "sh_6k", nop).name == \
        "stage1_prepass_sh_6k_benchmarkpage_abtt.json"

    cmd = runner.detector_gap_cmd(ces, "sh_6k", pathlib.Path("x.json"), False, False)
    assert "--pair-screen" in cmd and "ces" in cmd
    assert "--ces-artifact" in cmd and "--prepass-tag" in cmd
    cmd = runner.detector_gap_cmd(nop, "sh_6k", pathlib.Path("x.json"), False, False)
    assert "--pair-screen" in cmd and "none" in cmd
    assert "--geometry-space" in cmd and "--ces-artifact" not in cmd

    # unchanged behaviour for the committed pipelines
    raw = runner.load_pipeline(REPO / "pipelines" / "hnav_raw")
    cmd = runner.detector_gap_cmd(raw, "sh_6k", pathlib.Path("x.json"), False, False)
    assert "--pair-screen" not in cmd
    assert runner.prepass_file(Cfg, "sh_6k", raw).name == \
        "stage1_prepass_sh_6k_benchmarkpage.json"


def test_committed_ces_artifact_matches_the_pipeline_pin():
    art_path = REPO / "stage0_results" / "geometry_filter" / "ces_subspaces_k20.json"
    if not art_path.exists():
        pytest.skip("committed CES artifact not present")
    art, man = CESArtifact.load(art_path)
    spec = json.loads((REPO / "pipelines" / "hnav_ces" / "pipeline.json")
                      .read_text(encoding="utf-8"))
    assert man["fingerprint"] == spec["ces_fingerprint"] == art.fingerprint()
    assert man["provenance"]["fit_subsets"] == ["sh_6k", "sh_32k"]
    assert art.k == 20 and art.dim == 2560
