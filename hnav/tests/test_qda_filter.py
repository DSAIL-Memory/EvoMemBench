"""QDA conflict scorer — machinery checks and regression pins.  [QDA]

Three levels, matching the suite's standard:

1. **Closed-form / synthetic.** Rank selection, the conformal order
   statistic, the sign-flip null, the quadratic core, and the swap behaviour
   of the ordered term are each checked against answers known by
   construction, on synthetic data that needs no embedding cache.
2. **Committed-artifact consistency.** The weights manifest, spectrum, fit
   and calibration JSONs under ``stage0_results/qda_filter/`` must agree
   with each other (k_obj in the manifest == k_obj in spectrum.json, the
   gates' rules recomputed from their own recorded inputs, ...).
3. **Regression pins.** The headline numbers of the campaign, pinned the way
   ``test_question_strata.py`` pins the strata counts. Re-running
   ``run_all`` must reproduce them exactly (same seeds) — drift means the
   pipeline changed, not the world.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from hnav.qda_filter.calibrate import conformal_thresholds
from hnav.qda_filter.fit import QDAModel, signflip_test
from hnav.qda_filter.spectrum import nontrivial_spectrum, select_ranks

OUT = pathlib.Path(__file__).resolve().parents[2] / "stage0_results" / "qda_filter"

RNG = np.random.default_rng(11)


def _load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        pytest.skip(f"{name} not committed yet")
    return json.loads(p.read_text(encoding="utf-8"))


# ── 1. closed-form / synthetic ───────────────────────────────────────────────
def test_select_ranks_walks_until_first_failure_and_respects_caps():
    lam = np.array([10.0, 5.0, 2.0, 1.0, 1.0, 0.5, 0.1])
    top = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    bot = np.array([0.2, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9])
    # top: 10,5 exceed 3; 2 does not -> k_obj = 2
    # bottom (ascending 0.1, 0.5, 1.0, ...): 0.1<0.2, 0.5<0.6, 1.0>=0.9 -> 2
    k_obj, k_subj = select_ranks(lam, top, bot)
    assert (k_obj, k_subj) == (2, 2)
    assert select_ranks(lam, top, bot, cap_obj=1) == (1, 2)
    assert select_ranks(lam, top, bot, cap_subj=1)[1] == 1


def test_nontrivial_spectrum_matches_np_cov_eigenvalues():
    X = RNG.normal(size=(40, 7))
    lam, vt = nontrivial_spectrum(X)
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert lam.shape == (7,) and vt.shape == (7, 7)
    np.testing.assert_allclose(lam, ev[:7], atol=1e-10)
    # rank-deficient case: n-1 < dim
    lam2, _ = nontrivial_spectrum(RNG.normal(size=(5, 9)))
    assert lam2.shape == (4,)


def test_conformal_threshold_is_the_exact_order_statistic():
    scores = np.arange(99, dtype=float)          # 0..98, n=99
    out = conformal_thresholds(scores, alphas=(0.1, 1e-3, None))
    t = out["thresholds"]
    # k = ceil(100*0.9) = 90 -> 90th order statistic = 89.0
    assert t["0.1"]["threshold"] == 89.0 and t["0.1"]["achievable"]
    assert not t["0.001"]["achievable"] and t["0.001"]["threshold"] is None
    # floor alpha = 1/100 -> k = ceil(100*0.99) = 99 -> max score
    assert t["alpha_floor"]["threshold"] == 98.0
    assert out["alpha_floor"] == pytest.approx(0.01)


def test_signflip_null_separates_ordered_from_symmetric_data():
    informative = RNG.normal(loc=0.5, scale=1.0, size=(300, 6))
    signs = RNG.integers(0, 2, size=300) * 2 - 1
    symmetric = informative * signs[:, None]
    hit = signflip_test(informative, n_flips=300, seed=3)
    null = signflip_test(symmetric, n_flips=300, seed=3)
    assert hit["p_signflip"] < 0.01
    assert null["p_signflip"] > 0.05


def _tiny_model(ordered_on=True):
    dim = 6
    spec = {"U_obj": np.eye(dim)[:, :2], "U_subj": np.eye(dim)[:, 4:6],
            "lam_obj": np.array([4.0, 2.0]), "lam_subj": np.array([0.5, 0.25]),
            "k_obj": 2, "k_subj": 2, "sigma1sq": 1.0}
    mu1 = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    mu0 = np.zeros(dim)
    return QDAModel(np.eye(dim), spec, mu1, mu0, ordered_on=ordered_on)


def test_core_matches_the_closed_form_weight_by_weight():
    m = _tiny_model()
    z = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    # axes 0,1 obj with lam 4,2; axes 4,5 subj with lam .5,.25; 2,3 perp
    expect = 0.5 * ((1 - 1 / 4) * 1 + (1 - 1 / 2) * 4
                    + (1 - 1 / 0.5) * 25 + (1 - 1 / 0.25) * 36
                    + (1 - 1 / 1.0) * (9 + 16))
    assert m.core(z)[0] == pytest.approx(expect)


def test_ordered_term_projects_onto_the_object_subspace():
    m = _tiny_model()
    # mu1/sigma1^2 - mu0 = mu1; projected on span(e0,e1) -> only axis 0 left
    z = np.array([[2.0, 7.0, 100.0, 0.0, 0.0, 0.0]])
    assert m.ordered(z)[0] == pytest.approx(2.0 * 1.0)
    m_off = _tiny_model(ordered_on=False)
    assert m_off.ordered(z)[0] == 0.0


def test_v1_quantization_is_obj_energy_minus_subj_energy():
    m = _tiny_model()
    d = np.array([[3.0, 0.0, 1.0, 0.0, 4.0, 0.0]])
    d = d / np.linalg.norm(d)
    got = QDAModel.v1_quantized(d, m.U_obj, m.U_subj)[0]
    assert got == pytest.approx((9.0 - 16.0) / 26.0)


def test_score_api_swap_flips_only_the_ordered_term():
    score_mod = pytest.importorskip("hnav.qda_filter.score")
    if not score_mod.WEIGHTS_NPZ.exists():
        pytest.skip("weights.npz not committed yet")
    v1 = RNG.normal(size=2560)
    v2 = RNG.normal(size=2560)
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    s2 = score_mod.QDAScorer(variant="V2")
    a = s2.score_pairs(v1, v2, 3, 9)
    bswap = s2.score_pairs(v2, v1, 9, 3)     # same pair handed over swapped
    assert a == pytest.approx(bswap, rel=1e-6)   # serials fix orientation
    man = s2.manifest["scalars"]
    if man["ordered_on"]:
        s3 = score_mod.QDAScorer(variant="V3")
        fwd = s3.score_pairs(v1, v2, 3, 9)
        rev = s3.score_pairs(v1, v2, 9, 3)   # orientation genuinely reversed
        core = a
        assert fwd - core == pytest.approx(-(rev - core), rel=1e-5)


# ── 2. artifact consistency ──────────────────────────────────────────────────
def test_manifest_spectrum_and_fit_agree_on_the_model_shape():
    spec = _load("spectrum.json")
    man = _load("weights_manifest.json")
    assert man["scalars"]["k_obj"] == spec["k_obj"]
    assert man["scalars"]["k_subj"] == spec["k_subj"]
    assert man["scalars"]["sigma1_sq"] == pytest.approx(spec["sigma1_sq"])
    fit = _load("fit.json")
    assert man["scalars"]["ordered_on"] == fit["ordered_term_in_score"]


def test_gate_verdicts_follow_their_own_recorded_inputs():
    fit = _load("fit.json")
    g1, g2, g3 = (fit["gates"][k] for k in ("G1", "G2", "G3"))
    assert g1["pass"] == (abs(g1["V1_balanced_sh64k"]
                              - g1["V0_balanced_sh64k"]) <= 0.010)
    d = g2["delta_balanced"]
    assert g2["pass"] == (d["delta"] > (d["hi"] - d["lo"]) / 2
                          and g2["band_V2"] >= g2["band_V0"])
    assert g3["pass"] == (g3["signflip"] and g3["holdout_ok"])


def test_conformal_thresholds_are_reproducible_order_statistics():
    cal = _load("calibration.json")
    for var in ("conformal_V2", "conformal_V4"):
        blk = cal[var]
        n = blk["n0_cal"]
        assert blk["alpha_floor"] == pytest.approx(1.0 / (n + 1))
        for entry in blk["thresholds"].values():
            k = entry["order_statistic_k"]
            assert entry["achievable"] == (k <= n)
            if not entry["achievable"]:
                assert entry["threshold"] is None


# ── 3. regression pins (exact values of the committed campaign) ──────────────
def test_pinned_ranks_and_signflip():
    spec = _load("spectrum.json")
    assert spec["k_obj"] == PIN["k_obj"]
    assert spec["k_subj"] == PIN["k_subj"]
    ot = _load("ordered_term.json")
    assert ot["mu1_test"]["p_signflip"] == pytest.approx(PIN["p_signflip_mu1"])


def test_pinned_v1_vs_ces_agreement_and_v2_headline():
    fit = _load("fit.json")
    g1 = fit["gates"]["G1"]
    assert g1["V0_balanced_sh64k"] == pytest.approx(PIN["V0_bal_sh64k"], abs=1e-4)
    assert g1["V1_balanced_sh64k"] == pytest.approx(PIN["V1_bal_sh64k"], abs=1e-4)
    ej = _load("eval.json")
    m = ej["balanced"]["sh_64k"]["methods"]["V2"]
    assert m["auroc"] == pytest.approx(PIN["V2_bal_sh64k"], abs=1e-4)
    assert m["band_auroc"] == pytest.approx(PIN["V2_band_sh64k"], abs=1e-4)


def test_pinned_conformal_thresholds_v2():
    cal = _load("calibration.json")
    t = cal["conformal_V2"]["thresholds"]
    for key, want in PIN["conformal_V2"].items():
        assert t[key]["threshold"] == pytest.approx(want, rel=1e-6)


PIN = {
    # the committed campaign run (run_all.py, 2026-08-29, seeds in PREREG.md)
    "k_obj": 64,                       # hit the preregistered cap
    "k_subj": 0,                       # no significant variance-deficit tail
    "p_signflip_mu1": 0.0004997501249375312,   # = 1/2001, the add-one floor
    "V0_bal_sh64k": 0.9755888767662941,        # matches REPORT.md sec.7 CES
    "V1_bal_sh64k": 0.9337275870110742,        # pooled quantized core (G1 fail)
    "V2_bal_sh64k": 0.869460297597709,         # the G2 null result
    "V2_band_sh64k": 0.7971965829469158,
    "conformal_V2": {"0.1": -243.538858667581,
                     "0.01": -192.57251914164803,
                     "0.001": -155.91374567737594,
                     "alpha_floor": -155.91374567737594},
}
