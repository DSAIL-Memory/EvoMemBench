"""GEO identity screen — machinery checks and frozen-artifact pins.  [E2E-3]

Covers the paths standing between the money and the sh_64k one-shot (the
gaps the pre-shot review flagged): the string-tau round-trip through
frozen_cell/make_gate, the text-keyed margin cache, tau parsing, and the
committed artifact/operating-point consistency. Synthetic pieces need no
embedding cache; committed-artifact pieces skip when the files are absent.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from hnav.geometry_filter.geo_artifact import ARTIFACT_JSON, GeoIdentityScreen

REPO = pathlib.Path(__file__).resolve().parents[2]
OP = REPO / "stage0_results" / "geometry_filter" / "geo_operating_point.json"

RNG = np.random.default_rng(5)


def _toy(T_w=0.0, T_p=0.0, s_w=1.0, s_p=1.0) -> GeoIdentityScreen:
    dim = 8
    w = RNG.normal(size=dim)
    comps = np.linalg.qr(RNG.normal(size=(dim, 2)))[0].T   # 2 orthonormal rows
    return GeoIdentityScreen(w, 0.1, RNG.normal(size=dim) * 0.01, comps,
                             T_w, T_p, s_w, s_p)


class _Rec:
    def __init__(self, id, text, vector):
        self.id, self.text, self.vector = id, text, vector


# ── tau parsing ──────────────────────────────────────────────────────────────
def test_parse_tau_floats_strings_and_rectangles():
    assert GeoIdentityScreen.parse_tau(0.1) == (0.1, 0.1)
    assert GeoIdentityScreen.parse_tau("0.25") == (0.25, 0.25)
    assert GeoIdentityScreen.parse_tau("-0.4:0.2") == (-0.4, 0.2)
    with pytest.raises(ValueError):
        GeoIdentityScreen.parse_tau("abc")
    with pytest.raises(ValueError):
        GeoIdentityScreen.parse_tau("nan")          # NaN rejects every pair
    with pytest.raises(ValueError):
        GeoIdentityScreen.parse_tau("nan:0.2")


# ── margins ──────────────────────────────────────────────────────────────────
def test_margins_are_symmetric_and_zero_diff_is_rejected():
    art = _toy()
    v1, v2 = RNG.normal(size=8), RNG.normal(size=8)
    m12, m21 = art.margins_pair(v1, v2), art.margins_pair(v2, v1)
    assert m12 == pytest.approx(m21)                # |d| and cos_w symmetric
    assert art.margins_pair(v1, v1) == (-np.inf, -np.inf)
    assert not art.pair_filter(-100.0)(_Rec("a", "ta", v1), _Rec("b", "tb", v1))


def test_rectangle_filter_is_the_conjunction_of_both_margins():
    art = _toy()
    v1, v2 = RNG.normal(size=8), RNG.normal(size=8)
    mw, mp = art.margins_pair(v1, v2)
    a, b = _Rec("x", "tx", v1), _Rec("y", "ty", v2)
    eps = 1e-9
    assert art.pair_filter(f"{mw - eps}:{mp - eps}")(a, b)
    assert not art.pair_filter(f"{mw + eps}:{mp - eps}")(a, b)   # cos leg fails
    assert not art.pair_filter(f"{mw - eps}:{mp + eps}")(a, b)   # probe leg fails


def test_margin_cache_keys_on_text_not_id():
    """The review's poisoning case: same fact ids, different facts (subsets
    restart serials). Text-keyed caching must keep the verdicts apart."""
    art = _toy()
    v1, v2 = RNG.normal(size=8), RNG.normal(size=8)
    v3, v4 = RNG.normal(size=8), RNG.normal(size=8)
    f = art.pair_filter(-1000.0)                    # admit-everything tau
    assert f(_Rec("fact:1", "alpha", v1), _Rec("fact:2", "beta", v2))
    # same ids, different texts/vectors: must be scored fresh, not served
    # from the first pair's cache slot
    m_first = art._cache[("alpha", "beta")]
    f(_Rec("fact:1", "gamma", v3), _Rec("fact:2", "delta", v4))
    assert ("delta", "gamma") in art._cache        # keys are sorted texts
    assert art._cache[("delta", "gamma")] != m_first


# ── frozen artifacts ─────────────────────────────────────────────────────────
def _man():
    if not ARTIFACT_JSON.exists():
        pytest.skip("geo artifact not committed")
    return json.loads(ARTIFACT_JSON.read_text(encoding="utf-8"))


def test_committed_artifact_roundtrip_and_fingerprint():
    man = _man()
    art, man2 = GeoIdentityScreen.load(ARTIFACT_JSON)
    assert art.fingerprint() == man["fingerprint"]
    assert man2["provenance"]["fit_subsets"] == ["sh_6k", "sh_32k"]
    # provenance paths must be POSIX (the E2E-2 backslash lesson)
    assert "\\" not in man2["provenance"]["abtt_source"]
    assert "\\" not in man2["provenance"]["dataset"]


def test_operating_point_pins_the_committed_artifact():
    if not OP.exists():
        pytest.skip("geo operating point not committed")
    man = _man()
    op = json.loads(OP.read_text(encoding="utf-8"))
    assert op["pair_filter"] == "geo"
    assert op["ces"]["fingerprint"] == man["fingerprint"]
    assert op["metrics"]["n_suppressed_harmful"] == 0
    assert op["metrics"]["pair_precision"] == 1.0
    # GG1: strictly better than the best committed parser-free point
    assert op["metrics"]["pair_recall_pool"] > 0.4444
    # the frozen rectangle survives the JSON round-trip as a string
    tw, tp = GeoIdentityScreen.parse_tau(op["ces"]["tau"])
    assert np.isfinite(tw) and np.isfinite(tp)


def test_frozen_cell_and_make_gate_accept_the_string_tau():
    """The wet-run critical path: op JSON -> frozen_cell -> make_gate with a
    rectangle tau string must build a working gate filter."""
    if not OP.exists():
        pytest.skip("geo operating point not committed")
    from hnav.stage1.detector_gap import frozen_cell, make_gate
    art, _ = GeoIdentityScreen.load(ARTIFACT_JSON)
    cell = frozen_cell("raw", "geo")
    assert isinstance(cell["ces_tau"], str) and ":" in cell["ces_tau"]
    assert cell["ces_fingerprint"] == art.fingerprint()
    gate = make_gate(cell, replay=None, ces=art)
    assert gate.pair_filter is not None
    v = RNG.normal(size=art.probe_w.shape[0])
    out = gate.pair_filter(_Rec("fact:0", "t0", v),
                           _Rec("fact:1", "t1", RNG.normal(size=v.shape[0])))
    assert out in (True, False)
