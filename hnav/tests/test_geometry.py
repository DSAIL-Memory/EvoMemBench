"""Geometry and marginal-diff numerics, against constructed answers.  [T6]"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.diff_geometry import DiffGeometryModule, token_diff_spans
from hnav.core.geometry import (ABTTWhitening, GeometryModule, TauPolicy,
                                qr_residual, verbatim_values)
from hnav.core.types import Candidate
from hnav.tests.conftest import build_store, random_texts


# ── QR residual ──────────────────────────────────────────────────────────────
def test_qr_residual_is_zero_inside_the_span_and_one_outside():
    basis = np.eye(5)[:3]                      # spans e0, e1, e2
    inside = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    inside /= np.linalg.norm(inside)
    outside = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])

    r_in, sub = qr_residual(inside, basis)
    assert r_in[0] == pytest.approx(0.0, abs=1e-12)
    assert sub is False

    r_out, _ = qr_residual(outside, basis)
    assert r_out[0] == pytest.approx(1.0)


def test_qr_residual_partial_projection():
    basis = np.eye(3)[:1]                      # spans e0 only
    v = np.array([[0.6, 0.8, 0.0]])            # unit; component on e0 is 0.6
    r, _ = qr_residual(v, basis)
    assert r[0] == pytest.approx(0.8)


def test_qr_residual_subsampling_is_reported():
    bank = np.eye(40)
    r, sub = qr_residual(np.ones((1, 40)) / np.sqrt(40), bank, max_basis=10)
    assert sub is True, "subsampling must be reported, since r becomes an upper bound"
    assert r[0] > 0.0


def test_qr_residual_on_an_empty_bank_is_maximal():
    r, sub = qr_residual(np.ones((2, 4)), np.zeros((0, 4)))
    assert list(r) == [1.0, 1.0] and sub is False


# ── ABTT ─────────────────────────────────────────────────────────────────────
def test_abtt_refuses_to_fit_below_min_n():
    w = ABTTWhitening(n_components=3, min_fit_n=200)
    w.fit(np.random.default_rng(0).standard_normal((199, 16)))
    assert w.fitted is False and w.refused is True

    v = np.ones(16) / 4.0
    assert np.allclose(w.transform(v), v), "an unfitted whitener must pass through"


def test_abtt_removes_the_dominant_direction():
    """A shared component inflates every cosine; ABTT is there to remove it."""
    rng = np.random.default_rng(1)
    common = np.zeros(32)
    common[0] = 3.0
    raw = rng.standard_normal((400, 32)) + common
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    before = float(np.mean(raw @ raw.T))
    w = ABTTWhitening(n_components=3, min_fit_n=200).fit(raw)
    assert w.fitted is True and w.refused is False

    after_m = w.transform(raw)
    after = float(np.mean(after_m @ after_m.T))
    assert after < before
    assert np.allclose(np.linalg.norm(after_m, axis=1), 1.0, atol=1e-6)


def test_abtt_is_deterministic():
    m = np.random.default_rng(2).standard_normal((300, 12))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    a = ABTTWhitening(min_fit_n=200).fit(m).transform(m)
    b = ABTTWhitening(min_fit_n=200).fit(m).transform(m)
    assert np.array_equal(a, b)


# ── tau policy ───────────────────────────────────────────────────────────────
def test_tau_policy_holds_its_prior_until_min_n():
    tau = TauPolicy(init_tau=0.85, z=1.0, min_n=50)
    for _ in range(49):
        tau.update(0.10)
        assert tau.tau_t == 0.85
    tau.update(0.10)
    assert tau.tau_t == pytest.approx(0.50)      # clipped at lo


def test_tau_policy_tracks_mean_plus_z_sigma():
    tau = TauPolicy(init_tau=0.85, z=1.0, min_n=4, lo=0.0, hi=1.0)
    values = [0.6, 0.7, 0.8, 0.9]
    for v in values:
        tau.update(v)
    expected = np.mean(values) + np.std(values)
    assert tau.tau_t == pytest.approx(expected)


def test_tau_policy_ignores_nan():
    tau = TauPolicy(min_n=1, lo=0.0, hi=1.0)
    tau.update(float("nan"))
    assert tau.n == 0 and tau.tau_t == 0.85


# ── verbatim values ──────────────────────────────────────────────────────────
def test_verbatim_values_catches_the_answer_bearing_span():
    old = "The univeristy where Barack Obama was educated is Howard University."
    new = "The univeristy where Barack Obama was educated is Harvard University."
    new_values = verbatim_values(new) - verbatim_values(old)
    assert any("Harvard" in v for v in new_values)
    assert verbatim_values(new) - verbatim_values(new) == set()
    assert verbatim_values("") == set()


def test_verbatim_values_catches_numbers():
    assert "1999" in verbatim_values("released in 1999")


# ── GeometryModule ───────────────────────────────────────────────────────────
def test_geometry_module_exact_duplicate_and_sim_max(embedder):
    texts = random_texts(30, seed=20)
    store = build_store(texts, embedder)
    geom = GeometryModule(whitening=None, tau_policy=TauPolicy(min_n=1, lo=0.0, hi=1.0))
    for t in texts:
        geom.observe(t)

    dup = Candidate(id="c1", text=texts[7], vector=embedder.encode([texts[7]])[0])
    sig = geom.compute(dup, store)
    assert sig.is_exact_dup is True
    assert sig.sim_max == pytest.approx(1.0, abs=1e-5)
    assert sig.argmax_id == store.records[7].id
    assert sig.qr_residual_r == pytest.approx(0.0, abs=1e-5)

    novel = Candidate(id="c2", text="something entirely different",
                      vector=embedder.encode(["something entirely different"])[0])
    sig2 = geom.compute(novel, store)
    assert sig2.is_exact_dup is False
    assert sig2.sim_max < 0.9


def test_geometry_module_records_the_whitening_refusal(embedder):
    store = build_store(random_texts(40, seed=21), embedder)   # < min_fit_n
    geom = GeometryModule(whitening=ABTTWhitening(min_fit_n=200))
    geom.fit_whitening(store)
    sig = geom.compute(Candidate(id="c", text="x", vector=embedder.encode(["x"])[0]),
                       store)
    assert sig.whitened is False and sig.whitening_refused is True


def test_geometry_module_handles_an_empty_store(embedder):
    from hnav.core.types import StoreView
    geom = GeometryModule()
    sig = geom.compute(Candidate(id="c", text="x", vector=embedder.encode(["x"])[0]),
                       StoreView.from_records([]))
    assert np.isnan(sig.sim_max) and sig.argmax_id is None
    assert sig.tau_t == pytest.approx(0.85)


# ── diff geometry ────────────────────────────────────────────────────────────
def test_token_diff_spans_isolates_the_change():
    old = "Barack Obama was educated at Howard University."
    new = "Barack Obama was educated at Harvard University."
    o, n = token_diff_spans(old, new)
    assert o == "Howard" and n == "Harvard"

    assert token_diff_spans("same text", "same text") == ("", "")


def test_compute_if_update_returns_none_without_a_predecessor(embedder):
    mod = DiffGeometryModule(embedder)
    assert mod.compute_if_update(None, "a new fact") is None
    assert mod.compute_if_update("", "a new fact") is None


def test_critical_delta_fires_on_near_duplicate_with_disjoint_span(embedder):
    """The Howard/Harvard case: nearly identical blobs, disjoint changed span."""
    old = "Barack Obama was educated at Howard University."
    new = "Barack Obama was educated at Harvard University."

    class _Rigged:
        """Cosines pinned to the case under test, so the label logic is what is
        being checked rather than the embedder's opinion."""
        dim = 2

        def encode(self, texts):
            lookup = {old: [1.0, 0.0], new: [0.995, 0.0998],
                      "Howard": [1.0, 0.0], "Harvard": [0.2, 0.9798]}
            return np.array([lookup[t] for t in texts], dtype=np.float64)

    mod = DiffGeometryModule(_Rigged(), span_fn=None, tau_high=0.80, tau_low=0.30)
    d = mod.compute_if_update(old, new)
    assert d.whole_blob_sim > 0.98
    assert d.diff_sim < 0.30
    assert d.is_critical_delta is True
    assert d.parse_ok is False          # fell back to the token differ


def test_span_fn_failure_falls_back_and_records_it(embedder):
    def always_fails(old, new):
        raise ValueError("no template matched")

    mod = DiffGeometryModule(embedder, span_fn=always_fails)
    d = mod.compute_if_update("old fact here", "new fact here")
    assert d is not None and d.parse_ok is False
    assert d.diff_span_old == "old" and d.diff_span_new == "new"


def test_diff_novelty_uses_the_prior_object_vocabulary(embedder):
    mod = DiffGeometryModule(embedder)
    prior = embedder.encode(["London", "Madrid", "Paris"]).astype(np.float64)

    seen = mod.compute_if_update("X was born in the city of London.",
                                 "X was born in the city of Madrid.",
                                 prior_objects=prior)
    novel = mod.compute_if_update("X was born in the city of London.",
                                  "X was born in the city of Ulaanbaatar.",
                                  prior_objects=prior)
    assert seen.diff_novelty < novel.diff_novelty
    assert np.isnan(mod.compute_if_update("a b", "a c").diff_novelty)
