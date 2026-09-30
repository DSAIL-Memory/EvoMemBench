"""CrossEp-Know adapter signal wiring.  [T10]

Before T10, ``CLBenchAdapter.on_extract`` hardcoded ``geometry=None, diff=None,
retrieval_effect=None`` even when the modules were injected — the secondary
arena logged zero write-side signals. These tests pin the repaired behaviour to
the repo's closed-form standard:

* every logged number is checked against an ORACLE computed independently in
  the test (numpy SVD projection instead of the module's QR; a hand-rolled
  cosine instead of ``GeometryModule``), on :class:`HashEmbedder` vectors that
  are pure functions of the text;
* shadow legality is asserted mechanically — the inner backend is called
  exactly once with the caller's own kwargs, its return object is returned by
  identity, and the store handed to the adapter is never mutated;
* the mixed-space trap is exercised: a bank carrying native-API vectors of a
  different dimensionality must not crash signal computation, because the
  adapter re-embeds every bank text into its own space first.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav import config as _config
from hnav.adapters import clbench_adapter as cl
from hnav.core.audit import AuditLogger
from hnav.core.diff_geometry import DiffGeometryModule
from hnav.core.embedding import HashEmbedder
from hnav.core.geometry import GeometryModule
from hnav.core.replica import NumpyCosineReplica
from hnav.core.retrieval_signals import RetrievalSignals
from hnav.core.types import MemoryRecord, StoreView
from hnav.tests.conftest import build_store, random_texts

SHADOW = _config.HNavConfig(mode=_config.MODE_SHADOW)


def full_adapter(emb: HashEmbedder | None = None) -> cl.CLBenchAdapter:
    """An adapter with every optional module injected, audit in memory."""
    emb = emb or HashEmbedder(dim=64)
    return cl.CLBenchAdapter(
        cfg=SHADOW,
        embedder=emb,
        replica=NumpyCosineReplica(emb, top_k=10),
        audit=AuditLogger(None, run_id="test", suite="crossep"),
        signals=RetrievalSignals(top_m=50, k=10),
        geometry=GeometryModule(),
        diff=DiffGeometryModule(emb),
    )


# ── signals flow, checked against independent oracles ────────────────────────
def test_write_signals_flow_and_match_numpy_oracles():
    emb = HashEmbedder(dim=64)
    adapter = full_adapter(emb)
    texts = random_texts(6, seed=3)
    store = build_store(texts, emb)

    cand_text = "a genuinely new statement about delta and echo"
    cand = adapter.to_candidate(cand_text, task_id="t1", context_id="ctx1")
    decision = adapter.on_extract(cand, store)

    assert decision.action == "PASS" and decision.shadow is True
    assert decision.reasons["bank_space"] == "hnav"

    rec = adapter.audit.records[-1]
    geo = rec["geometry"]
    assert geo is not None, "geometry must flow when the module is injected"

    # Oracle: the same HashEmbedder is a pure function of the text, so the
    # bank can be rebuilt independently and the cosine computed by hand.
    bank = emb.encode(texts).astype(np.float64)
    cv = emb.encode([cand_text])[0].astype(np.float64)
    sims = bank @ cv
    assert geo["sim_max"] == pytest.approx(float(sims.max()), abs=1e-6)
    assert geo["argmax_id"] == f"rec:{int(sims.argmax())}"

    # Oracle for the QR residual, via SVD projection — a different algorithm
    # that must land on the same number.
    u, s, _ = np.linalg.svd(bank.T, full_matrices=False)
    u = u[:, s > 1e-10]
    resid = float(np.linalg.norm(cv - u @ (u.T @ cv)))
    assert geo["qr_residual_r"] == pytest.approx(resid, abs=1e-6)
    assert geo["qr_basis_subsampled"] is False
    assert geo["store_size"] == 6

    # Effect: the candidate is its own probe; cos(c, c) = 1.0 puts it at
    # rank 0 above the near-orthogonal store rows.
    eff = rec["retrieval_effect"]
    assert eff is not None, "retrieval effect must flow when replica+signals exist"
    assert eff["rank_self_after"] == 0
    assert eff["churn_at_k"] == 0 and eff["displaced_ids"] == []
    assert np.isfinite(eff["dH_self"])

    # Diff: prior was resolved to the nearest neighbour, so a DiffSignals
    # record must exist and cohere with the same oracle bank.
    diff = rec["diff"]
    assert diff is not None, "diff must flow once a nearest-neighbour prior exists"
    prior_vec = bank[int(sims.argmax())]
    assert diff["whole_blob_sim"] == pytest.approx(float(prior_vec @ cv), abs=1e-6)
    assert diff["parse_ok"] is False   # free text: token-level fallback only


def test_nearest_neighbour_prior_resolution():
    emb = HashEmbedder(dim=64)
    adapter = full_adapter(emb)
    texts = random_texts(5, seed=7)
    store = build_store(texts, emb)

    # Identical text -> cosine exactly 1.0 with rec:2, near-orthogonal to the
    # rest: the predecessor is deterministic.
    cand = adapter.to_candidate(texts[2], task_id="t1", context_id="ctx1")
    adapter.on_extract(cand, store)

    rec = adapter.audit.records[-1]
    assert rec["prior_id"] == "rec:2"
    assert rec["prior_text"] == texts[2]
    assert rec["hnav_reasons"]["has_prior"] is True
    assert rec["geometry"]["sim_max"] == pytest.approx(1.0, abs=1e-6)
    assert rec["diff"]["whole_blob_sim"] == pytest.approx(1.0, abs=1e-6)


def test_exact_duplicate_detected_across_writes():
    adapter = full_adapter()
    store = StoreView.from_records([])

    first = adapter.to_candidate("the same trajectory text", task_id="a")
    d1 = adapter.on_extract(first, store)
    assert d1.reasons["is_exact_dup"] is False
    assert d1.reasons["has_prior"] is False          # empty store: no prior
    assert d1.reasons["bank_space"] == "empty_store"

    second = adapter.to_candidate("the same trajectory text", task_id="b")
    d2 = adapter.on_extract(second, store.with_provisional(first))
    assert d2.reasons["is_exact_dup"] is True, \
        "MD5 dedup must see the first admission via geometry.observe"
    assert adapter.audit.records[-1]["geometry"]["is_exact_dup"] is True


# ── the mixed-space trap ─────────────────────────────────────────────────────
def _native_vector_store(n: int = 4) -> StoreView:
    """A bank the way a live backend hands it over: vectors from the
    benchmark's own embedding API, in a dimensionality the H-Nav embedder does
    not share (here 8-d vs HashEmbedder's 64-d)."""
    rng = np.random.default_rng(5)
    recs = []
    for i in range(n):
        v = rng.standard_normal(8).astype(np.float32)
        recs.append(MemoryRecord(id=f"nat:{i}", text=f"native chunk {i}",
                                 vector=v / np.linalg.norm(v), version=i))
    return StoreView.from_records(recs)


def test_native_vectors_are_reembedded_not_mixed():
    adapter = full_adapter()
    store = _native_vector_store()
    before_vecs = [r.vector for r in store.records]

    cand = adapter.to_candidate("a new write", task_id="t")
    adapter.on_extract(cand, store)

    rec = adapter.audit.records[-1]
    assert rec["geometry"] is not None
    assert np.isfinite(rec["geometry"]["sim_max"]), \
        "signals must be computed in the adapter's own space, not the native one"
    assert rec["retrieval_effect"] is not None
    # the caller's store is untouched: same records, same native vectors
    assert len(store) == 4
    assert all(store.records[i].vector is before_vecs[i] for i in range(4))


def test_on_retrieve_ranks_in_adapter_space():
    adapter = full_adapter()
    store = _native_vector_store()
    decision = adapter.on_retrieve("which chunk?", store, native_text="native")
    assert decision.action == "PASS" and decision.shadow
    assert decision.reasons["n_ranked"] == 4
    read_rec = adapter.audit.records[-1]
    assert read_rec["schema"] == "hnav.read.v1"
    assert read_rec["signals"] is not None


# ── degraded configurations stay honest, never crash ─────────────────────────
def test_no_embedder_logs_explicit_nulls():
    adapter = cl.CLBenchAdapter(cfg=SHADOW, audit=AuditLogger(None),
                                geometry=GeometryModule())
    store = build_store(random_texts(3, seed=1), HashEmbedder(dim=16))
    cand = adapter.to_candidate("text without a vector", task_id="t")
    decision = adapter.on_extract(cand, store)

    assert decision.action == "PASS" and decision.shadow
    assert decision.reasons["bank_space"] == "unavailable_no_embedder"
    rec = adapter.audit.records[-1]
    assert rec["retrieval_effect"] is None and rec["diff"] is None
    assert rec["geometry"]["store_size"] == 3   # dup/tau bookkeeping still runs
    sim = rec["geometry"]["sim_max"]            # not computed: NaN (or null)
    assert sim is None or not np.isfinite(sim)


# ── shadow legality through the wrapper, with modules injected ───────────────
class _RecordingMemory:
    """Counts calls and records kwargs so extra work is detectable."""

    def __init__(self) -> None:
        self.memory_dir = "memories/run_1/ctx_abc123"
        self.memory_bank: list = []
        self.embeddings: list = []
        self.n_extract = 0
        self.seen_kwargs: dict | None = None
        self.extract_result = {"latency_s": 0.1, "llm_usage": {}, "embed_usage": {}}

    def retrieve(self, query):
        return "", {"latency_s": 0.0, "embed_usage": {}}

    def extract(self, content, **kwargs):
        self.n_extract += 1
        self.seen_kwargs = kwargs
        self.memory_bank.append({"chunk_text": content})
        return self.extract_result


def test_wrapper_with_full_modules_stays_byte_neutral():
    inner = _RecordingMemory()
    adapter = full_adapter()
    wrapper = cl.HNavMemoryWrapper(inner, adapter, mode=_config.MODE_SHADOW)

    out = wrapper.extract("a trajectory", task_id="t9", query="q9")
    assert out is inner.extract_result          # identity, not a copy
    assert inner.n_extract == 1                 # exactly one native call
    assert inner.seen_kwargs == {"task_id": "t9", "query": "q9"}, \
        "the recovered context_id must never leak into the inner call"

    rec = adapter.audit.records[-1]
    assert rec["context_id"] == "ctx_abc123"    # from memory_dir's basename
    assert rec["schema"] == "hnav.write.v1"
    assert rec["hnav_action"] == "PASS" and rec["shadow"] is True


def test_live_singleton_does_not_cross_contaminate_contexts():
    """One adapter (the live path's module singleton) serving two wrapped
    backends must keep their dup/tau state apart: native banks are
    per-context isolated, so an identical chunk in two different contexts is
    NOT a duplicate anywhere the benchmark can see."""
    adapter = full_adapter()
    mem_a, mem_b = _RecordingMemory(), _RecordingMemory()
    mem_a.memory_dir = "memories/run_1/ctxA"
    mem_b.memory_dir = "memories/run_1/ctxB"
    wrap_a = cl.HNavMemoryWrapper(mem_a, adapter, mode=_config.MODE_SHADOW)
    wrap_b = cl.HNavMemoryWrapper(mem_b, adapter, mode=_config.MODE_SHADOW)

    shared = "an identical chunk of trajectory text"
    wrap_a.extract(shared, task_id="a1")
    wrap_b.extract(shared, task_id="b1")

    recs = adapter.audit.records
    assert recs[0]["hnav_reasons"]["is_exact_dup"] is False
    assert recs[1]["hnav_reasons"]["is_exact_dup"] is False, \
        "context B must not see context A's MD5 observe set"
    assert recs[1]["context_id"] == "ctxB"

    # ...while a genuine within-context repeat IS still flagged
    wrap_a.extract(shared, task_id="a2")
    assert recs[2]["hnav_reasons"]["is_exact_dup"] is True
    assert recs[2]["context_id"] == "ctxA"

    # the injected prototype module was never polluted by keyed contexts
    assert adapter.geometry._dup_hashes == set()


def test_per_context_state_is_lru_bounded():
    """A long multi-context run must not grow adapter state without bound."""
    from hnav.core.types import StoreView
    adapter = full_adapter()
    empty = StoreView.from_records([])
    for i in range(adapter.MAX_CONTEXT_STATES + 10):
        cand = adapter.to_candidate(f"text {i}", task_id=f"t{i}",
                                    context_id=f"ctx{i}")
        adapter.on_extract(cand, empty)
    assert len(adapter._memos) <= adapter.MAX_CONTEXT_STATES
    assert len(adapter._geo_states) <= adapter.MAX_CONTEXT_STATES
    # the most recent contexts survive; the oldest were evicted
    assert f"ctx{adapter.MAX_CONTEXT_STATES + 9}" in adapter._geo_states
    assert "ctx0" not in adapter._geo_states


def test_wrapper_off_mode_is_identity(monkeypatch):
    monkeypatch.setenv("HNAV_MODE", "off")
    _config.get_config(refresh=True)
    cl.reset_adapter()
    inner = _RecordingMemory()
    assert cl.wrap_memory(inner) is inner
    _config.get_config(refresh=True)
