"""Live read path, end to end through the adapter.  [T11]

The full seam — memorize → on_retrieve → gate → rerank → apply_read_decision —
exercised with :class:`StubNLI` and :class:`HashEmbedder` on chunks whose
conflict structure is known by construction, in all three modes. HashEmbedder
vectors are near-orthogonal, so the geometric screens are disabled
(``cos_pair=None, r_min=None, ambiguity_mode='none'``) and verification rides
on the NLI stage plus the adapter's key-equality pair filter — each of which
carries its own closed-form tests elsewhere. What THIS file pins is the
plumbing: who is allowed to act, and that acting is a permutation.
"""
from __future__ import annotations

import numpy as np

from hnav import config as _config
from hnav.adapters import mab_adapter as mab
from hnav.core.embedding import DiskCachedEmbedder, HashEmbedder
from hnav.core.read_gate import GateThresholds, ReadGate, StubNLI
from hnav.core.read_policy import ReadRerankPolicy

OFF = _config.HNavConfig(mode=_config.MODE_OFF)
SHADOW = _config.HNavConfig(mode=_config.MODE_SHADOW)
LIVE = _config.HNavConfig(mode=_config.MODE_LIVE)

# Chunk A holds the STALE fact (serial 1) plus a distractor; chunk B holds the
# superseding fact (serial 3). Native ranking puts A first — the arena's
# failure shape — so the correct rerank is exactly [B, A].
CHUNK_A = ("1. Thomas Kyd was born in the city of London. "
           "2. The chairperson of Fatah is Mahmoud Abbas.")
CHUNK_B = "3. Thomas Kyd was born in the city of Madrid."

NLI_ONLY = GateThresholds(cos_pair=None, r_min=None, ambiguity_mode="none")


def _adapter(cfg, embedder):
    policy = ReadRerankPolicy(ReadGate(StubNLI(), NLI_ONLY,
                                       pair_filter=mab.MABAdapter.same_key_pair))
    return mab.MABAdapter(cfg=cfg, embedder=embedder, read_policy=policy)


def _retrieve(adapter):
    return adapter.on_retrieve(
        query="Where was Thomas Kyd born?",
        ranked_texts=[CHUNK_A, CHUNK_B],
        scores=[95.0, 90.0], top_k=2, score_kind="similarity")


class _Doc:
    def __init__(self, page_content):
        self.page_content = page_content


SCORED = [(_Doc(CHUNK_A), 0.1), (_Doc(CHUNK_B), 0.2)]


def _memorize(adapter):
    adapter.before_memorize(CHUNK_A, context_id=0, chunk_index=0)
    adapter.before_memorize(CHUNK_B, context_id=0, chunk_index=1)


# ── live: the one armed seam ─────────────────────────────────────────────────
def test_live_end_to_end_rerank(embedder):
    adapter = _adapter(LIVE, embedder)
    _memorize(adapter)
    decision = _retrieve(adapter)

    assert decision.action == "RERANK"
    assert decision.shadow is False, "live mode must arm the decision"
    g = decision.reasons["groups"][0]
    assert g["latest_id"] == "fact:3" and g["stale_ids"] == ["fact:1"]

    page = adapter.apply_read_decision(decision, SCORED, top_k=2)
    assert page == [CHUNK_B, CHUNK_A]          # latest carrier promoted
    assert sorted(page) == sorted([CHUNK_A, CHUNK_B])   # token-neutral
    assert adapter.n_reranks_applied == 1 and adapter.n_rerank_fallbacks == 0


def test_live_distractor_key_never_joins_the_group(embedder):
    """The Fatah fact shares chunk A but not the key; with the pair filter on
    it must not appear in any group (Note-1 screen, wired end to end)."""
    adapter = _adapter(LIVE, embedder)
    _memorize(adapter)
    decision = _retrieve(adapter)
    members = {m for grp in decision.reasons["groups"] for m in grp["member_ids"]}
    assert members == {"fact:1", "fact:3"}


def test_live_corrupt_payload_falls_back_to_native(embedder):
    adapter = _adapter(LIVE, embedder)
    _memorize(adapter)
    decision = _retrieve(adapter)
    decision.payload = {"order": ["chunk:bogus", "chunk:other"]}
    page = adapter.apply_read_decision(decision, SCORED, top_k=2)
    assert page == [CHUNK_A, CHUNK_B]
    assert adapter.n_rerank_fallbacks == 1 and adapter.n_reranks_applied == 0


def test_live_pass_decision_returns_native(embedder):
    adapter = _adapter(LIVE, embedder)
    _memorize(adapter)
    decision = adapter.on_retrieve(
        query="q", ranked_texts=[CHUNK_B, CHUNK_A],   # latest already first
        scores=[95.0, 90.0], top_k=2, score_kind="similarity")
    assert decision.action == "PASS" and decision.shadow is False
    scored = [(_Doc(CHUNK_B), 0.1), (_Doc(CHUNK_A), 0.2)]
    assert adapter.apply_read_decision(decision, scored, 2) == [CHUNK_B, CHUNK_A]
    assert adapter.n_reranks_applied == 0


# ── shadow: decided, logged, never applied ───────────────────────────────────
def test_shadow_computes_the_decision_but_stays_disarmed(embedder):
    adapter = _adapter(SHADOW, embedder)
    _memorize(adapter)
    decision = _retrieve(adapter)
    assert decision.action == "RERANK", "shadow must still LOG what it would do"
    assert decision.shadow is True
    page = adapter.apply_read_decision(decision, SCORED, top_k=2)
    assert page == [CHUNK_A, CHUNK_B], "shadow may never reorder"
    assert adapter.n_reranks_applied == 0


# ── off: nothing at all ──────────────────────────────────────────────────────
def test_off_records_and_applies_nothing(embedder):
    adapter = _adapter(OFF, embedder)
    _memorize(adapter)
    decision = _retrieve(adapter)
    assert decision.action == "PASS" and decision.shadow is True
    assert adapter.facts == [] and adapter.n_read_decisions == 0
    page = adapter.apply_read_decision(decision, SCORED, top_k=2)
    assert page == [CHUNK_A, CHUNK_B]


# ── read-candidate pool ──────────────────────────────────────────────────────
def test_read_candidate_pool_caps_by_query_cosine_and_keeps_order(embedder):
    cfg = _config.HNavConfig(mode=_config.MODE_SHADOW, top_m=3)
    adapter = mab.MABAdapter(cfg=cfg, embedder=embedder)
    chunk = " ".join(f"{i}. fact number {i} says thing {i}." for i in range(1, 9))
    adapter.before_memorize(chunk, context_id=0, chunk_index=0)
    from hnav.core.types import RetrievalView
    cid = adapter.chunks[0].id
    target = adapter._fact_by_id["fact:5"]
    view = RetrievalView(query="q", query_vector=target.vector,
                         ids=[cid], scores=np.array([90.0]), top_k=1)
    pool = adapter._read_candidates(view)
    assert len(pool) == 3
    assert "fact:5" in {r.id for r in pool}, "its own vector must select itself"
    serials = [r.version for r in pool]
    assert serials == sorted(serials), "pool must keep (chunk, serial) order"


def test_read_candidate_pool_excludes_vectorless_facts(embedder):
    adapter = mab.MABAdapter(cfg=SHADOW, embedder=None)   # no embedder → no vectors
    _memorize(adapter)
    from hnav.core.types import RetrievalView
    view = RetrievalView(query="q", query_vector=None,
                         ids=[c.id for c in adapter.chunks],
                         scores=np.array([90.0, 80.0]), top_k=2)
    assert adapter._read_candidates(view) == []


# ── degradation and defaults ─────────────────────────────────────────────────
def test_get_adapter_off_is_none_and_shadow_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("HNAV_MODE", "off")
    _config.get_config(refresh=True)
    mab.reset_adapter()
    assert mab.get_adapter() is None

    # shadow on a box where the NLI cannot build: the adapter must still
    # exist and run its read path with read_policy=None (PASS decisions).
    # build_nli is forced to fail so the test is deterministic everywhere —
    # a machine with cached weights would otherwise load a real model here.
    import hnav.core.read_gate as rg

    def _no_nli(cfg):
        raise RuntimeError("nli unavailable in this test")

    monkeypatch.setattr(rg, "build_nli", _no_nli)
    monkeypatch.setenv("HNAV_MODE", "shadow")
    _config.get_config(refresh=True)
    mab.reset_adapter()
    adapter = mab.get_adapter()
    assert adapter is not None and adapter.read_policy is None
    assert adapter is not None and adapter.mode == "shadow"
    decision = adapter.on_retrieve("q", [CHUNK_A], [90.0], top_k=1)
    assert decision.action == "PASS" and decision.shadow is True

    mab.reset_adapter()
    monkeypatch.setenv("HNAV_MODE", "off")
    _config.get_config(refresh=True)


def test_embed_failure_degrades_to_no_vectors(tmp_path):
    class Exploding:
        dim = 8

        def encode(self, texts):
            raise ConnectionError("endpoint down")

    adapter = mab.MABAdapter(cfg=SHADOW, embedder=Exploding())
    out = adapter.before_memorize(CHUNK_B, context_id=0)
    assert out is CHUNK_B
    assert adapter.n_embed_errors >= 1
    assert adapter.facts[0].vector is None


def test_disk_cache_persist_false_never_writes(tmp_path):
    inner = HashEmbedder(dim=16)
    ro = DiskCachedEmbedder(inner, tmp_path, key="k", persist=False)
    v1 = ro.encode(["some text"])
    assert ro.n_misses == 1 and list(tmp_path.glob("*.npy")) == []
    rw = DiskCachedEmbedder(inner, tmp_path, key="k", persist=True)
    v2 = rw.encode(["some text"])
    assert rw.n_misses == 1 and len(list(tmp_path.glob("*.npy"))) == 1
    ro2 = DiskCachedEmbedder(inner, tmp_path, key="k", persist=False)
    v3 = ro2.encode(["some text"])
    assert ro2.n_hits == 1 and ro2.n_misses == 0
    assert np.array_equal(v1, v2) and np.array_equal(v2, v3)
