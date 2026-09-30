"""M0 — retriever-replica fidelity.  [T3]

The M0 criterion is "≥99.9% exact top-k identity vs the native retriever, 100%
for NumpyCosineReplica modulo documented ties". On a machine with no GPU and no
faiss the native retriever cannot be *run*, but for the CrossEp-Know arena it
does not need to be: the native computation is two lines of numpy
(``qwen3_embedding_memory.py:218-221``) and is inlined verbatim below as the
oracle. Agreement against that oracle is the same claim, computed the same way,
and is asserted here on 1,200 sampled (store, query) pairs.

What this file does NOT establish, and what therefore has to be re-measured on
the GPU box (see hnav/BUILD_NOTES.md):
  * agreement against a live LangChain FAISS index (tie order inside FAISS is
    not reproducible from outside it),
  * agreement when the vectors come from the real embedder rather than
    HashEmbedder. Fidelity is a property of the ranking arithmetic, not of the
    vectors, so this is expected to carry over — but "expected" is not measured.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.replica import (FaissFlatReplica, NumpyCosineReplica, TIE_NOTE,
                               kendall_tau, top_k_agreement)
from hnav.core.types import Candidate, StoreView
from hnav.tests.conftest import build_store, random_texts

N_PAIRS = 1200          # ≥1,000, per protocol §M0
STORE_SIZE = 200
TOP_K = 10


def native_qwen3_rank(bank_matrix: np.ndarray, query_vec: np.ndarray, k: int):
    """Verbatim transcription of ``Qwen3EmbeddingMemory.retrieve``.

    qwen3_embedding_memory.py:218 ::

        scores = (bank_matrix @ query_vec) * 100.0
        top_indices = np.argsort(scores)[::-1][:k]
    """
    scores = (bank_matrix @ query_vec) * 100.0
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices, scores


@pytest.fixture(scope="module")
def store_and_queries(embedder):
    store = build_store(random_texts(STORE_SIZE, seed=1, prefix="bank"), embedder)
    queries = random_texts(N_PAIRS, seed=2, prefix="query")
    return store, queries


def test_numpy_cosine_replica_is_bit_exact(embedder, store_and_queries):
    """100% exact top-k identity AND bit-identical scores over 1,200 pairs."""
    store, queries = store_and_queries
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)
    bank = store.matrix

    exact_topk = 0
    exact_top1 = 0
    max_abs_err = 0.0
    for q in queries:
        qv = embedder.encode([q])[0]
        native_idx, native_scores = native_qwen3_rank(bank, qv, TOP_K)
        native_ids = [store.records[int(i)].id for i in native_idx]

        view = rep.rank(store, q, query_vector=qv)
        exact_topk += int(top_k_agreement(view.top_ids, native_ids, TOP_K))
        exact_top1 += int(view.ids[0] == native_ids[0])

        # scores must be bit-identical, not merely close
        replica_scores = rep.rank_scores(store, q, query_vector=qv)
        assert np.array_equal(replica_scores, native_scores)
        max_abs_err = max(max_abs_err, float(np.abs(replica_scores - native_scores).max()))

    assert exact_topk == N_PAIRS, f"top-{TOP_K} identity {exact_topk}/{N_PAIRS}"
    assert exact_top1 == N_PAIRS
    assert max_abs_err == 0.0


def test_full_ranking_identity_and_tau(embedder, store_and_queries):
    """Full-ranking identity, not just the top-k page. τ = 1 by construction."""
    store, queries = store_and_queries
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)
    for q in queries[:50]:
        qv = embedder.encode([q])[0]
        order = np.argsort((store.matrix @ qv) * 100.0)[::-1]
        native_full = [store.records[int(i)].id for i in order]
        view = rep.rank(store, q, query_vector=qv)
        assert view.ids == native_full
        assert kendall_tau(view.ids, native_full) == pytest.approx(1.0)


def test_tie_order_matches_native_expression(embedder):
    """Duplicated bank rows produce exact score ties; replica and native still
    agree because both evaluate ``np.argsort(scores)[::-1]`` — see TIE_NOTE."""
    texts = ["duplicated row"] * 8 + random_texts(12, seed=3)
    store = build_store(texts, embedder)
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)

    qv = embedder.encode(["duplicated row"])[0]
    scores = (store.matrix @ qv) * 100.0
    n_tied = len(scores) - len(np.unique(scores))
    assert n_tied >= 7, "fixture failed to create ties"

    native_idx, _ = native_qwen3_rank(store.matrix, qv, len(texts))
    native_ids = [store.records[int(i)].id for i in native_idx]
    view = rep.rank(store, "duplicated row", query_vector=qv)
    assert view.ids == native_ids
    assert "np.argsort(scores)[::-1] is NOT a stable descending sort" in TIE_NOTE


def test_faiss_replica_matches_l2_distance_ordering(embedder):
    """Ascending squared-L2 (what FAISS returns) == descending cos×100 (what the
    replica reports), exactly, for L2-normalized vectors."""
    store = build_store(random_texts(120, seed=4), embedder)
    rep = FaissFlatReplica(embedder, top_k=TOP_K)
    for q in random_texts(40, seed=5, prefix="q"):
        qv = embedder.encode([q])[0]
        d2 = rep.native_distances(store, qv)
        faiss_order = np.argsort(d2, kind="stable")
        faiss_ids = [store.records[int(i)].id for i in faiss_order]

        view = rep.rank(store, q, query_vector=qv)
        assert view.ids == faiss_ids

        # cos = 1 - d2/2 must hold to float tolerance on unit vectors
        cos_from_d2 = (1.0 - d2[faiss_order] / 2.0) * 100.0
        assert np.allclose(view.scores, cos_from_d2, atol=1e-4)


def test_simulate_insert_is_non_mutating_and_appends(embedder):
    """I2: the store is never mutated; `after` is `before` plus one row."""
    store = build_store(random_texts(60, seed=6), embedder)
    before_ids = list(store.ids)
    before_matrix = store.matrix.copy()

    cand = Candidate(id="cand:new", text="a brand new fact",
                     vector=embedder.encode(["a brand new fact"])[0], version=999)
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)
    eff = rep.simulate_insert(store, cand, ["probe one", "probe two"])

    assert store.ids == before_ids
    assert np.array_equal(store.matrix, before_matrix)
    assert set(eff) == {"before", "after", "before:0", "after:0", "before:1", "after:1"}
    assert len(eff["after"].ids) == len(eff["before"].ids) + 1
    assert set(eff["after"].ids) - set(eff["before"].ids) == {"cand:new"}
    assert eff["before"] is eff["before:0"] and eff["after"] is eff["after:0"]


def test_simulate_insert_fast_path_equals_rebuild(embedder):
    """The O(d) dot-product append must give the same ranking as rebuilding the
    whole bank matrix. If this ever fails on the GPU box (different BLAS), use
    ``exact_rebuild=True`` and pay the copy."""
    store = build_store(random_texts(150, seed=7), embedder)
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)
    probes = random_texts(25, seed=8, prefix="probe")

    for i, p in enumerate(probes):
        cand = Candidate(id=f"cand:{i}", text=f"candidate text {i}",
                         vector=embedder.encode([f"candidate text {i}"])[0], version=1000 + i)
        fast = rep.simulate_insert(store, cand, [p], exact_rebuild=False)
        slow = rep.simulate_insert(store, cand, [p], exact_rebuild=True)
        assert fast["after"].ids == slow["after"].ids
        assert np.allclose(fast["after"].scores, slow["after"].scores, atol=1e-5)


def test_empty_store_and_missing_vector(embedder):
    rep = NumpyCosineReplica(embedder, top_k=TOP_K)
    empty = StoreView.from_records([])
    view = rep.rank(empty, "anything")
    assert view.ids == [] and len(view.scores) == 0

    with pytest.raises(ValueError, match="no vector"):
        rep.simulate_insert(empty, Candidate(id="c", text="t", vector=None), ["p"])


def test_kendall_tau_endpoints():
    a = ["a", "b", "c", "d"]
    assert kendall_tau(a, a) == pytest.approx(1.0)
    assert kendall_tau(a, list(reversed(a))) == pytest.approx(-1.0)
