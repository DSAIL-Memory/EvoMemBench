"""Retriever replicas.  [T3 / M0]

A replica reproduces the benchmark's *own* ranking exactly, so that H-Nav can
see the FULL pre-truncation ranking and can simulate a provisional insert
without touching the native store. If a replica is not faithful, then
``rank_self``, ``margin``, ``dH_self``, ``dH_neighbor`` and ``churn`` are all
invalid and the mechanisms that depend on them must be dropped, not silently
re-based on a different retriever (protocol §M0, STOP condition S1).

Two replicas, one per arena:

``NumpyCosineReplica``
    Targets ``Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/
    qwen3_embedding_memory.py:218``. The native computation is two lines::

        scores = (bank_matrix @ query_vec) * 100.0
        top_indices = np.argsort(scores)[::-1][:k]

    Both are reproduced verbatim below, so this replica is bit-exact given the
    same bank matrix — which is persisted at ``<memory_dir>/embeddings.jsonl``.

``FaissFlatReplica``
    Targets ``In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/
    embedding_retriever.py:185`` (LangChain FAISS, ``IndexFlatL2`` over
    L2-normalized embeddings). Native access goes through
    ``similarity_search_with_score_by_vector``; a pure-numpy emulation of the
    same index is provided so the logic is testable without faiss installed.

**Tie behaviour is documented, not silently different.** See ``TIE_NOTE``.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .types import Candidate, RetrievalView, StoreView

__all__ = ["RetrieverReplica", "NumpyCosineReplica", "FaissFlatReplica", "TIE_NOTE"]


TIE_NOTE = """\
np.argsort(scores)[::-1] is NOT a stable descending sort. numpy's default
kind='quicksort' (introsort) gives an unspecified order among equal scores, and
reversing it reverses whatever order that was. The native CrossEp-Know retriever
uses exactly this expression, so NumpyCosineReplica uses it too: reproducing the
native tie order matters more than imposing a deterministic one. Consequence:
when two bank vectors are exactly equal, replica and native agree only because
they run identical code on identical input — a re-implementation with
kind='stable' would NOT agree. Tie frequency is reported by M2; on the primary
arena all facts are unique strings, so exact score ties are expected at ~0.

FAISS breaks its own ties by internal heap order, which is not reproducible from
outside the index. FaissFlatReplica's numpy emulation therefore uses a stable
sort and reports the number of tied scores alongside every fidelity figure, so
any disagreement can be attributed rather than absorbed.
"""


class RetrieverReplica(Protocol):
    def rank(self, store: StoreView, query: str) -> RetrievalView: ...

    def simulate_insert(self, store: StoreView, cand: Candidate,
                        probes: Sequence[str]) -> dict[str, RetrievalView]: ...


class _BaseReplica:
    """Shared plumbing: query embedding + provisional-insert simulation."""

    def __init__(self, embedder, top_k: int = 10) -> None:
        self.embedder = embedder
        self.top_k = top_k

    # ── query vectors ────────────────────────────────────────────────────────
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self.embedder.encode([query])[0], dtype=np.float32)

    # ── scoring hooks, overridden per arena ──────────────────────────────────
    def _scores(self, matrix: np.ndarray, qv: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _score_one(self, vec: np.ndarray, qv: np.ndarray) -> float:
        raise NotImplementedError

    def _order(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ── public API ───────────────────────────────────────────────────────────
    def rank(self, store: StoreView, query: str, query_vector: np.ndarray | None = None,
             top_k: int | None = None) -> RetrievalView:
        """FULL pre-truncation ranking of ``store`` against ``query``."""
        qv = np.asarray(query_vector, dtype=np.float32) if query_vector is not None \
            else self.embed_query(query)
        if len(store) == 0:
            return RetrievalView(query=query, query_vector=qv, ids=[],
                                 scores=np.zeros(0), top_k=top_k or self.top_k)
        scores = self._scores(store.matrix, qv)
        order = self._order(scores)
        ids = store.ids
        return RetrievalView(
            query=query,
            query_vector=qv,
            ids=[ids[i] for i in order],
            scores=scores[order],
            top_k=top_k or self.top_k,
        )

    def rank_scores(self, store: StoreView, query: str,
                    query_vector: np.ndarray | None = None) -> np.ndarray:
        """Scores in *record order* (not ranked). Used by the fidelity harness."""
        qv = np.asarray(query_vector, dtype=np.float32) if query_vector is not None \
            else self.embed_query(query)
        return self._scores(store.matrix, qv)

    def simulate_insert(self, store: StoreView, cand: Candidate,
                        probes: Sequence[str], exact_rebuild: bool = False,
                        ) -> dict[str, RetrievalView]:
        """Rank every probe against the store before and after admitting ``cand``.

        Returns ``{"before": …, "after": …}`` for ``probes[0]`` (by convention the
        candidate's own probe) plus ``{"before:i": …, "after:i": …}`` for every
        probe ``i``. The store is never mutated: the "after" state comes from
        :meth:`StoreView.with_provisional`, which returns a new view.

        ``exact_rebuild=False`` (default) scores the provisional row with a
        single dot product instead of rebuilding the whole bank matrix — an
        O(d) operation instead of an O(Nd) copy, which matters at 18,332 facts.
        ``exact_rebuild=True`` takes the rebuild path;
        ``test_replica_fidelity.py`` asserts the two agree on rankings.
        """
        if cand.vector is None:
            raise ValueError("candidate has no vector; embed before simulating an insert")
        probes = list(probes) or [cand.text]
        cand_vec = np.asarray(cand.vector, dtype=np.float32)
        after_store = store.with_provisional(cand)

        out: dict[str, RetrievalView] = {}
        for i, probe in enumerate(probes):
            qv = self.embed_query(probe)
            before = self.rank(store, probe, query_vector=qv)

            if exact_rebuild:
                after = self.rank(after_store, probe, query_vector=qv)
            else:
                base = self._scores(store.matrix, qv) if len(store) else np.zeros(0)
                scores = np.concatenate([base, [self._score_one(cand_vec, qv)]])
                order = self._order(scores)
                ids = store.ids + [cand.id]
                after = RetrievalView(query=probe, query_vector=qv,
                                      ids=[ids[j] for j in order],
                                      scores=scores[order], top_k=self.top_k)

            out[f"before:{i}"] = before
            out[f"after:{i}"] = after
            if i == 0:
                out["before"], out["after"] = before, after
        return out


class NumpyCosineReplica(_BaseReplica):
    """Bit-exact replica of ``Qwen3EmbeddingMemory.retrieve``.

    Native (``qwen3_embedding_memory.py:218-221``)::

        scores = (bank_matrix @ query_vec) * 100.0
        top_indices = np.argsort(scores)[::-1][:k]

    Both bank and query are L2-normalized upstream, so ``scores`` is
    cosine × 100 — the scale that makes ``H_raw`` degenerate (hard rule 6).
    """

    SCALE = 100.0

    def _scores(self, matrix: np.ndarray, qv: np.ndarray) -> np.ndarray:
        return (matrix @ qv) * self.SCALE

    def _score_one(self, vec: np.ndarray, qv: np.ndarray) -> float:
        return float(np.dot(vec, qv) * self.SCALE)

    def _order(self, scores: np.ndarray) -> np.ndarray:
        # Verbatim native expression, including its unstable tie order. See TIE_NOTE.
        return np.argsort(scores)[::-1]


class FaissFlatReplica(_BaseReplica):
    """Replica of MemoryAgentBench's ``TextRetriever`` (LangChain FAISS flat L2).

    LangChain's ``FAISS.from_documents`` builds an ``IndexFlatL2`` and
    ``similarity_search_with_score_by_vector`` returns **squared L2 distance**,
    ascending. Both shipped embedders (``ContrieverEmbeddings``,
    ``Qwen3Embedding4BEmbeddings``) L2-normalize, so for unit vectors

        d² = 2 − 2·cos   ⟺   cos = 1 − d²/2

    which is an exact, strictly decreasing transform: ranking by ascending d² and
    by descending cosine are the same ranking. This replica therefore reports
    ``cos × 100`` — higher is better, and on the same scale as the other arena,
    so M2's score-scale analysis is comparable across arenas. Raw distances are
    available from :meth:`native_distances` for the fidelity check.
    """

    SCALE = 100.0

    def _scores(self, matrix: np.ndarray, qv: np.ndarray) -> np.ndarray:
        return (matrix @ qv) * self.SCALE

    def _score_one(self, vec: np.ndarray, qv: np.ndarray) -> float:
        return float(np.dot(vec, qv) * self.SCALE)

    def _order(self, scores: np.ndarray) -> np.ndarray:
        # FAISS's own tie order is not reproducible from outside the index; a
        # stable sort is used and tie counts are reported. See TIE_NOTE.
        return np.argsort(-scores, kind="stable")

    def native_distances(self, store: StoreView, qv: np.ndarray) -> np.ndarray:
        """Squared L2 distances in record order — what FAISS actually returns."""
        m = store.matrix
        return (np.linalg.norm(m, axis=1) ** 2
                - 2.0 * (m @ qv)
                + float(np.dot(qv, qv)))

    @staticmethod
    def from_langchain(vectorstore, embedder, top_k: int = 10) -> "FaissFlatReplica":
        """Attach to a live LangChain FAISS store (GPU path, T4+).

        Only used to *verify* the replica against the native index; the replica
        itself never queries FAISS during signal computation.
        """
        rep = FaissFlatReplica(embedder, top_k=top_k)
        rep.vectorstore = vectorstore  # type: ignore[attr-defined]
        return rep

    def native_rank(self, query_vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        """``(page_content, distance)`` straight from the native index."""
        vs = getattr(self, "vectorstore", None)
        if vs is None:
            raise RuntimeError("no LangChain vectorstore attached; use from_langchain()")
        scored = vs.similarity_search_with_score_by_vector(list(query_vector), k=k)
        return [(d.page_content, float(s)) for d, s in scored]


def top_k_agreement(a: Sequence[str], b: Sequence[str], k: int) -> bool:
    """Exact top-k identity — order included. The M0 criterion."""
    return list(a[:k]) == list(b[:k])


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    """Kendall τ between two full rankings of the same id set (protocol §M0)."""
    pos_b = {x: i for i, x in enumerate(b)}
    seq = [pos_b[x] for x in a if x in pos_b]
    n = len(seq)
    if n < 2:
        return float("nan")
    conc = disc = 0
    arr = np.asarray(seq)
    for i in range(n - 1):
        d = arr[i + 1:] - arr[i]
        conc += int((d > 0).sum())
        disc += int((d < 0).sum())
    return (conc - disc) / (0.5 * n * (n - 1))
