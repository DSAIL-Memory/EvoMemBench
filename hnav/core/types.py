"""Core H-Nav types.  [T3]

Field lists follow ``EVOMEMBENCH_HNAV_IMPLEMENTATION_PLAN.md`` §2.1.

These types are the *entire* interface between the adapters and the H-Nav core.
Nothing in ``hnav/core/`` may import from a benchmark; everything benchmark
specific (chunk parsing, serial numbers, prompt formats) lives in an adapter and
arrives here as a :class:`Candidate` or a :class:`StoreView`.

``StoreView.with_provisional`` is non-mutating by construction — that is what
makes ``dH`` and churn computable *before* a write commits (invariant I2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MemoryRecord",
    "Candidate",
    "StoreView",
    "RetrievalView",
    "Decision",
    "WRITE_ACTIONS",
    "READ_ACTIONS",
    "candidate_to_record",
]

# Write-side and read-side action vocabularies. Stage 0 only ever emits PASS,
# but the vocabularies are frozen here so audit records are schema-stable.
WRITE_ACTIONS = ("PASS", "SUPPRESS", "REWRITE", "DEFER")
READ_ACTIONS = ("PASS", "RERANK", "EXPAND", "TRIM", "ANNOTATE")


@dataclass(frozen=True, eq=False)
class MemoryRecord:
    """One admitted memory.

    ``id``      MAB: ``"fact:<serial>"``; CLBench: ``f"{task_id}_chunk{chunk_index}"``
    ``version`` MAB: the fact's serial number — the benchmark's own supersession
                key; CLBench: a monotone write counter.
    ``vector``  L2-normalized, in the native embedding space, or ``None`` when
                the record is only being used for bookkeeping.
    """

    id: str
    text: str
    vector: np.ndarray | None = None
    version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Candidate:
    """A proposed write, before it is admitted.

    ``op`` is always ``"ADD"`` on EvoMemBench: no write path anywhere in the
    benchmark overwrites (analysis §4.1), which is also why the
    ``WRITE_DESTRUCTIVE_OVERWRITE`` label is dropped from the protocol.

    ``prior_id`` / ``prior_text`` are resolved by the adapter using
    ``WriteConflictIndex.latest_before`` — never ``latest`` (hard rule 2).
    """

    id: str
    text: str
    vector: np.ndarray | None = None
    op: str = "ADD"
    prior_id: str | None = None
    prior_text: str | None = None
    version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_prior(self, prior: MemoryRecord | None) -> "Candidate":
        if prior is None:
            return self
        return replace(self, prior_id=prior.id, prior_text=prior.text)


def candidate_to_record(cand: Candidate) -> MemoryRecord:
    return MemoryRecord(id=cand.id, text=cand.text, vector=cand.vector,
                        version=cand.version, metadata=cand.metadata)


@dataclass
class StoreView:
    """An immutable-by-convention snapshot of a memory store.

    ``matrix`` is the (N, d) L2-normalized bank matrix in *record order*; it is
    built lazily from the record vectors and cached. Records carrying no vector
    are not permitted once ``matrix`` is requested — the adapter must embed
    first, so that row *i* of the matrix is always record *i*.
    """

    records: list[MemoryRecord] = field(default_factory=list)
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _index: dict[str, int] | None = field(default=None, repr=False)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_records(cls, records: Iterable[MemoryRecord]) -> "StoreView":
        return cls(records=list(records))

    def with_provisional(self, cand: Candidate) -> "StoreView":
        """Return a NEW view with ``cand`` appended. Never mutates ``self``.

        Appending is the correct simulation of the native behaviour on both
        arenas: the store is append-only, so the provisional state differs from
        the committed state by exactly one trailing row.
        """
        rec = candidate_to_record(cand)
        new_records = self.records + [rec]
        matrix = None
        if self._matrix is not None and rec.vector is not None:
            matrix = np.vstack([self._matrix, rec.vector.reshape(1, -1).astype(self._matrix.dtype)])
        return StoreView(records=new_records, _matrix=matrix, _index=None)

    def without(self, record_id: str) -> "StoreView":
        """Return a NEW view with ``record_id`` dropped — the ``S_without`` of
        the write-side counterfactual (plan §7.2). Because the retriever is a
        pure function of the store, this is the whole counterfactual: no re-run
        of the memorize phase is needed."""
        keep = [r for r in self.records if r.id != record_id]
        return StoreView(records=keep)

    # ── access ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.records:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                missing = [r.id for r in self.records if r.vector is None]
                if missing:
                    raise ValueError(
                        f"{len(missing)} record(s) without a vector, first={missing[0]!r}; "
                        "embed before requesting the bank matrix"
                    )
                self._matrix = np.stack(
                    [np.asarray(r.vector, dtype=np.float32) for r in self.records])
        return self._matrix

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.records]

    def index_of(self, record_id: str) -> int | None:
        if self._index is None:
            self._index = {r.id: i for i, r in enumerate(self.records)}
        return self._index.get(record_id)

    def get(self, record_id: str) -> MemoryRecord | None:
        i = self.index_of(record_id)
        return self.records[i] if i is not None else None


@dataclass
class RetrievalView:
    """A FULL pre-truncation ranking.

    ``ids``/``scores`` are aligned and cover the whole store, not just the page
    the benchmark will use. ``top_k`` records where the benchmark truncates, so
    signals computed over the full ranking can still be related to what the
    model actually saw.
    """

    query: str
    query_vector: np.ndarray | None
    ids: list[str]
    scores: np.ndarray
    top_k: int

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if len(self.ids) != len(self.scores):
            raise ValueError(f"ids ({len(self.ids)}) and scores ({len(self.scores)}) misaligned")

    @property
    def top_ids(self) -> list[str]:
        return self.ids[: self.top_k]

    @property
    def top_scores(self) -> np.ndarray:
        return self.scores[: self.top_k]

    def rank_of(self, record_id: str) -> int | None:
        """0-based rank in the full ranking, or ``None`` if absent."""
        try:
            return self.ids.index(record_id)
        except ValueError:
            return None


@dataclass
class Decision:
    """The output of a policy.

    ``shadow=True`` is defence in depth: in shadow mode the core still computes
    and logs a full decision (so it is analyzable), but the adapter asserts
    ``shadow is True`` and discards ``action``. No Stage-0 code path may act on
    a decision.
    """

    action: str = "PASS"
    payload: Any = None
    reasons: dict = field(default_factory=dict)
    shadow: bool = True

    def with_reason(self, key: str, value: Any) -> "Decision":
        self.reasons[key] = value
        return self


def as_float_list(x: Sequence[float] | np.ndarray) -> list[float]:
    """JSON-safe conversion for audit records."""
    return [float(v) for v in np.asarray(x).ravel()]
