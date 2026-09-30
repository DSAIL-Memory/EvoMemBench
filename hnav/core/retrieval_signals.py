"""Retrieval-side signals.  [T5]

Everything here is pure numpy over a ranking, so all of it is verifiable without
a GPU, a model, or a network — see ``hnav/tests/test_retrieval_signals.py``,
where every quantity is checked against a hand-computed value on fixtures with
known answers (orthonormal bases, deliberate ties, duplicated rows).

**Hard rule 6.** ``H_raw`` — the softmax entropy of the *raw* scores — is
computed and logged, and is barred from every decision path. Native scores are
``cosine × 100``, so a raw softmax saturates: at that scale a 0.05 cosine gap is
5 logits, and the distribution collapses onto the top item regardless of how
ambiguous the neighbourhood actually is. ``H_z`` (z-scored) is primary and
``H_vn`` (von Neumann, geometry-aware) is the secondary variant.
:meth:`RetrievalSignalSet.for_policy` is the only accessor a policy may use and
it does not contain ``H_raw``; ``hnav/tests/test_no_raw_entropy_in_policy.py``
enforces that.

M2 reports whether the degeneracy actually holds on this benchmark. A refutation
would revise the prior BFCL finding and is just as publishable as a confirmation,
so the measurement is made either way rather than assumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .types import RetrievalView

__all__ = ["RetrievalSignalSet", "DeltaSignalSet", "RetrievalSignals",
           "softmax", "entropy", "POLICY_FORBIDDEN"]

# Signals that may be logged but must never reach a policy.
POLICY_FORBIDDEN = ("H_raw",)

_EPS = 1e-12


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in **nats**. ``0 log 0 == 0``."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


@dataclass
class RetrievalSignalSet:
    """One ranking's worth of signals. All entropies are in nats."""

    n_ranked: int = 0
    top_m: int = 0
    rank_self: int | None = None          # 0-based rank of the probe's own item
    top1: float = float("nan")
    top2: float = float("nan")
    margin: float = float("nan")          # top1 - top2
    nmargin: float = float("nan")         # margin / |top1|
    H_raw: float = float("nan")           # LOGGED ONLY — never a decision input
    H_z: float = float("nan")             # primary
    H_vn: float = float("nan")            # von Neumann over the top-m Gram matrix
    eff_size: float = float("nan")        # exp(H_z): effective neighbourhood size
    dispersion: float = float("nan")      # std of the top-m scores
    score_mean: float = float("nan")
    score_std: float = float("nan")
    n_ties_top_m: int = 0                 # top-m scores that are not unique

    def to_dict(self) -> dict:
        return asdict(self)

    def for_policy(self) -> dict:
        """The ONLY view a policy may consume. ``H_raw`` is absent by design."""
        return {k: v for k, v in asdict(self).items() if k not in POLICY_FORBIDDEN}


@dataclass
class DeltaSignalSet:
    """The effect of a provisional insert on a ranking."""

    dH_self: float = float("nan")
    dH_neighbor: float = float("nan")
    churn_at_k: int = 0                   # pre-existing ids displaced from top-k
    churn_frac: float = float("nan")
    rank_shift: float = float("nan")      # mean |rank change| among survivors
    rank_self_after: int | None = None
    displaced_ids: list[str] = field(default_factory=list)
    k: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def for_policy(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in POLICY_FORBIDDEN}


class RetrievalSignals:
    """Computes :class:`RetrievalSignalSet` / :class:`DeltaSignalSet`.

    ``top_m`` is the neighbourhood depth for the entropy family (default 50, per
    the protocol's "m≈50"); ``k`` is the churn horizon.
    """

    def __init__(self, top_m: int = 50, k: int = 10) -> None:
        self.top_m = int(top_m)
        self.k = int(k)

    # ── single ranking ───────────────────────────────────────────────────────
    def compute(self, view: RetrievalView, self_id: str | None = None,
                vectors: np.ndarray | None = None) -> RetrievalSignalSet:
        """``vectors`` are the top-m item vectors, in ranked order; supplying
        them enables ``H_vn``. Without them ``H_vn`` stays NaN rather than being
        silently approximated by something else."""
        s = np.asarray(view.scores, dtype=np.float64)
        n = s.size
        if n == 0:
            return RetrievalSignalSet(n_ranked=0, top_m=0)

        m = min(self.top_m, n)
        head = s[:m]

        mean = float(s.mean())
        std = float(s.std())
        # A degenerate store (every score identical) has no scale to z-score by;
        # falling back to 1.0 makes H_z the uniform entropy, which is the honest
        # answer for "every item is equally good".
        z = (head - mean) / (std if std > _EPS else 1.0)

        top1 = float(s[0])
        top2 = float(s[1]) if n > 1 else float("nan")
        margin = top1 - top2 if n > 1 else float("nan")

        H_z = entropy(softmax(z))
        sig = RetrievalSignalSet(
            n_ranked=n,
            top_m=m,
            rank_self=view.rank_of(self_id) if self_id is not None else None,
            top1=top1,
            top2=top2,
            margin=margin,
            nmargin=float(margin / (abs(top1) + _EPS)) if n > 1 else float("nan"),
            H_raw=entropy(softmax(head)),
            H_z=H_z,
            H_vn=self.von_neumann(vectors) if vectors is not None else float("nan"),
            eff_size=float(np.exp(H_z)),
            dispersion=float(head.std()),
            score_mean=mean,
            score_std=std,
            n_ties_top_m=int(m - len(np.unique(head))),
        )
        return sig

    @staticmethod
    def von_neumann(vectors: np.ndarray) -> float:
        """von Neumann entropy of the normalized Gram matrix of the top-m items.

        ``G = V Vᵀ / m`` has unit trace for L2-normalized rows, so its
        eigenvalues are a probability distribution; ``H_vn = −Σ λ ln λ``. It is
        maximal (``ln m``) when the retrieved neighbourhood is orthogonal — many
        genuinely different things — and near 0 when every retrieved item points
        the same way. That is the distinction a score-based entropy cannot make.
        """
        v = np.asarray(vectors, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] == 0:
            return float("nan")
        m = v.shape[0]
        gram = (v @ v.T) / m
        eig = np.linalg.eigvalsh(gram)
        eig = np.clip(eig, 0.0, None)
        total = eig.sum()
        if total <= _EPS:
            return float("nan")
        return entropy(eig / total)

    # ── provisional insert ───────────────────────────────────────────────────
    def compute_delta(self, before: RetrievalView, after: RetrievalView,
                      self_id: str | None = None, k: int | None = None,
                      neighbors: Sequence[tuple[RetrievalView, RetrievalView]] = (),
                      ) -> DeltaSignalSet:
        """Signals across a simulated insert.

        ``dH_self`` is the change in ``H_z`` on the candidate's own probe;
        ``dH_neighbor`` is the mean change over neighbour probes. ``churn_at_k``
        counts *pre-existing* ids pushed out of the top-k — the newly inserted
        item is excluded, since its arrival is not itself churn.
        """
        k = int(k or self.k)
        b = self.compute(before)
        a = self.compute(after, self_id=self_id)

        before_top = list(before.ids[:k])
        after_top = set(after.ids[:k])
        displaced = [i for i in before_top if i not in after_top]

        pos_before = {x: i for i, x in enumerate(before.ids)}
        shifts = [abs(pos_before[x] - j) for j, x in enumerate(after.ids[:k])
                  if x in pos_before]

        dH_n = float("nan")
        if neighbors:
            deltas = [self.compute(nb_after).H_z - self.compute(nb_before).H_z
                      for nb_before, nb_after in neighbors]
            dH_n = float(np.mean(deltas)) if deltas else float("nan")

        return DeltaSignalSet(
            dH_self=a.H_z - b.H_z,
            dH_neighbor=dH_n,
            churn_at_k=len(displaced),
            churn_frac=len(displaced) / k if k else float("nan"),
            rank_shift=float(np.mean(shifts)) if shifts else 0.0,
            rank_self_after=a.rank_self,
            displaced_ids=displaced,
            k=k,
        )
