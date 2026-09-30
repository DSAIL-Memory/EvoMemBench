"""Embedding-geometry signals.  [T6]

``sim_max`` and its argmax, QR-residual novelty, the adaptive threshold
``tau_t``, ABTT whitening, and the exact-duplicate fast path.

Two design constraints inherited from the analysis and worth restating:

* **Whitening must refuse to fit on a small store.** ABTT estimates a mean and a
  few principal directions; on the 6k subsets (455 facts) and on early
  CrossEp-Know contexts (<20 chunks) those estimates are noise. The module
  refuses below ``min_fit_n`` (default 200) and falls back to raw cosine,
  recording which path it took so the fallback rate is reportable rather than
  invisible.
* **The QR residual is an upper bound when the basis is subsampled.** Stated in
  the return value (``qr_basis_subsampled``), not buried.

Nothing here decides anything. Stage 0 computes and logs; the policies that
would consume these signals are gated behind the T8 gate and are not written.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .types import Candidate, StoreView

__all__ = ["GeometrySignals", "ABTTWhitening", "TauPolicy", "GeometryModule",
           "qr_residual", "verbatim_values"]

_EPS = 1e-12


@dataclass
class GeometrySignals:
    sim_max: float = float("nan")
    argmax_id: str | None = None
    qr_residual_r: float = float("nan")
    qr_basis_subsampled: bool = False
    tau_t: float = float("nan")
    is_exact_dup: bool = False
    whitened: bool = False
    whitening_refused: bool = False
    store_size: int = 0
    verbatim_new_values: list[str] = field(default_factory=list)
    retrievability_floor_ok: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ── ABTT whitening ───────────────────────────────────────────────────────────
class ABTTWhitening:
    """All-but-the-top: subtract the mean, then project out the top-D principal
    directions, then renormalize.

    Common directions in an embedding space carry no discriminative information
    but dominate cosine similarity; removing them makes ``sim_max`` mean what it
    is supposed to mean. The cost is that both the mean and the directions have
    to be *estimated*, which is why :meth:`fit` refuses on a small store.
    """

    def __init__(self, n_components: int = 3, min_fit_n: int = 200) -> None:
        self.n_components = int(n_components)
        self.min_fit_n = int(min_fit_n)
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.fitted = False
        self.refused = False
        self.n_fit = 0

    def fit(self, matrix: np.ndarray) -> "ABTTWhitening":
        m = np.asarray(matrix, dtype=np.float64)
        self.n_fit = int(m.shape[0]) if m.ndim == 2 else 0
        if m.ndim != 2 or self.n_fit < self.min_fit_n:
            self.fitted, self.refused = False, True
            return self
        self.mean = m.mean(axis=0)
        centered = m - self.mean
        d = min(self.n_components, min(centered.shape) - 1)
        if d < 1:
            self.fitted, self.refused = False, True
            return self
        # economy SVD: right singular vectors are the principal directions
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components = vt[:d]
        self.fitted, self.refused = True, False
        return self

    # -- persistence ---------------------------------------------------------
    # Fitting offline once and shipping (mean, components) as constants is what
    # makes whitening usable at decision time: the read pool is 50 facts, far
    # below ``min_fit_n``, but a *pre-fitted* whitener has nothing left to
    # estimate and simply applies. The fingerprint exists so a consumer can
    # prove the vectors it is scoring were produced by the artifact it thinks
    # they were - the same guarantee the embedding cache namespace gives.
    def to_dict(self) -> dict:
        return {"n_components": self.n_components, "min_fit_n": self.min_fit_n,
                "fitted": self.fitted, "refused": self.refused,
                "n_fit": self.n_fit,
                "mean": None if self.mean is None else self.mean.tolist(),
                "components": None if self.components is None
                else self.components.tolist(),
                "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, d: dict) -> "ABTTWhitening":
        w = cls(int(d["n_components"]), int(d["min_fit_n"]))
        w.fitted, w.refused, w.n_fit = bool(d["fitted"]), bool(d["refused"]), int(d["n_fit"])
        w.mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=np.float64)
        w.components = (None if d.get("components") is None
                        else np.asarray(d["components"], dtype=np.float64))
        stamped = d.get("fingerprint")
        if stamped is not None and stamped != w.fingerprint():
            raise ValueError(
                f"whitening artifact fingerprint mismatch: file says {stamped}, "
                f"loaded parameters hash to {w.fingerprint()}. The artifact is "
                f"corrupt or was written by a different implementation.")
        return w

    def fingerprint(self) -> str | None:
        """sha256 over the exact bytes of (mean, components). None when unfitted."""
        if not self.fitted or self.mean is None or self.components is None:
            return None
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.mean, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.components, dtype=np.float64).tobytes())
        return h.hexdigest()

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Whiten and re-normalize. A pass-through when not fitted."""
        v = np.asarray(vectors, dtype=np.float64)
        single = v.ndim == 1
        if single:
            v = v[None, :]
        if not self.fitted:
            out = v
        else:
            out = v - self.mean
            out = out - (out @ self.components.T) @ self.components
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(norm > _EPS, out / np.maximum(norm, _EPS), out)
        return out[0] if single else out


# ── adaptive threshold ───────────────────────────────────────────────────────
class TauPolicy:
    """``tau_t = clip(mean + z·std)`` over the ``sim_max`` values seen so far.

    Adaptive rather than fixed because ``sim_max`` drifts with store size: in a
    larger store the nearest neighbour of anything is closer, so a constant
    threshold silently tightens as the run proceeds. Until ``min_n``
    observations exist there is nothing to adapt to and ``init_tau`` is used.

    Stateful and order-dependent by design — it models what an online policy
    would have known at time t — so :meth:`update` must be called in
    observation order, and never with a future candidate's value.
    """

    def __init__(self, init_tau: float = 0.85, z: float = 1.0, min_n: int = 50,
                 lo: float = 0.50, hi: float = 0.99) -> None:
        self.init_tau = float(init_tau)
        self.z = float(z)
        self.min_n = int(min_n)
        self.lo, self.hi = float(lo), float(hi)
        self.n = 0
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, sim_max: float) -> None:
        if not np.isfinite(sim_max):
            return
        self.n += 1
        self._sum += float(sim_max)
        self._sumsq += float(sim_max) ** 2

    @property
    def tau_t(self) -> float:
        if self.n < self.min_n:
            return self.init_tau
        mean = self._sum / self.n
        var = max(self._sumsq / self.n - mean * mean, 0.0)
        return float(np.clip(mean + self.z * np.sqrt(var), self.lo, self.hi))


# ── novelty ──────────────────────────────────────────────────────────────────
def qr_residual(vectors: np.ndarray, bank: np.ndarray, max_basis: int = 512,
                seed: int = 0) -> tuple[np.ndarray, bool]:
    """``(residual_norms, basis_was_subsampled)``.

    ``r = ‖v − P_bank v‖`` where ``P_bank`` projects onto the column space of the
    bank. The bank is subsampled to ``max_basis`` columns to keep the QR
    tractable; the residual is then an **upper bound** on true novelty, since a
    smaller span can only leave more of ``v`` unexplained. Same convention as
    T1's ``m1_geometry_calibration.qr_residual``, and the subsampling is
    reported alongside the value.
    """
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    b = np.asarray(bank, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] == 0:
        return np.ones(v.shape[0]), False
    subsampled = b.shape[0] > max_basis
    if subsampled:
        idx = np.random.default_rng(seed).choice(b.shape[0], max_basis, replace=False)
        b = b[idx]
    q, _ = np.linalg.qr(b.T)
    recon = (v @ q) @ q.T
    return np.linalg.norm(v - recon, axis=1), subsampled


# ── verbatim values ──────────────────────────────────────────────────────────
_VALUE_RE = re.compile(r"\b(?:[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*|\d[\d,.]*)\b")


def verbatim_values(text: str) -> set[str]:
    """Value-like spans: capitalized phrases and numbers.

    Deliberately crude and deliberately *recall*-oriented. Its only use is the
    veto in step 2 of the write policy — "does this candidate introduce a value
    the store has not seen?" — where a false positive costs a suppression that
    would have happened anyway, and a false negative destroys an answer. In an
    append-only supersession benchmark the asymmetry is not close.
    """
    return {m.group(0).strip() for m in _VALUE_RE.finditer(text or "") if m.group(0).strip()}


# ── the module ───────────────────────────────────────────────────────────────
class GeometryModule:
    def __init__(self, whitening: ABTTWhitening | None = None,
                 tau_policy: TauPolicy | None = None, max_basis: int = 512) -> None:
        self.whitening = whitening
        self.tau_policy = tau_policy or TauPolicy()
        self.max_basis = max_basis
        self._dup_hashes: set[str] = set()

    @staticmethod
    def text_hash(text: str) -> str:
        """MD5 of the exact string — the only dedup any EvoMemBench write path
        performs (analysis §4.1), reproduced here so ``is_exact_dup`` means what
        the benchmark means by it."""
        return hashlib.md5((text or "").encode()).hexdigest()

    def fit_whitening(self, store: StoreView) -> ABTTWhitening | None:
        """Fit ABTT on the store's bank matrix. Read-only w.r.t. the native
        retriever: whitening is applied to H-Nav's copy of the vectors and never
        to the store the benchmark searches."""
        if self.whitening is None or len(store) == 0:
            return self.whitening
        self.whitening.fit(store.matrix)
        return self.whitening

    def observe(self, text: str) -> None:
        """Record an admitted text so later exact duplicates are detectable."""
        self._dup_hashes.add(self.text_hash(text))

    def compute(self, candidate: Candidate, store: StoreView) -> GeometrySignals:
        sig = GeometrySignals(store_size=len(store))
        sig.is_exact_dup = self.text_hash(candidate.text) in self._dup_hashes

        if candidate.vector is None or len(store) == 0:
            sig.tau_t = self.tau_policy.tau_t
            return sig

        bank = store.matrix.astype(np.float64)
        vec = np.asarray(candidate.vector, dtype=np.float64)

        if self.whitening is not None and self.whitening.fitted:
            bank = self.whitening.transform(bank)
            vec = self.whitening.transform(vec)
            sig.whitened = True
        elif self.whitening is not None:
            sig.whitening_refused = self.whitening.refused

        sims = bank @ vec
        j = int(np.argmax(sims))
        sig.sim_max = float(sims[j])
        sig.argmax_id = store.records[j].id

        r, subsampled = qr_residual(vec, bank, max_basis=self.max_basis)
        sig.qr_residual_r = float(r[0])
        sig.qr_basis_subsampled = subsampled

        # tau_t reflects what was known BEFORE this candidate, then absorbs it.
        sig.tau_t = self.tau_policy.tau_t
        self.tau_policy.update(sig.sim_max)

        prior_text = candidate.prior_text or store.records[j].text
        sig.verbatim_new_values = sorted(
            verbatim_values(candidate.text) - verbatim_values(prior_text))
        sig.retrievability_floor_ok = True   # set by the adapter when a replica exists
        return sig
