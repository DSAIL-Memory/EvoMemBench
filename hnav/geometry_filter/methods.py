"""RCED and RCESP — relation-conditioned edit direction / subspace scoring.

Both operate on the *normalized difference vector* of a candidate pair,
``d_hat = (v_b - v_a) / ||v_b - v_a||``, and both are fit ONLY on calibration
gold-update difference vectors (oriented earlier→later).

RCED    per relation r: ``mu_r = normalize(mean_i d_hat_i)``.
        Score = ``|d_hat · mu_r|``. The absolute value is deliberate: at
        detection time a candidate pair is unordered, so a signed projection
        would leak orientation information that only labeled pairs have.

RCESP   per relation r: top-k right singular vectors of the stacked d_hat
        matrix (uncentered SVD — the subspace should CONTAIN the mean edit
        direction, not remove it). Score = ``||U_r^T d|| / ||d||`` — the
        fraction of the edit that lies inside the learned edit subspace.
        Naturally sign-invariant.

Both carry a *global* variant (all calibration edits pooled) which is what a
relation-disjoint evaluation is allowed to use, and which is the fallback when
a pair's relation was unseen or under-supported at fit time (< ``min_pairs``).
``rced_max`` scores ``max_r |d_hat·mu_r|`` for the no-relation-at-inference
setting. Which path scored each pair is counted, never silent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import PairView

_EPS = 1e-12
MIN_PAIRS_DEFAULT = 5
_CHUNK = 8192


def _chunked(pv: PairView, V: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn(d_chunk) -> 1d scores`` over (v_b - v_a) rows in chunks so a
    50k-pair view never materializes an (n, dim) matrix at once."""
    out = np.empty(len(pv))
    for s in range(0, len(pv), _CHUNK):
        e = min(s + _CHUNK, len(pv))
        d = V[pv.ib[s:e]] - V[pv.ia[s:e]]
        out[s:e] = fn(d)
    return out


class RCED:
    def __init__(self, min_pairs: int = MIN_PAIRS_DEFAULT) -> None:
        self.min_pairs = min_pairs
        self.mu: dict[str, np.ndarray] = {}
        self.mu_global: np.ndarray | None = None

    def fit(self, D_hat: np.ndarray, relations: list[str]) -> "RCED":
        by_rel = defaultdict(list)
        for i, r in enumerate(relations):
            by_rel[r].append(i)
        for r, idx in by_rel.items():
            if len(idx) >= self.min_pairs:
                m = D_hat[idx].mean(axis=0)
                self.mu[r] = m / max(np.linalg.norm(m), _EPS)
        g = D_hat.mean(axis=0)
        self.mu_global = g / max(np.linalg.norm(g), _EPS)
        return self

    def score(self, pv: PairView, V: np.ndarray) -> tuple[np.ndarray, dict]:
        """|d_hat · mu_r| per pair; global-mu fallback where r is unknown."""
        out = np.empty(len(pv))
        n_fallback = 0
        groups = defaultdict(list)
        for i, r in enumerate(pv.relation):
            groups[r if r in self.mu else None].append(i)
        for r, idx in groups.items():
            idx = np.array(idx)
            mu = self.mu[r] if r is not None else self.mu_global
            if r is None:
                n_fallback += len(idx)
            for s in range(0, len(idx), _CHUNK):
                ii = idx[s:s + _CHUNK]
                d = V[pv.ib[ii]] - V[pv.ia[ii]]
                out[ii] = np.abs(d @ mu) / np.maximum(
                    np.linalg.norm(d, axis=1), _EPS)
        return out, {"n_relation_fallback": n_fallback}

    def score_max(self, pv: PairView, V: np.ndarray) -> np.ndarray:
        """max over ALL trained relations — relation identity not used."""
        M = np.stack(list(self.mu.values()))          # (R, dim)
        return _chunked(pv, V, lambda d: np.abs(d @ M.T).max(axis=1)
                        / np.maximum(np.linalg.norm(d, axis=1), _EPS))

    def score_global(self, pv: PairView, V: np.ndarray) -> np.ndarray:
        return _chunked(pv, V, lambda d: np.abs(d @ self.mu_global)
                        / np.maximum(np.linalg.norm(d, axis=1), _EPS))


class RCESP:
    def __init__(self, k: int, min_pairs: int = MIN_PAIRS_DEFAULT,
                 k_global: int | None = None) -> None:
        self.k = int(k)
        self.k_global = int(k_global if k_global is not None else k)
        self.min_pairs = min_pairs
        self.U: dict[str, np.ndarray] = {}            # (dim, k_r)
        self.U_global: np.ndarray | None = None

    @staticmethod
    def _top_components(D_hat: np.ndarray, k: int) -> np.ndarray:
        # uncentered SVD: rows are unit edit vectors, right singular vectors
        # span the directions of maximum edit energy (mean direction included)
        _, _, vt = np.linalg.svd(D_hat, full_matrices=False)
        return vt[: min(k, vt.shape[0])].T            # (dim, k_eff)

    def fit(self, D_hat: np.ndarray, relations: list[str]) -> "RCESP":
        by_rel = defaultdict(list)
        for i, r in enumerate(relations):
            by_rel[r].append(i)
        for r, idx in by_rel.items():
            if len(idx) >= max(self.min_pairs, 2):
                self.U[r] = self._top_components(D_hat[idx], self.k)
        self.U_global = self._top_components(D_hat, self.k_global)
        return self

    def score(self, pv: PairView, V: np.ndarray) -> tuple[np.ndarray, dict]:
        out = np.empty(len(pv))
        n_fallback = 0
        groups = defaultdict(list)
        for i, r in enumerate(pv.relation):
            groups[r if r in self.U else None].append(i)
        for r, idx in groups.items():
            idx = np.array(idx)
            U = self.U[r] if r is not None else self.U_global
            if r is None:
                n_fallback += len(idx)
            for s in range(0, len(idx), _CHUNK):
                ii = idx[s:s + _CHUNK]
                d = V[pv.ib[ii]] - V[pv.ia[ii]]
                out[ii] = np.linalg.norm(d @ U, axis=1) / np.maximum(
                    np.linalg.norm(d, axis=1), _EPS)
        return out, {"n_relation_fallback": n_fallback}

    def score_global(self, pv: PairView, V: np.ndarray) -> np.ndarray:
        return _chunked(pv, V, lambda d: np.linalg.norm(d @ self.U_global, axis=1)
                        / np.maximum(np.linalg.norm(d, axis=1), _EPS))


def fit_training_edits(records, pv_all: PairView, V: np.ndarray,
                       train_mask, dedupe_transitions: bool = False,
                       transition_keys: list | None = None):
    """(D_hat, relations) for the pairs under ``train_mask``.

    ``dedupe_transitions`` keeps one exemplar per oriented (relation, o_a, o_b)
    transition — the fit-side half of a transition-disjoint protocol, so a
    frequent transition cannot dominate mu_r / U_r.
    """
    idx = np.flatnonzero(np.asarray(train_mask))
    if dedupe_transitions:
        assert transition_keys is not None
        seen, keep = set(), []
        for i in idx:
            t = transition_keys[i]
            if t is None or t not in seen:
                seen.add(t)
                keep.append(i)
        idx = np.array(keep)
    sub = pv_all.subset(np.isin(np.arange(len(pv_all)), idx))
    return sub.diff(V, normalize=True, oriented=True), list(sub.relation)
