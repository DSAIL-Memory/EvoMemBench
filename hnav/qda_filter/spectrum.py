"""Stage 2 — non-conflict whitening and the gold spectrum.  [QDA]

The whitener is fit on half A of the fit-split negatives (Ledoit-Wolf, so a
2432-dim covariance from ~3.9k samples is invertible without hand-tuning a
ridge), and the conflict-class structure is read off the eigenvalues of the
whitened gold covariance: eigenvalues > 1 are directions where conflicts have
EXCESS variance relative to non-conflicts (object-edit directions),
eigenvalues < 1 are directions where negatives have the excess (subject-swap
directions). Rank selection is by parallel analysis — a label-permutation
null, not a fixed cutoff.

Rank-deficiency rule (preregistered): with n1 = 989 gold edits in N' = 2432
dimensions the gold sample spectrum has at most n1 - 1 nontrivial
eigenvalues; real and permuted draws share that count exactly, so every
spectrum comparison here runs over the nontrivial part only. The structural
zeros beyond the rank say nothing about the population and are excluded from
the envelopes, from k_subj, and from sigma1^2.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def ledoit_wolf_whitener(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """(W0, info): W0 = Sigma_LW^{-1/2} restricted to the identifiable span.

    Difference vectors of a finite fact set span at most n_facts - 1
    dimensions, so the sample covariance has a genuine null space no matter
    how many PAIRS were drawn (the G1 smoke diagnosis: 2,190 calibration
    facts < N' = 2432). Ledoit-Wolf fills that null space with the shrinkage
    prior alpha*mu, and a naive Sigma^{-1/2} then amplifies it by
    1/sqrt(alpha*mu) — ~440x in the smoke run — turning the out-of-span
    energy of every held-out pair into dominant noise. The QDA weights are
    unidentified there (no calibration negative ever moved along those
    directions), so the fix is to give them weight 0: eigendirections whose
    LW eigenvalue equals the shrinkage floor (pure prior, zero data
    variance; relative tolerance 1e-6) are projected out of W0. See the
    PREREG addendum.
    """
    from sklearn.covariance import LedoitWolf

    X = np.asarray(X, dtype=np.float64)
    lw = LedoitWolf(assume_centered=False).fit(X)
    evals, evecs = np.linalg.eigh(lw.covariance_)
    assert evals.min() > 0, "LW covariance must be positive definite"
    Xc = X - X.mean(axis=0)
    mu = float((Xc ** 2).sum()) / (X.shape[0] * X.shape[1])  # sklearn's mu
    # fix B (PREREG addendum A): never amplify past the LW target scale mu —
    # eigenvalues below the bulk are span/sample artifacts of the fragmented
    # pair graph, so whitening only DAMPS well-estimated strong directions
    capped = np.maximum(evals, mu)
    W0 = (evecs * (1.0 / np.sqrt(capped))) @ evecs.T
    return W0, {"shrinkage": float(lw.shrinkage_),
                "n_fit": int(X.shape[0]),
                "lw_target_mu": mu,
                "n_capped_at_mu": int((evals < mu).sum()),
                "n_above_mu": int((evals >= mu).sum()),
                "eig_min": float(evals.min()),
                "eig_max": float(evals.max()),
                "max_damping": float(np.sqrt(evals.max() / mu))}


def nontrivial_spectrum(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(eigenvalues desc, eigenvectors as rows) of the ddof-1 covariance of
    the rows of Z, restricted to the nontrivial rank min(n-1, dim)."""
    Z = np.asarray(Z, dtype=np.float64)
    n, dim = Z.shape
    C = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    r = min(n - 1, dim)
    lam = (s[:r] ** 2) / (n - 1)
    return lam, vt[:r]


def _gram_spectrum(G_pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Nontrivial covariance eigenvalues of the rows selected by ``idx``,
    from a precomputed pooled Gram — the O(n1^3) inner loop of the
    permutation null (identical, by the kernel trick, to an SVD of the
    centered data rows)."""
    n1 = len(idx)
    Gs = G_pool[np.ix_(idx, idx)]
    rm = Gs.mean(axis=0)
    Gc = Gs - rm[None, :] - rm[:, None] + rm.mean()
    ev = np.linalg.eigvalsh(Gc) / (n1 - 1)
    ev = np.sort(ev)[::-1]
    return np.clip(ev[: n1 - 1], 0.0, None)  # nontrivial part


def parallel_analysis(Z_gold: np.ndarray, Z_neg_pool: np.ndarray,
                      n_perm: int, seed: int) -> dict:
    """Permutation-null envelopes for the gold spectrum.

    Pools gold with the half-B negatives, draws ``n_gold`` pseudo-gold rows
    ``n_perm`` times, and records the 95th percentile of each i-th largest
    and the 5th percentile of each i-th smallest nontrivial eigenvalue.
    """
    Zp = np.vstack([np.asarray(Z_gold, np.float64),
                    np.asarray(Z_neg_pool, np.float64)])
    n1 = Z_gold.shape[0]
    G = Zp @ Zp.T
    rng = np.random.default_rng(seed)
    tops = np.empty((n_perm, n1 - 1))
    for p in range(n_perm):
        idx = rng.choice(len(Zp), size=n1, replace=False)
        spec = _gram_spectrum(G, idx)
        # fix C (PREREG addendum A): TRACE-NORMALIZED spectra. The pooled
        # draws are ~80% negatives whose whitened variance dominates gold's
        # at every index, so raw-eigenvalue envelopes sit above the real
        # spectrum by total-variance alone; parallel analysis is a shape
        # comparison and is run on eigenvalue fractions, standard practice.
        tops[p] = spec / max(spec.sum(), _EPS)
    return {"null95_top": np.quantile(tops, 0.95, axis=0),
            "null05_bot": np.quantile(tops[:, ::-1], 0.05, axis=0),
            "null_mean_top1": float(tops[:, 0].mean()),
            "normalization": "trace",
            "n_perm": n_perm}


def select_ranks(lam: np.ndarray, null95_top: np.ndarray,
                 null05_bot: np.ndarray, cap_obj: int = 64,
                 cap_subj: int = 512) -> tuple[int, int]:
    """k_obj / k_subj per the preregistered rule, on nontrivial spectra."""
    r = len(lam)
    k_obj = 0
    for i in range(min(cap_obj, r)):
        if lam[i] > null95_top[i]:
            k_obj = i + 1
        else:
            break
    lam_asc = lam[::-1]
    k_subj = 0
    for i in range(min(cap_subj, r - k_obj)):
        if lam_asc[i] < null05_bot[i]:
            k_subj = i + 1
        else:
            break
    return k_obj, k_subj


def fit_spectrum(Z_gold: np.ndarray, Z_neg_pool: np.ndarray, n_perm: int,
                 seed: int, cap_obj: int = 64, cap_subj: int = 512) -> dict:
    """The whole Stage-2 read: spectrum, envelopes, ranks, sigma1^2, bases."""
    lam, vt = nontrivial_spectrum(Z_gold)
    pa = parallel_analysis(Z_gold, Z_neg_pool, n_perm=n_perm, seed=seed)
    lam_frac = lam / max(lam.sum(), _EPS)      # fix C: shape-vs-shape
    k_obj, k_subj = select_ranks(lam_frac, pa["null95_top"],
                                 pa["null05_bot"], cap_obj, cap_subj)
    r = len(lam)
    mid = lam[k_obj: r - k_subj] if r - k_subj > k_obj else lam[k_obj:]
    sigma1sq = float(mid.mean()) if len(mid) else float(lam.mean())
    n1 = Z_gold.shape[0]
    dim = Z_gold.shape[1]
    med = float(np.median(lam))
    return {
        "lam": lam, "vt": vt,
        "k_obj": k_obj, "k_subj": k_subj, "sigma1sq": sigma1sq,
        "U_obj": vt[:k_obj].T,                       # (dim, k_obj)
        "U_subj": vt[r - k_subj:].T if k_subj else np.zeros((dim, 0)),
        "lam_obj": lam[:k_obj],
        "lam_subj": lam[r - k_subj:] if k_subj else lam[:0],
        "null95_top": pa["null95_top"], "null05_bot": pa["null05_bot"],
        "n_perm": pa["n_perm"],
        "mp_edge_reference": {
            "sigma_sq_used": med,
            "top_edge": med * (1 + np.sqrt(dim / n1)) ** 2,
            "note": "reference only; the permutation null is authoritative "
                    "(bottom MP edge is 0 because N' > n1)"},
    }
