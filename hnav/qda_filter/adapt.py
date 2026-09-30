"""Stage 7 — label-free adaptation to a target store.  [QDA]

Three components, each usable without a single target label, each gated (G6):

  adapt_nuisance     re-centers and re-scales the whitener from the target's
                     own high-cosine pairs (approximately null at low
                     prevalence); subspaces, eigenvalues and mu1 stay frozen.
  estimate_prevalence single-parameter EM for the conflict rate pi under
                     p(s) = (1-pi) f0(s) + pi f1(s) with f0/f1 frozen from
                     the calibration subset.
  local_coherence    the additive-model signature that needs no fit at all:
                     same-transition edits across subjects are parallel
                     (measured 0.86 mean cosine), so the top eigenvalue of a
                     neighborhood Gram of unit difference vectors separates
                     edits from diffuse subject swaps.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ── 7.1 nuisance re-estimation ───────────────────────────────────────────────
def adapt_nuisance(b, W0: np.ndarray, target_mask: np.ndarray,
                   ref_mask: np.ndarray, cos_min: float = 0.85) -> dict:
    """(mu_t, scale) from the target's cos >= cos_min pairs; labels unread.

    Adapted input: z' = (sigma0_ref / sigma0_t) * W0 (d_t - mu_t).
    """
    tgt = np.flatnonzero(target_mask & (b.cos >= cos_min))
    mu_t = b.D_t[tgt].astype(np.float64).mean(axis=0)

    def _scale(idx: np.ndarray, mu: np.ndarray) -> float:
        acc, n = 0.0, 0
        for s in range(0, len(idx), 4096):
            ii = idx[s:s + 4096]
            Z = (b.D_t[ii].astype(np.float64) - mu) @ W0.T
            acc += float(np.einsum("ij,ij->", Z, Z))
            n += len(ii)
        return acc / (n * Z.shape[1])

    ref = np.flatnonzero(ref_mask)
    s_t = _scale(tgt, mu_t)
    s_ref = _scale(ref, np.zeros_like(mu_t))
    return {"mu_t": mu_t, "ratio": float(np.sqrt(s_ref / s_t)),
            "sigma0_t_sq": s_t, "sigma0_ref_sq": s_ref,
            "n_target_null_sample": int(len(tgt)), "cos_min": cos_min}


def whiten_adapted(b, W0: np.ndarray, idx: np.ndarray, ad: dict) -> np.ndarray:
    out = np.empty((len(idx), b.D_t.shape[1]), dtype=np.float64)
    for s in range(0, len(idx), 4096):
        ii = idx[s:s + 4096]
        out[s:s + len(ii)] = ((b.D_t[ii].astype(np.float64) - ad["mu_t"])
                              @ W0.T) * ad["ratio"]
    return out


# ── 7.2 prevalence EM ────────────────────────────────────────────────────────
class PrevalenceEM:
    """f0/f1 frozen as Gaussian KDEs of calibration-subset scores; EM over
    the single mixing parameter pi."""

    def __init__(self, scores_neg: np.ndarray, scores_pos: np.ndarray) -> None:
        from scipy.stats import gaussian_kde

        self.f0 = gaussian_kde(np.asarray(scores_neg, float))
        self.f1 = gaussian_kde(np.asarray(scores_pos, float))

    def estimate(self, scores: np.ndarray, pi0: float = 0.1,
                 max_iter: int = 500, tol: float = 1e-8) -> dict:
        s = np.asarray(scores, float)
        d0 = np.maximum(self.f0(s), 1e-300)
        d1 = np.maximum(self.f1(s), 1e-300)
        pi = float(pi0)
        for it in range(max_iter):
            post = pi * d1 / (pi * d1 + (1 - pi) * d0)
            new = float(post.mean())
            if abs(new - pi) < tol:
                pi = new
                break
            pi = new
        return {"pi_hat": pi, "n_iter": it + 1, "n": int(len(s))}

    def bayes_threshold(self, pi: float, lo: float, hi: float,
                        n_grid: int = 2001) -> float | None:
        """Smallest grid score whose posterior P(conflict|s) >= 0.5."""
        g = np.linspace(lo, hi, n_grid)
        post = (pi * np.maximum(self.f1(g), 1e-300)
                / (pi * np.maximum(self.f1(g), 1e-300)
                   + (1 - pi) * np.maximum(self.f0(g), 1e-300)))
        hits = np.flatnonzero(post >= 0.5)
        return float(g[hits[0]]) if len(hits) else None


def validate_prevalence(em: PrevalenceEM, pos_scores: np.ndarray,
                        neg_scores: np.ndarray,
                        pis=(0.01, 0.05, 0.2, 0.5), n_draw: int = 2000,
                        n_rep: int = 10, seed: int = 0) -> list[dict]:
    """Seeded mixtures with KNOWN pi from the target's labeled scores —
    labels used only to build the validation truth, never inside EM."""
    rng = np.random.default_rng(seed)
    out = []
    lo = float(min(pos_scores.min(), neg_scores.min()))
    hi = float(max(pos_scores.max(), neg_scores.max()))
    for pi in pis:
        ests = []
        for _ in range(n_rep):
            n1 = int(round(pi * n_draw))
            s = np.concatenate([
                rng.choice(pos_scores, size=n1, replace=True),
                rng.choice(neg_scores, size=n_draw - n1, replace=True)])
            ests.append(em.estimate(s)["pi_hat"])
        out.append({"pi_true": pi,
                    "pi_hat_mean": float(np.mean(ests)),
                    "pi_hat_sd": float(np.std(ests)),
                    "bayes_threshold_at_mean":
                        em.bayes_threshold(float(np.mean(ests)), lo, hi),
                    "n_draw": n_draw, "n_rep": n_rep})
    return out


# ── 7.3 local coherence ──────────────────────────────────────────────────────
def local_coherence(M_z: np.ndarray, D_hat: np.ndarray, K: int = 16,
                    chunk: int = 512) -> np.ndarray:
    """lambda_max(G)/K over each pair's K nearest neighbors (self excluded)
    by whitened-midpoint distance; G = Gram of the neighbors' unit
    difference vectors. Sign-invariant: flipping any d_hat conjugates G by a
    diagonal +-1 matrix and leaves the spectrum unchanged."""
    M = np.asarray(M_z, dtype=np.float32)
    D = np.asarray(D_hat, dtype=np.float32)
    n = len(M)
    sq = np.einsum("ij,ij->i", M, M)
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = sq[s:e, None] - 2.0 * (M[s:e] @ M.T) + sq[None, :]
        for r in range(e - s):
            d2[r, s + r] = np.inf                # self excluded
        nn = np.argpartition(d2, K, axis=1)[:, :K]
        for r in range(e - s):
            Dn = D[nn[r]].astype(np.float64)
            G = Dn @ Dn.T
            out[s + r] = float(np.linalg.eigvalsh(G)[-1]) / K
    return out


def rank_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of within-task ranks — the preregistered fusion rule."""
    from scipy.stats import rankdata

    return 0.5 * (rankdata(a) + rankdata(b))
