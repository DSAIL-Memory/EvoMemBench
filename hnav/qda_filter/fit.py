"""Stage 3 + Stage 4 — the ordered-term test and the score variants.  [QDA]

The score is the Gaussian log-likelihood ratio in coordinates whitened by the
non-conflict covariance (so Sigma_0 = I there):

    core(z)    = 1/2 [ sum_{i<=k_obj} (1 - 1/lam_i)(u_i.z)^2
                     + sum_{i in subj} (1 - 1/lam_i)(u_i.z)^2      (< 0)
                     + (1 - 1/sigma1^2) ||z_perp||^2 ]
    ordered(z) = (mu1/sigma1^2 - mu0) . z, restricted to span(U_obj)
    norm(d)    = log ||d_t||

CES is core() with the eigen-weights quantized to {+1, -1, 0} and the norm
discarded — V1 realizes exactly that quantization so G1 can check the
whitening/rank machinery against the committed screen.

The sign-flip null for the ordered term is exact: "no order information"
means d and -d are exchangeable, so flipping each z's sign uniformly at
random generates the null of ||mean||^2 with no distributional assumption.
"""
from __future__ import annotations

import numpy as np

from .spectrum import fit_spectrum

_EPS = 1e-12


# ── Stage 3 ──────────────────────────────────────────────────────────────────
def signflip_test(Z: np.ndarray, n_flips: int, seed: int) -> dict:
    """T = ||mean(Z)||^2 against the random-sign-flip null."""
    from hnav.geometry_filter.metrics import perm_pvalue

    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    T = float(mu @ mu)
    rng = np.random.default_rng(seed)
    null = np.empty(n_flips)
    for i in range(n_flips):
        s = rng.integers(0, 2, size=len(Z)) * 2 - 1
        m = (Z * s[:, None]).mean(axis=0)
        null[i] = m @ m
    return {"T": T, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "null_max": float(null.max()),
            "p_signflip": perm_pvalue(T, null), "n_flips": n_flips,
            "n_samples": int(len(Z))}


# ── Stage 4 ──────────────────────────────────────────────────────────────────
class QDAModel:
    """Frozen parameters + the closed-form score terms.

    Everything is a function of ``z = W0 Q d`` (and ``||Q d||`` for the norm
    term); the caller supplies z so eval code can batch/whiten once.
    """

    def __init__(self, W0: np.ndarray, spec: dict, mu1: np.ndarray,
                 mu0: np.ndarray, ordered_on: bool) -> None:
        self.W0 = W0
        self.U_obj = spec["U_obj"]
        self.U_subj = spec["U_subj"]
        self.lam_obj = spec["lam_obj"]
        self.lam_subj = spec["lam_subj"]
        self.k_obj = spec["k_obj"]
        self.k_subj = spec["k_subj"]
        self.sigma1sq = spec["sigma1sq"]
        self.mu1 = mu1
        self.mu0 = mu0
        self.ordered_on = bool(ordered_on)
        c = mu1 / self.sigma1sq - mu0
        # restricted to the object subspace: project the coefficient vector
        self.ordered_coef = self.U_obj @ (self.U_obj.T @ c)
        self.w_obj = 1.0 - 1.0 / np.maximum(self.lam_obj, _EPS)
        self.w_subj = 1.0 - 1.0 / np.maximum(self.lam_subj, _EPS)
        self.w_perp = 1.0 - 1.0 / max(self.sigma1sq, _EPS)

    def core(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        po = Z @ self.U_obj                       # (n, k_obj)
        ps = Z @ self.U_subj                      # (n, k_subj)
        sq = np.einsum("ij,ij->i", Z, Z)
        perp = sq - np.einsum("ij,ij->i", po, po) - np.einsum("ij,ij->i", ps, ps)
        return 0.5 * ((po ** 2) @ self.w_obj + (ps ** 2) @ self.w_subj
                      + self.w_perp * perp)

    def ordered(self, Z: np.ndarray) -> np.ndarray:
        if not self.ordered_on:
            return np.zeros(len(Z))
        return np.asarray(Z, np.float64) @ self.ordered_coef

    @staticmethod
    def v1_quantized(D_t_hat: np.ndarray, U_obj: np.ndarray,
                     U_subj: np.ndarray) -> np.ndarray:
        """CES's quantization: obj-energy minus subj-energy of the UNIT
        ABTT-complement difference (weights in {+1, -1, 0}, norm dropped)."""
        po = D_t_hat @ U_obj
        ps = D_t_hat @ U_subj
        return (np.einsum("ij,ij->i", po, po)
                - np.einsum("ij,ij->i", ps, ps))


def fit_beta_norm(score_fixed: np.ndarray, lognorm: np.ndarray,
                  y: np.ndarray) -> dict:
    """beta for V4 = coef(lognorm)/coef(score) from a 2-feature logistic on
    the fit split — the preregistered rule."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([score_fixed, lognorm])
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    lr.fit(X, y.astype(int))
    c_score, c_norm = float(lr.coef_[0][0]), float(lr.coef_[0][1])
    beta = c_norm / c_score if abs(c_score) > _EPS else 0.0
    return {"beta": beta, "coef_score": c_score, "coef_norm": c_norm}


def fit_v5(core: np.ndarray, ordered: np.ndarray, lognorm: np.ndarray,
           y: np.ndarray, seed: int = 0) -> tuple[object, dict]:
    """<= 4-parameter recalibration: logistic on [core, ordered, norm], 5-fold
    CV on the fit split for the honest fit-side number, then a full refit
    whose coefficients are the frozen V5."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from hnav.geometry_filter.metrics import auroc

    X = np.column_stack([core, ordered, lognorm])
    yy = y.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = np.empty(len(y))
    for tr, te in skf.split(X, yy):
        m = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        m.fit(X[tr], yy[tr])
        cv_scores[te] = m.decision_function(X[te])
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    lr.fit(X, yy)
    info = {"cv_auroc_fit_split": auroc(cv_scores[yy == 1], cv_scores[yy == 0]),
            "coef": [float(c) for c in lr.coef_[0]],
            "intercept": float(lr.intercept_[0]), "n_params": 4}
    return lr, info


# ── Stage 4b — relation-gated mixture ────────────────────────────────────────
class RelationMixture:
    """logsumexp_r [log p(r|m) + s_r(d)] with per-relation shrunk gold
    covariances (Woodbury against the structured pooled model).

    The eval pair's own relation identity is never read — the gate infers it
    from the whitened midpoint, which is what makes the relation-disjoint
    evaluation meaningful.
    """

    def __init__(self, model: QDAModel, min_rel_gold: int) -> None:
        self.model = model
        self.min_rel_gold = min_rel_gold
        self.rel_low: dict[str, dict] = {}       # relation -> Woodbury pieces
        self.classes: list[str] = []
        self.gate = None
        self.gate_kind = None
        self.gate_cv_acc: dict[str, float] = {}
        self.mu = None                            # mean used in the Gaussians

    # pooled structured Sigma1: apply inverse and logdet in closed form
    def _pooled_inv_apply(self, Y: np.ndarray) -> np.ndarray:
        m = self.model
        po = Y @ m.U_obj
        ps = Y @ m.U_subj
        out = (Y - po @ m.U_obj.T - ps @ m.U_subj.T) / m.sigma1sq
        out += (po / m.lam_obj) @ m.U_obj.T
        if m.k_subj:
            out += (ps / m.lam_subj) @ m.U_subj.T
        return out

    def _pooled_logdet(self, dim: int) -> float:
        m = self.model
        return (float(np.log(m.lam_obj).sum())
                + float(np.log(m.lam_subj).sum())
                + (dim - m.k_obj - m.k_subj) * np.log(m.sigma1sq))

    def fit(self, Z_gold: np.ndarray, rel_gold: list, Z_gate: np.ndarray,
            rel_gate: list, seed: int = 0) -> "RelationMixture":
        from collections import defaultdict

        m = self.model
        dim = Z_gold.shape[1]
        self.mu = (m.U_obj @ (m.U_obj.T @ m.mu1)) if m.ordered_on \
            else np.zeros(dim)
        self._logdet_pooled = self._pooled_logdet(dim)

        by_rel = defaultdict(list)
        for i, r in enumerate(rel_gold):
            if r is not None:
                by_rel[r].append(i)
        thresh = max(50, 3 * m.k_obj)
        for r, idx in by_rel.items():
            n_r = len(idx)
            if n_r < thresh:
                continue
            alpha = dim / (dim + n_r)
            C = Z_gold[idx] - Z_gold[idx].mean(axis=0)
            X = np.sqrt((1 - alpha) / (n_r - 1)) * C   # Sigma_r = aS1 + X^T X
            # Woodbury against B = alpha * Sigma1_pooled
            BinvXt = self._pooled_inv_apply(X) / alpha  # rows: B^-1 x_j
            K = np.eye(n_r) + X @ BinvXt.T
            sign, ld_K = np.linalg.slogdet(K)
            assert sign > 0
            self.rel_low[r] = {
                "alpha": alpha, "n_r": n_r, "X": X, "BinvXt": BinvXt,
                "Kinv": np.linalg.inv(K),
                "logdet": dim * np.log(alpha) + self._logdet_pooled + ld_K,
            }

        # gate on whitened midpoints; classes = relations with >= 5 fit pairs
        from collections import Counter
        cnt = Counter(r for r in rel_gate if r is not None)
        self.classes = sorted(r for r, c in cnt.items() if c >= 5)
        keep = [i for i, r in enumerate(rel_gate) if r in set(self.classes)]
        Xg = np.asarray(Z_gate[keep], dtype=np.float32)
        yg = np.array([rel_gate[i] for i in keep])

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import NearestCentroid

        cands = {"logreg": LogisticRegression(solver="lbfgs", C=1.0,
                                              max_iter=1000),
                 "centroid": NearestCentroid()}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for name, clf in cands.items():
            acc = []
            for tr, te in skf.split(Xg, yg):
                c = cands[name].__class__(**cands[name].get_params())
                c.fit(Xg[tr], yg[tr])
                acc.append(float((c.predict(Xg[te]) == yg[te]).mean()))
            self.gate_cv_acc[name] = float(np.mean(acc))
        self.gate_kind = max(self.gate_cv_acc, key=self.gate_cv_acc.get)
        self.gate = cands[self.gate_kind].fit(Xg, yg)
        return self

    def _log_p_rel(self, M_z: np.ndarray) -> np.ndarray:
        """log p(r|m) rows aligned with self.classes."""
        X = np.asarray(M_z, dtype=np.float32)
        if self.gate_kind == "logreg":
            lp = self.gate.predict_log_proba(X)
            order = [list(self.gate.classes_).index(c) for c in self.classes]
            return lp[:, order]
        # nearest-centroid: softmax over negative squared distances,
        # ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2 without a 3-d broadcast
        C = self.gate.centroids_.astype(np.float32)
        d2 = (np.einsum("ij,ij->i", X, X)[:, None]
              - 2.0 * (X @ C.T)
              + np.einsum("ij,ij->i", C, C)[None, :])
        order = [list(self.gate.classes_).index(c) for c in self.classes]
        lp = -0.5 * d2[:, order].astype(np.float64)
        mx = lp.max(axis=1, keepdims=True)
        return lp - (mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True)))

    def _s_r(self, Zc: np.ndarray, zz: np.ndarray, r: str) -> np.ndarray:
        """log N(z; mu, Sigma_r) - log N(z; 0, I) for one relation."""
        dim = Zc.shape[1]
        if r in self.rel_low:
            p = self.rel_low[r]
            a = self._pooled_inv_apply(Zc) / p["alpha"]
            quad = np.einsum("ij,ij->i", Zc, a)
            t = Zc @ p["BinvXt"].T                # (n, n_r)
            quad -= np.einsum("ij,jk,ik->i", t, p["Kinv"], t)
            logdet = p["logdet"]
        else:
            quad = np.einsum("ij,ij->i", Zc, self._pooled_inv_apply(Zc))
            logdet = self._logdet_pooled
        return 0.5 * zz - 0.5 * quad - 0.5 * logdet

    def score(self, Z: np.ndarray, M_z: np.ndarray,
              chunk: int = 2048) -> np.ndarray:
        from scipy.special import logsumexp

        out = np.empty(len(Z))
        for s in range(0, len(Z), chunk):
            e = min(s + chunk, len(Z))
            Zk = np.asarray(Z[s:e], dtype=np.float64)
            Zc = Zk - self.mu[None, :]
            zz = np.einsum("ij,ij->i", Zk, Zk)
            lp = self._log_p_rel(M_z[s:e])
            S = np.stack([lp[:, j] + self._s_r(Zc, zz, r)
                          for j, r in enumerate(self.classes)], axis=1)
            out[s:e] = logsumexp(S, axis=1)
        return out
