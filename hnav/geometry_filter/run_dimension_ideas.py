"""Experiment 4 — dimension-level difference-vector detectors vs ABTT-cosine.

The slot probe showed the axis-energy profile |d_hat| identifies which slot
changed (obj-vs-subj AUROC 0.96) but experiment 3 never converted that into a
conflict *score*. The exploratory pass showed why the naive version can't be a
one-liner: no single dimension discriminates (max |Cohen d| = 0.41, best
single-dim AUROC 0.40) — the signal is spread over hundreds of weakly
informative axes. Three detectors that combine dimensions, all fit on
calibration only:

  axis_lr     logistic regression on |d_hat| (2560 dims), cal gold vs cal
              hard negatives. The learned per-dimension weighting.
  ces         contrastive edit subspace: per relation, the object-edit
              subspace U_obj_r (top-20 SVD of gold d_hat — RCESP's subspace)
              AND the subject-edit subspace U_subj_r (top-20 SVD of hard-
              negative d_hat). Score = ||U_obj_r^T d_hat||^2 −
              ||U_subj_r^T d_hat||^2. RCESP used only the positive side;
              this also models what a NON-conflict edit looks like.
              k=20 is inherited from experiment 3, not retuned.
  topdim      energy fraction in a frozen mask of the top-m positive-effect
              dimensions (Cohen d of |d_hat_i|, calibration); m selected on
              calibration from {64, 128, 256, 512, 1024}.

All three are sign-invariant (|d|, squared energies), so pair order cannot
leak. Evaluated exactly like experiment 3: confirmatory hard-negative task
(+ inverted-win vs campaign cosine), balanced sh_64k set (+ 0.87–0.97 band),
transition-/subject-disjoint slices; ABTT-cosine and RCESP comparison rows are
recomputed in-process so every number shares one pipeline.

Usage:  python -m hnav.geometry_filter.run_dimension_ideas
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import data
from .methods import RCESP, fit_training_edits
from .metrics import (auprc, auroc, bootstrap_ci, inverted_win_rate,
                      paired_bootstrap_delta_auc)

K_CES = 20
M_GRID = (64, 128, 256, 512, 1024)
N_BOOT = 1000
_CHUNK = 8192


def _abs_diff(view: data.PairView, V: np.ndarray, s, e) -> np.ndarray:
    d = V[view.ib[s:e]] - V[view.ia[s:e]]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return np.abs(d)


def _chunked_scores(view, V, fn) -> np.ndarray:
    out = np.empty(len(view))
    for s in range(0, len(view), _CHUNK):
        e = min(s + _CHUNK, len(view))
        out[s:e] = fn(_abs_diff(view, V, s, e), s, e)
    return out


class ContrastiveSubspace:
    """Per-relation object-edit minus subject-edit subspace energy."""

    def __init__(self, k: int = K_CES, min_pairs: int = 5) -> None:
        self.k, self.min_pairs = k, min_pairs
        self.U_obj: dict[str, np.ndarray] = {}
        self.U_subj: dict[str, np.ndarray] = {}
        self.U_obj_g: np.ndarray | None = None
        self.U_subj_g: np.ndarray | None = None

    @staticmethod
    def _top(D: np.ndarray, k: int) -> np.ndarray:
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        return vt[: min(k, vt.shape[0])].T

    def fit(self, D_pos, rel_pos, D_neg, rel_neg) -> "ContrastiveSubspace":
        for D, rels, store in ((D_pos, rel_pos, self.U_obj),
                               (D_neg, rel_neg, self.U_subj)):
            by_rel = defaultdict(list)
            for i, r in enumerate(rels):
                by_rel[r].append(i)
            for r, idx in by_rel.items():
                if len(idx) >= self.min_pairs:
                    store[r] = self._top(D[idx], self.k)
        self.U_obj_g = self._top(D_pos, self.k)
        self.U_subj_g = self._top(D_neg, self.k)
        return self

    def score(self, view: data.PairView, V: np.ndarray) -> np.ndarray:
        out = np.empty(len(view))
        groups = defaultdict(list)
        for i, r in enumerate(view.relation):
            groups[r if (r in self.U_obj and r in self.U_subj) else None].append(i)
        for r, idx in groups.items():
            Uo = self.U_obj[r] if r is not None else self.U_obj_g
            Us = self.U_subj[r] if r is not None else self.U_subj_g
            idx = np.array(idx)
            for s in range(0, len(idx), _CHUNK):
                ii = idx[s:s + _CHUNK]
                d = V[view.ib[ii]] - V[view.ia[ii]]
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                out[ii] = (np.linalg.norm(d @ Uo, axis=1) ** 2
                           - np.linalg.norm(d @ Us, axis=1) ** 2)
        return out


def run() -> dict:
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    V = spaces["raw"]
    pv = data.PairView(records, index)

    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    in_eval = np.array([r["in_eval_set"] for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])
    camp_cos = np.array([r["cosine_similarity"] for r in records])
    subset = np.array([r["subset"] for r in records], object)
    trans_keys = [data.transition_key(r) for r in records]
    cal_subjects, cal_transitions = data.calibration_positive_sets(records)

    # ── calibration training material ──────────────────────────────────────
    pv_pos = pv.subset(gold & cal)
    pv_neg = pv.subset(hardneg & cal)
    P = np.abs(pv_pos.diff(V, normalize=True, oriented=False))
    N = np.abs(pv_neg.diff(V, normalize=True, oriented=False))

    # per-dimension effect sizes (Cohen d on |d_hat_i|)
    d_eff = (P.mean(0) - N.mean(0)) / np.sqrt(
        0.5 * (P.std(0) ** 2 + N.std(0) ** 2) + 1e-12)
    order = np.argsort(-np.abs(d_eff))

    # topdim: select m on calibration (in-sample, allowed — it is the dev set)
    m_auc = {}
    masks = {}
    for m in M_GRID:
        dims = order[:m][d_eff[order[:m]] > 0]
        masks[m] = dims
        m_auc[m] = auroc((P[:, dims] ** 2).sum(1), (N[:, dims] ** 2).sum(1))
    m_star = max(m_auc, key=m_auc.get)
    mask_star = masks[m_star]

    # axis_lr
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    lr.fit(np.vstack([P, N]).astype(np.float32),
           np.r_[np.ones(len(P)), np.zeros(len(N))])

    # ces (needs oriented gold edits + relations; negatives sign-free)
    D_pos, rel_pos = fit_training_edits(records, pv, V, gold & cal)
    D_neg = pv_neg.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace().fit(D_pos, rel_pos, D_neg, list(pv_neg.relation))

    # rcesp comparison row, same fit as experiment 3
    rcesp = RCESP(k=20).fit(D_pos, rel_pos)

    # abtt cosine comparison row
    V_abtt = spaces["abtt"]

    def all_scores(view: data.PairView) -> dict[str, np.ndarray]:
        return {
            "abtt_cos": view.cos(V_abtt),
            "campaign_cos": view.cos(V),
            "axis_lr": _chunked_scores(
                view, V, lambda A, s, e: lr.decision_function(
                    A.astype(np.float32))),
            "ces": ces.score(view, V),
            "topdim": _chunked_scores(
                view, V, lambda A, s, e: (A[:, mask_star] ** 2).sum(1)),
            "rcesp": rcesp.score(view, V)[0],
        }

    out = {"provenance": data.provenance(
        experiment="dimension_ideas", k_ces=K_CES, m_grid=list(M_GRID),
        n_boot=N_BOOT, n_cal_pos=int(len(P)), n_cal_hardneg=int(len(N))),
        "dimension_analysis": {
            "max_abs_effect": float(np.abs(d_eff).max()),
            "n_dims_effect_gt_0.2": int((np.abs(d_eff) > 0.2).sum()),
            "n_dims_effect_gt_0.5": int((np.abs(d_eff) > 0.5).sum()),
            "top20_dims": [[int(i), float(d_eff[i])] for i in order[:20]],
            "topdim_cal_auroc_by_m": {str(m): v for m, v in m_auc.items()},
            "m_star": int(m_star),
            "n_mask_dims": int(len(mask_star)),
        }}

    # ── confirmatory hard-negative task ────────────────────────────────────
    mask = (gold | hardneg) & ~cal
    view = pv.subset(mask)
    y = gold[mask]
    cc = camp_cos[mask]
    sc = all_scores(view)
    hard = {}
    for name, s in sc.items():
        hard[name] = {
            "auroc": auroc(s[y], s[~y]), "auprc": auprc(s[y], s[~y]),
            "auroc_ci95": bootstrap_ci(auroc, s[y], s[~y], N_BOOT, data.SEED),
            "inverted_vs_campaign_cos": inverted_win_rate(s[y], cc[y], s[~y], cc[~y]),
        }
    out["hard_negative_confirmatory"] = {
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "methods": hard}

    out["paired_bootstrap_confirmatory_hard"] = {
        f"{a}_minus_{b}": paired_bootstrap_delta_auc(
            sc[a][y], sc[a][~y], sc[b][y], sc[b][~y], N_BOOT, data.SEED)
        for a, b in (("axis_lr", "abtt_cos"), ("ces", "abtt_cos"),
                     ("axis_lr", "rcesp"), ("ces", "rcesp"),
                     ("topdim", "campaign_cos"))}

    # ── balanced sh_64k eval set (+ band) ──────────────────────────────────
    mask = in_eval & (subset == "sh_64k")
    view = pv.subset(mask)
    y = gold[mask]
    band = (camp_cos[mask] >= 0.87) & (camp_cos[mask] <= 0.97)
    sc = all_scores(view)
    bal = {}
    for name, s in sc.items():
        yb, sb = y[band], s[band]
        bal[name] = {"auroc": auroc(s[y], s[~y]),
                     "band_auroc": auroc(sb[yb], sb[~yb])}
    out["balanced_sh64k"] = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                             "band_n_pos": int(y[band].sum()),
                             "band_n_neg": int((~y[band]).sum()),
                             "methods": bal}

    # ── disjoint slices (confirmatory positives vs all conf hard negs) ─────
    def disjoint(keep_pos) -> dict:
        pos_idx = [i for i in np.flatnonzero(gold & ~cal) if keep_pos(i)]
        m = np.zeros(len(records), bool)
        m[pos_idx] = True
        m |= hardneg & ~cal
        view_g = pv.subset(m)
        y_g = gold[m]
        sc_g = all_scores(view_g)
        return {"n_pos": int(y_g.sum()),
                "auroc": {n: auroc(s[y_g], s[~y_g]) for n, s in sc_g.items()}}

    out["transition_disjoint"] = disjoint(
        lambda i: trans_keys[i] is not None and trans_keys[i] not in cal_transitions)
    out["subject_disjoint"] = disjoint(
        lambda i: records[i]["parser"]["both_parse"]
        and records[i]["parser"]["fact_a_parsed"]["subject"] not in cal_subjects
        and records[i]["parser"]["fact_b_parsed"]["subject"] not in cal_subjects)

    data.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = data.OUT_DIR / "dimension_ideas.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written:", dst)
    return out


if __name__ == "__main__":
    r = run()
    da = r["dimension_analysis"]
    print(f"m*={da['m_star']} ({da['n_mask_dims']} dims), cal grid "
          f"{da['topdim_cal_auroc_by_m']}")
    print("\nconfirmatory hard-negative task:")
    for name, m in r["hard_negative_confirmatory"]["methods"].items():
        print(f"  {name:13s} AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}"
              f"  inverted-win {m['inverted_vs_campaign_cos']['win_rate']:.3f}")
    print("\nbalanced sh_64k:")
    for name, m in r["balanced_sh64k"]["methods"].items():
        print(f"  {name:13s} AUROC {m['auroc']:.4f}  band {m['band_auroc']:.4f}")
    for tag in ("transition_disjoint", "subject_disjoint"):
        print(f"\n{tag} (n_pos={r[tag]['n_pos']}):",
              {n: round(v, 3) for n, v in r[tag]["auroc"].items()})
