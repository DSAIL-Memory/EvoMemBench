"""Evaluation metrics with uncertainty, numpy only.

Every quantity is a plain function of (positive scores, negative scores) so the
tests can check each against a closed-form or sklearn recomputation. Bootstrap
and permutation machinery is seeded and deterministic.
"""
from __future__ import annotations

import numpy as np

__all__ = ["auroc", "auprc", "prf_at", "best_f1_threshold", "bootstrap_ci",
           "paired_bootstrap_delta_auc", "inverted_win_rate", "perm_pvalue"]


def auroc(pos, neg) -> float:
    """Rank AUC = P(score_pos > score_neg) + 0.5·P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left")
    hi = np.searchsorted(ns, pos, side="right")
    return float((lo + (hi - lo) / 2.0).sum() / (len(pos) * len(neg)))


def auprc(pos, neg) -> float:
    """Area under precision–recall via the step interpolation sklearn uses
    (sum of ΔR·P at each threshold, descending scores; positives first on ties
    is NOT assumed — ties are grouped)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    # group ties: evaluate only at the last index of each distinct score
    distinct = np.flatnonzero(np.diff(scores)) if len(scores) > 1 else np.array([], int)
    idx = np.concatenate([distinct, [len(scores) - 1]])
    tp = np.cumsum(labels)[idx]
    fp = np.cumsum(1 - labels)[idx]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / len(pos)
    prev_r = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_r) * precision))


def prf_at(threshold: float, pos, neg) -> dict:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    tp = int((pos >= threshold).sum())
    fp = int((neg >= threshold).sum())
    fn = len(pos) - tp
    tn = len(neg) - fp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def best_f1_threshold(pos, neg) -> float:
    """Threshold maximizing F1, chosen over the observed score values.
    A calibration-split quantity: never call this on confirmatory data."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    cand = np.unique(np.concatenate([pos, neg]))
    best_t, best_f = float(cand[0]), -1.0
    for t in cand:
        f = prf_at(t, pos, neg)["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def bootstrap_ci(stat_fn, pos, neg, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap over pairs, resampling both classes independently."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = stat_fn(pos[rng.integers(0, len(pos), len(pos))],
                          neg[rng.integers(0, len(neg), len(neg))])
    return {"point": float(stat_fn(pos, neg)),
            "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "n_boot": n_boot}


def paired_bootstrap_delta_auc(pos_a, neg_a, pos_b, neg_b, n_boot: int = 1000,
                               seed: int = 0) -> dict:
    """AUC(method A) − AUC(method B) with the SAME resampled examples in both
    arms — the honest comparison when both methods score the same pairs."""
    pos_a, neg_a = np.asarray(pos_a, float), np.asarray(neg_a, float)
    pos_b, neg_b = np.asarray(pos_b, float), np.asarray(neg_b, float)
    assert len(pos_a) == len(pos_b) and len(neg_a) == len(neg_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pi = rng.integers(0, len(pos_a), len(pos_a))
        ni = rng.integers(0, len(neg_a), len(neg_a))
        deltas[i] = auroc(pos_a[pi], neg_a[ni]) - auroc(pos_b[pi], neg_b[ni])
    return {"delta": float(auroc(pos_a, neg_a) - auroc(pos_b, neg_b)),
            "lo": float(np.quantile(deltas, 0.025)),
            "hi": float(np.quantile(deltas, 0.975)),
            "p_delta_le_0": float(np.mean(deltas <= 0.0)),
            "n_boot": n_boot}


def inverted_win_rate(pos_scores, pos_cos, neg_scores, neg_cos,
                      chunk: int = 512) -> dict:
    """Win rate restricted to the comparisons where cosine is WRONG.

    Over all (positive, negative) pairs with ``cos(neg) > cos(pos)`` — where
    cosine's own win rate is 0 by construction — the fraction the method still
    orders correctly (score_pos > score_neg; ties count half). This is the
    stress test: a method living off cosine cannot beat 0 here.
    """
    ps, pc = np.asarray(pos_scores, float), np.asarray(pos_cos, float)
    nss, nc = np.asarray(neg_scores, float), np.asarray(neg_cos, float)
    n_comp = wins = 0.0
    for s in range(0, len(ps), chunk):
        e = min(s + chunk, len(ps))
        mask = nc[None, :] > pc[s:e, None]
        diff = ps[s:e, None] - nss[None, :]
        n_comp += mask.sum()
        wins += ((diff > 0) & mask).sum() + 0.5 * ((diff == 0) & mask).sum()
    return {"n_comparisons": int(n_comp),
            "win_rate": float(wins / n_comp) if n_comp else None}


def perm_pvalue(observed: float, null_samples, larger_is_extreme: bool = True) -> float:
    """(1 + #null ≥ obs) / (1 + n) — the add-one permutation p-value."""
    null_samples = np.asarray(null_samples, float)
    if larger_is_extreme:
        k = int((null_samples >= observed).sum())
    else:
        k = int((null_samples <= observed).sum())
    return float((1 + k) / (1 + len(null_samples)))
