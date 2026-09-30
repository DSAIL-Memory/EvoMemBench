"""Small closed-form statistics shared by the second-reader audit.  [offline]

Stdlib only. Every function here is checked against a hand-computed value in
``hnav/tests/test_audit_laya.py``.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def cohen_kappa(x: Sequence[bool], y: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters over the same items.

    Returns None when fewer than two items or when chance agreement is 1
    (both raters constant), where kappa is undefined.
    """
    if len(x) != len(y):
        raise ValueError("rater vectors differ in length")
    n = len(x)
    if n < 2:
        return None
    c = Counter(zip((bool(a) for a in x), (bool(b) for b in y)))
    po = (c[(True, True)] + c[(False, False)]) / n
    px = (c[(True, True)] + c[(True, False)]) / n
    py = (c[(True, True)] + c[(False, True)]) / n
    pe = px * py + (1 - px) * (1 - py)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def agreement(x: Sequence[bool], y: Sequence[bool]) -> float | None:
    if len(x) != len(y):
        raise ValueError("rater vectors differ in length")
    if not x:
        return None
    return sum(bool(a) == bool(b) for a, b in zip(x, y)) / len(x)


def table2x2(x: Sequence[bool], y: Sequence[bool]) -> dict:
    """Counts keyed by (x, y) as strings: 'TT', 'TF', 'FT', 'FF'."""
    c = Counter(zip((bool(a) for a in x), (bool(b) for b in y)))
    return {"TT": c[(True, True)], "TF": c[(True, False)],
            "FT": c[(False, True)], "FF": c[(False, False)]}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (two-sided 95 % default)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rank_auc(pos: Iterable[float], neg: Iterable[float]) -> float | None:
    """P(random positive scores above random negative); ties count half."""
    pos = list(pos)
    neg = list(neg)
    if not pos or not neg:
        return None
    both = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    wins = 0.0
    seen_neg = 0
    i = 0
    while i < len(both):
        j = i
        while j < len(both) and both[j][0] == both[i][0]:
            j += 1
        tie_neg = sum(1 for t in both[i:j] if t[1] == 0)
        tie_pos = (j - i) - tie_neg
        wins += tie_pos * (seen_neg + tie_neg / 2.0)
        seen_neg += tie_neg
        i = j
    return wins / (len(pos) * len(neg))


def rogan_gladen(observed_rate: float, sensitivity: float, false_positive_rate: float) -> float | None:
    """Prevalence corrected for an imperfect screen: (r - f) / (s - f), clipped to [0, 1]."""
    if sensitivity is None or false_positive_rate is None:
        return None
    if sensitivity - false_positive_rate <= 0:
        return None
    return min(1.0, max(0.0, (observed_rate - false_positive_rate) / (sensitivity - false_positive_rate)))


def largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    """Apportion ``total`` seats proportionally to ``weights`` (Hamilton method)."""
    s = float(sum(weights))
    if total <= 0 or s <= 0:
        return [0] * len(weights)
    quotas = [w / s * total for w in weights]
    seats = [int(math.floor(q)) for q in quotas]
    remaining = total - sum(seats)
    order = sorted(range(len(weights)), key=lambda i: quotas[i] - seats[i], reverse=True)
    for i in order[:remaining]:
        seats[i] += 1
    return seats
