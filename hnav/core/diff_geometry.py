"""Marginal-diff geometry.  [T6]

The hypothesis this module exists to test (H2): when a candidate is a *near
duplicate* of an admitted fact, the interesting question is not how similar the
two blobs are but whether the span that changed is semantically disjoint. A
fact that repeats an admitted one word-for-word is redundant; a fact identical
except that "Howard University" became "Harvard University" is an update, and
suppressing it destroys the answer.

  ``whole_blob_sim``  cosine between the two full fact strings
  ``diff_sim``        cosine between the two changed spans
  ``diff_novelty``    QR-residual novelty of the new span against the values the
                      store has already seen — genuinely different information
                      from ``diff_sim``, not a transform of it, which is what
                      makes the nested-model comparison in M4 meaningful
  ``is_critical_delta`` ``whole_blob_sim ≥ τ_high`` and ``diff_sim ≤ τ_low``

Span extraction is **adapter-supplied** because it is benchmark-specific. On the
primary arena it is exact and cheap — parse both facts with the induced relation
templates and take the object slots (99.5%+ coverage). The token-level fallback
runs when parsing fails, and ``parse_ok`` records which path was taken so the
two are never pooled silently.

Thresholds default to the values named in the protocol's
``WRITE_CRITICAL_DELTA`` definition. They are **calibration-split quantities**:
fit on ``sh_6k`` + ``sh_32k``, frozen, then applied unchanged. Nothing here may
be tuned on ``sh_64k`` or ``sh_262k``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Callable, Sequence

import numpy as np

from .geometry import qr_residual

__all__ = ["DiffSignals", "DiffGeometryModule", "token_diff_spans"]

_EPS = 1e-12

# Defaults, to be replaced by calibration-split quantiles before any live arm.
TAU_HIGH_DEFAULT = 0.80
TAU_LOW_DEFAULT = 0.30


@dataclass
class DiffSignals:
    whole_blob_sim: float = float("nan")
    diff_span_old: str = ""
    diff_span_new: str = ""
    diff_sim: float = float("nan")
    diff_novelty: float = float("nan")
    is_critical_delta: bool = False
    parse_ok: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def token_diff_spans(old: str, new: str) -> tuple[str, str]:
    """Fallback span extraction: the tokens that actually changed.

    Uses ``difflib.SequenceMatcher`` over whitespace tokens and returns the
    replaced/inserted/deleted runs. When the two strings are identical the spans
    are empty, which is the honest answer — there is no delta to score.
    """
    a, b = (old or "").split(), (new or "").split()
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    old_parts, new_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_parts.extend(a[i1:i2])
        new_parts.extend(b[j1:j2])
    return " ".join(old_parts), " ".join(new_parts)


class DiffGeometryModule:
    """``compute_if_update(old, new)`` -> :class:`DiffSignals` or ``None``.

    ``None`` means "this candidate is not an update" — there is no predecessor,
    so there is no diff to take. That is a different statement from "the diff is
    uninformative" and the two must not be collapsed.
    """

    def __init__(self, embedder, span_fn: Callable[[str, str], tuple[str, str, bool]] | None = None,
                 tau_high: float = TAU_HIGH_DEFAULT, tau_low: float = TAU_LOW_DEFAULT,
                 max_basis: int = 512) -> None:
        self.embedder = embedder
        self.span_fn = span_fn
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.max_basis = max_basis

    def spans(self, old: str, new: str) -> tuple[str, str, bool]:
        if self.span_fn is not None:
            try:
                return self.span_fn(old, new)
            except Exception:  # noqa: BLE001 — a parse failure is a fallback, not a crash
                pass
        o, n = token_diff_spans(old, new)
        return o, n, False

    def compute_if_update(self, old: str | None, new: str,
                          prior_objects: np.ndarray | Sequence | None = None,
                          ctx: dict | None = None) -> DiffSignals | None:
        if not old:
            return None

        old_span, new_span, parse_ok = self.spans(old, new)
        texts = [old, new, old_span or old, new_span or new]
        vecs = np.asarray(self.embedder.encode(texts), dtype=np.float64)

        whole = float(vecs[0] @ vecs[1])
        diff = float(vecs[2] @ vecs[3])

        novelty = float("nan")
        if prior_objects is not None and len(prior_objects) > 0:
            r, _ = qr_residual(vecs[3], np.asarray(prior_objects, dtype=np.float64),
                               max_basis=self.max_basis)
            novelty = float(r[0])

        return DiffSignals(
            whole_blob_sim=whole,
            diff_span_old=old_span,
            diff_span_new=new_span,
            diff_sim=diff,
            diff_novelty=novelty,
            is_critical_delta=bool(whole >= self.tau_high and diff <= self.tau_low),
            parse_ok=parse_ok,
        )
