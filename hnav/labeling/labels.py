"""Failure labels, exactly as retained in the Stage-0 protocol §3.  [T6]

Every retained label is a **deterministic function of an explicit context dict**
and carries its online/offline status in the registry, so nothing can quietly
use an offline label on an online path. Labels the benchmark does not generate
are kept in the registry with ``retained=False`` and a reason: dropping a label
because the class does not exist is a *finding*, not a gap, and deleting the
entry would erase the finding.

``WRITE_DESTRUCTIVE_OVERWRITE`` is dropped — no write path in EvoMemBench ever
overwrites (analysis §4.1).

Missing context keys are treated as "not established", so a label never fires on
absent evidence. That is deliberate: on this benchmark a false positive on the
write side means suppressing the only fact that carries the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["LabelDef", "WRITE_LABELS", "READ_LABELS", "ALL_LABELS",
           "apply_write_labels", "apply_read_labels", "retained"]


@dataclass(frozen=True)
class LabelDef:
    name: str
    side: str                 # "write" | "read"
    definition: str
    online: bool
    retained: bool
    fn: Callable[[dict], bool] | None = None
    dropped_reason: str = ""

    def __call__(self, ctx: dict) -> bool:
        if self.fn is None:
            return False
        return bool(self.fn(ctx))


def _g(ctx: dict, key: str, default: Any = None) -> Any:
    v = ctx.get(key, default)
    return default if v is None else v


def _num(ctx: dict, key: str) -> float:
    v = ctx.get(key)
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ── write side ───────────────────────────────────────────────────────────────
def _write_conflict(ctx: dict) -> bool:
    """Shares (relation, subject) with an already-admitted fact and differs in
    the object slot. `prior_object` comes from `latest_before`, never `latest`."""
    if not _g(ctx, "has_prior_same_key", False):
        return False
    return _g(ctx, "prior_object", None) != _g(ctx, "candidate_object", None)


def _write_critical_delta(ctx: dict) -> bool:
    if not _write_conflict(ctx):
        return False
    whole, diff = _num(ctx, "whole_blob_sim"), _num(ctx, "diff_sim")
    hi, lo = _num(ctx, "tau_high"), _num(ctx, "tau_low")
    if any(v != v for v in (whole, diff, hi, lo)):     # NaN check
        return False
    return whole >= hi and diff <= lo


def _write_stale_supersede(ctx: dict) -> bool:
    """A LATER fact exists for the same key. Needs the future, so offline only."""
    return bool(_g(ctx, "later_fact_exists", False))


def _write_redundant(ctx: dict) -> bool:
    sim, tau = _num(ctx, "sim_max"), _num(ctx, "tau")
    if sim != sim or tau != tau:
        return False
    return sim >= tau and not _g(ctx, "verbatim_new_values", [])


def _write_duplicate(ctx: dict) -> bool:
    return bool(_g(ctx, "is_exact_dup", False))


def _write_unnecessary(ctx: dict) -> bool:
    """Admitted, and no downstream question maps to its key. Offline: the
    question set is not visible at write time."""
    return not _g(ctx, "question_maps_to_key", True)


def _write_missed_update(ctx: dict) -> bool:
    """A fact present in the source text never became a candidate."""
    return bool(_g(ctx, "in_source_not_candidate", False))


WRITE_LABELS: dict[str, LabelDef] = {
    d.name: d for d in [
        LabelDef("WRITE_CRITICAL_DELTA", "write",
                 "shares (relation, subject) with an admitted fact, differs in object, "
                 "and whole_blob_sim >= tau_high while diff_sim <= tau_low",
                 online=True, retained=True, fn=_write_critical_delta),
        LabelDef("WRITE_CONFLICT", "write",
                 "shares (relation, subject) with an admitted fact, differs in object",
                 online=True, retained=True, fn=_write_conflict),
        LabelDef("WRITE_STALE_SUPERSEDE", "write",
                 "a later fact exists for the same key",
                 online=False, retained=True, fn=_write_stale_supersede),
        LabelDef("WRITE_REDUNDANT", "write",
                 "sim_max >= tau and the candidate introduces no new verbatim value",
                 online=True, retained=True, fn=_write_redundant),
        LabelDef("WRITE_DUPLICATE", "write",
                 "exact string / MD5 match against an admitted fact",
                 online=True, retained=True, fn=_write_duplicate),
        LabelDef("WRITE_UNNECESSARY", "write",
                 "admitted, and no downstream question maps to its key",
                 online=False, retained=True, fn=_write_unnecessary),
        LabelDef("WRITE_MISSED_UPDATE", "write",
                 "a fact present in the source text never became a candidate",
                 online=False, retained=True, fn=_write_missed_update),
        LabelDef("WRITE_DESTRUCTIVE_OVERWRITE", "write",
                 "(no operational definition — the class does not exist here)",
                 online=False, retained=False, fn=None,
                 dropped_reason="No write path in EvoMemBench ever overwrites "
                                "(analysis 4.1). Dropped as a finding, not a gap."),
    ]
}


# ── read side ────────────────────────────────────────────────────────────────
def _read_stale(ctx: dict) -> bool:
    """A retrieved fact is superseded by a higher-serial fact in the store.

    Online at read time: the whole store is legitimately visible then, which is
    why ReadConflictIndex.latest is the right API here and latest_before is not.
    """
    return len(_g(ctx, "stale_hits", [])) > 0


def _read_relevant_below_k(ctx: dict) -> bool:
    """The superseding fact exists but ranks below top_k."""
    return any(not h.get("superseder_retrieved", False)
               for h in _g(ctx, "stale_hits", []))


def _read_conflict(ctx: dict) -> bool:
    return int(_g(ctx, "n_conflicted_keys_in_topk", 0)) > 0


def _read_ambiguous(ctx: dict) -> bool:
    n, thr = _num(ctx, "nmargin"), _num(ctx, "nmargin_threshold")
    if n != n or thr != thr:
        return False
    return n < thr


def _read_high_interference(ctx: dict) -> bool:
    for value_key, thr_key in (("H_z", "H_z_threshold"), ("H_vn", "H_vn_threshold")):
        v, thr = _num(ctx, value_key), _num(ctx, thr_key)
        if v == v and thr == thr and v > thr:
            return True
    return False


def _read_distractor(ctx: dict) -> bool:
    """A retrieved fact shares the subject but not the queried relation."""
    return int(_g(ctx, "n_subject_only_matches", 0)) > 0


def _read_missing(ctx: dict) -> bool:
    return not _g(ctx, "key_matched_in_topk", True)


READ_LABELS: dict[str, LabelDef] = {
    d.name: d for d in [
        LabelDef("READ_STALE", "read",
                 "a retrieved fact is superseded by a higher-serial fact in the store",
                 online=True, retained=True, fn=_read_stale),
        LabelDef("READ_RELEVANT_BELOW_K", "read",
                 "the superseding fact exists but ranks below top_k",
                 online=True, retained=True, fn=_read_relevant_below_k),
        LabelDef("READ_CONFLICT", "read",
                 ">=2 retrieved facts share a key with differing objects",
                 online=True, retained=True, fn=_read_conflict),
        LabelDef("READ_AMBIGUOUS", "read",
                 "normalized margin below the calibrated percentile",
                 online=True, retained=True, fn=_read_ambiguous),
        LabelDef("READ_HIGH_INTERFERENCE", "read",
                 "H_z or H_vn above the calibrated percentile",
                 online=True, retained=True, fn=_read_high_interference),
        LabelDef("READ_DISTRACTOR", "read",
                 "a retrieved fact shares the subject but not the queried relation",
                 online=True, retained=True, fn=_read_distractor),
        LabelDef("READ_MISSING", "read",
                 "no retrieved fact matches the queried key",
                 online=True, retained=True, fn=_read_missing),
        LabelDef("READ_CLEAR", "read",
                 "none of the above — the do-no-harm stratum",
                 online=True, retained=True, fn=None),
    ]
}

ALL_LABELS: dict[str, LabelDef] = {**WRITE_LABELS, **READ_LABELS}


def retained(side: str | None = None) -> list[LabelDef]:
    labels = ALL_LABELS.values()
    return [d for d in labels if d.retained and (side is None or d.side == side)]


def apply_write_labels(ctx: dict, online_only: bool = False) -> list[str]:
    return [d.name for d in retained("write")
            if (not online_only or d.online) and d(ctx)]


def apply_read_labels(ctx: dict, online_only: bool = False) -> list[str]:
    """``READ_CLEAR`` is the complement of every other read label, so it is
    derived rather than defined — it can never be simultaneously true with one
    of them."""
    fired = [d.name for d in retained("read")
             if d.name != "READ_CLEAR" and (not online_only or d.online) and d(ctx)]
    return fired or ["READ_CLEAR"]
