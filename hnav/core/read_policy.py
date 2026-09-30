"""Read-path policies — Stage 1, Faz B.  [T11 rerank; T13 fact-level]

The only interventions H-Nav ships. Protocol trail: ``KAPI_KARARI.md`` §3/§6
(instruments GO, write_policy NO_GO forever, read_policy CONDITIONAL) →
user decision 2026-08-15 (``STAGE1_PLAN.md`` §0 R2: rerank **plus** stale
suppression and placement, "only the mechanism the probe proves") → this
module. ``test_no_raw_entropy_in_policy.py`` permits exactly this file under
``hnav/core/*policy*.py`` and AST-scans it for the forbidden raw-score entropy.

Three mechanisms, all driven by the same :class:`~hnav.core.read_gate.ReadGate`
decision and all born ``shadow=True``:

``RERANK`` (T11, :func:`rerank_order`, :class:`ReadRerankPolicy`)
    A token-neutral permutation of the retrieved page: each verified group's
    LATEST carrier is promoted immediately above that group's highest-ranked
    stale rival. **Kept as evidence, not as a recommendation** — the T11
    calibration measured it systematically harmful (162 cells, 0 net-positive,
    helped 228 / harmed 441 with a precision-1.00 detector;
    ``STAGE1_NULL_ANALIZI.md``). Deleting it would delete the finding.

``SUPPRESS`` (T13, :func:`suppress_ids`, ``ReadFactPolicy("suppress")``)
    Name the stale member(s) of every verified group so the caller can drop
    them from the assembled page. Reduces tokens.

``DEMOTE_LATE`` (T13, :func:`demote_ids`, ``ReadFactPolicy("demote_late")``)
    Name each verified group's LATEST carrier so the caller can move it to the
    END of the assembled page, deleting nothing. The fact multiset is preserved
    exactly; character count too, except for the separator bookkeeping described
    in ``mab_adapter.page_edit`` (which the adapter reports rather than assumes).

Why FACT granularity, and why that is a contract and not a detail
-----------------------------------------------------------------
The oracle probe (``hnav/stage1/stale_suppression_probe.py``, sh_6k, 473 real
calls) measured, on the conflicted stratum: native 4/74, suppressing the stale
fact 66/74, moving the latest fact to the very END 20/74, moving it to the
FRONT 1/74. The model anchors on the LATE-appearing value. The same corpus
shows the CHUNK-level version of the same idea to be harmful, because one chunk
carries 230-260 facts and reordering it scrambles hundreds of unrelated ones.
So both new mechanisms act on single facts inside the retrieved chunks.

The page contract. The benchmark hook returns a page of **chunk texts**, so a
fact-level edit has to be expressed as a rewrite of chunk text. This module
does NOT do that rewrite: it emits *ids* (``payload["drop_ids"]`` /
``payload["move_last_ids"]``) and the ADAPTER performs the splice, because
locating a fact inside a chunk means knowing the benchmark's serial numbering
and its chunker — knowledge that hard rule 2 keeps out of ``hnav/core/``. The
adapter side is ``mab_adapter.fact_spans`` / ``mab_adapter.page_edit``, which
splice original bytes and never renumber or reflow.

Unit mapping. The gate works on whatever candidates the adapter gave it (facts,
on the primary arena); the page being RERANKED may be a coarser unit (chunks),
so ``rerank_order`` takes an ``id_map``. The fact-level mechanisms need no map
— they name gate members directly, and the adapter resolves them against the
page. Groups without a LATEST (``latest_id=None``) are skipped by all three:
the gate refused to name a newest member, so there is nothing defensible to
suppress or move.

Determinism. Every output is a pure function of ``(decision, ...)``. Groups are
consumed in the gate's own order; within a group, members keep the gate's
order; duplicates are dropped keeping the first occurrence.

Ambiguity signals reach these policies only through the gate, which consumes
them via ``RetrievalSignalSet.for_policy()`` — the z-scored ``H_z`` family,
never the raw-score entropy (hard rule 6).
"""
from __future__ import annotations

from typing import Callable, Sequence

from .read_gate import (H_Z_CAL, NMARGIN_CAL, GateDecision, GateThresholds,
                        ReadGate)
from .types import Decision

__all__ = ["rerank_order", "suppress_ids", "demote_ids", "FACT_MECHANISMS",
           "FACT_ACTIONS", "ReadRerankPolicy", "ReadFactPolicy",
           "stage1_thresholds"]

# mechanism name -> the Decision.action and payload key it produces.
FACT_ACTIONS = {"suppress": ("SUPPRESS", "drop_ids"),
                "demote_late": ("DEMOTE_LATE", "move_last_ids")}
FACT_MECHANISMS = tuple(FACT_ACTIONS)


def stage1_thresholds() -> GateThresholds:
    """The gate operating point the live read path runs at.

    FROZEN by ``hnav/stage1/detector_gap.py --select`` on the calibration split
    (sh_6k + sh_32k, 200 questions), from DETECTION quality only — no LLM, no
    accuracy, no gold answer — before any arm was graded. The committed artifact
    is ``stage0_results/stage1_operating_point.json`` and
    ``test_threshold_provenance.py`` pins these values to it.

    At this point, on the calibration split: pair precision 1.0000, pair recall
    within the candidate pool 0.9784, 2,673 facts named stale of which 2,673 are
    genuinely superseded and **0** would change what the page says a key's
    current value is.

    ``ambiguity_mode="none"`` is a declared departure from the Faz A default,
    not an oversight. ``nmargin`` and ``H_z`` are still carried here for the
    record, but they are the only gate inputs derived from CHUNK embeddings,
    which were truncated at 512 of ~4096 tokens (T12) and are not yet re-fit;
    they are also the dominant recall bottleneck (conflicted-question recall
    collapses from 0.957 to ~0.16 with the screen on); and the volume-limiting
    job they were doing is now done by the adapter's identity screen plus the
    bidirectional NLI at the precision above. The artifact records the same
    argument under ``ambiguity_note``, and a campaign run after the embeddings
    are re-fit must revisit it.

    ``r_min=0.44`` is ``calibrate_read_policy.R_LOOSE`` — sqrt(1-0.90²)=0.436
    rounded up, i.e. a pass-through for every pair the ``cos_pair=0.90`` screen
    admits. It is hardcoded rather than imported because ``hnav/core/`` may not
    import from ``hnav/stage1/``; the value is pinned by the provenance test.
    """
    return GateThresholds(cos_pair=0.90, r_min=0.44,
                          nmargin=NMARGIN_CAL, H_z=H_Z_CAL,
                          ambiguity_mode="none", nli_contradiction=0.90)


def rerank_order(ordering: Sequence[str], decision: GateDecision,
                 id_map: Callable[[str], str | None] | None = None) -> list[str]:
    """Return the reranked page — a permutation of ``ordering``, or the
    identical list when the decision moves nothing.

    ``ordering``  page-unit ids, ranked (the benchmark's top-k page).
    ``decision``  a :class:`GateDecision`; only ``groups`` is consulted.
    ``id_map``    gate-member id -> page-unit id (or ``None`` when a member
                  has no page unit). Defaults to identity.
    """
    ids = list(ordering)
    if len(set(ids)) != len(ids):
        raise ValueError("ordering contains duplicate ids; refusing to rerank")
    if not decision.groups:
        return ids

    mapper = id_map if id_map is not None else (lambda x: x)
    pos0 = {u: i for i, u in enumerate(ids)}

    # Plan every promotion against the ORIGINAL positions, then apply in a
    # canonical order — the docstring's determinism contract.
    plans: list[tuple[int, int, str, str]] = []
    for g in decision.groups:
        if g.latest_id is None:
            continue                    # the gate refused to name a LATEST
        latest_u = mapper(g.latest_id)
        if latest_u is None or latest_u not in pos0:
            continue                    # not on the page; nothing to promote
        stale_us = sorted(
            {u for u in (mapper(s) for s in g.stale_ids)
             if u is not None and u in pos0 and u != latest_u},
            key=pos0.__getitem__)
        if not stale_us:
            continue                    # intra-unit conflict or off-page rivals
        first_stale = stale_us[0]
        if pos0[latest_u] <= pos0[first_stale]:
            continue                    # already above every rival
        plans.append((pos0[first_stale], pos0[latest_u], latest_u, first_stale))

    out = ids[:]
    for _, _, latest_u, first_stale in sorted(plans):
        cur, tgt = out.index(latest_u), out.index(first_stale)
        if cur > tgt:                   # may have been promoted past already
            out.pop(cur)
            out.insert(tgt, latest_u)

    if sorted(out) != sorted(ids):      # defence in depth: token neutrality
        raise AssertionError("rerank produced a non-permutation — refusing")
    return out


# ── fact-level mechanisms (T13) ──────────────────────────────────────────────
def _actionable(decision: GateDecision):
    """Verified groups the gate was willing to name a LATEST for.

    A group with ``latest_id=None`` (recency key missing or tied) is skipped by
    every mechanism: suppressing "the stale ones" of a group whose newest
    member is unknown would be a guess, and the gate exists to not guess.
    """
    return [g for g in decision.groups if g.latest_id is not None]


def _dedupe(ids) -> list[str]:
    """First occurrence wins; order otherwise preserved. Groups are disjoint
    components so this normally changes nothing — it is here so that a
    duplicate can never reach the adapter as a double edit."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def suppress_ids(decision: GateDecision) -> list[str]:
    """Ids to DROP from the assembled page: every stale member of every
    verified group that named a LATEST.

    The LATEST itself is never returned, and :func:`_check_disjoint` refuses
    the pathological case where an id appears as both — an overlap would mean
    the gate emitted non-disjoint components, and acting on that could delete
    the very fact another group wants kept.
    """
    out = _dedupe([s for g in _actionable(decision) for s in g.stale_ids
                   if s != g.latest_id])
    _check_disjoint(out, _dedupe([g.latest_id for g in _actionable(decision)]))
    return out


def demote_ids(decision: GateDecision) -> list[str]:
    """Ids to MOVE to the end of the assembled page: the LATEST carrier of
    every verified group that named one. Nothing is deleted."""
    out = _dedupe([g.latest_id for g in _actionable(decision)])
    _check_disjoint(_dedupe([s for g in _actionable(decision)
                             for s in g.stale_ids if s != g.latest_id]), out)
    return out


def _check_disjoint(stale: Sequence[str], latest: Sequence[str]) -> None:
    overlap = sorted(set(stale) & set(latest))
    if overlap:
        raise ValueError(
            f"gate emitted overlapping groups: {overlap[:3]} is both a LATEST "
            "and a stale member; refusing to edit the page")


# Gate counters copied into Decision.reasons — one auditable place.
_GATE_COUNTERS = ("ambiguous", "n_candidates", "n_pairs_screened", "n_pairs_cos",
                  "n_pairs_filter_rejected", "n_groups_cos", "n_groups_geometric",
                  "n_pairs_nli", "n_pairs_verified")


class ReadRerankPolicy:
    """Gate → rerank, packaged as a :class:`~hnav.core.types.Decision`.

    ``decide`` mutates nothing: not the candidates, not the ordering, not the
    store. ``Decision.shadow`` is returned ``True``; the ADAPTER flips it to
    ``False`` only under ``HNAV_MODE=live`` — the same defence-in-depth seam
    Stage 0 used, so a stray policy object can never act by itself.
    """

    def __init__(self, gate: ReadGate) -> None:
        self.gate = gate

    def decide(self, candidates, ordering: Sequence[str],
               signal_view=None,
               latest_key: Callable | None = None,
               id_map: Callable[[str], str | None] | None = None) -> Decision:
        gd = self.gate.decide(candidates, signal_view, latest_key=latest_key)
        new_order = rerank_order(ordering, gd, id_map=id_map)
        changed = new_order != list(ordering)
        reasons = _gate_reasons(gd)
        reasons["n_positions_changed"] = sum(
            1 for a, b in zip(new_order, ordering) if a != b)
        return Decision(
            action="RERANK" if changed else "PASS",
            payload={"order": new_order} if changed else None,
            reasons=reasons,
            shadow=True,
        )


def _gate_reasons(gd: GateDecision) -> dict:
    """The audit trail every policy attaches to its ``Decision``."""
    return {
        "gate": {k: getattr(gd, k) for k in _GATE_COUNTERS},
        "groups": [g.to_dict() for g in gd.groups],
        "thresholds": gd.reasons.get("thresholds"),
        "ambiguity": {k: v for k, v in gd.reasons.items() if k != "thresholds"},
    }


class ReadFactPolicy:
    """Gate → a FACT-level edit of the assembled page.  [T13]

    ``mechanism="suppress"``     -> ``Decision(action="SUPPRESS",
                                   payload={"drop_ids": [...]})``
    ``mechanism="demote_late"``  -> ``Decision(action="DEMOTE_LATE",
                                   payload={"move_last_ids": [...]})``

    Emits ids, never text: the splice that turns an id into rewritten chunk
    bytes belongs to the adapter (module docstring, "the page contract"). An
    empty id list is a ``PASS`` with no payload, byte-identical downstream to
    having no policy at all.

    ``decide`` mutates nothing: not the candidates, not the ordering, not the
    store. ``Decision.shadow`` is returned ``True``; the ADAPTER flips it to
    ``False`` only under ``HNAV_MODE=live``, so a stray policy object can never
    act by itself. ``ordering`` and ``id_map`` are accepted and unused — they
    keep one call signature across all read policies, and a fact-level edit
    needs neither a page order nor a coarser unit to project onto.
    """

    def __init__(self, gate: ReadGate, mechanism: str = "suppress") -> None:
        if mechanism not in FACT_ACTIONS:
            raise ValueError(f"mechanism={mechanism!r} not in {FACT_MECHANISMS}")
        self.gate = gate
        self.mechanism = mechanism

    def decide(self, candidates, ordering: Sequence[str] | None = None,
               signal_view=None,
               latest_key: Callable | None = None,
               id_map: Callable[[str], str | None] | None = None) -> Decision:
        gd = self.gate.decide(candidates, signal_view, latest_key=latest_key)
        action, payload_key = FACT_ACTIONS[self.mechanism]
        ids = (suppress_ids(gd) if self.mechanism == "suppress"
               else demote_ids(gd))
        reasons = _gate_reasons(gd)
        reasons["mechanism"] = self.mechanism
        reasons["n_ids"] = len(ids)
        reasons["n_groups_actionable"] = len(_actionable(gd))
        reasons["n_groups_without_latest"] = len(gd.groups) - len(_actionable(gd))
        return Decision(
            action=action if ids else "PASS",
            payload={payload_key: ids} if ids else None,
            reasons=reasons,
            shadow=True,
        )
