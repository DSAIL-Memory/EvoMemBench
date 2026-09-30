#!/usr/bin/env python3
"""Detector-to-oracle gap: how much of the probe's ceiling survives detection?  [T13]

The oracle probe (``hnav/stage1/stale_suppression_probe.py``, ``sh_6k``, 473
real calls against the frozen :8003 substrate) measured what a PERFECT conflict
detector would buy on the conflicted stratum:

    arm                       overall   unique   conflicted     McNemar vs native
    native                    0.290     25/26    4/74  (5.4%)   --
    native_repeat             0.290     25/26    4/74           0/0  (A/A floor)
    oracle_suppress           0.910     25/26    66/74 (89.2%)  net +62, p=4.3e-19
    oracle_recency            0.460     26/26    20/74 (27.0%)  net +17, p=7.6e-05
    anti (latest to FRONT)    0.260     25/26    1/74  (1.4%)   net -3, n.s.

Those arms read the ANSWER to decide which fact to cut or move. A shipped
policy may not. This module runs the identical 5-arm design with the arms
driven by :class:`~hnav.core.read_gate.ReadGate` plus the T13 fact-level
mechanisms in ``hnav/core/read_policy.py`` — embedding geometry and
bidirectional NLI only, no gold, no answers, no future facts — and reports, per
mechanism, the number the thesis actually needs:

    detector-achieved  /  oracle-achieved

together with the detector's precision and recall at the operating point used,
so a reader can attribute whatever is missing to *detection* rather than to the
mechanism (or the other way round).

Arms
----
``native``                 the untouched context.
``native_repeat``          the same prompt, an independent second call: the
                           in-harness A/A floor. The oracle probe measured it at
                           literally 0/0 discordant over 100 questions, so every
                           effect below is read against a measured zero.
``detector_suppress``      ``ReadFactPolicy("suppress")`` — every stale member of
                           every VERIFIED conflict group in the retrieved pool is
                           deleted. This is deliberately NOT the oracle edit
                           restricted to the queried key: the detector does not
                           know which key is queried, so it cleans the whole
                           page, and that collateral is part of what is measured.
``detector_demote_late``   ``ReadFactPolicy("demote_late")`` — every verified
                           group's LATEST carrier moves to the END of the
                           context in ascending serial order; nothing is deleted.
``detector_anti``          the mirror: the same LATEST carriers move to the
                           FRONT. **Measurement only** — no such mechanism exists
                           in ``hnav/core/``. It is the direction control: if
                           placement is the mechanism at all, ``demote_late`` and
                           ``anti`` must move the number in OPPOSITE directions.

Where the detector's inputs come from
-------------------------------------
``hnav/_out/stage1_prepass_<subset>.json`` (T11): the campaign-faithful chunk
ranking per question, the retrieval signals, the candidate pool built by the
adapter's own :func:`~hnav.adapters.mab_adapter.select_pool`, the pair geometry,
and the real DeBERTa-v3 bidirectional NLI scores. Those scores are replayed into
the REAL ``ReadGate`` through ``calibrate_read_policy.ReplayNLI`` — imported, so
no second gate exists to drift from the first.

Fact VECTORS come from the shared embedding cache and are then PROVEN to be the
prepass's own: every pair cosine recomputed from the loaded vectors must
reproduce the prepass's stored ``cos`` to 1e-6, over every pair of every
question, or the run refuses. The namespace is auto-detected across the
historical cache keys because ``cache_key`` gained its ``max_length`` component
after this prepass was written (T12) — and a stale cache hit is worse than a
miss, because it looks like a result.

Two declared limitations of that input, neither configurable away
----------------------------------------------------------------
1. The prepass's CHUNK vectors were embedded under the 512-token truncation
   defect while a chunk is ~4096 tokens, so ``nmargin``/``H_z`` — and only those
   — are contaminated. They feed exactly one thing: the gate's ambiguity
   precondition, which the frozen operating point therefore disables
   (``ambiguity_mode="none"``; the full argument is in :func:`select`). FACT
   vectors are unaffected — one short sentence is far under 512 tokens — and the
   cosine proof above shows the vectors used are the ones the prepass used.
2. The prepass predates NLI-config stamping. ``--allow-unstamped-prepass``
   accepts it; the artifact records that it did, alongside the measured
   character-length headroom that makes it harmless in this arena (a fact pair
   is tens of tokens against a 256-token budget).

Harness
-------
Identical to the probe's and imported from it: the whole context as a single
``Memory 1:`` block in context order, the benchmark's system message and query
template, ``substring_exact_match``, ``generation_max_length=10``, the frozen
:8003 chat server. Identical on purpose — the headline is a RATIO against the
oracle arms, and a ratio taken across two harnesses means nothing.

Arms are built with the probe's own ``suppress`` / ``move_to_end`` /
``move_to_front`` helpers and then re-derived through the SHIPPED page-contract
path (:func:`~hnav.adapters.mab_adapter.page_edit`) and asserted byte-identical.
So the measurement exercises the code that would ship, and the artifact reports
``n_page_edit_mismatch`` — which must be 0 — instead of leaving it to trust.

Split discipline
----------------
``sh_6k`` + ``sh_32k`` only; ``sh_64k``/``sh_262k`` are refused outright
(exit 2). The operating point is chosen by ``--select`` from DETECTION quality
alone — no LLM, no accuracy, no gold answer — and frozen to
``stage0_results/stage1_operating_point.json`` BEFORE any arm is graded.
``config.require_not_live()``: a measurement taken under intervention is not a
measurement.

Running it
----------
    python hnav/stage1/detector_gap.py --select                 # freeze, no LLM
    python hnav/stage1/detector_gap.py --subsets sh_6k --dry-run
    python hnav/stage1/detector_gap.py --subsets sh_6k --smoke-llm
    HNAV_LLM_BASE_URL=http://localhost:8003/v1 \
        python hnav/stage1/detector_gap.py --subsets sh_6k
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import (MABAdapter, _fact_separator,  # noqa: E402
                                       explode_facts, fact_spans, page_edit)
from hnav.core import read_gate as _rg  # noqa: E402
from hnav.core.read_gate import GateThresholds, ReadGate  # noqa: E402
from hnav.core.read_policy import demote_ids, suppress_ids  # noqa: E402
from hnav.core.types import MemoryRecord  # noqa: E402
from hnav.labeling.counterfactual import (normalize_answer,  # noqa: E402
                                          substring_exact_match)
from hnav.stage0.m2_retrieval_calibration import build_chunks  # noqa: E402
from hnav.labeling.question_strata import (STRATA, classify_questions,  # noqa: E402
                                           key_members)
# Imported, never re-transcribed: the gate replay, the grid axes, and the whole
# prompting / statistics harness the oracle numbers were produced with.
from hnav.stage1.calibrate_read_policy import (AMB_GRID, CALIBRATION,  # noqa: E402
                                               COS_GRID, FILTER_GRID,
                                               GENERATION_MAX_TOKENS, NLI_GRID,
                                               QUERY_TEMPLATE,
                                               R_MIN_GRID_LABELS, ReplayNLI,
                                               SYSTEM_MESSAGE, _CachedQR,
                                               build_user_prompt, r_min_of)
from hnav.stage1.stale_suppression_probe import (_acc, _sha,  # noqa: E402
                                                 build_prompt, move_to_end,
                                                 move_to_front, paired_cells,
                                                 primacy_stub, render_context,
                                                 split_context, suppress)

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
ORACLE_DIR = REPO / "stage0_results/stage1"
OPERATING_POINT = REPO / "stage0_results/stage1_operating_point.json"
CONFIRMATORY = ("sh_64k", "sh_262k")
ARMS = ("native", "native_repeat", "detector_suppress", "detector_demote_late",
        "detector_anti")
EDIT_ARMS = ("detector_suppress", "detector_demote_late", "detector_anti")
# Which oracle arm each detector arm is the achievable counterpart of.
ORACLE_COUNTERPART = {"detector_suppress": "oracle_suppress",
                      "detector_demote_late": "oracle_recency",
                      "detector_anti": "anti"}
COS_TOLERANCE = 1e-6
HARNESSES = ("whole_context", "retrieval")
# How the page is laid out for the model, per harness. Recorded in every
# artifact, because an artifact that misdescribes its own prompt shape is worse
# than one that omits it.
WHOLE_CONTEXT_SHAPE = r"RAGSystem: 'Memory 1:\n<whole context>\n' + templated query"
RETRIEVAL_SHAPE = (r"RAGSystem: 'Memory i:\n<chunk>' for the top-k retrieved "
                   r"chunks in similarity-rank order + templated query")
CHUNK_SIZE = 4096                       # the benchmark's own chunk_text_into_sentences
# The benchmark's OWN top-k page per question, produced by its own encoder
# (bfloat16, untruncated) and recorded by hnav/deploy/refit_chunk_embeddings.py.
# Re-encoding cannot reproduce it: matching dtype and truncation gets H-Nav's
# chunk vectors to min cosine 0.99997 of the benchmark's, and the sh_64k page
# still agrees on only 26/100 questions, because 17 tightly-clustered chunks
# reshuffle at the top-10/11 boundary under a 3e-5 perturbation. So the page is
# READ from the benchmark's vectors rather than recomputed from ours.
BENCHMARK_PAGES = REPO / "stage0_results/stage1/chunk_embedding_refit.json"
# r_min_label ordering, tightest first — used only as a deterministic tie-break.
R_RANK = {"frozen": 0, "loose": 1, "off": 2}
# Harm taxonomy, registered in pre-registration v2 Amendment 2. Tested in the
# order listed; EVERY class counts as harm and every class enters the McNemar
# b-cell exactly as before. The taxonomy is diagnostic, not permissive: the
# protective claim is voided by any unique-stratum harm that is not
# `malformed_generation`, which is the registered rule restated unchanged.
HARM_CLASSES = ("gold_cut", "malformed_generation", "refusal_after_edit",
                "information_loss")
MALFORMED_RATIO = 0.8
# Shapes the evaluator scores as a non-answer. "does not contain" is the exact
# phrasing observed on sh_32k retrieval q14, the case that named this class.
REFUSAL_MARKERS = ("does not contain", "not contain any", "no fact",
                   "cannot find", "no information", "not provided")
AMB_RANK = {"all": 0, "any": 1, "none": 2}


# ── ground truth, from fact TEXT alone (no answers involved) ─────────────────
def fact_table(item: dict) -> dict:
    """Everything the detector's ground truth needs, decided by the validated
    parser without reading a single answer.

    ``by_id``       fact id -> ``(serial, text, key, object)``
    ``superseded``  fact ids for which a LATER fact of the same key carries a
                    DIFFERENT object — genuinely stale under the benchmark's own
                    stated rule.
    ``latest``      key -> highest serial carrying that key.
    ``latest_obj``  key -> that fact's object.
    """
    members = key_members(item)
    by_id, superseded, latest, latest_obj = {}, set(), {}, {}
    for key, rows in members.items():
        top = max(rows)
        latest[key], latest_obj[key] = top[0], top[2]
        for s, text, obj in rows:
            by_id[f"fact:{s}"] = (s, text, key, obj)
            if any(s2 > s and o2 != obj for s2, _, o2 in rows):
                superseded.add(f"fact:{s}")
    return {"by_id": by_id, "superseded": superseded, "latest": latest,
            "latest_obj": latest_obj, "members": members}


def gt_pairs(ids, by_id) -> set[tuple[str, str]]:
    """Unordered pairs of ``ids`` that are TRUE supersession pairs: same
    ``(relation, subject)`` key, different object. Same definition
    ``calibrate_read_policy`` scores its false-verified rate against."""
    known = [i for i in ids if i in by_id]
    out = set()
    for a in range(len(known)):
        ia = known[a]
        ka, oa = by_id[ia][2], by_id[ia][3]
        if ka is None:
            continue
        for b in range(a + 1, len(known)):
            ib = known[b]
            if ka == by_id[ib][2] and oa != by_id[ib][3]:
                out.add((min(ia, ib), max(ia, ib)))
    return out


def count_gt_pairs(ids, by_id) -> int:
    """``len(gt_pairs(...))`` in linear time — a full page can hold thousands of
    facts and the quadratic form is only affordable on the 50-fact pool."""
    per_key: dict[tuple, list[str]] = {}
    for i in ids:
        row = by_id.get(i)
        if row and row[2] is not None:
            per_key.setdefault(row[2], []).append(row[3])
    total = 0
    for objs in per_key.values():
        n = len(objs)
        same = 0
        counts: dict[str, int] = {}
        for o in objs:
            counts[o] = counts.get(o, 0) + 1
        for c in counts.values():
            same += c * (c - 1) // 2
        total += n * (n - 1) // 2 - same
    return total


# ── vectors: the prepass's own, proven rather than assumed ───────────────────
class _FailOnMiss:
    dim = 0

    def encode(self, texts):
        raise RuntimeError(f"{len(texts)} text(s) missing from the cache, "
                           f"first={texts[0][:60]!r}")


def cosine_error(vecs: np.ndarray, order: list[str], prepasses) -> float:
    """Largest absolute deviation between a pair cosine recomputed from
    ``vecs`` and the cosine the prepass recorded for that pair. Zero pairs is
    reported as ``inf`` — an unverifiable namespace is not a verified one."""
    idx = {fid: i for i, fid in enumerate(order)}
    worst, n = 0.0, 0
    m = np.asarray(vecs, dtype=np.float64)
    for pp in prepasses:
        for q in pp["questions"]:
            for p in q["pairs"]:
                if p["a"] not in idx or p["b"] not in idx:
                    continue
                got = float(m[idx[p["a"]]] @ m[idx[p["b"]]])
                worst = max(worst, abs(got - float(p["cos"])))
                n += 1
    return worst if n else float("inf")


def load_fact_vectors(cfg, order, texts, prepasses,
                      whitener=None) -> tuple[np.ndarray, str, float]:
    """Fact vectors from the shared cache, under whichever namespace the prepass
    was written with — auto-detected, then proven by :func:`cosine_error`.

    [ABTT] ``whitener`` is applied immediately after the cache read and BEFORE
    the proof, so the guard keeps its full force: it now shows that the cache
    namespace *and* the whitening artifact together reproduce the geometry the
    prepass measured. Skipping the guard for whitened runs, or comparing raw
    cosines against a whitened prepass, would reintroduce exactly the T12
    failure this function exists to prevent.
    """
    from hnav.core.embedding import DiskCachedEmbedder, cache_key
    keys = [f"{cfg.embed_model}|{cfg.embed_dtype}".replace("/", "_"),
            cache_key(cfg.embed_model, cfg.embed_dtype, 512),
            cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length)]
    errors = []
    for key in dict.fromkeys(keys):
        try:
            emb = DiskCachedEmbedder(_FailOnMiss(), cfg.emb_cache_dir, key,
                                     persist=False)
            vecs = emb.encode(list(texts))
            if whitener is not None:
                vecs = whitener.transform(np.asarray(vecs, dtype=np.float64))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: {str(e)[:120]}")
            continue
        err = cosine_error(vecs, order, prepasses)
        if err <= COS_TOLERANCE:
            return vecs, key, err
        errors.append(f"{key}: max cosine error {err:.3g} > {COS_TOLERANCE}")
    raise SystemExit(
        " REFUSED: no embedding-cache namespace reproduces the prepass "
        "geometry.\n   " + "\n   ".join(errors) +
        "\n The gate would then be running on vectors that are not the ones the "
        "prepass\n measured, which is exactly the failure T12 exists to remove. "
        "Re-run the\n prepass with the current embedder, or point HNAV_CACHE_DIR "
        "at its cache.")


# ── prepass -> gate input ────────────────────────────────────────────────────
def build_records(order, texts, table, vecs) -> dict[str, MemoryRecord]:
    """``fact id -> MemoryRecord`` with the vector, the serial as ``version``
    and the parsed key/object in metadata — the exact shape the live adapter
    hands the gate (``MABAdapter.candidate_for``)."""
    out = {}
    for i, fid in enumerate(order):
        serial, text, key, obj = table["by_id"][fid]
        out[fid] = MemoryRecord(id=fid, text=text, vector=vecs[i], version=serial,
                                metadata={"key": key, "object": obj})
    return out


def make_gate(cell, replay, ces=None) -> ReadGate:
    """[E2E] ``cell["pair_filter"]`` is the identity screen: ``True`` = the
    parser's same-key check (the shipped arms), ``"ces"`` = the frozen CES
    artifact at ``cell["ces_tau"]`` (relation from the parser, subject identity
    from geometry), ``False`` = no identity screen at all (the known-dangerous
    setting the Faz A audit measured — allowed only for the preregistered
    abtt_noparser arm, where measuring that danger IS the experiment)."""
    pf = cell["pair_filter"]
    if pf == "idonly":
        # [E2E-4] symbolic identity ALONE. Same screen as the shipped parser
        # arm; what differs is the cell's nli_contradiction, which the idonly
        # grid drives to 0 — the arm exists to measure whether the semantic
        # contradiction gate is the binding constraint once identity is
        # certified. Harm is structurally 0 (see SELECTION_RULE notes).
        pair_filter = MABAdapter.same_key_pair
    elif pf in ("ces", "fusion", "geo"):
        if ces is None:
            raise SystemExit(f" REFUSED: this operating point uses the {pf!r} "
                             "screen; pass --pair-screen and its artifact.")
        tau = cell["ces_tau"] if pf == "geo" else float(cell["ces_tau"])
        pair_filter = ces.pair_filter(tau)
    elif pf:
        pair_filter = MABAdapter.same_key_pair
    else:
        pair_filter = None
    return ReadGate(replay, GateThresholds(
        cos_pair=cell["cos_pair"], r_min=cell["r_min"],
        ambiguity_mode=cell["ambiguity_mode"],
        nli_contradiction=cell["nli_contradiction"]),
        pair_filter=pair_filter)


def decide_all(prepass, recs, cells, replay, ces=None):
    """``{question index: {cell index: GateDecision}}`` for one subset.

    Questions outer, cells inner, with ``_CachedQR`` installed exactly as
    ``calibrate_read_policy.evaluate`` installs it (and restored), so an
    identical leave-one-out QR is computed once per question rather than once
    per (question, cell).
    """
    gates = [make_gate(c, replay, ces) for c in cells]
    qr_cache = _CachedQR()
    real_qr = _rg.qr_residual
    _rg.qr_residual = qr_cache
    out: dict[int, list] = {}
    try:
        for q in prepass["questions"]:
            pool = [recs[f] for f in q["pool"] if f in recs]
            if len(pool) >= 2:
                qr_cache.load(np.stack([np.asarray(r.vector, dtype=np.float64)
                                        for r in pool]))
            sig = {"nmargin": q["nmargin"], "H_z": q["H_z"]}
            out[q["index"]] = [g.decide(pool, sig,
                                        latest_key=MABAdapter.latest_key)
                               for g in gates]
    finally:
        _rg.qr_residual = real_qr
    return out


# ── detection metrics ────────────────────────────────────────────────────────
def blank_metrics() -> dict:
    return {"n_questions": 0, "tp": 0, "fp": 0, "gt_pool": 0, "gt_page": 0,
            "n_suppressed": 0, "n_suppressed_superseded": 0,
            "n_suppressed_same_value": 0, "n_suppressed_harmful": 0,
            "n_demoted": 0, "n_demoted_is_latest": 0,
            "n_conflicted": 0, "n_conflicted_hit": 0, "n_conflicted_gold_cut": 0,
            "n_unique": 0, "n_unique_touched": 0, "n_touched": 0,
            "n_ambiguous_pass": 0}


def classify_drops(m: dict, drop, table) -> None:
    """Split a suppression set into superseded / same-value / HARMFUL.

    The criterion is what the edit does to the PAGE, not what one fact looks
    like in isolation: for every key the drop set touches, take the members that
    survive and ask whether the key's newest surviving value is still the key's
    newest value. If it is not — or if the key loses every member — then every
    fact dropped from that key counts as harmful, because the page now says
    something different about that key than the corpus does.

    Getting this per-fact instead of per-key is exactly the mistake that would
    let a single-member key be waved through as a "duplicate restatement": its
    only fact IS its latest value, so ``object == latest_object`` is true right
    up to the moment deleting it erases the key.
    """
    by_id, members = table["by_id"], table["members"]
    dropped_serials = {by_id[f][0] for f in drop if f in by_id}
    per_key: dict[tuple, list[str]] = {}
    unknown = 0
    for fid in drop:
        row = by_id.get(fid)
        if row is None or row[2] is None:
            unknown += 1                    # unparsed: no defensible reasoning
            continue
        per_key.setdefault(row[2], []).append(fid)
    m["n_suppressed_harmful"] += unknown

    for key, fids in per_key.items():
        rows = sorted(members[key])
        survivors = [r for r in rows if r[0] not in dropped_serials]
        if not survivors or survivors[-1][2] != rows[-1][2]:
            m["n_suppressed_harmful"] += len(fids)
            continue
        for fid in fids:
            if fid in table["superseded"]:
                m["n_suppressed_superseded"] += 1
            else:
                m["n_suppressed_same_value"] += 1


def score_decision(m: dict, dec, table, strat, pool_ids, page_gt: int) -> None:
    """Accumulate one question's detection outcome into ``m``.

    Everything except the two ``n_conflicted_*`` counters is parse-derived: it
    uses fact text and serial order only. The conflicted-stratum counters use
    the question->key assignment from ``question_strata``, which is gold-derived
    — they are reported for ATTRIBUTION and are never used to choose an
    operating point (see :func:`select`).
    """
    by_id = table["by_id"]
    m["n_questions"] += 1
    if dec.ambiguous:
        m["n_ambiguous_pass"] += 1
    m["gt_pool"] += len(gt_pairs(pool_ids, by_id))
    m["gt_page"] += page_gt

    for pc in dec.pair_checks:
        if not pc.verified:
            continue
        ra, rb = by_id.get(pc.id_a), by_id.get(pc.id_b)
        true_pair = bool(ra and rb and ra[2] is not None and ra[2] == rb[2]
                         and ra[3] != rb[3])
        m["tp" if true_pair else "fp"] += 1

    drop = suppress_ids(dec)
    move = demote_ids(dec)
    m["n_suppressed"] += len(drop)
    m["n_demoted"] += len(move)
    if drop or move:
        m["n_touched"] += 1
    classify_drops(m, drop, table)
    for fid in move:
        row = by_id.get(fid)
        if row and row[2] is not None and table["latest"][row[2]] == row[0]:
            m["n_demoted_is_latest"] += 1

    if strat["stratum"] == "unique":
        m["n_unique"] += 1
        if drop or move:
            m["n_unique_touched"] += 1
    elif strat["stratum"] == "conflicted":
        m["n_conflicted"] += 1
        key = tuple(strat["key"])
        top = table["latest"][key]
        others = [s for s in strat["member_serials"] if s != top]
        cut = {int(f.split(":")[1]) for f in drop}
        if any(s in cut for s in others) and top not in cut:
            m["n_conflicted_hit"] += 1
        if strat["target_serial"] in cut:
            m["n_conflicted_gold_cut"] += 1


def finish_metrics(m: dict) -> dict:
    ver = m["tp"] + m["fp"]
    out = dict(m)
    out["pair_precision"] = (m["tp"] / ver) if ver else None
    out["pair_recall_pool"] = (m["tp"] / m["gt_pool"]) if m["gt_pool"] else None
    out["pair_recall_page"] = (m["tp"] / m["gt_page"]) if m["gt_page"] else None
    out["fact_precision"] = ((m["n_suppressed_superseded"] / m["n_suppressed"])
                             if m["n_suppressed"] else None)
    out["question_recall_conflicted"] = ((m["n_conflicted_hit"] / m["n_conflicted"])
                                         if m["n_conflicted"] else None)
    out["coverage"] = (m["n_touched"] / m["n_questions"]) if m["n_questions"] else None
    return out


# ── the grid and the pre-registered choice ───────────────────────────────────
CES_TAU_GRID = (-0.05, 0.0, 0.05, 0.10)   # preregistered 2026-08-27
# fusion thresholds are LOGITS of the calibration-fit logistic screen;
# preregistered 2026-08-27 (E2E-2)
FUSION_TAU_GRID = (0.0, 2.0, 4.0, 6.0, 8.0)
# geo thresholds are ANCHORED-MARGIN units of the GEO identity screen
# (0 = its calibration zero-FP anchor conjunction; negative explores the
# slack the key-level harm rule allows); preregistered 2026-08-29 (E2E-3,
# stage0_results/geometry_filter/GEO_PREREG.md)
GEO_TAU_GRID = (-0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25)


def grid_cells(cos_grid=None, pair_screen: str = "parser",
               ces_grid=None, nli_grid=None) -> list[dict]:
    """The SAME axes ``calibrate_read_policy`` declared for the rerank
    calibration — reused rather than redeclared so the two searches cannot
    silently diverge.

    [ABTT] ``cos_grid`` overrides the cosine axis, because 0.90/0.92/0.94 are
    coordinates in the RAW space. When it is overridden the ``loose`` r_min is
    also re-derived per cell as ``sqrt(1 - cos^2)`` — the documented coupling
    (HNAV_HOW_IT_WORKS §11.5) that keeps the residual screen a pass-through for
    whatever the cosine screen admitted. The frozen constant R_LOOSE=0.44 is
    that same quantity evaluated at cos=0.90 and is meaningless at cos=0.30.
    The raw path is untouched: with no override the constant is used verbatim.
    """
    custom = cos_grid is not None and tuple(cos_grid) != tuple(COS_GRID)
    # [E2E] the identity-screen axis depends on which arm is being selected:
    #   parser  -> the shipped grid (True, False), unchanged;
    #   none    -> pinned False (the abtt_noparser arm: no identity screen);
    #   ces     -> pinned "ces", crossed with the preregistered tau grid.
    if pair_screen == "parser":
        filter_axis = [(f, None) for f in FILTER_GRID]
    elif pair_screen == "idonly":
        # [E2E-4] the same same-key screen, pinned on, as its own arm
        filter_axis = [("idonly", None)]
    elif pair_screen == "none":
        filter_axis = [(False, None)]
    elif pair_screen in ("ces", "fusion", "geo"):
        taus = tuple(ces_grid) if ces_grid else (
            {"ces": CES_TAU_GRID, "fusion": FUSION_TAU_GRID,
             "geo": GEO_TAU_GRID}[pair_screen])
        # [E2E-3] geo taus may be 'tw:tp' rectangle strings (per-axis
        # offsets); everything else stays float
        filter_axis = [(pair_screen,
                        t if (pair_screen == "geo" and isinstance(t, str))
                        else float(t)) for t in taus]
    else:
        raise ValueError(f"unknown pair_screen {pair_screen!r}")
    cells = []
    for cos in (COS_GRID if cos_grid is None else tuple(cos_grid)):
        for rlab in R_MIN_GRID_LABELS:
            r = r_min_of(rlab)
            if custom and rlab == "loose":
                r = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
            for amb in AMB_GRID:
                for tau in (NLI_GRID if nli_grid is None else tuple(nli_grid)):
                    for filt, ces_tau in filter_axis:
                        cell = {"cos_pair": cos, "r_min_label": rlab,
                                "r_min": r,
                                "ambiguity_mode": amb,
                                "nli_contradiction": tau,
                                "pair_filter": filt}
                        if ces_tau is not None:
                            cell["ces_tau"] = ces_tau
                        cells.append(cell)
    return cells


SELECTION_RULE = {
    "require": [
        "pair_filter is True",
        "n_suppressed_harmful == 0",
    ],
    "maximise": "pair_recall_pool",
    "tie_break": ["higher cos_pair", "higher nli_contradiction",
                  "tighter r_min (frozen<loose<off)",
                  "stricter ambiguity_mode (all<any<none)"],
    "fit_on": "detection quality only - no LLM, no accuracy, no gold answer",
    "split": "sh_6k + sh_32k (calibration) ONLY",
}


def selection_rule_for(pair_screen: str) -> dict:
    """[E2E] The rule for an alternative identity screen. Documented method
    change (2026-08-27): the shipped ``pair_filter is True`` requirement said
    "the parser screen must be ON" — for the preregistered geometry arms the
    identity screen is pinned by the arm itself, so the requirement becomes
    membership in that arm's screen. ``n_suppressed_harmful == 0`` — the
    information-loss veto — stays hard for every screen; a screen that cannot
    reach it does not get an operating point, it gets a null result."""
    if pair_screen == "parser":
        return SELECTION_RULE
    rule = dict(SELECTION_RULE)
    rule["require"] = [f"pair_filter == {'False' if pair_screen == 'none' else pair_screen!r}"
                       " (pinned by the arm)",
                       "n_suppressed_harmful == 0"]
    if pair_screen in ("ces", "geo"):
        rule["tie_break"] = ["higher ces_tau (stricter identity screen)"] \
            + list(SELECTION_RULE["tie_break"])
    return rule


def select(cells: list[dict], pair_screen: str = "parser") -> dict | None:
    """Apply :func:`selection_rule_for`, frozen before any cell was scored.

    ``pair_filter is True`` is a REQUIREMENT, not a preference. The Faz A audit
    measured the NLI cross-encoder rubber-stamping same-template /
    different-subject pairs as bidirectional contradiction, and the T11
    calibration showed what that does downstream: with the identity screen OFF,
    ~86% of verifications were spurious and the resulting "gains" were noise
    (``STAGE1_NULL_ANALIZI.md`` §3). A gain built on a detector that cannot tell
    subjects apart is not a gain.

    ``n_suppressed_harmful == 0`` is the deletion criterion: not one suppressed
    fact may carry its key's CURRENT value. A fact that is genuinely superseded
    is safe to drop, and so is a same-valued duplicate; anything else is
    information loss, which no recall figure buys back.

    The objective is ``pair_recall_pool`` — TRUE supersession pairs verified,
    over the true supersession pairs present in the candidate pool. It is
    entirely parse-derived. The question-level and accuracy figures are reported
    but deliberately excluded from the objective: fitting the gate on either
    would make the calibration split a training set for the outcome.

    Note what a "none" ambiguity mode means if it wins: the frozen Stage-0
    ``nmargin``/``H_z`` precondition is disabled. That is defensible here and
    only here, for three stated reasons — those two signals are computed from
    CHUNK embeddings truncated at 512 of ~4096 tokens (the T12 defect, not yet
    re-fit because re-embedding is blocked while both servers hold the GPUs), so
    they are the one contaminated input the gate has; they are the dominant
    recall bottleneck; and the volume-limiting job they were doing is now done
    by the identity screen and the bidirectional NLI at measured precision 1.00.
    A campaign that re-fits the embeddings must revisit this.
    """
    want = {"parser": True, "none": False, "ces": "ces",
            "fusion": "fusion", "geo": "geo", "idonly": "idonly"}[pair_screen]
    feasible = [c for c in cells
                if c["pair_filter"] == want
                and c["metrics"]["n_suppressed_harmful"] == 0
                and c["metrics"]["pair_recall_pool"] is not None]
    if not feasible:
        return None
    def _tau_rank(c) -> float:
        """Stricter-first tie-break value; 'tw:tp' rectangles rank by the
        sum of their per-axis offsets."""
        t = c.get("ces_tau")
        if t is None:
            return 0.0
        if isinstance(t, str) and ":" in t:
            a, b = t.split(":", 1)
            return float(a) + float(b)
        return float(t)

    feasible.sort(key=lambda c: (-c["metrics"]["pair_recall_pool"],
                                 -_tau_rank(c),
                                 -c["cos_pair"], -c["nli_contradiction"],
                                 R_RANK[c["r_min_label"]],
                                 AMB_RANK[c["ambiguity_mode"]]))
    return feasible[0]


# ── arms ─────────────────────────────────────────────────────────────────────
def arm_facts(arm: str, facts, plan) -> list[tuple[int, str]]:
    """The fact list one arm shows the model. Built with the PROBE's own
    helpers, so the only difference between an oracle arm and its detector
    counterpart is which serials were chosen."""
    if arm in ("native", "native_repeat"):
        return list(facts)
    if arm == "detector_suppress":
        return suppress(facts, plan["suppress_serials"])
    out = list(facts)
    if arm == "detector_demote_late":
        for s in plan["demote_serials"]:            # ascending: newest ends LAST
            out = move_to_end(out, s)
        return out
    if arm == "detector_anti":
        for s in plan["demote_serials"]:            # ascending: newest ends FIRST
            out = move_to_front(out, s)
        return out
    raise ValueError(f"unknown arm {arm!r}")


def shipped_page(arm: str, context: str, plan) -> str | None:
    """The same edit through the SHIPPED page contract, or ``None`` for an arm
    that has no shipped mechanism (``detector_anti``)."""
    ids_s = [f"fact:{s}" for s in plan["suppress_serials"]]
    ids_d = [f"fact:{s}" for s in plan["demote_serials"]]
    if arm == "detector_suppress":
        return page_edit([context], drop_ids=ids_s)[0][0]
    if arm == "detector_demote_late":
        return page_edit([context], move_last_ids=ids_d)[0][0]
    return None


# ── the retrieval-path harness (prereg v2 section 10) ────────────────────────
# The deployed system never reads a whole-context block: it reads the top_k
# retrieved CHUNKS, numbered "Memory i:", in similarity-rank order. All T13
# calibration evidence is whole-context, which is a documented deviation that
# was justified by retrieval being COMPLETE on the calibration split (2 and 9
# chunks against top_k=10). It stops being justified at sh_64k, which is 17
# chunks. This harness closes that gap: on the calibration split it changes ONLY
# the block structure and order, so it isolates that variable from retrieval
# incompleteness, which appears only on the confirmatory subset.
#
# Here the arms ARE the shipped page contract - `page_edit` applied to the real
# retrieved page - rather than a probe-style rebuild that is then checked
# against it.


def benchmark_pages(subset: str, path: Path | None = None) -> list[list[int]] | None:
    """The benchmark's own top-k chunk indices per question, or ``None``.

    ``None`` means the artifact does not cover this subset, and the caller must
    then decide explicitly rather than silently falling back to a page H-Nav
    computed for itself — that is the whole point of Amendment 3.
    """
    # Resolved at CALL time, not bound at definition time, so the artifact path
    # stays a single source of truth that tests and callers can redirect.
    path = Path(path or BENCHMARK_PAGES)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("subsets", {}).get(subset)
    if not row or "benchmark" not in row.get("pages", {}):
        return None
    return [[int(i) for i in pg] for pg in row["pages"]["benchmark"]]


def chunk_texts_for(item, prepass) -> list[str]:
    """The benchmark's own chunks, rebuilt and CHECKED against the prepass.

    The prepass stores chunk ids and a fact->chunk map but not chunk text, so
    the text is regenerated with the benchmark's chunker and then verified:
    identical fact->chunk assignment, fact for fact. A silent chunker drift
    would otherwise put different bytes on the page than the geometry that
    drives the gate was measured on.
    """
    chunks, fallback = build_chunks(item["context"], CHUNK_SIZE)
    if fallback:
        raise SystemExit(
            " REFUSED: nltk/punkt is missing, so build_chunks fell back and "
            "these are not the benchmark's chunks.")
    got = {}
    for i, c in enumerate(chunks):
        for serial, _ in explode_facts(c):
            got[f"fact:{serial}"] = f"chunk:{i}"
    want = prepass["fact_chunk"]
    if got != want:
        diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise SystemExit(
            f" REFUSED: the rebuilt chunks disagree with the prepass on "
            f"{len(diff)} fact(s), first={diff[:3]}. The page would not be the "
            "one the gate's geometry was measured on.")
    return chunks


def page_move_front(page_texts, ids):
    """MEASUREMENT ONLY — the direction control's mirror of ``DEMOTE_LATE``.

    Deliberately NOT in ``hnav/adapters/``: no ``Decision`` action can produce a
    front-move, and putting one on the shipped surface would create a third
    mechanism nobody authorised. It reuses ``fact_spans`` and the adapter's own
    separator rule (imported, not re-transcribed) so that the bytes it moves are
    the same bytes ``page_edit`` would move, and it is exercised by
    ``detector_anti`` and nothing else.

    Facts are inserted in DESCENDING serial order before the first fact of the
    first chunk, so the newest ends up first — the exact mirror of
    ``page_edit``'s ascending append.
    """
    texts = list(page_texts)
    spans = [fact_spans(t) for t in texts]
    where: dict = {}
    for ci, sp in enumerate(spans):
        for fs in sp:
            where.setdefault(f"fact:{fs.serial}", (ci, fs))
    missing = [i for i in ids if i not in where]
    if missing:
        raise LookupError(f"{len(missing)} id(s) not on this page, "
                          f"first={missing[0]!r}")
    if not ids:
        return texts

    sep = _fact_separator(spans, texts)
    moved = sorted(((where[i][1].serial, i) for i in ids), reverse=True)
    moved_bytes = [texts[where[i][0]][where[i][1].start: where[i][1].own_end]
                   for _, i in moved]

    cuts: dict = {}
    for i in ids:
        ci, fs = where[i]
        cuts.setdefault(ci, []).append(fs)
    out = list(texts)
    for ci, ss in cuts.items():
        t = out[ci]
        for fs in sorted(ss, key=lambda x: x.del_start, reverse=True):
            t = t[: fs.del_start] + t[fs.del_end:]
        out[ci] = t

    head = fact_spans(out[0])
    pos = head[0].start if head else len(out[0].rstrip())
    out[0] = out[0][:pos] + "".join(m + sep for m in moved_bytes) + out[0][pos:]
    return out


def containment_violations(page, plan) -> list[str]:
    """Ids the policy names that the page does not carry.

    THE guard. The candidate pool is built from a page; if that page is not the
    page the model will be shown, the policy can name a fact that is simply not
    there. ``page_edit`` then raises, the live adapter fails open to the native
    page, and the artifact records a run in which the intervention never
    happened — which is indistinguishable, from the outside, from the
    intervention having no effect. On a one-shot confirmatory subset that is the
    worst failure available: a wiring bug wearing the costume of a null result.

    So containment is checked EXPLICITLY, per question, before anything is sent,
    and a violation refuses the run rather than degrading it.
    """
    present = {f"fact:{sp.serial}" for t in page for sp in fact_spans(t)}
    named = [f"fact:{x}" for x in plan["suppress_serials"]] +             [f"fact:{x}" for x in plan["demote_serials"]]
    return sorted(set(named) - present)


def pool_violations(page, pool_ids) -> list[str]:
    """Pool members the page does not carry (G1).

    Stronger than :func:`containment_violations` and checked alongside it: the
    named ids are a subset of the pool, so a pool that already escapes the page
    is the upstream bug, caught one step earlier and against the SAME page
    object that is handed to ``page_edit``.
    """
    present = {f"fact:{sp.serial}" for t in page for sp in fact_spans(t)}
    return sorted(set(pool_ids) - present)


def retrieval_arm_page(arm: str, page, plan):
    """The page one arm shows the model, through the SHIPPED edit path."""
    if arm in ("native", "native_repeat"):
        return list(page)
    drop = [f"fact:{s}" for s in plan["suppress_serials"]]
    move = [f"fact:{s}" for s in plan["demote_serials"]]
    if arm == "detector_suppress":
        return page_edit(page, drop_ids=drop)[0]
    if arm == "detector_demote_late":
        return page_edit(page, move_last_ids=move)[0]
    if arm == "detector_anti":
        return page_move_front(page, move)
    raise ValueError(f"unknown arm {arm!r}")


def page_facts(page) -> list:
    return sorted(f for t in page for f in explode_facts(t))


def retrieval_integrity(arm: str, page, edited, plan) -> bool:
    """Did the edit do exactly what the arm claims, and nothing else?

    Same chunk count always; the fact multiset is preserved by both placement
    arms and reduced by exactly the named facts by ``SUPPRESS``. Returns True on
    a violation, so the caller can count violations the way the whole-context
    harness counts ``page_edit`` mismatches — and refuse the run before sending
    anything.
    """
    if len(edited) != len(page):
        return True
    before, after = page_facts(page), page_facts(edited)
    if arm in ("native", "native_repeat"):
        return after != before
    if arm == "detector_suppress":
        drop = set(plan["suppress_serials"])
        return after != sorted(f for f in before if f[0] not in drop)
    return after != before


def plan_subset(item, name, prepass, decisions, cell_i, table,
                max_questions=None, harness: str = "whole_context",
                page_source: str | None = None) -> dict:
    """Build every arm for one subset, under either harness.

    ``whole_context`` reproduces the oracle probe exactly (one ``Memory 1:``
    block in context order) and CHECKS each edit against the shipped
    ``page_edit``. ``retrieval`` puts the benchmark's own top-k chunks on the
    page in rank order and performs the edit THROUGH ``page_edit``, checking
    instead that the result did exactly what the arm claims. Either way the
    counter is ``n_page_edit_mismatch`` and either way a non-zero value refuses
    the run before a call is sent.
    """
    if harness not in HARNESSES:
        raise ValueError(f"harness={harness!r} not in {HARNESSES}")
    preamble, facts = split_context(item["context"])
    strata = {r["index"]: r for r in classify_questions(item)}
    # Argument validation BEFORE any work: choosing the page source is the one
    # decision that cannot be defaulted, so it must fail before the chunker does.
    bpages = None
    if harness == "retrieval":
        if page_source is None:
            raise ValueError(
                "the retrieval harness needs an explicit page_source: "
                "'benchmark' reads the benchmark's own top-k page (Amendment 3), "
                "'prepass' uses the ranking H-Nav computed for itself. There is "
                "no default, because the two differ on 74% of sh_64k questions.")
        if page_source == "benchmark":
            bpages = benchmark_pages(name)
            if bpages is None:
                raise SystemExit(
                    f" REFUSED: no benchmark page recorded for {name} in "
                    f"{_rel(BENCHMARK_PAGES)}. Run "
                    "hnav/deploy/refit_chunk_embeddings.py first; do NOT fall "
                    "back to H-Nav's own ranking.")
    chunks = chunk_texts_for(item, prepass) if harness == "retrieval" else None

    questions, mismatches, pages, containment = [], 0, [], []
    page_edit_errors: list[dict] = []
    n_fired = n_suppressed_total = n_edits_applied = 0
    for q in prepass["questions"]:
        if max_questions and len(questions) >= max_questions:
            break
        dec = decisions[q["index"]][cell_i]
        plan = {
            "suppress_serials": sorted(int(f.split(":")[1])
                                       for f in suppress_ids(dec)),
            "demote_serials": sorted(int(f.split(":")[1])
                                     for f in demote_ids(dec)),
            "n_groups": len(dec.groups),
            "n_pairs_verified": dec.n_pairs_verified,
        }
        rec = strata[q["index"]]
        arms = {}
        if harness == "whole_context":
            for arm in ARMS:
                fl = arm_facts(arm, facts, plan)
                text = render_context(preamble, fl)
                if arm in ("detector_suppress", "detector_demote_late"):
                    if shipped_page(arm, item["context"], plan) != text:
                        mismatches += 1
                arms[arm] = {"prompt": build_prompt(text, rec["question"]),
                             "n_facts": len(fl)}
        else:
            if bpages is not None:
                page = [chunks[i] for i in bpages[q["index"]]]
            else:
                page = [chunks[int(c.split(":")[1])] for c in q["top_ids"]]
            missing = containment_violations(page, plan)
            if missing:
                containment.append({"index": q["index"], "kind": "named",
                                    "missing": missing})
            escaped = pool_violations(page, q.get("pool", []))
            if escaped:
                containment.append({"index": q["index"], "kind": "pool",
                                    "missing": escaped[:10]})
            if plan["suppress_serials"] or plan["demote_serials"]:
                n_fired += 1
            n_suppressed_total += len(plan["suppress_serials"])
            pages.append(len(page))
            message = QUERY_TEMPLATE.format(question=rec["question"])
            for arm in ARMS:
                try:
                    edited = retrieval_arm_page(arm, page, plan)
                except Exception as e:  # noqa: BLE001 — counted, never swallowed
                    page_edit_errors.append({"index": q["index"], "arm": arm,
                                             "error": repr(e)[:200]})
                    edited = list(page)
                if retrieval_integrity(arm, page, edited, plan):
                    mismatches += 1
                if arm in ("detector_suppress", "detector_demote_late")                         and edited != list(page):
                    n_edits_applied += 1
                arms[arm] = {"prompt": build_user_prompt(edited, message),
                             "n_facts": len(page_facts(edited))}
        native_prompt = arms["native"]["prompt"]
        questions.append({
            "index": q["index"], "question": rec["question"],
            "truths": rec["truths"], "stratum": rec["stratum"],
            "key": list(rec["key"]) if rec["key"] else None,
            "target_serial": rec.get("target_serial"),
            # Recorded so that claims about the pool cap are checkable from the
            # artifact alone; the confirmatory run's first draft attributed a
            # missed prediction to the pool without this, and could not be
            # verified.
            "n_pool": len(q.get("pool", [])),
            "pool": list(q.get("pool", [])),
            "plan": plan, "arms": arms,
            "identical_to_native": {a: arms[a]["prompt"] == native_prompt
                                    for a in ARMS},
        })
    return {"subset": name, "n_facts": len(facts), "questions": questions,
            "n_page_edit_mismatch": mismatches, "harness": harness,
            "page_source": page_source,
            "n_containment_violations": len(containment),
            "containment_violations": containment[:20],
            "n_page_edit_errors": len(page_edit_errors),
            "page_edit_errors": page_edit_errors[:20],
            "positive_control": {
                "n_questions": len(questions),
                "n_questions_policy_fired": n_fired,
                # ACCUMULATED ACROSS THE TWO EDITING ARMS (suppress and
                # demote_late), so the expected value is n_questions_policy_fired
                # x n_editing_arms, NOT n_questions_policy_fired. The
                # pre-registration's VC8 wording ("must equal the number of
                # questions fired, expected 100") describes the PER-ARM figure;
                # both are reported so neither can be misread as a failure.
                "n_fact_edits_applied": n_edits_applied,
                "n_editing_arms": 2,
                "n_fact_edits_applied_per_arm": n_edits_applied / 2,
                "n_facts_suppressed": n_suppressed_total,
                "ok": bool(n_fired and n_suppressed_total
                           and n_edits_applied >= n_fired)},
            "n_chunks_total": len(chunks) if chunks is not None else None,
            "n_chunks_on_page": max(pages) if pages else None,
            "retrieval_complete": (len(chunks) <= max(pages)
                                   if chunks is not None and pages else None),
            "strata_counts": {s: sum(1 for q in questions if q["stratum"] == s)
                              for s in STRATA}}


def budget(plans) -> dict:
    rows, calls, chars = [], 0, 0
    for p in plans:
        prompts, c = set(), 0
        for q in p["questions"]:
            for arm in ARMS:
                if arm == "native_repeat":
                    continue
                t = q["arms"][arm]["prompt"]
                if t not in prompts:
                    prompts.add(t)
                    c += len(t)
        c += sum(len(q["arms"]["native"]["prompt"]) for q in p["questions"])
        n = len(prompts) + len(p["questions"])
        rows.append({"subset": p["subset"], "n_questions": len(p["questions"]),
                     "n_distinct_prompts": len(prompts),
                     "n_aa_repeats": len(p["questions"]), "n_calls": n,
                     "approx_prompt_tokens": c // 4,
                     "n_identical_to_native": {
                         a: sum(1 for q in p["questions"]
                                if q["identical_to_native"][a])
                         for a in EDIT_ARMS}})
        calls += n
        chars += c
    return {"per_subset": rows, "total_calls": calls,
            "approx_total_prompt_tokens": chars // 4}


# ── execution ────────────────────────────────────────────────────────────────
def make_answer_fn(args, cfg):
    if args.smoke_llm:
        return primacy_stub()
    from openai import OpenAI  # noqa: PLC0415 — lazy: no network at import

    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def answer(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                      {"role": "user", "content": prompt}],
            temperature=cfg.llm_temperature, max_tokens=GENERATION_MAX_TOKENS)
        return resp.choices[0].message.content or ""
    return answer


def run_subset(plan, answer_fn) -> dict:
    cache, per_question, n_calls = {}, [], 0
    for q in plan["questions"]:
        row = {"index": q["index"], "stratum": q["stratum"], "key": q["key"],
               "truths": q["truths"], "target_serial": q.get("target_serial"),
               "n_pool": q.get("n_pool"), "pool": q.get("pool"),
               "plan": q["plan"], "arms": {}}
        for arm in ARMS:
            prompt = q["arms"][arm]["prompt"]
            if arm == "native_repeat":
                out = answer_fn(prompt)              # deliberately uncached: A/A
                n_calls += 1
            else:
                if prompt not in cache:
                    cache[prompt] = answer_fn(prompt)
                    n_calls += 1
                out = cache[prompt]
            row["arms"][arm] = {
                "output": out,
                "correct": bool(substring_exact_match(out, q["truths"])),
                "n_facts": q["arms"][arm]["n_facts"],
                "prompt_chars": len(prompt), "prompt_sha": _sha(prompt),
                "identical_to_native": q["identical_to_native"][arm]}
        per_question.append(row)

    flags = {a: [r["arms"][a]["correct"] for r in per_question] for a in ARMS}
    by_stratum = {}
    for s in STRATA:
        idx = [i for i, r in enumerate(per_question) if r["stratum"] == s]
        if not idx:
            continue
        by_stratum[s] = {
            "arms": {a: _acc([flags[a][i] for i in idx]) for a in ARMS},
            "paired_vs_native": {
                a: paired_cells([flags["native"][i] for i in idx],
                                [flags[a][i] for i in idx])
                for a in ARMS if a != "native"}}

    nat_chars = sum(r["arms"]["native"]["prompt_chars"] for r in per_question)
    tokens = {a: {"prompt_chars": sum(r["arms"][a]["prompt_chars"]
                                      for r in per_question)}
              for a in ARMS}
    for a, t in tokens.items():
        t["delta_chars_vs_native"] = t["prompt_chars"] - nat_chars
        t["delta_pct"] = (100.0 * t["delta_chars_vs_native"] / nat_chars
                          if nat_chars else None)
        t["approx_prompt_tokens"] = t["prompt_chars"] // 4

    return {"subset": plan["subset"], "n_facts": plan["n_facts"],
            "strata_counts": plan["strata_counts"],
            "n_page_edit_mismatch": plan["n_page_edit_mismatch"],
            "n_containment_violations": plan["n_containment_violations"],
            "containment_violations": plan["containment_violations"],
            "n_page_edit_errors": plan["n_page_edit_errors"],
            "page_edit_errors": plan["page_edit_errors"],
            "positive_control": plan["positive_control"],
            "harness": plan["harness"],
            "page_source": plan["page_source"],
            "n_chunks_total": plan["n_chunks_total"],
            "n_chunks_on_page": plan["n_chunks_on_page"],
            "retrieval_complete": plan["retrieval_complete"],
            "n_llm_calls": n_calls,
            "arms": {a: _acc(flags[a]) for a in ARMS},
            "paired_vs_native": {a: paired_cells(flags["native"], flags[a])
                                 for a in ARMS if a != "native"},
            "aa_floor": paired_cells(flags["native"], flags["native_repeat"]),
            "by_stratum": by_stratum, "tokens": tokens,
            "harm": {a: harm_report(per_question, a) for a in ARMS
                     if a != "native"},
            "per_question": per_question}


# ── harm taxonomy (pre-registration v2, Amendment 2) ─────────────────────────
def classify_harm(native_out: str, arm_out: str, gold_cut: bool) -> str:
    """Name one harmed question, by the rule registered before the run.

    ``gold_cut`` first, because a deleted gold fact explains everything after it.
    Then ``malformed_generation`` — the measured "Shinzo Abe" -> "Sinzo Abe"
    shape, where the answer is right and the string is not. Then
    ``refusal_after_edit`` — the model declines although nothing it needed was
    removed (sh_32k retrieval q14). Anything else is ``information_loss``.
    """
    from difflib import SequenceMatcher  # stdlib; local to keep import cost nil
    if gold_cut:
        return "gold_cut"
    native_out, arm_out = native_out or "", arm_out or ""
    if SequenceMatcher(a=native_out, b=arm_out).ratio() >= MALFORMED_RATIO:
        return "malformed_generation"
    low = arm_out.lower()
    if not normalize_answer(arm_out).strip() or any(m in low for m in REFUSAL_MARKERS):
        return "refusal_after_edit"
    return "information_loss"


def harm_report(per_question, arm: str) -> dict:
    """Every question this arm lost, classified and counted.

    ``protective_claim_void`` restates §5b unchanged: a unique-stratum harm that
    is not a malformed generation voids the protective claim regardless of net.
    """
    rows = []
    for q in per_question:
        if not (q["arms"]["native"]["correct"] and not q["arms"][arm]["correct"]):
            continue
        target = q.get("target_serial")
        gold_cut = bool(target is not None
                        and target in set(q["plan"]["suppress_serials"]))
        cls = classify_harm(q["arms"]["native"]["output"],
                            q["arms"][arm]["output"], gold_cut)
        rows.append({"index": q["index"], "stratum": q["stratum"], "class": cls,
                     "native_output": q["arms"]["native"]["output"],
                     "arm_output": q["arms"][arm]["output"],
                     "target_serial": target, "gold_cut": gold_cut})
    counts = {c: sum(1 for r in rows if r["class"] == c) for c in HARM_CLASSES}
    voiding = [r for r in rows if r["stratum"] == "unique"
               and r["class"] != "malformed_generation"]
    return {"n_harmed": len(rows), "counts": counts,
            "by_stratum": {s: {c: sum(1 for r in rows if r["stratum"] == s
                                      and r["class"] == c)
                               for c in HARM_CLASSES}
                           for s in sorted({r["stratum"] for r in rows})},
            "protective_claim_void": bool(voiding),
            "voiding_questions": [r["index"] for r in voiding],
            "harms": rows}


# ── void conditions, assembled in one place (pre-registration v2 §7) ─────────
NATIVE_BAND = (0.30, 0.50)      # VC2, derived in advance from the m3 offset
UNIQUE_NATIVE_FLOOR = 0.80      # VC2, the protective design's premise
CONFIRMATORY_ARM = "detector_suppress"


def void_condition_report(res: dict, table: dict, page_source: str | None) -> dict:
    """Every pre-registered void condition, evaluated and named.

    Assembled here rather than left scattered across counters: an external
    reviewer should be able to read the verdict, not reconstruct it. Each entry
    carries the observed value that decided it.

    Condition 5 is the only one that voids the PROTECTIVE CLAIM alone; every
    other condition voids the whole run (Amendment 4, R2).
    """
    arms, strat = res["arms"], res["by_stratum"]
    harm = res["harm"][CONFIRMATORY_ARM]
    pc = res["positive_control"]
    m = blank_metrics()
    for q in res["per_question"]:
        classify_drops(m, [f"fact:{x}" for x in q["plan"]["suppress_serials"]],
                       table)
    nat = arms["native"]["accuracy"]
    uniq = (strat.get("unique") or {}).get("arms", {}).get("native", {}).get("accuracy")
    aa = res["aa_floor"]

    def cond(num, scope, ok, observed, note=""):
        return {"condition": num, "voids": scope,
                "status": "pass" if ok else "fail",
                "observed": observed, "note": note}

    out = {
        "1_page_edit_mismatch": cond(
            1, "run", res["n_page_edit_mismatch"] == 0,
            res["n_page_edit_mismatch"]),
        "2_native_in_band": cond(
            2, "run",
            bool(nat is not None and NATIVE_BAND[0] <= nat <= NATIVE_BAND[1]
                 and (uniq is None or uniq >= UNIQUE_NATIVE_FLOOR)),
            {"native_overall": nat, "band": list(NATIVE_BAND),
             "unique_native": uniq, "unique_floor": UNIQUE_NATIVE_FLOOR}),
        "3_aa_floor_zero": cond(
            3, "run", aa["b_native_only"] + aa["c_arm_only"] == 0,
            {"b": aa["b_native_only"], "c": aa["c_arm_only"]}),
        "4_no_harmful_suppression": cond(
            4, "run", m["n_suppressed_harmful"] == 0,
            {"n_suppressed_harmful": m["n_suppressed_harmful"],
             "n_suppressed_superseded": m["n_suppressed_superseded"],
             "n_suppressed_same_value": m["n_suppressed_same_value"]}),
        "5_protected_stratum": cond(
            5, "protective_claim_only", not harm["protective_claim_void"],
            {"voiding_questions": harm["voiding_questions"],
             "counts": harm["counts"]},
            "the ONLY condition that leaves the run and the accuracy result "
            "standing; the shot is still spent"),
        "6_7_page_source": cond(
            6, "run", page_source == "benchmark" or page_source is None,
            {"page_source": page_source},
            "Amendments 1 and 3: chunk vectors valid AND the page is the "
            "benchmark's own"),
        "8_guards_and_positive_control": cond(
            8, "run",
            bool(res.get("n_containment_violations", 0) == 0
                 and res.get("n_page_edit_errors", 0) == 0 and pc["ok"]),
            {"containment": res.get("n_containment_violations"),
             "page_edit_errors": res.get("n_page_edit_errors"),
             "positive_control": pc}),
    }
    run_failed = [k for k, v in out.items()
                  if v["status"] == "fail" and v["voids"] == "run"]
    out["verdict"] = {
        "run_void": bool(run_failed),
        "run_void_because": run_failed,
        "protective_claim_void": bool(
            out["5_protected_stratum"]["status"] == "fail"),
        "shot_spent": not bool(run_failed),
    }
    return out


# ── the headline: detector vs oracle ─────────────────────────────────────────
def load_oracle(subset: str) -> dict | None:
    """The committed oracle-probe result for this subset, whatever it is named."""
    for path in sorted(ORACLE_DIR.glob("stale_suppression_probe*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("smoke_llm"):
            continue
        for res in payload.get("results", []):
            if res["subset"] == subset:
                return {"file": str(path.relative_to(REPO)).replace("\\", "/"),
                        "harness": payload.get("harness", {}), "result": res}
    return None


def _ratio(num, den):
    if den in (None, 0) or num is None:
        return None
    return num / den


def compare_to_oracle(res: dict, oracle: dict | None) -> dict | None:
    """``detector-achieved / oracle-achieved`` per mechanism, on the paired
    McNemar net and on the conflicted-stratum gain over native.

    Both harnesses are identical and the ``native`` arm is a byte-identical
    prompt in both, so the two runs' native arms are compared directly as a
    cross-run A/A check. A drift there would invalidate every ratio below,
    which is why it is reported rather than assumed.
    """
    if oracle is None:
        return None
    o = oracle["result"]
    o_conf = o["by_stratum"].get("conflicted", {}).get("arms", {})
    d_conf = res["by_stratum"].get("conflicted", {}).get("arms", {})
    o_nat = o["arms"]["native"]
    d_nat = res["arms"]["native"]

    rows = {}
    for arm, oarm in ORACLE_COUNTERPART.items():
        d_net = res["paired_vs_native"][arm]["net"]
        o_net = o["paired_vs_native"][oarm]["net"]
        d_gain = (d_conf.get(arm, {}).get("correct", 0)
                  - d_conf.get("native", {}).get("correct", 0))
        o_gain = (o_conf.get(oarm, {}).get("correct", 0)
                  - o_conf.get("native", {}).get("correct", 0))
        rows[arm] = {
            "oracle_arm": oarm,
            "detector_net": d_net, "oracle_net": o_net,
            "net_ratio": _ratio(d_net, o_net),
            "detector_conflicted_correct": d_conf.get(arm, {}).get("correct"),
            "oracle_conflicted_correct": o_conf.get(oarm, {}).get("correct"),
            "detector_conflicted_gain": d_gain, "oracle_conflicted_gain": o_gain,
            "conflicted_gain_ratio": _ratio(d_gain, o_gain),
            "detector_accuracy": res["arms"][arm]["accuracy"],
            "oracle_accuracy": o["arms"][oarm]["accuracy"]}
    o_harness = "whole_context"          # every oracle probe arm is whole-context
    return {
        "source": oracle["file"],
        "oracle_harness": o_harness,
        "harness_match": res.get("harness", "whole_context") == o_harness,
        "harness_caveat": None if res.get("harness", "whole_context") == o_harness else (
            "This run uses the RETRIEVAL-PATH harness; the oracle probe is "
            "whole-context. The ratios below therefore compare across harnesses "
            "and are NOT the detector/oracle ratio of the confirmatory design - "
            "read them as 'how much of the whole-context oracle ceiling survives "
            "being applied to a rank-ordered multi-block page'."),
        "same_harness": {
            "llm_model": oracle["harness"].get("llm_model"),
            "llm_base_url": oracle["harness"].get("llm_base_url"),
            "prompt_shape": oracle["harness"].get("prompt_shape")},
        "native_cross_run": {
            "detector_native": d_nat, "oracle_native": o_nat,
            "identical": d_nat["correct"] == o_nat["correct"]},
        "by_mechanism": rows}


# ── reporting ────────────────────────────────────────────────────────────────
def _fmt(x, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def format_detection(cells, chosen) -> str:
    lines = ["", "=" * 100,
             " DETECTOR OPERATING-POINT SELECTION (detection quality only - "
             "no LLM, no gold)", "=" * 100,
             f"{'cos':>5}{'r_min':>8}{'amb':>6}{'nli':>6}{'filt':>9}"
             f"{'prec':>8}{'rec_pool':>10}{'rec_page':>10}{'sup':>7}"
             f"{'harm':>6}{'q_rec':>8}{'cover':>7}"]
    # [E2E] all cells, not only pair_filter-truthy ones: the 'none' screen's
    # grid is entirely False and used to render an empty table.
    top = sorted(cells,
                 key=lambda c: -(c["metrics"]["pair_recall_pool"] or 0))[:12]
    for c in top:
        m = c["metrics"]
        filt = str(c["pair_filter"])
        if c.get("ces_tau") is not None:
            t = c["ces_tau"]
            filt += f"@{t}" if isinstance(t, str) else f"@{t:g}"
        lines.append(
            f"{c['cos_pair']:>5}{c['r_min_label']:>8}{c['ambiguity_mode']:>6}"
            f"{c['nli_contradiction']:>6}{filt:>9}"
            f"{_fmt(m['pair_precision']):>8}{_fmt(m['pair_recall_pool']):>10}"
            f"{_fmt(m['pair_recall_page']):>10}"
            f"{m['n_suppressed']:>7}{m['n_suppressed_harmful']:>6}"
            f"{_fmt(m['question_recall_conflicted']):>8}"
            f"{_fmt(m['coverage']):>7}")
    if chosen:
        m = chosen["metrics"]
        lines += ["",
                  f" CHOSEN: cos_pair={chosen['cos_pair']} "
                  f"r_min={chosen['r_min_label']} "
                  f"ambiguity={chosen['ambiguity_mode']} "
                  f"nli={chosen['nli_contradiction']} "
                  f"pair_filter={chosen['pair_filter']}"
                  + (f" ces_tau={chosen['ces_tau']}"
                     if chosen.get("ces_tau") is not None else ""),
                  f"   pair precision {_fmt(m['pair_precision'])}  "
                  f"recall(pool) {_fmt(m['pair_recall_pool'])}  "
                  f"recall(page) {_fmt(m['pair_recall_page'])}",
                  f"   suppressed {m['n_suppressed']} facts "
                  f"({m['n_suppressed_superseded']} superseded, "
                  f"{m['n_suppressed_same_value']} same-value duplicates, "
                  f"{m['n_suppressed_harmful']} harmful)",
                  f"   conflicted-question recall "
                  f"{m['n_conflicted_hit']}/{m['n_conflicted']} "
                  f"(gold-valued fact cut in {m['n_conflicted_gold_cut']})",
                  f"   unique-stratum questions touched "
                  f"{m['n_unique_touched']}/{m['n_unique']}"]
    else:
        lines.append("\n NO cell satisfies the pre-registered requirements. "
                     "REPORT AND STOP.")
    return "\n".join(lines)


def format_run(results, comparisons) -> str:
    lines = []
    for res in results:
        cmp_ = comparisons.get(res["subset"])
        cov = ""
        if res.get("page_source"):
            cov += f", page={res['page_source']}"
        if res.get("n_chunks_total") is not None:
            cov += (f", {res['n_chunks_on_page']}/{res['n_chunks_total']} chunks "
                   f"on page{'' if res['retrieval_complete'] else ' - INCOMPLETE'}")
        lines += ["", "=" * 100,
                  f" {res['subset']}  [{res.get('harness', 'whole_context')}]  "
                  f"({res['n_facts']} facts, {res['strata_counts']}{cov})", "=" * 100,
                  f"{'arm':<24}{'overall':>18}{'unique':>18}{'conflicted':>18}"
                  f"{'b/c':>10}{'tok d%':>9}"]
        for arm in ARMS:
            cells = ""
            if arm != "native":
                pc = res["paired_vs_native"][arm]
                cells = f"{pc['b_native_only']}/{pc['c_arm_only']}"
            row = f"{arm:<24}"
            for scope in ("overall", "unique", "conflicted"):
                a = (res["arms"][arm] if scope == "overall"
                     else (res["by_stratum"].get(scope) or {}).get("arms", {}).get(arm))
                row += ("{:>18}".format("-") if not a or a["accuracy"] is None
                        else "{:>18}".format("{}/{} {:.3f}".format(
                            a["correct"], a["n"], a["accuracy"])))
            dp = res["tokens"][arm]["delta_pct"]
            lines.append(row + f"{cells:>10}" +
                         ("{:>9}".format("-") if dp is None else f"{dp:>+9.2f}"))
        aa = res["aa_floor"]
        lines.append(f"  A/A floor: {aa['b_native_only']}/{aa['c_arm_only']} "
                     f"discordant of {aa['n']} (p={aa['p_exact']:.3f})")
        pc = res.get("positive_control") or {}
        lines.append(f"  GUARDS  mismatches={res['n_page_edit_mismatch']} "
                     f"containment={res.get('n_containment_violations', 0)} "
                     f"page_edit_errors={res.get('n_page_edit_errors', 0)}"
                     "   (all MUST be 0 - void condition 8)")
        if pc:
            lines.append(
                f"  POSITIVE CONTROL  fired={pc['n_questions_policy_fired']}"
                f"/{pc['n_questions']} edits={pc['n_fact_edits_applied']}"
                f" suppressed={pc['n_facts_suppressed']} "
                f"-> {'OK' if pc['ok'] else 'FAILED'}")
        vc = res.get("void_conditions") or {}
        if vc:
            v = vc["verdict"]
            lines.append(
                "  VOID CONDITIONS  " + "  ".join(
                    f"{k.split('_')[0]}:{vv['status']}"
                    for k, vv in vc.items() if k != "verdict"))
            lines.append(f"  VERDICT  run_void={v['run_void']} "
                         f"protective_claim_void={v['protective_claim_void']} "
                         f"shot_spent={v['shot_spent']}")
        for arm, h in res.get("harm", {}).items():
            if not h["n_harmed"]:
                continue
            named = ", ".join(f"{c}={n}" for c, n in h["counts"].items() if n)
            void = "  PROTECTIVE CLAIM VOID" if h["protective_claim_void"] else ""
            lines.append(f"  harm {arm:<22} {h['n_harmed']}: {named}{void}")
        for arm in ARMS:
            if arm in ("native", "native_repeat"):
                continue
            pc = res["paired_vs_native"][arm]
            lines.append(f"  {arm:<22} net {pc['net']:+d}  "
                         f"(b={pc['b_native_only']} c={pc['c_arm_only']}, "
                         f"exact p={pc['p_exact']:.4g})")
        if cmp_:
            lines += ["", "  DETECTOR / ORACLE", "  " + "-" * 92,
                      f"  {'mechanism':<24}{'oracle arm':<18}"
                      f"{'det net':>9}{'orc net':>9}{'ratio':>8}"
                      f"{'det conf':>10}{'orc conf':>10}{'gain ratio':>12}"]
            for arm, r in cmp_["by_mechanism"].items():
                nr = _fmt(r["net_ratio"])
                gr = _fmt(r["conflicted_gain_ratio"])
                lines.append(
                    f"  {arm:<24}{r['oracle_arm']:<18}"
                    f"{r['detector_net']:>+9d}{r['oracle_net']:>+9d}{nr:>8}"
                    f"{str(r['detector_conflicted_correct']):>10}"
                    f"{str(r['oracle_conflicted_correct']):>10}{gr:>12}")
            nx = cmp_["native_cross_run"]
            lines.append(f"  cross-run native check: this run "
                         f"{nx['detector_native']['correct']}/"
                         f"{nx['detector_native']['n']}, oracle run "
                         f"{nx['oracle_native']['correct']}/"
                         f"{nx['oracle_native']['n']} "
                         f"({'identical' if nx['identical'] else 'DRIFTED'})")
    return "\n".join(lines)


# ── loading ──────────────────────────────────────────────────────────────────
def subset_of(item: dict) -> str:
    return item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
        .replace("factconsolidation_", "")


def prepass_path(cfg, subset: str, page_source: str | None,
                 geometry_space: str = "raw", prepass_tag: str = ""):
    """Where the prepass for this page source lives.

    Two page sources means two prepasses, and they are NOT interchangeable: the
    candidate pool is built from the page, so a pool derived from H-Nav's own
    ranking can name a fact absent from the benchmark's page. Separate filenames
    stop that by accident; the stamped ``page_source`` stops it on purpose.
    """
    suffix = "_benchmarkpage" if page_source == "benchmark" else ""
    # [ABTT] the geometry space is part of the identity of a prepass for the
    # same reason the page source is: the pairs and the cosines differ, so a
    # raw prepass replayed into a whitened gate would be a silent mismatch.
    suffix += "_abtt" if geometry_space == "abtt" else ""
    # [E2E] a screen that admits pairs below the raw loose bound needs its own
    # prepass superset; the tag keeps it from colliding with the shipped one.
    suffix += prepass_tag
    return cfg.out_dir / f"stage1_prepass_{subset}{suffix}.json"


def load_context(cfg, subsets, args) -> dict:
    """Dataset items, prepasses, ground-truth tables and gate-ready records.

    The prepass NLI-config stamp is checked here rather than trusted: a table
    scored by a different engine, replayed into this gate, would produce numbers
    that look valid and are not (the T12 lesson). An unstamped prepass predates
    the stamp and is accepted only behind ``--allow-unstamped-prepass``, with
    the fact recorded in every artifact this run writes.
    """
    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {subset_of(i): i for i in data if subset_of(i) in subsets}
    missing = [s for s in subsets if s not in items]
    if missing:
        raise SystemExit(f" REFUSED: no dataset item for {missing}")

    want_source = getattr(args, "page_source", None)
    prepasses, unstamped = {}, []
    for s in subsets:
        p = prepass_path(cfg, s, want_source,
                         getattr(args, "geometry_space", "raw"),
                         getattr(args, "prepass_tag", ""))
        if not p.exists():
            extra = (" --page-source benchmark"
                     if want_source == "benchmark" else "")
            raise SystemExit(
                f" REFUSED: {p} missing. Run\n"
                f"   python hnav/stage1/calibrate_read_policy.py --prepass "
                f"--subsets {s}{extra}\n first (it needs the embedder and NLI).")
        pp = json.loads(p.read_text(encoding="utf-8"))
        # GUARD: the prepass must have been built from the SAME page this run
        # will edit. A pool derived from a different ranking can name a fact the
        # page does not carry; page_edit would raise, the adapter would fall
        # back, and the artifact would look exactly like a null result.
        got_source = pp.get("page_source", "prepass")
        if want_source is not None and got_source != want_source:
            raise SystemExit(
                f" REFUSED: {p.name} was built with page_source={got_source!r} "
                f"but this run uses {want_source!r}. The candidate pool would "
                "come from a different page than the model sees.")
        stamp = pp.get("nli_config")
        if stamp is None:
            unstamped.append(s)
            if not args.allow_unstamped_prepass:
                raise SystemExit(
                    f" REFUSED: {p.name} predates NLI-config stamping (T12), so "
                    "the engine that\n produced its scores cannot be verified. "
                    "Re-run --prepass, or pass\n --allow-unstamped-prepass to "
                    "accept it (the artifact records that you did).")
        elif stamp.get("stub"):
            raise SystemExit(f" REFUSED: {p.name} was scored with a STUB NLI.")
        want_space = getattr(args, "geometry_space", "raw")
        got_space = pp.get("geometry_space", "raw")
        if got_space != want_space:
            raise SystemExit(
                f" REFUSED: {p.name} was written in the {got_space!r} geometry "
                f"space but this run scores in {want_space!r}.")
        w = getattr(args, "_whitener", None)
        if w is not None and pp.get("whitening_fingerprint") != w.fingerprint():
            raise SystemExit(
                f" REFUSED: {p.name} was written with whitening fingerprint "
                f"{pp.get('whitening_fingerprint')}, this run loaded "
                f"{w.fingerprint()}. Different artifact, different space.")
        prepasses[s] = pp

    tables = {s: fact_table(items[s]) for s in subsets}

    records, namespaces, cos_err = {}, set(), 0.0
    for s in subsets:
        table = tables[s]
        order = sorted(table["by_id"], key=lambda f: table["by_id"][f][0])
        texts = [table["by_id"][f][1] for f in order]
        vecs, key, err = load_fact_vectors(cfg, order, texts, [prepasses[s]],
                                           whitener=getattr(args, "_whitener", None))
        records[s] = build_records(order, texts, table, vecs)
        namespaces.add(key)
        cos_err = max(cos_err, err)

    return {"items": items, "prepasses": prepasses, "tables": tables,
            "records": records, "unstamped": unstamped,
            "embed_namespaces": sorted(namespaces), "max_cosine_error": cos_err}


def page_fact_ids(prepass, table) -> list[str]:
    """Every fact the retrieved page carries, from the prepass's own
    ``fact_chunk`` map. This is the denominator for ``pair_recall_page`` — what
    a detector with no pool cap could in principle have seen."""
    tops = {c for q in prepass["questions"] for c in q["top_ids"]}
    return [f for f, c in prepass["fact_chunk"].items()
            if c in tops and f in table["by_id"]]


# ── selection driver ─────────────────────────────────────────────────────────
def measure_grid(ctx, subsets, cells, ces=None) -> None:
    """Fill ``cell["metrics"]`` (pooled over the split) and ``cell["by_subset"]``."""
    for c in cells:
        c["metrics"] = blank_metrics()
        c["by_subset"] = {s: blank_metrics() for s in subsets}
    for s in subsets:
        pp, table, recs = ctx["prepasses"][s], ctx["tables"][s], ctx["records"][s]
        strata = {r["index"]: r for r in classify_questions(ctx["items"][s])}
        page_gt = count_gt_pairs(page_fact_ids(pp, table), table["by_id"])
        decisions = decide_all(pp, recs, cells, ReplayNLI(pp["nli_table"]), ces)
        for q in pp["questions"]:
            decs = decisions[q["index"]]
            pool = [f for f in q["pool"] if f in table["by_id"]]
            for ci, c in enumerate(cells):
                score_decision(c["metrics"], decs[ci], table,
                               strata[q["index"]], pool, page_gt)
                score_decision(c["by_subset"][s], decs[ci], table,
                               strata[q["index"]], pool, page_gt)
    for c in cells:
        c["metrics"] = finish_metrics(c["metrics"])
        c["by_subset"] = {s: finish_metrics(m) for s, m in c["by_subset"].items()}


AMBIGUITY_NOTE = (
    "ambiguity_mode='none' disables the frozen Stage-0 nmargin/H_z "
    "precondition. Declared, not incidental: those two signals are the only "
    "gate input computed from CHUNK embeddings, which were truncated at 512 of "
    "~4096 tokens (T12) and are not yet re-fit; they are also the dominant "
    "recall bottleneck (question-level recall collapses from 0.97 to 0.16 with "
    "the screen on); and the volume-limiting role they played is now carried by "
    "the identity screen plus bidirectional NLI at the measured precision "
    "recorded here. A campaign run after the embeddings are re-fit must "
    "revisit this.")


def freeze(chosen, ctx, subsets, cfg=None, args=None) -> Path:
    art = {
        "task": "T13",
        "thresholds": {
            "cos_pair": chosen["cos_pair"], "r_min": chosen["r_min"],
            "nmargin": _rg.NMARGIN_CAL, "H_z": _rg.H_Z_CAL,
            "ambiguity_mode": chosen["ambiguity_mode"],
            "nli_contradiction": chosen["nli_contradiction"]},
        "r_min_label": chosen["r_min_label"],
        "pair_filter": chosen["pair_filter"],
        "ces": ({"tau": chosen["ces_tau"],
                 "artifact": {"fusion": getattr(args, "fusion_artifact", None),
                              "geo": getattr(args, "geo_artifact", None),
                              "ces": getattr(args, "ces_artifact", None)
                              }[chosen["pair_filter"]],
                 "fingerprint": (args._ces.fingerprint()
                                 if getattr(args, "_ces", None) is not None
                                 else None)}
                if chosen["pair_filter"] in ("ces", "fusion", "geo") else None),
        "mechanisms": ["suppress", "demote_late"],
        "selection_rule": selection_rule_for(
            getattr(args, "pair_screen", "parser")),
        "metrics": chosen["metrics"],
        "by_subset": chosen["by_subset"],
        "ambiguity_note": AMBIGUITY_NOTE if chosen["ambiguity_mode"] == "none"
                          else None,
        "provenance": {
            "generated_at_utc":
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": _git_head(),
            "fit_subsets": sorted(subsets),
            "confirmatory_refused": list(CONFIRMATORY),
            "prepass": {s: prepass_path(cfg, s,
                                        getattr(args, "page_source", None),
                                        getattr(args, "geometry_space", "raw"),
                                        getattr(args, "prepass_tag", "")).name
                        for s in subsets} if cfg is not None else None,
            "prepass_unstamped_nli_config": ctx["unstamped"],
            "embed_cache_namespace": ctx["embed_namespaces"],
            "max_pair_cosine_error_vs_prepass": ctx["max_cosine_error"],
            "geometry_space": getattr(args, "geometry_space", "raw"),
            "pair_screen": getattr(args, "pair_screen", "parser"),
            "whitening_artifact": getattr(args, "whitening_artifact", None),
            "whitening_fingerprint": (args._whitener.fingerprint()
                                      if getattr(args, "_whitener", None)
                                      is not None else None),
            "grid": {"cos_pair": list(getattr(args, "cos_grid", None)
                                      or COS_GRID),
                     "r_min": list(R_MIN_GRID_LABELS),
                     "ambiguity_mode": list(AMB_GRID),
                     "nli_contradiction": list(getattr(args, "nli_grid", None)
                                               or NLI_GRID),
                     "pair_filter": list(FILTER_GRID),
                     "ces_tau": (list(getattr(args, "ces_grid", None)
                                      or CES_TAU_GRID)
                                 if getattr(args, "pair_screen", "parser")
                                 == "ces" else None),
                     "geo_tau": ([t.strip() for t in
                                  (getattr(args, "geo_grid", None) or ""
                                   ).split(",") if t.strip()]
                                 or list(GEO_TAU_GRID)
                                 if getattr(args, "pair_screen", "parser")
                                 == "geo" else None)},
        },
    }
    # The SHIPPED operating point is never overwritten by an alternative
    # geometry or an alternative identity screen: a reader finding foreign
    # thresholds in the file the thesis cites would have no way to tell.
    # Each preregistered arm gets its own artifact file.
    dest = operating_point_path(getattr(args, "geometry_space", "raw"),
                                getattr(args, "pair_screen", "parser"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the operating point is pinned by sha256 in every
    # pipeline.json, and those pins are the git-blob (LF) hashes the Linux box
    # checks out. Letting Python's universal-newline translation write CRLF on
    # Windows silently breaks every pin while `git diff` stays empty.
    dest.write_text(json.dumps(art, indent=1, default=str), encoding="utf-8",
                    newline="\n")
    return dest


def operating_point_path(geometry_space: str, pair_screen: str) -> Path:
    """[E2E] One frozen operating-point file per preregistered arm. Undefined
    combinations are refused rather than given a default file — an arm that was
    not preregistered must not silently acquire an operating point."""
    if pair_screen == "parser":
        if geometry_space == "abtt":
            return REPO / "stage0_results" / "abtt" / "abtt_operating_point.json"
        return OPERATING_POINT
    gf = REPO / "stage0_results" / "geometry_filter"
    if pair_screen == "ces" and geometry_space == "raw":
        return gf / "ces_operating_point.json"
    if pair_screen == "fusion" and geometry_space == "raw":
        return gf / "fusion_operating_point.json"
    if pair_screen == "geo" and geometry_space == "raw":
        return gf / "geo_operating_point.json"
    if pair_screen == "idonly" and geometry_space == "raw":
        return REPO / "stage0_results" / "stage1" / "idonly_operating_point.json"
    if pair_screen == "none" and geometry_space == "abtt":
        return gf / "abtt_noparser_operating_point.json"
    raise SystemExit(
        f" REFUSED: no preregistered arm for pair_screen={pair_screen!r} with "
        f"geometry_space={geometry_space!r}. The preregistered arms are "
        "ces+raw, fusion+raw, geo+raw, idonly+raw and none+abtt; parser goes "
        "with either space.")


def frozen_cell(geometry_space: str = "raw", pair_screen: str = "parser",
                override: str | None = None) -> dict:
    """The committed operating point, as a grid cell. Refuses to invent one.

    [ABTT] The whitened arm reads its OWN frozen artifact. Falling back to the
    shipped raw point here would score whitened vectors against thresholds
    calibrated in the raw space -- the arms would differ by two things at once
    and the contrast would mean nothing. [E2E] Same for the identity-screen
    arms: each reads its own file, and the file's recorded screen must agree
    with the one the caller asked for.
    """
    global OPERATING_POINT
    if override is not None:
        # [EXPLORATORY] an explicitly-supplied operating-point file. The screen
        # and fingerprint checks below still apply; the caller announces the
        # deviation, this function only refuses to let it be silent.
        OPERATING_POINT = Path(override)
        print(f" NOTE: OPERATING-POINT OVERRIDE -> {_rel(OPERATING_POINT)} "
              "(exploratory run; not the preregistered arm artifact)")
    else:
        OPERATING_POINT = operating_point_path(geometry_space, pair_screen)
    if not OPERATING_POINT.exists():
        raise SystemExit(
            f" REFUSED: {_rel(OPERATING_POINT)} does not exist. Run\n"
            "   python hnav/stage1/detector_gap.py --select\n"
            " first: the operating point is frozen from DETECTION quality, "
            "before anything is graded.")
    art = json.loads(OPERATING_POINT.read_text(encoding="utf-8"))
    got_screen = art.get("provenance", {}).get("pair_screen", "parser")
    if got_screen != pair_screen:
        raise SystemExit(
            f" REFUSED: {_rel(OPERATING_POINT)} was frozen for the "
            f"{got_screen!r} identity screen but this run asked for "
            f"{pair_screen!r}.")
    thr = art["thresholds"]
    cell = {"cos_pair": thr["cos_pair"], "r_min": thr["r_min"],
            "r_min_label": art.get("r_min_label", "custom"),
            "ambiguity_mode": thr["ambiguity_mode"],
            "nli_contradiction": thr["nli_contradiction"],
            "pair_filter": art["pair_filter"], "artifact": art}
    if art.get("ces"):
        cell["ces_tau"] = art["ces"]["tau"]
        cell["ces_fingerprint"] = art["ces"]["fingerprint"]
    return cell


def _rel(path: Path) -> str:
    """Repo-relative when possible; the absolute path otherwise, so a message
    about a missing file cannot itself raise."""
    try:
        return str(Path(path).relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="*", default=list(CALIBRATION))
    ap.add_argument("--select", action="store_true",
                    help="score the grid on DETECTION quality only and freeze "
                         "the operating point. No LLM, no grading.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every arm, print the exact call budget, send nothing")
    ap.add_argument("--smoke-llm", action="store_true",
                    help="SMOKE ONLY: the probe's deterministic primacy stub "
                         "instead of the answer model. Writes *_SMOKE.json; the "
                         "numbers are MEANINGLESS as evidence about any model.")
    ap.add_argument("--harness", choices=list(HARNESSES),
                    default="whole_context",
                    help="whole_context reproduces the oracle probe exactly; "
                         "retrieval puts the benchmark's own top-k chunks on the "
                         "page in rank order and edits them through the shipped "
                         "page contract (pre-registration v2 section 10)")
    ap.add_argument("--page-source", choices=("benchmark", "prepass"),
                    default=None,
                    help="retrieval harness only, and REQUIRED there. "
                         "'benchmark' reads the benchmark's own top-k page from "
                         "the refit artifact (Amendment 3); 'prepass' uses the "
                         "ranking H-Nav computed for itself, which agrees with "
                         "the benchmark on 26/100 sh_64k questions.")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--allow-unstamped-prepass", action="store_true")
    ap.add_argument("--confirmatory", action="store_true",
                    help="THE ACT OF FIRING. Permits sh_64k, and ONLY sh_64k, "
                         "and only with --harness retrieval --page-source "
                         "benchmark. sh_262k stays refused. Use with --dry-run "
                         "first to produce the guard pre-flight evidence.")
    ap.add_argument("--geometry-space", choices=("raw", "abtt"), default="raw",
                    help="'abtt' scores the gate in the ABTT-whitened space; "
                         "the prepass must have been produced the same way.")
    ap.add_argument("--whitening-artifact", default=None)
    ap.add_argument("--cos-grid", nargs="+", type=float, default=None,
                    help="override the cosine axis (required for abtt: the "
                         "default 0.90/0.92/0.94 are raw-space coordinates)")
    ap.add_argument("--pair-screen",
                    choices=("parser", "idonly", "ces", "fusion", "geo",
                             "none"),
                    default="parser",
                    help="[E2E] identity screen: 'parser' = the shipped "
                         "same-key check; 'ces' = the frozen contrastive-edit-"
                         "subspace artifact (relation from the parser, subject "
                         "identity from geometry); 'idonly' = the same "
                         "same-key screen as its own arm, whose grid relaxes "
                         "the NLI axis (E2E-4: is the semantic gate the "
                         "binding constraint?); 'fusion' = the calibration-"
                         "fit logistic over (CES, ABTT-cosine); 'geo' = the "
                         "fully parser-free GEO identity screen (slot probe x "
                         "whitened cosine, E2E-3); 'none' = no identity screen "
                         "(the abtt_noparser arm only).")
    ap.add_argument("--ces-artifact", default=None,
                    help="path to ces_subspaces_k20.json (required for "
                         "--pair-screen ces)")
    ap.add_argument("--fusion-artifact", default=None,
                    help="path to fusion_screen.json (required for "
                         "--pair-screen fusion)")
    ap.add_argument("--geo-artifact", default=None,
                    help="path to geo_identity_screen.json (required for "
                         "--pair-screen geo)")
    ap.add_argument("--ces-grid", nargs="+", type=float, default=None,
                    help="tau axis for --select with --pair-screen ces "
                         "(default: the preregistered CES_TAU_GRID)")
    ap.add_argument("--nli-grid", nargs="+", type=float, default=None,
                    help="override the bidirectional-NLI contradiction axis "
                         "(default 0.5/0.9/0.99). Required for --pair-screen "
                         "idonly, whose preregistered axis includes 0.0 = the "
                         "gate off.")
    ap.add_argument("--geo-grid", default=None,
                    help="tau axis for --select with --pair-screen geo: ONE "
                         "comma-separated string of floats (diagonal family) "
                         "and/or 'tw:tp' rectangles (per-axis offsets, the "
                         "E2E-3 amendment). Pass with '=' so leading minus "
                         "signs survive argparse: --geo-grid=-1.0:0.2,0:0.4")
    ap.add_argument("--prepass-tag", default=None,
                    help="extra prepass filename suffix; defaults to '_ces' "
                         "for --pair-screen ces and '' otherwise")
    ap.add_argument("--operating-point", default=None,
                    help="[EXPLORATORY] read this operating-point file instead "
                         "of the arm's frozen one. For explicitly-labeled "
                         "deviation runs only; screen/fingerprint checks still "
                         "apply and the artifact records the override.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()

    # [ABTT] Resolve the geometry space before anything reads a prepass or a
    # vector, so every later guard compares like with like.
    args._whitener = None
    if args.geometry_space == "abtt":
        if not args.whitening_artifact:
            print(" REFUSED: --geometry-space abtt needs --whitening-artifact "
                  "(fit one with hnav/stage0/fit_abtt_artifact.py).")
            return 2
        if args.select and not args.cos_grid:
            print(" REFUSED: --select in the abtt space needs --cos-grid. The "
                  "default 0.90/0.92/0.94 are RAW-space coordinates and would "
                  "search a region the whitened distribution barely reaches.")
            return 2
        from hnav.stage1.calibrate_read_policy import load_whitening
        args._whitener, _art = load_whitening(args.whitening_artifact)
        print(f" ABTT space: D={args._whitener.components.shape[0]}  "
              f"fingerprint={args._whitener.fingerprint()[:16]}...  "
              f"fit_on={_art['fit_subsets']}")

    # [E2E] Resolve the identity screen the same way: before any prepass read.
    args._ces = None
    if args.prepass_tag is None:
        # fusion shares CES's raw cos>=0.80 frame and therefore its prepasses
        args.prepass_tag = "_ces" if args.pair_screen in ("ces", "fusion") else ""
    if args.pair_screen == "fusion":
        if args.geometry_space != "raw":
            print(" REFUSED: the fusion screen scores raw-space vectors "
                  "internally; --pair-screen fusion needs the raw space.")
            return 2
        if not args.fusion_artifact:
            print(" REFUSED: --pair-screen fusion needs --fusion-artifact "
                  "(fit one with python -m hnav.geometry_filter.fusion_screen).")
            return 2
        if args.select and not args.cos_grid:
            print(" REFUSED: --select with --pair-screen fusion needs "
                  "--cos-grid (the fusion frame is cos >= 0.80).")
            return 2
        from hnav.geometry_filter.fusion_screen import FusionScreen
        args._ces, _f_man = FusionScreen.load(args.fusion_artifact)
        print(f" FUSION screen: w={np.round(args._ces.w, 3).tolist()} "
              f"b={args._ces.b:.3f}  fingerprint="
              f"{args._ces.fingerprint()[:16]}...")
    if args.pair_screen == "ces":
        if args.geometry_space != "raw":
            print(" REFUSED: the CES artifact is fit in the RAW space; "
                  "--pair-screen ces cannot be combined with another "
                  "geometry space.")
            return 2
        if not args.ces_artifact:
            print(" REFUSED: --pair-screen ces needs --ces-artifact "
                  "(fit one with python -m hnav.geometry_filter.ces_artifact).")
            return 2
        if args.select and not args.cos_grid:
            print(" REFUSED: --select with --pair-screen ces needs --cos-grid. "
                  "The CES validation frame is cos >= 0.80 (the gold dataset "
                  "says nothing below it); pass the frame explicitly, e.g. "
                  "--cos-grid 0.80.")
            return 2
        from hnav.geometry_filter.ces_artifact import CESArtifact
        args._ces, _ces_man = CESArtifact.load(args.ces_artifact)
        print(f" CES screen: k={args._ces.k} relations={len(args._ces.relations)}"
              f"  fingerprint={args._ces.fingerprint()[:16]}...  "
              f"fit_on={_ces_man['provenance']['fit_subsets']}")
    if args.pair_screen == "geo":
        # [E2E-3] fully parser-free identity screen; scores RAW vectors (the
        # probe) and whitens internally from its own copied ABTT parameters,
        # so it runs on the shipped raw prepasses at the shipped cos grid.
        if args.geometry_space != "raw":
            print(" REFUSED: the GEO screen scores raw-space vectors "
                  "internally; --pair-screen geo needs the raw space.")
            return 2
        if not args.geo_artifact:
            print(" REFUSED: --pair-screen geo needs --geo-artifact "
                  "(fit one with python -m hnav.geometry_filter.geo_artifact).")
            return 2
        from hnav.geometry_filter.geo_artifact import GeoIdentityScreen
        args._ces, _geo_man = GeoIdentityScreen.load(args.geo_artifact)
        print(f" GEO screen: anchors T_w={args._ces.T_w:.4f} "
              f"T_p={args._ces.T_p:.4f}  fingerprint="
              f"{args._ces.fingerprint()[:16]}...  "
              f"fit_on={_geo_man['provenance']['fit_subsets']}")
    elif args.pair_screen == "none" and args.geometry_space != "abtt":
        print(" REFUSED: --pair-screen none is preregistered only as the "
              "abtt_noparser arm (--geometry-space abtt).")
        return 2

    if args.confirmatory:
        if args.select:
            print(" REFUSED: --confirmatory and --select are mutually exclusive."
                  "\n --select FITS the operating-point grid. Running it on the"
                  " confirmatory subset\n would tune on HELD-OUT data and write"
                  " an artifact that a later reader could\n mistake for the"
                  " frozen operating point, which is frozen at commit 4f66c52.")
            return 2
        wrong = [s for s in args.subsets if s != "sh_64k"]
        if wrong or not args.subsets:
            print(" REFUSED: --confirmatory covers sh_64k and nothing else; got "
                  + str(args.subsets) + ".\n sh_262k is outside the campaign "
                  "(pre-registration v2 section 2).")
            return 2
        if args.harness != "retrieval" or args.page_source != "benchmark":
            print(" REFUSED: the pre-registered confirmatory design is "
                  "--harness retrieval\n --page-source benchmark (Branch A, "
                  "Amendments 2 and 3). Got harness="
                  + repr(args.harness) + " page_source="
                  + repr(args.page_source) + ".")
            return 2
        print("=" * 100)
        print(" CONFIRMATORY RUN - sh_64k, single shot, pre-registered at")
        print("   stage0_results/stage1_preregistration_v2.md")
        print(" Success: conflicted McNemar net >= +10 AND exact p < 0.01 AND"
              " the protective criterion")
        print(" Void conditions 1-4 and 6-8 void the RUN; condition 5 voids the"
              " PROTECTIVE CLAIM only")
        print("=" * 100)
    bad = [s for s in args.subsets
           if s not in CALIBRATION and not (args.confirmatory and s == "sh_64k")]
    if bad:
        print(f" REFUSED: {bad} is outside the calibration split {CALIBRATION}.\n"
              f" {CONFIRMATORY} is the confirmatory arena; measuring or tuning "
              "there would\n void the campaign (STAGE1_PLAN.md). Nothing was "
              "computed.")
        return 2
    subsets = list(dict.fromkeys(args.subsets))

    ctx = load_context(cfg, subsets, args)
    print(f" embedding cache namespace(s): {ctx['embed_namespaces']}  "
          f"(max pair-cosine error vs prepass {ctx['max_cosine_error']:.2e})")
    if ctx["unstamped"]:
        print(f" NOTE: prepass for {ctx['unstamped']} predates NLI-config "
              "stamping; accepted under --allow-unstamped-prepass.")

    # [E2E] The prepass NLI table only covers pairs inside its loose
    # components. A gate cosine below the prepass's own cos_loose would ask
    # ReplayNLI about pairs it never scored — that is a hard KeyError three
    # layers down; refuse it here with the fix in the message instead.
    def check_cos_floor(min_cos: float) -> None:
        for s in subsets:
            used = ctx["prepasses"][s].get("cos_loose_used")
            if used is not None and min_cos < float(used) - 1e-9:
                raise SystemExit(
                    f" REFUSED: gate cos_pair {min_cos} is below the prepass "
                    f"loose bound {used} for {s}: the NLI replay table cannot "
                    f"cover the admitted pairs. Rebuild that prepass with "
                    f"--cos-loose {min_cos} (or lower).")

    if args.select:
        if sorted(subsets) != sorted(CALIBRATION):
            print(f" NOTE: fitting on {subsets}, not the full calibration split "
                  f"{list(CALIBRATION)}. Recorded in the artifact.")
        geo_taus = ([t.strip() for t in args.geo_grid.split(",") if t.strip()]
                    if (args.pair_screen == "geo" and args.geo_grid) else None)
        cells = grid_cells(args.cos_grid, args.pair_screen,
                           geo_taus if geo_taus else args.ces_grid,
                           nli_grid=args.nli_grid)
        check_cos_floor(min(c["cos_pair"] for c in cells))
        measure_grid(ctx, subsets, cells, args._ces)
        chosen = select(cells, args.pair_screen)
        # write BEFORE printing: a formatting hiccup must not lose the grid
        (cfg.out_dir / "detector_gap_selection.json").write_text(
            json.dumps({"cells": cells, "chosen": chosen,
                        "selection_rule": selection_rule_for(args.pair_screen),
                        "pair_screen": args.pair_screen,
                        "fit_subsets": subsets}, indent=1, default=str),
            encoding="utf-8")
        print(format_detection(cells, chosen))
        if chosen is None:
            return 3
        print(f"\n froze operating point -> "
              f"{_rel(freeze(chosen, ctx, subsets, cfg, args))}")
        return 0

    cell = frozen_cell(getattr(args, "geometry_space", "raw"), args.pair_screen,
                       override=args.operating_point)
    check_cos_floor(cell["cos_pair"])
    if cell["pair_filter"] in ("ces", "fusion", "geo"):
        if args._ces is None or \
                args._ces.fingerprint() != cell.get("ces_fingerprint"):
            print(" REFUSED: the frozen CES operating point was selected with "
                  f"fingerprint {str(cell.get('ces_fingerprint'))[:16]}... but "
                  "this run loaded "
                  + (args._ces.fingerprint()[:16] + "..." if args._ces else
                     "no artifact")
                  + ". Different artifact, different screen.")
            return 2
    print(f" harness: {args.harness}"
          + (f" (page source: {args.page_source})"
             if args.harness == "retrieval" else ""))
    print(f" operating point: cos_pair={cell['cos_pair']} "
          f"r_min={cell['r_min_label']} ambiguity={cell['ambiguity_mode']} "
          f"nli={cell['nli_contradiction']} pair_filter={cell['pair_filter']}"
          + (f" ces_tau={cell['ces_tau']}"
             if cell["pair_filter"] in ("ces", "fusion", "geo") else ""))

    plans = []
    for s in subsets:
        pp, table, recs = ctx["prepasses"][s], ctx["tables"][s], ctx["records"][s]
        decisions = decide_all(pp, recs, [cell], ReplayNLI(pp["nli_table"]),
                               args._ces)
        plans.append(plan_subset(ctx["items"][s], s, pp, decisions, 0, table,
                                 args.max_questions, harness=args.harness,
                                 page_source=args.page_source))

    bud = budget(plans)
    print("=" * 100)
    print(" DETECTOR-GAP RUN - call budget (exact, pre-flight)")
    print("=" * 100)
    for row in bud["per_subset"]:
        print(f"  {row['subset']:<9} questions={row['n_questions']:<5}"
              f" distinct prompts={row['n_distinct_prompts']:<5}"
              f" A/A repeats={row['n_aa_repeats']:<5}"
              f" calls={row['n_calls']:<6} "
              f"~{row['approx_prompt_tokens']:,} prompt tokens")
        print(f"             arms byte-identical to native: "
              f"{row['n_identical_to_native']}")
    print(f"  TOTAL {bud['total_calls']} calls, "
          f"~{bud['approx_total_prompt_tokens']:,} prompt tokens "
          f"(max_tokens={GENERATION_MAX_TOKENS} out)")
    print(f"  server: {'STUB (no endpoint)' if args.smoke_llm else cfg.llm_base_url}"
          f"   model: {'stub' if args.smoke_llm else cfg.llm_model}"
          f"   mode: {cfg.mode}")

    # GUARDS, checked before a single call is sent. Both are VOID conditions,
    # not "the intervention did nothing": a run whose edits silently failed to
    # apply looks exactly like a null result, and on a one-shot confirmatory
    # subset that mistake is unrecoverable.
    bad_edit = [p["subset"] for p in plans if p["n_page_edit_mismatch"]]
    bad_contain = [(p["subset"], p["n_containment_violations"],
                    p["containment_violations"][:2])
                   for p in plans if p["n_containment_violations"]]
    if bad_contain:
        print("\n REFUSED - VOID, not null: the policy names facts that are NOT"
              " on the page it\n would edit: " + str(bad_contain) + "\n"
              " The candidate pool came from a different page than the model"
              " sees. Such a run\n would fall back to the native page and record"
              " an intervention that never\n happened. Nothing was sent.")
        return 5
    if bad_edit:
        print("\n REFUSED - VOID, not null: the probe-style arm and the shipped"
              f" page_edit path\n disagree on {bad_edit}. The measurement would"
              " not be measuring the mechanism\n that ships. Nothing was sent.")
        return 4
    bad_err = [(p["subset"], p["n_page_edit_errors"]) for p in plans
               if p["n_page_edit_errors"]]
    if bad_err:
        print("\n REFUSED - VOID, not null: page_edit raised on "
              + str(bad_err) + ".\n That is the false-null signature"
              " (void condition 8). Nothing was sent.")
        return 6
    bad_ctrl = [(p["subset"], p["positive_control"]) for p in plans
                if not p["positive_control"]["ok"]]
    if bad_ctrl:
        print("\n REFUSED - VOID, not null: the positive control failed on "
              + str([b[0] for b in bad_ctrl]) + ".\n The policy never fired, or"
              " edits were applied while nothing was suppressed: the\n"
              " silent-gutting signature (void condition 8). Nothing was sent.\n"
              " " + str(bad_ctrl))
        return 7

    print("\n GUARD PRE-FLIGHT (void condition 8 - all must be zero / OK)")
    for pl in plans:
        pc = pl["positive_control"]
        print(f"  {pl['subset']:<9} page_source={str(pl['page_source']):<10}"
              f" containment={pl['n_containment_violations']}"
              f" page_edit_errors={pl['n_page_edit_errors']}"
              f" mismatches={pl['n_page_edit_mismatch']}")
        print(f"            positive control: fired "
              f"{pc['n_questions_policy_fired']}/{pc['n_questions']}, "
              f"edits applied {pc['n_fact_edits_applied']}, "
              f"facts suppressed {pc['n_facts_suppressed']} -> "
              f"{'OK' if pc['ok'] else 'FAILED'}")

    if args.dry_run:
        print("\n --dry-run: nothing sent.")
        return 0

    answer_fn = make_answer_fn(args, cfg)
    if args.smoke_llm:
        print("\n *** SMOKE RUN - primacy-anchored stub, not a model. ***")
    results = [run_subset(p, answer_fn) for p in plans]
    for res, pl in zip(results, plans):
        res["void_conditions"] = void_condition_report(
            res, ctx["tables"][res["subset"]], pl["page_source"])
    comparisons = {}
    for res in results:
        c = compare_to_oracle(res, load_oracle(res["subset"]))
        if c:
            comparisons[res["subset"]] = c
    print(format_run(results, comparisons))

    payload = {
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(), "mode": cfg.mode,
        "smoke_llm": bool(args.smoke_llm),
        "harness": args.harness,
        "page_source": args.page_source,
        "confirmatory": bool(args.confirmatory),
        "preregistration": "stage0_results/stage1_preregistration_v2.md",
        "operating_point": cell["artifact"],
        "harness": {
            "llm_model": None if args.smoke_llm else cfg.llm_model,
            "llm_base_url": None if args.smoke_llm else cfg.llm_base_url,
            "temperature": cfg.llm_temperature,
            "max_tokens": GENERATION_MAX_TOKENS,
            "system_message": SYSTEM_MESSAGE,
            "prompt_shape": (
                WHOLE_CONTEXT_SHAPE if args.harness == "whole_context"
                else RETRIEVAL_SHAPE),
            "prompt_source": (
                "hnav.stage1.stale_suppression_probe (imported verbatim)"
                if args.harness == "whole_context" else
                "hnav.stage1.calibrate_read_policy.build_user_prompt "
                "(imported verbatim)"),
            "grader": "hnav.labeling.counterfactual.substring_exact_match",
            "comparability_to_oracle_probe": (
                "same prompt shape, same system message, same grader, same "
                "frozen :8003 substrate - the headline is a RATIO against the "
                "oracle arms and a ratio taken across harnesses is meaningless"
                if args.harness == "whole_context" else
                "SAME system message, grader and frozen :8003 substrate, but a "
                "DIFFERENT prompt shape from the oracle probe (rank-ordered "
                "multi-block page vs one whole-context block). Ratios against "
                "the oracle are cross-harness; see "
                "detector_vs_oracle.harness_caveat."),
        },
        "arms": {
            "native": "untouched context",
            "native_repeat":
                "same prompt, independent second call (A/A floor)",
            "detector_suppress":
                "ReadFactPolicy('suppress'): drop every stale member of every "
                "verified group; serials NOT renumbered",
            "detector_demote_late":
                "ReadFactPolicy('demote_late'): every verified group's LATEST "
                "carrier moves to the END, ascending serial order",
            "detector_anti":
                "measurement-only mirror: the same LATEST carriers move to the "
                "FRONT",
        },
        "detector_inputs": {
            "prepass": {s: str(prepass_path(cfg, s, getattr(args, "page_source", None)))
                        for s in subsets},
            "prepass_unstamped_nli_config": ctx["unstamped"],
            "embed_cache_namespace": ctx["embed_namespaces"],
            "max_pair_cosine_error_vs_prepass": ctx["max_cosine_error"],
            "gate": "hnav.core.read_gate.ReadGate (real); NLI replayed from the "
                    "prepass via calibrate_read_policy.ReplayNLI",
            "policy": "hnav.core.read_policy.ReadFactPolicy",
            "page_contract_check":
                "each suppress/demote arm re-derived through "
                "mab_adapter.page_edit and asserted byte-identical",
            "containment_check":
                "every id the policy names is asserted present on the page it "
                "will edit, per question, before any call is sent; a violation "
                "REFUSES the run (exit 5) instead of falling back to native",
            "void_if_nonzero": ["n_page_edit_mismatch",
                                "n_containment_violations"],
        },
        "split": {"calibration": list(CALIBRATION),
                  "refused": list(CONFIRMATORY)},
        "budget": bud,
        "results": results,
        "detector_vs_oracle": comparisons,
    }
    tag = "" if args.harness == "whole_context" else f"_{args.harness}"
    out = Path(args.out) if args.out else (
        cfg.out_dir / (f"detector_gap{tag}_SMOKE.json" if args.smoke_llm
                       else f"detector_gap{tag}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
