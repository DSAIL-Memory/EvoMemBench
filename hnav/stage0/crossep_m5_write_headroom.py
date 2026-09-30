#!/usr/bin/env python3
"""M5 — CrossEp-Know write-side headroom.  MEASUREMENT ONLY, NO POLICY.  [T10]

The KAPI_KARARI write-path NO_GO was measured on MemoryAgentBench context
chunks — an append-only, dedup-free substrate where duplicate_rate measured
0.0 everywhere and the veto chain left 0–1.6% headroom. CrossEp-Know is a
materially different substrate (real memory APIs, LLM-mediated extraction,
free-text candidates, per-context banks), so the write-cascade question is
UNMEASURED there — neither covered by the NO_GO nor authorized past it
(``HNAV_VISION_GAP.md`` §2.2). This script is the Stage-0-style measurement
that the [GATE] decision requires. It fits nothing, decides nothing, and
refuses to: every number is a base rate or a distribution.

What it does: replays a per-context write stream through the (T10-wired)
``CLBenchAdapter`` shadow instrumentation — geometry (sim_max / QR novelty /
MD5 exact-dup), nearest-neighbour marginal diff, and the provisional-insert
effect with the candidate as its own probe — then adds two offline layers the
adapter deliberately does not carry: a lexical Jaccard near-dup proxy (like
``labeling/marginal_diff.py``, a proxy only — no threshold may be derived
from it) and, optionally, bidirectional NLI contradiction scoring of
(nearest-prior, candidate) pairs via the read gate's cross-encoder.

Write streams:

``qwen3_embedding`` (default)
    Reconstructed exactly from a benchmark output JSONL: per sample,
    ``query`` is the last user message, the trajectory is re-serialized the
    way ``cl_bench_memory/trajectory.py`` does it, and the backend's own
    1024-token sentence chunker is transcribed below (``fallback_chunker`` is
    recorded if nltk/tiktoken are unavailable, so a fallback run can never be
    mistaken for the real chunking). This is the LLM-free backend: its write
    stream is fully replayable offline.

``mem0_history``
    Reads ``<run_dir>/<context_id>/history.db`` from any completed mem0 run
    (the vendored mem0 logs every ADD with its text). No such run artifact
    exists yet — the mem0/A-mem streams are LLM-extracted and can only be
    produced by running the benchmark; this reader makes the measurement a
    file-drop away once one exists.

``generic_jsonl``
    ``{"context_id": ..., "task_id": ..., "text": ...}`` events, one per
    line, for any future backend capture (A-mem has no persisted store in the
    CL-bench wiring, so its stream must be captured at run time).

Stratification: by backend, by calibration/held-out split
(``hnav/labeling/crossep_split.json``, cluster = ``context_id``), and by
cluster. Aggregates are CLUSTER-FIRST (means/medians over per-cluster
statistics) — candidate-level pooling across clusters is never reported
(ICC 0.346, design effect 3.20).

Leakage: reads benchmark output records (offline Stage-0 layer). It uses
``messages`` + ``model_output`` + ``metadata`` only — the write stream a real
run would produce — and never touches rubrics or any evaluation field.

    python hnav/stage0/crossep_m5_write_headroom.py --smoke-embedder
    python hnav/stage0/crossep_m5_write_headroom.py --smoke-embedder --nli stub
    python hnav/stage0/crossep_m5_write_headroom.py            # real embedder (GPU)
    python hnav/stage0/crossep_m5_write_headroom.py --backend mem0_history \
        --source Cross-Episode-Knowledge/CROSSEP-KNOW/memories/context_memory/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.clbench_adapter import CLBenchAdapter  # noqa: E402
from hnav.core.audit import AuditLogger  # noqa: E402
from hnav.core.diff_geometry import DiffGeometryModule  # noqa: E402
from hnav.core.geometry import GeometryModule  # noqa: E402
from hnav.core.replica import NumpyCosineReplica  # noqa: E402
from hnav.core.retrieval_signals import RetrievalSignals  # noqa: E402
from hnav.core.types import StoreView, candidate_to_record  # noqa: E402
from hnav.labeling.crossep_split import load_split  # noqa: E402

DEFAULT_SOURCE = (REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/outputs/"
                  "context_nomemory/"
                  "CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory.jsonl")

SIM_GRID = (0.80, 0.85, 0.90, 0.95, 0.99)
JACCARD_GRID = (0.50, 0.70, 0.90)
NLI_GRID = (0.50, 0.70, 0.90)
CHUNK_SIZE = 1024        # the CL-bench backend's own setting


# ── the CL-bench chunker, transcribed ────────────────────────────────────────
def clbench_chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Transcription of ``cl_bench_memory/chunking.py::chunk_text``.

    A copy rather than an import: ``cl_bench_memory/__init__`` drags in every
    backend (mem0, chromadb, ...) at import time, and ``hnav/stage0`` must
    stay importable without them. Raises when nltk/tiktoken are missing;
    :func:`build_chunks` catches that and records the fallback.
    """
    import nltk
    import tiktoken

    nltk.download("punkt_tab", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = nltk.sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sentence in sentences:
        n = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if count + n > chunk_size and current:
            chunks.append(" ".join(current))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def _fallback_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """~4 chars/token paragraph grouping, used only when the real chunker's
    dependencies are missing. Marked in the output; never silently pooled."""
    budget = chunk_size * 4
    chunks, current, size = [], [], 0
    for para in text.split("\n"):
        if size + len(para) > budget and current:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def build_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> tuple[list[str], bool]:
    """``(chunks, used_fallback)``."""
    try:
        return clbench_chunk_text(text, chunk_size), False
    except Exception:  # noqa: BLE001 — nltk/tiktoken absent or punkt undownloadable
        return _fallback_chunks(text, chunk_size), True


# ── trajectory re-serialization (transcribed from cl_bench_memory) ───────────
def format_trajectory(messages: list[dict], model_output: str) -> str:
    parts = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            parts.append(f"**[System Context]**\n{content}")
        elif role == "user":
            parts.append(f"**[User Question]**\n{content}")
        elif role == "assistant":
            parts.append(f"**[Assistant Response]**\n{content}")
    if model_output:
        parts.append(f"**[Model Output]**\n{model_output}")
    return "\n\n".join(parts)


def last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# ── write-stream sources ─────────────────────────────────────────────────────
def iter_qwen3_stream(source: Path, chunk_size: int = CHUNK_SIZE,
                      ) -> tuple["OrderedDict[str, list[dict]]", bool]:
    """``context_id -> [write event]`` in file order, exactly as the
    ``qwen3_embedding_4b`` backend would have chunked and admitted them."""
    streams: "OrderedDict[str, list[dict]]" = OrderedDict()
    any_fallback = False
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            cid = meta.get("context_id", "unknown")
            task_id = meta.get("task_id", rec.get("idx", ""))
            query = last_user_message(rec.get("messages", []))
            trajectory = format_trajectory(rec.get("messages", []),
                                           rec.get("model_output", ""))
            full = f"{query}\n\n{trajectory}" if query else trajectory
            chunks, used_fallback = build_chunks(full, chunk_size)
            any_fallback = any_fallback or used_fallback
            events = streams.setdefault(cid, [])
            for i, chunk in enumerate(chunks):
                events.append({
                    "context_id": cid, "task_id": task_id, "chunk_index": i,
                    "context_category": meta.get("context_category", ""),
                    "text": chunk,
                })
    return streams, any_fallback


def iter_mem0_history(run_dir: Path) -> tuple["OrderedDict[str, list[dict]]", bool]:
    """ADD events from every ``<run_dir>/<context_id>/history.db``."""
    import sqlite3

    streams: "OrderedDict[str, list[dict]]" = OrderedDict()
    dbs = sorted(run_dir.glob("*/history.db"))
    if not dbs:
        raise FileNotFoundError(
            f"no <context_id>/history.db under {run_dir} — a completed mem0 "
            "run is required before this backend can be measured")
    for db in dbs:
        cid = db.parent.name
        con = sqlite3.connect(db)
        try:
            rows = con.execute(
                "SELECT memory_id, new_memory FROM history "
                "WHERE event = 'ADD' AND (is_deleted IS NULL OR is_deleted = 0) "
                "ORDER BY created_at, rowid").fetchall()
        finally:
            con.close()
        streams[cid] = [
            {"context_id": cid, "task_id": mid, "chunk_index": i,
             "context_category": "", "text": text or ""}
            for i, (mid, text) in enumerate(rows) if text
        ]
    return streams, False


def iter_generic_jsonl(source: Path) -> tuple["OrderedDict[str, list[dict]]", bool]:
    streams: "OrderedDict[str, list[dict]]" = OrderedDict()
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            cid = ev["context_id"]
            events = streams.setdefault(cid, [])
            ev.setdefault("chunk_index", len(events))
            ev.setdefault("context_category", "")
            events.append(ev)
    return streams, False


# ── replay ───────────────────────────────────────────────────────────────────
def replay_context(context_id: str, events: list[dict], embedder, cfg,
                   compute_diff: bool = True) -> list[dict]:
    """Replay one context's write stream through a FRESH adapter (the native
    runner gives each context an isolated bank) and return its audit records,
    each augmented with the offline lexical Jaccard proxy."""
    adapter = CLBenchAdapter(
        cfg=cfg,
        embedder=embedder,
        replica=NumpyCosineReplica(embedder, top_k=10),
        audit=AuditLogger(None, run_id=cfg.run_id, suite="crossep_m5",
                          subset=context_id),
        signals=RetrievalSignals(top_m=cfg.top_m, k=cfg.churn_k),
        geometry=GeometryModule(),          # whitening: refused below 200 anyway
        diff=DiffGeometryModule(embedder) if compute_diff else None,
    )
    store = StoreView.from_records([])
    seen_word_sets: list[set[str]] = []
    for ev in events:
        cand = adapter.to_candidate(
            ev["text"], task_id=ev.get("task_id", ""),
            context_id=context_id, chunk_index=ev.get("chunk_index", 0),
            context_category=ev.get("context_category", ""))
        adapter.on_extract(cand, store)

        words = set(ev["text"].lower().split())
        jac = max((len(words & prev) / len(words | prev)
                   for prev in seen_word_sets if words | prev), default=float("nan"))
        adapter.audit.records[-1]["lex_jaccard_max"] = (
            float(jac) if np.isfinite(jac) else None)
        seen_word_sets.append(words)

        store = store.with_provisional(cand)   # native behaviour: always admit
    return adapter.audit.records


# ── statistics (cluster-first, never pooled) ─────────────────────────────────
def embedder_report(emb) -> dict:
    """Everything needed to reproduce (or impeach) this run's vectors.

    Written because the 2026-08-16 artifact could NOT self-certify: that run
    survived only thanks to the GQA fused-kernel fix (T13), and
    ``HFEmbedder.gqa_expansion_applied`` records whether the fix actually
    engaged — but M5 captured no embedder provenance, so the artifact was
    silent about the one field that would have proven it. A silent fallback
    to the MATH path is exactly what that flag exists to catch, and "the run
    did not OOM" is indirect evidence, not a record.

    Must be read AFTER the replay: ``gqa_expansion_applied`` is set during
    the first forward pass, so capturing it before any encoding would always
    log ``None``. The cache namespace is recorded verbatim
    (``model|dtype|Lmax``) so a reader can locate — or fail to locate — these
    exact vectors on disk.
    """
    inner = getattr(emb, "inner", emb)      # unwrap DiskCachedEmbedder
    out = {
        "class": type(inner).__name__,
        "wrapper": type(emb).__name__ if inner is not emb else None,
        "model": getattr(inner, "model_name", None),
        "dtype": getattr(inner, "dtype_name", None),
        "device": getattr(inner, "device", None),
        "max_length": getattr(inner, "max_length", None),
        "max_batch_tokens": getattr(inner, "max_batch_tokens", None),
        "batch": getattr(inner, "batch", None),
        "dim": getattr(inner, "dim", None),
        "cache_namespace": getattr(emb, "key", None),
        "cache_dir": str(getattr(emb, "cache_dir", "")) or None,
    }
    # None (no forward ran yet) and False (the fix did not engage) are
    # DIFFERENT states and stay distinct; "not_applicable" means this
    # embedder has no such flag at all (e.g. HashEmbedder).
    out["gqa_expansion_applied"] = getattr(inner, "gqa_expansion_applied",
                                           "not_applicable")
    return out


def pcts(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "mean": float(x.mean()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "min": float(x.min()), "max": float(x.max())}


def _rate(flags) -> float:
    flags = [bool(f) for f in flags]
    return float(np.mean(flags)) if flags else float("nan")


def _finite(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def cluster_stats(context_id: str, records: list[dict], split: str) -> dict:
    geo = [r.get("geometry") or {} for r in records]
    eff = [r.get("retrieval_effect") or {} for r in records]
    dif = [r for r in (r.get("diff") for r in records) if r]

    sim_max = _finite(g.get("sim_max") for g in geo)
    jac = _finite(r.get("lex_jaccard_max") for r in records)
    rank_self = [e.get("rank_self_after") for e in eff if e.get("rank_self_after") is not None]

    return {
        "context_id": context_id,
        "split": split,
        "context_category": records[0].get("context_category", ""),
        "n_candidates": len(records),
        "store_size_final": len(records),
        "exact_dup_rate": _rate(g.get("is_exact_dup") for g in geo),
        "n_exact_dup": int(sum(bool(g.get("is_exact_dup")) for g in geo)),
        "sim_max": pcts(sim_max),
        "sim_max_frac_ge": {f"{t:.2f}": _rate(v >= t for v in sim_max)
                            for t in SIM_GRID} if sim_max else {},
        "qr_residual": pcts(_finite(g.get("qr_residual_r") for g in geo)),
        "lex_jaccard_max": pcts(jac),
        "lex_jaccard_frac_ge": {f"{t:.2f}": _rate(v >= t for v in jac)
                                for t in JACCARD_GRID} if jac else {},
        "n_diff": len(dif),
        "whole_blob_sim": pcts(_finite(d.get("whole_blob_sim") for d in dif)),
        "diff_sim": pcts(_finite(d.get("diff_sim") for d in dif)),
        "critical_delta_rate_default_tau": _rate(d.get("is_critical_delta")
                                                 for d in dif),
        "rank_self_after": pcts(rank_self),
        "rank_self_top1_rate": _rate(r == 0 for r in rank_self),
        "churn_at_k": pcts(_finite(e.get("churn_at_k") for e in eff)),
        "dH_self": pcts(_finite(e.get("dH_self") for e in eff)),
    }


_SCALAR_KEYS = ("exact_dup_rate", "rank_self_top1_rate",
                "critical_delta_rate_default_tau")
_PCT_KEYS = ("sim_max", "qr_residual", "lex_jaccard_max", "whole_blob_sim",
             "rank_self_after", "churn_at_k", "dH_self")


def aggregate_split(rows: list[dict]) -> dict:
    """Cluster-first aggregation: statistics OF the per-cluster statistics."""
    if not rows:
        return {"n_clusters": 0}
    out: dict = {
        "n_clusters": len(rows),
        "n_candidates_total": int(sum(r["n_candidates"] for r in rows)),
        "candidates_per_cluster": pcts([r["n_candidates"] for r in rows]),
    }
    for key in _SCALAR_KEYS:
        out[f"{key}__over_clusters"] = pcts([r[key] for r in rows])
    for key in _PCT_KEYS:
        p50s = [r[key].get("p50") for r in rows if r[key]]
        out[f"{key}_p50__over_clusters"] = pcts(p50s)
    for grid_key, grid in (("sim_max_frac_ge", SIM_GRID),
                           ("lex_jaccard_frac_ge", JACCARD_GRID)):
        out[f"{grid_key}__over_clusters"] = {
            f"{t:.2f}": pcts([r[grid_key].get(f"{t:.2f}") for r in rows
                              if r[grid_key]])
            for t in grid}
    return out


# ── optional offline NLI (labeling layer privilege) ──────────────────────────
def run_nli(per_context_records: "OrderedDict[str, list[dict]]",
            splits: dict[str, str], engine: str, allowed_splits: set[str],
            max_pairs: int) -> dict:
    """Bidirectional contradiction rates on (nearest-prior, candidate) pairs.

    Offline only: this uses the read gate's cross-encoder from the labeling
    side of the fence. Pair selection is deterministic (context order, then
    candidate order) and capped, and rates are reported per cluster.

    **Truncation is STRUCTURAL here and is reported, never assumed away**
    (T12 note 5). DeBERTa-v3's own ``max_position_embeddings`` is 512 and
    premise and hypothesis share that budget, while CrossEp chunks run to
    ~300 tokens per side after the 1200-char cut — so ~71.5% of pairs are cut
    at the model's ceiling. The knob cannot be raised without leaving the
    checkpoint's validated range, so the residue is measured: the returned
    block carries a ``truncation`` sub-block from the engine itself, and
    every contradiction number in this block must be quoted alongside it.
    """
    if engine == "stub":
        from hnav.core.read_gate import StubNLI
        nli = StubNLI()
    else:
        from hnav.core.read_gate import CrossEncoderNLI
        cfg = _config.get_config()
        # max_length BY KEYWORD and from config (T12 note 5): this call used
        # to stop before it and inherit 256, which premise and hypothesis had
        # to SHARE — so each 1200-char chunk side was cut to ~128 tokens.
        nli = CrossEncoderNLI(model_name=cfg.nli_model, device=engine,
                              dtype="float32", batch=8,
                              max_length=cfg.nli_max_length)

    pairs, owners = [], []
    for cid, records in per_context_records.items():
        if splits.get(cid, "unknown") not in allowed_splits:
            continue
        for r in records:
            prior = r.get("prior_text")
            if prior and len(pairs) < max_pairs:
                a = prior[:1200]
                b = (r.get("candidate_text") or "")[:1200]
                pairs.append((a, b))
                owners.append(cid)
    if not pairs:
        return {"engine": engine, "n_pairs": 0,
                "truncation": (nli.truncation_report()
                               if hasattr(nli, "truncation_report") else None)}

    fwd = nli.score_pairs(pairs)
    bwd = nli.score_pairs([(b, a) for a, b in pairs])
    per_cluster: dict[str, list[float]] = {}
    for cid, f, b in zip(owners, fwd, bwd):
        per_cluster.setdefault(cid, []).append(
            min(f.contradiction, b.contradiction))

    rows = []
    for cid, contras in per_cluster.items():
        rows.append({
            "context_id": cid, "split": splits.get(cid, "unknown"),
            "n_pairs": len(contras),
            "bidir_contra_min": pcts(contras),
            "bidir_contra_frac_ge": {f"{t:.2f}": _rate(v >= t for v in contras)
                                     for t in NLI_GRID},
        })
    return {
        "engine": engine,
        "n_pairs": len(pairs),
        "allowed_splits": sorted(allowed_splits),
        # CrossEp chunk pairs structurally exceed even DeBERTa-v3's 512-token
        # limit (measured: 71.5% of pairs), so the residual truncation is
        # REPORTED rather than assumed away (T12 note 5).
        "truncation": (nli.truncation_report()
                       if hasattr(nli, "truncation_report") else None),
        "per_cluster": rows,
        "bidir_contra_frac_ge__over_clusters": {
            f"{t:.2f}": pcts([r["bidir_contra_frac_ge"][f"{t:.2f}"] for r in rows])
            for t in NLI_GRID},
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3_embedding",
                    choices=["qwen3_embedding", "mem0_history", "generic_jsonl"])
    ap.add_argument("--source", type=Path, default=None,
                    help="output JSONL (qwen3_embedding / generic_jsonl) or a "
                         "mem0 run directory (mem0_history)")
    ap.add_argument("--max-contexts", type=int, default=None)
    ap.add_argument("--max-events-per-context", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--no-diff", action="store_true",
                    help="skip marginal-diff (halves the unique-embed count)")
    ap.add_argument("--nli", default="off",
                    help="off | stub (smoke only) | cpu | cuda:N — offline "
                         "bidirectional contradiction rates on nearest-prior pairs")
    ap.add_argument("--nli-max-pairs", type=int, default=400)
    ap.add_argument("--nli-splits", nargs="*", default=["calibration"],
                    help="which split(s) NLI pairs are drawn from")
    ap.add_argument("--smoke-embedder", action="store_true",
                    help="HashEmbedder instead of the real model: exercises "
                         "every code path with no GPU. MD5 exact-dup and "
                         "lexical-Jaccard numbers are REAL (embedder-free); "
                         "every cosine-based number is MEANINGLESS. Written "
                         "to a separate _SMOKE file.")
    args = ap.parse_args()

    if args.nli == "stub" and not args.smoke_embedder:
        print("--nli stub is a smoke-only engine; add --smoke-embedder", file=sys.stderr)
        return 2

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f" M5 — CrossEp write-side headroom   backend={args.backend}")
    if args.smoke_embedder:
        print("   *** SMOKE RUN — HashEmbedder. Cosine-based numbers are "
              "meaningless; MD5-dup and Jaccard numbers are real. ***")
    else:
        print(f"   model={cfg.embed_model}  dtype={cfg.embed_dtype}  "
              f"top_m={cfg.top_m}  k={cfg.churn_k}")
    print("=" * 74)

    # write stream
    if args.backend == "qwen3_embedding":
        source = args.source or DEFAULT_SOURCE
        streams, used_fallback = iter_qwen3_stream(source, args.chunk_size)
    elif args.backend == "mem0_history":
        if args.source is None:
            print("--backend mem0_history requires --source <run_dir>", file=sys.stderr)
            return 2
        streams, used_fallback = iter_mem0_history(args.source)
        source = args.source
    else:
        if args.source is None:
            print("--backend generic_jsonl requires --source <events.jsonl>",
                  file=sys.stderr)
            return 2
        streams, used_fallback = iter_generic_jsonl(args.source)
        source = args.source

    split_art = load_split()
    calib = set(split_art["calibration"])
    held = set(split_art["heldout"])
    splits = {cid: ("calibration" if cid in calib
                    else "heldout" if cid in held else "unknown")
              for cid in streams}

    if args.max_contexts:
        streams = OrderedDict(list(streams.items())[: args.max_contexts])

    if args.smoke_embedder:
        from hnav.core.embedding import HashEmbedder
        emb = HashEmbedder(dim=64)
    else:
        from hnav.core.embedding import build_embedder
        emb = build_embedder(cfg)

    per_context_records: "OrderedDict[str, list[dict]]" = OrderedDict()
    rows = []
    n_events_total = sum(len(v) for v in streams.values())
    print(f" {len(streams)} contexts, {n_events_total} write events"
          + ("  [FALLBACK CHUNKER]" if used_fallback else ""))

    for i, (cid, events) in enumerate(streams.items()):
        if args.max_events_per_context:
            events = events[: args.max_events_per_context]
        records = replay_context(cid, events, emb, cfg,
                                 compute_diff=not args.no_diff)
        # context_category rides along for the stats row
        for r, ev in zip(records, events):
            r["context_category"] = ev.get("context_category", "")
        per_context_records[cid] = records
        rows.append(cluster_stats(cid, records, splits.get(cid, "unknown")))
        if (i + 1) % 20 == 0 or i + 1 == len(streams):
            print(f"   replayed {i + 1}/{len(streams)} contexts", flush=True)

    by_split = {
        s: aggregate_split([r for r in rows if r["split"] == s])
        for s in ("calibration", "heldout", "unknown")
        if any(r["split"] == s for r in rows)
    }

    nli_block = None
    if args.nli != "off":
        print(f" scoring NLI pairs ({args.nli}, splits={args.nli_splits}, "
              f"cap={args.nli_max_pairs}) ...", flush=True)
        nli_block = run_nli(per_context_records, splits, args.nli,
                            set(args.nli_splits), args.nli_max_pairs)
        if args.nli == "stub":
            nli_block["NLI_STUB"] = True

    embed_accounting = None
    if hasattr(emb, "n_hits"):
        embed_accounting = {"cache_hits": emb.n_hits, "cache_misses": emb.n_misses}
    embedder_provenance = embedder_report(emb)

    # the honest headroom sketch this measurement exists for — descriptive
    # only; any operating threshold must later be fit on calibration clusters
    def _mean_over(split_name: str, key: str):
        blk = by_split.get(split_name, {})
        return (blk.get(f"{key}__over_clusters") or {}).get("mean")

    verdict = {
        "mab_baseline": {"duplicate_rate": 0.0,
                         "would_intervene_after_vetoes": "0.000-0.016"},
        "exact_dup_rate_cluster_mean": {
            s: _mean_over(s, "exact_dup_rate") for s in by_split},
        "near_dup_sim_ge_0.95_cluster_mean": {
            s: ((by_split[s].get("sim_max_frac_ge__over_clusters") or {})
                .get("0.95") or {}).get("mean") for s in by_split},
        "lex_jaccard_ge_0.70_cluster_mean": {
            s: ((by_split[s].get("lex_jaccard_frac_ge__over_clusters") or {})
                .get("0.70") or {}).get("mean") for s in by_split},
        "note": ("Cosine-based entries are meaningless under --smoke-embedder; "
                 "MD5 exact-dup and lexical Jaccard are embedder-free and real."
                 if args.smoke_embedder else
                 "Measured in the H-Nav embedding space "
                 f"({cfg.embed_model}/{cfg.embed_dtype}), not the native "
                 "DashScope space — declared deviation, same methodology as "
                 "the primary arena's Stage-0."),
    }

    result = {
        "measurement": "m5_crossep_write_headroom",
        "backend": args.backend,
        "source": str(source),
        "smoke": bool(args.smoke_embedder),
        "embed_model": None if args.smoke_embedder else cfg.embed_model,
        "embed_dtype": None if args.smoke_embedder else cfg.embed_dtype,
        "fallback_chunker": used_fallback,
        "chunk_size": args.chunk_size,
        "diff_computed": not args.no_diff,
        "split_artifact": "hnav/labeling/crossep_split.json",
        "n_contexts": len(streams),
        "n_write_events": n_events_total,
        "embed_accounting": embed_accounting,
        "embedder_provenance": embedder_provenance,
        "by_split": by_split,
        "per_cluster": rows,
        "nli": nli_block,
        "headroom_indicators": verdict,
    }
    if args.smoke_embedder:
        result["SMOKE_HASH_EMBEDDER"] = True

    suffix = "_SMOKE" if args.smoke_embedder else ""
    out = cfg.out_dir / f"m5_crossep_write_headroom_{args.backend}{suffix}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(" M5 SUMMARY  (cluster-first — never pooled across clusters)")
    print("=" * 74)
    for s, blk in by_split.items():
        dup = (blk.get("exact_dup_rate__over_clusters") or {}).get("mean", float("nan"))
        near = ((blk.get("sim_max_frac_ge__over_clusters") or {})
                .get("0.95") or {}).get("mean", float("nan"))
        jacc = ((blk.get("lex_jaccard_frac_ge__over_clusters") or {})
                .get("0.70") or {}).get("mean", float("nan"))
        print(f" {s:<12} clusters={blk['n_clusters']:>3}  "
              f"cands={blk.get('n_candidates_total', 0):>5}  "
              f"dup(mean)={dup:.4f}  sim>=.95(mean)={near:.4f}  "
              f"jacc>=.70(mean)={jacc:.4f}")
    if nli_block:
        top = nli_block.get("bidir_contra_frac_ge__over_clusters", {})
        p = (top.get("0.90") or {}).get("mean", float("nan"))
        print(f" NLI ({nli_block['engine']}): pairs={nli_block['n_pairs']}  "
              f"bidir-contra>=0.90 cluster-mean={p:.4f}")
        # The contradiction figure is uninterpretable without this: DeBERTa's
        # 512-token ceiling cuts most CrossEp chunk pairs (T12 note 5).
        trunc = nli_block.get("truncation")
        if trunc:
            print(f"   truncation: {trunc.get('n_truncated')}/{trunc.get('n_scored')} "
                  f"inputs cut at max_length={trunc.get('max_length')} "
                  f"(rate {trunc.get('truncation_rate', float('nan')):.4f}) "
                  f"— quote this beside every contradiction number above")
    print(f"\n wrote {out}")
    print(" This measurement fits NOTHING. Operating thresholds, if the [GATE]")
    print(" ever authorizes any, may be fit on the calibration clusters only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
