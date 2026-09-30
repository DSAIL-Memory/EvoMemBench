#!/usr/bin/env python3
"""M3 — headroom, per candidate H-Nav family.  [T6]

Write side: total write decisions, duplicate / near-duplicate / update /
critical-delta / conflict / stale-replacement rates, how often a geometry gate
*would* intervene, and how often that intervention *could* change downstream
correctness.

Read side: total reads, ambiguous / low-margin / stale / high-interference /
relevant-below-k rates, and the downstream failure rate within each.

**Thresholds are fit on the calibration split only** (``sh_6k`` + ``sh_32k``),
frozen, and applied unchanged to the confirmatory split. The script refuses to
fit on a confirmatory subset; ``--allow-no-calibration`` exists only for smoke
runs and stamps the output as unfit-for-analysis.

Counterfactual labeling costs one answer call per (candidate x affected
question); ``--max-counterfactuals`` bounds it and ``--no-llm`` skips it
entirely, in which case base rates and would-intervene rates are still produced
and the correctness columns are reported as absent rather than as zero.

    python hnav/stage0/m3_headroom.py --no-llm
    python hnav/stage0/m3_headroom.py --max-counterfactuals 400
    python hnav/stage0/m3_headroom.py --smoke-embedder --no-llm --subsets sh_6k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import explode_facts, fact_key  # noqa: E402
from hnav.core.diff_geometry import DiffGeometryModule  # noqa: E402
from hnav.core.embedding import build_embedder  # noqa: E402
from hnav.core.geometry import (ABTTWhitening, GeometryModule, TauPolicy,  # noqa: E402
                                qr_residual, verbatim_values)
from hnav.core.replica import FaissFlatReplica  # noqa: E402
from hnav.core.retrieval_signals import RetrievalSignals  # noqa: E402
from hnav.core.types import Candidate, MemoryRecord, StoreView  # noqa: E402
from hnav.labeling import labels as L  # noqa: E402
from hnav.labeling.conflict_index import Entry, ReadConflictIndex, WriteConflictIndex  # noqa: E402
from hnav.labeling.counterfactual import (CounterfactualLabeler,  # noqa: E402
                                          local_answer_fn, map_questions_to_keys)
from hnav.stage0.m2_retrieval_calibration import build_chunks, pcts, subset_name  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"

CALIBRATION = ("sh_6k", "sh_32k")
CONFIRMATORY = ("sh_64k", "sh_262k")


# ── span extraction for the primary arena (adapter-supplied, exact) ──────────
def object_spans(old: str, new: str) -> tuple[str, str, bool]:
    """Both facts parsed with the induced templates; take the object slots."""
    po, pn = fact_key(old), fact_key(new)
    if po[1] is not None and pn[1] is not None:
        return po[1], pn[1], True
    raise ValueError("parse failed")


class _QRBasis:
    """Cached QR basis over a random subsample of the store.

    Recomputing a 512-column QR per candidate is ~1.3 GFLOP; on 18,332 candidates
    that is not a measurement, it is a weekend. The basis is a random subsample
    to begin with, so refreshing it every ``interval`` admissions is a change of
    sampling frequency, not of estimator. The interval is recorded in the output.
    """

    def __init__(self, interval: int = 500, max_basis: int = 512) -> None:
        self.interval = interval
        self.max_basis = max_basis
        self.q: np.ndarray | None = None
        self.built_at = -1
        self.subsampled = False

    def residual(self, vec: np.ndarray, bank: np.ndarray, n: int) -> float:
        if bank.shape[0] == 0:
            return 1.0
        if self.q is None or n - self.built_at >= self.interval:
            b = bank
            self.subsampled = b.shape[0] > self.max_basis
            if self.subsampled:
                idx = np.random.default_rng(n).choice(b.shape[0], self.max_basis,
                                                      replace=False)
                b = b[idx]
            self.q, _ = np.linalg.qr(b.T.astype(np.float64))
            self.built_at = n
        v = np.asarray(vec, dtype=np.float64)
        return float(np.linalg.norm(v - self.q @ (self.q.T @ v)))


# ── per-subset state ─────────────────────────────────────────────────────────
def build_state(item, name, emb, cfg, args) -> dict:
    context = item["context"]
    facts = explode_facts(context)
    chunks, fallback = build_chunks(context, args.chunk_size)

    fact_vecs = emb.encode([t for _, t in facts])
    chunk_vecs = emb.encode(chunks)

    fact_store = StoreView.from_records([
        MemoryRecord(id=f"fact:{s}", text=t, vector=fact_vecs[i], version=s,
                     metadata={"key": fact_key(t)[0], "object": fact_key(t)[1]})
        for i, (s, t) in enumerate(facts)])
    chunk_store = StoreView.from_records([
        MemoryRecord(id=f"chunk:{i}", text=t, vector=chunk_vecs[i], version=i)
        for i, t in enumerate(chunks)])

    # chunk membership, both directions — the reverse map keeps the read pass
    # linear instead of scanning every chunk for every stale hit.
    fact_of_chunk: dict[str, list[str]] = {}
    chunk_of_fact: dict[str, str] = {}
    for i, c in enumerate(chunks):
        members = [f"fact:{s}" for s, _ in explode_facts(c)]
        fact_of_chunk[f"chunk:{i}"] = members
        for f in members:
            chunk_of_fact[f] = f"chunk:{i}"

    read_index = ReadConflictIndex()
    for r in fact_store.records:
        read_index.add(r.metadata["key"],
                       Entry(r.version, r.id, r.text, r.metadata["object"]))

    return {"name": name, "facts": facts, "fact_vecs": fact_vecs,
            "fact_store": fact_store, "chunk_store": chunk_store,
            "chunks": chunks, "fallback_chunker": fallback,
            "fact_of_chunk": fact_of_chunk, "chunk_of_fact": chunk_of_fact,
            "read_index": read_index,
            "questions": map_questions_to_keys(item)}


# ── write pass ───────────────────────────────────────────────────────────────
def write_pass(state, cfg, args) -> list[dict]:
    """Replays the write path in serial order, computing exactly the signals an
    online policy would have had — ``latest_before`` only, no future facts."""
    facts, vecs = state["facts"], state["fact_vecs"]
    read_index = state["read_index"]

    geom = GeometryModule(
        whitening=ABTTWhitening(cfg.whiten_components, cfg.whiten_min_fit_n),
        tau_policy=TauPolicy())
    span_emb = _SpanEmbedder(state)
    diff_mod = DiffGeometryModule(span_emb, span_fn=object_spans)
    write_index = WriteConflictIndex()
    qr = _QRBasis(interval=args.qr_refresh)

    keys_with_question = {q["key"] for q in state["questions"] if q["key"]}
    rows: list[dict] = []
    admitted_vecs: list[np.ndarray] = []
    seen_texts: set[str] = set()
    object_vecs: list[np.ndarray] = []

    for i, (serial, text) in enumerate(facts):
        key, obj = fact_key(text)
        vec = np.asarray(vecs[i], dtype=np.float64)
        bank = np.asarray(admitted_vecs, dtype=np.float64) if admitted_vecs \
            else np.zeros((0, vec.shape[0]))

        sim_max, argmax_id = float("nan"), None
        if bank.shape[0]:
            sims = bank @ vec
            j = int(np.argmax(sims))
            sim_max, argmax_id = float(sims[j]), f"fact:{facts[j][0]}"
        r = qr.residual(vec, bank, len(admitted_vecs))

        prior = write_index.latest_before(key, serial)     # ONLINE lookup
        newest = read_index.latest(key)                    # OFFLINE, for the labels
        later_exists = bool(newest and newest.version > serial
                            and newest.object != obj)

        d = None
        if prior is not None:
            d = diff_mod.compute_if_update(
                prior.text, text,
                prior_objects=np.asarray(object_vecs) if object_vecs else None)

        tau_t = geom.tau_policy.tau_t
        geom.tau_policy.update(sim_max)

        ctx = {
            "has_prior_same_key": prior is not None,
            "prior_object": prior.object if prior else None,
            "candidate_object": obj,
            "whole_blob_sim": d.whole_blob_sim if d else None,
            "diff_sim": d.diff_sim if d else None,
            "tau_high": diff_mod.tau_high, "tau_low": diff_mod.tau_low,
            "sim_max": sim_max, "tau": tau_t,
            "verbatim_new_values": sorted(
                verbatim_values(text) - verbatim_values(prior.text if prior else "")),
            "is_exact_dup": text in seen_texts,
            "later_fact_exists": later_exists,
            "question_maps_to_key": key in keys_with_question,
            "in_source_not_candidate": False,   # every source fact becomes a candidate here
        }
        fired = L.apply_write_labels(ctx)

        rows.append({
            "id": f"fact:{serial}", "serial": serial, "index": i,
            "key": list(key) if key else None, "object": obj,
            "sim_max": sim_max, "argmax_id": argmax_id, "qr_residual": r,
            "tau_t": tau_t,
            "whole_blob_sim": d.whole_blob_sim if d else None,
            "diff_sim": d.diff_sim if d else None,
            "diff_novelty": d.diff_novelty if d else None,
            "is_critical_delta": bool(d.is_critical_delta) if d else False,
            "parse_ok": bool(d.parse_ok) if d else None,
            "has_prior": prior is not None,
            "has_new_verbatim": bool(ctx["verbatim_new_values"]),
            "is_newest_for_key": not later_exists,
            "labels": fired,
        })

        seen_texts.add(text)
        admitted_vecs.append(vec)
        write_index.observe(key, Entry(serial, f"fact:{serial}", text, obj))
        if obj:
            object_vecs.append(span_emb.one(obj))

    return rows


class _SpanEmbedder:
    """Embeds object spans, memoized per subset — many facts share an object."""

    def __init__(self, state) -> None:
        self._emb = state.setdefault("_span_emb_cache", {})
        self._backend = state["_embedder"]
        self.dim = getattr(self._backend, "dim", 0)

    def one(self, text: str) -> np.ndarray:
        if text not in self._emb:
            self._emb[text] = np.asarray(self._backend.encode([text])[0], dtype=np.float64)
        return self._emb[text]

    def encode(self, texts):
        return np.stack([self.one(t) for t in texts])


# ── read pass ────────────────────────────────────────────────────────────────
def read_pass(state, cfg, args) -> list[dict]:
    chunk_store = state["chunk_store"]
    replica = FaissFlatReplica(state["_embedder"], top_k=args.top_k)
    signals = RetrievalSignals(top_m=cfg.top_m, k=cfg.churn_k)
    read_index = state["read_index"]
    fact_of_chunk = state["fact_of_chunk"]
    by_id = {r.id: r for r in state["fact_store"].records}

    rows = []
    for q in state["questions"]:
        view = replica.rank(chunk_store, q["question"], top_k=args.top_k)
        sig = signals.compute(view)
        retrieved_chunks = set(view.top_ids)
        retrieved_facts = [by_id[f] for c in retrieved_chunks
                           for f in fact_of_chunk.get(c, []) if f in by_id]

        stale_hits = []
        for rec in retrieved_facts:
            newest = read_index.superseder_of(rec.id)
            if newest is None:
                continue
            holder = state["chunk_of_fact"].get(newest.id)
            stale_hits.append({"old_id": rec.id, "new_id": newest.id,
                               "superseder_retrieved": holder in retrieved_chunks})

        key = tuple(q["key"]) if q["key"] else None
        conflicted = {r.metadata["key"] for r in retrieved_facts
                      if read_index.is_conflicted(r.metadata["key"])}
        # READ_DISTRACTOR / READ_MISSING use the OFFLINE question->key mapping.
        # The online form needs a query-side parser, which Stage 0 does not
        # build, so these two rates are an UPPER BOUND on what an online
        # detector could reach. Recorded here, not silently equated.
        subject_only = sum(1 for r in retrieved_facts
                           if key and r.metadata["key"]
                           and r.metadata["key"][1] == key[1]
                           and r.metadata["key"][0] != key[0])
        key_hit = any(r.metadata["key"] == key for r in retrieved_facts) if key else True

        rows.append({
            "_label_ctx": {"stale_hits": stale_hits,
                           "n_conflicted_keys_in_topk": len(conflicted - {None}),
                           "nmargin": sig.nmargin, "H_z": sig.H_z,
                           "n_subject_only_matches": subject_only,
                           "key_matched_in_topk": key_hit},
            "index": q["index"], "question": q["question"], "truths": q["truths"],
            "key": q["key"], "conflicted_key": q["conflicted"],
            "target_serial": q["target_serial"],
            "top_ids": view.top_ids,
            "nmargin": sig.nmargin, "H_z": sig.H_z, "H_vn": sig.H_vn,
            "margin": sig.margin, "eff_size": sig.eff_size,
            "n_stale": len(stale_hits),
            "n_superseder_missing": sum(1 for h in stale_hits
                                        if not h["superseder_retrieved"]),
            "missing_superseder_ids": [h["new_id"] for h in stale_hits
                                       if not h["superseder_retrieved"]],
        })
    return rows


def label_read_rows(rows: list[dict], thresholds: dict) -> list[dict]:
    """Apply the read labels with FROZEN thresholds.

    Split from :func:`read_pass` so the expensive ranking happens once while the
    thresholds are still being fit on the calibration split — labelling a
    confirmatory subset never influences the thresholds applied to it.
    """
    for r in rows:
        ctx = dict(r["_label_ctx"])
        ctx["nmargin_threshold"] = thresholds["nmargin"]
        ctx["H_z_threshold"] = thresholds["H_z"]
        r["labels"] = L.apply_read_labels(ctx)
    return rows


# ── thresholds ───────────────────────────────────────────────────────────────
def fit_thresholds(read_rows: list[dict], write_rows: list[dict]) -> dict:
    """Calibration-split percentiles. Frozen after this call.

    ``nmargin`` p25 and ``H_z`` p75 define the ambiguous / high-interference
    strata; ``r_min`` p10 of the QR residual defines "not novel enough to be
    worth admitting" for the hypothetical geometry gate. Percentile choices are
    stated here rather than searched — searching them would be tuning on the
    outcome.
    """
    nm = [r["nmargin"] for r in read_rows if np.isfinite(r["nmargin"])]
    hz = [r["H_z"] for r in read_rows if np.isfinite(r["H_z"])]
    rr = [r["qr_residual"] for r in write_rows if np.isfinite(r["qr_residual"])]
    return {"nmargin": float(np.percentile(nm, 25)) if nm else float("nan"),
            "H_z": float(np.percentile(hz, 75)) if hz else float("nan"),
            "r_min": float(np.percentile(rr, 10)) if rr else float("nan"),
            "percentiles": {"nmargin": 25, "H_z": 75, "r_min": 10},
            "n_fit_read_rows": len(read_rows), "n_fit_write_rows": len(write_rows)}


# ── headroom tables ──────────────────────────────────────────────────────────
def write_headroom(rows: list[dict], outcomes: dict, thresholds: dict) -> dict:
    n = len(rows)
    counts = Counter(l for r in rows for l in r["labels"])
    classes = Counter(o["counterfactual_class"] for o in outcomes.values())

    # The hypothetical geometry gate: sim_max >= tau_t AND r < r_min (plan §3,
    # step 5). Stage 0 only measures how often it WOULD fire; the policy that
    # would act on it is gated behind T8 and is not written.
    r_min = thresholds.get("r_min", float("nan"))
    would = [r for r in rows
             if np.isfinite(r["sim_max"]) and np.isfinite(r["qr_residual"])
             and r["sim_max"] >= r["tau_t"] and r["qr_residual"] < r_min]
    # Steps 2-4 are vetoes ON suppression, never triggers for it: a candidate
    # that introduces a new verbatim value, or is a critical delta, is never
    # suppressed however close it sits to an admitted fact.
    after_vetoes = [r for r in would
                    if not r["is_critical_delta"] and not r["has_new_verbatim"]]
    would_ids = {r["id"] for r in after_vetoes}
    could_change = [o for cid, o in outcomes.items()
                    if cid in would_ids
                    and o["counterfactual_class"] in ("must_write", "must_suppress")]
    graded_would = [cid for cid in outcomes if cid in would_ids]

    def rate(k: str) -> float:
        return counts[k] / n if n else float("nan")

    return {
        "n_write_decisions": n,
        "duplicate_rate": rate("WRITE_DUPLICATE"),
        "conflict_rate": rate("WRITE_CONFLICT"),
        "critical_delta_rate": rate("WRITE_CRITICAL_DELTA"),
        "redundant_rate": rate("WRITE_REDUNDANT"),
        "stale_supersede_rate": rate("WRITE_STALE_SUPERSEDE"),
        "unnecessary_rate": rate("WRITE_UNNECESSARY"),
        "update_rate": sum(r["has_prior"] for r in rows) / n if n else float("nan"),
        "would_intervene_rate": len(would) / n if n else float("nan"),
        "would_intervene_after_vetoes_rate": len(after_vetoes) / n if n else float("nan"),
        "n_vetoed": len(would) - len(after_vetoes),
        "label_counts": dict(counts),
        "n_counterfactuals": len(outcomes),
        "counterfactual_classes": dict(classes),
        "n_would_intervene_graded": len(graded_would),
        "could_change_correctness_rate":
            len(could_change) / len(graded_would) if graded_would else None,
    }


def read_headroom(rows: list[dict], outcomes: list[dict]) -> dict:
    n = len(rows)
    counts = Counter(l for r in rows for l in r["labels"])
    by_index = {o["question_index"]: o for o in outcomes}

    strata = {}
    for label in sorted(counts):
        members = [r for r in rows if label in r["labels"]]
        graded = [by_index[r["index"]] for r in members if r["index"] in by_index]
        strata[label] = {
            "n": len(members),
            "rate": len(members) / n if n else float("nan"),
            "n_graded": len(graded),
            "failure_rate_native": (sum(1 for g in graded if not g["correct_native"])
                                    / len(graded)) if graded else None,
            "repair_helped": sum(1 for g in graded if g["repair_helped"]) or 0,
            "repair_harmed": sum(1 for g in graded if g["repair_harmed"]) or 0,
        }

    graded_all = list(by_index.values())
    return {
        "n_reads": n,
        "label_counts": dict(counts),
        "strata": strata,
        "n_graded": len(graded_all),
        "accuracy_native": (sum(g["correct_native"] for g in graded_all)
                            / len(graded_all)) if graded_all else None,
        "accuracy_repaired": (sum(g["correct_repaired"] for g in graded_all)
                              / len(graded_all)) if graded_all else None,
        "repair_helped": sum(g["repair_helped"] for g in graded_all),
        "repair_harmed": sum(g["repair_harmed"] for g in graded_all),
    }


def _stub_answer_fn():
    """A deterministic stand-in for the answer model: replies with the object of
    the highest-serial fact in the prompt — the benchmark's own "latest wins"
    rule. It exists so the counterfactual plumbing (retrieval, prompt caching,
    class assignment, accounting) can be exercised end to end without an
    endpoint. It measures nothing about a real model.
    """
    import re as _re
    fact_line = _re.compile(r"^\s*(\d+)\.\s+(.*)$", _re.M)

    def answer(prompt: str) -> str:
        body = prompt.split("Facts:\n")[-1].split("\n\nQuestion")[0]
        best, best_obj = -1, ""
        for serial, text in fact_line.findall(body):
            _, obj = fact_key(text)
            if obj and int(serial) > best:
                best, best_obj = int(serial), obj
        return best_obj or "no idea"

    return answer


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--qr-refresh", type=int, default=500)
    ap.add_argument("--max-counterfactuals", type=int, default=300,
                    help="per subset, write side; read side is one per question")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip counterfactuals; base rates only")
    ap.add_argument("--stub-llm", action="store_true",
                    help="SMOKE ONLY: replace the answer model with a deterministic "
                         "stub, to exercise the counterfactual wiring with no "
                         "endpoint. Results are MEANINGLESS.")
    ap.add_argument("--allow-no-calibration", action="store_true",
                    help="SMOKE ONLY: fit thresholds on whatever subsets were run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-embedder", action="store_true")
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(" M3 — headroom")
    if args.smoke_embedder:
        print("   *** SMOKE RUN — HashEmbedder. No result here is interpretable. ***")
        from hnav.core.embedding import HashEmbedder
        emb = HashEmbedder(dim=64)
    else:
        print(f"   model={cfg.embed_model}  dtype={cfg.embed_dtype}")
        emb = build_embedder(cfg)
    print("=" * 74)

    data = json.load(open(DATA, encoding="utf-8"))
    states = {}
    for item in data:
        name = subset_name(item)
        if name.startswith("mh_"):
            continue
        if args.subsets and name not in args.subsets:
            continue
        print(f"\n building {name} ...", flush=True)
        st = build_state(item, name, emb, cfg, args)
        st["_embedder"] = emb
        states[name] = st

    if not states:
        print(" no subsets selected")
        return 1

    # ── pass 1: signals everywhere, thresholds from the CALIBRATION split only
    cal = [n for n in states if n in CALIBRATION]
    if not cal and not args.allow_no_calibration:
        print(f"\n REFUSING to fit thresholds: none of {CALIBRATION} was run.\n"
              " Nothing may be tuned on the confirmatory split. Re-run including a\n"
              " calibration subset, or pass --allow-no-calibration for a smoke run.")
        return 2

    signals_by_subset = {}
    for name, st in states.items():
        print(f"\n signals for {name} ...", flush=True)
        signals_by_subset[name] = {"write": write_pass(st, cfg, args),
                                   "read": read_pass(st, cfg, args)}

    fit_from = cal or list(states)
    thresholds = fit_thresholds(
        [r for n in fit_from for r in signals_by_subset[n]["read"]],
        [r for n in fit_from for r in signals_by_subset[n]["write"]])
    thresholds["fit_subsets"] = fit_from
    thresholds["unfit_for_analysis"] = not cal
    print(f"\n thresholds fit on {fit_from}: nmargin<{thresholds['nmargin']:.4f}  "
          f"H_z>{thresholds['H_z']:.4f}  r_min<{thresholds['r_min']:.4f}")

    # ── pass 2: frozen thresholds applied unchanged everywhere ──────────────
    rng = random.Random(args.seed)
    results = []
    for name, st in states.items():
        print(f"\n── {name}", flush=True)
        w_rows = signals_by_subset[name]["write"]
        r_rows = label_read_rows(signals_by_subset[name]["read"], thresholds)

        w_outcomes: dict = {}
        r_outcomes: list = []
        if not args.no_llm:
            replica = FaissFlatReplica(emb, top_k=args.top_k)
            answer_fn = _stub_answer_fn() if args.stub_llm else local_answer_fn(cfg)
            labeler = CounterfactualLabeler(replica, answer_fn, top_k=args.top_k)
            by_key_questions = defaultdict(list)
            for q in st["questions"]:
                if q["key"]:
                    by_key_questions[tuple(q["key"])].append(q)

            targets = [r for r in w_rows if r["key"]
                       and tuple(r["key"]) in by_key_questions]
            rng.shuffle(targets)
            for row in targets[: args.max_counterfactuals]:
                affected = by_key_questions[tuple(row["key"])]
                out = labeler.label_write(st["fact_store"], row["id"], affected,
                                          is_newest_for_key=row["is_newest_for_key"])
                w_outcomes[row["id"]] = out.to_dict()

            texts = {r.id: r.text for r in st["fact_store"].records}
            chunk_texts = {r.id: r.text for r in st["chunk_store"].records}
            for r in r_rows:
                native_texts = [chunk_texts[c] for c in r["top_ids"]]
                injected = [texts[i] for i in r["missing_superseder_ids"] if i in texts]
                r_outcomes.append(labeler.label_read(
                    {"index": r["index"], "question": r["question"],
                     "truths": r["truths"]}, native_texts, injected).to_dict())
            print(f"   answer calls: {labeler.n_calls}")

        res = {
            "subset": name,
            "split": "calibration" if name in CALIBRATION else
                     ("confirmatory" if name in CONFIRMATORY else "other"),
            "n_facts": len(st["facts"]), "n_chunks": len(st["chunks"]),
            "fallback_chunker": st["fallback_chunker"],
            "thresholds": thresholds,
            "qr_refresh_interval": args.qr_refresh,
            "write": write_headroom(w_rows, w_outcomes, thresholds),
            "read": read_headroom(r_rows, r_outcomes),
            "sim_max": pcts([r["sim_max"] for r in w_rows]),
            "qr_residual": pcts([r["qr_residual"] for r in w_rows]),
            "whole_blob_sim": pcts([r["whole_blob_sim"] for r in w_rows]),
            "diff_sim": pcts([r["diff_sim"] for r in w_rows]),
        }
        if args.smoke_embedder:
            res["SMOKE_HASH_EMBEDDER"] = True
        results.append(res)

        w, rd = res["write"], res["read"]
        print(f"   write: n={w['n_write_decisions']}  conflict={w['conflict_rate']:.3f}"
              f"  critical-delta={w['critical_delta_rate']:.3f}"
              f"  stale={w['stale_supersede_rate']:.3f}"
              f"  dup={w['duplicate_rate']:.3f}"
              f"  would-intervene={w['would_intervene_rate']:.3f}"
              f" (after vetoes {w['would_intervene_after_vetoes_rate']:.3f})")
        print(f"   read : n={rd['n_reads']}  " +
              "  ".join(f"{k}={v['rate']:.2f}" for k, v in sorted(rd["strata"].items())))

        # per-candidate rows and outcomes are M4's input, written out in full
        (cfg.out_dir / f"m3_write_rows_{name}.json").write_text(json.dumps(w_rows))
        (cfg.out_dir / f"m3_read_rows_{name}.json").write_text(json.dumps(r_rows))
        (cfg.out_dir / f"m3_write_outcomes_{name}.json").write_text(
            json.dumps(w_outcomes))
        (cfg.out_dir / f"m3_read_outcomes_{name}.json").write_text(
            json.dumps(r_outcomes))

    out = cfg.out_dir / ("m3_headroom_SMOKE.json" if args.smoke_embedder
                         else "m3_headroom.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n wrote {out}")
    if args.no_llm:
        print(" NOTE: --no-llm — correctness columns are ABSENT, not zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
