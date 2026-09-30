#!/usr/bin/env python3
"""M2 — retrieval calibration.  [T5]

Over all 400 single-hop primary-arena questions, using the **full pre-truncation
ranking**: score scale and tie frequency; ``top1``/``top2``/``margin``/``nmargin``;
``H_raw`` (logged only), ``H_z``, ``H_vn``; effective neighbourhood size;
dispersion; and ``dH_self``/``dH_neighbor``/``churn@k`` across simulated
provisional inserts.

Reports an explicit **verdict on raw-score entropy degeneracy** at the native
``cosine × 100`` scale. The criterion is stated below and fixed before the run;
a refutation would revise the prior BFCL finding and is reported as readily as a
confirmation.

Leakage: reads ``context`` and ``questions``. The questions are the retrieval
queries — that is what they are at read time — and this is an offline Stage-0
module, where the protocol permits them. It never touches the answer list, and
nothing computed here feeds an online decision. Stratified by subset throughout:
store sizes span 455 → 18,332 facts and retrieval difficulty scales with them,
so pooling would hide the effect the thesis is about.

    python hnav/stage0/m2_retrieval_calibration.py
    python hnav/stage0/m2_retrieval_calibration.py --subsets sh_6k --max-questions 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import explode_facts, fact_key  # noqa: E402
from hnav.core.embedding import build_embedder  # noqa: E402
from hnav.core.replica import FaissFlatReplica  # noqa: E402
from hnav.core.retrieval_signals import RetrievalSignals  # noqa: E402
from hnav.core.types import Candidate, MemoryRecord, StoreView  # noqa: E402
from hnav.labeling.conflict_index import to_probe  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"

# Pre-registered criterion for the raw-entropy verdict.
DEGENERACY_H_RAW_FRAC = 0.10   # median H_raw below this fraction of ln(m) ...
DEGENERACY_RATIO = 5.0         # ... and H_z at least this many times larger


# ── the benchmark's own chunker, transcribed ─────────────────────────────────
def chunk_text_into_sentences(text: str, model_name: str = "gpt-4o-mini",
                              chunk_size: int = 4096) -> list[str]:
    """Transcription of ``utils/eval_other_utils.py:173``.

    Kept as a copy rather than an import so that ``hnav/stage0/`` does not take
    a dependency on the benchmark package's import side effects. The behaviour
    that matters downstream is the ``" ".join(...)``: it replaces the newlines
    between facts with spaces, which is why ``explode_facts`` needs its inline
    fallback.

    Raises if nltk/tiktoken are missing; :func:`build_chunks` catches that and
    records the fallback in the output, so a run made without them is never
    mistaken for the real chunking.
    """
    import nltk
    import tiktoken

    nltk.download("punkt", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    sentences = nltk.sent_tokenize(text)

    chunks, current, count = [], [], 0
    for sentence in sentences:
        n = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if count + n > chunk_size:
            chunks.append(" ".join(current))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _fallback_chunks(text: str, chunk_size: int) -> list[str]:
    """~4 chars/token line grouping. Used only when nltk/tiktoken are missing."""
    budget = chunk_size * 4
    chunks, current, size = [], [], 0
    for line in text.splitlines():
        if size + len(line) > budget and current:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(context: str, chunk_size: int = 4096) -> tuple[list[str], bool]:
    """``(chunks, used_fallback)``."""
    try:
        return chunk_text_into_sentences(context, chunk_size=chunk_size), False
    except Exception:  # noqa: BLE001 — nltk/tiktoken absent, or punkt undownloadable
        return _fallback_chunks(context, chunk_size), True


# ── stats helpers ────────────────────────────────────────────────────────────
def pcts(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "mean": float(x.mean()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


def subset_name(item: dict) -> str:
    return item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
        .replace("factconsolidation_", "")


# ── main ─────────────────────────────────────────────────────────────────────
def run_subset(item, name, emb, cfg, args) -> dict:
    context = item["context"]
    questions = list(item["questions"])          # the retrieval queries
    if args.max_questions:
        questions = questions[: args.max_questions]

    chunks, used_fallback = build_chunks(context, args.chunk_size)
    facts = explode_facts(context)
    print(f"\n── {name}: {len(facts)} facts, {len(chunks)} chunks, "
          f"{len(questions)} questions"
          + ("  [FALLBACK CHUNKER]" if used_fallback else ""), flush=True)

    # ── chunk-level store: the unit the native retriever actually indexes ────
    chunk_vecs = emb.encode(chunks)
    chunk_store = StoreView.from_records([
        MemoryRecord(id=f"chunk:{i}", text=t, vector=chunk_vecs[i], version=i)
        for i, t in enumerate(chunks)])

    replica = FaissFlatReplica(emb, top_k=args.top_k)
    signals = RetrievalSignals(top_m=cfg.top_m, k=cfg.churn_k)

    q_vecs = emb.encode(questions)
    rows = []
    for qi, q in enumerate(questions):
        view = replica.rank(chunk_store, q, query_vector=q_vecs[qi], top_k=args.top_k)
        m = min(cfg.top_m, len(view.ids))
        idx = [int(chunk_store.index_of(i)) for i in view.ids[:m]]
        sig = signals.compute(view, vectors=chunk_vecs[idx])
        rows.append(sig.to_dict())

    # ── fact-level provisional inserts (write-side effect) ───────────────────
    groups: dict[tuple, list] = defaultdict(list)
    for serial, text in facts:
        key, obj = fact_key(text)
        if key:
            groups[key].append((serial, text, obj))
    conflicted = [sorted(v) for v in groups.values() if len({o for _, _, o in v}) > 1]
    rng = random.Random(args.seed)
    rng.shuffle(conflicted)
    sample = conflicted[: args.max_inserts]

    fact_texts = [t for _, t in facts]
    fact_vecs = emb.encode(fact_texts)
    fact_store = StoreView.from_records([
        MemoryRecord(id=f"fact:{serial}", text=t, vector=fact_vecs[i], version=serial)
        for i, (serial, t) in enumerate(facts)])
    fact_pos = {f"fact:{serial}": i for i, (serial, _) in enumerate(facts)}

    deltas = []
    for group in sample:
        (old_serial, old_text, _), (new_serial, new_text, _) = group[0], group[-1]
        key, _ = fact_key(new_text)
        probe = to_probe(key[0], key[1]) if key else new_text[:200]

        # S_without: the store as it was before the superseding fact arrived.
        pre = StoreView.from_records(
            [r for r in fact_store.records if r.version < new_serial])
        if len(pre) < 2:
            continue
        cand = Candidate(id=f"fact:{new_serial}", text=new_text,
                         vector=fact_vecs[fact_pos[f"fact:{new_serial}"]],
                         version=new_serial,
                         metadata={"key": key, "prior_id": f"fact:{old_serial}"})
        eff = replica.simulate_insert(pre, cand, [probe])
        d = signals.compute_delta(eff["before"], eff["after"],
                                  self_id=cand.id, k=cfg.churn_k)
        deltas.append(d.to_dict())

    # ── raw-entropy degeneracy verdict ───────────────────────────────────────
    m_used = int(np.median([r["top_m"] for r in rows])) if rows else 0
    ceiling = float(np.log(m_used)) if m_used > 1 else float("nan")
    med_raw = float(np.median([r["H_raw"] for r in rows])) if rows else float("nan")
    med_z = float(np.median([r["H_z"] for r in rows])) if rows else float("nan")
    degenerate = bool(np.isfinite(ceiling)
                      and med_raw < DEGENERACY_H_RAW_FRAC * ceiling
                      and med_z >= DEGENERACY_RATIO * max(med_raw, 1e-12))

    res = {
        "subset": name,
        "model": cfg.embed_model, "dtype": cfg.embed_dtype,
        "n_facts": len(facts), "n_chunks": len(chunks), "n_questions": len(questions),
        "top_k": args.top_k, "top_m": cfg.top_m, "churn_k": cfg.churn_k,
        "fallback_chunker": used_fallback,
        "score_scale": pcts([r["top1"] for r in rows]),
        "top2": pcts([r["top2"] for r in rows]),
        "margin": pcts([r["margin"] for r in rows]),
        "nmargin": pcts([r["nmargin"] for r in rows]),
        "H_raw": pcts([r["H_raw"] for r in rows]),
        "H_z": pcts([r["H_z"] for r in rows]),
        "H_vn": pcts([r["H_vn"] for r in rows]),
        "eff_size": pcts([r["eff_size"] for r in rows]),
        "dispersion": pcts([r["dispersion"] for r in rows]),
        "tie_rate_top_m": float(np.mean([r["n_ties_top_m"] > 0 for r in rows]))
        if rows else float("nan"),
        "mean_ties_top_m": float(np.mean([r["n_ties_top_m"] for r in rows]))
        if rows else float("nan"),
        "n_conflicted_keys": len(conflicted),
        "n_simulated_inserts": len(deltas),
        "dH_self": pcts([d["dH_self"] for d in deltas]),
        "churn_at_k": pcts([d["churn_at_k"] for d in deltas]),
        "churn_frac": pcts([d["churn_frac"] for d in deltas]),
        "rank_shift": pcts([d["rank_shift"] for d in deltas]),
        "rank_self_after": pcts([d["rank_self_after"] for d in deltas]),
        "entropy_ceiling_ln_m": ceiling,
        "median_H_raw": med_raw, "median_H_z": med_z,
        "raw_entropy_verdict": "DEGENERATE" if degenerate else "NOT_DEGENERATE",
    }

    print(f"   top1      : p50={res['score_scale'].get('p50', float('nan')):.2f}  "
          f"margin p50={res['margin'].get('p50', float('nan')):.3f}  "
          f"nmargin p50={res['nmargin'].get('p50', float('nan')):.4f}")
    print(f"   entropy   : H_raw p50={med_raw:.4f}  H_z p50={med_z:.4f}  "
          f"H_vn p50={res['H_vn'].get('p50', float('nan')):.4f}  "
          f"ceiling ln(m)={ceiling:.3f}")
    print(f"   verdict   : raw-score entropy is {res['raw_entropy_verdict']}")
    print(f"   inserts   : n={len(deltas)}  dH_self p50="
          f"{res['dH_self'].get('p50', float('nan')):.4f}  churn@k p50="
          f"{res['churn_at_k'].get('p50', float('nan')):.1f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--include-mh", action="store_true",
                    help="also run multi-hop subsets (exploratory only)")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--max-inserts", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-embedder", action="store_true",
                    help="run the pipeline against HashEmbedder instead of the real "
                         "model: exercises every code path with no GPU. The numbers "
                         "are MEANINGLESS and are written to a separate _SMOKE file.")
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(" M2 — retrieval calibration")
    if args.smoke_embedder:
        print("   *** SMOKE RUN — HashEmbedder. No result here is interpretable. ***")
    else:
        print(f"   model={cfg.embed_model}  device=cuda:{cfg.embed_device}  "
              f"dtype={cfg.embed_dtype}  top_m={cfg.top_m}  k={cfg.churn_k}")
    print("=" * 74)

    data = json.load(open(DATA, encoding="utf-8"))
    if args.smoke_embedder:
        from hnav.core.embedding import HashEmbedder
        emb = HashEmbedder(dim=64)
    else:
        emb = build_embedder(cfg)

    results = []
    for item in data:
        name = subset_name(item)
        if not args.include_mh and name.startswith("mh_"):
            continue
        if args.subsets and name not in args.subsets:
            continue
        res = run_subset(item, name, emb, cfg, args)
        if args.smoke_embedder:
            res["SMOKE_HASH_EMBEDDER"] = True
        results.append(res)

    out = cfg.out_dir / ("m2_retrieval_calibration_SMOKE.json" if args.smoke_embedder
                         else "m2_retrieval_calibration.json")
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print(" M2 SUMMARY  (stratified — never pooled across subsets)")
    print("=" * 74)
    print(f" {'subset':<9} {'qs':>4} {'top1 p50':>9} {'margin':>8} {'H_raw':>7} "
          f"{'H_z':>6} {'H_vn':>6} {'eff':>6}  raw-entropy")
    for r in results:
        print(f" {r['subset']:<9} {r['n_questions']:>4} "
              f"{r['score_scale'].get('p50', float('nan')):>9.2f} "
              f"{r['margin'].get('p50', float('nan')):>8.3f} "
              f"{r['median_H_raw']:>7.4f} {r['median_H_z']:>6.3f} "
              f"{r['H_vn'].get('p50', float('nan')):>6.3f} "
              f"{r['eff_size'].get('p50', float('nan')):>6.2f}  "
              f"{r['raw_entropy_verdict']}")
    print(f"\n wrote {out}")
    print("\n Criterion (pre-registered): raw-score entropy is DEGENERATE iff")
    print(f"   median H_raw < {DEGENERACY_H_RAW_FRAC} x ln(m)  AND  "
          f"median H_z >= {DEGENERACY_RATIO} x median H_raw.")
    print(" H_raw is reported for this verdict only. It never feeds a decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
