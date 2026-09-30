#!/usr/bin/env python3
"""M0 — live-index retriever-replica fidelity.  [closes NEXT_STEPS.md §5]

The one Stage-0 measurement that had no script. ``test_replica_fidelity.py``
compares ``FaissFlatReplica`` against the native *expression* (the two lines from
``qwen3_embedding_memory.py:218``, inlined) over synthetic vectors, and is exact.
It does **not** touch a live LangChain FAISS index, and it does not use real
embeddings. This module does both.

    python hnav/stage0/m0_live_fidelity.py
    python hnav/stage0/m0_live_fidelity.py --subsets sh_6k --max-pairs 50   # smoke

Writes ``hnav/_out/m0_replica_fidelity.json`` with exactly the fields
``report.py::render_m0`` reads: ``arena``, ``n_pairs``, ``top1_agreement``,
``topk_agreement``, ``kendall_tau``, ``max_abs_score_error``, ``tie_rate``.

──────────────────────────────────────────────────────────────────────────────
WHAT IS AND IS NOT BEING MEASURED
──────────────────────────────────────────────────────────────────────────────
This measures **replica fidelity**, not embedder agreement. The replica is fed
the vectors reconstructed out of the live FAISS index itself
(``index.reconstruct_n``) and the same query vector the native search used, so
any disagreement is attributable to the ranking logic alone. Feeding the replica
its own freshly computed embeddings instead would confound two different things
and inflate the error with embedder nondeterminism.

The native side goes through the raw ``vectorstore.index.search`` rather than
``similarity_search_with_score_by_vector`` deliberately: the latter returns
``Document`` objects whose ``page_content`` is ambiguous when two chunks are
byte-identical, whereas the raw index returns **positions**, which are not.
``similarity_search_with_score_by_vector`` is a thin wrapper over exactly this
call, so nothing about the comparison is weakened.

Requires the :8001 embedding server — see ``hnav/deploy/mab.env.template`` for
why (``TextRetriever`` never uses the in-process embedder class, contrary to
``NEXT_STEPS.md`` §6 Correction 2).

Leakage: reads ``context`` and ``questions``; both are permitted in
``hnav/stage0/``. Never reads ``answers``. Nothing here feeds an online decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.core.replica import FaissFlatReplica, kendall_tau, top_k_agreement  # noqa: E402
from hnav.core.types import MemoryRecord, StoreView  # noqa: E402
from hnav.stage0.m2_retrieval_calibration import build_chunks  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
MAB = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"

ARENA = "InEp-Know (MemoryAgentBench, LangChain FAISS IndexFlatL2)"


@contextmanager
def in_mab():
    """Run inside MemoryAgentBench/ with it on sys.path.

    Both are required, for different reasons:
      - ``methods/embedding_retriever.py`` does ``from utils.eval_data_utils
        import format_chat``, so the package root must be importable.
      - ``TextRetriever.__init__`` calls ``dotenv.dotenv_values()``, which reads
        ``./.env`` — the CWD's, not the repo root's. That file is what points
        the embeddings at :8001. Without the chdir they would silently fall back
        to ``os.environ["OPENAI_BASE_URL"]``, which points at the *LLM* on :8000,
        and every embedding call would fail or return garbage.
    """
    prev_cwd = os.getcwd()
    added = False
    try:
        if str(MAB) not in sys.path:
            sys.path.insert(0, str(MAB))
            added = True
        os.chdir(MAB)
        yield
    finally:
        os.chdir(prev_cwd)
        if added and str(MAB) in sys.path:
            sys.path.remove(str(MAB))


def subset_name(item: dict) -> str:
    return item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]


def measure_subset(name: str, context: str, questions: list[str],
                   args) -> dict | None:
    """One subset: build the live index, compare every question against it."""
    chunks, used_fallback = build_chunks(context, args.chunk_size)
    if used_fallback:
        print(f"   !! FALLBACK CHUNKER on {name} — nltk/punkt missing. "
              f"The units being indexed are NOT the benchmark's. Fix before trusting this.",
              flush=True)

    qs = questions[: args.max_pairs] if args.max_pairs else questions
    print(f"\n── {name}: {len(chunks)} chunks, {len(qs)} queries", flush=True)

    with in_mab():
        from methods.embedding_retriever import TextRetriever  # noqa: PLC0415

        retriever = TextRetriever(embedding_model_name=args.embed_model)
        print("   building live FAISS index ...", flush=True)
        retriever.build_vectorstore(chunks)
        vs = retriever.vectorstore
        index = vs.index
        n = index.ntotal

        # The replica is fed the index's OWN vectors — see module docstring.
        bank = index.reconstruct_n(0, n).astype(np.float32)
        norms = np.linalg.norm(bank, axis=1)

        store = StoreView.from_records([
            MemoryRecord(id=f"pos:{i}", text=chunks[i] if i < len(chunks) else "",
                         vector=bank[i], version=i)
            for i in range(n)])
        replica = FaissFlatReplica(embedder=None, top_k=args.top_k)

        k = min(args.top_k, n)
        top1 = topk = 0
        taus: list[float] = []
        max_score_err = 0.0
        tied = 0

        for qi, q in enumerate(qs):
            qv = np.asarray(retriever.embedding_model.embed_query(q), dtype=np.float32)

            # native: raw FAISS. D = squared L2, I = positions, both ascending by D.
            D, I = index.search(qv.reshape(1, -1), n)
            native_ids = [f"pos:{int(i)}" for i in I[0] if int(i) >= 0]
            native_d2 = np.asarray(D[0][: len(native_ids)], dtype=np.float64)

            # replica: same vectors, same query vector, its own ordering logic.
            view = replica.rank(store, q, query_vector=qv, top_k=k)

            if native_ids[:1] == view.ids[:1]:
                top1 += 1
            if top_k_agreement(native_ids, view.ids, k):
                topk += 1
            taus.append(kendall_tau(native_ids, view.ids))

            # cos = 1 - d²/2, exact for unit vectors; replica reports cos × 100.
            expected = (1.0 - native_d2 / 2.0) * 100.0
            pos_of = {rid: j for j, rid in enumerate(view.ids)}
            got = np.array([view.scores[pos_of[rid]] for rid in native_ids])
            max_score_err = max(max_score_err, float(np.max(np.abs(expected - got))))

            tied += int(len(np.unique(np.round(view.scores, 12))) < len(view.scores))

            if (qi + 1) % 25 == 0:
                print(f"   {qi + 1}/{len(qs)}  top-k so far {topk / (qi + 1):.4f}", flush=True)

    n_pairs = len(qs)
    if n_pairs == 0:
        return None

    row = {
        "arena": f"{ARENA} — {name}",
        "n_pairs": n_pairs,
        "n_docs": int(n),
        "top_k": int(k),
        "top1_agreement": top1 / n_pairs,
        "topk_agreement": topk / n_pairs,
        "kendall_tau": float(np.nanmean(taus)) if taus else float("nan"),
        "max_abs_score_error": max_score_err,
        "tie_rate": tied / n_pairs,
        "embed_model": args.embed_model,
        "bank_norm_min": float(norms.min()),
        "bank_norm_max": float(norms.max()),
        "unit_normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "fallback_chunker": used_fallback,
    }
    if not row["unit_normalized"]:
        # cos = 1 - d²/2 assumes unit vectors. Say so rather than quietly
        # reporting a score error that means something else.
        row["WARNING"] = (
            f"bank vectors are NOT unit-normalized (norms {norms.min():.4f}–{norms.max():.4f}). "
            "The d²→cos identity does not hold; max_abs_score_error is not interpretable. "
            "Ranking agreement (top1/topk/tau) is still valid — it only needs a monotone map."
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="default: all single-hop factconsolidation subsets")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap queries per subset (a capped run is a SAMPLED run — "
                         "say so in the report)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()

    print("=" * 74)
    print(" M0 — live-index replica fidelity")
    print(f"   embed_model={args.embed_model}  top_k={args.top_k}")
    print(f"   endpoint   ={os.environ.get('OPENAI_BASE_URL', '(from MemoryAgentBench/.env)')}")
    print("=" * 74)

    data = json.load(open(DATA, encoding="utf-8"))
    rows = []
    for item in data:
        name = subset_name(item)
        if "_sh_" not in name:
            continue
        short = name.replace("factconsolidation_", "")
        if args.subsets and short not in args.subsets and name not in args.subsets:
            continue
        row = measure_subset(short, item["context"], list(item["questions"]), args)
        if row:
            rows.append(row)
            print(f"   {short}: top1={row['top1_agreement']:.4f} "
                  f"topk={row['topk_agreement']:.4f} tau={row['kendall_tau']:.4f} "
                  f"maxΔ={row['max_abs_score_error']:.2e}", flush=True)

    if not rows:
        print("\n No subsets matched. Nothing written.")
        return 1

    out_dir = Path(args.out).parent if args.out else cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / "m0_replica_fidelity.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    total = sum(r["n_pairs"] for r in rows)
    worst = min(r["topk_agreement"] for r in rows)
    print("\n" + "=" * 74)
    print(f" wrote {out}   ({len(rows)} subset(s), {total} pairs)")
    if total < 1000:
        print(f" NOTE: {total} pairs < the protocol's 1,000. Report it as a sampled run.")
    if worst < 0.999:
        print(f" S1 FIRED — worst top-k agreement {worst:.4f} < 0.999.")
        print(" rank_self, margin, dH_self, dH_neighbor and churn are INVALID.")
        print(" Report the maximum achievable fidelity and state which signals die.")
        print("=" * 74)
        return 2
    print(f" top-k agreement ≥ {worst:.4f} on every subset. Replica is faithful.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
