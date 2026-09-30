#!/usr/bin/env python3
"""Faz B calibration — coverage-balanced operating point for the read gate.  [T11]

CALIBRATION SPLIT ONLY (``sh_6k`` + ``sh_32k``). The script refuses every other
subset: sh_64k is the confirmatory arena and touching it here would void the
campaign (hard invariant; STAGE1_PLAN.md §0/§3).

Two phases, split so that the GPU (NLI) and the chat server (grading) never
have to coexist on one card:

``--prepass``   Embeddings (cache-first), the benchmark-faithful chunk ranking
                per question, retrieval signals, candidate pools, pair
                geometry (raw AND ABTT-whitened, logged side by side — V1 A/B
                evidence, no behaviour change), bidirectional NLI scores for
                every pair inside a loose-screen component, and pair ground
                truth from the validated parser. Writes
                ``hnav/_out/stage1_prepass_<subset>.json``.

``--evaluate``  Replays the REAL ``ReadGate`` + ``rerank_order`` (not a
                reimplementation) over the grid, using a replay NLI client fed
                from the prepass. Per cell: coverage, verification precision,
                the supervisor-Note-1 false-verified rate (with and without
                the key screen), and — unless ``--no-llm`` — net help/harm
                from grading the native vs reranked chunk order with the
                benchmark's own prompt shape against the Stage-1 chat server.
                Writes ``hnav/_out/stage1_calibration.json`` and, with
                ``--freeze``, the committed operating-point artifact.

Query faithfulness. The campaign's retriever is NOT queried with the bare
question: ``RAGSystem.answer_query`` extracts everything after
"Now Answer the Question:" from the templated message and retrieves on that.
This harness reproduces the template and the extraction verbatim, so rankings,
signals and realized ambiguity-coverage here are the campaign's. (M2/M3 used
bare questions; the frozen nmargin/H_z CONSTANTS keep their Stage-0 provenance
and are applied unchanged — what this harness measures is how often they fire
under campaign queries.)

Pre-registered objective (frozen before any grading was looked at):
  feasible(cell) :=  pooled harmed ≤ 2% of graded questions
                 AND per-subset harmed ≤ helped
                 AND false_verified_rate ≤ 0.05 (when any pair verified)
  score(cell)    :=  pooled (helped − harmed)
  choose argmax score among feasible; ties → fewer harmed, then higher
  cos_pair, then higher nli_contradiction, then fewer touched questions,
  then ambiguity_mode in the order all > any > none. If no feasible cell has
  score > 0, REPORT AND STOP — that is a result, not a tuning failure.

Smoke flags (``--smoke-embedder``, ``--stub-nli``, ``--stub-llm``) write
``*_SMOKE`` files; nothing from a smoke run may feed a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import (MABAdapter, explode_facts, fact_key,  # noqa: E402
                                       select_pool)
from hnav.core import read_gate as _rg  # noqa: E402
from hnav.core.geometry import ABTTWhitening, qr_residual  # noqa: E402
from hnav.core.read_gate import (GateThresholds, NLIScores, ReadGate,  # noqa: E402
                                 StubNLI, build_nli)
from hnav.core.read_policy import rerank_order  # noqa: E402
from hnav.core.replica import FaissFlatReplica  # noqa: E402
from hnav.core.retrieval_signals import RetrievalSignals  # noqa: E402
from hnav.core.types import MemoryRecord, StoreView  # noqa: E402
from hnav.labeling.counterfactual import substring_exact_match  # noqa: E402
from hnav.stage0.m2_retrieval_calibration import build_chunks, subset_name  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
CALIBRATION = ("sh_6k", "sh_32k")

# ── the benchmark's own prompt shape, verbatim ───────────────────────────────
# utils/templates.py: SYSTEM_MESSAGE + the Conflict_Resolution 'rag_agent'
# query template; methods/embedding_retriever.py RAGSystem.answer_query for the
# retrieval-query extraction and the "Memory i:" join. Copied, not imported —
# importing the benchmark from hnav would break the layering tests.
SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
QUERY_TEMPLATE = ("Pretend you are a knowledge management system. Each fact in the knowledge pool "
                  "is provided with a serial number at the beginning, and the newer fact has larger "
                  "serial number. \n You need to solve the conflicts of facts in the knowledge pool "
                  "by finding the newest fact with larger serial number. You need to answer a "
                  "question based on this rule. You should give a very concise answer without "
                  "saying other words for the question **only** from the knowledge pool you have "
                  "memorized rather than the real facts in real world. \n\nFor example:\n\n "
                  "[Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is "
                  "the name of the current president of Russia? \nAnswer: Donald Trump \n\n "
                  "Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:")
RETRIEVAL_EXTRACT = re.compile(r"Now Answer the Question:\s*(.*)", re.DOTALL)
GENERATION_MAX_TOKENS = 10   # configs/data_conf/Conflict_Resolution/*.yaml generation_max_length

# ── grid axes (declared, not searched) ───────────────────────────────────────
COS_GRID = (0.90, 0.92, 0.94)          # M1b best-F1 tau family on/around calibration
R_MIN_GRID_LABELS = ("frozen", "loose", "off")
NLI_GRID = (0.5, 0.9, 0.99)
AMB_GRID = ("all", "any", "none")
FILTER_GRID = (True, False)
COS_LOOSE = min(COS_GRID)
R_LOOSE = 0.44                          # sqrt(1-0.90^2)=0.436: pass-through for pairs
                                        # the loosest cos screen admits


def r_min_of(label: str) -> float | None:
    return {"frozen": _rg.R_MIN_CAL, "loose": R_LOOSE, "off": None}[label]


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ── embedder choices ─────────────────────────────────────────────────────────
class _FailOnMiss:
    """Inner embedder for --cache-only: any miss is a hard, named error."""

    dim = 0

    def encode(self, texts):
        raise RuntimeError(
            f"--cache-only but {len(texts)} text(s) missing from the embedding "
            f"cache, first={texts[0][:80]!r}. Run --prepass with the real "
            f"embedder (GPU) first, or drop --cache-only.")


def make_embedder(args, cfg):
    from hnav.core.embedding import DiskCachedEmbedder, cache_key
    if args.smoke_embedder:
        from hnav.core.embedding import HashEmbedder
        return HashEmbedder(dim=64)
    key = cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length)
    if args.cache_only:
        return DiskCachedEmbedder(_FailOnMiss(), cfg.emb_cache_dir, key)
    from hnav.core.embedding import build_embedder
    return build_embedder(cfg)


# ── replay NLI ───────────────────────────────────────────────────────────────
class ReplayNLI:
    """Serves prepass NLI scores back to the real gate. Direction-aware;
    unknown pairs are a hard error — silence would mean the prepass superset
    was not a superset, which invalidates the whole grid."""

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self.table = table

    def score_pairs(self, pairs):
        out = []
        for p, h in pairs:
            k = f"{_sha(p)}|{_sha(h)}"
            if k not in self.table:
                raise KeyError(f"NLI score missing for a directed pair: "
                               f"{p[:60]!r} -> {h[:60]!r}. The prepass loose "
                               f"screen did not cover this grid cell.")
            e, n, c = self.table[k]
            out.append(NLIScores(entailment=e, neutral=n, contradiction=c))
        return out


class _CachedQR:
    """Per-question leave-one-out residual cache keyed by the member row's
    bytes. Installed as ``read_gate.qr_residual`` during --evaluate ONLY (and
    restored), so 162 grid cells do not redo an identical QR 162 times. Falls
    back to the real computation on a miss — behaviour, not speed, is the
    contract, and ``test_stage1_calibration.py`` pins the equivalence."""

    def __init__(self) -> None:
        self.table: dict[bytes, float] = {}

    def load(self, mat: np.ndarray) -> None:
        self.table.clear()
        for i in range(mat.shape[0]):
            r, _ = qr_residual(mat[i], np.delete(mat, i, axis=0))
            self.table[np.ascontiguousarray(mat[i]).tobytes()] = float(r[0])

    def __call__(self, vectors, bank, max_basis: int = 512, seed: int = 0):
        v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if v.shape[0] == 1:
            hit = self.table.get(np.ascontiguousarray(v[0]).tobytes())
            if hit is not None:
                return np.array([hit]), False
        return qr_residual(vectors, bank, max_basis=max_basis, seed=seed)


# ── subset state ─────────────────────────────────────────────────────────────
def build_state(item, name, emb, cfg, args):
    """Facts, chunks and per-question queries for one subset.

    ``args.page_source == "benchmark"`` skips CHUNK embedding entirely (T13,
    pre-registration v2 Amendment 3): the page is read from the benchmark's own
    vectors, so a chunk ranking computed here would be both unused and
    misleading. It also avoids embedding 5,243-token chunks in fp32, which does
    not fit on a 24 GB card next to 17 GB of fp32 weights.
    """
    context = item["context"]
    page_source = getattr(args, "page_source", "prepass")
    facts = explode_facts(context)
    chunks, fallback = build_chunks(context, args.chunk_size)
    fact_vecs = emb.encode([t for _, t in facts])
    chunk_vecs = None if page_source == "benchmark" else emb.encode(chunks)

    # [ABTT] Whiten the fact vectors ONCE, here, so that every downstream
    # consumer -- select_pool, the loose screen, sims, the leave-one-out span
    # residual, ReadGate.decide -- operates in the corrected space without
    # knowing it. ABTT renormalizes, so ``mat @ mat.T`` is still a cosine
    # matrix and the gate needs no change.
    #
    # Chunk vectors are deliberately NOT whitened: the chunk ranking exists to
    # reproduce what the benchmark's own index retrieves, and transforming
    # H-Nav's copy would make it reproduce something else. On the confirmatory
    # path ``page_source="benchmark"`` skips chunk embedding entirely.
    whiten_space = getattr(args, "_whitener", None)
    if whiten_space is not None and getattr(args, "whiten_scope", "pairs") == "all":
        fact_vecs = whiten_space.transform(np.asarray(fact_vecs, dtype=np.float64))

    records = []
    for i, (s, t) in enumerate(facts):
        key, obj = fact_key(t)
        records.append(MemoryRecord(id=f"fact:{s}", text=t, vector=fact_vecs[i],
                                    version=s, metadata={"key": key, "object": obj}))
    chunk_store = None if chunk_vecs is None else StoreView.from_records([
        MemoryRecord(id=f"chunk:{i}", text=t, vector=chunk_vecs[i], version=i)
        for i, t in enumerate(chunks)])

    fact_chunk: dict[str, str] = {}
    fact_of_chunk: dict[str, list[str]] = {}
    for i, c in enumerate(chunks):
        members = [f"fact:{s}" for s, _ in explode_facts(c)]
        fact_of_chunk[f"chunk:{i}"] = members
        for f in members:
            fact_chunk[f] = f"chunk:{i}"

    by_id = {r.id: r for r in records}
    qs = []
    for qi, question in enumerate(item.get("questions", [])):
        if args.max_questions and qi >= args.max_questions:
            break
        message = QUERY_TEMPLATE.format(question=question)
        rq = RETRIEVAL_EXTRACT.search(message).group(1)   # RAGSystem, verbatim
        qs.append({"index": qi, "question": question, "message": message,
                   "retrieval_query": rq,
                   "truths": item.get("answers", [])[qi]})
    return {"name": name, "facts": facts, "records": records, "by_id": by_id,
            "chunks": chunks, "chunk_store": chunk_store,
            "fallback_chunker": fallback, "fact_chunk": fact_chunk,
            "fact_of_chunk": fact_of_chunk, "qs": qs,
            "page_source": page_source}


# ── prepass ──────────────────────────────────────────────────────────────────
def loose_components(pool, sims, cos_loose: float = COS_LOOSE) -> list[list[int]]:
    """Connected components of the loose cosine screen.

    ``cos_loose`` is a parameter because it is a *coordinate in the embedding
    space*, not a constant of the method. Whitening moves the whole cosine
    distribution down, so reusing the raw 0.90 in whitened space would admit
    almost nothing and the prepass superset would no longer contain the pairs
    the grid can select. [ABTT]
    """
    n = len(pool)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if sims[i, j] >= cos_loose]
    return _rg._components(n, edges)


# ── ABTT geometry space  [ABTT Phase 3] ──────────────────────────────────────
def load_whitening(path):
    """Load the offline-fitted whitening artifact, or None for the raw space.

    Fitted offline on the calibration split by
    ``hnav/stage0/fit_abtt_artifact.py`` so that nothing has to be estimated at
    decision time, where the 50-fact pool is below ``min_fit_n``.
    ``ABTTWhitening.from_dict`` re-checks the fingerprint, so a corrupt or
    mismatched artifact raises here rather than silently scoring in a slightly
    different space.
    """
    if path is None:
        return None, None
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    w = ABTTWhitening.from_dict(art["whitening"])
    if not w.fitted:
        raise SystemExit(f" REFUSED: {path} holds an unfitted whitener.")
    return w, art


def benchmark_top_ids(subset: str, path=None):
    """The benchmark's OWN top-k page per question, as ``chunk:i`` ids.

    Read from ``stage0_results/stage1/chunk_embedding_refit.json``. Measured
    there: re-encoding cannot reproduce this page — matching the benchmark's
    dtype and truncation gets our chunk vectors to min cosine 0.99997 and the
    sh_64k page still agrees on only 26/100 questions, because 17
    tightly-clustered chunks reshuffle at the top-10/11 boundary. So it is read,
    never recomputed (pre-registration v2 Amendment 3).
    """
    from pathlib import Path as _P
    path = _P(path or REPO / "stage0_results/stage1/chunk_embedding_refit.json")
    if not path.exists():
        raise SystemExit(f" REFUSED: {path} missing; run "
                         "hnav/deploy/refit_chunk_embeddings.py first.")
    row = json.loads(path.read_text(encoding="utf-8")).get("subsets", {}).get(subset)
    if not row or "benchmark" not in row.get("pages", {}):
        raise SystemExit(f" REFUSED: no benchmark page recorded for {subset}. "
                         "Do NOT fall back to a ranking computed here.")
    return [[f"chunk:{int(i)}" for i in pg] for pg in row["pages"]["benchmark"]]


def prepass_subset(st, emb, nli, cfg, args) -> dict:
    page_source = st.get("page_source", "prepass")
    bench_pages = benchmark_top_ids(st["name"]) if page_source == "benchmark" else None
    replica = (FaissFlatReplica(emb, top_k=args.top_k)
               if st["chunk_store"] is not None else None)
    signals = RetrievalSignals(top_m=cfg.top_m, k=cfg.churn_k)

    # [ABTT] When the vectors are ALREADY whitened by the offline artifact, the
    # per-subset A/B whitener must not run: fitting it here would whiten twice
    # and ``cos_w`` would describe a space nothing is scored in. Left unfitted
    # it is a documented pass-through, so ``cos_w == cos`` and the stamped
    # ``geometry_space`` tells a reader which space that is.
    space_whitener = getattr(args, "_whitener", None)
    scope = getattr(args, "whiten_scope", "pairs")
    cos_loose = float(getattr(args, "cos_loose", None) or COS_LOOSE)
    whiten = ABTTWhitening(cfg.whiten_components, cfg.whiten_min_fit_n)
    fact_matrix = np.stack([np.asarray(r.vector, dtype=np.float64)
                            for r in st["records"]])
    if space_whitener is None:
        whiten.fit(fact_matrix)

    out_qs = []
    nli_table: dict[str, tuple[float, float, float]] = {}
    directed_todo: list[tuple[str, str]] = []
    directed_keys: list[str] = []
    directed_seen: set[str] = set()

    for q in st["qs"]:
        if bench_pages is not None:
            # The page is the BENCHMARK's. nmargin/H_z would be derived from a
            # chunk ranking that does not exist here; they are inert at the
            # frozen operating point (ambiguity_mode="none") and are recorded as
            # null rather than fabricated.
            top_ids = list(bench_pages[q["index"]])
            qv = emb.encode([q["retrieval_query"]])[0]
            # [ABTT] select_pool ranks facts by ``fact @ qv``; the facts are
            # whitened in build_state, so an unwhitened query would score two
            # different spaces against each other.
            if space_whitener is not None and scope == "all":
                qv = space_whitener.transform(np.asarray(qv, dtype=np.float64))
            nmargin = H_z = None
        else:
            view = replica.rank(st["chunk_store"], q["retrieval_query"],
                                top_k=args.top_k)
            sig = signals.compute(view)
            top_ids, qv = list(view.top_ids), view.query_vector
            if space_whitener is not None and scope == "all":   # [ABTT] see above
                qv = space_whitener.transform(np.asarray(qv, dtype=np.float64))
            nmargin, H_z = float(sig.nmargin), float(sig.H_z)
        # Adapter-faithful page-fact order: MABAdapter.facts_in iterates chunks
        # in ADMISSION (index) order filtered by the retrieved set — not in
        # rank order — so the harness does the same.
        wanted = set(top_ids)
        page_facts = [st["by_id"][f] for i in range(len(st["chunks"]))
                      if f"chunk:{i}" in wanted
                      for f in st["fact_of_chunk"][f"chunk:{i}"]]
        pool = select_pool(page_facts, qv, cfg.top_m)

        mat = np.stack([np.asarray(r.vector, dtype=np.float64) for r in pool]) \
            if pool else np.zeros((0, 1))
        # [ABTT] Under scope="pairs" the pool was selected in the RAW space
        # (retrieval faithfulness) and only the fact-fact geometry the gate
        # decides on is whitened here. Under scope="all" `mat` is already
        # whitened in build_state, so it must not be transformed twice.
        if space_whitener is not None and scope == "pairs" and len(pool):
            mat = space_whitener.transform(mat)
        sims = mat @ mat.T
        mat_w = whiten.transform(mat) if whiten.fitted and len(pool) else mat
        sims_w = mat_w @ mat_w.T

        comps = loose_components(pool, sims, cos_loose) if len(pool) >= 2 else []
        members = sorted({i for g in comps for i in g})
        loo, loo_w = {}, {}
        for i in members:
            r, _ = qr_residual(mat[i], np.delete(mat, i, axis=0))
            loo[i] = float(r[0])
            rw, _ = qr_residual(mat_w[i], np.delete(mat_w, i, axis=0))
            loo_w[i] = float(rw[0])

        pairs = []
        for g in comps:
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    i, j = g[a], g[b]
                    ra, rb = pool[i], pool[j]
                    ka, kb = ra.metadata.get("key"), rb.metadata.get("key")
                    pairs.append({
                        "a": ra.id, "b": rb.id,
                        "cos": float(sims[i, j]), "cos_w": float(sims_w[i, j]),
                        "r_a": loo[i], "r_b": loo[j],
                        "r_a_w": loo_w[i], "r_b_w": loo_w[j],
                        "same_key": bool(ka is not None and ka == kb),
                        "same_object": bool(ra.metadata.get("object") is not None
                                            and ra.metadata.get("object")
                                            == rb.metadata.get("object")),
                    })
                    for p, h in ((ra.text, rb.text), (rb.text, ra.text)):
                        k = f"{_sha(p)}|{_sha(h)}"
                        if k not in directed_seen:
                            directed_seen.add(k)
                            directed_todo.append((p, h))
                            directed_keys.append(k)

        out_qs.append({"index": q["index"], "truths": q["truths"],
                       "retrieval_query": q["retrieval_query"],
                       "top_ids": top_ids,
                       "nmargin": nmargin, "H_z": H_z,
                       "pool": [r.id for r in pool],
                       "pairs": pairs})

    # one batched NLI sweep over the union of directed pairs
    if directed_todo:
        scores = nli.score_pairs(directed_todo)
        for k, s in zip(directed_keys, scores):
            nli_table[k] = (s.entailment, s.neutral, s.contradiction)

    # ABTT A/B summary: does whitening separate true supersession pairs better?
    flat = [p for qq in out_qs for p in qq["pairs"]]
    truth = [p["same_key"] and not p["same_object"] for p in flat]
    ab = {
        "n_pairs": len(flat),
        "n_true_supersession": int(sum(truth)),
        "auc_cos_raw": _auc([p["cos"] for p in flat], truth),
        "auc_cos_whitened": _auc([p["cos_w"] for p in flat], truth),
        "auc_rmin_raw": _auc([-min(p["r_a"], p["r_b"]) for p in flat], truth),
        "auc_rmin_whitened": _auc([-min(p["r_a_w"], p["r_b_w"]) for p in flat], truth),
        "whitening_fitted": bool(whiten.fitted),
        "note": "logged only - no decision consumes whitened values (V1 A/B evidence). "
                "Pair support = raw loose screen.",
    }

    return {"subset": st["name"], "n_facts": len(st["facts"]),
            "geometry_space": "abtt" if space_whitener is not None else "raw",
            "whiten_scope": scope if space_whitener is not None else None,
            "whitening_fingerprint": (space_whitener.fingerprint()
                                      if space_whitener is not None else None),
            "cos_loose_used": cos_loose,
            "n_chunks": len(st["chunks"]),
            "fallback_chunker": st["fallback_chunker"],
            # Stamped so no future reader can conflate the two page sources.
            "page_source": page_source,
            "signals_available": page_source != "benchmark",
            "top_k": args.top_k, "pool_cap": cfg.top_m,
            "cos_loose": COS_LOOSE,
            "queries": "templated (RAGSystem extraction, campaign-faithful)",
            # The prepass PERSISTS NLI scores (nli_table) and --evaluate
            # replays them, so the table is a cache — and its key is
            # sha1(premise)|sha1(hypothesis) with no engine parameters in it.
            # Stamping the engine config here is the analogue of putting
            # max_length in the embedding cache namespace (T12): it makes a
            # mismatched replay detectable instead of silent.
            "nli_config": {"model": cfg.nli_model,
                           "max_length": cfg.nli_max_length,
                           "stub": bool(getattr(args, "stub_nli", False))},
            "questions": out_qs, "nli_table": nli_table,
            "fact_chunk": st["fact_chunk"], "abtt_ab": ab}


def _auc(scores, labels) -> float | None:
    """Mann-Whitney AUC, closed form; None when one class is empty."""
    pos = [s for s, t in zip(scores, labels) if t]
    neg = [s for s, t in zip(scores, labels) if not t]
    if not pos or not neg:
        return None
    allv = sorted((s, i) for i, s in enumerate(pos + neg))
    ranks = {}
    k = 0
    while k < len(allv):
        j = k
        while j + 1 < len(allv) and allv[j + 1][0] == allv[k][0]:
            j += 1
        mean_rank = (k + j) / 2 + 1
        for m in range(k, j + 1):
            ranks[allv[m][1]] = mean_rank
        k = j + 1
    rsum = sum(ranks[i] for i in range(len(pos)))
    return float((rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ── evaluate ─────────────────────────────────────────────────────────────────
def grid_cells():
    for cos in COS_GRID:
        for rlab in R_MIN_GRID_LABELS:
            for amb in AMB_GRID:
                for tau in NLI_GRID:
                    for filt in FILTER_GRID:
                        yield {"cos_pair": cos, "r_min_label": rlab,
                               "r_min": r_min_of(rlab), "ambiguity_mode": amb,
                               "nli_contradiction": tau, "pair_filter": filt}


def make_answer_fn(args, cfg):
    if args.stub_llm:
        fact_line = re.compile(r"(?:(?<=\s)|(?<=\A))(\d+)\.\s+(.+?)(?=\s+\d+\.\s|\Z)", re.S)

        def stub(prompt: str) -> str:
            best, best_obj = -1, "no idea"
            for m in fact_line.finditer(prompt):
                _, obj = fact_key(m.group(2).strip())
                if obj and int(m.group(1)) > best:
                    best, best_obj = int(m.group(1)), obj
            return best_obj
        return stub

    from openai import OpenAI  # noqa: PLC0415
    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def answer(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                      {"role": "user", "content": prompt}],
            temperature=cfg.llm_temperature,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    return answer


def build_user_prompt(chunk_texts, message) -> str:
    """RAGSystem.answer_query, verbatim: Memory-numbered chunks + the message."""
    memory = "\n".join(f"Memory {i+1}:\n{t}" for i, t in enumerate(chunk_texts))
    return memory + "\n" + message


def evaluate(prepasses, states, cfg, args) -> dict:
    cells = [dict(c) for c in grid_cells()]
    for c in cells:
        c.update({"n_verified": 0, "n_true_supersession": 0, "n_fv_diff_key": 0,
                  "n_fv_same_object": 0, "n_changed": 0, "n_ambiguous": 0,
                  "helped": 0, "harmed": 0,
                  "by_subset": defaultdict(lambda: {"helped": 0, "harmed": 0,
                                                    "changed": 0})})

    prompt_cache: dict[str, str] = {}
    grade_cache: dict[tuple, bool] = {}
    answer_fn = make_answer_fn(args, cfg) if not args.no_llm else None
    n_llm_calls = [0]

    def graded(subset, q, order_texts, message, truths) -> bool:
        key = (subset, q, tuple(_sha(t) for t in order_texts))
        if key not in grade_cache:
            prompt = build_user_prompt(order_texts, message)
            if prompt not in prompt_cache:
                prompt_cache[prompt] = answer_fn(prompt)
                n_llm_calls[0] += 1
            grade_cache[key] = substring_exact_match(prompt_cache[prompt], truths)
        return grade_cache[key]

    qr_cache = _CachedQR()
    real_qr = _rg.qr_residual
    _rg.qr_residual = qr_cache
    n_questions = 0
    try:
        for pp in prepasses:
            st = states[pp["subset"]]
            chunk_text = {f"chunk:{i}": t for i, t in enumerate(st["chunks"])}
            replay = ReplayNLI(pp["nli_table"])
            for q in pp["questions"]:
                n_questions += 1
                pool = [st["by_id"][fid] for fid in q["pool"]]
                if len(pool) >= 2:
                    mat = np.stack([np.asarray(r.vector, dtype=np.float64)
                                    for r in pool])
                    qr_cache.load(mat)
                sigview = {"nmargin": q["nmargin"], "H_z": q["H_z"]}
                native = [chunk_text[c] for c in q["top_ids"]]
                orders_to_grade = {}
                for c in cells:
                    thr = GateThresholds(
                        cos_pair=c["cos_pair"], r_min=c["r_min"],
                        ambiguity_mode=c["ambiguity_mode"],
                        nli_contradiction=c["nli_contradiction"])
                    gate = ReadGate(replay, thr,
                                    pair_filter=MABAdapter.same_key_pair
                                    if c["pair_filter"] else None)
                    dec = gate.decide(pool, sigview,
                                      latest_key=MABAdapter.latest_key)
                    if dec.ambiguous:
                        c["n_ambiguous"] += 1
                    for pc in dec.pair_checks:
                        if not pc.verified:
                            continue
                        c["n_verified"] += 1
                        ma = st["by_id"][pc.id_a].metadata
                        mb = st["by_id"][pc.id_b].metadata
                        if ma["key"] is not None and ma["key"] == mb["key"]:
                            if ma["object"] == mb["object"]:
                                c["n_fv_same_object"] += 1
                            else:
                                c["n_true_supersession"] += 1
                        else:
                            c["n_fv_diff_key"] += 1
                    order = rerank_order(q["top_ids"], dec,
                                         id_map=pp["fact_chunk"].get)
                    if order != q["top_ids"]:
                        c["n_changed"] += 1
                        c["by_subset"][pp["subset"]]["changed"] += 1
                        orders_to_grade[id(c)] = (c, order)
                    else:
                        orders_to_grade[id(c)] = (c, None)

                if answer_fn is None:
                    continue
                distinct = {tuple(o) for _, o in orders_to_grade.values()
                            if o is not None}
                if not distinct:
                    continue
                message = st["qs"][q["index"]]["message"]
                nat_ok = graded(pp["subset"], q["index"], native, message,
                                q["truths"])
                order_ok = {}
                for o in distinct:
                    texts = [chunk_text[c_] for c_ in o]
                    order_ok[o] = graded(pp["subset"], q["index"], texts,
                                         message, q["truths"])
                for c, o in orders_to_grade.values():
                    if o is None:
                        continue
                    ok = order_ok[tuple(o)]
                    if ok and not nat_ok:
                        c["helped"] += 1
                        c["by_subset"][pp["subset"]]["helped"] += 1
                    elif nat_ok and not ok:
                        c["harmed"] += 1
                        c["by_subset"][pp["subset"]]["harmed"] += 1
    finally:
        _rg.qr_residual = real_qr

    # feasibility + choice (the pre-registered objective in the docstring)
    graded_n = n_questions if answer_fn is not None else 0
    for c in cells:
        fv = c["n_fv_diff_key"] + c["n_fv_same_object"]
        c["false_verified_rate"] = fv / c["n_verified"] if c["n_verified"] else None
        c["net"] = c["helped"] - c["harmed"]
        c["coverage"] = c["n_changed"] / n_questions if n_questions else None
        per_subset_ok = all(v["harmed"] <= v["helped"]
                            for v in c["by_subset"].values())
        c["feasible"] = bool(
            graded_n
            and c["harmed"] <= 0.02 * graded_n
            and per_subset_ok
            and (c["n_verified"] == 0 or c["false_verified_rate"] <= 0.05))
        c["by_subset"] = {k: dict(v) for k, v in c["by_subset"].items()}

    chosen = None
    if graded_n:
        feas = [c for c in cells if c["feasible"] and c["net"] > 0]
        amb_rank = {"all": 0, "any": 1, "none": 2}
        feas.sort(key=lambda c: (-c["net"], c["harmed"], -c["cos_pair"],
                                 -c["nli_contradiction"], c["n_changed"],
                                 amb_rank[c["ambiguity_mode"]]))
        chosen = feas[0] if feas else None

    return {"n_questions": n_questions, "graded": bool(answer_fn),
            "n_llm_calls": n_llm_calls[0], "cells": cells, "chosen": chosen}


def write_operating_point(res: dict) -> Path:
    """The committed freeze artifact, from a finished evaluation result."""
    ch = res["chosen"]
    if ch is None:
        raise SystemExit("no chosen cell — nothing to freeze (report and stop)")
    art = {
        "thresholds": {
            "cos_pair": ch["cos_pair"], "r_min": ch["r_min"],
            "nmargin": _rg.NMARGIN_CAL, "H_z": _rg.H_Z_CAL,
            "ambiguity_mode": ch["ambiguity_mode"],
            "nli_contradiction": ch["nli_contradiction"]},
        "pair_filter": ch["pair_filter"],
        "cell": {k: v for k, v in ch.items() if k != "by_subset"},
        "by_subset": ch["by_subset"],
        "provenance": res["provenance"],
    }
    dst = REPO / "stage0_results/stage1_operating_point.json"
    dst.write_text(json.dumps(art, indent=1, default=str))
    return dst


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=list(CALIBRATION))
    ap.add_argument("--prepass", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="with --evaluate: write the committed operating-point "
                         "artifact to stage0_results/stage1_operating_point.json")
    ap.add_argument("--freeze-from-json", action="store_true",
                    help="write the operating-point artifact from an EXISTING "
                         "hnav/_out/stage1_calibration.json without recomputing "
                         "anything (grading is expensive and its cache is "
                         "in-memory)")
    ap.add_argument("--page-source", choices=("prepass", "benchmark"),
                    default="prepass",
                    help="'prepass' ranks chunks here (the pre-T13 behaviour, "
                         "unchanged by default); 'benchmark' reads the "
                         "benchmark's own top-k page and builds the candidate "
                         "pool from IT, so the facts the gate inspects are the "
                         "facts the model will see (pre-registration v2 "
                         "Amendment 3). Writes *_benchmarkpage.json.")
    ap.add_argument("--geometry-space", choices=("raw", "abtt"), default="raw",
                    help="'abtt' whitens fact AND query vectors with the offline "
                         "artifact before any cosine is taken, so the gate "
                         "decides in the isotropy-corrected space. Default 'raw' "
                         "reproduces every committed number unchanged.")
    ap.add_argument("--whitening-artifact", default=None,
                    help="path to the artifact from fit_abtt_artifact.py; "
                         "required by --geometry-space abtt")
    ap.add_argument("--whiten-scope", choices=("pairs", "all"), default="pairs",
                    help="'pairs' (default) whitens ONLY the fact-fact geometry "
                         "the gate decides on, leaving query->fact retrieval "
                         "raw; 'all' whitens the query too. Measured: 'all' "
                         "costs ~27%% of reachable true-supersession pairs "
                         "because select_pool then builds a worse pool.")
    ap.add_argument("--prepass-tag", default="",
                    help="[E2E] extra output-filename suffix (e.g. '_ces') so "
                         "a raw-space prepass with a lowered --cos-loose does "
                         "not collide with the shipped raw prepass")
    ap.add_argument("--cos-loose", type=float, default=None,
                    help="loose-screen threshold. MUST be re-set for the abtt "
                         "space: 0.90 is a raw-space coordinate.")
    ap.add_argument("--dry-run-lo", type=float, default=0.30,
                    help="lowest threshold in the --dry-run-pairs grid")
    ap.add_argument("--dry-run-pairs", action="store_true",
                    help="report loose-pair counts and NLI-cache coverage over a "
                         "threshold grid, then stop. No NLI, no writes.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--cache-only", action="store_true",
                    help="embeddings must all be cache hits (no GPU, no model)")
    ap.add_argument("--no-llm", action="store_true",
                    help="grid + false-verified rates only; no grading")
    ap.add_argument("--smoke-embedder", action="store_true")
    ap.add_argument("--stub-nli", action="store_true")
    ap.add_argument("--stub-llm", action="store_true")
    args = ap.parse_args()

    bad = [s for s in args.subsets if s not in CALIBRATION]
    if bad:
        print(f" REFUSED: {bad} is not the calibration split {CALIBRATION}. "
              f"Nothing is EVER tuned on the confirmatory subsets.")
        return 2

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if args.freeze_from_json:
        res = json.loads((cfg.out_dir / "stage1_calibration.json").read_text())
        if res.get("provenance", {}).get("smoke"):
            print(" REFUSED: the stored calibration is a SMOKE run.")
            return 2
        dst = write_operating_point(res)
        print(f" froze operating point -> {dst}")
        return 0

    smoke = args.smoke_embedder or args.stub_nli or args.stub_llm
    tag = ("_benchmarkpage" if args.page_source == "benchmark" else "")         + ("_SMOKE" if smoke else "")
    if smoke:
        print(" *** SMOKE RUN — no number below may feed a threshold. ***")

    if args.geometry_space == "abtt" and not args.whitening_artifact:
        print(" REFUSED: --geometry-space abtt needs --whitening-artifact. "
              "Fit one with hnav/stage0/fit_abtt_artifact.py (calibration "
              "split only).")
        return 2
    args._whitener, whiten_art = load_whitening(
        args.whitening_artifact if args.geometry_space == "abtt" else None)
    if args._whitener is not None:
        if args.cos_loose is None:
            print(" REFUSED: --geometry-space abtt without --cos-loose. The "
                  f"raw default {COS_LOOSE} is a coordinate in the RAW space; "
                  "reusing it here would silently admit almost nothing. Run "
                  "--dry-run-pairs to choose one.")
            return 2
        print(f" ABTT space: D={args._whitener.components.shape[0]}"
              f"  fingerprint={args._whitener.fingerprint()[:16]}..."
              f"  fit_on={whiten_art['fit_subsets']}"
              f"  cos_loose={args.cos_loose}")
    tag = tag + ("_abtt" if args._whitener is not None else "") + args.prepass_tag

    emb = make_embedder(args, cfg)
    data = json.load(open(DATA, encoding="utf-8"))
    states = {}
    for item in data:
        name = subset_name(item)
        if name in args.subsets:
            print(f" building {name} ...", flush=True)
            states[name] = build_state(item, name, emb, cfg, args)

    if args.dry_run_pairs:
        # [ABTT Phase 3b] The cached NLI table covers pairs the RAW loose screen
        # admitted. Whitening changes which pairs are components, so the sweep
        # cost is the pairs that are NEW -- measure it before paying it.
        cached = set()
        for name in states:
            for suffix in (tag, tag.replace("_abtt", "")):
                pth = cfg.out_dir / f"stage1_prepass_{name}{suffix}.json"
                if pth.exists():
                    cached |= set(json.loads(pth.read_text()).get("nli_table", {}))
        print(f" NLI cache holds {len(cached)} directed scores")
        # the same grid in both spaces, so the RAW ceiling is a control for the
        # whitened one: a difference in reachable true pairs is then a property
        # of the pool select_pool built, not of the threshold sweep
        grid = [round(x, 2) for x in np.arange(args.dry_run_lo, 0.99, 0.05)]
        for name, st in states.items():
            print()
            print(f" {name} (facts={len(st['facts'])})")
            print(f"   {'cos_loose':>10s} {'pairs':>8s} {'directed':>9s} "
                  f"{'cached':>8s} {'NEW':>8s} {'true_sup':>9s}")
            for thr in grid:
                pairs, directed = 0, set()
                n_true = 0
                for q in st["qs"]:
                    qv = emb.encode([q["retrieval_query"]])[0]
                    # mirror the real path exactly: scope decides whether the
                    # QUERY is whitened (pool composition) as well as the pair
                    # geometry, or only the pair geometry
                    if args._whitener is not None and args.whiten_scope == "all":
                        qv = args._whitener.transform(np.asarray(qv, dtype=np.float64))
                    pool = select_pool(list(st["records"]), qv, cfg.top_m)
                    if len(pool) < 2:
                        continue
                    mat = np.stack([np.asarray(r.vector, dtype=np.float64) for r in pool])
                    if args._whitener is not None:
                        mat = args._whitener.transform(mat)
                    sims = mat @ mat.T
                    for g in loose_components(pool, sims, thr):
                        for a in range(len(g)):
                            for b in range(a + 1, len(g)):
                                ra, rb = pool[g[a]], pool[g[b]]
                                pairs += 1
                                ka, kb = ra.metadata.get("key"), rb.metadata.get("key")
                                oa, ob = ra.metadata.get("object"), rb.metadata.get("object")
                                if ka is not None and ka == kb and oa != ob:
                                    n_true += 1
                                directed.add(f"{_sha(ra.text)}|{_sha(rb.text)}")
                                directed.add(f"{_sha(rb.text)}|{_sha(ra.text)}")
                new = len(directed - cached)
                print(f"   {thr:10.2f} {pairs:8d} {len(directed):9d} "
                      f"{len(directed) - new:8d} {new:8d} {n_true:9d}")
        return 0

    if args.prepass:
        nli = StubNLI() if args.stub_nli else build_nli(cfg)
        for name, st in states.items():
            print(f" prepass {name} ...", flush=True)
            pp = prepass_subset(st, emb, nli, cfg, args)
            out = cfg.out_dir / f"stage1_prepass_{name}{tag}.json"
            out.write_text(json.dumps(pp))
            npairs = sum(len(q['pairs']) for q in pp['questions'])
            print(f"   {out.name}: {len(pp['questions'])} questions, "
                  f"{npairs} loose pairs, {len(pp['nli_table'])} directed NLI, "
                  f"abtt AUC cos raw={pp['abtt_ab']['auc_cos_raw']} "
                  f"whitened={pp['abtt_ab']['auc_cos_whitened']}")

    if args.evaluate:
        prepasses = []
        for name in states:
            p = cfg.out_dir / f"stage1_prepass_{name}{tag}.json"
            if not p.exists():
                print(f" missing {p} — run --prepass first"); return 1
            pp = json.loads(p.read_text())
            # The replayed nli_table is a cache keyed only by the pair text.
            # Refuse to replay scores produced by a DIFFERENT engine config —
            # the T12 lesson: a stale hit is worse than a miss because it
            # looks like a result.
            want = {"model": cfg.nli_model, "max_length": cfg.nli_max_length,
                    "stub": bool(args.stub_nli)}
            got = pp.get("nli_config")
            if got is None:
                print(f" REFUSED: {p.name} predates NLI-config stamping "
                      f"(T12). Re-run --prepass.")
                return 2
            if got != want:
                print(f" REFUSED: {p.name} was scored with {got}, but this "
                      f"run is configured {want}. Re-run --prepass.")
                return 2
            prepasses.append(pp)
        res = evaluate(prepasses, states, cfg, args)
        res["provenance"] = {
            "git": _git_head(),
            "date": datetime.now(timezone.utc).isoformat(),
            "subsets": sorted(states),
            "smoke": smoke,
            "embed": {"model": cfg.embed_model, "dtype": cfg.embed_dtype,
                      "cache_misses": getattr(emb, "n_misses", None),
                      "cache_hits": getattr(emb, "n_hits", None)},
            "nli_model": cfg.nli_model,
            "nli_max_length": cfg.nli_max_length,
            "llm": None if args.no_llm else {
                "base_url": cfg.llm_base_url, "model": cfg.llm_model,
                "max_tokens": GENERATION_MAX_TOKENS,
                "temperature": cfg.llm_temperature,
                "prompt": "benchmark RAGSystem shape (Memory i + templated query), "
                          "system=SYSTEM_MESSAGE", "stub": args.stub_llm},
            "grid": {"cos_pair": COS_GRID, "r_min": R_MIN_GRID_LABELS,
                     "ambiguity_mode": AMB_GRID, "nli_contradiction": NLI_GRID,
                     "pair_filter": FILTER_GRID},
            "frozen_constants": {"nmargin": _rg.NMARGIN_CAL, "H_z": _rg.H_Z_CAL,
                                 "r_min_frozen": _rg.R_MIN_CAL,
                                 "r_min_loose": R_LOOSE},
            "abtt_ab": {p["subset"]: p["abtt_ab"] for p in prepasses},
        }
        out = cfg.out_dir / f"stage1_calibration{tag}.json"
        out.write_text(json.dumps(res, indent=1, default=str))
        print(f"\n wrote {out}")

        ch = res["chosen"]
        if ch is None:
            print("\n NO feasible operating point with net > 0 "
                  "(or --no-llm). REPORT AND STOP — do not force one.")
        else:
            print(f"\n chosen: cos_pair={ch['cos_pair']} r_min={ch['r_min_label']}"
                  f" amb={ch['ambiguity_mode']} nli={ch['nli_contradiction']}"
                  f" filter={ch['pair_filter']}")
            print(f"   net={ch['net']} (helped {ch['helped']} / harmed {ch['harmed']})"
                  f" coverage={ch['coverage']:.3f}"
                  f" false_verified={ch['false_verified_rate']}")
            if args.freeze and not smoke:
                print(f" froze operating point -> {write_operating_point(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
