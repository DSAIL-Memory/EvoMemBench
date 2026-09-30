#!/usr/bin/env python3
"""M6 - ABTT geometry re-measurement: does correcting anisotropy before the
cosine screen improve conflict discrimination?  [ABTT Phase 1]

The shipped detector thresholds raw cosine in a space measured to be strongly
anisotropic: unrelated facts sit at cos ~0.604 and no candidate pair at sh_262k
falls below 0.65, so the usable band is roughly ``[0.52, 1.00]``.
``ABTTWhitening`` (``hnav/core/geometry.py``) has been built and tested since
Stage 0 but has never fed a decision. This script measures what it would change.

**The three mechanisms by which whitening could help.** They are kept separate
because they need different evidence and only one of them is cheap:

  M3  *ranking inside the screened pool* - do supersession pairs rank above
      non-supersession pairs better after whitening? Measurable from the
      committed prepass alone (``--from-prepass``), because the prepass already
      stores ``cos_w`` / ``r_*_w`` beside the raw values.
  M2  *pairs the raw screen never admitted* - invisible to M3, because the
      prepass pair support **is** the raw loose screen. Needs vectors.
  M1  *pool composition* - ``select_pool`` keeps the 50 facts most similar to
      the query, so whitening changes **which facts are candidates at all**.
      Needs vectors and query embeddings; not covered here.

The committed A/B (``stage1_calibration.json -> provenance.abtt_ab``) measured a
slice of M3 only, reported +0.019 / +0.003 AUC, and was logged rather than acted
on. An AUC gain is not a decision gain, so every comparison here is made **at
equal coverage** - matching the number of pairs admitted - or on a
threshold-free statistic. Comparing raw ``cos >= 0.90`` against whitened
``cos_w >= 0.90`` is meaningless: whitening moves the distribution by
construction, so an unmatched comparison measures the shift, not the signal.

    # M3, no GPU, no vectors, runs anywhere the prepass JSON exists
    python hnav/stage0/m6_abtt_geometry.py --from-prepass hnav/_out/stage1_prepass_sh_6k.json

    # M2 + anisotropy + candidate floor, needs the embedding cache
    python hnav/stage0/m6_abtt_geometry.py --subsets sh_6k sh_32k
    python hnav/stage0/m6_abtt_geometry.py --subsets sh_6k --smoke-embedder

Fit-basis regimes are swept rather than assumed. The documented reason for
leaving ABTT off - "the 50-fact pool is below ``min_fit_n=200``" - conflates the
fit basis with the decision pool; ``MABAdapter.facts`` holds the whole store and
``select_pool`` only selects from it, so ``per_store`` is available online.
``pool_level`` is measured anyway, so the refusal is evidenced, not asserted.

Leakage: reads ONLY ``context`` (same footprint as M1b). No gold, no answers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config                                        # noqa: E402
from hnav.adapters.mab_adapter import explode_facts                       # noqa: E402
from hnav.core.embedding import build_embedder                            # noqa: E402
from hnav.core.geometry import ABTTWhitening                              # noqa: E402
from hnav.stage0.m1b_grouping_ablation import knn_candidates, truth_pairs  # noqa: E402
from hnav.stage0.m4_marginal_diff_test import auc                         # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
CALIBRATION = ("sh_6k", "sh_32k")

# Swept, not assumed. D=0 is mean-centering only: it removes the additive part
# of the anisotropy without estimating a single principal direction, so it is
# the variance-cheap baseline any D>0 result has to beat.
D_GRID = (0, 1, 2, 3, 5, 8, 16)
REGIMES = ("per_store", "frozen_global", "pool_level")

INTERPRETATION = (
    "A gain that appears in AUC but vanishes at equal coverage is a rescaling, "
    "not a detection improvement: the pipeline thresholds cosine, so only a "
    "change in the ORDER of pairs can change a decision. Report the equal-"
    "coverage deltas as the headline and AUC as context, never the reverse."
)


# -- embedder choice ---------------------------------------------------------
class _FailOnMiss:
    """Inner embedder for --cache-only: any miss is a hard, named error.

    Mirrors ``calibrate_read_policy._FailOnMiss`` rather than importing it, so
    a Stage-0 script keeps no dependency on Stage 1 (M1 inlines its embedder for
    the same reason). M1 already paid for these vectors, so a miss here means
    the namespace moved - which must stop the run, not silently re-embed into a
    different geometry.
    """

    dim = 0

    def encode(self, texts):
        raise RuntimeError(
            f"--cache-only but {len(texts)} text(s) missing from the embedding "
            f"cache, first={texts[0][:80]!r}. The cache namespace encodes model, "
            f"dtype and max_length - a miss means one of them changed. Run M1 "
            f"with the real embedder first, or drop --cache-only.")


def make_embedder(args, cfg):
    if args.smoke_embedder:
        from hnav.core.embedding import HashEmbedder
        return HashEmbedder(dim=64)
    from hnav.core.embedding import DiskCachedEmbedder, cache_key
    key = cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length)
    if args.cache_only:
        return DiskCachedEmbedder(_FailOnMiss(), cfg.emb_cache_dir, key)
    return build_embedder(cfg)


# -- shared helpers ----------------------------------------------------------
def describe(v: np.ndarray) -> dict:
    """Distribution summary. ``band_p10_p90`` is the working range a threshold
    actually has to discriminate inside."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {}
    p10, p90 = float(np.percentile(v, 10)), float(np.percentile(v, 90))
    return {"n": int(v.size), "min": float(v.min()), "p10": p10,
            "p50": float(np.percentile(v, 50)), "p90": p90,
            "max": float(v.max()), "mean": float(v.mean()),
            "std": float(v.std()), "band_p10_p90": p90 - p10}


def equal_coverage_delta(raw: np.ndarray, whit: np.ndarray, y: np.ndarray,
                         raw_thresholds) -> list[dict]:
    """Precision/recall of the whitened score at the SAME number of admitted
    pairs as each raw threshold.

    Matching coverage is what makes the two spaces comparable: whitening shifts
    the whole distribution down, so any fixed threshold admits fewer pairs and
    would look more precise for a reason that has nothing to do with signal.
    """
    rows: list[dict] = []
    n_true = int(y.sum())
    if n_true == 0 or raw.size != whit.size:
        return rows
    order_w = np.sort(whit)[::-1]
    for thr in raw_thresholds:
        sel_r = raw >= thr
        n_adm = int(sel_r.sum())
        if n_adm == 0 or n_adm > whit.size:
            continue
        thr_w = float(order_w[n_adm - 1])       # admits exactly n_adm pairs
        sel_w = whit >= thr_w
        p_r, r_r = float(y[sel_r].mean()), float(y[sel_r].sum() / n_true)
        p_w, r_w = float(y[sel_w].mean()), float(y[sel_w].sum() / n_true)
        rows.append({
            "raw_threshold": float(thr), "n_admitted": n_adm,
            "whitened_threshold_equal_coverage": thr_w,
            "raw_precision": p_r, "raw_recall": r_r,
            "whitened_precision": p_w, "whitened_recall": r_w,
            "delta_precision": p_w - p_r, "delta_recall": r_w - r_r,
        })
    return rows


def recall_at_precision(score: np.ndarray, y: np.ndarray,
                        targets=(0.90, 0.95, 0.99, 1.0)) -> dict:
    """Best recall achievable at or above each precision target.

    This is the shape the shipped system runs in: the operating point was
    selected under ``n_suppressed_harmful == 0`` (precision 1.000) and then
    maximised recall, so recall-at-precision is the decision-relevant readout
    and plain AUC is not.
    """
    y = np.asarray(y, dtype=bool)
    n_true = int(y.sum())
    if n_true == 0 or y.size == 0:
        return {}
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / n_true
    out = {}
    for t in targets:
        ok = prec >= t
        out[f"recall_at_precision_{t:g}"] = float(rec[ok].max()) if ok.any() else 0.0
    return out


def threshold_at_precision(score: np.ndarray, y: np.ndarray,
                           target: float) -> float | None:
    """The loosest threshold whose admitted set reaches ``target`` precision.

    Loosest rather than tightest because the screen's job is recall: among the
    thresholds that meet the precision bar, the pipeline wants the one that
    admits most. Returns None when the bar is unreachable.
    """
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.size == 0 or y.sum() == 0:
        return None
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.flatnonzero(prec >= target)
    return float(ss[ok.max()]) if ok.size else None


def apply_threshold(score: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """Precision/recall of an absolute threshold — the transfer readout."""
    y = np.asarray(y, dtype=bool)
    sel = np.asarray(score, dtype=float) >= thr
    n_true = int(y.sum())
    return {"threshold": float(thr), "n_admitted": int(sel.sum()),
            "coverage": float(sel.mean()),
            "precision": float(y[sel].mean()) if sel.any() else float("nan"),
            "recall": float(y[sel].sum() / n_true) if n_true else float("nan")}


def transfer_matrix(per_subset: dict, target: float = 0.95) -> dict:
    """Fit an absolute threshold on one subset, spend it on another.

    This is the portability question, and it is the one place ABTT has a
    mechanism-level reason to win: the raw cosine floor rises with store size
    (0.58 -> 0.65 across the subsets), so a constant raw threshold silently
    tightens as the store grows. Removing the shared directions should remove
    that drift. ``REFIT_PER_SUBSET`` exists in the shipped gate precisely
    because pooling a threshold across these subsets was indefensible; if
    whitening makes one threshold transfer, that constraint relaxes.

    ``degradation`` is the precision lost by spending a foreign threshold
    instead of the subset's own — lower is better, and the raw-vs-whitened
    difference in it is the headline.
    """
    out: dict = {"precision_target": target, "space": {}}
    for space in ("raw", "whitened"):
        rows = []
        for fit_s, fit_d in per_subset.items():
            thr = threshold_at_precision(fit_d[space], fit_d["y"], target)
            if thr is None:
                rows.append({"fit_on": fit_s, "threshold": None,
                             "note": f"precision {target} unreachable"})
                continue
            for eval_s, eval_d in per_subset.items():
                got = apply_threshold(eval_d[space], eval_d["y"], thr)
                own = threshold_at_precision(eval_d[space], eval_d["y"], target)
                own_p = (apply_threshold(eval_d[space], eval_d["y"], own)["precision"]
                         if own is not None else float("nan"))
                rows.append({"fit_on": fit_s, "eval_on": eval_s,
                             "transferred": fit_s != eval_s, **got,
                             "own_threshold": own,
                             "degradation_vs_own": own_p - got["precision"]})
        out["space"][space] = rows
    cross = {sp: [r for r in rs if r.get("transferred")]
             for sp, rs in out["space"].items()}
    out["summary"] = {
        sp: {"mean_transferred_precision":
             float(np.nanmean([r["precision"] for r in rs])) if rs else None,
             "mean_degradation":
             float(np.nanmean([r["degradation_vs_own"] for r in rs])) if rs else None,
             "threshold_spread":
             float(np.nanmax([r["threshold"] for r in rs])
                   - np.nanmin([r["threshold"] for r in rs])) if rs else None}
        for sp, rs in cross.items()}
    return out


def rank_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """How much whitening actually reorders pairs.

    A monotone rescale cannot change any thresholded decision, so a Spearman
    near 1.0 predicts the equal-coverage deltas will be ~0 no matter how far the
    raw cosines moved. This is the diagnostic that explains a null.
    """
    if a.size < 3:
        return {}
    from scipy.stats import kendalltau, spearmanr
    return {"spearman": float(spearmanr(a, b).statistic),
            "kendall_tau": float(kendalltau(a, b).statistic)}


# -- path A: mechanism M3, from the committed prepass ------------------------
def analyse_prepass(path: Path) -> dict:
    """Re-analyse a prepass artifact's per-pair raw and whitened geometry.

    Reproduces the committed ``abtt_ab`` AUCs as a self-check, then asks the
    question that block did not: does the gain survive equal coverage?
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    flat = [p for q in d.get("questions", []) for p in q.get("pairs", [])]
    if not flat:
        return {"source": str(path), "error": "no pairs in prepass"}

    y = np.array([bool(p["same_key"]) and not bool(p["same_object"]) for p in flat])
    cos = np.array([p["cos"] for p in flat], dtype=float)
    cos_w = np.array([p["cos_w"] for p in flat], dtype=float)
    rmin = np.array([-min(p["r_a"], p["r_b"]) for p in flat], dtype=float)
    rmin_w = np.array([-min(p["r_a_w"], p["r_b_w"]) for p in flat], dtype=float)

    out = {
        "source": path.name,
        "subset": d.get("subset"),
        "n_facts": d.get("n_facts"),
        "cos_loose": d.get("cos_loose"),
        "n_pairs": len(flat),
        "n_true_supersession": int(y.sum()),
        "base_rate": float(y.mean()),
        # The single most important caveat about this whole path.
        "pair_support": ("RAW loose screen - pairs the whitened screen would "
                         "admit but the raw screen rejected are NOT in this "
                         "population, so mechanism M2 is invisible here"),
        "distributions": {"cos_raw": describe(cos), "cos_whitened": describe(cos_w)},
        "auc": {"cos_raw": auc(y, cos), "cos_whitened": auc(y, cos_w),
                "rmin_raw": auc(y, rmin), "rmin_whitened": auc(y, rmin_w)},
        "class_separation": {
            "raw_mean_gap": float(cos[y].mean() - cos[~y].mean()),
            "whitened_mean_gap": float(cos_w[y].mean() - cos_w[~y].mean()),
        },
        "rank_agreement_cos_vs_cos_w": rank_agreement(cos, cos_w),
        "equal_coverage": equal_coverage_delta(cos, cos_w, y,
                                               [0.90, 0.92, 0.94, 0.96]),
        "recall_at_precision": {"cos_raw": recall_at_precision(cos, y),
                                "cos_whitened": recall_at_precision(cos_w, y)},
    }
    a = out["auc"]
    ec = out["equal_coverage"]
    out["verdict_m3"] = {
        "auc_delta_cos": a["cos_whitened"] - a["cos_raw"],
        "auc_delta_rmin": a["rmin_whitened"] - a["rmin_raw"],
        "max_equal_coverage_precision_gain":
            max((r["delta_precision"] for r in ec), default=0.0),
        "max_equal_coverage_recall_gain":
            max((r["delta_recall"] for r in ec), default=0.0),
    }
    # kept private (stripped before serialisation) so the cross-subset transfer
    # can be computed without re-reading and re-parsing every prepass file
    out["_arrays"] = {"raw": cos, "whitened": cos_w, "y": y}
    return out


# -- path B: mechanism M2 + anisotropy, from vectors -------------------------
def fit_whitener(regime: str, d: int, store: np.ndarray,
                 global_matrix: np.ndarray | None, min_fit_n: int,
                 pool_cap: int = 50, seed: int = 0) -> ABTTWhitening:
    """Fit ABTT under one regime.

    ``d == 0`` is mean-centering only. ``ABTTWhitening`` always removes at least
    one direction, so it is fitted with one component and the component matrix
    is then emptied - explicit rather than clever, and it keeps the mean.

    ``pool_level`` fits on ``pool_cap`` rows, not on the store: it exists to
    *demonstrate* the refusal that the documented "ABTT cannot run at gate time"
    argument rests on. Handing it the full store would silently turn it into a
    second copy of ``per_store`` and the refusal would never be exercised.
    """
    w = ABTTWhitening(n_components=max(int(d), 1), min_fit_n=min_fit_n)
    if regime == "pool_level":
        n = min(int(pool_cap), store.shape[0])
        idx = np.random.default_rng(seed).choice(store.shape[0], n, replace=False)
        basis = store[np.sort(idx)]
    else:
        basis = {"per_store": store, "frozen_global": global_matrix}[regime]
    if basis is None:
        w.refused = True
        return w
    w.fit(basis)
    if w.fitted and int(d) == 0:
        w.components = np.zeros((0, basis.shape[1]))
    return w


def analyse_vectors(name: str, facts, mat: np.ndarray, global_matrix,
                    args) -> dict:
    """Anisotropy, the candidate-pair floor, and the grouping PR sweep, raw vs
    whitened, over the regime x D grid."""
    truth, _, n_parsed = truth_pairs(facts)
    rng = np.random.default_rng(args.seed)

    def anisotropy(m: np.ndarray) -> dict:
        """Mean cosine between random pairs - 0 in a perfectly isotropic space."""
        n = m.shape[0]
        k = min(args.control_pairs, max(n * (n - 1) // 2, 1))
        i = rng.integers(0, n, size=k * 2)
        j = rng.integers(0, n, size=k * 2)
        ok = i != j
        i, j = i[ok][:k], j[ok][:k]
        if i.size == 0:
            return {}
        return describe(np.einsum("ij,ij->i", m[i], m[j]))

    def grouping(m: np.ndarray) -> dict:
        cands = knn_candidates(m, args.top_n)
        s = np.array([c[2] for c in cands], dtype=float)
        yy = np.array([(c[0], c[1]) in truth for c in cands], dtype=bool)
        return {"n_candidates": len(cands),
                "candidate_floor_min_cos": float(s.min()) if s.size else None,
                "auc": auc(yy, s),
                "recall_at_precision": recall_at_precision(s, yy),
                "_scores": s, "_truth": yy}

    base = grouping(mat)
    pub = lambda g: {k: v for k, v in g.items() if not k.startswith("_")}  # noqa: E731
    out = {
        "subset": name, "n_facts": len(facts), "n_parsed": n_parsed,
        "n_truth_pairs": len(truth),
        "raw": {"anisotropy_random_pairs": anisotropy(mat),
                "grouping": pub(base)},
        "regimes": {},
    }
    for regime in args.regimes:
        for d in args.d_grid:
            w = fit_whitener(regime, d, mat, global_matrix, args.min_fit_n,
                             pool_cap=args.pool_cap, seed=args.seed)
            tag = f"{regime}|D={d}"
            if not w.fitted:
                out["regimes"][tag] = {"fitted": False, "refused": bool(w.refused),
                                       "n_fit": int(w.n_fit),
                                       "min_fit_n": args.min_fit_n}
                continue
            mw = np.asarray(w.transform(mat), dtype=np.float64)
            g = grouping(mw)
            same_support = (base["_scores"].size == g["_scores"].size)
            out["regimes"][tag] = {
                "fitted": True, "n_fit": int(w.n_fit),
                "n_components_removed": int(w.components.shape[0]),
                "anisotropy_random_pairs": anisotropy(mw),
                "grouping": pub(g),
                "auc_delta_vs_raw": g["auc"] - base["auc"],
                "candidate_set_identical_to_raw": same_support,
                "equal_coverage_vs_raw": equal_coverage_delta(
                    base["_scores"], g["_scores"], base["_truth"],
                    [0.90, 0.92, 0.94]) if same_support else
                    "kNN candidate sets differ - compare via recall_at_precision",
            }
    return out


def main() -> int:
    cfg = _config.get_config()
    cfg.require_not_live()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-prepass", nargs="+", default=None,
                    help="analyse prepass artifacts (mechanism M3); no vectors needed")
    ap.add_argument("--subsets", nargs="+", default=list(CALIBRATION))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    ap.add_argument("--d-grid", nargs="+", type=int, default=list(D_GRID))
    ap.add_argument("--min-fit-n", type=int, default=None,
                    help="override whiten_min_fit_n (default: config)")
    ap.add_argument("--max-facts", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--control-pairs", type=int, default=20000)
    ap.add_argument("--transfer-precision", type=float, default=0.95,
                    help="precision bar a transferred threshold must clear")
    ap.add_argument("--pool-cap", type=int, default=50,
                    help="rows the pool_level regime fits on (the read-gate pool size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-only", action="store_true",
                    help="read vectors from the disk cache only; a miss is fatal")
    ap.add_argument("--smoke-embedder", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.min_fit_n is None:
        args.min_fit_n = cfg.whiten_min_fit_n

    result = {
        "measurement": "M6 ABTT geometry re-measurement",
        "interpretation": INTERPRETATION,
        "config": {"embed_model": cfg.embed_model, "embed_dtype": cfg.embed_dtype,
                   "embed_max_length": cfg.embed_max_length,
                   "whiten_min_fit_n": args.min_fit_n,
                   "d_grid": args.d_grid, "regimes": args.regimes},
    }

    if args.from_prepass:
        result["mode"] = "prepass (mechanism M3 only)"
        result["prepass"] = [analyse_prepass(Path(p)) for p in args.from_prepass]
        for r in result["prepass"]:
            if "error" in r:
                print(f"  {r['source']}: {r['error']}")
                continue
            v = r["verdict_m3"]
            print(f"\n{r['subset']}  n_pairs={r['n_pairs']}  "
                  f"true={r['n_true_supersession']} ({r['base_rate']:.1%})")
            print(f"  AUC cos   raw={r['auc']['cos_raw']:.4f} -> "
                  f"whitened={r['auc']['cos_whitened']:.4f} "
                  f"({v['auc_delta_cos']:+.4f})")
            print(f"  AUC r_min raw={r['auc']['rmin_raw']:.4f} -> "
                  f"whitened={r['auc']['rmin_whitened']:.4f} "
                  f"({v['auc_delta_rmin']:+.4f})")
            print(f"  band p10-p90 raw="
                  f"{r['distributions']['cos_raw']['band_p10_p90']:.4f} -> "
                  f"whitened={r['distributions']['cos_whitened']['band_p10_p90']:.4f}"
                  f"   mean class gap "
                  f"{r['class_separation']['raw_mean_gap']:.4f} -> "
                  f"{r['class_separation']['whitened_mean_gap']:.4f}")
            ra = r["rank_agreement_cos_vs_cos_w"]
            if ra:
                print(f"  rank agreement spearman={ra['spearman']:.4f} "
                      f"kendall={ra['kendall_tau']:.4f}")
            print("  EQUAL COVERAGE (the decision-relevant readout):")
            for row in r["equal_coverage"]:
                print(f"    raw>={row['raw_threshold']:.2f} admits "
                      f"{row['n_admitted']:5d} | P "
                      f"{row['raw_precision']:.4f}->{row['whitened_precision']:.4f}"
                      f" ({row['delta_precision']:+.4f}) | R "
                      f"{row['raw_recall']:.4f}->{row['whitened_recall']:.4f}"
                      f" ({row['delta_recall']:+.4f})")

        # cross-subset threshold transfer needs >= 2 subsets
        per = {r["subset"]: r.pop("_arrays") for r in result["prepass"]
               if "_arrays" in r}
        if len(per) >= 2:
            tm = transfer_matrix(per, target=args.transfer_precision)
            result["transfer"] = tm
            print(f"\nTHRESHOLD TRANSFER (fit at precision "
                  f">= {args.transfer_precision}, spend on the other subset):")
            for sp, s in tm["summary"].items():
                if s["mean_transferred_precision"] is None:
                    continue
                print(f"  {sp:9s} transferred precision="
                      f"{s['mean_transferred_precision']:.4f}"
                      f"  degradation={s['mean_degradation']:+.4f}"
                      f"  threshold spread={s['threshold_spread']:.4f}")
            for sp in ("raw", "whitened"):
                for r in tm["space"][sp]:
                    if r.get("transferred"):
                        print(f"    {sp:9s} fit {r['fit_on']:6s} -> eval "
                              f"{r['eval_on']:6s}  thr={r['threshold']:.4f}"
                              f"  P={r['precision']:.4f} R={r['recall']:.4f}"
                              f"  (own thr={r['own_threshold']:.4f})")
        else:
            for r in result["prepass"]:
                r.pop("_arrays", None)
    else:
        result["mode"] = "vectors (mechanism M2 + anisotropy + candidate floor)"
        data = json.loads(DATA.read_text(encoding="utf-8"))
        emb = make_embedder(args, cfg)
        loaded = {}
        for item in data:
            # same derivation as M1b, so subset names cannot drift between them
            nm = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
                .replace("factconsolidation_", "")
            if nm not in args.subsets:
                continue
            facts = explode_facts(item["context"])
            if args.max_facts:
                facts = facts[:args.max_facts]
            mat = np.asarray(emb.encode([t for _, t in facts]), dtype=np.float64)
            loaded[nm] = (facts, mat)
        if not loaded:
            print(f"  no subsets matched {args.subsets}", file=sys.stderr)
            return 1
        # frozen_global is fitted on the calibration split ONLY - never on
        # sh_64k / sh_262k, whatever --subsets asks for.
        cal = [m for s, (_, m) in loaded.items() if s in CALIBRATION]
        global_matrix = np.vstack(cal) if cal else None
        result["frozen_global_fit_subsets"] = [s for s in loaded if s in CALIBRATION]
        result["subsets"] = {}
        for s, (facts, mat) in loaded.items():
            r = analyse_vectors(s, facts, mat, global_matrix, args)
            result["subsets"][s] = r
            print(f"\n{s}  facts={r['n_facts']}  truth_pairs={r['n_truth_pairs']}")
            print(f"  raw  anisotropy(mean random-pair cos)="
                  f"{r['raw']['anisotropy_random_pairs'].get('mean', float('nan')):.4f}"
                  f"  grouping AUC={r['raw']['grouping']['auc']:.4f}"
                  f"  floor={r['raw']['grouping']['candidate_floor_min_cos']:.4f}")
            for tag, g in r["regimes"].items():
                if not g["fitted"]:
                    print(f"  {tag:22s} REFUSED (n_fit={g['n_fit']} "
                          f"< {g['min_fit_n']})")
                    continue
                print(f"  {tag:22s} anisotropy="
                      f"{g['anisotropy_random_pairs'].get('mean', float('nan')):+.4f}"
                      f"  AUC={g['grouping']['auc']:.4f} "
                      f"({g['auc_delta_vs_raw']:+.4f})"
                      f"  floor={g['grouping']['candidate_floor_min_cos']:.4f}")

    outdir = REPO / "stage0_results" / "abtt"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) if args.out else outdir / "m6_abtt_geometry.json"
    dest.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
