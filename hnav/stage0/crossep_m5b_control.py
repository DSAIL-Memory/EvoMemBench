#!/usr/bin/env python3
"""M5b — the controls the M5 redundancy claim rests on.  CACHED VECTORS ONLY.

M5 reports ``sim_max ≥ 0.95`` at a cluster-mean of 0.863/0.853. That number
alone cannot support a redundancy claim, for two independent reasons, and this
script measures both rather than arguing them:

**1. Does the threshold discriminate, or has cosine saturated?**
Cosine compresses on long text: two unrelated ~1,000-token chunks from the
same domain can sit high without being redundant. So the threshold is
calibrated against a control — random chunk pairs drawn from DIFFERENT
contexts, which by construction share no episode. If those also reached 0.95
the metric would be meaningless. (Measured: they never do.)

**2. Is ``sim_max`` even the right statistic?**
It is a NEAREST-NEIGHBOUR maximum over a growing store, so it conflates
"how redundant is this corpus" with "how many prior records were available to
match against". :func:`naive_nn_prediction` makes that explicit: under
independence, a pairwise rate ``p`` over ``i`` priors predicts
``1 − (1 − p)^i`` — which for these store sizes (30–250) saturates near 1.0.
The honest base rate is therefore the PAIRWISE within-context rate, and that
is the number the report leads with.

**3. How much of the mass is a harness artifact?**
CrossEp rebuilds each sample's trajectory from the same System Context block,
so the chunker re-emits that shared text once per sample. This attributes the
near-duplicate mass to that mechanism two ways: verbatim substring containment
(exact-duplicate driver) and 5-gram shingle containment (boundary-shifted
near-duplicate driver), plus whether a near-duplicate's nearest neighbour came
from a DIFFERENT sample — i.e. genuine cross-episode accumulation.

No GPU, no model, no network: every vector is read from the shared embedding
cache and a MISS IS A HARD ERROR rather than a silent re-embed, so this can
never quietly compute its numbers in a different space from the run it is
validating. Reproducible from ``--seed`` alone; the sampling method is stated
in :func:`control_pairs`.

    python hnav/stage0/crossep_m5b_control.py
    python hnav/stage0/crossep_m5b_control.py --pairs 5000 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.core.embedding import cache_key  # noqa: E402
from hnav.labeling.crossep_split import load_split  # noqa: E402
from hnav.stage0.crossep_m5_write_headroom import (DEFAULT_SOURCE,  # noqa: E402
                                                   iter_qwen3_stream)

SEED = 20260816
NEAR_DUP_TAU = 0.95          # the threshold M5 reports; validated, not assumed
SHINGLE_K = 5
CONTAINMENT_GRID = (0.9, 0.5, 0.1)
ORGANIC_MAX_CONTAINMENT = 0.1


# ── cache access (read-only, miss is fatal) ──────────────────────────────────
class CachedVectors:
    """Reads the shared embedding cache. Never writes, never embeds."""

    def __init__(self, namespace: str, cache_dir: Path) -> None:
        self.namespace = namespace
        self.cache_dir = Path(cache_dir)
        self.n_read = 0

    def path_for(self, text: str) -> Path:
        h = hashlib.sha256((self.namespace + "||" + text).encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def get(self, text: str) -> np.ndarray:
        p = self.path_for(text)
        if not p.exists():
            raise FileNotFoundError(
                f"vector missing from the cache for namespace {self.namespace!r}.\n"
                "M5b validates an existing run and must read that run's own "
                "vectors; it will not embed. Run M5 first with matching pins "
                "(model/dtype/max_length), or point --cache-dir at the right cache."
            )
        v = np.asarray(np.load(p), dtype=np.float64)
        self.n_read += 1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# ── text helpers ─────────────────────────────────────────────────────────────
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def shingles(s: str, k: int = SHINGLE_K) -> set[str]:
    w = norm_ws(s).lower().split()
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def system_blocks(source: Path) -> dict[str, str]:
    """``context_id -> system message``. It is identical across a context's
    samples — it *is* the context — so the first one seen is definitive."""
    out: dict[str, str] = {}
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["metadata"]["context_id"]
            if cid in out:
                continue
            for m in rec.get("messages", []):
                if m["role"] == "system":
                    out[cid] = m["content"]
                    break
    return out


def pcts(x) -> dict:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": float(a.mean()),
            "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()),
            f"frac_ge_{NEAR_DUP_TAU}": float((a >= NEAR_DUP_TAU).mean())}


# ── control 1: does the threshold discriminate? ──────────────────────────────
def control_pairs(ctx_vecs: dict[str, list[np.ndarray]], n_pairs: int,
                  seed: int) -> dict:
    """Random-pair similarity, within vs across contexts.

    Sampling (stated so the draws are reproducible from the seed alone):
      * ACROSS — draw 2 distinct context_ids uniformly without replacement,
        then 1 chunk uniformly from each. Shares no episode by construction.
      * WITHIN — draw 1 context_id uniformly from those with ≥2 chunks, then
        2 distinct chunk indices uniformly without replacement.
    Both use one ``random.Random(seed)`` consumed in this order.
    """
    rng = random.Random(seed)
    cids = [c for c, v in ctx_vecs.items() if len(v) >= 2]

    across = []
    for _ in range(n_pairs):
        a, b = rng.sample(cids, 2)
        across.append(float(rng.choice(ctx_vecs[a]) @ rng.choice(ctx_vecs[b])))

    within = []
    for _ in range(n_pairs):
        vs = ctx_vecs[rng.choice(cids)]
        i, j = rng.sample(range(len(vs)), 2)
        within.append(float(vs[i] @ vs[j]))

    return {"n_pairs_each": n_pairs, "seed": seed,
            "across_contexts": pcts(across), "within_context": pcts(within)}


# ── control 2: is sim_max just a store-size artifact? ────────────────────────
def naive_nn_prediction(ctx_sizes: dict[str, int], pairwise_p: float) -> dict:
    """Expected ``P(sim_max ≥ τ)`` if within-context pairs were INDEPENDENT.

    For the i-th write (0-based) there are ``i`` prior records, so under
    independence ``P(max ≥ τ) = 1 − (1 − p)^i``. Averaging over every write
    in every context gives what the nearest-neighbour statistic would read
    with NO redundancy beyond the pairwise base rate. If the observed value
    is at or below this, ``sim_max`` carries no information the pairwise rate
    did not already carry — which is exactly why the report leads with the
    pairwise number instead.
    """
    preds = []
    for n in ctx_sizes.values():
        for i in range(1, n):
            preds.append(1.0 - (1.0 - pairwise_p) ** i)
    return {"pairwise_p": pairwise_p,
            "mean_predicted_nn_rate": float(np.mean(preds)) if preds else float("nan"),
            "n_writes_considered": len(preds),
            "interpretation": (
                "Expected nearest-neighbour rate under INDEPENDENCE given the "
                "pairwise rate and the store sizes. Observed sim_max rates at "
                "or below this carry no evidence beyond the pairwise rate.")}


# ── control 3: how much is the shared System Context block? ──────────────────
def attribution(streams, ctx_vecs, sysmap, calib: set[str]) -> dict:
    """Attribute the near-duplicate mass to the shared System Context block.

    Recomputes each write's ``sim_max`` against its own prior writes — the
    same causal order M5 replays — then splits the ``≥ τ`` mass by how much
    of the chunk is System-Context text, and by whether the matched neighbour
    came from a different sample.
    """
    rows = []
    verbatim = 0
    total = 0
    for cid, events in streams.items():
        sys_norm = norm_ws(sysmap.get(cid, ""))
        sys_sh = shingles(sysmap.get(cid, ""))
        vecs = ctx_vecs[cid]
        for i, ev in enumerate(events):
            total += 1
            t_norm = norm_ws(ev["text"])
            if t_norm and t_norm in sys_norm:
                verbatim += 1
            if i == 0:
                continue
            sims = np.array([float(vecs[i] @ vecs[j]) for j in range(i)])
            k = int(sims.argmax())
            sh = shingles(ev["text"])
            rows.append({
                "split": "calibration" if cid in calib else "heldout",
                "sim_max": float(sims.max()),
                "containment": (len(sh & sys_sh) / len(sh)) if sh else 0.0,
                "cross_sample": events[k].get("task_id") != ev.get("task_id"),
            })

    out = {
        "n_write_events": total,
        "verbatim_system_substring": {
            "n": verbatim, "rate": verbatim / total if total else float("nan"),
            "note": ("Chunks that are literally inside the context's System "
                     "Context block. This is the exact-duplicate driver; "
                     "compare against M5's MD5 exact_dup_rate."),
        },
    }
    for split in ("calibration", "heldout"):
        R = [r for r in rows if r["split"] == split]
        if not R:
            continue
        hi = [r for r in R if r["sim_max"] >= NEAR_DUP_TAU]
        organic = [r for r in R if r["containment"] < ORGANIC_MAX_CONTAINMENT]
        out[split] = {
            "n_comparisons": len(R),
            "sim_max_frac_ge_tau_event_level": len(hi) / len(R),
            "near_dup_mass": {
                "n": len(hi),
                "cross_sample_frac": float(np.mean([r["cross_sample"] for r in hi]))
                if hi else float("nan"),
                "containment_frac_ge": {
                    f"{t:.1f}": (float(np.mean([r["containment"] >= t for r in hi]))
                                 if hi else float("nan"))
                    for t in CONTAINMENT_GRID},
            },
            "organic_chunks": {
                "definition": f"system-shingle containment < {ORGANIC_MAX_CONTAINMENT}",
                "n": len(organic),
                "sim_max_frac_ge_tau": float(np.mean(
                    [r["sim_max"] >= NEAR_DUP_TAU for r in organic]))
                if organic else float("nan"),
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--max-length", type=int, default=None,
                    help="cache namespace length; defaults to the configured pin")
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    max_length = args.max_length or getattr(cfg, "embed_max_length", 8192)
    namespace = cache_key(cfg.embed_model, cfg.embed_dtype, max_length)
    cache = CachedVectors(namespace, args.cache_dir or cfg.emb_cache_dir)

    print("=" * 74)
    print(" M5b — controls for the CrossEp redundancy claim  (cached vectors only)")
    print(f"   namespace={namespace}  pairs={args.pairs}  seed={args.seed}")
    print("=" * 74)

    streams, used_fallback = iter_qwen3_stream(args.source)
    sysmap = system_blocks(args.source)
    split = load_split()
    calib = set(split["calibration"])

    ctx_vecs = {cid: [cache.get(e["text"]) for e in evs]
                for cid, evs in streams.items()}
    print(f" {len(streams)} contexts, {cache.n_read} vectors read from cache")

    ctrl = control_pairs(ctx_vecs, args.pairs, args.seed)
    attr = attribution(streams, ctx_vecs, sysmap, calib)
    pairwise_p = ctrl["within_context"][f"frac_ge_{NEAR_DUP_TAU}"]
    naive = naive_nn_prediction({c: len(v) for c, v in ctx_vecs.items()}, pairwise_p)

    result = {
        "measurement": "m5b_crossep_control",
        "validates": "m5_crossep_write_headroom_qwen3_embedding.json",
        "source": str(args.source),
        "cache_namespace": namespace,
        "n_vectors_read": cache.n_read,
        "fallback_chunker": used_fallback,
        "near_dup_tau": NEAR_DUP_TAU,
        "shingle_k": SHINGLE_K,
        "threshold_control": ctrl,
        "store_size_effect": naive,
        "system_context_attribution": attr,
    }
    out = cfg.out_dir / "m5b_crossep_control.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    a, w = ctrl["across_contexts"], ctrl["within_context"]
    tau = f"frac_ge_{NEAR_DUP_TAU}"
    print("\n THRESHOLD CONTROL (does 0.95 discriminate, or has cosine saturated?)")
    print(f"   ACROSS contexts : p50={a['p50']:.4f} p95={a['p95']:.4f} "
          f"  >=tau {a[tau]:.4f}")
    print(f"   WITHIN context  : p50={w['p50']:.4f} p95={w['p95']:.4f} "
          f"  >=tau {w[tau]:.4f}   <-- the honest base rate")
    print("\n STORE-SIZE EFFECT (is sim_max informative beyond the pair rate?)")
    print(f"   naive independent prediction for the NN max: "
          f"{naive['mean_predicted_nn_rate']:.4f}")
    print("   (compare against M5's observed sim_max frac_ge 0.95 ~0.86)")
    print("\n SYSTEM-CONTEXT ATTRIBUTION")
    v = attr["verbatim_system_substring"]
    print(f"   verbatim system substrings: {v['n']}/{attr['n_write_events']} "
          f"({v['rate']:.4f})  <-- the exact-duplicate driver")
    for sp in ("calibration", "heldout"):
        if sp not in attr:
            continue
        b = attr[sp]
        print(f"   [{sp}] near-dup mass n={b['near_dup_mass']['n']}  "
              f"cross-sample {b['near_dup_mass']['cross_sample_frac']:.3f}  "
              f"containment>=0.9 {b['near_dup_mass']['containment_frac_ge']['0.9']:.3f}")
        print(f"      ORGANIC chunks n={b['organic_chunks']['n']}  "
              f">=tau {b['organic_chunks']['sim_max_frac_ge_tau']:.4f}")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
