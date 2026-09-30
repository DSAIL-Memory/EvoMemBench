#!/usr/bin/env python3
"""Fit the ABTT whitening artifact offline, once, from the calibration split.
[ABTT Phase 3a]

Why an artifact rather than a runtime fit. The read gate decides over a 50-fact
pool, which is far below ``whiten_min_fit_n = 200``, and that is the documented
reason ABTT was never armed. Fitting *offline* dissolves it: a pre-fitted
whitener has nothing left to estimate at decision time, it just applies. G1
also measured the global fit to be the better one - it matched or beat a
per-store fit and degraded far more gracefully at high D, because it estimates
its directions from 2,765 rows instead of 455.

**Fit subsets are the calibration split and nothing else.** ABTT is unsupervised
- it reads no answer and no label - so pooling held-out text into the fit would
not be leakage in the gold sense. It is still refused here, because the
transform applied to the held-out arena would then be partly derived from that
arena, which is a free objection to hand a reviewer for no measured benefit:
G1 showed the calibration-only global fit transfers.

    python hnav/stage0/fit_abtt_artifact.py --components 128
    python hnav/stage0/fit_abtt_artifact.py --components 64 --cache-only

Writes ``stage0_results/abtt/abtt_whitening_D<k>.json`` holding the mean, the
components, a sha256 fingerprint over both, and the provenance a consumer needs
to prove it is scoring the vectors it thinks it is.

Leakage: reads ONLY ``context``. No gold, no answers.
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

from hnav import config as _config                                   # noqa: E402
from hnav.adapters.mab_adapter import explode_facts                  # noqa: E402
from hnav.core.embedding import cache_key                            # noqa: E402
from hnav.core.geometry import ABTTWhitening                         # noqa: E402
from hnav.stage0.m6_abtt_geometry import DATA, make_embedder         # noqa: E402

CALIBRATION = ("sh_6k", "sh_32k")
OUTDIR = REPO / "stage0_results" / "abtt"


def main() -> int:
    cfg = _config.get_config()
    cfg.require_not_live()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", type=int, default=128,
                    help="D, the number of principal directions removed")
    ap.add_argument("--subsets", nargs="+", default=list(CALIBRATION))
    ap.add_argument("--cache-only", action="store_true", default=True)
    ap.add_argument("--gpu", dest="cache_only", action="store_false",
                    help="allow a real embedder if the cache misses")
    ap.add_argument("--smoke-embedder", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    illegal = [s for s in args.subsets if s not in CALIBRATION]
    if illegal:
        print(f" REFUSED: {illegal} is not in the calibration split "
              f"{list(CALIBRATION)}. The whitening artifact is applied to the "
              f"held-out arena, so fitting it there would make the transform "
              f"partly a function of the data it is spent on.", file=sys.stderr)
        return 2

    emb = make_embedder(args, cfg)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    mats, counts = [], {}
    for item in data:
        nm = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
            .replace("factconsolidation_", "")
        if nm not in args.subsets:
            continue
        facts = explode_facts(item["context"])
        mats.append(np.asarray(emb.encode([t for _, t in facts]), dtype=np.float64))
        counts[nm] = len(facts)
    if not mats:
        print(f" no subsets matched {args.subsets}", file=sys.stderr)
        return 1

    matrix = np.vstack(mats)
    w = ABTTWhitening(n_components=args.components,
                      min_fit_n=cfg.whiten_min_fit_n).fit(matrix)
    if not w.fitted:
        print(f" REFUSED to fit: n={matrix.shape[0]} < min_fit_n="
              f"{cfg.whiten_min_fit_n}", file=sys.stderr)
        return 2

    art = {
        "artifact": "ABTT whitening (frozen_global)",
        "regime": "frozen_global",
        "fit_subsets": sorted(counts),
        "facts_per_subset": counts,
        "n_fit": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "held_out_refused": ["sh_64k", "sh_262k"],
        "embed_cache_namespace": cache_key(cfg.embed_model, cfg.embed_dtype,
                                           cfg.embed_max_length),
        "embed_model": cfg.embed_model, "embed_dtype": cfg.embed_dtype,
        "embed_max_length": cfg.embed_max_length,
        "selected_on": "G1 offline detection quality, calibration split only "
                       "(stage0_results/abtt/G1_GATE_REPORT.md)",
        "whitening": w.to_dict(),
    }
    dest = Path(args.out) if args.out else OUTDIR / f"abtt_whitening_D{args.components}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(art), encoding="utf-8")

    print(f" fitted D={args.components} on {sorted(counts)} "
          f"({matrix.shape[0]} facts x {matrix.shape[1]}d)")
    print(f" fingerprint {w.fingerprint()[:16]}...")
    print(f" wrote {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")

    # round-trip proof: the artifact must reconstruct bit-identically, or a
    # consumer would silently score in a slightly different space
    back = ABTTWhitening.from_dict(json.loads(dest.read_text(encoding="utf-8"))["whitening"])
    probe = matrix[: min(64, matrix.shape[0])]
    assert np.array_equal(w.transform(probe), back.transform(probe)), \
        "artifact round-trip is not bit-identical"
    print(" round-trip verified: reloaded artifact transforms bit-identically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
