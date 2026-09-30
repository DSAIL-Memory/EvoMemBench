#!/usr/bin/env python3
"""The confirmatory subset's prepass — detection inputs only, nothing tunable.  [T13]

``calibrate_read_policy.py`` refuses ``sh_64k`` outright, and that refusal must
stay absolute: it is the script that fits an operating point, and nothing is ever
fitted on a confirmatory subset. But a Branch A run still needs `sh_64k`
*detection inputs* — the candidate pool, the pair geometry and the bidirectional
NLI scores — computed at the **already-frozen** operating point.

So the confirmatory prepass is a separate, deliberately smaller act. It has no
grid, no thresholds, no objective and no way to write an operating point; it
cannot tune because there is nothing in it to tune with. What it does have is
three refusals of its own:

* only ``sh_64k`` — ``sh_262k`` is outside the campaign (pre-registration v2 §2)
  and every calibration subset belongs to ``calibrate_read_policy``;
* **page source is forced to ``benchmark``** and cannot be overridden. The pool
  must be built from the page the model will actually be shown (Amendment 3);
* **embeddings must be cache hits.** No vector is computed fresh on the
  confirmatory subset — they were produced and audited by
  ``hnav/deploy/refit_chunk_embeddings.py``, and a miss here would mean the
  geometry is not the geometry that was checked.

The heavy work is imported from ``calibrate_read_policy``, not re-implemented:
``build_state`` and ``prepass_subset`` are the same functions that produced every
committed prepass, so the confirmatory inputs cannot drift from the calibration
inputs by construction.

    CUDA_VISIBLE_DEVICES=1 python hnav/stage1/confirmatory_prepass.py
    python hnav/stage1/confirmatory_prepass.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.stage0.m2_retrieval_calibration import subset_name  # noqa: E402
from hnav.stage1.calibrate_read_policy import (DATA, build_state,  # noqa: E402
                                               make_embedder, prepass_subset)

CONFIRMATORY = "sh_64k"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", default=CONFIRMATORY)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--geometry-space", choices=("raw", "abtt"), default="raw",
                    help="'abtt' whitens the fact-fact geometry with the "
                         "offline artifact, matching the whitened arm's "
                         "operating point. Writes *_abtt.")
    ap.add_argument("--whitening-artifact", default=None)
    ap.add_argument("--cos-loose", type=float, default=None,
                    help="loose-screen threshold; required for abtt (0.90 is a "
                         "raw-space coordinate)")
    ap.add_argument("--prepass-tag", default="",
                    help="[E2E] extra output-filename suffix (e.g. '_ces') so "
                         "a raw-space prepass with a lowered --cos-loose does "
                         "not collide with the shipped raw prepass")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub-nli", action="store_true",
                    help="SMOKE ONLY: writes *_SMOKE and may not feed a run.")
    args = ap.parse_args()

    if args.subset != CONFIRMATORY:
        print(f" REFUSED: this script builds the {CONFIRMATORY} prepass only.\n"
              " Calibration subsets belong to calibrate_read_policy.py; "
              "sh_262k is outside\n the campaign (pre-registration v2 §2).")
        return 2

    cfg = _config.get_config()
    cfg.require_not_live()

    # Forced, not defaulted: neither can be overridden from the command line.
    # [ABTT] The artifact is FIT on the calibration split; here it is only
    # SPENT. Nothing about the whitening is derived from sh_64k.
    whitener = None
    if args.geometry_space == "abtt":
        if not args.whitening_artifact or args.cos_loose is None:
            print(" REFUSED: --geometry-space abtt needs --whitening-artifact "
                  "and --cos-loose.")
            return 2
        from hnav.stage1.calibrate_read_policy import load_whitening
        whitener, art = load_whitening(args.whitening_artifact)
        if CONFIRMATORY in art.get("fit_subsets", []):
            print(f" REFUSED: the artifact was fit on {art['fit_subsets']}, "
                  f"which includes the confirmatory subset.")
            return 2

    opts = SimpleNamespace(top_k=args.top_k, chunk_size=args.chunk_size,
                           max_questions=0, page_source="benchmark",
                           cache_only=True, smoke_embedder=False,
                           stub_nli=args.stub_nli,
                           whiten_scope="pairs", cos_loose=args.cos_loose)
    opts._whitener = whitener

    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    item = next((i for i in data if subset_name(i) == args.subset), None)
    if item is None:
        print(f" REFUSED: no dataset item for {args.subset}")
        return 2

    print("=" * 88)
    print(f" CONFIRMATORY PREPASS — {args.subset}")
    print("=" * 88)
    print("  page source : benchmark  (forced; Amendment 3)")
    print("  embeddings  : cache-only (forced; no fresh vectors on the "
          "confirmatory subset)")
    print("  tunables    : none. This script cannot fit or write an operating "
          "point.")
    if args.dry_run:
        print("\n --dry-run: nothing loaded.")
        return 0

    emb = make_embedder(opts, cfg)
    print(f"\n building {args.subset} ...", flush=True)
    st = build_state(item, args.subset, emb, cfg, opts)

    from hnav.core.read_gate import StubNLI, build_nli  # noqa: PLC0415
    nli = StubNLI() if args.stub_nli else build_nli(cfg)
    print(f" prepass {args.subset} ...", flush=True)
    pp = prepass_subset(st, emb, nli, cfg, opts)

    tag = ("_abtt" if whitener is not None else "") + args.prepass_tag \
        + ("_SMOKE" if args.stub_nli else "")
    out = cfg.out_dir / f"stage1_prepass_{args.subset}_benchmarkpage{tag}.json"
    out.write_text(json.dumps(pp), encoding="utf-8")
    npairs = sum(len(q["pairs"]) for q in pp["questions"])
    pool_sizes = {len(q["pool"]) for q in pp["questions"]}
    print(f"\n {out.name}")
    print(f"   questions {len(pp['questions'])}, loose pairs {npairs}, "
          f"directed NLI {len(pp['nli_table'])}")
    print(f"   pool sizes {sorted(pool_sizes)}, page_source "
          f"{pp['page_source']!r}, signals_available {pp['signals_available']}")
    print(f"   embedding cache: hits {getattr(emb, 'n_hits', None)}, "
          f"misses {getattr(emb, 'n_misses', None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
