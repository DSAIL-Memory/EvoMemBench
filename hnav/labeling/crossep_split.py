#!/usr/bin/env python3
"""CrossEp-Know calibration / held-out split — cluster-level, frozen.  [T10]

Threshold discipline requires a calibration subset BEFORE anything is fit
(brief hard rule; ``HNAV_VISION_GAP.md`` §2.5 lists the missing CrossEp split
as a prerequisite artifact, not an afterthought). This module defines it.

Design decisions, recorded here because they are load-bearing:

* **The unit is the cluster (``context_id``), never the sample.** Samples
  within a context share a memory bank and a system context; ICC = 0.346 and
  design effect 3.20 mean sample-level splitting would leak calibration
  information into the held-out set through shared clusters (effective
  N ≈ 276 of 884).
* **~40% of clusters to calibration**, mirroring the primary arena's ratio
  (sh_6k + sh_32k are 2 of 4 subsets; by write volume they are the smaller
  pair). Assignment is stratified by ``context_category`` so neither side is
  starved of a domain; within each category, clusters are shuffled by a fixed
  seeded RNG and the first ``round(0.4·n)`` (at least 1 when n ≥ 2) go to
  calibration. A category with a single cluster goes to held-out — it cannot
  be split, and thresholds must never be fit on the only exemplar of a
  stratum.
* **The artifact is committed** (``crossep_split.json``, next to this file)
  and re-derivable: ``derive_split`` is a pure function of the dataset file
  and the constants below. ``hnav/tests/test_crossep_split.py`` fails if the
  committed artifact and a fresh derivation ever disagree — the split cannot
  drift silently.

Thresholds, gate settings, and any tuned quantity for the CrossEp arena may
be fit on the CALIBRATION clusters only. Held-out clusters are for one-shot
confirmation. ``crossep_m5_write_headroom.py`` stratifies by this split and
refuses to fit anything at all (it is measurement-only).

This is a labeling-layer module: it may read the dataset file (it uses only
``metadata.context_id`` / ``metadata.context_category`` / ``task_id``).
Nothing under ``hnav/core`` or ``hnav/adapters`` imports it.

    python hnav/labeling/crossep_split.py          # derive, verify, (re)write
"""
from __future__ import annotations

import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASET = REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl"
SPLIT_PATH = Path(__file__).with_name("crossep_split.json")

SEED = 20260815          # date-derived, fixed forever
CALIB_FRAC = 0.40        # of clusters, per category
ICC = 0.346              # measured; recorded for the analysis layer
DESIGN_EFFECT = 3.20

__all__ = ["derive_split", "load_split", "split_of", "DATASET", "SPLIT_PATH",
           "SEED", "CALIB_FRAC"]


def _clusters(dataset_path: Path) -> "OrderedDict[str, dict]":
    """``context_id -> {category, n_samples}`` in file order."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line).get("metadata", {})
            cid = meta["context_id"]
            entry = out.setdefault(cid, {
                "context_category": meta.get("context_category", ""),
                "n_samples": 0,
            })
            entry["n_samples"] += 1
    return out


def derive_split(dataset_path: Path = DATASET, seed: int = SEED,
                 calib_frac: float = CALIB_FRAC) -> dict:
    """Pure function of the dataset file and the constants. Deterministic."""
    clusters = _clusters(dataset_path)

    by_cat: "OrderedDict[str, list[str]]" = OrderedDict()
    for cid, info in clusters.items():
        by_cat.setdefault(info["context_category"], []).append(cid)

    calibration: list[str] = []
    heldout: list[str] = []
    per_category = {}
    for cat, cids in by_cat.items():
        n = len(cids)
        if n < 2:
            take = 0                       # a lone cluster cannot be split
        else:
            take = max(1, round(calib_frac * n))
            take = min(take, n - 1)        # never empty the held-out side
        shuffled = list(cids)
        random.Random(f"{seed}|{cat}").shuffle(shuffled)
        cal, held = shuffled[:take], shuffled[take:]
        calibration.extend(cal)
        heldout.extend(held)
        per_category[cat] = {
            "n_clusters": n,
            "n_calibration": len(cal),
            "n_samples": sum(clusters[c]["n_samples"] for c in cids),
            "n_samples_calibration": sum(clusters[c]["n_samples"] for c in cal),
        }

    calibration.sort()
    heldout.sort()
    n_samples = sum(c["n_samples"] for c in clusters.values())
    n_samples_cal = sum(clusters[c]["n_samples"] for c in calibration)

    return {
        "schema": "hnav.crossep_split.v1",
        "dataset": str(DATASET.relative_to(REPO)).replace("\\", "/"),
        "seed": seed,
        "calib_frac_clusters": calib_frac,
        "unit": "context_id",
        "icc": ICC,
        "design_effect": DESIGN_EFFECT,
        "n_clusters": len(clusters),
        "n_clusters_calibration": len(calibration),
        "n_clusters_heldout": len(heldout),
        "n_samples": n_samples,
        "n_samples_calibration": n_samples_cal,
        "n_samples_heldout": n_samples - n_samples_cal,
        "per_category": per_category,
        "calibration": calibration,
        "heldout": heldout,
        "rationale": (
            "Cluster-level split: samples within a context share one memory "
            "bank and one system context (ICC 0.346, design effect 3.20), so "
            "sample-level splitting would leak. ~40% of clusters per "
            "context_category to calibration, mirroring the primary arena's "
            "sh_6k+sh_32k ratio; seeded shuffle per category; single-cluster "
            "categories go to held-out. Any CrossEp threshold may be fit on "
            "the calibration clusters ONLY."
        ),
    }


def load_split(path: Path = SPLIT_PATH) -> dict:
    """The frozen artifact. Raises if it has not been generated/committed."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run python hnav/labeling/crossep_split.py")
    return json.loads(path.read_text(encoding="utf-8"))


def split_of(context_id: str, split: dict) -> str:
    """``"calibration"`` / ``"heldout"`` / ``"unknown"`` for one cluster."""
    if context_id in split.get("_calib_set", ()):   # memoized by caller
        return "calibration"
    if context_id in split["calibration"]:
        return "calibration"
    if context_id in split["heldout"]:
        return "heldout"
    return "unknown"


def main() -> int:
    fresh = derive_split()
    if SPLIT_PATH.exists():
        committed = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if committed == fresh:
            print(f"OK: {SPLIT_PATH.name} matches a fresh derivation "
                  f"(seed={fresh['seed']}, "
                  f"{fresh['n_clusters_calibration']}/{fresh['n_clusters']} "
                  f"clusters calibration).")
            return 0
        print("MISMATCH between the committed artifact and a fresh derivation.")
        print("If the dataset or the constants changed DELIBERATELY, delete "
              f"{SPLIT_PATH.name} and re-run; otherwise investigate.")
        return 1
    SPLIT_PATH.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SPLIT_PATH}")
    print(f"  clusters : {fresh['n_clusters_calibration']} calibration / "
          f"{fresh['n_clusters_heldout']} held-out (of {fresh['n_clusters']})")
    print(f"  samples  : {fresh['n_samples_calibration']} / "
          f"{fresh['n_samples_heldout']} (of {fresh['n_samples']})")
    for cat, row in fresh["per_category"].items():
        print(f"  {cat:<35} {row['n_calibration']:>3}/{row['n_clusters']:<3} clusters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
