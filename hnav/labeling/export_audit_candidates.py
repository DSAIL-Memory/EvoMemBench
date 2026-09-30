#!/usr/bin/env python3
"""Full candidate set for the LLM semantic audit: all pairs with cos >= 0.80.

Builds, for sh_6k / sh_32k / sh_64k, every unordered fact pair whose cosine
similarity meets the threshold, using the EXACT campaign embeddings
(``Qwen/Qwen3-Embedding-4B``, float32, max_length 8192 — the namespace every
committed geometry number was computed in). The cache is loaded with the same
all-or-nothing rule as M7: a single missing vector aborts the run rather than
silently re-embedding into a different geometry.

Differences from the M7 pair space, on purpose:
  * ALL facts are included — also the ~0.5% the template parser cannot parse.
    M7's matrices dropped those; an audit hunting parser misses must not.
  * No eligibility filtering: parser-tagged conflicts, same-key duplicates,
    everything above the threshold is kept. Parser output rides along as
    METADATA only (``parser_tagged_conflict``, key/relation/subject/objects).
  * One flat threshold (0.80) for all subsets, not each subset's own conflict
    floor. The earlier ~75,486 figure used the per-subset floors (0.8339 /
    0.8339 / 0.8011) on parsed facts only, so THIS count is expected to be
    larger; the summary quantifies exactly where the difference comes from.

Output: JSONL (one canonical pair per line, serial_a < serial_b) plus a
summary JSON. Nothing is sent to any LLM here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.adapters.mab_adapter import explode_facts            # noqa: E402
from hnav.config import get_config                             # noqa: E402
from hnav.labeling.conflict_analysis import parse              # noqa: E402
from hnav.stage0.m7_delta_geometry import load_vectors         # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
OUT_DIR = REPO / "stage0_results" / "conflict_pairs"
SUBSETS = ("sh_6k", "sh_32k", "sh_64k")
# The committed M7 baseline these counts must be reconciled against:
# eligible (non-same-key, PARSED facts) pairs above each subset's own conflict
# floor, plus the tagged conflicts themselves.
# Floors at full precision from m7_delta_geometry.json. M7 computed its Gram
# in float32 while this exporter uses float64, so a handful of pairs sit within
# one float32 ULP of the floor and can flip across it; the reconciliation is
# therefore expected to land within a few pairs, not exactly.
M7_BASELINE = {"sh_6k": {"floor": 0.8338865637779236, "eligible_above": 306, "tagged": 160},
               "sh_32k": {"floor": 0.8338865637779236, "eligible_above": 6716, "tagged": 835},
               "sh_64k": {"floor": 0.8010916709899902, "eligible_above": 65782, "tagged": 1687}}


def describe(v: np.ndarray) -> dict:
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "mean": float(v.mean()), "max": float(v.max())}


def build_subset(name: str, context: str, cfg, thr: float):
    facts = explode_facts(context)                     # ALL facts, parsed or not
    serials = [n for n, _ in facts]
    texts = [t for _, t in facts]
    assert len(set(serials)) == len(serials), "duplicate serials"
    V, ns, rep = load_vectors(texts, cfg, embed=False)
    V = np.asarray(V, dtype=np.float64)
    nrm = np.linalg.norm(V, axis=1)
    assert abs(float(nrm.min()) - 1.0) < 1e-5 and abs(float(nrm.max()) - 1.0) < 1e-5, \
        "vectors are not unit-norm; wrong artifact"

    parsed = {}
    for s, t in facts:
        p = parse(t)
        parsed[s] = None if p is None else {"relation": p[0], "subject": p[1],
                                            "object": p[2]}

    gram = V @ V.T
    iu, ju = np.triu_indices(len(facts), 1)            # i<j: no self, no dupes
    cos = gram[iu, ju]
    keep = np.flatnonzero(cos >= thr)

    records = []
    n_tagged = 0
    for k in keep:
        i, j = int(iu[k]), int(ju[k])
        sa, sb = serials[i], serials[j]
        pa, pb = parsed[sa], parsed[sb]
        both = pa is not None and pb is not None
        same_key = both and (pa["relation"], pa["subject"]) == (pb["relation"], pb["subject"])
        tagged = bool(same_key and pa["object"] != pb["object"])
        n_tagged += tagged
        records.append({
            "pair_id": f"{name}:{sa}-{sb}",
            "subset": name,
            "fact_a_id": f"fact:{sa}", "fact_b_id": f"fact:{sb}",
            "fact_a": texts[i], "fact_b": texts[j],
            "cosine_similarity": float(cos[k]),
            "parser_tagged_conflict": tagged,
            "parser_metadata": {
                "fact_a_parsed": pa, "fact_b_parsed": pb,
                "both_parse": both, "same_key": bool(same_key),
                "same_relation": bool(both and pa["relation"] == pb["relation"]),
                "same_subject": bool(both and pa["subject"] == pb["subject"]),
                "same_object": bool(both and pa["object"] == pb["object"]),
                "superseding_serial": sb if tagged else None,   # later wins
            },
            "llm_audit_status": "pending",
        })
    records.sort(key=lambda r: -r["cosine_similarity"])

    n = len(facts)
    base = M7_BASELINE[name]
    summary = {
        "n_facts": n, "n_unparsed": sum(1 for v in parsed.values() if v is None),
        "n_possible_unordered_pairs": n * (n - 1) // 2,
        "n_selected": len(records),
        "n_selected_parser_tagged": n_tagged,
        "n_selected_untagged": len(records) - n_tagged,
        "selected_cosine": describe(cos[keep]) if keep.size else {},
        "reconciliation_vs_m7": {
            "m7_figure": base["eligible_above"] + base["tagged"],
            "m7_floor": base["floor"],
            "n_at_m7_floor_parsed_only": int(sum(
                1 for k in keep
                if cos[k] >= base["floor"]
                and parsed[serials[int(iu[k])]] is not None
                and parsed[serials[int(ju[k])]] is not None)),
            "n_added_by_flat_threshold": int(sum(1 for k in keep
                                                 if cos[k] < base["floor"])),
            "n_added_by_unparsed_facts": int(sum(
                1 for k in keep
                if parsed[serials[int(iu[k])]] is None
                or parsed[serials[int(ju[k])]] is None)),
        },
    }
    return records, summary, ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--out", default=str(OUT_DIR / "audit_candidates_cos080.jsonl"))
    args = ap.parse_args()

    cfg = get_config()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {it["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
             .replace("factconsolidation_", ""): it for it in data}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_summaries, namespace = {}, None
    n_total = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for s in SUBSETS:
            recs, summ, ns = build_subset(s, items[s]["context"], cfg,
                                          args.threshold)
            namespace = ns
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_summaries[s] = summ
            n_total += summ["n_selected"]
            print(f"{s:7s} facts={summ['n_facts']:5d} "
                  f"possible={summ['n_possible_unordered_pairs']:>10,d} "
                  f"selected={summ['n_selected']:>7,d} "
                  f"(tagged={summ['n_selected_parser_tagged']}, "
                  f"untagged={summ['n_selected_untagged']})")

    summary = {
        "what": f"all unordered fact pairs with cosine >= {args.threshold} — "
                f"candidate set for the LLM semantic audit",
        "threshold": args.threshold,
        "embedding": {"model": cfg.embed_model, "dtype": cfg.embed_dtype,
                      "max_length": cfg.embed_max_length,
                      "cache_namespace": namespace,
                      "cache_dir": str(cfg.emb_cache_dir),
                      "provenance": "the campaign cache pulled from ozonderlab2 "
                                    "2026-08-23; identical vectors to every "
                                    "committed geometry result"},
        "n_total_candidates": n_total,
        "output_jsonl": str(out),
        "subsets": all_summaries,
        "note": "parser output is METADATA only; nothing was filtered by it. "
                "Unparsed facts are INCLUDED (M7's pair space excluded them), "
                "and the flat threshold sits below the per-subset conflict "
                "floors M7 used — both differences vs the ~75,486 figure are "
                "quantified in reconciliation_vs_m7.",
    }
    sp = out.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                  encoding="utf-8")
    print(f"\ntotal candidates: {n_total:,}")
    print(f"wrote {out}\nwrote {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
