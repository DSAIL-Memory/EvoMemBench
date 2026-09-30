"""Experiment 1 — do the difference-vector observations survive null controls?

The earlier scratchpad analysis reported striking raw numbers (same-transition
cross-subject d_hat cosine ≈ 0.86). Raw numbers can be inflated by embedding
anisotropy: if all embeddings share dominant common directions, all difference
vectors do too. This runner measures the *gap* between genuine statistics and
matched nulls, in all three spaces:

  genuine   same-transition different-subject pairwise cos(d_hat_i, d_hat_j)
            same-relation different-transition pairwise cos
  nulls     A  shuffled-transition: within each relation, permute which later
               fact each earlier fact is paired with, recompute d_hat
            B  across-relation pairwise cos
            C  random unit vectors, same count and dim
  extras    global mean-direction norm vs the 1/sqrt(n) random null

Bootstrap CIs (resampled pairs) on the genuine statistics; permutation
p-values against the shuffle distribution. Everything stratified by split.

Usage:  python -m hnav.geometry_filter.run_nulls
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import data
from .metrics import perm_pvalue

N_SHUFFLE = 50
N_BOOT = 200


def _masked_mean(G: np.ndarray, mask: np.ndarray) -> float:
    iu = np.triu_indices_from(G, k=1)
    m = mask[iu]
    return float(G[iu][m].mean()) if m.any() else float("nan")


def _ids(values: list) -> np.ndarray:
    table: dict = {}
    return np.array([table.setdefault(v, len(table)) for v in values])


def _stats_for(D_hat: np.ndarray, rels: list, subjs: list, trans: list) -> dict:
    G = D_hat @ D_hat.T
    n = len(rels)
    rel_eq = np.equal.outer(_ids(rels), _ids(rels))
    sub_eq = np.equal.outer(_ids(subjs), _ids(subjs))
    tra_eq = np.equal.outer(_ids(trans), _ids(trans))
    same_tr = rel_eq & tra_eq & ~sub_eq
    same_rel = rel_eq & ~tra_eq
    diff_rel = ~rel_eq
    return {
        "n_vectors": n,
        "same_transition_diff_subject": _masked_mean(G, same_tr),
        "same_relation_diff_transition": _masked_mean(G, same_rel),
        "across_relation": _masked_mean(G, diff_rel),
        "across_relation_abs": _masked_mean(np.abs(G), diff_rel),
        "global_mean_direction_norm": float(np.linalg.norm(D_hat.mean(axis=0))),
    }


def run() -> dict:
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    rng = np.random.default_rng(data.SEED)

    gold = [r for r in records if r["gold_update"]]
    out = {"provenance": data.provenance(experiment="null_baselines",
                                         n_shuffle=N_SHUFFLE, n_boot=N_BOOT),
           "splits": {}}

    for split in ("calibration", "confirmatory"):
        srecs = [r for r in gold if r["split"] == split]
        pv = data.PairView(srecs, index)
        rels = [data.relation_of(r) for r in srecs]
        subjs = [r["parser"]["fact_a_parsed"]["subject"] for r in srecs]
        trans = [data.transition_key(r) for r in srecs]
        blob: dict = {"n_pairs": len(srecs), "spaces": {}}

        for name, V in spaces.items():
            D = pv.diff(V, normalize=True, oriented=True)
            genuine = _stats_for(D, rels, subjs, trans)

            # bootstrap CI on the two headline genuine statistics
            boots = {"same_transition_diff_subject": [],
                     "same_relation_diff_transition": []}
            for _ in range(N_BOOT):
                bi = rng.integers(0, len(srecs), len(srecs))
                s = _stats_for(D[bi], [rels[i] for i in bi],
                               [subjs[i] for i in bi], [trans[i] for i in bi])
                for k in boots:
                    boots[k].append(s[k])
            genuine["ci95"] = {k: [float(np.quantile(v, 0.025)),
                                   float(np.quantile(v, 0.975))]
                               for k, v in boots.items()}

            # shuffled-transition null: permute later facts within relation
            by_rel = defaultdict(list)
            for i, r in enumerate(rels):
                by_rel[r].append(i)
            null_same_rel, null_same_tr, null_mean_norm = [], [], []
            for _ in range(N_SHUFFLE):
                ib = pv.ib.copy()
                for idx in by_rel.values():
                    idx = np.array(idx)
                    ib[idx] = ib[rng.permutation(idx)]
                Dn = V[ib] - V[pv.ia]
                Dn /= np.maximum(np.linalg.norm(Dn, axis=1, keepdims=True), 1e-12)
                s = _stats_for(Dn, rels, subjs, trans)
                null_same_rel.append(s["same_relation_diff_transition"])
                null_same_tr.append(s["same_transition_diff_subject"])
                null_mean_norm.append(s["global_mean_direction_norm"])

            # random unit-vector control
            R = rng.standard_normal(D.shape)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            random_ctrl = _stats_for(R, rels, subjs, trans)

            blob["spaces"][name] = {
                "genuine": genuine,
                "shuffled_transition_null": {
                    "same_relation_diff_transition_mean": float(np.mean(null_same_rel)),
                    "same_relation_diff_transition_sd": float(np.std(null_same_rel)),
                    "same_transition_diff_subject_mean": float(np.mean(null_same_tr)),
                    "global_mean_direction_norm_mean": float(np.mean(null_mean_norm)),
                    "p_same_relation": perm_pvalue(
                        genuine["same_relation_diff_transition"], null_same_rel),
                    "p_same_transition": perm_pvalue(
                        genuine["same_transition_diff_subject"], null_same_tr),
                },
                "random_unit_control": random_ctrl,
            }
        out["splits"][split] = blob

    data.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = data.OUT_DIR / "null_baselines.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written:", dst)
    return out


if __name__ == "__main__":
    r = run()
    for split, blob in r["splits"].items():
        for space, s in blob["spaces"].items():
            g, n = s["genuine"], s["shuffled_transition_null"]
            print(f"{split:13s} {space:9s} same-trans {g['same_transition_diff_subject']:+.3f}"
                  f" (shuffle {n['same_transition_diff_subject_mean']:+.3f})"
                  f"  same-rel {g['same_relation_diff_transition']:+.3f}"
                  f" (shuffle {n['same_relation_diff_transition_mean']:+.3f},"
                  f" p={n['p_same_relation']:.3f})")
