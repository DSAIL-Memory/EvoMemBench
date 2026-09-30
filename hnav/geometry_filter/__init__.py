"""Geometry-filter experiments on the gold conflict dataset.  [Audit series]

Offline tier: everything here may read gold labels and judge verdicts. Nothing
under ``hnav/core/`` or ``hnav/adapters/`` may import from this package.

The research question (2026-08-26): do sentence-embedding difference vectors
``d = v_b - v_a`` carry information about *which semantic slot changed*, beyond
what plain cosine similarity already provides — and can a lightweight
relation-conditioned detector (RCED / RCESP) separate gold conflicts from
cosine-matched hard negatives?

Inputs (all committed):
    stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz
    stage0_results/abtt/abtt_whitening_D128.json      (calibration-fit ABTT)
    hnav/_cache/emb/                                   (campaign embeddings)

Outputs land in ``stage0_results/geometry_filter/``.
"""
