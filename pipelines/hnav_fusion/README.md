# hnav_fusion — CES + ABTT-cosine logistic fusion as the identity screen

Experiment 5 measured CES and ABTT-cosine to be *tail-complementary* (CES owns
the extreme tail on seen transitions, TPR 0.598 @ FPR 1e-4; ABTT-cosine owns
unseen, 0.404). This arm fuses them: a balanced logistic over the
z-standardized pair scores, fit on calibration gold vs hard negatives, frozen
in `stage0_results/geometry_filter/fusion_screen.json` (fingerprint
`2ffb0c85…`, which itself pins the CES artifact `34e3abc1…` and the ABTT
whitening `3fdacc1f…`). Held-out pair level it dominates both components:
hard-task TPR@1e-4 0.748, transition-disjoint 0.536.

Screen: raw cosine ≥ 0.80 frame (the CES validation frame — reuses the `_ces`
prepasses) + fusion logit > τ. Parser supplies relation identity inside the
CES component (partial parser removal, honestly named). Stage 2 NLI and
`detector_suppress` are byte-identical to every other arm.

Before the first run (once, on the box; `_ces` prepasses assumed present):

    python hnav/stage1/detector_gap.py --select --subsets sh_6k sh_32k \
           --page-source benchmark --pair-screen fusion \
           --fusion-artifact stage0_results/geometry_filter/fusion_screen.json \
           --cos-grid 0.80 --ces-grid 0 2 4 6 8
    # then pin the new operating point's sha256 into pipeline.json in the
    # same commit that adds the file.

Run:

    python pipelines/hnav_fusion/run.py --llm-model <served-name> [--llm-base-url URL] [--dry-run]

Everything else: ../README.md.
