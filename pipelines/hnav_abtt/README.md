# hnav_abtt — ABTT whitening before the cosine screen

Facts are whitened (mean-centred, top-128 principal directions removed,
renormalised) with the frozen calibration-split artifact
`stage0_results/abtt/abtt_whitening_D128.json` (fingerprint `3fdacc1f…`)
before the pair screen; thresholds re-selected in that space
(cos_pair 0.30, r_min 0.954). Scope is `pairs`: retrieval stays raw, only the
fact-fact comparison is whitened. Committed reference result: sh_64k
conflicted 37/66 — exactly equal to hnav_raw (net 0, CI [0,0]); the two
pipelines exist so that equality can be re-tested on other answering models.

    python pipelines/hnav_abtt/run.py --llm-model <served-name> [--llm-base-url URL] [--dry-run]

Everything else: ../README.md.
