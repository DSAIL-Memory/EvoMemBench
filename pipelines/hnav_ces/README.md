# hnav_ces — the contrastive-edit-subspace identity screen

The parser's `same_key` pair filter (subject identity from string equality) is
replaced by the frozen CES artifact
`stage0_results/geometry_filter/ces_subspaces_k20.json` (fingerprint
`34e3abc1…`, k=20, fit on the sh_6k+sh_32k gold conflict dataset only): a pair
survives when `‖U_obj_r^T d̂‖² − ‖U_subj_r^T d̂‖² > τ` — "does the difference
vector look like a value replacement or like a change of subject?". The parser
still supplies the RELATION template (honest naming: this is a partial parser
removal — the parser-free global variant failed its pair-level gate at 0.8725
AUROC on 2026-08-27 and was not promoted). The cosine screen is the CES
validation frame, cos ≥ 0.80 raw — the gold dataset says nothing below it.

Pair-level evidence: `stage0_results/geometry_filter/REPORT.md` §7
(held-out hard-task AUROC 0.981; 0.980 win rate on the comparisons cosine
orders backwards). Stage 2 (bidirectional NLI, threshold 0.90) and the
`detector_suppress` mechanism are byte-identical to the committed arms.

Before the first run (once, on the box):

    python hnav/stage1/calibrate_read_policy.py --prepass --subsets sh_6k sh_32k \
           --page-source benchmark --cos-loose 0.80 --prepass-tag _ces
    python hnav/stage1/confirmatory_prepass.py --subset sh_64k \
           --cos-loose 0.80 --prepass-tag _ces
    python hnav/stage1/detector_gap.py --select --subsets sh_6k sh_32k \
           --pair-screen ces --ces-artifact stage0_results/geometry_filter/ces_subspaces_k20.json \
           --cos-grid 0.80
    # then pin the new operating point: put its sha256 into pipeline.json
    # (operating_point_sha256) in the same commit that adds the file.

Run:

    python pipelines/hnav_ces/run.py --llm-model <served-name> [--llm-base-url URL] [--dry-run]

Everything else: ../README.md.
