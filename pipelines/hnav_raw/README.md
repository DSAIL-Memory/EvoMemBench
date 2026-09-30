# hnav_raw — the shipped configuration

Raw-cosine geometry; operating point `stage0_results/stage1_operating_point.json`
(cos_pair 0.90, r_min 0.44, pair_filter, NLI 0.90), sha256-pinned in
`pipeline.json`. Committed reference result (Qwen3-4B-Instruct-2507, frozen
:8003): sh_64k conflicted 17/66 native → 37/66.

    python pipelines/hnav_raw/run.py --llm-model <served-name> [--llm-base-url URL] [--dry-run]

Everything else: ../README.md.
