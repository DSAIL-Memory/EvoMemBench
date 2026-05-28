#!/usr/bin/env bash
# Smoke wrapper: ReasoningBank backend, 8k context, 1 sample/subset.
# Usage:
#   bash scripts/run_eval_reasoning_bank_inepisode_fc_8k_lc_mini.sh [extra flags forwarded]
exec "$(dirname "${BASH_SOURCE[0]}")/run_eval_reasoning_bank_inepisode_fc_8k_lc.sh" \
    --limit-per-subset 1 \
    --workers 1 \
    --run-name reasoning_bank_inepisode_fc_8k_lc_mini \
    "$@"
