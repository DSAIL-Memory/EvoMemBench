#!/usr/bin/env bash
# In-episode evaluator (FC mode, long-context, 16k context budget).
# Usage:
#   bash scripts/run_eval_inepisode_fc_16k_lc.sh [--workers N] [--limit-per-subset N] ...
exec "$(dirname "${BASH_SOURCE[0]}")/run_eval_inepisode_fc_128k_lc.sh" \
    --context-budget 16000 \
    --run-name inepisode_fc_16k_lc \
    "$@"
