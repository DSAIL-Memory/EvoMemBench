#!/usr/bin/env bash
# Smoke wrapper: ACE backend, 64k context, 1 sample/subset.
# Usage:
#   bash scripts/run_eval_ace_inepisode_fc_64k_lc_mini.sh [extra flags forwarded]
exec "$(dirname "${BASH_SOURCE[0]}")/run_eval_ace_inepisode_fc_64k_lc.sh" \
    --limit-per-subset 1 \
    --workers 1 \
    --run-name ace_inepisode_fc_64k_lc_mini \
    "$@"
