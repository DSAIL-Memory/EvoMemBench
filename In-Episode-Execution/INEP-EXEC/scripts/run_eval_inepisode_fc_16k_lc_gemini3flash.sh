#!/usr/bin/env bash
# In-episode evaluator (FC mode, long-context, 16k context budget, gpt-5-mini).
exec "$(dirname "${BASH_SOURCE[0]}")/run_eval_inepisode_fc_128k_lc_gemini3flash.sh" \
    --context-budget 16000 \
    --run-name inepisode_fc_16k_lc_gemini3flash \
    "$@"
