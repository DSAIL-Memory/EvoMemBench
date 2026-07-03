#!/usr/bin/env bash
# Run all 5 context-budget variants of the qwen3_embedding in-episode evaluator in sequence.
#
# Usage:
#   bash scripts/run_eval_qwen3_embedding_inepisode_fc_all_lc.sh \
#     [--limit-per-subset N]   # applied to every sub-run
#     [--workers N]
#     [--keep-going]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1"); shift
done

BUDGETS=(8000 16000 32000 64000 128000)
LABELS=(8k   16k   32k   64k   128k)

overall_failed=0

for i in "${!BUDGETS[@]}"; do
    budget="${BUDGETS[$i]}"
    label="${LABELS[$i]}"
    script="${SCRIPT_DIR}/run_eval_qwen3_embedding_inepisode_fc_${label}_lc.sh"

    echo ""
    echo "================================================================"
    echo "  [$((i+1))/${#BUDGETS[@]}]  qwen3_embedding  context-budget=${budget} (${label})"
    echo "================================================================"

    if bash "$script" "${EXTRA_ARGS[@]}"; then
        echo "[OK] qwen3_embedding ${label} completed successfully."
    else
        rc=$?
        echo "[WARN] qwen3_embedding ${label} exited with code $rc - continuing to next budget."
        overall_failed=$((overall_failed + 1))
    fi
done

echo ""
echo "================================================================"
echo "  qwen3_embedding all-budgets sweep done. Failed sub-runs: ${overall_failed}/${#BUDGETS[@]}"
echo "================================================================"
exit "$overall_failed"
