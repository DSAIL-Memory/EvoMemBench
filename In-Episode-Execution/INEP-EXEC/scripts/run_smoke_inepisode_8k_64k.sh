#!/usr/bin/env bash
# Smoke runner: exercises bm25 / graphrag / memagent / qwen3_embedding
# at both 8k and 64k context budgets, sequentially, 1 sample per subset.
#
# Total: 8 invocations × 4 subsets × 1 sample = 32 samples.
# A failed invocation is recorded but does NOT abort the loop.
# Exit code = number of failed invocations.
#
# Usage:
#   bash scripts/run_smoke_inepisode_8k_64k.sh
set -uo pipefail   # intentionally no -e — failures must not abort the loop

ENV_NAME="INEP-EXEC"

# ── Locate repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/smoke_logs"
mkdir -p "${LOG_DIR}"

# ── Activate conda env ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Preflight: .env ──────────────────────────────────────────────────────────
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    echo "[smoke] ERROR: .env not found at ${REPO_ROOT}/.env — aborting." >&2
    exit 1
fi

# ── Preflight: production parent scripts ─────────────────────────────────────
for parent in \
    run_eval_bm25_inepisode_fc_8k_lc.sh \
    run_eval_bm25_inepisode_fc_64k_lc.sh \
    run_eval_graphrag_inepisode_fc_8k_lc.sh \
    run_eval_graphrag_inepisode_fc_64k_lc.sh \
    run_eval_memagent_inepisode_fc_8k_lc.sh \
    run_eval_memagent_inepisode_fc_64k_lc.sh \
    run_eval_qwen3_embedding_inepisode_fc_8k_lc.sh \
    run_eval_qwen3_embedding_inepisode_fc_64k_lc.sh; do
    if [[ ! -f "${SCRIPT_DIR}/${parent}" ]]; then
        echo "[smoke] ERROR: parent script not found: ${SCRIPT_DIR}/${parent}" >&2
        exit 1
    fi
done

# ── Run order: 8k first, then 64k ────────────────────────────────────────────
RUNS=(
    "bm25:8k"
    "graphrag:8k"
    "memagent:8k"
    "qwen3_embedding:8k"
    "bm25:64k"
    "graphrag:64k"
    "memagent:64k"
    "qwen3_embedding:64k"
)

failed=()
passed=()

echo "================================================================"
echo "  BFCL Smoke Test  [8k + 64k | sequential | 1 sample/subset]"
echo "  Repo:   ${REPO_ROOT}"
echo "  Logs:   ${LOG_DIR}/"
echo "  Total runs: ${#RUNS[@]}"
echo "================================================================"
echo ""

for entry in "${RUNS[@]}"; do
    backend="${entry%%:*}"
    ctx="${entry##*:}"
    label="${backend}-${ctx}"
    log="${LOG_DIR}/${backend}_${ctx}.log"
    parent="run_eval_${backend}_inepisode_fc_${ctx}_lc.sh"
    run_name="smoke_${backend}_${ctx}"

    echo "[smoke] ── ${label} ────────────────────────────────────────────"
    echo "[smoke]   script: scripts/${parent}"
    echo "[smoke]   log:    smoke_logs/${backend}_${ctx}.log"

    set +e
    bash "${SCRIPT_DIR}/${parent}" \
        --limit-per-subset 1 \
        --workers 1 \
        --run-name "${run_name}" \
        >"${log}" 2>&1
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
        echo "[smoke]   ✓ ${label} PASSED"
        passed+=("${label}")
    else
        echo "[smoke]   ✗ ${label} FAILED (rc=${rc})  →  ${log}"
        failed+=("${label}")
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  SMOKE SUMMARY"
echo "================================================================"
echo "  Passed (${#passed[@]}):"
for l in "${passed[@]}"; do echo "    ✓ ${l}"; done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "  Failed (${#failed[@]}):"
    for l in "${failed[@]}"; do echo "    ✗ ${l}"; done
fi
echo "  Logs: ${LOG_DIR}/"
echo "================================================================"

exit "${#failed[@]}"
