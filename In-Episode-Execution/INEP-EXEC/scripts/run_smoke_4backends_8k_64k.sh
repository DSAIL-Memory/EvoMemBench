#!/usr/bin/env bash
# Smoke orchestrator: mem0 / ace / reasoning_bank / no-memory @ 8k + 64k.
# 4 groups, each runs one backend's 8k+64k mini in parallel (peak 2 processes).
#
# Groups:
#   G1 no-memory:     nm_8k     ∥ nm_64k
#   G2 mem0:          mem0_8k   ∥ mem0_64k
#   G3 ACE:           ace_8k    ∥ ace_64k
#   G4 ReasoningBank: rb_8k     ∥ rb_64k
#
# All extra flags forwarded to every mini script unchanged (e.g. --workers N).
#
# Usage:
#   bash scripts/run_smoke_4backends_8k_64k.sh [extra flags]
#
# Logs: in_episode_results/runs/smoke_<TS>/logs/<label>.log
set -uo pipefail   # no -e — failures must not abort the loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Log directory ─────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/../in_episode_results/runs/smoke_${TS}/logs"
mkdir -p "${LOG_DIR}"

EXTRA=( "$@" )

# ── Banner ────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  Smoke Orchestrator  [mem0 / ace / reasoning_bank / no-memory]"
echo "  Context budgets:   8k + 64k"
echo "  Parallelism:       2 (same backend, both contexts)"
echo "  Samples/subset:    1  (--limit-per-subset 1)"
echo "  Logs: ${LOG_DIR}"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "  Extra flags: ${EXTRA[*]}"
echo "════════════════════════════════════════════════════════════════"

# ── Tracking ──────────────────────────────────────────────────────────────────
passed=()
failed=()

# ── run_pair <group-label> <label1> <script1> <label2> <script2> ──────────────
# Runs two mini scripts in parallel; waits for both; updates passed/failed.
run_pair() {
    local glabel="$1" l1="$2" s1="$3" l2="$4" s2="$5"
    local log1="${LOG_DIR}/${l1}.log"
    local log2="${LOG_DIR}/${l2}.log"

    echo ""
    echo "┌─── ${glabel} ───────────────────────────────────────"
    echo "│  ${l1}  →  $(basename "${log1}")"
    echo "│  ${l2}  →  $(basename "${log2}")"
    echo "└────────────────────────────────────────────────────"

    set +e
    bash "${SCRIPT_DIR}/${s1}" "${EXTRA[@]}" >"${log1}" 2>&1 &
    local p1=$!
    bash "${SCRIPT_DIR}/${s2}" "${EXTRA[@]}" >"${log2}" 2>&1 &
    local p2=$!

    if wait "${p1}"; then
        echo "  [done]   ${l1}"
        passed+=("${l1}")
    else
        echo "  [FAILED] ${l1} — see ${log1}"
        failed+=("${l1}")
    fi
    if wait "${p2}"; then
        echo "  [done]   ${l2}"
        passed+=("${l2}")
    else
        echo "  [FAILED] ${l2} — see ${log2}"
        failed+=("${l2}")
    fi
    set -e
}

# ── Groups ────────────────────────────────────────────────────────────────────
run_pair "G1 no-memory" \
    nm_8k  "run_eval_inepisode_fc_8k_lc_mini.sh" \
    nm_64k "run_eval_inepisode_fc_64k_lc_mini.sh"

run_pair "G2 mem0" \
    mem0_8k  "run_eval_mem0_inepisode_fc_8k_lc_mini.sh" \
    mem0_64k "run_eval_mem0_inepisode_fc_64k_lc_mini.sh"

run_pair "G3 ACE" \
    ace_8k  "run_eval_ace_inepisode_fc_8k_lc_mini.sh" \
    ace_64k "run_eval_ace_inepisode_fc_64k_lc_mini.sh"

run_pair "G4 ReasoningBank" \
    rb_8k  "run_eval_reasoning_bank_inepisode_fc_8k_lc_mini.sh" \
    rb_64k "run_eval_reasoning_bank_inepisode_fc_64k_lc_mini.sh"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SMOKE SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo "  Passed (${#passed[@]}):"
for l in "${passed[@]}"; do echo "    ✓ ${l}"; done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "  Failed (${#failed[@]}):"
    for l in "${failed[@]}"; do echo "    ✗ ${l}"; done
fi
echo "  Logs: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════════"

exit "${#failed[@]}"
