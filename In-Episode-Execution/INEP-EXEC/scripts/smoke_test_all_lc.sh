#!/usr/bin/env bash
# Smoke test: verify all 6 *_all_lc*.sh chains on 8k + 64k budgets.
#
# Strategy:
#   Phase 0  — bash -n syntax-check the 6 all_lc wrappers (< 1 s).
#   Phase 1  — run 12 sub-scripts (6 baselines × {8k,64k}) with
#              --limit-per-subset 1 --keep-going.
#              Baselines are paired into 3 batches; within each batch
#              the two tracks run in parallel; batches are sequential.
#   Phase 2  — print a 6×2 PASS/FAIL matrix.
#
# Batch pairing (mixed API families to spread rate-limit load):
#   Batch 1: amem (DeepSeek)   ∥ gpt5mini (OpenAI)
#   Batch 2: memos (DeepSeek)  ∥ gemini3flash (Gemini)
#   Batch 3: memobrain + memoryos (both DeepSeek — unavoidable, placed last)
#
# Usage:
#   bash scripts/smoke_test_all_lc.sh [--workers N]
#
# Outputs land under:
#   in_episode_results/smoke_runs/all_lc_<TS>/
#     orchestrator.log   — full combined log
#     logs/              — per-subrun logs (<baseline>_<label>.log)
#     matrix.txt         — 6×2 PASS/FAIL table
#     results.tsv        — raw (baseline, label, rc) rows
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
WORKERS=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Create smoke output dir ───────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
SMOKE_DIR="${SCRIPT_DIR}/../in_episode_results/smoke_runs/all_lc_${TS}"
mkdir -p "${SMOKE_DIR}/logs"

RESULTS_TSV="${SMOKE_DIR}/results.tsv"
printf 'baseline\tlabel\trc\n' > "$RESULTS_TSV"

# Tee all subsequent output to orchestrator.log
exec > >(tee "${SMOKE_DIR}/orchestrator.log") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  smoke_test_all_lc.sh"
echo "  Timestamp  : ${TS}"
echo "  Smoke dir  : ${SMOKE_DIR}"
echo "  Workers    : ${WORKERS} per subset"
echo "  Budgets    : 8k, 64k"
echo "  Batches    : amem+gpt5mini | memos+gemini3flash | memobrain+memoryos"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 0: bash -n on the 6 wrappers ───────────────────────────────────────
echo ""
echo "── Phase 0: syntax check (bash -n) ─────────────────────────────────────"

SYNTAX_FAILED=()
for f in \
    run_eval_amem_inepisode_fc_all_lc.sh \
    run_eval_memobrain_inepisode_fc_all_lc.sh \
    run_eval_memos_inepisode_fc_all_lc.sh \
    run_eval_memoryos_inepisode_fc_all_lc.sh \
    run_eval_inepisode_fc_all_lc_gpt5mini.sh \
    run_eval_inepisode_fc_all_lc_gemini3flash.sh; do
    if bash -n "${SCRIPT_DIR}/$f" 2>/dev/null; then
        echo "  [OK]   $f"
    else
        echo "  [FAIL] $f — syntax error detected"
        SYNTAX_FAILED+=("$f")
    fi
done

if [[ ${#SYNTAX_FAILED[@]} -eq 0 ]]; then
    echo "Phase 0: all 6 wrappers pass syntax check."
else
    echo "Phase 0: ${#SYNTAX_FAILED[@]} wrapper(s) failed syntax check: ${SYNTAX_FAILED[*]}"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# Resolve (baseline, label) → sub-script filename
resolve_sub_script() {
    local base="$1" label="$2"
    case "$base" in
        amem|memobrain|memos|memoryos)
            echo "run_eval_${base}_inepisode_fc_${label}_lc.sh" ;;
        gpt5mini|gemini3flash)
            echo "run_eval_inepisode_fc_${label}_lc_${base}.sh" ;;
        *) echo "[ERROR] Unknown baseline: $base" >&2; return 1 ;;
    esac
}

# Run one baseline track: 8k sub-script, then 64k sub-script (sequential).
# Writes per-run logs to SMOKE_DIR/logs/ and appends to results.tsv.
# Returns number of sub-script failures in this track.
run_track() {
    local baseline="$1"
    local track_failures=0
    for label in 8k 64k; do
        local sub_script log rc
        sub_script="$(resolve_sub_script "$baseline" "$label")"
        log="${SMOKE_DIR}/logs/${baseline}_${label}.log"
        rc=0

        echo "  [start] ${baseline} ${label}  →  ${sub_script}"
        bash "${SCRIPT_DIR}/${sub_script}" \
            --limit-per-subset 1 \
            --keep-going \
            --workers "${WORKERS}" \
            --run-name "smoke_${baseline}_${label}" \
            >"$log" 2>&1 || rc=$?

        if [[ $rc -eq 0 ]]; then
            echo "  [PASS]  ${baseline} ${label}"
        else
            echo "  [FAIL]  ${baseline} ${label}  rc=${rc}  log: ${log}"
            track_failures=$((track_failures + 1))
        fi
        # Append result (atomic for short writes on Linux)
        printf '%s\t%s\t%s\n' "$baseline" "$label" "$rc" >> "$RESULTS_TSV"
    done
    return "$track_failures"
}

# ── Phase 1: pre-flight — verify all 12 sub-scripts exist ────────────────────
echo ""
echo "── Phase 1: pre-flight check ────────────────────────────────────────────"

MISSING=()
for base in amem memos memobrain memoryos gpt5mini gemini3flash; do
    for label in 8k 64k; do
        sub="$(resolve_sub_script "$base" "$label")"
        [[ -f "${SCRIPT_DIR}/$sub" ]] || MISSING+=("$sub")
    done
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "[ERROR] Missing sub-scripts (aborting):"
    printf '  %s\n' "${MISSING[@]}"
    exit 1
fi
echo "  All 12 target sub-scripts present."

# ── Phase 1: 3 batches ────────────────────────────────────────────────────────
echo ""
echo "── Phase 1: end-to-end sub-script runs ──────────────────────────────────"

BATCHES=(
    "amem|gpt5mini"
    "memos|gemini3flash"
    "memobrain|memoryos"
)

PHASE1_FAILED=0
batch_num=0

for batch in "${BATCHES[@]}"; do
    batch_num=$((batch_num + 1))
    IFS='|' read -r base_a base_b <<<"$batch"

    echo ""
    echo "  ── Batch ${batch_num}/3: ${base_a} ∥ ${base_b}"

    run_track "$base_a" &
    pid_a=$!
    run_track "$base_b" &
    pid_b=$!

    batch_fail=0
    wait "$pid_a" || batch_fail=$((batch_fail + 1))
    wait "$pid_b" || batch_fail=$((batch_fail + 1))

    PHASE1_FAILED=$((PHASE1_FAILED + batch_fail))
    if [[ $batch_fail -eq 0 ]]; then
        echo "  ── Batch ${batch_num} done: both tracks PASS"
    else
        echo "  ── Batch ${batch_num} done: ${batch_fail} track(s) had failures"
    fi
done

# ── Phase 2: result matrix ────────────────────────────────────────────────────
echo ""
echo "── Phase 2: result matrix ────────────────────────────────────────────────"
echo ""

BASELINES=(amem memos memobrain memoryos gpt5mini gemini3flash)

{
    printf '%-18s  %-12s  %-12s\n' "baseline" "8k" "64k"
    printf '%-18s  %-12s  %-12s\n' "------------------" "------------" "------------"

    for base in "${BASELINES[@]}"; do
        cell_8k="?"
        cell_64k="?"
        while IFS=$'\t' read -r b l rc; do
            if [[ "$b" == "$base" && "$l" == "8k" ]]; then
                if [[ "$rc" == "0" ]]; then cell_8k="PASS"; else cell_8k="FAIL(rc=${rc})"; fi
            elif [[ "$b" == "$base" && "$l" == "64k" ]]; then
                if [[ "$rc" == "0" ]]; then cell_64k="PASS"; else cell_64k="FAIL(rc=${rc})"; fi
            fi
        done < <(tail -n +2 "$RESULTS_TSV")
        printf '%-18s  %-12s  %-12s\n' "$base" "$cell_8k" "$cell_64k"
    done
} | tee "${SMOKE_DIR}/matrix.txt"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Smoke complete."
echo "  Syntax failures : ${#SYNTAX_FAILED[@]}"
echo "  Run failures    : ${PHASE1_FAILED}"
echo "  Logs            : ${SMOKE_DIR}/logs/"
echo "  Matrix          : ${SMOKE_DIR}/matrix.txt"
echo "  Full log        : ${SMOKE_DIR}/orchestrator.log"
echo "════════════════════════════════════════════════════════════════"

exit $((${#SYNTAX_FAILED[@]} + PHASE1_FAILED))
