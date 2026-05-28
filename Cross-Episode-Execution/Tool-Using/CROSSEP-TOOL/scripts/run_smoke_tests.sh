#!/usr/bin/env bash
# Smoke-test wrapper: runs all four CROSSEP-TOOL eval scripts serially with a
# small --limit-per-subset cap, then prints a PASS/FAIL summary.
#
# Usage:
#   bash scripts/run_smoke_tests.sh \
#     [--limit N]          # samples per subset per backend (default: 2)
#     [--only LIST]        # comma-separated subset of {mem0,ace,reasoning_bank,no_memory}
#     [--skip LIST]        # comma-separated backends to exclude
#     [--run-name NAME]    # prefix for child --run-name (default: smoke_<TS>)
#     [--keep-going]       # continue running later backends if an earlier one fails
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
LIMIT=2
ONLY_RAW=""
SKIP_RAW=""
RUN_PREFIX=""
KEEP_GOING=false

# ── Flag parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)      LIMIT="$2";      shift 2 ;;
        --only)       ONLY_RAW="$2";   shift 2 ;;
        --skip)       SKIP_RAW="$2";   shift 2 ;;
        --run-name)   RUN_PREFIX="$2"; shift 2 ;;
        --keep-going) KEEP_GOING=true; shift   ;;
        *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Repo root ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ── Backend list and validation ───────────────────────────────────────────────
ALL_BACKENDS=(mem0 ace reasoning_bank no_memory bm25 qwen3_embedding graphrag)

_in_comma_list() {
    local needle="$1" list="$2"
    local IFS=','
    for item in $list; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

_validate_list() {
    local list="$1" flag="$2"
    local IFS=','
    for item in $list; do
        local found=false
        for b in "${ALL_BACKENDS[@]}"; do [[ "$b" == "$item" ]] && found=true && break; done
        if [[ "$found" == "false" ]]; then
            echo "[ERROR] Unknown backend '$item' in $flag. Valid: ${ALL_BACKENDS[*]}"
            exit 1
        fi
    done
}

[[ -n "$ONLY_RAW" ]] && _validate_list "$ONLY_RAW" "--only"
[[ -n "$SKIP_RAW" ]] && _validate_list "$SKIP_RAW" "--skip"

# Build the ordered list to run
BACKENDS=()
for b in "${ALL_BACKENDS[@]}"; do
    if [[ -n "$ONLY_RAW" ]] && ! _in_comma_list "$b" "$ONLY_RAW"; then continue; fi
    if [[ -n "$SKIP_RAW" ]] && _in_comma_list "$b" "$SKIP_RAW"; then continue; fi
    BACKENDS+=("$b")
done

if [[ ${#BACKENDS[@]} -eq 0 ]]; then
    echo "[ERROR] No backends selected after applying --only / --skip."
    exit 1
fi

# ── Timestamp & shared log dir ────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
[[ -z "$RUN_PREFIX" ]] && RUN_PREFIX="smoke_${TS}"
SMOKE_LOG_DIR="cross_episode_results/runs/_smoke_${RUN_PREFIX}"
mkdir -p "${SMOKE_LOG_DIR}"
WRAPPER_LOG="${SMOKE_LOG_DIR}/wrapper.log"

# ── Backend → script + flags mapping ─────────────────────────────────────────
backend_script() {
    case "$1" in
        mem0)           echo "scripts/run_eval_mem0_inenv_allcrossenv_fc.sh" ;;
        ace)            echo "scripts/run_eval_ace_inenv_allcrossenv_fc.sh" ;;
        reasoning_bank) echo "scripts/run_eval_reasoning_bank_inenv_allcrossenv_fc.sh" ;;
        no_memory)      echo "scripts/run_eval_no_memory_inenv_fc.sh" ;;
        bm25)           echo "scripts/run_eval_bm25_inenv_allcrossenv_fc.sh" ;;
        qwen3_embedding) echo "scripts/run_eval_qwen3_embedding_inenv_allcrossenv_fc.sh" ;;
        graphrag)       echo "scripts/run_eval_graphrag_inenv_allcrossenv_fc.sh" ;;
    esac
}

backend_extra_flags() {
    # no_memory uses a different flag name (--workers) and has no Phase 2
    case "$1" in
        no_memory) echo "" ;;
        *)         echo "" ;;
    esac
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  BFCL Cross-Episode  — smoke test wrapper"
echo "  Backends (serial):   ${BACKENDS[*]}"
echo "  Limit/subset:        ${LIMIT}"
echo "  Run prefix:          ${RUN_PREFIX}"
echo "  Keep-going:          ${KEEP_GOING}"
echo "  Smoke log dir:       ${SMOKE_LOG_DIR}"
echo "================================================================"
echo "" | tee -a "${WRAPPER_LOG}"

# ── Per-backend tracking ──────────────────────────────────────────────────────
declare -A STATUS_CODE    # exit code
declare -A STATUS_SECS    # elapsed seconds
STATUS_ORDER=()           # ordered list for summary (same order as BACKENDS)

aborted=()  # backends that were skipped due to an earlier failure

run_backend() {
    local backend="$1"
    local child_run_name
    case "$backend" in
        mem0)           child_run_name="smoke_mem0_${TS}" ;;
        ace)            child_run_name="smoke_ace_${TS}" ;;
        reasoning_bank) child_run_name="smoke_rb_${TS}" ;;
        no_memory)      child_run_name="smoke_nomem_${TS}" ;;
        bm25)           child_run_name="smoke_bm25_${TS}" ;;
        qwen3_embedding) child_run_name="smoke_qwen3_${TS}" ;;
        graphrag)       child_run_name="smoke_graph_${TS}" ;;
    esac

    local script
    script="$(backend_script "$backend")"
    local backend_log="${SMOKE_LOG_DIR}/${backend}.log"

    echo "================================================================" | tee -a "${WRAPPER_LOG}"
    echo "  smoke test: ${backend}  (limit=${LIMIT})" | tee -a "${WRAPPER_LOG}"
    echo "  script:     ${script}" | tee -a "${WRAPPER_LOG}"
    echo "  log:        ${backend_log}" | tee -a "${WRAPPER_LOG}"
    echo "================================================================" | tee -a "${WRAPPER_LOG}"

    local t_start
    t_start=$(date +%s)

    local exit_code=0
    bash "${script}" \
        --limit-per-subset "${LIMIT}" \
        --run-name         "${child_run_name}" \
        2>&1 | tee -a "${WRAPPER_LOG}" "${backend_log}" || exit_code=$?

    local t_end elapsed
    t_end=$(date +%s)
    elapsed=$(( t_end - t_start ))

    STATUS_CODE["$backend"]="$exit_code"
    STATUS_SECS["$backend"]="$elapsed"
    STATUS_ORDER+=("$backend")

    local mm ss label
    mm=$(( elapsed / 60 ))
    ss=$(( elapsed % 60 ))
    if [[ $exit_code -eq 0 ]]; then
        label="PASS"
    else
        label="FAIL"
    fi

    printf "[%s] %s  in %02dm%02ds" "$backend" "$label" "$mm" "$ss" \
        | tee -a "${WRAPPER_LOG}"
    [[ $exit_code -ne 0 ]] && printf "  exit=%d" "$exit_code" | tee -a "${WRAPPER_LOG}"
    echo "" | tee -a "${WRAPPER_LOG}"

    return "$exit_code"
}

# ── Serial run loop ───────────────────────────────────────────────────────────
abort=false
for backend in "${BACKENDS[@]}"; do
    if [[ "$abort" == "true" ]]; then
        aborted+=("$backend")
        continue
    fi

    if ! run_backend "$backend"; then
        if [[ "$KEEP_GOING" != "true" ]]; then
            abort=true
        fi
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "${WRAPPER_LOG}"
echo "================================================================" | tee -a "${WRAPPER_LOG}"
echo "  Smoke test summary" | tee -a "${WRAPPER_LOG}"
echo "----------------------------------------------------------------" | tee -a "${WRAPPER_LOG}"

n_fail=0
for b in "${STATUS_ORDER[@]}"; do
    ec="${STATUS_CODE[$b]}"
    secs="${STATUS_SECS[$b]}"
    mm=$(( secs / 60 ))
    ss=$(( secs % 60 ))
    if [[ $ec -eq 0 ]]; then
        printf "  PASS  %-18s  %02dm%02ds\n" "$b" "$mm" "$ss" | tee -a "${WRAPPER_LOG}"
    else
        printf "  FAIL  %-18s  %02dm%02ds  exit=%d  log: %s\n" \
            "$b" "$mm" "$ss" "$ec" "${SMOKE_LOG_DIR}/${b}.log" | tee -a "${WRAPPER_LOG}"
        (( n_fail++ )) || true
    fi
done

for b in "${aborted[@]}"; do
    printf "  SKIP  %-18s  (aborted — pass --keep-going to continue)\n" "$b" \
        | tee -a "${WRAPPER_LOG}"
    (( n_fail++ )) || true
done

echo "----------------------------------------------------------------" | tee -a "${WRAPPER_LOG}"
echo "  Smoke logs: ${SMOKE_LOG_DIR}" | tee -a "${WRAPPER_LOG}"
echo "================================================================" | tee -a "${WRAPPER_LOG}"

exit "$n_fail"
