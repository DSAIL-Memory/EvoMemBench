#!/usr/bin/env bash
# smoke_new_backends.sh — E2E smoke test for the 4 newly-migrated memory backends:
#   agent_kb, agent_workflow, lightweight, skillweaver
#
# Usage:
#   bash scripts/smoke_new_backends.sh [--static-only]
#
#   --static-only   Syntax + precondition checks only; no API calls.
#
# Each E2E run uses --max-samples 2 --workers 1 --eval-workers 1 --top-k 1 --skip-compare.
# Outputs are moved to outputs/_smoke_new/<backend>/ after each run.
# A Markdown report is written to outputs/_smoke_new/smoke_new_backends_<timestamp>.md.

set -euo pipefail

STATIC_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --static-only) STATIC_ONLY=true ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash scripts/smoke_new_backends.sh [--static-only]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SMOKE_LOG_DIR="outputs/_smoke_new"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${SMOKE_LOG_DIR}/smoke_new_backends_${TIMESTAMP}.md"
mkdir -p "${SMOKE_LOG_DIR}"

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

_pass() { PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|$1"); }
_fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|$1"); }
_skip() { SKIP_COUNT=$((SKIP_COUNT + 1)); RESULTS+=("SKIP|$1"); }
_log()  { echo "$*"; }

# ── Phase 1: Python syntax check ──────────────────────────────────────────────
_log ""
_log "=== Phase 1: Python syntax check ==="

declare -A BACKEND_SYNTAX_OK
for _pair in \
    "agent_kb:cl_bench_memory/agent_kb_memory.py" \
    "agent_workflow:cl_bench_memory/agent_workflow_memory.py" \
    "lightweight:cl_bench_memory/lightweight_memory.py" \
    "skillweaver:cl_bench_memory/skillweaver_memory.py"
do
    _name="${_pair%%:*}"
    _file="${_pair#*:}"
    if python -m py_compile "$_file" 2>/dev/null; then
        _log "  [ok]  $_name  ($_ file)"
        _pass "syntax:${_name}"
        BACKEND_SYNTAX_OK[$_name]=true
    else
        _log "  [ERR] $_name syntax error:"
        python -m py_compile "$_file" 2>&1 | sed 's/^/       /'
        _fail "syntax:${_name}"
        BACKEND_SYNTAX_OK[$_name]=false
    fi
done

# ── Phase 2: Precondition checks ───────────────────────────────────────────────
_log ""
_log "=== Phase 2: Precondition checks ==="

PRECOND_OK=true

_check() {
    local label="$1" ok="$2" hint="$3"
    if [[ "$ok" == true ]]; then
        _log "  [ok]  $label"
        _pass "precond:${label}"
    else
        _log "  [ERR] $label"
        _log "        $hint"
        _fail "precond:${label}"
        PRECOND_OK=false
    fi
}

[[ -f run_context_memory.sh ]] && _rc=true || _rc=false
_check "run_context_memory.sh exists" "$_rc" "Expected at ${PROJECT_ROOT}/run_context_memory.sh"

[[ -f CL-bench_context_ge5.jsonl ]] && _ds=true || _ds=false
_check "CL-bench_context_ge5.jsonl exists" "$_ds" \
    "Run: modelscope download --dataset 'littlewyy/CL-bench_context_ge5' CL-bench_context_ge5.jsonl --local_dir ."

# registry import check
if python -c "
from cl_bench_memory import AgentKBMemory, AgentWorkflowMemory, LightweightMemory, SkillWeaverMemory
from cl_bench_memory import build_memory
" 2>/dev/null; then
    _log "  [ok]  all 4 backends importable from cl_bench_memory"
    _pass "precond:registry_imports"
else
    _log "  [ERR] import check failed:"
    python -c "from cl_bench_memory import AgentKBMemory, AgentWorkflowMemory, LightweightMemory, SkillWeaverMemory" \
        2>&1 | sed 's/^/       /'
    _fail "precond:registry_imports"
    PRECOND_OK=false
fi

if python -c "import volcenginesdkarkruntime" 2>/dev/null; then
    _log "  [ok]  volcenginesdkarkruntime importable"
    _pass "precond:ark_sdk"
else
    _log "  [WARN] volcenginesdkarkruntime not installed — inference may not route through Ark batch endpoint"
    _log "         Install with: pip install volcengine-python-sdk"
    _pass "precond:ark_sdk (warned)"
fi

if python -c "
from dotenv import load_dotenv; import os
load_dotenv()
import sys; sys.exit(0 if os.getenv('BATCH_MODEL') else 1)
" 2>/dev/null; then
    _log "  [ok]  BATCH_MODEL set"
    _pass "precond:BATCH_MODEL"
else
    _log "  [WARN] BATCH_MODEL not set — inference will use DEEPSEEK_* endpoint instead of Ark batch"
    _pass "precond:BATCH_MODEL (warned)"
fi

if [[ "$PRECOND_OK" == false ]]; then
    _log ""
    _log "Preconditions failed. Switching to --static-only."
    STATIC_ONLY=true
fi

# ── Phase 3: E2E mini-runs ─────────────────────────────────────────────────────
if [[ "$STATIC_ONLY" == true ]]; then
    _log ""
    _log "=== Phase 3: E2E skipped (--static-only or precondition failure) ==="
else
    _log ""
    _log "=== Phase 3: E2E mini-runs (--max-samples 2 --workers 1 --eval-workers 1 --top-k 1 --skip-compare) ==="

    INPUT_NAME="CL-bench_context_ge5"

    _sanitize() {
        local s="${1//\//_}"
        echo "${s//:/_}"
    }

    _run_backend() {
        local mem_type="$1"
        local label="e2e:${mem_type}"
        local log_file="${SMOKE_LOG_DIR}/${mem_type}.log"
        local backend_dir="${SMOKE_LOG_DIR}/${mem_type}"
        local mt_safe
        mt_safe=$(_sanitize "$mem_type")

        if [[ "${BACKEND_SYNTAX_OK[$mem_type]:-false}" == false ]]; then
            _log ""
            _log "  [skip] $mem_type — syntax error, fix the file first"
            _skip "$label"
            return
        fi

        _log ""
        _log "  Running $mem_type ..."
        local t_start=$SECONDS

        local exit_code=0
        bash run_context_memory.sh \
            --memory-type "$mem_type" \
            --max-samples 2 \
            --workers 1 \
            --eval-workers 1 \
            --top-k 1 \
            --skip-compare \
            >"${log_file}" 2>&1 || exit_code=$?

        local elapsed=$(( SECONDS - t_start ))

        if [[ $exit_code -ne 0 ]]; then
            _log "  [FAIL] $mem_type (${elapsed}s) — pipeline exited $exit_code"
            _log "         Log: ${log_file}"
            tail -10 "${log_file}" | sed 's/^/         /'
            _fail "${label} (${elapsed}s)"
            return
        fi

        local raw_file graded_file
        raw_file=$(ls -t "outputs/context_${mt_safe}/${INPUT_NAME}_"*"_ctx_memory_topk1"*.jsonl \
            2>/dev/null | grep -v "_graded" | head -1 || true)
        graded_file="${raw_file%.jsonl}_graded.jsonl"

        local assert_ok=true
        if [[ -z "$raw_file" || ! -f "$raw_file" ]]; then
            _log "  [FAIL] $mem_type — inference output not found under outputs/context_${mt_safe}/"
            assert_ok=false
        elif [[ $(wc -l < "$raw_file") -lt 1 ]]; then
            _log "  [FAIL] $mem_type — inference output is empty: $raw_file"
            assert_ok=false
        fi

        if [[ "$assert_ok" == true ]]; then
            if [[ ! -f "$graded_file" ]]; then
                _log "  [FAIL] $mem_type — graded output not found: $graded_file"
                assert_ok=false
            elif [[ $(wc -l < "$graded_file") -lt 1 ]]; then
                _log "  [FAIL] $mem_type — graded output is empty: $graded_file"
                assert_ok=false
            fi
        fi

        if [[ "$assert_ok" == true ]]; then
            mkdir -p "${backend_dir}"
            mv "$raw_file"    "${backend_dir}/"
            mv "$graded_file" "${backend_dir}/"
            _log "  [PASS] $mem_type (${elapsed}s) — outputs in ${backend_dir}/"
            _pass "${label} (${elapsed}s)"
        else
            _log "         Log: ${log_file}"
            tail -10 "${log_file}" | sed 's/^/         /'
            _fail "${label} (${elapsed}s)"
        fi
    }

    for backend in agent_kb agent_workflow lightweight skillweaver; do
        _run_backend "$backend"
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────────
_log ""
_log "=== Summary ==="
for r in "${RESULTS[@]}"; do
    status="${r%%|*}"
    detail="${r#*|}"
    _log "  [${status}] ${detail}"
done
_log ""
_log "  ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped"

# ── Write Markdown report ──────────────────────────────────────────────────────
{
    echo "# Smoke Test Report — New Backends"
    echo ""
    echo "- **Date:** $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- **Mode:** $([ "$STATIC_ONLY" == true ] && echo 'static only' || echo 'E2E (--max-samples 2, --workers 1)')"
    echo "- **Backends:** agent_kb, agent_workflow, lightweight, skillweaver"
    echo ""
    echo "## Results"
    echo ""
    for r in "${RESULTS[@]}"; do
        status="${r%%|*}"
        detail="${r#*|}"
        case "$status" in
            PASS) echo "- ✅ ${detail}" ;;
            FAIL) echo "- ❌ ${detail}" ;;
            SKIP) echo "- ⏭️  ${detail}" ;;
        esac
    done
    echo ""
    echo "## Summary"
    echo ""
    echo "**${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped**"
    echo ""
    if [[ $FAIL_COUNT -gt 0 ]]; then
        echo "**Status: FAILED**"
        echo ""
        echo "Per-backend logs in \`${SMOKE_LOG_DIR}/\`."
    else
        echo "**Status: OK**"
        echo ""
        echo "E2E outputs kept in \`${SMOKE_LOG_DIR}/<backend>/\`."
        echo "To clear: \`rm -rf ${SMOKE_LOG_DIR}/\`"
    fi
} > "${REPORT_FILE}"

_log ""
if [[ $FAIL_COUNT -gt 0 ]]; then
    _log "FAILED — report: ${REPORT_FILE}"
    exit 1
else
    _log "OK — report: ${REPORT_FILE}"
fi
