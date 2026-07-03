#!/usr/bin/env bash
# smoke_test.sh — lightweight smoke tests for all scripts in scripts/
#
# Usage: bash scripts/smoke_test.sh [--static-only]
#
#   --static-only   Syntax + dependency checks only, no API calls.
#
# E2E outputs are moved to outputs/_smoke/<backend>/ after each run.
# A Markdown report is written to outputs/_smoke/smoke_report_<timestamp>.md.
# To clear all smoke artifacts: rm -rf outputs/_smoke/

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Flags
# ──────────────────────────────────────────────────────────────────────────────
STATIC_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --static-only) STATIC_ONLY=true ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash scripts/smoke_test.sh [--static-only]"
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Resolve project root (one level up from scripts/)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SMOKE_LOG_DIR="outputs/_smoke"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${SMOKE_LOG_DIR}/smoke_report_${TIMESTAMP}.md"
mkdir -p "${SMOKE_LOG_DIR}"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

_pass() { PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS|$1"); }
_fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL|$1"); }
_skip() { SKIP_COUNT=$((SKIP_COUNT + 1)); RESULTS+=("SKIP|$1"); }

# Print to console and accumulate for the report
_log() { echo "$*"; }

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Syntax check
# ──────────────────────────────────────────────────────────────────────────────
_log ""
_log "=== Phase 1: Bash syntax check ==="

SYNTAX_OK=true
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
for sh in "${SCRIPT_DIR}"/*.sh; do
    name="$(basename "$sh")"
    [[ "$name" == "$SELF_NAME" ]] && continue
    if bash -n "$sh" 2>/dev/null; then
        _log "  [ok]  $name"
        _pass "syntax:${name}"
    else
        _log "  [ERR] $name — syntax error:"
        bash -n "$sh" 2>&1 | sed 's/^/       /'
        _fail "syntax:${name}"
        SYNTAX_OK=false
    fi
done

if [[ "$SYNTAX_OK" == false ]]; then
    _log ""
    _log "Syntax errors found. Fix them before proceeding."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Precondition checks
# ──────────────────────────────────────────────────────────────────────────────
_log ""
_log "=== Phase 2: Precondition checks ==="

PRECOND_OK=true

check() {
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
check "run_context_memory.sh exists" "$_rc" "File not found at ${PROJECT_ROOT}/run_context_memory.sh"

[[ -f CL-bench_context_ge5.jsonl ]] && _ds=true || _ds=false
check "CL-bench_context_ge5.jsonl exists" "$_ds" \
    "Run: modelscope download --dataset 'littlewyy/CL-bench_context_ge5' CL-bench_context_ge5.jsonl --local_dir ."

[[ -f .env ]] && _env=true || _env=false
if [[ "$_env" == false ]]; then
    _log "  [WARN] .env not found — API keys must be exported in the environment."
    _pass "precond:.env (warned)"
else
    _log "  [ok]  .env exists"
    _pass "precond:.env"
fi

# Ark batch SDK — inference must route through Volcengine Ark batch endpoint
if python -c "import volcenginesdkarkruntime" 2>/dev/null; then
    _log "  [ok]  volcenginesdkarkruntime importable"
    _pass "precond:ark_sdk"
else
    _log "  [ERR] volcenginesdkarkruntime not installed"
    _log "        Install with: pip install volcengine-python-sdk"
    _fail "precond:ark_sdk"
    PRECOND_OK=false
fi

# BATCH_MODEL must be set (read via python-dotenv to mirror infer_context_memory.py:297)
if python -c "from dotenv import load_dotenv; import os; load_dotenv(override=True); import sys; sys.exit(0 if os.getenv('BATCH_MODEL') else 1)" 2>/dev/null; then
    _log "  [ok]  BATCH_MODEL set in .env / environment"
    _pass "precond:batch_model"
else
    _log "  [ERR] BATCH_MODEL not set — inference will not route through Ark batch endpoint"
    _log "        Add 'BATCH_MODEL=<ark_endpoint_id>' to .env"
    _fail "precond:batch_model"
    PRECOND_OK=false
fi

_CORE_FILES=(
    cl_bench_memory/base.py
    cl_bench_memory/trajectory.py
    cl_bench_memory/api_client.py
    cl_bench_memory/io.py
    infer_context_memory.py
    eval.py
)
if python -m py_compile "${_CORE_FILES[@]}" 2>/dev/null; then
    _log "  [ok]  Python syntax (core files)"
    _pass "precond:python_syntax"
else
    _log "  [ERR] Python syntax error in core files:"
    python -m py_compile "${_CORE_FILES[@]}" 2>&1 | sed 's/^/       /'
    _fail "precond:python_syntax"
    PRECOND_OK=false
fi

_log "  Checking per-backend file syntax ..."
declare -A BACKEND_IMPORT_OK
for _pair in \
    "reasoning_bank:cl_bench_memory/reasoning_bank.py" \
    "ace:cl_bench_memory/ace_memory.py" \
    "mem0:cl_bench_memory/mem0_memory.py" \
    "bm25:cl_bench_memory/bm25_memory.py" \
    "graphrag:cl_bench_memory/graphrag_memory.py" \
    "qwen3_embedding_4b:cl_bench_memory/qwen3_embedding_memory.py" \
    "agent_kb:cl_bench_memory/agent_kb_memory.py" \
    "agent_workflow:cl_bench_memory/agent_workflow_memory.py" \
    "lightweight:cl_bench_memory/lightweight_memory.py" \
    "skillweaver:cl_bench_memory/skillweaver_memory.py" \
    "a_mem:cl_bench_memory/a_mem_memory.py" \
    "memos:cl_bench_memory/memos_memory.py" \
    "memoryos:cl_bench_memory/memoryos_memory.py"
do
    _bname="${_pair%%:*}"
    _file="${_pair#*:}"
    if python -m py_compile "$_file" 2>/dev/null; then
        _log "    [ok]  $_bname"
        BACKEND_IMPORT_OK[$_bname]=true
    else
        _log "    [WARN] $_bname syntax error (will skip E2E for this backend)"
        BACKEND_IMPORT_OK[$_bname]=false
    fi
done

if [[ "$PRECOND_OK" == false ]]; then
    _log ""
    _log "Preconditions failed. Cannot continue with E2E tests."
    STATIC_ONLY=true
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: E2E mini-runs
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$STATIC_ONLY" == true ]]; then
    _log ""
    _log "=== Phase 3: E2E skipped (--static-only or precondition failure) ==="
else
    _log ""
    _log "=== Phase 3: E2E mini-runs (--max-samples 2 --workers 1 --eval-workers 1 --skip-compare) ==="

    INPUT_NAME="CL-bench_context_ge5"

    # Mirrors run_context_memory.sh:69-70
    _sanitize() {
        local s="${1//\//_}"
        echo "${s//:/_}"
    }

    _run_smoke() {
        local script="$1"
        local label="$2"
        local extra_args=("${@:3}")
        local backend_dir="${SMOKE_LOG_DIR}/${script%.sh}"

        # Skip if the backend's Python file had a syntax error
        local mem_arg="" is_no_mem=false
        for a in "${extra_args[@]}"; do
            [[ "$a" == "--no-memory" ]] && is_no_mem=true && break
        done
        if [[ "$is_no_mem" == false ]]; then
            for i in "${!extra_args[@]}"; do
                if [[ "${extra_args[$i]}" == "--memory-type" ]]; then
                    mem_arg="${extra_args[$((i+1))]}"
                    break
                fi
            done
            if [[ -n "$mem_arg" && "${BACKEND_IMPORT_OK[$mem_arg]:-true}" == false ]]; then
                _log ""
                _log "  [skip] $label (backend syntax error — fix the file first)"
                _skip "e2e:${label}"
                return
            fi
        fi

        local log_file="${SMOKE_LOG_DIR}/${script%.sh}.log"
        _log ""
        _log "  Running $label ..."
        local t_start=$SECONDS

        local smoke_args=(
            "--max-samples" "2"
            "--workers" "1"
            "--eval-workers" "1"
            "--top-k" "1"
            "--skip-compare"
        )

        local exit_code=0
        bash run_context_memory.sh "${extra_args[@]}" "${smoke_args[@]}" \
            >"${log_file}" 2>&1 || exit_code=$?

        local elapsed=$(( SECONDS - t_start ))

        if [[ $exit_code -ne 0 ]]; then
            _log "  [FAIL] $label (${elapsed}s) — pipeline exited with code $exit_code"
            _log "         Log: ${log_file}"
            tail -8 "${log_file}" | sed 's/^/         /'
            _fail "e2e:${label} (${elapsed}s)"
            return
        fi

        # Locate output files
        local no_mem=false
        for a in "${extra_args[@]}"; do
            [[ "$a" == "--no-memory" ]] && no_mem=true && break
        done

        local raw_file="" graded_file=""
        if [[ "$no_mem" == true ]]; then
            raw_file=$(ls -t "outputs/context_nomemory/${INPUT_NAME}_"*"_ctx_nomemory_"*.jsonl \
                2>/dev/null | grep -v "_graded" | head -1 || true)
        else
            local mem_type="" mt_safe
            for i in "${!extra_args[@]}"; do
                if [[ "${extra_args[$i]}" == "--memory-type" ]]; then
                    mem_type="${extra_args[$((i+1))]}"
                    break
                fi
            done
            mt_safe=$(_sanitize "$mem_type")
            raw_file=$(ls -t "outputs/context_${mt_safe}/${INPUT_NAME}_"*"_ctx_memory_topk1"*.jsonl \
                2>/dev/null | grep -v "_graded" | head -1 || true)
        fi
        [[ -n "$raw_file" ]] && graded_file="${raw_file%.jsonl}_graded.jsonl"

        local assert_ok=true
        if [[ -z "$raw_file" || ! -f "$raw_file" ]]; then
            _log "  [FAIL] $label — inference output not found"
            assert_ok=false
        elif [[ $(wc -l < "$raw_file") -lt 1 ]]; then
            _log "  [FAIL] $label — inference output is empty: $raw_file"
            assert_ok=false
        fi
        if [[ "$assert_ok" == true ]]; then
            if [[ -z "$graded_file" || ! -f "$graded_file" ]]; then
                _log "  [FAIL] $label — graded output not found (expected: $graded_file)"
                assert_ok=false
            elif [[ $(wc -l < "$graded_file") -lt 1 ]]; then
                _log "  [FAIL] $label — graded output is empty: $graded_file"
                assert_ok=false
            fi
        fi

        # Assert inference routed through Ark batch endpoint (infer_context_memory.py:362)
        # Only applies to deepseek-* models; gemini/gpt variants use a different inference endpoint.
        local _infer_model=""
        for i in "${!extra_args[@]}"; do
            if [[ "${extra_args[$i]}" == "--infer-model" ]]; then
                _infer_model="${extra_args[$((i+1))]}"
                break
            fi
        done
        local _model_lower
        _model_lower=$(echo "${_infer_model:-deepseek-v3-2-251201}" | tr '[:upper:]' '[:lower:]')
        if [[ "$assert_ok" == true && "$_model_lower" == deepseek* ]]; then
            if ! grep -q "Using Volcengine Ark batch API" "${log_file}"; then
                _log "  [FAIL] $label — inference did not route through Ark batch endpoint"
                _log "         Expected: 'Using Volcengine Ark batch API' in ${log_file}"
                assert_ok=false
            fi
        fi

        if [[ "$assert_ok" == true ]]; then
            mkdir -p "${backend_dir}"
            mv "$raw_file"    "${backend_dir}/"
            mv "$graded_file" "${backend_dir}/"
            _log "  [PASS] $label (${elapsed}s) — outputs in ${backend_dir}/"
            _pass "e2e:${label} (${elapsed}s)"
        else
            _log "         Log: ${log_file}"
            tail -8 "${log_file}" | sed 's/^/         /'
            _fail "e2e:${label} (${elapsed}s)"
        fi
    }

    # Parse and run each wrapper script; skip group-level smoke scripts only
    for sh in "${SCRIPT_DIR}"/*.sh; do
        name="$(basename "$sh")"
        [[ "$name" == "$SELF_NAME"                        ]] && continue
        [[ "$name" == "smoke_agent_kb_agent_workflow_lightweight_skillweaver.sh" ]] && continue
        [[ "$name" == "smoke_amem_memos_memoryos.sh"     ]] && continue
        [[ "$name" == "smoke_bm25_qwen3_graphrag.sh"     ]] && continue
        [[ "$name" == "smoke_mem0_reasoning_bank_ace.sh" ]] && continue
        [[ "$name" == "nomemory_gpt-5-mini.sh"           ]] && continue  # legacy name, skip if present

        mem_type=$(grep -oE '\-\-memory-type\s+\S+' "$sh" 2>/dev/null | awk '{print $2}' | head -1 || true)
        no_memory=$(grep -qE '\-\-no-memory' "$sh" 2>/dev/null && echo "true" || echo "false")
        infer_model=$(grep -oE '\-\-infer-model\s+\S+' "$sh" 2>/dev/null | awk '{print $2}' | head -1 || true)

        backend_args=()
        if [[ "$no_memory" == "true" ]]; then
            backend_args+=("--no-memory")
        elif [[ -n "$mem_type" ]]; then
            backend_args+=("--memory-type" "$mem_type")
        else
            _log "  [WARN] Could not detect memory args in $name — skipping."
            _skip "e2e:${name}"
            continue
        fi
        [[ -n "$infer_model" ]] && backend_args+=("--infer-model" "$infer_model")

        _run_smoke "$name" "$name" "${backend_args[@]}"
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary (console)
# ──────────────────────────────────────────────────────────────────────────────
_log ""
_log "=== Smoke Test Summary ==="
if [[ ${#RESULTS[@]} -gt 0 ]]; then
    for r in "${RESULTS[@]}"; do
        status="${r%%|*}"
        detail="${r#*|}"
        _log "  [${status}] ${detail}"
    done
fi
_log ""
_log "  ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped"

# ──────────────────────────────────────────────────────────────────────────────
# Write Markdown report
# ──────────────────────────────────────────────────────────────────────────────
{
    echo "# Smoke Test Report"
    echo ""
    echo "- **Date:** $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- **Mode:** $([ "$STATIC_ONLY" == true ] && echo 'static only (no API calls)' || echo 'full E2E (--max-samples 2, --workers 1)')"
    echo "- **Project:** ${PROJECT_ROOT}"
    echo ""
    echo "## Results"
    echo ""
    if [[ ${#RESULTS[@]} -gt 0 ]]; then
        for r in "${RESULTS[@]}"; do
            status="${r%%|*}"
            detail="${r#*|}"
            case "$status" in
                PASS) echo "- ✅ ${detail}" ;;
                FAIL) echo "- ❌ ${detail}" ;;
                SKIP) echo "- ⏭️  ${detail}" ;;
            esac
        done
    fi
    echo ""
    echo "## Summary"
    echo ""
    echo "**${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped**"
    echo ""
    if [[ $FAIL_COUNT -gt 0 ]]; then
        echo "**Status: FAILED**"
        echo ""
        echo "See per-backend logs in \`${SMOKE_LOG_DIR}/\`."
    else
        echo "**Status: OK**"
        echo ""
        echo "E2E outputs kept in \`${SMOKE_LOG_DIR}/<backend>/\`."
        echo "To clear all smoke artifacts: \`rm -rf ${SMOKE_LOG_DIR}/\`"
    fi
} > "${REPORT_FILE}"

_log ""
if [[ $FAIL_COUNT -gt 0 ]]; then
    _log "FAILED — report: ${REPORT_FILE}"
    exit 1
else
    _log "OK — report: ${REPORT_FILE}"
fi
