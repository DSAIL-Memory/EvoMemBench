#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Complete MemEvolve Pipeline
#
# Runs two benchmarks sequentially with a single shared memory provider so
# that knowledge accumulated in Phase 1 is available to Phase 2.
#
# Default order : XBench (Phase 1) → WebWalkerQA (Phase 2)
# Reversed order: WebWalkerQA (Phase 1) → XBench (Phase 2)  [--ww-first]
#
# Usage:
#   cd /data/kcyu/MemEvolve/Flash-Searcher-main
#   bash run_full_pipeline.sh                          # XBench → WebWalkerQA
#   bash run_full_pipeline.sh --ww-first               # WebWalkerQA → XBench
#   bash run_full_pipeline.sh --xbench-only            # XBench only
#   bash run_full_pipeline.sh --webwalkerqa-only       # WebWalkerQA only
#   bash run_full_pipeline.sh --no-memory              # baseline (no memory)
#   bash run_full_pipeline.sh --xbench-n 50            # limit XBench to 50 tasks
#   bash run_full_pipeline.sh --ww-n 50                # limit WebWalkerQA to 50 tasks
#
# All results go to OUTPUT_DIR (default: ./pipeline_output/).
# =============================================================================

# Must run with bash (not sh): uses arrays, pipefail, BASH_SOURCE, etc.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script requires bash. Run: bash run_full_pipeline.sh" >&2
    exit 1
fi

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# ▸▸▸  CONFIGURATION  ◂◂◂
# Adjust these variables to control the run.
# =============================================================================

# -- Execution order --
# "xbench_webwalkerqa" → XBench first (default)
# "webwalkerqa_xbench" → WebWalkerQA first
ORDER="xbench_webwalkerqa"

# -- Memory --
MEMORY_PROVIDER="memory_os"   # agent_kb | lightweight_memory | cerebra_fusion_memory | etc.
                                       # Set to "" to run without memory (baseline)

# -- Datasets --
XBENCH_INFILE="./data/xbench/DeepSearch.csv"
WW_INFILE="./data/webwalkerqa/webwalkerqa_subset_170.jsonl"

# Optional: limit number of tasks (set to "" for the full dataset)
XBENCH_N=""        # e.g. "100" → process first 100 XBench tasks
WW_N=""            # e.g. "170" → process first 170 WebWalkerQA tasks

# Optional: specific task index ranges (overrides XBENCH_N / WW_N if set)
# Format: "1-50" or "1,3,5-20,30"
XBENCH_INDICES=""
WW_INDICES=""

# -- Agent --
MAX_STEPS=40
CONCURRENCY=50
SUMMARY_INTERVAL=8
PROMPTS_TYPE="default"

# -- Output --
OUTPUT_DIR="./pipeline_output"

# -- Judge model --
# Defaults to DEFAULT_JUDGE_MODEL env var, or deepseek-chat as fallback.
# Override here if needed, e.g. JUDGE_MODEL="gpt-5"
JUDGE_MODEL=""

# =============================================================================
# ▸▸▸  CLI FLAGS  ◂◂◂  (override config above without editing the file)
# =============================================================================

SKIP_XBENCH=false
SKIP_WW=false
BOTH_ORDERS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ww-first)          ORDER="webwalkerqa_xbench" ;;
        --xbench-first)      ORDER="xbench_webwalkerqa" ;;
        --both-orders)       BOTH_ORDERS=true ;;
        --xbench-only)       SKIP_WW=true ;;
        --webwalkerqa-only)  SKIP_XBENCH=true ;;
        --no-memory)         MEMORY_PROVIDER="" ;;
        --memory)            shift; MEMORY_PROVIDER="$1" ;;
        --xbench-n)          shift; XBENCH_N="$1" ;;
        --ww-n)              shift; WW_N="$1" ;;
        --xbench-indices)    shift; XBENCH_INDICES="$1" ;;
        --ww-indices)        shift; WW_INDICES="$1" ;;
        --max-steps)         shift; MAX_STEPS="$1" ;;
        --concurrency)       shift; CONCURRENCY="$1" ;;
        --output-dir)        shift; OUTPUT_DIR="$1" ;;
        --judge-model)       shift; JUDGE_MODEL="$1" ;;
        -h|--help)
            sed -n '2,24p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1  (use --help for usage)" >&2
            exit 1
            ;;
    esac
    shift
done

# =============================================================================
# ▸▸▸  ENVIRONMENT SETUP  ◂◂◂
# =============================================================================

# Load .env
set -a
source .env 2>/dev/null || true
set +a

# Bridge env var name mismatch: .env sets OPENAI_BASE_URL but code reads OPENAI_API_BASE
export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"

# Judge model: CLI flag > env var > deepseek-chat fallback
if [[ -z "$JUDGE_MODEL" ]]; then
    JUDGE_MODEL="${DEFAULT_JUDGE_MODEL:-deepseek-chat}"
fi
export DEFAULT_JUDGE_MODEL="$JUDGE_MODEL"

# =============================================================================
# ▸▸▸  LOGGING SETUP  ◂◂◂
# =============================================================================

mkdir -p "$OUTPUT_DIR"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/run_${RUN_TS}.log"

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " MemEvolve Full Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Configuration:"
printf "  %-24s %s\n" "Execution order:"  "$ORDER"
printf "  %-24s %s\n" "Memory provider:"  "${MEMORY_PROVIDER:-<none — baseline>}"
printf "  %-24s %s\n" "XBench input:"     "$XBENCH_INFILE"
printf "  %-24s %s\n" "WebWalkerQA input:" "$WW_INFILE"
printf "  %-24s %s\n" "Skip XBench:"      "$SKIP_XBENCH"
printf "  %-24s %s\n" "Skip WebWalkerQA:" "$SKIP_WW"
printf "  %-24s %s\n" "XBench tasks:"     "${XBENCH_INDICES:-${XBENCH_N:-all}}"
printf "  %-24s %s\n" "WW tasks:"         "${WW_INDICES:-${WW_N:-all}}"
printf "  %-24s %s\n" "Max steps:"        "$MAX_STEPS"
printf "  %-24s %s\n" "Concurrency:"      "$CONCURRENCY"
printf "  %-24s %s\n" "Judge model:"      "$JUDGE_MODEL"
printf "  %-24s %s\n" "Output dir:"       "$OUTPUT_DIR"
printf "  %-24s %s\n" "Log file:"         "$LOG_FILE"
printf "  %-24s %s\n" "Model:"            "${DEFAULT_MODEL:-<not set>}"
printf "  %-24s %s\n" "API base:"         "${OPENAI_API_BASE:-<not set>}"
echo ""

# =============================================================================
# ▸▸▸  BUILD PIPELINE COMMAND  ◂◂◂
# =============================================================================

CMD=(python run_pipeline.py
    --output_dir    "$OUTPUT_DIR"
    --order         "$ORDER"
    --max_steps     "$MAX_STEPS"
    --concurrency   "$CONCURRENCY"
    --summary_interval "$SUMMARY_INTERVAL"
    --prompts_type  "$PROMPTS_TYPE"
    --judge_model   "$JUDGE_MODEL"
    --xbench_infile "$XBENCH_INFILE"
    --webwalkerqa_infile "$WW_INFILE"
)

# Memory
if [[ -n "$MEMORY_PROVIDER" ]]; then
    CMD+=(--memory_provider "$MEMORY_PROVIDER")
fi

# Phase skips
if $SKIP_XBENCH; then CMD+=(--skip_xbench); fi
if $SKIP_WW;     then CMD+=(--skip_webwalkerqa); fi

# Task limits (indices take precedence over sample_num)
if [[ -n "$XBENCH_INDICES" ]]; then
    CMD+=(--xbench_task_indices "$XBENCH_INDICES")
elif [[ -n "$XBENCH_N" ]]; then
    CMD+=(--xbench_sample_num "$XBENCH_N")
fi

if [[ -n "$WW_INDICES" ]]; then
    CMD+=(--webwalkerqa_task_indices "$WW_INDICES")
elif [[ -n "$WW_N" ]]; then
    CMD+=(--webwalkerqa_sample_num "$WW_N")
fi

# =============================================================================
# ▸▸▸  RUN PIPELINE  ◂◂◂
# =============================================================================

echo "------------------------------------------------------------"
echo " Running pipeline..."
echo " CMD: ${CMD[*]}"
echo "------------------------------------------------------------"
echo ""

START_EPOCH=$(date +%s)

# Qdrant storage path used by memos_provider
STORAGE_PATH="./storage/memory_os"

PIPELINE_SUBDIR=""  # set below only when BOTH_ORDERS=false
if $BOTH_ORDERS; then
    # Storage paths are now isolated per-order via storage_suffix in run_pipeline.py,
    # so both orders can run in parallel without interference.
    PIDS=()
    for ord in xbench_webwalkerqa webwalkerqa_xbench; do
        OUTDIR_ORD="${OUTPUT_DIR}/memory_os_${ord}"
        mkdir -p "$OUTDIR_ORD"
        LOG_ORD="${OUTDIR_ORD}/run_${ord}.log"

        ORD_CMD=(python run_pipeline.py
            --output_dir    "$OUTDIR_ORD"
            --order         "$ord"
            --max_steps     "$MAX_STEPS"
            --concurrency   "$CONCURRENCY"
            --summary_interval "$SUMMARY_INTERVAL"
            --prompts_type  "$PROMPTS_TYPE"
            --judge_model   "$JUDGE_MODEL"
            --xbench_infile "$XBENCH_INFILE"
            --webwalkerqa_infile "$WW_INFILE"
        )
        if [[ -n "$MEMORY_PROVIDER" ]]; then
            ORD_CMD+=(--memory_provider "$MEMORY_PROVIDER")
        fi
        if [[ -n "$XBENCH_INDICES" ]]; then
            ORD_CMD+=(--xbench_task_indices "$XBENCH_INDICES")
        elif [[ -n "$XBENCH_N" ]]; then
            ORD_CMD+=(--xbench_sample_num "$XBENCH_N")
        fi
        if [[ -n "$WW_INDICES" ]]; then
            ORD_CMD+=(--webwalkerqa_task_indices "$WW_INDICES")
        elif [[ -n "$WW_N" ]]; then
            ORD_CMD+=(--webwalkerqa_sample_num "$WW_N")
        fi

        echo " Launching order=$ord in background → $LOG_ORD"
        echo " CMD: ${ORD_CMD[*]}"
        "${ORD_CMD[@]}" > >(tee -a "$LOG_ORD") 2>&1 &
        PIDS+=($!)
    done

    echo " Waiting for both orders to complete (PIDs: ${PIDS[*]})..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    echo ""
    echo "============================================================"
    echo " Both-orders complete. Summary:"
    echo "============================================================"
    for ord in xbench_webwalkerqa webwalkerqa_xbench; do
        OUTDIR_ORD="${OUTPUT_DIR}/memory_os_${ord}"
        RPT="$(ls -t "${OUTDIR_ORD}"/pipeline_memory_os_*/pipeline_report.txt 2>/dev/null | head -1 || true)"
        if [[ -f "$RPT" ]]; then
            echo ""
            echo "--- order: $ord ---"
            cat "$RPT"
        fi
        XBENCH_J="$(ls -t "${OUTDIR_ORD}"/pipeline_memory_os_*/xbench_results.jsonl 2>/dev/null | head -1 || true)"
        WW_J="$(ls -t "${OUTDIR_ORD}"/pipeline_memory_os_*/webwalkerqa_results.jsonl 2>/dev/null | head -1 || true)"
        AN_ARGS=()
        [[ -f "$XBENCH_J" ]] && AN_ARGS+=("$XBENCH_J")
        [[ -f "$WW_J" ]]    && AN_ARGS+=("$WW_J")
        if [[ ${#AN_ARGS[@]} -gt 0 ]]; then
            python analyze_results.py "${AN_ARGS[@]}" --output "${OUTDIR_ORD}/analysis_report.txt"
            echo "Analysis: ${OUTDIR_ORD}/analysis_report.txt"
        fi
    done
else
    "${CMD[@]}"
fi

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
echo ""
echo "Pipeline finished in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

if ! $BOTH_ORDERS; then
# =============================================================================
# ▸▸▸  FIND OUTPUT FILES  ◂◂◂
# Locate the timestamped subdirectory created by run_pipeline.py
# =============================================================================

# The pipeline creates: OUTPUT_DIR/pipeline_{memory_tag}{timestamp}/
MEM_TAG="${MEMORY_PROVIDER:-nomemory}"
PIPELINE_SUBDIR="$(ls -td "$OUTPUT_DIR"/pipeline_${MEM_TAG}_* 2>/dev/null | head -1 || true)"

if [[ -z "$PIPELINE_SUBDIR" ]]; then
    # Fallback: find any pipeline_ dir modified in the last 60 seconds
    PIPELINE_SUBDIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'pipeline_*' \
                        -newer "$LOG_FILE" 2>/dev/null | sort | tail -1 || true)"
fi

XBENCH_JSONL="$PIPELINE_SUBDIR/xbench_results.jsonl"
WW_JSONL="$PIPELINE_SUBDIR/webwalkerqa_results.jsonl"

echo ""
echo "------------------------------------------------------------"
echo " Output directory: ${PIPELINE_SUBDIR:-<not found>}"
echo "------------------------------------------------------------"

# =============================================================================
# ▸▸▸  ANALYZE RESULTS  ◂◂◂
# =============================================================================

echo ""
echo "============================================================"
echo " Results Analysis"
echo "============================================================"
echo ""

ANALYSIS_CMD=(python analyze_results.py)
FOUND_ANY=false

if [[ -f "$XBENCH_JSONL" ]]; then
    ANALYSIS_CMD+=("$XBENCH_JSONL")
    FOUND_ANY=true
    echo "  XBench results   : $XBENCH_JSONL"
fi

if [[ -f "$WW_JSONL" ]]; then
    ANALYSIS_CMD+=("$WW_JSONL")
    FOUND_ANY=true
    echo "  WebWalkerQA results: $WW_JSONL"
fi

ANALYSIS_REPORT="$PIPELINE_SUBDIR/analysis_report.txt"
ANALYSIS_CMD+=(--output "$ANALYSIS_REPORT")

echo ""

if $FOUND_ANY; then
    "${ANALYSIS_CMD[@]}"
    echo ""
    echo "Analysis report saved to: $ANALYSIS_REPORT"
else
    echo "Warning: no JSONL result files found in $PIPELINE_SUBDIR" >&2
    echo "Pipeline report (if any): $PIPELINE_SUBDIR/pipeline_report.txt"
fi
fi # end: if ! $BOTH_ORDERS

# =============================================================================
# ▸▸▸  SUMMARY  ◂◂◂
# =============================================================================

echo ""
echo "============================================================"
echo " DONE"
echo " $(date '+%Y-%m-%d %H:%M:%S')  |  elapsed: ${ELAPSED}s"
echo "============================================================"
echo ""
echo "Output files:"
if [[ -n "$PIPELINE_SUBDIR" ]]; then
    for f in "$PIPELINE_SUBDIR"/*; do
        [[ -e "$f" ]] && printf "  %s\n" "$f"
    done
fi
echo ""
echo "Full log: $LOG_FILE"
echo "============================================================"
