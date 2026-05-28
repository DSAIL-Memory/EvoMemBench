#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline_graphrag.sh — GraphRAG (semantic + concept-graph) memory provider
#
# Runs two benchmarks sequentially with the GraphRAG memory provider so that
# trajectory chunks accumulated in Phase 1 are available to Phase 2.
#
# Default order : XBench (Phase 1) → WebWalkerQA (Phase 2)
# Reversed order: WebWalkerQA (Phase 1) → XBench (Phase 2)  [--ww-first]
#
# Usage:
#   cd /data/zhongjianzhang/EvoMemBench-XbenchWebWalkQA/Flash-Searcher-main
#   bash run_full_pipeline_graphrag.sh                          # XBench → WebWalkerQA
#   bash run_full_pipeline_graphrag.sh --ww-first               # WebWalkerQA → XBench
#   bash run_full_pipeline_graphrag.sh --both-orders            # Run both orderings
#   bash run_full_pipeline_graphrag.sh --xbench-only            # XBench only
#   bash run_full_pipeline_graphrag.sh --webwalkerqa-only       # WebWalkerQA only
#   bash run_full_pipeline_graphrag.sh --xbench-n 50            # Limit XBench to 50 tasks
#   bash run_full_pipeline_graphrag.sh --ww-n 50                # Limit WebWalkerQA to 50 tasks
#
# All results go to OUTPUT_DIR (default: ./pipeline_output/).
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script requires bash. Run: bash run_full_pipeline_graphrag.sh" >&2
    exit 1
fi

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# ▸▸▸  CONFIGURATION  ◂◂◂
# =============================================================================

ORDER="xbench_webwalkerqa"

MEMORY_PROVIDER="graphrag"

XBENCH_INFILE="./data/xbench/DeepSearch.csv"
WW_INFILE="./data/webwalkerqa/webwalkerqa_subset_170.jsonl"

XBENCH_N=""
WW_N=""

XBENCH_INDICES=""
WW_INDICES=""

MAX_STEPS=40
CONCURRENCY=50
SUMMARY_INTERVAL=8
PROMPTS_TYPE="default"

OUTPUT_DIR="./pipeline_output"

JUDGE_MODEL=""

# =============================================================================
# ▸▸▸  CLI FLAGS  ◂◂◂
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

set -a
source .env 2>/dev/null || true
set +a

export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"

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

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " MemEvolve Full Pipeline — GraphRAG"
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

if [[ -n "$MEMORY_PROVIDER" ]]; then
    CMD+=(--memory_provider "$MEMORY_PROVIDER")
fi

if $SKIP_XBENCH; then CMD+=(--skip_xbench); fi
if $SKIP_WW;     then CMD+=(--skip_webwalkerqa); fi

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

# GraphRAG storage: memory_bank.jsonl + embeddings.jsonl + graph.gpickle
GRAPHRAG_DIR="./storage/graphrag"

if $BOTH_ORDERS; then
    declare -a PIDS=()
    declare -A PID_TO_ORD

    for ord in xbench_webwalkerqa webwalkerqa_xbench; do
        OUTDIR_ORD="${OUTPUT_DIR}/graphrag_${ord}"
        GRAPHRAG_DIR_ORD="./storage/graphrag_${ord}"
        ORD_LOG="${OUTPUT_DIR}/run_${ord}_${RUN_TS}.log"

        mkdir -p "$OUTDIR_ORD"
        rm -rf "$GRAPHRAG_DIR_ORD"
        mkdir -p "$GRAPHRAG_DIR_ORD"

        ORD_CMD=(python run_pipeline.py
            --output_dir         "$OUTDIR_ORD"
            --order              "$ord"
            --memory_dir         "$GRAPHRAG_DIR_ORD"
            --max_steps          "$MAX_STEPS"
            --concurrency        "$CONCURRENCY"
            --summary_interval   "$SUMMARY_INTERVAL"
            --prompts_type       "$PROMPTS_TYPE"
            --judge_model        "$JUDGE_MODEL"
            --xbench_infile      "$XBENCH_INFILE"
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

        echo "============================================================"
        echo " [graphrag] --both-orders: launching order=$ord in background"
        printf "  %-16s %s\n" "storage:"  "$GRAPHRAG_DIR_ORD"
        printf "  %-16s %s\n" "output:"   "$OUTDIR_ORD"
        printf "  %-16s %s\n" "log:"      "$ORD_LOG"
        echo " CMD: ${ORD_CMD[*]}"
        echo "============================================================"

        "${ORD_CMD[@]}" > "$ORD_LOG" 2>&1 &
        pid=$!
        PIDS+=("$pid")
        PID_TO_ORD[$pid]="$ord"
    done

    echo ""
    echo "Both orderings running in parallel. Waiting for completion..."
    BOTH_FAILED=0
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            echo "[graphrag] order=${PID_TO_ORD[$pid]} (pid=$pid) finished OK"
        else
            echo "[graphrag] order=${PID_TO_ORD[$pid]} (pid=$pid) FAILED (exit $?)"
            BOTH_FAILED=1
        fi
    done

    echo ""
    echo "============================================================"
    echo " Both-orders complete. Summary:"
    echo "============================================================"
    for ord in xbench_webwalkerqa webwalkerqa_xbench; do
        echo ""
        echo "--- order: $ord ---"
        echo "  log: ${OUTPUT_DIR}/run_${ord}_${RUN_TS}.log"
        OUTDIR_ORD="${OUTPUT_DIR}/graphrag_${ord}"
        RPT="$(ls -t "${OUTDIR_ORD}"/pipeline_graphrag_*/pipeline_report.txt 2>/dev/null | head -1 || true)"
        if [[ -f "$RPT" ]]; then
            cat "$RPT"
        else
            echo "  (no pipeline_report.txt found in $OUTDIR_ORD)"
        fi
    done

    if [[ "$BOTH_FAILED" -ne 0 ]]; then
        echo ""
        echo "WARNING: one or more orderings failed. Check the logs above." >&2
        exit 1
    fi
else
    "${CMD[@]}"
fi

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
echo ""
echo "Pipeline finished in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

# =============================================================================
# ▸▸▸  FIND OUTPUT FILES  ◂◂◂
# =============================================================================

MEM_TAG="${MEMORY_PROVIDER:-nomemory}"
PIPELINE_SUBDIR="$(ls -td "$OUTPUT_DIR"/pipeline_${MEM_TAG}_* 2>/dev/null | head -1 || true)"

if [[ -z "$PIPELINE_SUBDIR" ]]; then
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
