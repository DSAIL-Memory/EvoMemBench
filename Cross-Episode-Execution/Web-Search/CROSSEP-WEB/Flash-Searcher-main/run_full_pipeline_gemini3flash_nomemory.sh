#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline_gemini3flash_nomemory.sh — gemini-3-flash-preview 无 memory 基线
#
# 基于 run_full_pipeline.sh，agent 模型换成 gemini-3-flash-preview（aigcbest OpenAI 兼容
# 端点），不使用任何 memory。judge 仍保留 deepseek Ark batch（与 deepseek 基线
# 同口径，方便对比）。
#
# Usage:
#   bash run_full_pipeline_gemini3flash_nomemory.sh                      # 全量
#   bash run_full_pipeline_gemini3flash_nomemory.sh --xbench-n 3 --ww-n 3  # 小样验证
#   bash run_full_pipeline_gemini3flash_nomemory.sh --xbench-only
#   bash run_full_pipeline_gemini3flash_nomemory.sh --webwalkerqa-only
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script requires bash. Run: bash run_full_pipeline_gemini3flash_nomemory.sh" >&2
    exit 1
fi

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# ▸▸▸  CONFIGURATION  ◂◂◂
# =============================================================================

ORDER="webwalkerqa_xbench"

# gemini-3-flash-preview 基线：不带 memory
MEMORY_PROVIDER=""

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

# 输出目录与 deepseek / gpt5mini 隔离
OUTPUT_DIR="./pipeline_output_gemini3flash_nomemory"

JUDGE_MODEL=""

# =============================================================================
# ▸▸▸  CLI FLAGS  ◂◂◂
# =============================================================================

SKIP_XBENCH=false
SKIP_WW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ww-first)          ORDER="webwalkerqa_xbench" ;;
        --xbench-first)      ORDER="xbench_webwalkerqa" ;;
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
            sed -n '2,12p' "$0" | sed 's/^# \?//'
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

# 覆写 agent 模型为 gemini-3-flash-preview；触发 ArkBatchServerModel 走 OpenAI 直连分支
export DEFAULT_MODEL="gemini-3-flash-preview"
export OPENAI_AGENT_MODEL="gemini-3-flash-preview"
export ANALYSIS_MODEL="$DEFAULT_MODEL"
export GENERATION_MODEL="$DEFAULT_MODEL"
export MSWEA_MODEL_NAME="$DEFAULT_MODEL"
# DEFAULT_JUDGE_MODEL / BATCH_MODEL / DEEPSEEK_API_KEY 不动 → judge 走 Ark batch

# gemini-3-flash-preview agent 走 aigcbest，models.py 读 AIGCBEST_* 变量
export OPENAI_API_BASE="${AIGCBEST_BASE_URL:-}"

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
echo " MemEvolve Full Pipeline — gemini-3-flash-preview no-memory baseline"
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
"${CMD[@]}"
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
