#!/usr/bin/env bash
# run_context_memory.sh - Full pipeline: context-memory inference → eval → comparison analysis
#
# Usage:
#   bash run_context_memory.sh [options]
#
# Options:
#   --input NAME          Dataset basename without .jsonl (default: CL-bench_context_ge5)
#   --infer-model MODEL   Inference model (default: deepseek-v3-2-251201)
#   --judge-model MODEL   Judge model for eval (default: gpt-5.1)
#   --workers N           Parallel contexts for inference (default: 8)
#   --eval-workers N      Parallel workers for eval (default: 8)
#   --top-k N             Memory retrieval top-k (default: 10)
#   --memory-type TYPE    Memory backend type (default: reasoning_bank)
#   --no-memory           Disable memory (plain inference, skips retrieve/extract)
#   --max-samples N       Limit total samples (for testing)
#   --baseline PATH       Baseline graded file for comparison (optional)
#   --skip-compare        Skip comparison analysis
#   --help                Show this help
#
# Example:
#   bash run_context_memory.sh --workers 10 --top-k 10
#   bash run_context_memory.sh --memory-type reasoning_bank --workers 10
#   bash run_context_memory.sh --no-memory --workers 50
#   bash run_context_memory.sh --max-samples 15 --workers 3
#   bash run_context_memory.sh --baseline outputs/CL-bench_context_ge5_deepseek-chat_..._graded.jsonl

set -e

# Defaults
INPUT_NAME="CL-bench_context_ge5"
INFER_MODEL="deepseek-v3-2-251201"
JUDGE_MODEL="gpt-5.1"
WORKERS=50
EVAL_WORKERS=50
TOP_K=10
MEMORY_TYPE="mem0"
MAX_SAMPLES=""
BASELINE="outputs/context_nomemory/CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl"
SKIP_COMPARE=false
NO_MEMORY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)        INPUT_NAME="$2";   shift 2 ;;
        --infer-model)  INFER_MODEL="$2";  shift 2 ;;
        --judge-model)  JUDGE_MODEL="$2";  shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        --eval-workers) EVAL_WORKERS="$2"; shift 2 ;;
        --top-k)        TOP_K="$2";        shift 2 ;;
        --memory-type)  MEMORY_TYPE="$2";  shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2";  shift 2 ;;
        --baseline)     BASELINE="$2";     shift 2 ;;
        --skip-compare) SKIP_COMPARE=true; shift ;;
        --no-memory)    NO_MEMORY=true;    shift ;;
        --help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Sanitize memory type for use in file paths (mirrors Python's _sanitize_for_filename)
MEMORY_TYPE_SAFE="${MEMORY_TYPE//\//_}"
MEMORY_TYPE_SAFE="${MEMORY_TYPE_SAFE//:/_}"

INPUT_FILE="${INPUT_NAME}.jsonl"

# Build infer args
INFER_ARGS=(
    "--model" "${INFER_MODEL}"
    "--input" "${INPUT_FILE}"
    "--workers" "${WORKERS}"
)
if [[ "${NO_MEMORY}" == true ]]; then
    INFER_ARGS+=("--no-memory")
else
    INFER_ARGS+=("--top-k" "${TOP_K}" "--memory-type" "${MEMORY_TYPE}")
    if [[ "${MEMORY_TYPE}" == "qwen3_embedding_4b" ]]; then
        INFER_ARGS+=("--embed-provider" "qwen3" "--embed-model" "Qwen/Qwen3-Embedding-4B")
    fi
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
    INFER_ARGS+=("--max-samples" "${MAX_SAMPLES}")
fi

echo "============================================================"
echo "Step 1: Context-memory inference"
echo "  Input:   ${INPUT_FILE}"
echo "  Model:   ${INFER_MODEL}"
if [[ "${NO_MEMORY}" == true ]]; then
    echo "  Memory:  none (disabled)"
else
    echo "  Memory:  ${MEMORY_TYPE}"
    echo "  Top-K:   ${TOP_K}"
fi
echo "  Workers (infer): ${WORKERS}"
echo "  Workers (eval):  ${EVAL_WORKERS}"
[[ -n "${MAX_SAMPLES}" ]] && echo "  Max samples: ${MAX_SAMPLES}"
echo "============================================================"

python infer_context_memory.py "${INFER_ARGS[@]}"

# Find the inference output file
if [[ "${NO_MEMORY}" == true ]]; then
    INFER_OUTPUT=$(ls -t "outputs/context_nomemory/${INPUT_NAME}_"*"_ctx_nomemory_"*.jsonl 2>/dev/null \
        | grep -v "_graded" | head -1)
else
    INFER_OUTPUT=$(ls -t "outputs/context_${MEMORY_TYPE_SAFE}/${INPUT_NAME}_"*"_ctx_memory_topk${TOP_K}"*.jsonl 2>/dev/null \
        | grep -v "_graded" | head -1)
fi

if [[ -z "${INFER_OUTPUT}" ]]; then
    echo "ERROR: Could not find inference output file."
    exit 1
fi

GRADED_OUTPUT="${INFER_OUTPUT%.jsonl}_graded.jsonl"

echo ""
echo "============================================================"
echo "Step 2: Evaluation (LLM-as-judge)"
echo "  Input:   ${INFER_OUTPUT}"
echo "  Output:  ${GRADED_OUTPUT}"
echo "  Judge:   ${JUDGE_MODEL}"
echo "============================================================"

python eval.py \
    --input "${INFER_OUTPUT}" \
    --output "${GRADED_OUTPUT}" \
    --judge-model "${JUDGE_MODEL}" \
    --workers "${EVAL_WORKERS}"

if [[ "${SKIP_COMPARE}" == true || -z "${BASELINE}" ]]; then
    echo ""
    echo "Skipping comparison analysis (no --baseline provided or --skip-compare set)."
    echo ""
    echo "To run comparison later:"
    echo "  python analyze_comparison_new.py \\"
    echo "      --baseline <BASELINE_GRADED_FILE> \\"
    echo "      --baseline-infer <BASELINE_RAW_INFER_FILE> \\"
    echo "      --memory ${GRADED_OUTPUT} \\"
    echo "      --infer-memory ${INFER_OUTPUT}"
    exit 0
fi

echo ""
echo "============================================================"
echo "Step 3: Comparison analysis"
echo "  Baseline: ${BASELINE}"
echo "  Memory:   ${GRADED_OUTPUT}"
echo "============================================================"

# Derive baseline raw inference file (strip _graded suffix) for efficiency comparison
BASELINE_INFER_OUTPUT="${BASELINE/_graded.jsonl/.jsonl}"
if [[ ! -f "${BASELINE_INFER_OUTPUT}" ]]; then
    BASELINE_INFER_OUTPUT=""
fi

# Same directory as inference output; name aligned with infer stem (not _graded)
REPORT_FILE="${INFER_OUTPUT%.jsonl}_comparison_report.md"

COMPARE_ARGS=(
    "--baseline" "${BASELINE}"
    "--memory"   "${GRADED_OUTPUT}"
    "--infer-memory" "${INFER_OUTPUT}"
    "--output"   "${REPORT_FILE}"
)
[[ -n "${BASELINE_INFER_OUTPUT}" ]] && COMPARE_ARGS+=("--baseline-infer" "${BASELINE_INFER_OUTPUT}")

python analyze_comparison_new.py "${COMPARE_ARGS[@]}"

echo ""
echo "Done. Report: ${REPORT_FILE}"
