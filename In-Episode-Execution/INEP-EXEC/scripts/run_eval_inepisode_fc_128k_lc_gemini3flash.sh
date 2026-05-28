#!/usr/bin/env bash
# In-episode evaluator (FC mode, long-context, gpt-5-mini): 4 subsets in parallel.
# Long-context noise injection ON by default.
#
# Usage:
#   bash scripts/run_eval_inepisode_fc_128k_lc_gemini3flash.sh \
#     [--limit-per-subset N]    # default: all 50 per subset
#     [--workers N]             # default: 25 (per subset; total = 4 × N)
#     [--run-name NAME]         # default: inepisode_fc_128k_lc_gemini3flash
#     [--context-budget N]      # token budget per LLM call (default: 128000 = 128k)
#     [--keep-going]            # continue to merge even if some subsets failed
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="INEP-EXEC"
LIMIT_PER_SUBSET=""
WORKERS=25
RUN_NAME="inepisode_fc_128k_lc_gemini3flash"
CONTEXT_BUDGET=128000
KEEP_GOING=false
LONG_CONTEXT=true

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit-per-subset)   LIMIT_PER_SUBSET="$2"; shift 2 ;;
        --workers)            WORKERS="$2";           shift 2 ;;
        --run-name)           RUN_NAME="$2";          shift 2 ;;
        --context-budget)     CONTEXT_BUDGET="$2";    shift 2 ;;
        --keep-going)         KEEP_GOING=true;        shift   ;;
        --long-context)       LONG_CONTEXT=true;      shift   ;;
        *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Activate conda env ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Move to repo root (one level up from this script) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ── Verify required env vars ─────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[ERROR] .env not found in $SCRIPT_DIR — please create it from bfcl_eval/.env.example"
    exit 1
fi
set -a
# shellcheck source=/dev/null
source .env
set +a

for var in GEMINI3FLASH_API_KEY BATCH_MODEL_GEMINI3FLASH; do
    val="${!var:-}"
    if [[ -z "$val" || "$val" == *"XXXXXX"* ]]; then
        echo "[ERROR] $var is not set (or still placeholder) in .env"
        exit 1
    fi
done

export BATCH_BACKEND=gemini3flash
export BATCH_MODEL="$BATCH_MODEL_GEMINI3FLASH"
export OPENAI_API_KEY="$GEMINI3FLASH_API_KEY"
export OPENAI_BASE_URL="$GEMINI3FLASH_BASE_URL"

# ── Helpers ───────────────────────────────────────────────────────────────────
IDS_DIR="bfcl_eval/scripts/in_episode/ids"

extract_ids() {
    local subset="$1"
    python3 -c \
        "import json,sys; d=json.load(open(sys.argv[1]))['multi_turn_ours']; \
         n=int(sys.argv[2]) if sys.argv[2] else len(d); print(','.join(d[:n]))" \
        "${IDS_DIR}/ids_${subset}.json" "${LIMIT_PER_SUBSET:-}"
}

# ── Paths and timestamps ──────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
MEMORY_TYPE="inepisode"
BASE_DIR="in_episode_results/runs/${RUN_NAME}_${MEMORY_TYPE}_${TS}"
mkdir -p "${BASE_DIR}/logs"

SUBSETS=(gorilla_fs vehicle_control trading_bot travel_api)

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  BFCL In-Episode  [FC mode | gpt-5-mini]"
echo "  Model:           ${BATCH_MODEL}"
echo "  Base dir:        ${BASE_DIR}"
echo "  Workers/subset:  ${WORKERS}"
echo "  Context budget:  ${CONTEXT_BUDGET} tokens"
echo "  Long context:    ON"
[[ -n "$LIMIT_PER_SUBSET" ]] && echo "  Limit/subset:    ${LIMIT_PER_SUBSET}"
echo "================================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Subset jobs in parallel
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Run] Launching ${#SUBSETS[@]} subset jobs in parallel..."

pids=()
names=()

for subset in "${SUBSETS[@]}"; do
    ids="$(extract_ids "$subset")"
    log="${BASE_DIR}/logs/phase1_${subset}.log"

    cmd=(
        python -m bfcl_eval.scripts.in_episode.run_batch_eval
        --num-workers "${WORKERS}"
        --output-dir  "${BASE_DIR}/phase1"
        --run-name    "${subset}"
        --context-budget "${CONTEXT_BUDGET}"
        --mode        FC
        --ids         "$ids"
    )
    [[ "$LONG_CONTEXT" == "true" ]] && cmd+=(--long-context)

    "${cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
    names+=("$subset")
    echo "  started ${subset} (pid $!), log: $log"
done

failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        failed+=("${names[$i]}")
        echo "[Run] FAILED: ${names[$i]} — see ${BASE_DIR}/logs/phase1_${names[$i]}.log"
    else
        echo "[Run] done:   ${names[$i]}"
    fi
done

if [[ ${#failed[@]} -gt 0 && "$KEEP_GOING" != "true" ]]; then
    echo "[Run] ${#failed[@]} subset(s) failed: ${failed[*]}"
    echo "[Run] Aborting before merge. Pass --keep-going to merge anyway."
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Merge summaries
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Merge] Building combined summary..."
python3 scripts/merge_summaries.py "${BASE_DIR}"

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results at:              ${BASE_DIR}"
echo "  combined_summary.txt:    ${BASE_DIR}/combined_summary.txt"
echo "  combined_summary.csv:    ${BASE_DIR}/combined_summary.csv"
echo "  combined_per_sample.csv: ${BASE_DIR}/combined_per_sample.csv"
echo "================================================================"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "[WARN] ${#failed[@]} subset(s) did not complete successfully: ${failed[*]}"
fi
exit "${#failed[@]}"
