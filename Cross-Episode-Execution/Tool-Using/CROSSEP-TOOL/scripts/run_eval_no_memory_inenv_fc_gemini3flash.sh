#!/usr/bin/env bash
# Cross-episode no-memory evaluator using gpt-5-mini (zhizengzeng OpenAI-compatible gateway).
# Identical flow to run_eval_no_memory_inenv_fc.sh but routes through the gemini3flash handler.
#
# Usage:
#   bash scripts/run_eval_no_memory_inenv_fc_gemini3flash.sh \
#     [--limit-per-subset N]   # default: all 50 per subset
#     [--workers N]            # default: 25 (per subset; total = 4 × N)
#     [--run-name NAME]        # default: inenv_fc_gemini3flash
#     [--keep-going]           # continue to merge even if some subsets failed
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="CROSSEP-TOOL"
LIMIT_PER_SUBSET=""
WORKERS=25
RUN_NAME="inenv_fc_gemini3flash"
KEEP_GOING=false

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit-per-subset) LIMIT_PER_SUBSET="$2"; shift 2 ;;
        --workers)          WORKERS="$2";          shift 2 ;;
        --run-name)         RUN_NAME="$2";         shift 2 ;;
        --keep-going)       KEEP_GOING=true;       shift   ;;
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
IDS_DIR="bfcl_eval/scripts/cross_episode/ids"

extract_ids() {
    local subset="$1"
    python3 -c \
        "import json,sys; d=json.load(open(sys.argv[1]))['multi_turn_ours']; \
         n=int(sys.argv[2]) if sys.argv[2] else len(d); print(','.join(d[:n]))" \
        "${IDS_DIR}/ids_${subset}.json" "${LIMIT_PER_SUBSET:-}"
}

# ── Paths and timestamps ──────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
MEMORY_TYPE="no_memory"
BASE_DIR="cross_episode_results/runs/${RUN_NAME}_${MEMORY_TYPE}_${TS}"
mkdir -p "${BASE_DIR}/logs"

SUBSETS=(gorilla_fs vehicle_control trading_bot travel_api)

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  BFCL Cross-Episode  [no memory | in-env | FC mode | gemini-3-flash]"
echo "  Model:           ${BATCH_MODEL}"
echo "  Base dir:        ${BASE_DIR}"
echo "  Workers/subset:  ${WORKERS}"
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
    python -m bfcl_eval.scripts.cross_episode.run_batch_eval \
        --num-workers "${WORKERS}" \
        --output-dir  "${BASE_DIR}/phase1" \
        --run-name    "${subset}" \
        --memory-type none \
        --mode        FC \
        --ids         "$ids" \
        >"$log" 2>&1 &
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
