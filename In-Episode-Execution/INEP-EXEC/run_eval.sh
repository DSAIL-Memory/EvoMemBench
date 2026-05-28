#!/usr/bin/env bash
# In-episode batch evaluation runner.
# Usage:  bash run_eval.sh [--limit N] [--workers N] [--run-name NAME] [extra args...]
# The script activates the 'bfcl' conda environment automatically.
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="bfcl"
NUM_WORKERS=8
LIMIT=""
RUN_NAME="auto"
EXTRA_ARGS=()

# ── Parse simple overrides ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)    LIMIT="$2";      shift 2 ;;
        --workers)  NUM_WORKERS="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2";   shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Activate conda env ────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Move to repo root (directory this script lives in) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Verify required env vars ─────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[ERROR] .env not found in $SCRIPT_DIR — please create it from bfcl_eval/.env.example"
    exit 1
fi
# Load .env so we can check the keys exist
set -a
# shellcheck source=/dev/null
source .env
set +a

for var in DEEPSEEK_API_KEY BATCH_MODEL; do
    val="${!var:-}"
    if [[ -z "$val" || "$val" == *"XXXXXX"* ]]; then
        echo "[ERROR] $var is not set (or still placeholder) in .env"
        exit 1
    fi
done

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python -m bfcl_eval.scripts.in_episode.run_batch_eval
    --num-workers "$NUM_WORKERS"
    --run-name    "$RUN_NAME"
    --output-dir  in_episode_results
)
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
CMD+=("${EXTRA_ARGS[@]}")

echo "============================================================"
echo "  BFCL In-Episode Batch Eval"
echo "  Model:    ${BATCH_MODEL}"
echo "  Workers:  ${NUM_WORKERS}"
[[ -n "$LIMIT" ]] && echo "  Limit:    ${LIMIT}"
echo "  Run name: ${RUN_NAME}"
echo "  Command:  ${CMD[*]}"
echo "============================================================"

exec "${CMD[@]}"
