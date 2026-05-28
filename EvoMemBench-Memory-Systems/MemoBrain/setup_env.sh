#!/bin/bash
# Full environment setup for MemoBrain.
# Run from the EvoMemBench-Memory-Systems repo root:
#   bash MemoBrain/setup_env.sh [OPTIONS]
#
# Options:
#   --conda-env     Conda env name           (default: memobrain)
#   --python        Python version           (default: 3.10)
#   --download-dir  HuggingFace cache dir    (default: ./llms)
#   --prefetch      Pre-download model weights (flag, off by default)
#   --skip-vllm     Skip vLLM install (useful if already installed)

set -euo pipefail

# ── Detect where this script lives so we always find the package ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="memobrain"
PYTHON_VER="3.10"
DOWNLOAD_DIR="${HF_HOME:-$REPO_ROOT/llms}"
PREFETCH=0
SKIP_VLLM=0
MODEL="TommyChien/MemoBrain-14B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)    CONDA_ENV="$2";    shift 2 ;;
        --python)       PYTHON_VER="$2";   shift 2 ;;
        --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
        --prefetch)     PREFETCH=1;        shift   ;;
        --skip-vllm)    SKIP_VLLM=1;       shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo " MemoBrain environment setup"
echo "=========================================="
echo "  Conda env   : $CONDA_ENV"
echo "  Python      : $PYTHON_VER"
echo "  Package dir : $SCRIPT_DIR"
echo "  Model cache : $DOWNLOAD_DIR"
echo "  Pre-fetch   : $([ $PREFETCH -eq 1 ] && echo yes || echo no)"
echo "=========================================="

# ── Step 1: conda environment ─────────────────────────────────────────────────
echo ""
echo "[1/4] Setting up conda environment '$CONDA_ENV' ..."

# Source conda so it's available in non-interactive shells
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found. Install Miniconda/Anaconda first."
    exit 1
}
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "  Env '$CONDA_ENV' already exists — skipping creation."
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VER" -y
    echo "  Created env '$CONDA_ENV'."
fi

conda activate "$CONDA_ENV"
echo "  Activated '$CONDA_ENV'."

# ── Step 2: pip / vLLM ───────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing server dependencies ..."
pip install --upgrade pip --quiet

if [[ $SKIP_VLLM -eq 1 ]]; then
    echo "  --skip-vllm set, skipping vLLM install."
else
    pip install vllm
fi

# Smoke test
python -c "import vllm; print('  vLLM', vllm.__version__, 'OK')"

# ── Step 3: MemoBrain package ─────────────────────────────────────────────────
echo ""
echo "[3/4] Installing MemoBrain package ..."
pip install -e "$SCRIPT_DIR" --quiet
python -c "from memobrain import MemoBrain; print('  MemoBrain import OK')"

# ── Step 4: optional model pre-fetch ─────────────────────────────────────────
echo ""
if [[ $PREFETCH -eq 1 ]]; then
    echo "[4/4] Pre-downloading model weights for '$MODEL' ..."
    export HF_HOME="$DOWNLOAD_DIR"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    # hf-mirror.com is a domestic mirror; bypass the local proxy to avoid SSL errors
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    echo "  HF endpoint: $HF_ENDPOINT"
    for attempt in 1 2 3 4 5; do
        echo "  Download attempt $attempt/5 ..."
        hf download "$MODEL" && break
        echo "  Attempt $attempt failed, retrying in 10s ..."
        sleep 10
    done
    echo "  Model cached in $DOWNLOAD_DIR"
else
    echo "[4/4] Skipping model pre-download (pass --prefetch to download now)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Setup complete."
echo ""
echo " To start the server (single GPU ≥ 40 GB):"
echo "   bash MemoBrain/run_server.sh --conda-env $CONDA_ENV --gpus 0"
echo ""
echo " Two GPUs with tensor parallelism (2 × 24 GB):"
echo "   bash MemoBrain/run_server.sh --conda-env $CONDA_ENV --gpus 0,1 --tp 2"
echo ""
echo " Verify the server is up:"
echo "   curl http://localhost:8001/v1/models"
echo "=========================================="
