#!/bin/bash
# Full environment setup for RL-MemoryAgent vLLM server.
# Run from the EvoMemBench-Memory-Systems repo root:
#   bash memagent/setup_env.sh [OPTIONS]
#
# Options:
#   --conda-env     Conda env name           (default: vllm_memagent)
#   --python        Python version           (default: 3.10)
#   --download-dir  HuggingFace cache dir    (default: ./llms)
#   --prefetch      Pre-download model weights (flag, off by default)
#
# Note: vLLM 0.8.5 + PyTorch 2.6+cu124 is pinned intentionally.
# Newer vLLM (>=0.20) requires CUDA 13.x and will not work on machines
# with CUDA 12.x drivers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="vllm_memagent"
PYTHON_VER="3.10"
DOWNLOAD_DIR="${HF_HOME:-$REPO_ROOT/llms}"
PREFETCH=0
MODEL="BytedTsinghua-SIA/RL-MemoryAgent-14B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)    CONDA_ENV="$2";    shift 2 ;;
        --python)       PYTHON_VER="$2";   shift 2 ;;
        --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
        --prefetch)     PREFETCH=1;        shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo " RL-MemoryAgent environment setup"
echo "=========================================="
echo "  Conda env   : $CONDA_ENV"
echo "  Python      : $PYTHON_VER"
echo "  Model cache : $DOWNLOAD_DIR"
echo "  Pre-fetch   : $([ $PREFETCH -eq 1 ] && echo yes || echo no)"
echo "=========================================="

# ── Step 1: conda environment ─────────────────────────────────────────────────
echo ""
echo "[1/3] Setting up conda environment '$CONDA_ENV' ..."

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

# ── Step 2: pip / vLLM (pinned for CUDA 12.x) ────────────────────────────────
echo ""
echo "[2/3] Installing server dependencies (pinned for CUDA 12.4) ..."
pip install --upgrade pip --quiet

pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 --quiet
python -c "import torch; print('  torch', torch.__version__, 'OK')"

pip install vllm==0.8.5 --quiet
python -c "import vllm; print('  vLLM', vllm.__version__, 'OK')"

pip install transformers==4.51.3 huggingface_hub --quiet
python -c "import transformers; print('  transformers', transformers.__version__, 'OK')"

# ── Step 3: optional model pre-fetch ─────────────────────────────────────────
echo ""
if [[ $PREFETCH -eq 1 ]]; then
    echo "[3/3] Pre-downloading model weights for '$MODEL' ..."
    export HF_HOME="$DOWNLOAD_DIR"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
    echo "  HF endpoint: $HF_ENDPOINT"
    for attempt in 1 2 3 4 5; do
        echo "  Download attempt $attempt/5 ..."
        huggingface-cli download "$MODEL" && break
        echo "  Attempt $attempt failed, retrying in 10s ..."
        sleep 10
    done
    echo "  Model cached in $DOWNLOAD_DIR"
else
    echo "[3/3] Skipping model pre-download (pass --prefetch to download now)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Setup complete."
echo ""
echo " To start the server (two GPUs, tensor parallelism):"
echo "   bash memagent/run_server.sh --gpus 1,2 --tp 2"
echo ""
echo " Single GPU (≥ 40 GB):"
echo "   bash memagent/run_server.sh --gpus 0 --tp 1"
echo ""
echo " Verify the server is up:"
echo "   curl http://localhost:8000/v1/models"
echo "=========================================="
