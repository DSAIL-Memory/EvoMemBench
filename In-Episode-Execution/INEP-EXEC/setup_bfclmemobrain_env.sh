#!/bin/bash
# Create and configure the bfclmemobrain conda environment.
# Installs: vllm (via bfcl_eval[oss_eval_vllm]), bfcl_eval, memobrain
# Usage: bash setup_bfclmemobrain_env.sh

set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="bfclmemobrain"
BFCL_DIR="/data/chimo/BFCL/in-episode/ours"
MEMOBRAIN_DIR="/data/chimo/MemoBrain"

echo "=== Creating conda environment: $ENV_NAME (Python 3.10) ==="
conda create -y -n "$ENV_NAME" python=3.10

conda activate "$ENV_NAME"

echo "=== Installing bfcl_eval + vllm ==="
pip install -e "$BFCL_DIR[oss_eval_vllm]"

echo "=== Installing memobrain ==="
pip install -e "$MEMOBRAIN_DIR"

echo "=== Pinning transformers to vllm-compatible version ==="
pip install transformers==4.51.3

echo "=== Installing volcengine SDK (required for Ark/DeepSeek inference) ==="
pip install volcengine-python-sdk --index-url https://pypi.org/simple/

echo "=== Done. Activate with: conda activate $ENV_NAME ==="
