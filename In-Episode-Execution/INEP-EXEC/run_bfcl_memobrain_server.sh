#!/bin/bash
# Start a dedicated vLLM server for BFCL MemoBrain (GPU 5, port 8002).
# Uses the bfclmemobrain conda environment.
# Prerequisites: run setup_bfclmemobrain_env.sh first
# Usage: bash run_bfcl_memobrain_server.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bfclmemobrain
export CUDA_VISIBLE_DEVICES=5
export HF_HOME=/data/chimo/llms

python -m vllm.entrypoints.openai.api_server \
    --model TommyChien/MemoBrain-14B \
    --download-dir /data/chimo/llms \
    --port 8002 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95
