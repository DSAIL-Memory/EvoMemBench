#!/bin/bash
# Start vLLM OpenAI-compatible server for MemoBrain-14B (memory processor)
# Prerequisites: run setup_vllm_env.sh first
# Usage: bash run_memobrain_server.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-memobrain
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf_cache}"

python -m vllm.entrypoints.openai.api_server \
    --model TommyChien/MemoBrain-14B \
    --download-dir "${VLLM_DOWNLOAD_DIR:-${HF_HOME}}" \
    --port "${VLLM_PORT:-8001}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-2}" \
    --dtype "${VLLM_DTYPE:-bfloat16}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
