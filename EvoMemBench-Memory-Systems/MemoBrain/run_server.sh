#!/bin/bash
# Start a local vLLM server for TommyChien/MemoBrain-14B.
# Usage: bash run_server.sh [OPTIONS]
#
# Options (all optional):
#   --port              Port to serve on              (default: 8001)
#   --gpus              CUDA_VISIBLE_DEVICES value    (default: 0)
#   --tp                Tensor parallel size          (default: 1)
#   --model             Model name/path               (default: TommyChien/MemoBrain-14B)
#   --download-dir      HuggingFace cache dir         (default: ~/.cache/huggingface)
#   --max-model-len     Max context length in tokens  (default: 32768)
#   --gpu-util          GPU memory utilization ratio  (default: 0.95)
#   --conda-env         Conda environment to activate (default: none)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PORT=8001
GPUS=2,3
TP=2
MODEL="TommyChien/MemoBrain-14B"
DOWNLOAD_DIR="${HF_HOME:-$REPO_ROOT/llms}"
MAX_MODEL_LEN=32768
GPU_UTIL=0.85
CONDA_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)         PORT="$2";         shift 2 ;;
        --gpus)         GPUS="$2";         shift 2 ;;
        --tp)           TP="$2";           shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util)     GPU_UTIL="$2";     shift 2 ;;
        --conda-env)    CONDA_ENV="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="$DOWNLOAD_DIR"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
unset ALL_PROXY all_proxy

# Resolve model path from local HF cache to avoid huggingface_hub version issues
if [[ "$MODEL" != /* ]]; then
    OWNER="${MODEL%%/*}"
    NAME="${MODEL##*/}"
    SNAPSHOT_DIR=$(ls -d "$DOWNLOAD_DIR/hub/models--${OWNER}--${NAME}/snapshots/"*/ 2>/dev/null | tail -1)
    if [[ -n "$SNAPSHOT_DIR" ]]; then
        MODEL="${SNAPSHOT_DIR%/}"
        echo "  Resolved local path: $MODEL"
    fi
fi

echo "Starting MemoBrain vLLM server:"
echo "  Model:       $MODEL"
echo "  Port:        $PORT"
echo "  GPUs:        $GPUS  (tp=$TP)"
echo "  Max length:  $MAX_MODEL_LEN"
echo "  Download dir: $DOWNLOAD_DIR"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --download-dir "$DOWNLOAD_DIR" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL"
