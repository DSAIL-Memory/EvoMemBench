#!/usr/bin/env bash
# Stage-1 FROZEN embedding endpoint — :8001, GPU1, **bfloat16**.  [T11]
#
# DECLARED DEVIATION (same shape as the t4/S2 Part-2 one, and for the same
# arithmetic): the campaign chat window (65536, fp8 KV, ~13 GiB with weights)
# plus an fp32 embed server (~16 GiB weights alone) cannot share one 23.52 GiB
# card, and GPU0 belongs to the user's :8000. bf16 is the checkpoint's native
# dtype; t4 Part 2 already served it at --gpu-memory-utilization 0.33 next to
# a chat server on the same card, deterministically.
#
# What this affects and what it does not:
#   * benchmark retrieval (chunk + query embeddings) runs in bf16 — BOTH arms
#     (off and live) see the identical retrieval, so the paired campaign
#     comparison is internally valid;
#   * H-Nav's GATE geometry does NOT run in bf16: fact vectors come from the
#     fp32 disk cache (cache-first, endpoint fallback NON-persisting —
#     hnav/adapters/mab_adapter._build_components), so the calibrated
#     cos_pair/r_min thresholds keep their fp32 space. Cache misses are
#     counted; the campaign analysis treats misses > 0 as a flag.
#   * the frozen nmargin/H_z ambiguity constants were calibrated on fp32
#     rankings; their firing rate under bf16 rankings is a declared transfer
#     assumption, bounded by the A/A + shadow coverage checks on sh_32k.
#
#   nohup bash hnav/deploy/serve_stage1_embed.sh > hnav/_out/pipeline/embed_stage1.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p hnav/_out/pipeline

VLLM_BIN="${HNAV_VLLM_BIN:-/mnt/nvmes/nvme1/egekutlu/programs/conda/vllm_0.9.1/bin/vllm}"
MODEL="${HNAV_EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
PORT="${HNAV_EMBED_PORT:-8001}"
DEV="${HNAV_STAGE1_EMBED_DEVICE:-1}"

# shellcheck source=/dev/null
source hnav/deploy/_activate.sh   # HF_HOME on the nvme so weights resolve
# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh
hnav_gpu_report
hnav_gpu_guard "$DEV" 8 || {
    echo "Aborting — GPU$DEV lacks the ~8 GiB the bf16 embed server needs."
    exit 1
}

echo "Stage-1 embed: $MODEL on :$PORT (GPU$DEV), bf16 0.33 — see header for the deviation."
CUDA_VISIBLE_DEVICES="$DEV" exec "$VLLM_BIN" serve "$MODEL" \
    --task embed \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.33 \
    --served-model-name "$MODEL"
