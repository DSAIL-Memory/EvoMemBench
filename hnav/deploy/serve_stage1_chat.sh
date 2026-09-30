#!/usr/bin/env bash
# Stage-1 FROZEN chat substrate — :8003 on GPU1.  [T11]
#
# One configuration for calibration grading, the A/A noise measurement and the
# sh_64k confirmatory campaign. Changing ANY flag after the A/A runs re-opens
# the noise measurement (stage0_results/stage1_preregistration.md) — do not.
#
# Frozen flags and why:
#   --max-model-len 65536        sh_64k RAG prompts are ~53k tokens (m3
#                                measured top-10 chunk payloads 49.6-52.2k +
#                                template); 65536 covers them with margin.
#   --kv-cache-dtype fp8         m3 precedent (:8003, KAPI_KARARI deviation 3);
#                                bf16 KV at 65k (9.2 GiB) would not leave room
#                                for the embed server on the same card.
#   --enforce-eager              m3 + t4-Part-2 precedent; no cudagraph memory.
#   --max-num-seqs 1             t4 Part 2 determinism precedent; the benchmark
#                                asks one question at a time anyway.
#   --no-enable-prefix-caching   t4 Part 2: prefix caching is the measured
#                                source of run-to-run coupling on :8000.
#   --gpu-memory-utilization 0.58  weights 7.61 GiB (bf16) + fp8 KV ~5.6 GiB
#                                at 65536 + activations; leaves 0.33 for the
#                                bf16 embed server (serve_stage1_embed.sh) and
#                                ~2 GiB slack on a 23.52 GiB card.
#   GPU0 is NOT used: the user's :8000 (PID 52259/52520) owns it and we never
#   touch processes we did not start.
#
#   nohup bash hnav/deploy/serve_stage1_chat.sh > hnav/_out/pipeline/chat_stage1.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p hnav/_out/pipeline

# The vLLM 0.9.1 install this box already serves with (:8000, :8002, :8003
# history). We only READ the binary; we never touch the user's processes.
VLLM_BIN="${HNAV_VLLM_BIN:-/mnt/nvmes/nvme1/egekutlu/programs/conda/vllm_0.9.1/bin/vllm}"
MODEL="${HNAV_STAGE1_MODEL:-/mnt/nvmes/nvme1/egekutlu/models/Qwen3-4B-Instruct-2507}"
# [multi-model] the advertised name must track the weights, or a new model is
# served under the reference model's name and every artifact mislabels it.
# Default preserves the frozen Stage-1 substrate exactly.
SERVED_NAME="${HNAV_STAGE1_SERVED_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
# sh_64k prompts reach ~42.5k tokens; anything below ~48k cannot run it.
MAXLEN="${HNAV_STAGE1_MAX_MODEL_LEN:-65536}"
PORT="${HNAV_STAGE1_CHAT_PORT:-8003}"
DEV="${HNAV_STAGE1_CHAT_DEVICE:-1}"

# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh
hnav_gpu_report
hnav_gpu_guard "$DEV" 13 || {
    echo "Aborting — GPU$DEV lacks the ~13 GiB the frozen chat config needs."
    exit 1
}

echo "Stage-1 chat: $MODEL on :$PORT (GPU$DEV), frozen flags — see header."
CUDA_VISIBLE_DEVICES="$DEV" exec "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAXLEN" \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.58
