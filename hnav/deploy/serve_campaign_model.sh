#!/usr/bin/env bash
# Serve ONE campaign answering model on :8003 / GPU1.            [E2E-5]
#
# A separate script from serve_stage1_chat.sh on purpose: that one is the
# FROZEN Stage-1 substrate for the reference Qwen3-4B result and its flags are
# load-bearing for a measurement that is already committed. This one keeps the
# determinism-critical half of those flags identical and takes everything
# model-specific from a models.d/*.env file, so a substituted model can never
# silently change the serving contract.
#
#   bash hnav/deploy/serve_campaign_model.sh hnav/deploy/models.d/01_phi4_mini.env
#
# Normally invoked by run_model_campaign.sh, which owns the lifecycle.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONF="${1:?usage: serve_campaign_model.sh <models.d/*.env>}"
[ -f "$CONF" ] || { echo "no such model config: $CONF"; exit 2; }
# shellcheck disable=SC1090
source "$CONF"

: "${MODEL_KEY:?}" "${MODEL_PATH:?}" "${SERVED_NAME:?}" "${VLLM_ENV:?}"
: "${MAX_MODEL_LEN:?}" "${GPU_MEM_UTIL:?}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"

# ── explicit overrides, applied AFTER the source ─────────────────────────────
# `source "$CONF"` overwrites the caller's environment, so `KV_CACHE_DTYPE=auto
# bash serve_campaign_model.sh conf` silently served fp8 anyway. The KV-dtype
# A/B would have compared fp8 against fp8 and "proved" the dtype innocent.
# Diagnostics therefore set HNAV_FORCE_*, which nothing in a config file can
# clobber, and the resolved values are echoed below so the log is the record.
VLLM_ENV="${HNAV_FORCE_VLLM_ENV:-$VLLM_ENV}"
KV_CACHE_DTYPE="${HNAV_FORCE_KV_DTYPE:-$KV_CACHE_DTYPE}"
MAX_MODEL_LEN="${HNAV_FORCE_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
GPU_MEM_UTIL="${HNAV_FORCE_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"
EXTRA_SERVE_FLAGS="${HNAV_FORCE_EXTRA_SERVE_FLAGS-$EXTRA_SERVE_FLAGS}"

NVME="${HNAV_NVME:-/mnt/nvmes/nvme1/egekutlu}"
export HF_HOME="${HF_HOME:-$NVME/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$NVME/.cache/tmp}"
mkdir -p "$TMPDIR"

case "$VLLM_ENV" in
  legacy) VLLM_BIN="${HNAV_VLLM_BIN:-$NVME/programs/conda/vllm_0.9.1/bin/vllm}" ;;
  modern)
    VLLM_BIN="${HNAV_VLLM_MODERN_BIN:-$NVME/programs/venvs/vllm_modern/bin/vllm}"
    # This box has no CUDA toolkit (`nvcc` absent, /usr/local/cuda absent), and
    # vLLM 0.28 reaches for FlashInfer in two independent places that both JIT-
    # compile on first use: the attention backend and the SAMPLER. Either one
    # aborts engine startup with
    #   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
    # so switching only the attention backend still died, in
    # flashinfer/sampling.py::top_k_mask_logits.
    #
    # Disabling the FlashInfer sampler cannot change a single answer in this
    # campaign: the harness fixes temperature at 0, so sampling is argmax, and
    # argmax is argmax in either implementation. The attention backend is set
    # per model in models.d/, where it is a serving choice that IS visible to
    # the numerics and therefore probed before any measured cell.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    ;;
  *) echo "VLLM_ENV must be legacy|modern, got '$VLLM_ENV'"; exit 2 ;;
esac
[ -x "$VLLM_BIN" ] || { echo "vllm binary missing: $VLLM_BIN"; exit 2; }
[ -d "$MODEL_PATH" ] || { echo "weights missing: $MODEL_PATH"; exit 2; }

PORT="${HNAV_STAGE1_CHAT_PORT:-8003}"
DEV="${HNAV_STAGE1_CHAT_DEVICE:-1}"          # GPU0 belongs to the user. Never.

# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh
hnav_gpu_report
hnav_gpu_guard "$DEV" 8 || { echo "Aborting - GPU$DEV is not free."; exit 1; }

echo "== serving $MODEL_KEY =="
echo "   weights : $MODEL_PATH"
echo "   as      : $SERVED_NAME"
echo "   vllm    : $VLLM_BIN ($VLLM_ENV)"
echo "   context : $MAX_MODEL_LEN   kv=$KV_CACHE_DTYPE  util=$GPU_MEM_UTIL"
echo "   extra   : $EXTRA_SERVE_FLAGS"

# Frozen, determinism-critical (see models.d/README.md):
#   --max-num-seqs 1 --no-enable-prefix-caching --enforce-eager
# word-splitting EXTRA_SERVE_FLAGS is intentional; the values carry no spaces.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$DEV" exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --enforce-eager \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    $EXTRA_SERVE_FLAGS
