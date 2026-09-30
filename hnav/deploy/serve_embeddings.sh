#!/usr/bin/env bash
# Serve Qwen3-Embedding-4B on :8001, pinned to GPU1.
#
# NOT needed for T1/T2/T5/T7 — those embed in-process via hnav/core/embedding.py,
# so the Stage-0 gate is reachable without any serving infrastructure.
#
# REQUIRED for M0 (hnav/stage0/m0_live_fidelity.py) and T4, where the benchmark's
# own code path runs. NEXT_STEPS.md §6 "Correction 2" claims otherwise — it says
# TextRetriever loads Qwen3-Embedding-4B in-process. That is wrong. The class
# Qwen3Embedding4BEmbeddings exists at methods/embedding_retriever.py:51 but
# TextRetriever.__init__ never selects it; line 176 routes
# "Qwen/Qwen3-Embedding-4B" to LangChain OpenAIEmbeddings against
# OPENAI_BASE_URL. See hnav/deploy/mab.env.template.
#
#   tmux new -s embed
#   bash hnav/deploy/serve_embeddings.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source hnav/deploy/_activate.sh
# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh

MODEL="${HNAV_EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
PORT="${HNAV_EMBED_PORT:-8001}"
DEV="${HNAV_EMBED_DEVICE:-1}"
# Serve in the campaign's pinned dtype (hnav/config.py, default float32 — what
# T1 calibrated with). vLLM's default is the checkpoint's bf16, whose
# quantization leaves "normalized" vectors at norms 1±2e-3 and fired gate S1
# in M0 on 2026-08-14. Do not drop this flag.
DTYPE="${HNAV_EMBED_DTYPE:-float32}"

if ! python -c "import vllm" 2>/dev/null; then
    echo "vllm not installed in the '${HNAV_CONDA_ENV:-hnav}' conda env."
    echo "    python -m pip install vllm"
    echo
    echo "Note: GPU0 already runs the LLM server. This script pins the embedder"
    echo "to GPU${DEV} via CUDA_VISIBLE_DEVICES so the two never contend."
    echo "Do not remove that pin."
    exit 1
fi

# vLLM takes --gpu-memory-utilization 0.85 of the WHOLE card, so it needs the
# card effectively to itself. Refuse if something else is already on it.
hnav_gpu_report
echo
hnav_gpu_guard "$DEV" 9 || {
    echo "Aborting — not starting a second allocator on a busy card."
    exit 1
}

echo "Serving $MODEL on :$PORT (GPU$DEV) ..."
# --task embed is the flag on vLLM >=0.6; older builds use --task embedding and
# newer ones --runner pooling. If this errors, check `vllm serve --help`.
CUDA_VISIBLE_DEVICES="$DEV" exec vllm serve "$MODEL" \
    --task embed \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "$MODEL"
