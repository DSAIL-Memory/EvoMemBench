#!/usr/bin/env bash
# H-Nav remote setup — ozonderlab2 (2x RTX 4090, CUDA 13.0 driver)
#
# Run ONCE on the GPU machine after transferring the repo.
#   bash hnav/deploy/setup_remote.sh
#
# Assumes full internet (huggingface.co + PyPI). Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV="${HNAV_VENV:-$REPO_ROOT/.venv-hnav}"
EMBED_MODEL="${HNAV_EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
PY="${HNAV_PYTHON:-python3.11}"

echo "=============================================================="
echo " H-Nav remote setup"
echo "   repo   : $REPO_ROOT"
echo "   venv   : $VENV"
echo "   embed  : $EMBED_MODEL"
echo "=============================================================="

# ── 1. Python ────────────────────────────────────────────────────────────────
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "!! $PY not found."
    echo "   Either module-load a 3.11 toolchain, or set HNAV_PYTHON=python3.x"
    echo "   (3.10-3.12 all fine; avoid 3.13 — torch wheel coverage is patchy)."
    exit 1
fi
echo "[1/6] Python: $($PY -V)"

# ── 2. venv ──────────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
    echo "[2/6] created venv"
else
    echo "[2/6] venv exists, reusing"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

# ── 3. torch (CUDA) ──────────────────────────────────────────────────────────
# cu124 wheels run fine under a CUDA 13.0 driver (drivers are backward
# compatible with older runtimes). Override with HNAV_TORCH_INDEX if needed.
TORCH_INDEX="${HNAV_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[3/6] torch already present with CUDA"
else
    echo "[3/6] installing torch from $TORCH_INDEX ..."
    python -m pip install -q torch --index-url "$TORCH_INDEX"
fi

# ── 4. Python deps ───────────────────────────────────────────────────────────
echo "[4/6] installing python deps ..."
python -m pip install -q \
    numpy scipy scikit-learn \
    transformers accelerate safetensors sentencepiece \
    huggingface_hub \
    nltk tiktoken \
    openai python-dotenv httpx tqdm

# ── 5. Pre-cache tokenizer / NLTK data (needed by chunk_text_into_sentences) ──
echo "[5/6] pre-caching nltk + tiktoken ..."
python - <<'PYEOF'
import nltk, tiktoken
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"   warn: nltk {pkg}: {e}")
try:
    tiktoken.encoding_for_model("gpt-4o-mini")
except Exception as e:
    print(f"   warn: tiktoken: {e}")
print("   ok")
PYEOF

# ── 6. Embedding model weights ───────────────────────────────────────────────
echo "[6/6] fetching $EMBED_MODEL (~8GB, first run only) ..."
python - "$EMBED_MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
path = snapshot_download(model, allow_patterns=[
    "*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
print(f"   cached at: {path}")
PYEOF

echo
echo "=============================================================="
echo " Setup complete."
echo
echo "   source $VENV/bin/activate"
echo "   cp hnav/deploy/.env.template .env     # then edit"
echo "   python hnav/deploy/check_env.py"
echo "=============================================================="
