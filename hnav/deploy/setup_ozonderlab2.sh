#!/usr/bin/env bash
# H-Nav setup for the lab GPU boxes (ozonderlab2 and friends).
#
#   bash hnav/deploy/setup_ozonderlab2.sh
#
# Lab policy, respected here:
#   - Do NOT install your own conda. The system-wide install at /opt/anaconda3
#     is used; this script only CREATES AN ENV inside it.
#   - `conda init` is not run at login, so every script sources the lab helper
#     (`source activate_conda`) itself. See hnav/deploy/_activate.sh.
#   - No workload manager: check nvidia-smi before anything touches a GPU.
#
# Also pins HF_HOME to the nvme — the embedder weights are ~8 GB and home dirs
# are small. Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${HNAV_CONDA_ENV:-hnav}"
PYVER="${HNAV_PYVER:-3.11}"
NVME="${HNAV_NVME:-/mnt/nvmes/nvme1/egekutlu}"
export HF_HOME="${HF_HOME:-$NVME/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

echo "=============================================================="
echo " H-Nav setup"
echo "   repo    : $REPO_ROOT"
echo "   conda   : system-wide (never self-installed)"
echo "   env     : $ENV_NAME  (python $PYVER)"
echo "   HF_HOME : $HF_HOME"
echo "=============================================================="

# ── 0. GPU state, for information — setup itself allocates nothing ───────────
# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh
echo
echo "[0/5] current GPU state:"
hnav_gpu_report
echo

# ── 1. system conda ──────────────────────────────────────────────────────────
# `command -v conda` is NOT sufficient: /opt/anaconda3/condabin/conda is on PATH
# but is a standalone binary, while `conda activate` is a shell FUNCTION that
# only exists after conda's hook is sourced. Testing the binary and then calling
# activate yields "CondaError: Run 'conda init' before 'conda activate'".
# The right test is `type -t conda` = function.
if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
    for _c in activate_conda "$HOME/activate_conda" /opt/activate_conda; do
        # shellcheck disable=SC1090
        source "$_c" >/dev/null 2>&1 && break
    done
fi
if [ "$(type -t conda 2>/dev/null)" != "function" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
    for _h in /opt/anaconda3/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do
        # shellcheck disable=SC1090
        [ -f "$_h" ] && { source "$_h"; break; }
    done
fi
if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
    echo "!! 'conda activate' unavailable (shell function not defined)."
    echo "   Do NOT install your own conda. Activate by hand and re-run:"
    echo "       conda activate $ENV_NAME"
    echo "       HNAV_NO_ACTIVATE=1 bash hnav/deploy/setup_ozonderlab2.sh"
    exit 1
fi
echo "[1/5] conda: $(conda --version)  (activate function available)"

# ── 2. env ───────────────────────────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/5] env '$ENV_NAME' exists, reusing"
else
    echo "[2/5] creating env '$ENV_NAME' (python $PYVER) ..."
    conda create -y -n "$ENV_NAME" "python=$PYVER"
fi
if [ "${HNAV_NO_ACTIVATE:-0}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    # shellcheck disable=SC1091
    conda activate "$ENV_NAME"
fi
python -m pip install -q --upgrade pip wheel setuptools
echo "        python: $(python -V 2>&1)  at  $(command -v python)"

# ── 3. torch ─────────────────────────────────────────────────────────────────
# Plain PyPI: on Linux, `pip install torch` ships CUDA 12.x wheels by default,
# which run fine under the 580 / CUDA 13.0 driver (backward compatible) on the
# 4090 (Ada, sm_89). The dedicated cu124 index is only a FALLBACK — used as the
# sole index it replaces PyPI entirely and pip can fail with
# ResolutionImpossible while resolving torch's own dependencies there.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[3/5] torch present with CUDA: $(python -c 'import torch;print(torch.__version__)')"
else
    echo "[3/5] installing torch from PyPI (CUDA wheels are the Linux default) ..."
    if ! python -m pip install -q torch; then
        TORCH_INDEX="${HNAV_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
        echo "      PyPI failed; retrying with --extra-index-url $TORCH_INDEX ..."
        python -m pip install -q torch --extra-index-url "$TORCH_INDEX"
    fi
    python -c "import torch; assert torch.cuda.is_available(), 'torch installed but CUDA unavailable'" || {
        echo "!! torch imported but torch.cuda.is_available() is False."
        echo "   Check the driver (nvidia-smi) and that you are not in a CPU-only wheel."
        exit 1
    }
fi

# ── 4. deps + nltk/tiktoken data ─────────────────────────────────────────────
echo "[4/5] installing deps ..."
python -m pip install -q \
    numpy scipy scikit-learn \
    transformers accelerate safetensors sentencepiece \
    huggingface_hub \
    nltk tiktoken \
    openai python-dotenv httpx tqdm pytest

# nltk's downloader REFUSES world/group-writable target trees (its CVE
# hardening), and /home on this box is exactly that — so `nltk.download` was
# silently a no-op and the real chunker never ran (m2's tripwire caught it).
# Fetch the punkt zips by hand into $NLTK_DATA instead: only the DOWNLOADER
# has the permission check, the loader does not.
export NLTK_DATA="${NLTK_DATA:-$NVME/nltk_data}"
python - <<'PYEOF'
import os, pathlib, sys, urllib.request, zipfile

root = pathlib.Path(os.environ["NLTK_DATA"])
tok = root / "tokenizers"
tok.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(root, 0o700)   # best effort; the manual path works regardless
except OSError:
    pass

for pkg in ("punkt", "punkt_tab"):
    if (tok / pkg).exists():
        print(f"   nltk {pkg}: present")
        continue
    url = f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{pkg}.zip"
    z = tok / f"{pkg}.zip"
    print(f"   nltk {pkg}: fetching ...")
    urllib.request.urlretrieve(url, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(tok)
    z.unlink()

import nltk
try:
    n = len(nltk.sent_tokenize("One sentence. Two sentences."))
    assert n == 2
    print("   nltk punkt: VERIFIED (sent_tokenize works)")
except Exception as e:
    sys.exit(f"!! punkt still unusable after manual install: {e}")

import tiktoken
tiktoken.encoding_for_model("gpt-4o-mini")
print("   tiktoken cached")
PYEOF

# ── 4b. minimal benchmark deps, for M0 + T4 ──────────────────────────────────
# The full MemoryAgentBench requirements.txt drags in deepspeed, bitsandbytes,
# faiss-gpu (broken on pip) and a dozen memory backends. The m0/t4 import
# chains actually need only this subset — mapped by reading the imports of
# main.py → agent.py → embedding_retriever.py → utils/*. faiss-cpu is correct
# here: LangChain's FAISS.from_documents builds IndexFlatL2 on CPU either way,
# and flat search is exact.
echo "[4b/6] installing minimal benchmark deps (m0/t4) ..."
python -m pip install -q \
    pyyaml datasets rouge-score \
    langchain langchain-community langchain-openai langchain-core \
    faiss-cpu

# ── 5. weights + .env files ──────────────────────────────────────────────────
EMBED_MODEL="${HNAV_EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
NLI_MODEL="${HNAV_NLI_MODEL:-cross-encoder/nli-deberta-v3-large}"
echo "[5/5] fetching $EMBED_MODEL (~8 GB) + $NLI_MODEL (~1.7 GB), first run only ..."
python - "$EMBED_MODEL" "$NLI_MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for model in sys.argv[1:]:
    path = snapshot_download(model, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "spm.model"])
    print(f"   cached at: {path}")
PYEOF

if [ ! -f "$REPO_ROOT/.env" ]; then
    cp hnav/deploy/ozonderlab2.env "$REPO_ROOT/.env"
    echo "   wrote .env — CHECK HNAV_LLM_MODEL matches what :8000 serves"
else
    echo "   .env exists, left alone"
fi

MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
if [ ! -f "$MAB/.env" ]; then
    cp hnav/deploy/mab.env.template "$MAB/.env"
    echo "   wrote MemoryAgentBench/.env (embeddings -> :8001)"
else
    echo "   MemoryAgentBench/.env exists, left alone"
fi

cat <<EOF

==============================================================
 Setup complete.

 Every new shell needs the env — conda init is off by default:

     source activate_conda
     conda activate $ENV_NAME
     export HF_HOME=$HF_HOME

 (The H-Nav launcher scripts do this themselves via
  hnav/deploy/_activate.sh, so nohup/tmux runs are safe.)

 Next:
     python hnav/deploy/check_env.py     # must exit 0
     pytest hnav/tests/ -q               # must be 175 passed
==============================================================
EOF
