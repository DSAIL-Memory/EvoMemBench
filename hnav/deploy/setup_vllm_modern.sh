#!/usr/bin/env bash
# Build the SECOND vLLM environment, for the models the frozen one cannot load.
#                                                            [multi-model E2E-5]
#
# Why two environments at all: the campaign's answering models split cleanly by
# architecture support, and this was measured, not assumed —
#
#   vllm_0.9.1 (the existing, frozen serving env)
#       Phi3ForCausalLM                        supported
#       Gemma3ForConditionalGeneration         supported
#       Gemma4ForConditionalGeneration         NOT registered
#       Qwen3_5ForConditionalGeneration        NOT registered
#
# The existing env is NEVER upgraded: it is the environment the reference
# Qwen3-4B result was produced in, and re-running that model later under a
# different vLLM would silently change the reference. A new env in a new prefix
# leaves it untouched.
#
# The install is CPU/disk only — no GPU is touched — so this may run while a
# campaign is answering on GPU1.
#
#   nohup bash hnav/deploy/setup_vllm_modern.sh > hnav/_out/campaign/vllm_modern_setup.log 2>&1 &
#
# Result: $VENV/bin/vllm , recorded in hnav/_out/campaign/vllm_modern.json
set -uo pipefail

NVME="${HNAV_NVME:-/mnt/nvmes/nvme1/egekutlu}"
PROGRAMS="$NVME/programs"
VENV="${HNAV_VLLM_MODERN_VENV:-$PROGRAMS/venvs/vllm_modern}"
UV_DIR="$PROGRAMS/uv/bin"
WANT_VERSION="${HNAV_VLLM_MODERN_VERSION:-0.28.0}"
PY="${HNAV_VLLM_MODERN_PYTHON:-3.12}"

# Everything on the nvme: the wheel set (torch + vllm + deps) is several GB and
# /home on this box is small and group-writable.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NVME/.cache/uv}"
export TMPDIR="${TMPDIR:-$NVME/.cache/tmp}"
export HF_HOME="${HF_HOME:-$NVME/hf_cache}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$PROGRAMS/venvs"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p hnav/_out/campaign
OUT=hnav/_out/campaign/vllm_modern.json

echo "== vLLM modern env =="
echo "   venv    : $VENV"
echo "   version : $WANT_VERSION   python $PY"
echo "   uv cache: $UV_CACHE_DIR"

# ── uv, installed into our own prefix (no sudo, no ~/.local surprise) ─────────
if [ ! -x "$UV_DIR/uv" ]; then
    echo "-- installing uv into $UV_DIR"
    mkdir -p "$UV_DIR"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$UV_DIR" INSTALLER_NO_MODIFY_PATH=1 sh \
        || { echo "FAILED: uv install"; exit 2; }
fi
export PATH="$UV_DIR:$PATH"
uv --version || exit 2

# ── the venv ─────────────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
    echo "-- creating venv (uv fetches its own CPython $PY; the system has 3.10)"
    uv venv --python "$PY" "$VENV" || { echo "FAILED: uv venv"; exit 3; }
fi

echo "-- installing vllm==$WANT_VERSION (prebuilt wheels only; no local compile)"
uv pip install --python "$VENV/bin/python" "vllm==$WANT_VERSION" \
    || { echo "FAILED: vllm install"; exit 4; }

# ── the only question that matters: are the two archs registered now? ────────
"$VENV/bin/python" - "$OUT" <<'PY'
import json, subprocess, sys, pathlib
out = pathlib.Path(sys.argv[1])
import vllm, torch
from vllm.model_executor.models.registry import ModelRegistry as R
archs = set(R.get_supported_archs())
need = {"Gemma4ForConditionalGeneration": None,
        "Qwen3_5ForConditionalGeneration": None,
        "Phi3ForCausalLM": None,
        "Gemma3ForConditionalGeneration": None}
for a in need:
    need[a] = a in archs
rec = {"vllm": vllm.__version__, "torch": torch.__version__,
       "cuda": torch.version.cuda, "python": sys.version.split()[0],
       "archs": need, "venv": sys.prefix}
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps(rec, indent=1))
missing = [a for a, ok in need.items() if not ok]
if missing:
    print("!! STILL MISSING:", missing)
    print("   The campaign driver will refuse these models rather than serve")
    print("   them wrong. Try a nightly:")
    print("     uv pip install --python $VENV/bin/python --prerelease=allow \\")
    print("       --extra-index-url https://wheels.vllm.ai/nightly vllm")
    sys.exit(5)
print("OK: every architecture this campaign needs is registered.")
PY
rc=$?
echo "setup exit=$rc"
exit $rc
