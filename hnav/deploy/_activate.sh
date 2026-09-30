#!/usr/bin/env bash
# Shared activation for every H-Nav script that runs on the lab GPU boxes.
#   source hnav/deploy/_activate.sh
#
# Lab policy: do NOT install your own conda. A system-wide install lives at
# /opt/anaconda3. `conda init` is deliberately not run at login (it slows bash
# down), so every script must make conda usable itself — inheriting it from an
# interactive shell is not reliable under nohup/tmux.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY `command -v conda` IS THE WRONG TEST
# ─────────────────────────────────────────────────────────────────────────────
# /opt/anaconda3/condabin/conda is on PATH by default, so `command -v conda`
# succeeds — but that is a standalone BINARY. `conda activate` is a shell
# FUNCTION that only exists after `conda init` or after sourcing conda's hook.
# Testing the binary and then calling activate gives:
#
#     CondaError: Run 'conda init' before 'conda activate'
#
# The correct test is `type -t conda` = "function". `conda shell.bash hook` is
# the canonical way to define it from the binary without touching ~/.bashrc.
#
# Override the env name with HNAV_CONDA_ENV. Set HNAV_NO_ACTIVATE=1 to skip
# entirely (useful if you have already activated something by hand).

HNAV_CONDA_ENV="${HNAV_CONDA_ENV:-hnav}"

if [ "${HNAV_NO_ACTIVATE:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

# ── already in the target env? then there is nothing to do ───────────────────
if [ "${CONDA_DEFAULT_ENV:-}" = "$HNAV_CONDA_ENV" ]; then
    _hnav_already=1
else
    _hnav_already=0
fi

if [ "$_hnav_already" -eq 0 ]; then
    # ── make `conda activate` available (the FUNCTION, not the binary) ───────
    if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
        # 1. the lab's own helper. `source` searches PATH for a bare name.
        for _c in activate_conda "$HOME/activate_conda" /opt/activate_conda; do
            # shellcheck disable=SC1090
            source "$_c" >/dev/null 2>&1 && break
        done
    fi
    if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
        # 2. conda's own hook, driven by whatever conda binary is on PATH.
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook 2>/dev/null)" || true
        fi
    fi
    if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
        # 3. source the profile script from the known system-wide installs.
        for _h in /opt/anaconda3/etc/profile.d/conda.sh \
                  /opt/miniconda3/etc/profile.d/conda.sh; do
            # shellcheck disable=SC1090
            [ -f "$_h" ] && { source "$_h"; break; }
        done
    fi

    if [ "$(type -t conda 2>/dev/null)" != "function" ]; then
        echo "!! 'conda activate' is unavailable (the shell function is not defined)."
        echo "   Tried: source activate_conda / conda shell.bash hook /"
        echo "          /opt/anaconda3/etc/profile.d/conda.sh"
        echo "   Do NOT install your own conda. Either ask the admin where the"
        echo "   helper lives, or activate by hand and re-run with:"
        echo "       conda activate $HNAV_CONDA_ENV"
        echo "       HNAV_NO_ACTIVATE=1 bash <script>"
        return 1 2>/dev/null || exit 1
    fi

    if ! conda activate "$HNAV_CONDA_ENV" 2>/dev/null; then
        echo "!! conda env '$HNAV_CONDA_ENV' not found."
        echo "   Create it once with:  bash hnav/deploy/setup_ozonderlab2.sh"
        return 1 2>/dev/null || exit 1
    fi
fi

# ── caches on the nvme, not the home dir (weights are ~8 GB) ─────────────────
NVME="${HNAV_NVME:-/mnt/nvmes/nvme1/egekutlu}"
export HF_HOME="${HF_HOME:-$NVME/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# nltk refuses to DOWNLOAD into a world/group-writable tree (CVE hardening in
# its downloader) — /home on this box is exactly that, so punkt silently never
# arrived and the chunker fell back (caught by m2's tripwire). Setup places the
# data here by hand, bypassing the downloader; the LOADER has no such check.
export NLTK_DATA="${NLTK_DATA:-$NVME/nltk_data}"
