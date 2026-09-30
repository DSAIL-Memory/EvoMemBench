#!/usr/bin/env bash
# GPU preflight. There is no Slurm on these boxes, so nothing stops two jobs
# landing on the same card and OOM-ing each other. Every H-Nav launcher sources
# this and refuses to start if the target GPU is already busy.
#
#   source hnav/deploy/gpu_guard.sh
#   hnav_gpu_guard 1 17          # require GPU1 to have >= 17 GB free
#
# Override the refusal with HNAV_FORCE_GPU=1 only if you know who the other
# process belongs to and have agreed to share.

hnav_gpu_report() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "── nvidia-smi ────────────────────────────────────────────────"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader 2>/dev/null \
            || nvidia-smi
        echo "── compute processes ─────────────────────────────────────────"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null | head -10 | sed 's/^/   /' || true
        echo "──────────────────────────────────────────────────────────────"
        echo "   Watch live in another terminal:  watch -n 1 nvidia-smi"
    else
        echo "!! nvidia-smi not found — cannot verify GPU state."
    fi
}

# hnav_gpu_guard <device_index> <required_free_gb>
hnav_gpu_guard() {
    local dev="${1:-1}" need_gb="${2:-17}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "!! nvidia-smi missing; skipping the GPU guard. You are on your own."
        return 0
    fi

    local free_mib
    free_mib="$(nvidia-smi --id="$dev" --query-gpu=memory.free \
                --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [ -z "$free_mib" ]; then
        echo "!! Could not read GPU$dev. Does it exist? Run nvidia-smi."
        return 1
    fi

    local free_gb=$(( free_mib / 1024 ))
    local nproc
    nproc="$(nvidia-smi --id="$dev" --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null | grep -c . || true)"

    echo "   GPU$dev: ${free_gb} GB free, ${nproc} compute process(es), need ~${need_gb} GB"

    if [ "$nproc" -gt 0 ]; then
        echo
        echo "   !! GPU$dev already has ${nproc} compute process(es) on it:"
        nvidia-smi --id="$dev" --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null | head -6 | sed 's/^/      /'
        [ "$nproc" -gt 6 ] && echo "      ... and $(( nproc - 6 )) more"
        echo
        if [ "${HNAV_FORCE_GPU:-0}" != "1" ]; then
            echo "   Refusing to launch. There is no scheduler here — starting anyway"
            echo "   risks OOM-ing both jobs. Options:"
            echo "     - pick a free card:   HNAV_EMBED_DEVICE=<n> bash <script>"
            echo "     - share deliberately: HNAV_FORCE_GPU=1 bash <script>"
            return 1
        fi
        echo "   HNAV_FORCE_GPU=1 set — continuing anyway."
    fi

    if [ "$free_gb" -lt "$need_gb" ]; then
        echo
        echo "   !! Only ${free_gb} GB free on GPU$dev, need ~${need_gb} GB."
        echo "      float32 needs ~17 GB, float16 ~9 GB. If you switch to float16,"
        echo "      set HNAV_EMBED_DTYPE=float16 and PIN IT for every later task —"
        echo "      dtype drift changes cosines and moves every threshold."
        if [ "${HNAV_FORCE_GPU:-0}" != "1" ]; then
            return 1
        fi
    fi
    return 0
}
