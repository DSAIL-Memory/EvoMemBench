#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="${HF_HOME:-$REPO_ROOT/llms}/RL-MemoryAgent-14B"
BASE="https://hf-mirror.com/BytedTsinghua-SIA/RL-MemoryAgent-14B/resolve/main"
LOG="$SCRIPT_DIR/download_model.log"

mkdir -p "$DEST"
exec > "$LOG" 2>&1

echo "[$(date)] Starting model download ..."

# Download all 7 shards in parallel using background jobs
for i in 01 02 03 04 05 06 07; do
    f="model-000${i}-of-00007.safetensors"
    out="$DEST/$f"
    if [ -f "$out" ]; then
        actual_size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        echo "[$(date)] Shard $i: existing file $actual_size bytes, will resume"
    fi
    echo "[$(date)] Starting shard $i download ..."
    wget --no-proxy --continue --quiet --show-progress \
        --tries=20 --timeout=120 --waitretry=30 \
        -O "$out" "$BASE/$f" &
    echo "[$(date)] Shard $i wget PID: $!"
done

echo "[$(date)] Waiting for all downloads ..."
wait
echo "[$(date)] All downloads complete. Final sizes:"
ls -lh "$DEST"/*.safetensors 2>/dev/null
