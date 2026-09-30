#!/usr/bin/env bash
# Fill the ONE gap in the five-model comparison table.             [E2E-5]
#
# hnav_raw x Qwen3-4B-Instruct-2507 x {sh_6k, sh_32k} has never been measured
# at page_source=benchmark. The committed reference numbers for those two
# subsets (94/100 and 86/100) come from detector_gap_retrieval_sh{6,32}k.json,
# which ran with page_source=None - the prepass page, not the benchmark page.
# Their native rows say so plainly: 28 and 48, against 30 and 53 for every
# pipelines/ arm of the same model. sh_64k is unaffected; the confirmatory run
# already used page_source=benchmark, which is why its native (45) agrees.
#
# This is a NEW cell, not a re-roll: a configuration that was never run. Every
# other Qwen3-4B cell stays exactly as committed.
#
#   nohup bash hnav/deploy/run_reference_raw_fill.sh > hnav/_out/campaign/ref_raw_fill.log 2>&1 &
#
# The server config here is the FROZEN Stage-1 substrate, flag for flag
# (serve_stage1_chat.sh): kv fp8, max-model-len 65536, gpu-memory-utilization
# 0.58. Not the campaign defaults. The point of this run is comparability with
# the reference model's other cells, so its substrate must match theirs even
# where a campaign default would be harmless.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_KEY=qwen3_4b_ref
SERVED_NAME=Qwen/Qwen3-4B-Instruct-2507
MODEL_PATH=/mnt/nvmes/nvme1/egekutlu/models/Qwen3-4B-Instruct-2507
TAG=Qwen_Qwen3-4B-Instruct-2507_benchmarkpage_2026-08-31
PORT="${HNAV_STAGE1_CHAT_PORT:-8003}"
BASE_URL="http://localhost:${PORT}/v1"
OUTDIR="hnav/_out/campaign/${MODEL_KEY}"
mkdir -p "$OUTDIR"
PROG="$OUTDIR/progress"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PROG"; }

LOCK="$OUTDIR/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "REFUSED: $LOCK exists"; exit 9; fi
SRV=""
stop_server() {
    [ -n "$SRV" ] || return 0
    kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
    kill -KILL -"$SRV" 2>/dev/null; SRV=""
}
trap 'stop_server; rmdir "$LOCK" 2>/dev/null' EXIT
trap 'log "[signal] aborting"; stop_server; rmdir "$LOCK" 2>/dev/null; exit 130' INT TERM

source hnav/deploy/_activate.sh || { log "FAILED activate"; exit 2; }
log "=== reference hnav_raw fill :: sh_6k sh_32k :: tag $TAG ==="

if curl -sf --max-time 5 "$BASE_URL/models" >/dev/null 2>&1; then
    log "FAILED: something already serves :$PORT"; exit 3
fi

log "starting FROZEN Stage-1 substrate (serve_stage1_chat.sh, unmodified)"
setsid bash hnav/deploy/serve_stage1_chat.sh > "$OUTDIR/server.log" 2>&1 < /dev/null &
SRV=$!
ok=0
for _ in $(seq 1 120); do
    kill -0 "$SRV" 2>/dev/null || break
    curl -sf --max-time 5 "$BASE_URL/models" >/dev/null 2>&1 && { ok=1; break; }
    sleep 10
done
[ "$ok" = 1 ] || { log "FAILED: server never ready"; tail -20 "$OUTDIR/server.log" \
    | sed 's/^/    /' | tee -a "$PROG"; exit 4; }
log "server ready"

# Health probe only. The full preflight's one_shot check would refuse this
# model - idonly and geo ARE measured for it, and correctly so; only this one
# arm/subset pair is new. --probe-only skips one_shot and still runs the
# repetition detector and the unique-stratum accuracy floor.
log "probe"
if ! python hnav/deploy/preflight_model.py --probe-only \
        --model-key "$MODEL_KEY" --model-path "$MODEL_PATH" \
        --served-name "$SERVED_NAME" --base-url "$BASE_URL" \
        > "$OUTDIR/probe.log" 2>&1; then
    log "FAILED probe"; tail -20 "$OUTDIR/probe.log" | sed 's/^/    /' | tee -a "$PROG"
    exit 5
fi
grep -E '^\s+\[(ok |FAIL)\]' "$OUTDIR/probe.log" | tee -a "$PROG"

log "--- hnav_raw : sh_6k sh_32k (one shot, ~1,000 completions) ---"
if python pipelines/hnav_raw/run.py --llm-model "$SERVED_NAME" \
        --llm-base-url "$BASE_URL" --subsets sh_6k sh_32k --tag "$TAG" \
        > "$OUTDIR/wet.log" 2>&1; then
    log "ok -> pipelines/hnav_raw/results/$TAG"
    tail -5 "$OUTDIR/wet.log" | sed 's/^/    /' | tee -a "$PROG"
    touch "$OUTDIR/fill.done"
else
    log "FAILED wet run"; tail -25 "$OUTDIR/wet.log" | sed 's/^/    /' | tee -a "$PROG"
    exit 6
fi
stop_server
log "=== done ==="
