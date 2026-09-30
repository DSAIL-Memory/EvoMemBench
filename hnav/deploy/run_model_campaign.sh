#!/usr/bin/env bash
# The whole campaign for ONE answering model, unattended.        [E2E-5]
#
#   serve -> preflight (hard gate) -> hnav_raw -> hnav_idonly -> hnav_geo -> stop
#
# ~4,500 chat completions (3 arms x 5 internal arms x 100 questions x 3
# subsets), several hours. Everything runs under one detached process, so the
# ssh session that started it is irrelevant once it is running.
#
#   nohup bash hnav/deploy/run_model_campaign.sh hnav/deploy/models.d/01_phi4_mini.env \
#        > hnav/_out/campaign/phi4_mini/driver.log 2>&1 &
#
# Discipline this script enforces, none of it optional:
#   * ONE server instance per model, shared by all three arms - otherwise the
#     within-model cross-arm comparison is confounded by the serving config.
#   * the preflight must pass before a single completion is spent.
#   * one shot per (model, arm): the pipeline runner refuses to overwrite, and
#     the arm tag is computed ONCE here so a run crossing midnight does not
#     scatter the three arms across two tags and break cross-arm pairing.
#   * a failing arm does not abort the model; the remaining arms still run and
#     the failure is recorded. A void is reported, never re-rolled.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONF="${1:?usage: run_model_campaign.sh <models.d/*.env>}"
[ -f "$CONF" ] || { echo "no such model config: $CONF"; exit 2; }
# shellcheck disable=SC1090
source "$CONF"
: "${MODEL_KEY:?}" "${MODEL_PATH:?}" "${SERVED_NAME:?}"

ARMS="${HNAV_CAMPAIGN_ARMS:-hnav_raw hnav_idonly hnav_geo}"
PORT="${HNAV_STAGE1_CHAT_PORT:-8003}"
BASE_URL="http://localhost:${PORT}/v1"
OUTDIR="hnav/_out/campaign/${MODEL_KEY}"
PROG="$OUTDIR/progress"
mkdir -p "$OUTDIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PROG"; }

# ── single instance (mkdir is atomic; a check-then-act test is not) ──────────
LOCK="$OUTDIR/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "REFUSED: $LOCK exists - a ${MODEL_KEY} campaign is already in flight"
    echo "(or one died; remove the directory by hand after checking)."
    exit 9
fi
SERVER_PID=""
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server pid $SERVER_PID"
        # the launcher execs vllm, but vllm forks an engine worker; kill the
        # whole process group or the worker keeps the card and the next model's
        # GPU guard refuses to start.
        kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 2
        done
        kill -KILL -"$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
cleanup() { stop_server; rmdir "$LOCK" 2>/dev/null; }
# INT/TERM must EXIT, not just clean up: bash defers the signal until the
# running arm returns and then continues the loop, so a non-exiting handler
# makes `kill` a no-op and the next arm starts anyway.
trap cleanup EXIT
trap 'log "[signal] aborting ${MODEL_KEY}"; cleanup; exit 130' INT TERM

source hnav/deploy/_activate.sh || { log "FAILED conda activate"; exit 2; }

MODEL_TAG="$(printf '%s' "$SERVED_NAME" | sed 's/[^A-Za-z0-9._-]\+/_/g; s/^_//; s/_$//')"
RUN_TAG="${HNAV_CAMPAIGN_TAG:-${MODEL_TAG}_$(date -u +%F)}"
log "=== campaign ${MODEL_KEY} :: tag ${RUN_TAG} :: arms ${ARMS} ==="
log "git $(git rev-parse --short HEAD)"

# ── the port and the card must be ours before we start ──────────────────────
if curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    log "FAILED: something is already serving :$PORT. Stop it first - this"
    log "        campaign will not answer against a server it did not start."
    exit 3
fi

log "starting server"
# setsid: its own process group, so stop_server can signal vllm AND the engine
# worker it forks. Without it the worker survives, holds GPU1, and the next
# model's gpu_guard refuses to launch.
setsid bash hnav/deploy/serve_campaign_model.sh "$CONF" \
    > "$OUTDIR/server.log" 2>&1 < /dev/null &
SERVER_PID=$!
log "server pid $SERVER_PID"

# The 9B loads ~19 GB from nvme; 30 min is slack, not an expectation.
READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "FAILED: server process died during startup - see $OUTDIR/server.log"
        tail -30 "$OUTDIR/server.log" | sed 's/^/    /' | tee -a "$PROG"
        exit 4
    fi
    if curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        READY=1; log "server ready after ~$((i * 10))s"; break
    fi
    sleep 10
done
[ "$READY" = 1 ] || { log "FAILED: server never became ready"; \
    tail -30 "$OUTDIR/server.log" | sed 's/^/    /' | tee -a "$PROG"; exit 4; }

# ── the gate ────────────────────────────────────────────────────────────────
log "preflight"
if ! python hnav/deploy/preflight_model.py \
        --model-key "$MODEL_KEY" --model-path "$MODEL_PATH" \
        --served-name "$SERVED_NAME" --base-url "$BASE_URL" \
        > "$OUTDIR/preflight.log" 2>&1; then
    log "FAILED preflight - not spending 4,500 completions on this model"
    tail -40 "$OUTDIR/preflight.log" | sed 's/^/    /' | tee -a "$PROG"
    exit 5
fi
grep -E '^\s+\[(ok |FAIL)\]' "$OUTDIR/preflight.log" | tee -a "$PROG"
log "preflight PASSED"

# ── the arms ────────────────────────────────────────────────────────────────
FAILED_ARMS=""
for ARM in $ARMS; do
    RUN="pipelines/${ARM}/run.py"
    [ -f "$RUN" ] || { log "SKIP $ARM (no $RUN)"; FAILED_ARMS="$FAILED_ARMS $ARM"; continue; }

    log "--- $ARM : dry-run (guards + call budget, sends nothing) ---"
    DRY_TAG="_dryrun_${MODEL_KEY}_${ARM}"
    if python "$RUN" --llm-model "$SERVED_NAME" --llm-base-url "$BASE_URL" \
            --tag "$DRY_TAG" --dry-run > "$OUTDIR/${ARM}_dryrun.log" 2>&1; then
        log "$ARM dry-run ok"
    else
        log "FAILED $ARM dry-run - skipping this arm, continuing with the rest"
        tail -25 "$OUTDIR/${ARM}_dryrun.log" | sed 's/^/    /' | tee -a "$PROG"
        FAILED_ARMS="$FAILED_ARMS $ARM"
        rm -rf "pipelines/${ARM}/results/${DRY_TAG}"
        continue
    fi
    rm -rf "pipelines/${ARM}/results/${DRY_TAG}"

    log "--- $ARM : WET (one shot, ~1,500 completions) ---"
    if python "$RUN" --llm-model "$SERVED_NAME" --llm-base-url "$BASE_URL" \
            --tag "$RUN_TAG" > "$OUTDIR/${ARM}_wet.log" 2>&1; then
        log "$ARM wet ok -> pipelines/${ARM}/results/${RUN_TAG}"
        tail -6 "$OUTDIR/${ARM}_wet.log" | sed 's/^/    /' | tee -a "$PROG"
    else
        log "FAILED $ARM wet run (rc=$?) - continuing with the remaining arms"
        tail -25 "$OUTDIR/${ARM}_wet.log" | sed 's/^/    /' | tee -a "$PROG"
        FAILED_ARMS="$FAILED_ARMS $ARM"
    fi
done

log "stopping server for $MODEL_KEY"
stop_server
# Let the card come back before the next model's guard runs. `grep -c .` on
# empty input prints "0" AND exits 1, so `|| echo 0` made this "0\n0" and the
# comparison never matched - the loop always burned its full five minutes.
# `wc -l` always exits 0.
for _ in $(seq 1 60); do
    used=$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader \
           2>/dev/null | sed '/^$/d' | wc -l)
    [ "$used" -eq 0 ] && break
    sleep 5
done

if [ -n "$FAILED_ARMS" ]; then
    log "=== ${MODEL_KEY} DONE WITH FAILURES:${FAILED_ARMS} ==="
    touch "$OUTDIR/campaign.partial"
    exit 6
fi
log "=== ${MODEL_KEY} DONE - all arms ==="
touch "$OUTDIR/campaign.done"
exit 0
