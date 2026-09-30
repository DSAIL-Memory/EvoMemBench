#!/usr/bin/env bash
# The whole multi-model campaign, one model at a time, unattended.  [E2E-5]
#
#   nohup bash hnav/deploy/run_multimodel_campaign.sh \
#        > hnav/_out/campaign/campaign.log 2>&1 &
#
# Runs every hnav/deploy/models.d/*.env in filename order - smallest weights
# first, so a failure on the largest model costs nothing that already ran - and
# for each one: serve on GPU1, preflight, hnav_raw, hnav_idonly, hnav_geo, stop.
#
# Total ~18,000 chat completions across four models. The reference model
# (Qwen3-4B-Instruct-2507) is NOT re-run: its numbers are already committed and
# a second shot at a measured cell is a re-roll, not a replication.
#
# A model that fails does not stop the campaign. Resume is free: a model whose
# campaign.done sentinel exists is skipped, and the pipeline runners refuse to
# overwrite a results folder in any case.
#
# Args: optional list of model keys to restrict to (e.g. `qwen35_9b`).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT=hnav/_out/campaign
mkdir -p "$OUT"
PROG="$OUT/campaign_progress"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PROG"; }

LOCK="$OUT/.campaign.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "REFUSED: $LOCK exists - a multi-model campaign is already running."
    exit 9
fi
# The handler MUST exit. bash defers a signal until the foreground child (a
# multi-hour model campaign) returns, then runs the handler and CARRIES ON with
# the loop - so a trap that only cleans up turns `kill` into a no-op and the
# next model launches anyway. Observed on 2026-08-30: SIGTERM to this script
# left it queued to start gemma4 with the serving config that had just been
# found defective. Only SIGKILL stopped it.
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
trap 'echo "[signal] stopping after the current model"; rmdir "$LOCK" 2>/dev/null; exit 130' INT TERM

ONLY="$*"
CONFS=$(ls hnav/deploy/models.d/*.env 2>/dev/null | sort)
[ -n "$CONFS" ] || { log "no models.d/*.env found"; exit 2; }

log "================= MULTI-MODEL CAMPAIGN ================="
log "git $(git rev-parse --short HEAD)"
[ -n "$ONLY" ] && log "restricted to: $ONLY"

DONE=""; FAILED=""; SKIPPED=""
for CONF in $CONFS; do
    KEY=$(grep -E '^MODEL_KEY=' "$CONF" | head -1 | cut -d= -f2)
    if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qx "$KEY"; then
        SKIPPED="$SKIPPED $KEY"; continue
    fi
    if [ -f "$OUT/$KEY/campaign.done" ]; then
        log "SKIP $KEY - campaign.done exists (one shot per model)"
        SKIPPED="$SKIPPED $KEY"; continue
    fi
    mkdir -p "$OUT/$KEY"
    log ">>> $KEY  ($CONF)"
    START=$(date +%s)
    bash hnav/deploy/run_model_campaign.sh "$CONF" \
        > "$OUT/$KEY/driver.log" 2>&1
    RC=$?
    MINS=$(( ($(date +%s) - START) / 60 ))
    if [ "$RC" = 0 ]; then
        log "<<< $KEY OK  (${MINS} min)"
        DONE="$DONE $KEY"
    else
        log "<<< $KEY rc=$RC  (${MINS} min) - see $OUT/$KEY/driver.log"
        tail -15 "$OUT/$KEY/progress" 2>/dev/null | sed 's/^/    /' | tee -a "$PROG"
        FAILED="$FAILED $KEY"
    fi
done

log "================= CAMPAIGN FINISHED ================="
log "  done   :${DONE:- (none)}"
log "  failed :${FAILED:- (none)}"
log "  skipped:${SKIPPED:- (none)}"
touch "$OUT/campaign.finished"
[ -z "$FAILED" ]
