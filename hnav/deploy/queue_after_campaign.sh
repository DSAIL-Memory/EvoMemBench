#!/usr/bin/env bash
# Wait for the running multi-model campaign to finish, then start another one
# restricted to the given model keys.                              [E2E-5]
#
#   nohup bash hnav/deploy/queue_after_campaign.sh gemma3_4b \
#        > hnav/_out/campaign/queued_gemma3.log 2>&1 &
#
# Why this exists: a model can be skipped mid-campaign (gemma-3's re-run
# tripped the one_shot guard on its own preserved VOID directories) and the
# orchestrator correctly moves on rather than stopping. Re-queuing by hand
# means the GPU idles until someone notices - which already cost three hours
# once. This chains the next run instead.
#
# It waits for the campaign.finished sentinel, clears it, and launches. The
# orchestrator's own locks still apply, so this cannot start a second
# concurrent campaign.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

KEYS="${*:?usage: queue_after_campaign.sh <model_key> [model_key ...]}"
OUT=hnav/_out/campaign
SENTINEL="$OUT/campaign.finished"
DEADLINE=$(( $(date +%s) + 86400 ))     # 24 h; the campaign is ~11 h

echo "[$(date -u +%FT%TZ)] queued: $KEYS (waiting for $SENTINEL)"

while [ ! -f "$SENTINEL" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[$(date -u +%FT%TZ)] GAVE UP: no campaign.finished within 24 h"
        exit 1
    fi
    # If no campaign is running at all, there is nothing to wait for.
    if ! pgrep -f run_multimodel_campaign.sh > /dev/null 2>&1; then
        echo "[$(date -u +%FT%TZ)] no campaign running; proceeding now"
        break
    fi
    sleep 300
done

rm -f "$SENTINEL"
echo "[$(date -u +%FT%TZ)] starting campaign for: $KEYS"
# shellcheck disable=SC2086
exec bash hnav/deploy/run_multimodel_campaign.sh $KEYS
