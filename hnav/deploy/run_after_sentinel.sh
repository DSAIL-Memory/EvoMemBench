#!/usr/bin/env bash
# Wait for a sentinel file, then exec a command.                    [E2E-5]
#
#   nohup bash hnav/deploy/run_after_sentinel.sh \
#       hnav/_out/campaign/campaign.finished \
#       bash hnav/deploy/run_reference_raw_fill.sh \
#       > hnav/_out/campaign/chained.log 2>&1 &
#
# queue_after_campaign.sh chains ANOTHER ORCHESTRATOR RUN and takes models.d
# keys; this chains an arbitrary command. Passing a non-key to that script
# starts an orchestrator that matches no model and does nothing - which is
# exactly what happened once, and is why the two jobs are now separate scripts
# with names that say which is which.
#
# Also waits for GPU1 to be free, because a sentinel can appear a moment before
# the server holding the card has finished exiting.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SENTINEL="${1:?usage: run_after_sentinel.sh <sentinel-path> <command...>}"
shift
[ "$#" -gt 0 ] || { echo "no command given"; exit 2; }

DEADLINE=$(( $(date +%s) + 86400 ))
echo "[$(date -u +%FT%TZ)] waiting for $SENTINEL, then: $*"

while [ ! -f "$SENTINEL" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[$(date -u +%FT%TZ)] GAVE UP: $SENTINEL never appeared (24 h)"
        exit 1
    fi
    sleep 300
done
echo "[$(date -u +%FT%TZ)] sentinel present"

# `grep -c .` on empty input prints 0 and exits 1; `wc -l` always exits 0.
for _ in $(seq 1 120); do
    n=$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader \
        2>/dev/null | sed '/^$/d' | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 10
done
echo "[$(date -u +%FT%TZ)] GPU1 free; starting"
exec "$@"
