#!/usr/bin/env bash
# Run M1 (the Stage-0 gate) under nohup so it survives the SSH session dropping.
#
#   bash hnav/deploy/run_t1.sh              # all single-hop subsets
#   bash hnav/deploy/run_t1.sh --subsets sh_6k --max-pairs 50   # 2-minute smoke
#
# Watch:  tail -f hnav/_out/m1.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# System-wide conda + nvme caches. conda init is off at login, so this must
# happen inside the script — a nohup'd job does not inherit an interactive shell.
# shellcheck source=/dev/null
source hnav/deploy/_activate.sh
# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh

mkdir -p hnav/_out
LOG="hnav/_out/m1.log"

# ── GPU preflight — there is no scheduler on these boxes ─────────────────────
DEV="${HNAV_EMBED_DEVICE:-$(grep -E '^HNAV_EMBED_DEVICE=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo 1)}"
DEV="${DEV:-1}"
DTYPE="${HNAV_EMBED_DTYPE:-$(grep -E '^HNAV_EMBED_DTYPE=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo float32)}"
NEED=17; [ "$DTYPE" = "float16" ] && NEED=9

hnav_gpu_report
echo
echo "Target: GPU$DEV, dtype=$DTYPE"
hnav_gpu_guard "$DEV" "$NEED" || {
    echo
    echo "Aborting before anything allocates. Nothing has been run."
    exit 1
}

echo
echo "Pre-flight ..."
python hnav/deploy/check_env.py || { echo "check_env failed — fix before running M1."; exit 1; }

echo
echo "Launching M1 (nohup). Log: $LOG"
nohup python hnav/stage0/m1_geometry_calibration.py "$@" > "$LOG" 2>&1 &
PID=$!
echo "  pid=$PID"
echo
echo "  tail -f $LOG"
echo "  results land in hnav/_out/m1_geometry_calibration.json"
echo
echo "Exit codes: 0 = gate PASSED, 2 = GATE S3 FIRED (stop, report to a human)."
