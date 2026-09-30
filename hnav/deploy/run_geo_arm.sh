#!/usr/bin/env bash
# E2E-3: the hnav_geo wet run (all three subsets) against the frozen
# Qwen3-4B-Instruct-2507 chat server on :8003. Launch under nohup from the
# repo root on the box:
#
#   nohup bash hnav/deploy/run_geo_arm.sh > hnav/_out/geo_arm_launch.log 2>&1 &
#
# Progress: hnav/_out/geo_arm_progress ; sentinel: hnav/_out/geo_arm.done
set -u
cd "$(dirname "$0")/../.."
source hnav/deploy/_activate.sh
ARM_NAME=geo_arm

# [E2E-4] single-instance lock. Two launchers fired 8 s apart on 2026-08-30 and
# both cleared the runner's one-shot guard, because that guard looks for
# detector_gap_*.json which is only written when a subset FINISHES. Two
# detector_gap processes then targeted the same output file. mkdir is atomic on
# POSIX, so this is a real mutex, not a check-then-act race.
LOCK="hnav/_out/${ARM_NAME}.lock"
mkdir -p hnav/_out
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSED: $LOCK exists - another ${ARM_NAME} run is in flight (or a"
  echo "previous one died; remove the directory by hand after checking)."
  exit 9
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
PROG=hnav/_out/geo_arm_progress
echo "start $(date -u +%FT%TZ)" >> "$PROG"

curl -sf http://localhost:8003/v1/models > /dev/null || {
  echo "FAILED no-chat-server $(date -u +%FT%TZ)" >> "$PROG"; exit 3; }

python pipelines/hnav_geo/run.py --llm-model Qwen/Qwen3-4B-Instruct-2507 --dry-run \
  && echo "ok dry-run $(date -u +%FT%TZ)" >> "$PROG" \
  || { echo "FAILED dry-run $(date -u +%FT%TZ)" >> "$PROG"; exit 4; }

python pipelines/hnav_geo/run.py --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --llm-base-url http://localhost:8003/v1 \
  && echo "ok wet $(date -u +%FT%TZ)" >> "$PROG" \
  || { echo "FAILED wet $(date -u +%FT%TZ)" >> "$PROG"; exit 5; }

touch hnav/_out/geo_arm.done
echo "done $(date -u +%FT%TZ)" >> "$PROG"
