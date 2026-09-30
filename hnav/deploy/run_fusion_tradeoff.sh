#!/usr/bin/env bash
# [EXPLORATORY] Relaxed-harm fusion trade-off: four one-shot sh_64k runs at
# tau 0/2/4/6 (predefined on calibration; user directive 2026-08-27). NOT a
# preregistered result — the fusion screen FAILS the zero-harm rule, see
# pipelines/hnav_fusion/pipeline.json "status".
#
#   nohup bash hnav/deploy/run_fusion_tradeoff.sh > hnav/_out/tradeoff_launch.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source hnav/deploy/_activate.sh
: > hnav/_out/tradeoff_progress
rm -f hnav/_out/tradeoff.done
for T in 0 2 4 6; do
  echo "start tau${T} $(date -u +%FT%TZ)" >> hnav/_out/tradeoff_progress
  HNAV_LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
  HNAV_LLM_BASE_URL="http://localhost:8003/v1" \
  python hnav/stage1/detector_gap.py --confirmatory --subsets sh_64k \
      --harness retrieval --page-source benchmark --pair-screen fusion \
      --fusion-artifact stage0_results/geometry_filter/fusion_screen.json \
      --operating-point "stage0_results/geometry_filter/fusion_exploratory_tau${T}_op.json" \
      --out "hnav/_out/fusion_tradeoff_tau${T}_sh64k.json" \
      > "hnav/_out/fusion_tradeoff_tau${T}.log" 2>&1 \
    && echo "ok tau${T} $(date -u +%FT%TZ)" >> hnav/_out/tradeoff_progress \
    || echo "FAILED tau${T} rc=$? $(date -u +%FT%TZ)" >> hnav/_out/tradeoff_progress
done
touch hnav/_out/tradeoff.done
