#!/usr/bin/env bash
# Two-phase cross-episode mem0 evaluator — ALL 12 cross-env pairs (FC mode).
#
# Identical to run_eval_mem0_inenv_crossenv_fc.sh but Phase 2 covers every
# ordered (a, b) pair where a ≠ b (12 pairs total) instead of only the 4
# same-cluster pairs.
#
# Phase 1 (in-env): 4 subsets run in parallel; each subset runs SERIALLY
#   (workers=1) so memory accumulates causally; each subset has its OWN store.
# Phase 2 (all cross-env): 12 a→b pairs run in parallel; each loads Phase-1
#   subset-a memory as READ-ONLY and generates subset-b samples.
#
# IMPORTANT — qdrant concurrency constraint:
#   qdrant local mode holds an exclusive flock on <storage>/qdrant/.lock, so two
#   processes CANNOT share one source bank dir simultaneously.  This script
#   pre-stages a private cp -r per pair under <base_dir>/_banks/ before
#   launching the parallel Phase-2 workers.  3 pairs share each Phase-1 source
#   bank, so 3 copies are made per subset.
#
# Usage:
#   bash scripts/run_eval_mem0_inenv_allcrossenv_fc.sh \
#     [--limit-per-subset N]   # default: all 50 per subset (use 2-3 for smoke test)
#     [--p2-workers N]         # default: 1  (per pair, generate phase)
#     [--top-k K]              # default: 3
#     [--run-name NAME]        # default: inenv_allcrossenv_mem0_fc
#     [--keep-going]           # continue Phase 2 even if some Phase-1 subset failed
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="CROSSEP-TOOL"
LIMIT_PER_SUBSET=""
P2_WORKERS=1
TOP_K=3
RUN_NAME="inenv_allcrossenv_mem0_fc"
KEEP_GOING=false

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit-per-subset) LIMIT_PER_SUBSET="$2"; shift 2 ;;
        --p2-workers)       P2_WORKERS="$2";        shift 2 ;;
        --top-k)            TOP_K="$2";             shift 2 ;;
        --run-name)         RUN_NAME="$2";          shift 2 ;;
        --keep-going)       KEEP_GOING=true;        shift   ;;
        *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Activate conda env ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Move to repo root (one level up from this script) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ── Verify required env vars ─────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[ERROR] .env not found in $SCRIPT_DIR — please create it from bfcl_eval/.env.example"
    exit 1
fi
set -a
# shellcheck source=/dev/null
source .env
set +a

for var in DEEPSEEK_API_KEY BATCH_MODEL DASHSCOPE_API_KEY DASHSCOPE_BASE_URL; do
    val="${!var:-}"
    if [[ -z "$val" || "$val" == *"XXXXXX"* ]]; then
        echo "[ERROR] $var is not set (or still placeholder) in .env"
        exit 1
    fi
done

# Disable mem0's shared internal telemetry qdrant (~/.mem0/migrations_qdrant).
export MEM0_TELEMETRY=false

# ── Helpers ───────────────────────────────────────────────────────────────────
IDS_DIR="bfcl_eval/scripts/cross_episode/ids"

extract_ids() {
    local subset="$1"
    python3 -c \
        "import json,sys; d=json.load(open(sys.argv[1]))['multi_turn_ours']; \
         n=int(sys.argv[2]) if sys.argv[2] else len(d); print(','.join(d[:n]))" \
        "${IDS_DIR}/ids_${subset}.json" "${LIMIT_PER_SUBSET:-}"
}

_in_list() {
    local needle="$1"; shift
    for el; do [[ "$el" == "$needle" ]] && return 0; done
    return 1
}

# ── Paths and timestamps ──────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
MEMORY_TYPE="mem0"
BASE_DIR="cross_episode_results/runs/${RUN_NAME}_${MEMORY_TYPE}_${TS}"
mkdir -p "${BASE_DIR}/logs"

# ── Subset + setting config ───────────────────────────────────────────────────
SUBSETS=(gorilla_fs vehicle_control trading_bot travel_api)

# All 12 ordered (a, b) pairs where a ≠ b
P2_A_SUBSETS=(
    gorilla_fs      gorilla_fs      gorilla_fs
    vehicle_control vehicle_control vehicle_control
    trading_bot     trading_bot     trading_bot
    travel_api      travel_api      travel_api
)
P2_B_SUBSETS=(
    vehicle_control trading_bot     travel_api
    gorilla_fs      trading_bot     travel_api
    gorilla_fs      vehicle_control travel_api
    gorilla_fs      vehicle_control trading_bot
)

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  BFCL Cross-Episode  [mem0 | in-env + all 12 cross-env | FC mode]"
echo "  Model:            ${BATCH_MODEL}"
echo "  Base dir:         ${BASE_DIR}"
echo "  Phase-1 workers:  1 (fixed, serial per subset)"
echo "  Phase-2 workers:  ${P2_WORKERS} (per pair, generate phase)"
echo "  Phase-2 pairs:    ${#P2_A_SUBSETS[@]}"
echo "  top_k:            ${TOP_K}"
[[ -n "$LIMIT_PER_SUBSET" ]] && echo "  Limit/subset:     ${LIMIT_PER_SUBSET}"
echo "================================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — in-env, generate then evaluate, serial within each subset
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 1] Launching ${#SUBSETS[@]} subset jobs in parallel..."

p1_pids=()
p1_names=()

for subset in "${SUBSETS[@]}"; do
    ids="$(extract_ids "$subset")"
    log="${BASE_DIR}/logs/phase1_${subset}.log"
    (
        python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
            --num-workers  1 \
            --output-dir   "${BASE_DIR}/phase1" \
            --run-name     "${subset}" \
            --memory-type  mem0 \
            --memory-top-k "${TOP_K}" \
            --mode         FC \
            --ids          "$ids" \
        && python -m bfcl_eval.scripts.cross_episode.run_batch_evaluate \
            --input        "${BASE_DIR}/phase1/${subset}/result.jsonl" \
            --num-workers  4
    ) >"$log" 2>&1 &
    p1_pids+=("$!")
    p1_names+=("$subset")
    echo "  started ${subset} (pid $!), log: $log"
done

p1_failed=()
for i in "${!p1_pids[@]}"; do
    if ! wait "${p1_pids[$i]}"; then
        p1_failed+=("${p1_names[$i]}")
        echo "[Phase 1] FAILED: ${p1_names[$i]} — see ${BASE_DIR}/logs/phase1_${p1_names[$i]}.log"
    else
        echo "[Phase 1] done:   ${p1_names[$i]}"
    fi
done

if [[ ${#p1_failed[@]} -gt 0 ]]; then
    echo "[Phase 1] ${#p1_failed[@]} subset(s) failed: ${p1_failed[*]}"
    if [[ "$KEEP_GOING" != "true" ]]; then
        echo "[Phase 1] Aborting before Phase 2. Pass --keep-going to run Phase 2 anyway."
        exit 1
    fi
    echo "[Phase 1] --keep-going set; proceeding to Phase 2 (affected pairs will be skipped)."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Stage per-pair private bank copies (required by qdrant's exclusive flock)
# Each Phase-1 source bank is shared by 3 Phase-2 pairs — copy it 3 times.
# ══════════════════════════════════════════════════════════════════════════════
scratch_root="${BASE_DIR}/_banks"
mkdir -p "${scratch_root}"
echo ""
echo "[Stage] Copying Phase-1 banks to per-pair scratch dirs..."

for i in "${!P2_A_SUBSETS[@]}"; do
    a="${P2_A_SUBSETS[$i]}"
    setting="${a}__to__${P2_B_SUBSETS[$i]}"

    if _in_list "$a" "${p1_failed[@]+"${p1_failed[@]}"}"; then
        continue
    fi

    src="${BASE_DIR}/phase1/${a}/memory"
    dst="${scratch_root}/${setting}"
    cp -r "${src}" "${dst}"
    # Remove the stale flock file; qdrant recreates it at init.
    rm -f "${dst}/qdrant/.lock"
    echo "  staged ${setting} → ${dst}"
done

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — all cross-env pairs, read-only bank, parallel across pairs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 2] Launching ${#P2_A_SUBSETS[@]} cross-env pairs in parallel..."

p2_pids=()
p2_names=()
p2_skipped=()

for i in "${!P2_A_SUBSETS[@]}"; do
    a="${P2_A_SUBSETS[$i]}"
    b="${P2_B_SUBSETS[$i]}"
    setting="${a}__to__${b}"
    mem_from="${scratch_root}/${setting}"

    if _in_list "$a" "${p1_failed[@]+"${p1_failed[@]}"}"; then
        echo "  skipping ${setting} — Phase-1 of ${a} failed"
        p2_skipped+=("$setting")
        continue
    fi

    ids="$(extract_ids "$b")"
    log="${BASE_DIR}/logs/phase2_${setting}.log"
    (
        python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
            --num-workers     "${P2_WORKERS}" \
            --output-dir      "${BASE_DIR}/phase2" \
            --run-name        "${setting}" \
            --memory-type     mem0 \
            --memory-top-k    "${TOP_K}" \
            --memory-load-from "$mem_from" \
            --memory-readonly \
            --mode            FC \
            --ids              "$ids" \
        && python -m bfcl_eval.scripts.cross_episode.run_batch_evaluate \
            --input           "${BASE_DIR}/phase2/${setting}/result.jsonl" \
            --num-workers     4
    ) >"$log" 2>&1 &
    p2_pids+=("$!")
    p2_names+=("$setting")
    echo "  started ${setting} (pid $!), log: $log"
done

p2_failed=()
for i in "${!p2_pids[@]}"; do
    if ! wait "${p2_pids[$i]}"; then
        p2_failed+=("${p2_names[$i]}")
        echo "[Phase 2] FAILED: ${p2_names[$i]} — see ${BASE_DIR}/logs/phase2_${p2_names[$i]}.log"
    else
        echo "[Phase 2] done:   ${p2_names[$i]}"
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Merge summaries across all sub-runs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Merge] Building combined summary..."
python3 scripts/merge_summaries.py "${BASE_DIR}" --title "mem0 | in-env + all 12 cross-env"

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results at:              ${BASE_DIR}"
echo "  combined_summary.txt:    ${BASE_DIR}/combined_summary.txt"
echo "  combined_summary.csv:    ${BASE_DIR}/combined_summary.csv"
echo "  combined_per_sample.csv: ${BASE_DIR}/combined_per_sample.csv"
echo "  (Scratch bank copies at: ${scratch_root}/)"
echo "================================================================"

total_failed=$(( ${#p1_failed[@]} + ${#p2_failed[@]} + ${#p2_skipped[@]} ))
if [[ $total_failed -gt 0 ]]; then
    echo "[WARN] ${total_failed} sub-run(s) did not complete successfully:"
    if [[ ${#p1_failed[@]} -gt 0 ]]; then echo "  Phase 1 failed:  ${p1_failed[*]}"; fi
    if [[ ${#p2_failed[@]} -gt 0 ]]; then echo "  Phase 2 failed:  ${p2_failed[*]}"; fi
    if [[ ${#p2_skipped[@]} -gt 0 ]]; then echo "  Phase 2 skipped: ${p2_skipped[*]}"; fi
fi
exit "$total_failed"
