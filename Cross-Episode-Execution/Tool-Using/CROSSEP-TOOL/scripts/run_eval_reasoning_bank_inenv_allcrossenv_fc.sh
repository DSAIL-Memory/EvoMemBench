#!/usr/bin/env bash
# Two-phase cross-episode ReasoningBank evaluator — ALL 12 cross-env pairs (FC mode).
#
# Identical to run_eval_reasoning_bank_inenv_crossenv_fc.sh but Phase 2 covers
# every ordered (a, b) pair where a ≠ b (12 pairs total) instead of only the 4
# same-cluster pairs.
#
# Phase 1 (in-env): 4 subsets run in parallel; each subset runs SERIALLY
#   (workers=1) so the bank accumulates causally; each subset has its OWN
#   bank.jsonl store.
# Phase 2 (all cross-env): 12 a→b pairs run in parallel; each loads Phase-1
#   subset-a bank as READ-ONLY and generates subset-b samples.
#   ReasoningBank loads bank.jsonl once at init — concurrent readers safe,
#   no bank copying required.
#
# Usage:
#   bash scripts/run_eval_reasoning_bank_inenv_allcrossenv_fc.sh \
#     [--limit-per-subset N]   # default: all 50 per subset (use 2-3 for smoke test)
#     [--p2-workers N]         # default: 1  (per pair, generate phase)
#     [--run-name NAME]        # default: inenv_allcrossenv_reasoning_bank_fc
#     [--keep-going]           # continue Phase 2 even if some Phase-1 subset failed
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="CROSSEP-TOOL"
LIMIT_PER_SUBSET=""
P2_WORKERS=1
RUN_NAME="inenv_allcrossenv_reasoning_bank_fc"
KEEP_GOING=false

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit-per-subset) LIMIT_PER_SUBSET="$2"; shift 2 ;;
        --p2-workers)       P2_WORKERS="$2";        shift 2 ;;
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
MEMORY_TYPE="reasoning_bank"
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
echo "  BFCL Cross-Episode  [ReasoningBank | in-env + all 12 cross-env | FC mode]"
echo "  Model:            ${BATCH_MODEL}"
echo "  Base dir:         ${BASE_DIR}"
echo "  Phase-1 workers:  1 (fixed, serial per subset)"
echo "  Phase-2 workers:  ${P2_WORKERS} (per pair, generate phase)"
echo "  Phase-2 pairs:    ${#P2_A_SUBSETS[@]}"
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
            --memory-type  reasoning_bank \
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
# Phase 2 — all cross-env pairs, read-only bank, parallel across pairs
# ReasoningBank loads bank.jsonl once at init — concurrent readers safe.
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
    mem_from="${BASE_DIR}/phase1/${a}/memory"

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
            --memory-type     reasoning_bank \
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
python3 scripts/merge_summaries.py "${BASE_DIR}" --title "ReasoningBank | in-env + all 12 cross-env"

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results at:              ${BASE_DIR}"
echo "  combined_summary.txt:    ${BASE_DIR}/combined_summary.txt"
echo "  combined_summary.csv:    ${BASE_DIR}/combined_summary.csv"
echo "  combined_per_sample.csv: ${BASE_DIR}/combined_per_sample.csv"
echo "================================================================"

total_failed=$(( ${#p1_failed[@]} + ${#p2_failed[@]} + ${#p2_skipped[@]} ))
if [[ $total_failed -gt 0 ]]; then
    echo "[WARN] ${total_failed} sub-run(s) did not complete successfully:"
    if [[ ${#p1_failed[@]} -gt 0 ]]; then echo "  Phase 1 failed:  ${p1_failed[*]}"; fi
    if [[ ${#p2_failed[@]} -gt 0 ]]; then echo "  Phase 2 failed:  ${p2_failed[*]}"; fi
    if [[ ${#p2_skipped[@]} -gt 0 ]]; then echo "  Phase 2 skipped: ${p2_skipped[*]}"; fi
fi
exit "$total_failed"
