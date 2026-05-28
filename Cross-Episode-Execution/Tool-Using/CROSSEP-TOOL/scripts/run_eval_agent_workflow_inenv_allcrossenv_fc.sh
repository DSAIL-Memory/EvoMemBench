#!/usr/bin/env bash
# Two-phase cross-episode evaluator using the AgentWorkflow memory backend (FC mode).
# Phase 2 covers ALL 12 ordered (a, b) cross-env pairs where a ≠ b.
#
# AgentWorkflow abstracts each successful trajectory into a generic, reusable
# workflow description via LLM induction, then stores it with a sentence embedding.
# At retrieval the top-K most query-similar workflows are injected as guidance.
# No DashScope embedder required — sentence-transformers run locally.
#
# Phase 1 (in-env): 4 subsets in parallel, each serial (workers=1) with its own store.
#   generate → evaluate per subset.
# Phase 2 (all cross-env): 12 a→b transfer pairs in parallel, read-only memory
#   loaded from Phase-1's store for subset-a, evaluated over subset-b.
#
# Usage:
#   bash scripts/run_eval_agent_workflow_inenv_allcrossenv_fc.sh \
#     [--limit-per-subset N]   # default: all 50 per subset
#     [--p2-workers N]         # default: 2
#     [--top-k K]              # default: 3
#     [--run-name NAME]        # default: inenv_allcrossenv_fc
#     [--keep-going]           # continue Phase 2 even if Phase 1 partially failed
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="CROSSEP-TOOL"
LIMIT_PER_SUBSET=""
P2_WORKERS=2
TOP_K=3
RUN_NAME="inenv_allcrossenv_fc"
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
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Move to repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ── Load env vars ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[ERROR] .env not found in $SCRIPT_DIR — please create it from bfcl_eval/.env.example"
    exit 1
fi
set -a; source .env; set +a

for var in DEEPSEEK_API_KEY BATCH_MODEL; do
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
MEMORY_TYPE="agent_workflow"
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

echo "================================================================"
echo "  BFCL Cross-Episode  [agent_workflow | in-env + all 12 cross-env | FC mode]"
echo "  Model:            ${BATCH_MODEL}"
echo "  Base dir:         ${BASE_DIR}"
echo "  Phase-1 workers:  1 (fixed, serial per subset)"
echo "  Phase-2 workers:  ${P2_WORKERS} (per pair, generate phase)"
echo "  Phase-2 pairs:    ${#P2_A_SUBSETS[@]}"
echo "  top_k:            ${TOP_K}"
[[ -n "$LIMIT_PER_SUBSET" ]] && echo "  Limit/subset:     ${LIMIT_PER_SUBSET}"
echo "================================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — in-env
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 1] Launching ${#SUBSETS[@]} subset jobs in parallel..."

p1_pids=(); p1_names=()

for subset in "${SUBSETS[@]}"; do
    ids="$(extract_ids "$subset")"
    log="${BASE_DIR}/logs/phase1_${subset}.log"
    (
        python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
            --num-workers  1 \
            --output-dir   "${BASE_DIR}/phase1" \
            --run-name     "${subset}" \
            --memory-type  agent_workflow \
            --memory-top-k "${TOP_K}" \
            --mode         FC \
            --ids          "$ids" \
        && python -m bfcl_eval.scripts.cross_episode.run_batch_evaluate \
            --input        "${BASE_DIR}/phase1/${subset}/result.jsonl" \
            --num-workers  4
    ) >"$log" 2>&1 &
    p1_pids+=("$!"); p1_names+=("$subset")
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
# Phase 2 — all cross-env pairs, read-only memory, parallel across pairs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 2] Launching ${#P2_A_SUBSETS[@]} cross-env pairs in parallel..."

p2_pids=(); p2_names=(); p2_skipped=()

for i in "${!P2_A_SUBSETS[@]}"; do
    a="${P2_A_SUBSETS[$i]}"
    b="${P2_B_SUBSETS[$i]}"
    setting="${a}__to__${b}"
    mem_from="${BASE_DIR}/phase1/${a}/memory"

    if _in_list "$a" "${p1_failed[@]+"${p1_failed[@]}"}"; then
        echo "  skipping ${setting} — Phase-1 of ${a} failed"
        p2_skipped+=("$setting"); continue
    fi

    ids="$(extract_ids "$b")"
    log="${BASE_DIR}/logs/phase2_${setting}.log"
    (
        python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
            --num-workers      "${P2_WORKERS}" \
            --output-dir       "${BASE_DIR}/phase2" \
            --run-name         "${setting}" \
            --memory-type      agent_workflow \
            --memory-top-k     "${TOP_K}" \
            --memory-load-from "$mem_from" \
            --memory-readonly \
            --mode             FC \
            --ids              "$ids" \
        && python -m bfcl_eval.scripts.cross_episode.run_batch_evaluate \
            --input            "${BASE_DIR}/phase2/${setting}/result.jsonl" \
            --num-workers      4
    ) >"$log" 2>&1 &
    p2_pids+=("$!"); p2_names+=("$setting")
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
# Merge summaries
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Merge] Building combined summary..."
python3 scripts/merge_summaries.py "${BASE_DIR}" --title "agent_workflow | in-env + all 12 cross-env"

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
    [[ ${#p1_failed[@]} -gt 0 ]] && echo "  Phase 1 failed:  ${p1_failed[*]}"
    [[ ${#p2_failed[@]} -gt 0 ]] && echo "  Phase 2 failed:  ${p2_failed[*]}"
    [[ ${#p2_skipped[@]} -gt 0 ]] && echo "  Phase 2 skipped: ${p2_skipped[*]}"
fi
exit "$total_failed"
