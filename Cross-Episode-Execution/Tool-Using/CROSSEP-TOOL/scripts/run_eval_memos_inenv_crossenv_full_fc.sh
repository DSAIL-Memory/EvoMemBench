#!/usr/bin/env bash
# Two-phase cross-episode memos evaluator — FULL 4×4 cross-env matrix (FC mode).
#
# Identical to run_eval_memos_inenv_crossenv_fc.sh but Phase 2 tests all
# N*(N-1) = 12 ordered pairs (a→b, a≠b) instead of the 4-pair subset.
#
# Phase 1 (in-env): 4 subsets run in parallel; each subset runs SERIALLY
#   (workers=1) so memory accumulates causally; each subset has its OWN store.
#   Within each subset: generate (inference, writes result.jsonl) then evaluate
#   (post-hoc decode + checker + progress_score, writes per-sample JSONs + summary).
#   Skip with --phase1-dir to reuse an existing Phase-1 output directory.
# Phase 2 (cross-env): 12 a→b settings run in parallel; each loads Phase-1
#   subset-a memory as a READ-ONLY initial store and generates subset-b samples
#   in parallel (no memory writes → no causality constraint), then evaluates.
#
# Usage:
#   bash scripts/run_eval_memos_inenv_crossenv_full_fc.sh \
#     [--limit-per-subset N]   # default: all 50 per subset (use 3 for smoke test)
#     [--p2-workers N]         # default: 2  (per Phase-2 setting, generate phase)
#     [--top-k K]              # default: 3
#     [--run-name NAME]        # default: inenv_crossenv_full_memos_fc
#     [--keep-going]           # continue Phase 2 even if some Phase-1 subset failed
#     [--phase1-dir PATH]      # skip Phase 1 and load memory from PATH/{subset}/memory
#
# Example (reuse existing Phase-1):
#   bash scripts/run_eval_memos_inenv_crossenv_full_fc.sh \
#     --phase1-dir cross_episode_results/runs/inenv_crossenv_memos_fc_memos_20260430_031031/phase1
#
# Phase-2 settings (12 pairs):
#   gorilla_fs      → vehicle_control, trading_bot, travel_api
#   vehicle_control → gorilla_fs,      trading_bot, travel_api
#   trading_bot     → gorilla_fs,      vehicle_control, travel_api
#   travel_api      → gorilla_fs,      vehicle_control, trading_bot
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
ENV_NAME="CROSSEP-TOOL"
LIMIT_PER_SUBSET=""
P2_WORKERS=2
TOP_K=3
RUN_NAME="inenv_crossenv_full_memos_fc"
KEEP_GOING=false
PHASE1_DIR=""   # if set, skip Phase 1 and load memory from here

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit-per-subset) LIMIT_PER_SUBSET="$2"; shift 2 ;;
        --p2-workers)       P2_WORKERS="$2";        shift 2 ;;
        --top-k)            TOP_K="$2";             shift 2 ;;
        --run-name)         RUN_NAME="$2";          shift 2 ;;
        --keep-going)       KEEP_GOING=true;        shift   ;;
        --phase1-dir)       PHASE1_DIR="$2";        shift 2 ;;
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

# ── Resolve Phase-1 source dir ────────────────────────────────────────────────
if [[ -n "$PHASE1_DIR" ]]; then
    PHASE1_DIR="$(cd "$PHASE1_DIR" && pwd)"
fi

# ── Paths and timestamps ──────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
MEMORY_TYPE="memos"
BASE_DIR="cross_episode_results/runs/${RUN_NAME}_${MEMORY_TYPE}_${TS}"
mkdir -p "${BASE_DIR}/logs"

# ── Subset config ─────────────────────────────────────────────────────────────
SUBSETS=(gorilla_fs vehicle_control trading_bot travel_api)

# ── Banner ────────────────────────────────────────────────────────────────────
N=${#SUBSETS[@]}
N_PAIRS=$(( N * (N - 1) ))
echo "================================================================"
echo "  BFCL Cross-Episode  [memos | in-env + FULL cross-env | FC mode]"
echo "  Model:            ${BATCH_MODEL}"
echo "  Base dir:         ${BASE_DIR}"
if [[ -n "$PHASE1_DIR" ]]; then
echo "  Phase-1:          SKIPPED — loading from ${PHASE1_DIR}"
else
echo "  Phase-1 subsets:  ${N} (serial per subset)"
fi
echo "  Phase-2 pairs:    ${N_PAIRS} (all ordered a→b, a≠b)"
echo "  Phase-2 workers:  ${P2_WORKERS} (per setting, generate phase)"
echo "  top_k:            ${TOP_K}"
[[ -n "$LIMIT_PER_SUBSET" ]] && echo "  Limit/subset:     ${LIMIT_PER_SUBSET}"
echo "================================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — in-env, generate then evaluate, serial within each subset
#   Skipped when --phase1-dir is provided.
# ══════════════════════════════════════════════════════════════════════════════
p1_failed=()

if [[ -n "$PHASE1_DIR" ]]; then
    echo ""
    echo "[Phase 1] Skipped — using existing Phase-1 dir: ${PHASE1_DIR}"
    for subset in "${SUBSETS[@]}"; do
        mem_check="${PHASE1_DIR}/${subset}/memory"
        if [[ ! -d "$mem_check" ]]; then
            echo "[ERROR] Expected memory dir not found: ${mem_check}"
            exit 1
        fi
        echo "  verified ${subset}: ${mem_check}"
    done
else
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
                --memory-type  memos \
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
        echo "[Phase 1] --keep-going set; proceeding to Phase 2 (affected settings will be skipped)."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — full 4×4 cross-env matrix (a≠b), read-only memory
#
# memos uses Qdrant in local (file-based) mode, which prohibits concurrent
# access to the same storage directory. Because each source subset (a) feeds
# N-1 = 3 target settings that all run in parallel, we copy the source memory
# once per setting so every job gets its own private Qdrant instance.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 2] Pre-copying source memories (one copy per setting)..."
MEM_COPIES_DIR="${BASE_DIR}/phase2_mem_copies"
mkdir -p "${MEM_COPIES_DIR}"
p1_root="${PHASE1_DIR:-${BASE_DIR}/phase1}"

for a in "${SUBSETS[@]}"; do
    if _in_list "$a" "${p1_failed[@]+"${p1_failed[@]}"}"; then
        continue
    fi
    src="${p1_root}/${a}/memory"
    for b in "${SUBSETS[@]}"; do
        [[ "$a" == "$b" ]] && continue
        dst="${MEM_COPIES_DIR}/${a}__to__${b}"
        cp -r "$src" "$dst"
        echo "  copied ${a}/memory → phase2_mem_copies/${a}__to__${b}"
    done
done

echo ""
echo "[Phase 2] Launching ${N_PAIRS} cross-env settings in parallel..."

p2_pids=()
p2_names=()
p2_skipped=()

for a in "${SUBSETS[@]}"; do
    for b in "${SUBSETS[@]}"; do
        [[ "$a" == "$b" ]] && continue

        setting="${a}__to__${b}"
        mem_from="${MEM_COPIES_DIR}/${setting}"

        if _in_list "$a" "${p1_failed[@]+"${p1_failed[@]}"}"; then
            echo "  skipping ${setting} — Phase-1 of ${a} failed"
            p2_skipped+=("$setting")
            continue
        fi

        ids="$(extract_ids "$b")"
        log="${BASE_DIR}/logs/phase2_${setting}.log"
        (
            python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
                --num-workers      "${P2_WORKERS}" \
                --output-dir       "${BASE_DIR}/phase2" \
                --run-name         "${setting}" \
                --memory-type      memos \
                --memory-top-k     "${TOP_K}" \
                --memory-load-from "$mem_from" \
                --memory-readonly \
                --mode             FC \
                --ids              "$ids" \
            && python -m bfcl_eval.scripts.cross_episode.run_batch_evaluate \
                --input            "${BASE_DIR}/phase2/${setting}/result.jsonl" \
                --num-workers      4
        ) >"$log" 2>&1 &
        p2_pids+=("$!")
        p2_names+=("$setting")
        echo "  started ${setting} (pid $!), log: $log"
    done
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
python3 scripts/merge_summaries.py "${BASE_DIR}"

# ── Cleanup memory copies ─────────────────────────────────────────────────────
if [[ -d "${MEM_COPIES_DIR:-}" ]]; then
    echo "[Cleanup] Removing memory copies: ${MEM_COPIES_DIR}"
    rm -rf "${MEM_COPIES_DIR}"
fi

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
