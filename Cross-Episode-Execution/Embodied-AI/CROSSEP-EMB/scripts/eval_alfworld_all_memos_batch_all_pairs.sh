#!/bin/bash
# Evaluate ALFWorld with MemOS (memos) memory in two settings:
#
#   Phase 1 in-env:    each task type accumulates its own memory; 6-way parallelism across tasks and serial execution within each task (--parallel 1)
#   Phase 2 cross-env: all 30 pairs (6 sources x 5 targets), 5 waves x 6 pairs in round-robin order
#
# Wave scheduling (each source appears only once per wave to avoid Qdrant exclusive file-lock conflicts):
#   Wave k: TASKS[i] → TASKS[(i+k) mod 6]  (i=0..5, k=1..5)
#   Within a wave, the Qdrant directory for a given source is read by only one process, so no per-pair directory copy is needed.
#
# Phase 2 starts after all 6 in-env runs finish. cross-env uses --parallel 1 internally.
#
# Dependencies:
#   - ALFWorld server is already running on port 36005
#   - pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
#   - pip install tiktoken
#   - pip install 'volcengine-python-sdk[ark]'
#   - .env contains: DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DEEPSEEK_API_KEY,
#                  DEEPSEEK_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL, BATCH_MODEL

set -euo pipefail

# ──────────────────────────────────────────────
# Command-line argument parsing
# ──────────────────────────────────────────────
MEMOS_TOPK=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topk|-k) MEMOS_TOPK="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$BASE_DIR/.env" ] && source "$BASE_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/alfworld_memos_batch_all_pairs_${TIMESTAMP}"
MEMORY_DIR="$OUTPUT_BASE/memory"
CONFIG_DIR="$OUTPUT_BASE/configs"
LOG_DIR="$OUTPUT_BASE/logs"

PORT=36005
MAX_ROUNDS=20
PARALLEL_CROSS=1

API_MODE="batch"

# ──────────────────────────────────────────────
# Task type names and test-set indices
# ──────────────────────────────────────────────
TASK_PAP="pick_and_place_simple"
TASK_PTO="pick_two_obj_and_place"
TASK_PC="pick_clean_then_place_in_recep"
TASK_PCO="pick_cool_then_place_in_recep"
TASK_PH="pick_heat_then_place_in_recep"
TASK_LA="look_at_obj_in_light"

INDICES_PAP='[2423,2426,2429,2433,2438,2441,2444,2445,2450,2457,2468,2485,2491,2492,2497,2503,2504,2512,2514,2518,2525,2528,2529,2533,2540,2544,2561,2562,2565,2569,2572,2576,2583,2585,2587,2588,2590,2591,2600,2601,2606,2607,2610,2616,2617,2618]'
INDICES_PTO='[2421,2428,2430,2432,2437,2439,2442,2446,2447,2466,2469,2472,2473,2475,2479,2489,2490,2495,2496,2498,2502,2508,2513,2519,2520,2523,2530,2538,2541,2546,2559,2560,2564,2566,2573,2575,2582,2586,2589,2596,2605,2608,2609,2615,2619]'
INDICES_PC='[2422,2425,2434,2440,2448,2452,2455,2456,2460,2463,2467,2480,2501,2505,2516,2517,2524,2526,2534,2536,2539,2542,2545,2548,2550,2552,2553,2554,2558,2568,2574,2578,2592,2593,2604,2613,2614]'
INDICES_PCO='[2420,2431,2435,2451,2454,2459,2461,2462,2470,2476,2477,2481,2493,2499,2506,2509,2510,2511,2515,2521,2522,2537,2556,2563,2579,2580,2599,2612]'
INDICES_PH='[2424,2436,2443,2449,2453,2464,2465,2471,2474,2478,2494,2500,2527,2531,2535,2543,2549,2555,2567,2577,2594,2595,2597,2602,2611]'
INDICES_LA='[2427,2458,2482,2483,2484,2486,2487,2488,2507,2532,2547,2551,2557,2570,2571,2581,2584,2598,2603]'

# Ordered task array plus index map for wave round-robin scheduling
TASKS=("$TASK_PAP" "$TASK_PTO" "$TASK_PC" "$TASK_PCO" "$TASK_PH" "$TASK_LA")
declare -A INDICES_MAP=(
    ["$TASK_PAP"]="$INDICES_PAP"
    ["$TASK_PTO"]="$INDICES_PTO"
    ["$TASK_PC"]="$INDICES_PC"
    ["$TASK_PCO"]="$INDICES_PCO"
    ["$TASK_PH"]="$INDICES_PH"
    ["$TASK_LA"]="$INDICES_LA"
)

mkdir -p "$MEMORY_DIR" "$CONFIG_DIR" "$LOG_DIR"
echo "$OUTPUT_BASE" > /tmp/alfworld_memos_all_pairs_current_output

# ──────────────────────────────────────────────
# Logging and status helpers
# ──────────────────────────────────────────────
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

wait_for_done() {
    local marker="$1"
    while [ ! -f "$marker" ]; do
        sleep 5
    done
}

# ──────────────────────────────────────────────
# Generate memos config JSON
# ──────────────────────────────────────────────
generate_config() {
    local task="$1"
    local out_file="$CONFIG_DIR/${task}.json"

    python3 - <<PYEOF
import json, os

task    = "$task"
mem_dir = "$MEMORY_DIR"
config = {
    "agent_id": task,
    "search_limit": $MEMOS_TOPK,
    "qdrant_path": os.path.join(mem_dir, "qdrant", task),
    "collection_name": task,
    "vector_dimension": 1024,
    "llm": {
        "model": "deepseek-v3-2-251201",
        "api_key": "\${DEEPSEEK_API_KEY}",
        "base_url": "\${DEEPSEEK_BASE_URL}"
    },
    "embed": {
        "model": "text-embedding-v4",
        "api_key": "\${DASHSCOPE_API_KEY}",
        "base_url": "\${DASHSCOPE_BASE_URL}"
    }
}
with open("$out_file", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Config written: $out_file")
PYEOF
}

# ──────────────────────────────────────────────
# In-env evaluation: accumulate and read memory on each task's own test set
# ──────────────────────────────────────────────
run_in_env() {
    local task="$1"
    local indices="$2"
    local out_dir="$OUTPUT_BASE/in_env/$task"
    local log_file="$LOG_DIR/in_env_${task}.log"
    local cfg="$CONFIG_DIR/${task}.json"

    log "▶ START  in-env: $task"
    mkdir -p "$out_dir"

    python "$BASE_DIR/scripts/eval_alfworld_with_memory.py" \
        --port "$PORT" \
        --indices "$indices" \
        --max_rounds "$MAX_ROUNDS" \
        --parallel 1 \
        --api_mode "$API_MODE" \
        --memory_type memos \
        --memory_config "$cfg" \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"

    log "✓ DONE   in-env: $task"
}

# ──────────────────────────────────────────────
# Cross-env evaluation: use the source memory in read-only mode on the target test set
#
# Qdrant concurrency note: wave round-robin ensures each source directory is used by at most one process per wave.
# Therefore, no per-pair directory copy is needed when combined with serial wave waits.
# ──────────────────────────────────────────────
run_cross_env() {
    local src_task="$1"
    local tgt_task="$2"
    local tgt_indices="$3"
    local pair_label="${src_task}__to__${tgt_task}"
    local out_dir="$OUTPUT_BASE/cross_env/$pair_label"
    local log_file="$LOG_DIR/cross_${pair_label}.log"
    local src_cfg="$CONFIG_DIR/${src_task}.json"

    log "▶ START  cross-env: $src_task → $tgt_task"
    mkdir -p "$out_dir"

    python "$BASE_DIR/scripts/eval_alfworld_with_memory.py" \
        --port "$PORT" \
        --indices "$tgt_indices" \
        --max_rounds "$MAX_ROUNDS" \
        --parallel "$PARALLEL_CROSS" \
        --api_mode "$API_MODE" \
        --memory_type memos \
        --memory_config "$src_cfg" \
        --readonly_memory \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"

    log "✓ DONE   cross-env: $src_task → $tgt_task"
}

# ──────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────
generate_final_report() {
    local output_base="$1"
    python3 "$BASE_DIR/scripts/generate_alfworld_report.py" \
        "$output_base" \
        --title "ALFWorld MemOS All-Pairs Batch Final Report"
}

# ──────────────────────────────────────────────
# Main flow
# ──────────────────────────────────────────────
log "Output dir    : $OUTPUT_BASE"
log "Memory dir    : $MEMORY_DIR"
log "Port          : $PORT"
log "MemOS TopK    : $MEMOS_TOPK"
log "Cross PARALLEL: $PARALLEL_CROSS  (in-env is always 1)"
log ""
log "Generating memory configs..."

for task in "${TASKS[@]}"; do
    generate_config "$task"
done
log "All configs generated."
log ""

log "════════════════════════════════════════"
log "Phase 1: launching 6 in-env tests in parallel"
log "Phase 2: all 30 cross-env pairs in 5 waves after all in-env complete"
log "════════════════════════════════════════"

# -- Phase 1: 6 in-env runs in parallel ----------------
run_in_env "$TASK_PAP" "$INDICES_PAP" &
run_in_env "$TASK_PTO" "$INDICES_PTO" &
run_in_env "$TASK_PC"  "$INDICES_PC"  &
run_in_env "$TASK_PCO" "$INDICES_PCO" &
run_in_env "$TASK_PH"  "$INDICES_PH"  &
run_in_env "$TASK_LA"  "$INDICES_LA"  &

# -- Wait for all in-env summary.json files ------------
log "Waiting for all 6 in-env summaries..."
for task in "${TASKS[@]}"; do
    wait_for_done "$OUTPUT_BASE/in_env/$task/summary.json"
    log "✓ in-env done: $task"
done
wait || true

# -- Phase 2: 5 waves x 6 round-robin pairs (30 total) --
# Wave k: TASKS[i] → TASKS[(i+k) mod 6], each source is unique within a wave (Qdrant-safe)
log ""
log "════════════════════════════════════════"
log "Phase 2: 5 waves × 6 pairs = 30 cross-env (--parallel $PARALLEL_CROSS each)"
log "════════════════════════════════════════"

for offset in 1 2 3 4 5; do
    log "── Wave $offset/5 ──"
    for i in 0 1 2 3 4 5; do
        tgt_i=$(( (i + offset) % 6 ))
        src="${TASKS[$i]}"
        tgt="${TASKS[$tgt_i]}"
        run_cross_env "$src" "$tgt" "${INDICES_MAP[$tgt]}" &
    done
    wait || true
    log "── Wave $offset done ──"
done

log ""
log "════════════════════════════════════════"
log "All 36 experiments finished (6 in-env + 30 cross-env)!"
log "Generating final report..."
log "════════════════════════════════════════"

generate_final_report "$OUTPUT_BASE"

log ""
log "Output layout:"
log "  in_env/          — 6 task-type directories, each with per-sample JSON + summary"
log "  cross_env/       — 30 transfer pair directories"
log "  memory/qdrant/   — Qdrant vector stores per task"
log "  configs/         — JSON config files (for reference)"
log "  logs/            — Full stdout logs per experiment"
log "  final_report.txt — Cross-experiment comparison table"
