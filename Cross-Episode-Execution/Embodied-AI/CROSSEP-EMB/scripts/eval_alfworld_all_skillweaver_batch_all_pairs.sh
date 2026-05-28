#!/bin/bash
# Full evaluation: SkillWeaver memory backend, all in-env + all-pairs cross-env

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topk|-k) shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/alfworld_skillweaver_batch_all_pairs_${TIMESTAMP}"
MEMORY_DIR="$OUTPUT_BASE/memory"
CONFIG_DIR="$OUTPUT_BASE/configs"
LOG_DIR="$OUTPUT_BASE/logs"

PORT=36005
MAX_ROUNDS=20
PARALLEL_CROSS=1
API_MODE="batch"

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
echo "$OUTPUT_BASE" > /tmp/alfworld_skillweaver_all_pairs_current_output

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

wait_for_done() {
    local marker="$1"
    while [ ! -f "$marker" ]; do sleep 5; done
}

generate_config() {
    local task="$1"
    local out_file="$CONFIG_DIR/${task}.json"
    python3 - <<PYEOF
import json, os
task    = "$task"
mem_dir = "$MEMORY_DIR"
config = {
    "skills_dir": os.path.join(mem_dir, f"skillweaver_{task}"),
    "ark_batch_config": {
        "api_key": "\${DEEPSEEK_API_KEY}",
        "model":   "\${BATCH_MODEL}"
    }
}
with open("$out_file", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Config written: $out_file")
PYEOF
}

run_in_env() {
    local task="$1"; local indices="$2"
    local out_dir="$OUTPUT_BASE/in_env/$task"
    local log_file="$LOG_DIR/in_env_${task}.log"
    local cfg="$CONFIG_DIR/${task}.json"
    log "▶ START  in-env: $task"
    mkdir -p "$out_dir"
    python "$BASE_DIR/scripts/eval_alfworld_with_memory.py" \
        --port "$PORT" --indices "$indices" --max_rounds "$MAX_ROUNDS" \
        --parallel 1 --api_mode "$API_MODE" \
        --memory_type skillweaver --memory_config "$cfg" \
        --output_dir "$out_dir" 2>&1 | tee "$log_file"
    log "✓ DONE   in-env: $task"
}

run_cross_env() {
    local src_task="$1"; local tgt_task="$2"; local tgt_indices="$3"
    local pair_label="${src_task}__to__${tgt_task}"
    local out_dir="$OUTPUT_BASE/cross_env/$pair_label"
    local log_file="$LOG_DIR/cross_${pair_label}.log"
    local src_cfg="$CONFIG_DIR/${src_task}.json"
    log "▶ START  cross-env: $src_task → $tgt_task"
    mkdir -p "$out_dir"
    python "$BASE_DIR/scripts/eval_alfworld_with_memory.py" \
        --port "$PORT" --indices "$tgt_indices" --max_rounds "$MAX_ROUNDS" \
        --parallel "$PARALLEL_CROSS" --api_mode "$API_MODE" \
        --memory_type skillweaver --memory_config "$src_cfg" \
        --readonly_memory --output_dir "$out_dir" 2>&1 | tee "$log_file"
    log "✓ DONE   cross-env: $src_task → $tgt_task"
}

generate_final_report() {
    python3 "$BASE_DIR/scripts/generate_alfworld_report.py" "$1" --title "ALFWorld SkillWeaver All-Pairs Batch Final Report"
}

log "Output dir : $OUTPUT_BASE"
log "Memory dir : $MEMORY_DIR"
log "Port       : $PORT"

for task in "${TASKS[@]}"; do generate_config "$task"; done
log "All configs generated."

log "════════════════════════════════════════"
log "Phase 1 — In-Environment Evaluation (6 tasks in parallel)"
log "════════════════════════════════════════"

run_in_env "$TASK_PAP" "$INDICES_PAP" &
run_in_env "$TASK_PTO" "$INDICES_PTO" &
run_in_env "$TASK_PC"  "$INDICES_PC"  &
run_in_env "$TASK_PCO" "$INDICES_PCO" &
run_in_env "$TASK_PH"  "$INDICES_PH"  &
run_in_env "$TASK_LA"  "$INDICES_LA"  &

log "Waiting for all 6 in-env summaries..."
for task in "${TASKS[@]}"; do
    wait_for_done "$OUTPUT_BASE/in_env/$task/summary.json"
    log "✓ in-env done: $task"
done
wait || true

log ""
log "════════════════════════════════════════"
log "Phase 2 — Cross-Environment Evaluation (all-pairs, 5 waves of 6)"
log "════════════════════════════════════════"

for offset in 1 2 3 4 5; do
    log "── Wave $offset/5 ──"
    for i in 0 1 2 3 4 5; do
        tgt_i=$(( (i + offset) % 6 ))
        src="${TASKS[$i]}"; tgt="${TASKS[$tgt_i]}"
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
log "  memory/          — skillweaver_<task>/ skill directories per task type"
log "  configs/         — JSON config files"
log "  logs/            — Full stdout logs per experiment"
log "  final_report.txt — Cross-experiment comparison table"
