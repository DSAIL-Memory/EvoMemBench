#!/bin/bash
# Smoke test: AWM (Agent Workflow Memory) backend, 1 sample per task, 1 wave cross-env

set -euo pipefail

TOPK=3
INDUCTION=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topk|-k) TOPK="$2"; shift 2 ;;
        --no-induction) INDUCTION=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/smoke_awm_all_pairs_${TIMESTAMP}"
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

INDICES_PAP='[2423]'
INDICES_PTO='[2421]'
INDICES_PC='[2422]'
INDICES_PCO='[2420]'
INDICES_PH='[2424]'
INDICES_LA='[2427]'

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
echo "$OUTPUT_BASE" > /tmp/alfworld_smoke_awm_all_pairs_current_output

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
    "store_path": os.path.join(mem_dir, f"awm_{task}.json"),
    "top_k": $TOPK,
    "enable_induction": $( [ "$INDUCTION" = "true" ] && echo "True" || echo "False" ),
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
        --memory_type agent_workflow_memory --memory_config "$cfg" \
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
        --memory_type agent_workflow_memory --memory_config "$src_cfg" \
        --readonly_memory --output_dir "$out_dir" 2>&1 | tee "$log_file"
    log "✓ DONE   cross-env: $src_task → $tgt_task"
}

generate_final_report() {
    python3 "$BASE_DIR/scripts/generate_alfworld_report.py" "$1" --title "ALFWorld AWM All-Pairs Smoke Test Report"
}

log "Output dir : $OUTPUT_BASE"
log "Memory dir : $MEMORY_DIR"
log "Port       : $PORT"
log "TopK       : $TOPK"
log "Induction  : $INDUCTION"

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
log "Phase 2 — Cross-Environment Evaluation (1 wave of 6)"
log "════════════════════════════════════════"

for offset in 1; do
    log "── Wave $offset/1 ──"
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
log "All 12 experiments finished (6 in-env + 6 cross-env)!"
log "Generating final report..."
log "════════════════════════════════════════"

generate_final_report "$OUTPUT_BASE"

log ""
log "Output layout:"
log "  in_env/          — 6 task-type directories, each with per-sample JSON + summary"
log "  cross_env/       — 6 transfer pair directories"
log "  memory/          — awm_<task>.json workflow store files"
log "  configs/         — JSON config files"
log "  logs/            — Full stdout logs per experiment"
log "  final_report.txt — Cross-experiment comparison table"
