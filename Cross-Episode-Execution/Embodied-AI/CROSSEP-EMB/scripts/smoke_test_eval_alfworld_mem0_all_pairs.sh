#!/bin/bash
# Smoke test: mem0 all-pairs end-to-end pipeline validation (minimal scale)
#
#   Phase 1 in-env:    take 1 sample from each task type, with 6-way parallelism
#   Phase 2 cross-env: only 1 wave (offset=1) x 6 pairs, with 1 sample per pair
#
# Total: 12 experiments (6 + 6), about 10-20 minutes, mostly waiting for the async Batch API.
# Differs from eval_alfworld_all_mem0_batch_all_pairs.sh:
#   - 1 sample index per task
#   - PORT=36005 (matches the currently running ALFWorld server)
#   - Phase 2 only runs one wave with offset=1

set -euo pipefail

MEM0_TOPK=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topk|-k) MEM0_TOPK="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/smoke_mem0_all_pairs_${TIMESTAMP}"
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

# Use only the first index for each task
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
echo "$OUTPUT_BASE" > /tmp/alfworld_smoke_mem0_all_pairs_current_output

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

wait_for_done() {
    local marker="$1"
    while [ ! -f "$marker" ]; do
        sleep 5
    done
}

generate_config() {
    local task="$1"
    local out_file="$CONFIG_DIR/${task}.json"

    python3 - <<PYEOF
import json, os

task      = "$task"
mem_dir   = "$MEMORY_DIR"
config = {
    "agent_id": task,
    "search_limit": $MEM0_TOPK,
    "history_db_path": os.path.join(mem_dir, f"history_{task}.db"),
    "ark_batch_config": {
        "api_key": "\${DEEPSEEK_API_KEY}",
        "model": "\${BATCH_MODEL}"
    },
    "embedder_config": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v4",
            "api_key": "\${DASHSCOPE_API_KEY}",
            "openai_base_url": "\${DASHSCOPE_BASE_URL}"
        }
    },
    "llm_config": {
        "provider": "openai",
        "config": {
            "model": "deepseek-v3-2-251201",
            "api_key": "\${DEEPSEEK_API_KEY}",
            "openai_base_url": "\${DEEPSEEK_BASE_URL}"
        }
    },
    "vector_store_config": {
        "provider": "chroma",
        "config": {
            "collection_name": task,
            "path": os.path.join(mem_dir, task)
        }
    }
}
with open("$out_file", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Config written: $out_file")
PYEOF
}

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
        --memory_type mem0 \
        --memory_config "$cfg" \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"

    log "✓ DONE   in-env: $task"
}

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
        --memory_type mem0 \
        --memory_config "$src_cfg" \
        --readonly_memory \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"

    log "✓ DONE   cross-env: $src_task → $tgt_task"
}

generate_final_report() {
    local output_base="$1"
    python3 "$BASE_DIR/scripts/generate_alfworld_report.py" \
        "$output_base" \
        --title "ALFWorld Mem0 All-Pairs Smoke Test Report"
}

log "=== Mem0 All-Pairs Smoke Test ==="
log "Output dir : $OUTPUT_BASE"
log "Memory dir : $MEMORY_DIR"
log "Port       : $PORT"
log "Mem0 TopK  : $MEM0_TOPK"
log "Samples    : 1 per task type"
log "Cross-env  : 1 wave (offset=1), 6 pairs"
log ""
log "Generating memory configs..."

for task in "${TASKS[@]}"; do
    generate_config "$task"
done
log "All configs generated."
log ""

log "════════════════════════════════════════"
log "Phase 1: 6 in-env (1 sample each) in parallel"
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
log "Phase 2: 1 wave × 6 pairs cross-env"
log "════════════════════════════════════════"

for offset in 1; do
    log "── Wave $offset/1 ──"
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
log "All 12 smoke experiments finished (6 in-env + 6 cross-env)!"
log "Generating report..."
log "════════════════════════════════════════"

generate_final_report "$OUTPUT_BASE"

log ""
log "Smoke test output: $OUTPUT_BASE"
log "Verify: ls $OUTPUT_BASE/in_env/ $OUTPUT_BASE/cross_env/"
