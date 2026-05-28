#!/bin/bash
# 冒烟测试：ACE all-pairs 端到端链路验证（最小规模）
#
#   Phase 1 in-env:    每种任务类型取 1 条样本，6 个并行
#   Phase 2 cross-env: 仅 1 波（offset=1）× 6 对，每对 1 条样本
#
# 总计 12 个实验（6 + 6），约 10-20 分钟（batch API 异步等待为主）。
# 与 eval_alfworld_all_ace_batch_all_pairs.sh 的唯一差异：
#   - 每任务 1 个 sample index
#   - Phase 2 只跑 offset=1 一波

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/smoke_ace_all_pairs_${TIMESTAMP}"
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

# 每任务仅取首个索引
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
echo "$OUTPUT_BASE" > /tmp/alfworld_smoke_ace_all_pairs_current_output

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
base_dir  = "$BASE_DIR"

config = {
    "playbook_path": os.path.join(mem_dir, f"playbook_{task}.txt"),
    "initial_playbook_path": os.path.join(base_dir, "scripts", "prompts", "alfworld_initial_playbook.txt"),
    "ark_batch_config": {
        "api_key": "\${DEEPSEEK_API_KEY}",
        "model": "\${BATCH_MODEL}"
    },
    "reflector_prompt_path": os.path.join(base_dir, "scripts", "prompts", "alfworld_ace_reflector_prompt.txt"),
    "curator_prompt_path": os.path.join(base_dir, "scripts", "prompts", "alfworld_ace_curator_prompt.txt"),
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
        --memory_type ace \
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
        --memory_type ace \
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
        --title "ALFWorld ACE All-Pairs Smoke Test Report"
}

log "=== ACE All-Pairs Smoke Test ==="
log "Output dir : $OUTPUT_BASE"
log "Memory dir : $MEMORY_DIR"
log "Port       : $PORT"
log "Samples    : 1 per task type"
log "Cross-env  : 1 wave (offset=1), 6 pairs"
log ""
log "Generating ACE configs..."

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
