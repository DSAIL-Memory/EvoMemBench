#!/bin/bash
# 冒烟测试：gpt-5-mini no-memory baseline 端到端链路验证（最小规模）
#
#   in-env only: 每种任务类型取 1 条样本，6 个并行（无 cross-env）
#
# 总计 6 个实验。
# 与 eval_alfworld_all_no_memory_gpt5mini.sh 的唯一差异：
#   - 每任务 1 个 sample index
#   - PARALLEL=1

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$BASE_DIR/.env" ] && source "$BASE_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/smoke_no_memory_gpt5mini_${TIMESTAMP}"
LOG_DIR="$OUTPUT_BASE/logs"

PORT=36005
MAX_ROUNDS=20
PARALLEL=1
API_MODE="standard"
MODEL="gpt-5-mini"

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

mkdir -p "$LOG_DIR"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

run_in_env() {
    local task="$1"
    local indices="$2"
    local out_dir="$OUTPUT_BASE/in_env/$task"
    local log_file="$LOG_DIR/in_env_${task}.log"

    log "▶ START  in-env: $task"
    mkdir -p "$out_dir"

    python "$BASE_DIR/scripts/eval_alfworld_with_memory.py" \
        --port "$PORT" \
        --indices "$indices" \
        --max_rounds "$MAX_ROUNDS" \
        --parallel "$PARALLEL" \
        --api_mode "$API_MODE" \
        --model "$MODEL" \
        --no_memory \
        --output_dir "$out_dir" \
        2>&1 | tee "$log_file"

    log "✓ DONE   in-env: $task"
}

generate_final_report() {
    local output_base="$1"
    local report_file="$output_base/final_report.txt"

    python3 - <<PYEOF
import json, os, glob
from datetime import datetime

output_base = "$output_base"
report_file = "$report_file"

SHORT = {
    "pick_and_place_simple":          "pick_and_place",
    "pick_two_obj_and_place":         "pick_two_obj",
    "pick_clean_then_place_in_recep": "pick_clean",
    "pick_cool_then_place_in_recep":  "pick_cool",
    "pick_heat_then_place_in_recep":  "pick_heat",
    "look_at_obj_in_light":           "look_at_obj",
}

def shorten(name):
    for long, short in SHORT.items():
        name = name.replace(long, short)
    return name

def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

rows = []
for p in sorted(glob.glob(os.path.join(output_base, "in_env", "*", "summary.json"))):
    task = os.path.basename(os.path.dirname(p))
    s = load_summary(p)
    if s:
        rows.append({
            "task":    task,
            "n":       s.get("num_samples", 0),
            "success": s.get("success_rate", 0),
            "steps":   s.get("avg_rounds_per_sample", 0),
            "tokens":  s.get("avg_tokens_per_sample", 0),
            "lat":     s.get("avg_sample_latency_per_sample", 0),
            "in_tok":  s.get("total_prompt_tokens", 0) // max(s.get("num_samples", 1), 1),
            "out_tok": s.get("total_completion_tokens", 0) // max(s.get("num_samples", 1), 1),
        })

SEP = "=" * 80
HDR = "-" * 80
H = f"{'Task':<38} {'N':>3}  {'SuccRate':>8}  {'Avg.Step':>8}  {'Tkns/s':>6}  {'AvgLat':>7}  {'InTok':>6}  {'OutTok':>6}"

lines = [
    SEP,
    "ALFWorld No-Memory gpt-5-mini Smoke Test Report",
    f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Output : {output_base}",
    SEP,
    "",
    "[In-Env Results]",
    H, HDR,
]
for r in rows:
    lines.append(
        f"{shorten(r['task']):<38} {r['n']:>3}  {r['success']:>8.4f}  {r['steps']:>8.2f}  "
        f"{r['tokens']:>6}  {r['lat']:>6.2f}s  {r['in_tok']:>6}  {r['out_tok']:>6}"
    )
if not rows:
    lines.append("  (no results found)")

if rows:
    total_n = sum(r["n"] for r in rows)
    if total_n:
        w_succ  = sum(r["success"] * r["n"] for r in rows) / total_n
        w_steps = sum(r["steps"]   * r["n"] for r in rows) / total_n
        w_tok   = int(sum(r["tokens"] * r["n"] for r in rows) / total_n)
        w_lat   = sum(r["lat"]     * r["n"] for r in rows) / total_n
        w_in    = int(sum(r["in_tok"]  * r["n"] for r in rows) / total_n)
        w_out   = int(sum(r["out_tok"] * r["n"] for r in rows) / total_n)
    else:
        w_succ = w_steps = w_tok = w_lat = w_in = w_out = 0
    lines += [
        HDR,
        f"{'OVERALL (weighted)':<38} {total_n:>3}  {w_succ:>8.4f}  {w_steps:>8.2f}  "
        f"{w_tok:>6}  {w_lat:>6.2f}s  {w_in:>6}  {w_out:>6}"
    ]

lines += ["", SEP, f"Report saved to: {report_file}", ""]
report = "\n".join(lines)
print(report)
with open(report_file, "w") as f:
    f.write(report)
PYEOF
}

log "=== No-Memory gpt-5-mini Smoke Test ==="
log "Output dir : $OUTPUT_BASE"
log "Port       : $PORT"
log "Model      : $MODEL"
log "Samples    : 1 per task type"
log ""

log "════════════════════════════════════════"
log "Launching 6 in-env tests in parallel (no memory, $MODEL)"
log "════════════════════════════════════════"

run_in_env "$TASK_PAP" "$INDICES_PAP" &
run_in_env "$TASK_PTO" "$INDICES_PTO" &
run_in_env "$TASK_PC"  "$INDICES_PC"  &
run_in_env "$TASK_PCO" "$INDICES_PCO" &
run_in_env "$TASK_PH"  "$INDICES_PH"  &
run_in_env "$TASK_LA"  "$INDICES_LA"  &

wait || true

log ""
log "════════════════════════════════════════"
log "All 6 smoke experiments finished!"
log "Generating report..."
log "════════════════════════════════════════"

generate_final_report "$OUTPUT_BASE"

log ""
log "Smoke test output: $OUTPUT_BASE"
log "Verify: ls $OUTPUT_BASE/in_env/"
