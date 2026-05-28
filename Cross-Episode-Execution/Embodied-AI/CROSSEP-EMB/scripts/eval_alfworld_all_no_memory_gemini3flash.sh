#!/bin/bash
# gemini-3-flash-preview 不使用 memory，仅测试 in-env，6 种任务类型并行，每种内部并行度 8
#
# agent 模型: gemini-3-flash-preview（aigcbest OpenAI 兼容端点，从 .env OPENAI_* 读取）
# api_mode: standard（非 Ark batch）
# judge: 无（ALFWorld 环境自带 success/fail 判定）
#
# 依赖：
#   - ALFWorld server 已在 port 36005 运行
#   - .env 中配置：OPENAI_API_KEY, OPENAI_BASE_URL

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$BASE_DIR/.env" ] && source "$BASE_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$BASE_DIR/output/alfworld_no_memory_gemini3flash_${TIMESTAMP}"
LOG_DIR="$OUTPUT_BASE/logs"

PORT=36005
MAX_ROUNDS=20
PARALLEL=8
API_MODE="standard"
MODEL="gemini-3-flash-preview"

# ──────────────────────────────────────────────
# 各任务类型的名称及测试集索引（从 mappings_test.json 提取）
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

mkdir -p "$LOG_DIR"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ──────────────────────────────────────────────
# in-env 评估（无 memory，并行度 8）
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# 汇总报告
# ──────────────────────────────────────────────
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
    "ALFWorld No-Memory gemini-3-flash-preview Final Report",
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

# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
log "Output dir : $OUTPUT_BASE"
log "Port       : $PORT"
log "Parallel   : $PARALLEL  (per task)"
log "Model      : $MODEL"
log ""

log "════════════════════════════════════════"
log "Launching 6 in-env tests in parallel (no memory, gemini-3-flash-preview)"
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
log "All 6 experiments finished!"
log "Generating final report..."
log "════════════════════════════════════════"

generate_final_report "$OUTPUT_BASE"

log ""
log "Output layout:"
log "  in_env/          — 6 task-type directories, each with per-sample JSON + summary"
log "  logs/            — Full stdout logs per experiment"
log "  final_report.txt — Cross-task comparison table"
