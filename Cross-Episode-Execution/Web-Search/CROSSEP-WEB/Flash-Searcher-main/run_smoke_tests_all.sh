#!/usr/bin/env bash
# 串行执行 16 个 pipeline 脚本的冒烟测试，失败不中断，最后汇总 PASS/FAIL
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
SMOKE_ROOT="$SCRIPT_DIR/smoke_tests/$TS"
mkdir -p "$SMOKE_ROOT"
SUMMARY="$SMOKE_ROOT/SUMMARY.txt"

# 14 个默认 both-orders 脚本（会产 2 份 pipeline_report.txt）
SCRIPTS_BOTH=(
    run_full_pipeline_ace.sh
    run_full_pipeline_agent_kb.sh
    run_full_pipeline_agent_workflow_memory.sh
    run_full_pipeline_a_mem.sh
    run_full_pipeline_bm25.sh
    run_full_pipeline_gpt-5-mini.sh
    run_full_pipeline_graphrag.sh
    run_full_pipeline_lightweight_memory.sh
    run_full_pipeline_mem0.sh
    run_full_pipeline_memory_os.sh
    run_full_pipeline_memos.sh
    run_full_pipeline_qwen3_embedding.sh
    run_full_pipeline_reasoning_bank.sh
    run_full_pipeline_skillweaver.sh
)
# 2 个 nomemory 单向脚本（产 1 份 pipeline_report.txt）
SCRIPTS_SINGLE=(
    run_full_pipeline_gemini3flash_nomemory.sh
    run_full_pipeline_gpt5mini_nomemory.sh
)

declare -a RESULTS

run_one() {
    local sh="$1" expected_reports="$2"
    local name="${sh%.sh}"
    local out_dir="$SMOKE_ROOT/$name/output"
    local log_file="$SMOKE_ROOT/$name/run.log"
    mkdir -p "$out_dir"

    echo "════════════════════════════════════════════════════════════"
    echo "▶ $sh   (expected reports: $expected_reports)   $(date '+%F %T')"
    echo "════════════════════════════════════════════════════════════"

    local t0=$(date +%s)
    bash "$sh" \
        --xbench-n 3 --ww-n 3 --max-steps 10 --concurrency 2 \
        --output-dir "$out_dir" \
        > "$log_file" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local dur=$((t1 - t0))

    local report_count
    report_count=$(find "$out_dir" -maxdepth 5 -name pipeline_report.txt 2>/dev/null | wc -l)

    local status
    if [[ $rc -eq 0 && $report_count -ge $expected_reports ]]; then
        status="PASS"
    elif [[ $report_count -gt 0 ]]; then
        status="PARTIAL"
    else
        status="FAIL"
    fi

    local line
    line=$(printf "%-7s %-50s %4ds   reports=%d/%d   log=%s" \
                  "$status" "$sh" "$dur" "$report_count" "$expected_reports" "$log_file")
    echo "$line"
    RESULTS+=("$line")
}

START_ALL=$(date +%s)
for sh in "${SCRIPTS_BOTH[@]}";   do run_one "$sh" 2; done
for sh in "${SCRIPTS_SINGLE[@]}"; do run_one "$sh" 1; done
END_ALL=$(date +%s)

{
    echo "Smoke test summary  ($(date '+%F %T'))"
    echo "Total wall time: $((END_ALL - START_ALL)) s   ( $(( (END_ALL-START_ALL)/60 )) min )"
    echo "Output root: $SMOKE_ROOT"
    echo "------------------------------------------------------------"
    for r in "${RESULTS[@]}"; do echo "$r"; done
} | tee "$SUMMARY"
