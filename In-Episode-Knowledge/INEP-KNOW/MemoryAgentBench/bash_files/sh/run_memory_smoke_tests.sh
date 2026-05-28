#!/bin/bash
# Smoke test the 4 newly-migrated memory agents: 2 queries on Factconsolidation_sh_262k with --force
# Each agent runs main.py, then verifies the output JSON contains at least 1 result.
# memobrain is skipped automatically if vLLM is not reachable on port 8001.
#
# Prerequisites: .env loaded with DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL /
#                BATCH_MODEL / OPENAI_API_KEY as needed
# Usage: bash bash_files/sh/run_memory_smoke_tests.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

SMOKE_DATASET="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml"
QUERIES=2

# ── baseline name → (agent_cfg, output_dir) ─────────────────────────────────
declare -a NAMES=(amem memos memoryos memobrain)

declare -A AGENT_CFG
AGENT_CFG[amem]="configs/agent_conf/RAG_Agents/amem/Structure_rag_amem.yaml"
AGENT_CFG[memos]="configs/agent_conf/RAG_Agents/memos/Embedding_rag_memos.yaml"
AGENT_CFG[memoryos]="configs/agent_conf/RAG_Agents/memoryos/Embedding_rag_memoryos.yaml"
AGENT_CFG[memobrain]="configs/agent_conf/RAG_Agents/memobrain/Recurrent_memory_memobrain.yaml"

declare -A OUTPUT_DIR
OUTPUT_DIR[amem]="./outputs/deepseek-chat-amem"
OUTPUT_DIR[memos]="./outputs/deepseek-chat-memos"
OUTPUT_DIR[memoryos]="./outputs/deepseek-chat-memoryos"
OUTPUT_DIR[memobrain]="./outputs/memobrain-14b-deepseek"

# ── counters ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
FAIL_NAMES=()

# ── helpers ──────────────────────────────────────────────────────────────────
check_output_json() {
    local output_dir="$1"
    local json_file
    json_file=$(find "${output_dir}/Conflict_Resolution" \
        -name "factconsolidation_sh_262k*.json" 2>/dev/null | head -1)

    if [[ -z "$json_file" ]]; then
        echo "  ERROR: output JSON not found under ${output_dir}/Conflict_Resolution/"
        return 1
    fi

    local n
    n=$(python3 -c "
import json, sys
d = json.load(open('$json_file'))
print(len(d.get('data', [])))
" 2>/dev/null)

    if [[ "${n:-0}" -ge 1 ]]; then
        echo "  OK: ${json_file##*/} — $n result(s)"
        return 0
    else
        echo "  ERROR: JSON exists but data is empty (${json_file##*/})"
        return 1
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
for name in "${NAMES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Smoke: $name"
    echo "════════════════════════════════════════════════════"

    # memobrain requires a live vLLM server on port 8001
    if [[ "$name" == "memobrain" ]]; then
        if ! curl -sf --noproxy '*' http://localhost:8001/v1/models > /dev/null 2>&1; then
            echo "  SKIP: vLLM server not reachable on port 8001"
            echo "        Start MemoBrain-14B vLLM before running this agent."
            ((SKIP++))
            continue
        fi
    fi

    env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
        HF_ENDPOINT=https://hf-mirror.com \
    python main.py \
        --agent_config  "${AGENT_CFG[$name]}" \
        --dataset_config "$SMOKE_DATASET" \
        --max_test_queries_ablation "$QUERIES" \
        --force
    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo "  FAIL: main.py exited with code $exit_code"
        ((FAIL++))
        FAIL_NAMES+=("$name")
        continue
    fi

    if check_output_json "${OUTPUT_DIR[$name]}"; then
        ((PASS++))
    else
        ((FAIL++))
        FAIL_NAMES+=("$name")
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Smoke test summary"
echo "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
if [[ ${#FAIL_NAMES[@]} -gt 0 ]]; then
    echo "  Failed baselines: ${FAIL_NAMES[*]}"
fi
echo "════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]   # exit 0 only if no failures (SKIP is acceptable)
