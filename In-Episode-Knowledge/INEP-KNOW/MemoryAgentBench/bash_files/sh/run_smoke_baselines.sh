#!/bin/bash
# Smoke test all 8 baselines: 2 queries on Factconsolidation_sh_262k with --force
# Each baseline runs main.py, then verifies the output JSON contains at least 1 result.
# memagent is skipped automatically if vLLM is not reachable on port 8000.
#
# Prerequisites: .env loaded with OPENAI_API_KEY / DEEPSEEK_API_KEY etc. as needed
# Usage: bash bash_files/sh/run_smoke_baselines.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

SMOKE_DATASET="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml"
QUERIES=2

# ── baseline name → (agent_cfg, output_dir) ─────────────────────────────────
declare -a NAMES=(bm25 graphrag mem0 qwen3_emb deepseek_lc gpt5_mini gemini3_flash memagent)

declare -A AGENT_CFG
AGENT_CFG[bm25]="configs/agent_conf/RAG_Agents/deepseek-chat/Simple_rag_deepseek-chat-bm25.yaml"
AGENT_CFG[graphrag]="configs/agent_conf/RAG_Agents/deepseek-chat/Structure_rag_deepseek-chat-graph-rag.yaml"
AGENT_CFG[mem0]="configs/agent_conf/RAG_Agents/deepseek-chat/Structure_rag_deepseek-chat-mem0.yaml"
AGENT_CFG[qwen3_emb]="configs/agent_conf/RAG_Agents/deepseek-chat/Embedding_rag_deepseek-chat-qwen3_embedding_4b.yaml"
AGENT_CFG[deepseek_lc]="configs/agent_conf/Long_Context_Agents/Long_context_agent_deepseek-chat.yaml"
AGENT_CFG[gpt5_mini]="configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-5-mini.yaml"
AGENT_CFG[gemini3_flash]="configs/agent_conf/Long_Context_Agents/Long_context_agent_gemini-3-flash.yaml"
AGENT_CFG[memagent]="configs/agent_conf/RAG_Agents/memagent/Recurrent_memory_memagent-14b.yaml"

declare -A OUTPUT_DIR
OUTPUT_DIR[bm25]="./outputs/deepseek-chat-bm25"
OUTPUT_DIR[graphrag]="./outputs/deepseek-chat-graph-rag"
OUTPUT_DIR[mem0]="./outputs/deepseek-chat-mem0"
OUTPUT_DIR[qwen3_emb]="./outputs/deepseek-chat-qwen3_embedding_4b"
OUTPUT_DIR[deepseek_lc]="./outputs/deepseek-chat"
OUTPUT_DIR[gpt5_mini]="./outputs/gpt-5-mini"
OUTPUT_DIR[gemini3_flash]="./outputs/gemini-3-flash"
OUTPUT_DIR[memagent]="./outputs/deepseek-chat-memagent-14b"

# ── counters ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
FAIL_NAMES=()

# ── helpers ──────────────────────────────────────────────────────────────────
check_output_json() {
    local output_dir="$1"
    # Find any matching JSON in the CR sub-directory (file may be pre-existing if
    # all queries were already done and save_results_to_file was skipped this run)
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

    # memagent requires a live vLLM server
    if [[ "$name" == "memagent" ]]; then
        if ! curl -sf --noproxy '*' http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "  SKIP: vLLM server not reachable on port 8000"
            echo "        Start it with: bash <EvoMemBench-Memory-Systems>/memagent/run_server.sh"
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
