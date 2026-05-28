#!/bin/bash
# Run memagent-14b (recurrent memory) + DeepSeek-chat over Conflict_Resolution(2) + Accurate_Retrieval(4) datasets
# Prerequisites:
#   1. vLLM server running on port 8000:
#      bash <EvoMemBench-Memory-Systems>/memagent/run_server.sh
#   2. DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, BATCH_MODEL set in .env
# Usage: bash bash_files/sh/run_exp_memagent.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

AGENT_CFG="configs/agent_conf/RAG_Agents/memagent/Recurrent_memory_memagent-14b.yaml"

AR_DATASETS=(
    "Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml"
    "Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml"
    "Accurate_Retrieval/Ruler/QA/Ruler_qa2_421k.yaml"
    "Accurate_Retrieval/EventQA/Eventqa_full.yaml"
)

CR_DATASETS=(
    "Conflict_Resolution/Factconsolidation_sh_262k.yaml"
    "Conflict_Resolution/Factconsolidation_mh_262k.yaml"
)

ALL_DATASETS=("${CR_DATASETS[@]}" "${AR_DATASETS[@]}")

for DATA_CFG in "${ALL_DATASETS[@]}"; do
    echo "========================================================"
    echo "Agent:   $AGENT_CFG"
    echo "Dataset: $DATA_CFG"
    echo "========================================================"

    python main.py \
        --agent_config  "$AGENT_CFG" \
        --dataset_config "configs/data_conf/${DATA_CFG}"

    echo "Done: $AGENT_CFG  x  $DATA_CFG"
    echo ""
done
