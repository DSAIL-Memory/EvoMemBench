#!/bin/bash
# Run MemoBrain on all AR + CR datasets
# Usage: bash bash_files/sh/run_exp_memobrain.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

AGENT_CFG="configs/agent_conf/RAG_Agents/memobrain/Recurrent_memory_memobrain.yaml"

ALL_DATASETS=(
    "Conflict_Resolution/Factconsolidation_sh_262k.yaml"
    "Conflict_Resolution/Factconsolidation_mh_262k.yaml"
    "Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml"
    "Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml"
    "Accurate_Retrieval/Ruler/QA/Ruler_qa2_421k.yaml"
    "Accurate_Retrieval/EventQA/Eventqa_full.yaml"
)

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
