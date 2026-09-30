#!/usr/bin/env bash
# Stage-1 A/A noise measurement — 1 discarded warmup + 5 off-runs on sh_32k,
# against the FROZEN substrate (:8003 chat via serve_stage1_chat.sh, :8001
# bf16 embed via serve_stage1_embed.sh).  [T11]
#
# Why sh_32k and not sh_64k: sh_64k is the CONFIRMATORY subset — every
# inference spent on it before the pre-registered campaign is a spent shot.
# The transfer assumption (sh_32k noise ~ sh_64k noise on the same frozen
# substrate) is declared in stage0_results/stage1_preregistration.md.
#
# Run from the repo root on the box, with both servers up:
#   nohup bash stage0_results/stage1_aa_driver.sh > hnav/_out/pipeline/stage1_aa.log 2>&1 &
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
BASE_CFG="$MAB/configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml"
DS="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml"
PIPE="$REPO_ROOT/hnav/_out/pipeline"
mkdir -p "$PIPE"

CHAT="${HNAV_STAGE1_CHAT_URL:-http://localhost:8003/v1}"
N_RUNS=5

log() { echo "[$(date '+%F %T')] $*"; }

# preconditions: both frozen servers answer, and the MAB .env points embeds at :8001
curl -sf "$CHAT/models" >/dev/null || { log "chat $CHAT is not up — start serve_stage1_chat.sh"; exit 1; }
curl -sf "http://localhost:8001/v1/models" >/dev/null || { log ":8001 embed is not up — start serve_stage1_embed.sh"; exit 1; }
grep -q "OPENAI_BASE_URL=http://localhost:8001/v1" "$MAB/.env" || { log "$MAB/.env must route embeddings to :8001 (hnav/deploy/mab.env.template)"; exit 1; }

run_one() {  # $1=tag
    local tag="$1"
    local out_rel="./outputs/hnav_stage1aa_${tag}"
    local cfg="$PIPE/stage1aa_${tag}.yaml"
    sed -e "s|^output_dir:.*|output_dir: $out_rel|" "$BASE_CFG" > "$cfg"
    rm -rf "$MAB/outputs/hnav_stage1aa_${tag}"
    log "run $tag (HNAV_MODE=off, chat=$CHAT) ..."
    (cd "$MAB" && CUDA_VISIBLE_DEVICES= HNAV_MODE=off HNAV_DOTENV_NO_OVERRIDE=1 \
        OPENAI_BASE_URL="$CHAT" OPENAI_API_KEY=EMPTY \
        python main.py --agent_config "$cfg" --dataset_config "$DS") \
        > "$PIPE/stage1aa_${tag}.runlog" 2>&1
    local rc=$?
    log "run $tag exit=$rc"
    return $rc
}

run_one warmup || { log "WARMUP FAILED — aborting"; exit 1; }
for i in $(seq 1 $N_RUNS); do
    run_one "off_$i" || { log "off_$i FAILED — aborting"; exit 1; }
done
log "A/A DONE: $N_RUNS off runs (+1 discarded warmup). Now run:"
log "  python stage0_results/stage1_aa_analysis.py"
