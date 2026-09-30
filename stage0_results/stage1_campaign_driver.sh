#!/usr/bin/env bash
# Stage-1 CONFIRMATORY campaign — sh_64k, 7 off + 7 live, SINGLE SHOT.  [T11]
#
#   *** DO NOT RUN before stage0_results/stage1_preregistration.md and
#   *** stage0_results/stage1_operating_point.json are COMMITTED and the
#   *** supervisor audit of both has passed. This script refuses otherwise.
#
# Pre-registered execution order (fixed, no optional stopping):
#   W  O L  O L  O L  O L  O L  O L  O L      (W = off warmup, DISCARDED)
# Both arms run against the FROZEN substrate (:8003 chat, :8001 bf16 embed).
# Live arms: HNAV_MODE=live, NLI on CPU (arms never see a GPU), per-run
# HNAV_RUN_ID so each live run leaves its own audit trail under
# hnav/_out/audit/stage1c_live_<i>.read.jsonl.
#
#   nohup bash stage0_results/stage1_campaign_driver.sh > hnav/_out/pipeline/stage1_campaign.log 2>&1 &
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
BASE_CFG="$MAB/configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml"
DS="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_64k.yaml"
PIPE="$REPO_ROOT/hnav/_out/pipeline"
CHAT="${HNAV_STAGE1_CHAT_URL:-http://localhost:8003/v1}"
mkdir -p "$PIPE"

log() { echo "[$(date '+%F %T')] $*"; }

# ── refusals: the campaign is one shot; every precondition is hard ───────────
cd "$REPO_ROOT"
git ls-files --error-unmatch stage0_results/stage1_preregistration.md >/dev/null 2>&1 \
    || { log "REFUSED: stage1_preregistration.md is not committed."; exit 1; }
# A committed file is not a REGISTERED one. The sh_64k pre-registration was
# WITHDRAWN on 2026-08-15 (mechanism refuted by the stratified finding;
# STAGE1_PLAN.md R2) and retained as evidence, so "it exists in git" no longer
# implies "it authorizes a campaign". A new pre-registration must be written
# after the oracle probe; until then this driver stays shut.
grep -q "WITHDRAWN" stage0_results/stage1_preregistration.md \
    && { log "REFUSED: the sh_64k pre-registration is WITHDRAWN (STAGE1_PLAN.md R2)."; \
         log "         Write a NEW pre-registration after the oracle probe."; exit 1; }
git ls-files --error-unmatch stage0_results/stage1_operating_point.json >/dev/null 2>&1 \
    || { log "REFUSED: stage1_operating_point.json is not committed."; exit 1; }
curl -sf "$CHAT/models" >/dev/null || { log "REFUSED: chat $CHAT not up."; exit 1; }
curl -sf "http://localhost:8001/v1/models" >/dev/null || { log "REFUSED: :8001 embed not up."; exit 1; }
grep -q "OPENAI_BASE_URL=http://localhost:8001/v1" "$MAB/.env" \
    || { log "REFUSED: $MAB/.env must route embeddings to :8001."; exit 1; }

# live-stack preflight: the adapter must actually assemble its read policy —
# a live arm that silently degraded to PASS would burn the campaign.
HNAV_MODE=live HNAV_NLI_DEVICE=cpu CUDA_VISIBLE_DEVICES= python - <<'PY' || { log "REFUSED: live read stack failed preflight."; exit 1; }
import sys
from hnav import config as _config
from hnav.adapters import mab_adapter as mab
cfg = _config.get_config(refresh=True)
assert cfg.mode == "live", cfg.mode
mab.reset_adapter()
a = mab.get_adapter()
assert a is not None, "adapter did not build"
assert a.read_policy is not None, "read_policy did not build (NLI? embedder?)"
assert a.embedder is not None, "endpoint embedder did not build"
print("live preflight OK: policy + embedder + signals assembled")
PY

run_one() {  # $1=arm(off|live) $2=tag
    local arm="$1" tag="$2"
    local out_rel="./outputs/hnav_stage1c_${tag}"
    local cfg="$PIPE/stage1c_${tag}.yaml"
    sed -e "s|^output_dir:.*|output_dir: $out_rel|" "$BASE_CFG" > "$cfg"
    rm -rf "$MAB/outputs/hnav_stage1c_${tag}"
    log "run $tag (HNAV_MODE=$arm) ..."
    (cd "$MAB" && CUDA_VISIBLE_DEVICES= HNAV_MODE="$arm" HNAV_NLI_DEVICE=cpu \
        HNAV_RUN_ID="stage1c_${tag}" HNAV_DOTENV_NO_OVERRIDE=1 \
        OPENAI_BASE_URL="$CHAT" OPENAI_API_KEY=EMPTY \
        python main.py --agent_config "$cfg" --dataset_config "$DS") \
        > "$PIPE/stage1c_${tag}.runlog" 2>&1
    local rc=$?
    log "run $tag exit=$rc"
    return $rc
}

run_one off warmup || { log "WARMUP FAILED — aborting"; exit 1; }
for i in $(seq 1 7); do
    run_one off  "off_$i"  || { log "off_$i FAILED — aborting";  exit 1; }
    run_one live "live_$i" || { log "live_$i FAILED — aborting"; exit 1; }
done
log "CAMPAIGN DONE: 7 off + 7 live (+1 discarded warmup). Analysis:"
log "  python stage0_results/stage1_campaign_analysis.py"
