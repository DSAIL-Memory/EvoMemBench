#!/usr/bin/env bash
# Realized-coverage shadow check — ONE sh_32k run, HNAV_MODE=shadow.  [T11]
#
# Purpose: the calibration harness ranks RAW chunks; the benchmark stores
# MEMORIZE-TEMPLATED chunks (wrapper + timestamp). This run pushes the frozen
# operating point through the REAL stack (benchmark process, :8001 bf16
# retrieval, endpoint-embedder fact vectors, CPU NLI) on a CALIBRATION subset
# and records what the gate actually does — decisions logged, nothing applied.
# It is the rehearsal for every failure mode that would VOID the one-shot
# campaign: stack must assemble, audit must land, embed-cache misses must be 0,
# realized RERANK coverage must be in the neighbourhood of the calibration
# expectation (stage0_results/stage1_operating_point.json, by_subset sh_32k).
#
#   bash stage0_results/stage1_shadow_check.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
BASE_CFG="$MAB/configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml"
DS="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml"
PIPE="$REPO_ROOT/hnav/_out/pipeline"
CHAT="${HNAV_STAGE1_CHAT_URL:-http://localhost:8003/v1}"
mkdir -p "$PIPE"

log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO_ROOT"
git ls-files --error-unmatch stage0_results/stage1_operating_point.json >/dev/null 2>&1 \
    || { log "REFUSED: freeze the operating point first."; exit 1; }
curl -sf "$CHAT/models" >/dev/null || { log "REFUSED: chat $CHAT not up."; exit 1; }
curl -sf "http://localhost:8001/v1/models" >/dev/null || { log "REFUSED: :8001 embed not up."; exit 1; }

rm -f "$REPO_ROOT/hnav/_out/audit/stage1_shadowcheck.read.jsonl"
cfg="$PIPE/stage1_shadowcheck.yaml"
sed -e "s|^output_dir:.*|output_dir: ./outputs/hnav_stage1_shadowcheck|" "$BASE_CFG" > "$cfg"
rm -rf "$MAB/outputs/hnav_stage1_shadowcheck"

log "shadow coverage-check run (sh_32k, HNAV_MODE=shadow) ..."
(cd "$MAB" && CUDA_VISIBLE_DEVICES= HNAV_MODE=shadow HNAV_NLI_DEVICE=cpu \
    HNAV_RUN_ID=stage1_shadowcheck HNAV_DOTENV_NO_OVERRIDE=1 \
    OPENAI_BASE_URL="$CHAT" OPENAI_API_KEY=EMPTY \
    python main.py --agent_config "$cfg" --dataset_config "$DS") \
    > "$PIPE/stage1_shadowcheck.runlog" 2>&1
rc=$?
log "run exit=$rc"

log "audit digest:"
python - <<'PY'
import json
from pathlib import Path
p = Path("hnav/_out/audit/stage1_shadowcheck.read.jsonl")
if not p.exists():
    raise SystemExit("VOID-SHAPED: no audit file was written")
n = rerank = misses = errors = 0
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r.get("schema") != "hnav.read.v1":
        continue
    n += 1
    if r.get("hnav_action") == "RERANK":
        rerank += 1
    ec = (r.get("hnav_reasons") or {}).get("embed_cache") or {}
    misses = max(misses, int(ec.get("misses") or 0))
    errors = max(errors, int(ec.get("embed_errors") or 0))
op = json.loads(Path("stage0_results/stage1_operating_point.json").read_text())
expect = op["by_subset"].get("sh_32k", {}).get("changed")
print(f"reads={n} rerank_decisions={rerank} embed_cache_misses={misses} "
      f"embed_errors={errors}")
print(f"calibration expectation (sh_32k changed): {expect}/100")
print("shadow applied nothing by construction; compare rerank_decisions to "
      "the expectation and treat misses>0 or reads!=100 as campaign blockers.")
PY
