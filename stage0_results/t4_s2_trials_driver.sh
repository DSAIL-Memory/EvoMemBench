#!/usr/bin/env bash
# T4/S2 pre-registered trials driver — see t4_s2_protocol.md Part 1.
# Runs FROM the repo root on the box. Assumes: hnav env active, :8000 healthy,
# :8001 fp32 embed server up, GPU never touched by the arms themselves.
#
#   bash stage0_results/t4_s2_trials_driver.sh
#
# Writes per-run results under MAB/outputs/hnav_s2trial_<arm>_<i>/ and a
# progress log to hnav/_out/pipeline/s2_trials.log (nohup-friendly).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"
BASE_CFG="$REPO_ROOT/hnav/_out/pipeline/t4_sh_6k_off.yaml"   # model+temp0 already set
DS="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml"
PIPE="$REPO_ROOT/hnav/_out/pipeline"

# Pre-registered execution order (O=off, S=shadow), warmup first (discarded):
ORDER=(W O S O S O S O S O S O O O O O)

log() { echo "[$(date '+%F %T')] $*"; }

run_one() {  # $1=arm(off|shadow) $2=tag
    local arm="$1" tag="$2"
    local out_rel="./outputs/hnav_s2trial_${tag}"
    local cfg="$PIPE/s2trial_${tag}.yaml"
    sed -e "s|^output_dir:.*|output_dir: $out_rel|" "$BASE_CFG" > "$cfg"
    rm -rf "$MAB/outputs/hnav_s2trial_${tag}"
    log "run $tag (HNAV_MODE=$arm) ..."
    (cd "$MAB" && CUDA_VISIBLE_DEVICES= HNAV_MODE="$arm" HNAV_DOTENV_NO_OVERRIDE=1 \
        OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
        python main.py --agent_config "$cfg" --dataset_config "$DS") \
        > "$PIPE/s2trial_${tag}.runlog" 2>&1
    local rc=$?
    log "run $tag exit=$rc"
    return $rc
}

no=0; ns=0
for slot in "${ORDER[@]}"; do
    case "$slot" in
        W) run_one off warmup || { log "WARMUP FAILED — aborting"; exit 1; } ;;
        O) no=$((no+1)); run_one off    "off_$no"    || { log "off_$no FAILED — aborting"; exit 1; } ;;
        S) ns=$((ns+1)); run_one shadow "sha_$ns"    || { log "sha_$ns FAILED — aborting"; exit 1; } ;;
    esac
done
log "ALL TRIALS DONE: $no off + $ns shadow (+1 discarded warmup)"
