#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# H-Nav Stage-0 unattended pipeline — T1 through T8 in one nohup'able run.
#
#   tmux new -s stage0
#   bash hnav/deploy/run_stage0.sh
#         — or —
#   nohup bash hnav/deploy/run_stage0.sh > hnav/_out/pipeline/console.log 2>&1 &
#
# Resume after an interruption: just run it again. Every stage records
# PASS/FAIL/SKIP/GATE in hnav/_out/pipeline/<stage>.status and completed stages
# are skipped on re-run. Force one stage over: --redo <stage>. Stages:
#   preflight t1_smoke t1 t2 m0 t4 m2 m3 m4 report
#
# ── GATE SEMANTICS, decided deliberately ────────────────────────────────────
# The brief says "stop at every [GATE] and report". Unattended, that means:
#
#   gate PASSES → continuing is exactly what the human would order; continue.
#   gate FIRES  → HALT THE PIPELINE IMMEDIATELY. Nothing downstream runs.
#                 The status file names the gate; a human decides what's next.
#
#   S3 (T1, exit 2)  geometry premise dead        → halt
#   S1 (M0, exit 2)  replica infidelity           → halt (rank signals invalid)
#   S2 (T4, diff)    shadow not byte-neutral      → halt (hard stop)
#   T8               always the end; the GO/NO_GO gate itself is HUMAN-ONLY.
#
# A stage that cannot run for environmental reasons (missing benchmark deps,
# vllm env absent) is SKIPped with the reason recorded, and the pipeline
# continues — report.py already renders missing measurements as NOT RUN, which
# is the designed behaviour, and M2–M4 use H-Nav's own offline replica, so they
# do not depend on M0/T4 having run. The final summary lists every SKIP.
#
# ── GPU choreography (no scheduler on this box) ─────────────────────────────
#   GPU0: the user's vLLM LLM server (:8000). NEVER touched.
#   GPU1: time-shared between two mutually exclusive tenants:
#     - in-process HFEmbedder      (t1_smoke t1 t2 m2 m3 m4)   ~17 GB
#     - vLLM embed server :8001    (m0 t4)                     0.85 × 24 GB
#   The pipeline starts the server only for m0/t4, kills it afterwards, and
#   WAITS for the VRAM to actually drain before the next in-process stage.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source hnav/deploy/_activate.sh || exit 1
# shellcheck source=/dev/null
source hnav/deploy/gpu_guard.sh

PIPE="$REPO_ROOT/hnav/_out/pipeline"
mkdir -p "$PIPE"

EMBED_DEV="${HNAV_EMBED_DEVICE:-1}"
EMBED_VLLM_PY="${HNAV_EMBED_VLLM_PY:-/mnt/nvmes/nvme1/egekutlu/programs/conda/vllm_0.9.1/bin/python3.11}"
EMBED_MODEL="${HNAV_EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
# The campaign's pinned embedding dtype (hnav/config.py: HNAV_EMBED_DTYPE,
# default float32 — the dtype T1 calibrated with). MUST match the served dtype:
# vLLM otherwise defaults to the checkpoint's bf16, whose quantization leaves
# the "normalized" served vectors at norms 1±2e-3, which breaks the unit-norm
# assumption FaissFlatReplica's dot-product ordering relies on and fired S1 on
# 2026-08-14 (m0 topk_agreement 0.24 on sh_32k, bank norms 0.9981–1.0022).
EMBED_DTYPE="${HNAV_EMBED_DTYPE:-float32}"
MAB="$REPO_ROOT/In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench"

REDO="${2:-}"
[ "${1:-}" = "--redo" ] || REDO=""

# ── one instance at a time ───────────────────────────────────────────────────
PIDFILE="$PIPE/pipeline.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Another pipeline instance is running (pid $(cat "$PIDFILE")). Refusing."
    exit 1
fi
echo $$ > "$PIDFILE"

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -f "$PIDFILE"
}
trap cleanup EXIT

log()  { echo "[$(date '+%F %T')] $*"; }
mark() { echo "$2 $(date '+%F %T') $3" > "$PIPE/$1.status"; }

done_already() {  # $1=stage → 0 if PASS/SKIP recorded and not being redone
    # --redo accepts a comma list: --redo m0,t4
    case ",$REDO," in *",$1,"*) return 1 ;; esac
    [ -f "$PIPE/$1.status" ] && grep -qE '^(PASS|SKIP)' "$PIPE/$1.status"
}

halt() {  # $1=stage $2=gate-name $3=message
    mark "$1" "GATE" "$2"
    log "════════════════════════════════════════════════════════════"
    log " $2 FIRED at stage '$1' — PIPELINE HALTED."
    log " $3"
    log " A human evaluates this. Nothing downstream has run."
    log " Full log: $PIPE/$1.log"
    log "════════════════════════════════════════════════════════════"
    summary
    exit 2
}

run_stage() {  # $1=stage $2...=command. PASS on 0, FAIL+halt on nonzero.
    local stage="$1"; shift
    if done_already "$stage"; then log "── $stage: already $(cut -d' ' -f1 "$PIPE/$stage.status"), skipping"; return 0; fi
    log "── $stage: starting  →  $PIPE/$stage.log"
    # capture rc directly: `if cmd; then...fi` with no else leaves $?=0 after fi
    "$@" > "$PIPE/$stage.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        mark "$stage" "PASS" "-"
        log "── $stage: PASS"
        return 0
    fi
    mark "$stage" "FAIL" "exit=$rc"
    log "── $stage: FAILED (exit $rc) — pipeline halted. See $PIPE/$stage.log"
    summary
    exit 1
}

wait_vram_free() {  # $1=device $2=need_gb — wait until that much is free
    local dev="$1" need="$2" i
    for i in $(seq 1 60); do
        local free_mib
        free_mib="$(nvidia-smi --id="$dev" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        [ -n "$free_mib" ] && [ "$(( free_mib / 1024 ))" -ge "$need" ] && return 0
        sleep 5
    done
    log "!! GPU$dev never freed ${need} GB (waited 5 min)."
    return 1
}

wait_http() {  # $1=url $2=timeout_s
    local url="$1" t="$2" i
    for i in $(seq 1 "$t"); do
        curl -sf "$url" -o /dev/null 2>/dev/null && return 0
        sleep 1
    done
    return 1
}

served_model() {  # prints the first model id :8000 serves, or nothing
    curl -sf http://localhost:8000/v1/models 2>/dev/null \
      | python -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null
}

summary() {
    {
        echo "# Stage-0 pipeline summary — $(date '+%F %T')"
        echo
        printf "%-10s %-6s %s\n" "stage" "state" "detail"
        for s in preflight t1_smoke t1 t2 m0 t4 m2 m3 m4 report; do
            if [ -f "$PIPE/$s.status" ]; then
                printf "%-10s %s\n" "$s" "$(cat "$PIPE/$s.status")"
            else
                printf "%-10s %s\n" "$s" "not-reached"
            fi
        done
        echo
        echo "Outputs: hnav/_out/*.json, STAGE0_REPORT.md (if report ran)"
        echo "hnav/_out/ is gitignored — copy off or commit deliberately."
    } | tee "$PIPE/SUMMARY.md"
}

# ═══ preflight ═══════════════════════════════════════════════════════════════
stage_preflight() {
    # sync HNAV_LLM_MODEL with reality before check_env judges it
    local sid
    sid="$(served_model || true)"
    if [ -n "$sid" ] && ! grep -q "^HNAV_LLM_MODEL=$sid$" .env 2>/dev/null; then
        log "syncing .env HNAV_LLM_MODEL -> '$sid' (what :8000 actually serves)"
        sed -i "s|^HNAV_LLM_MODEL=.*|HNAV_LLM_MODEL=$sid|" .env
    fi
    python hnav/deploy/check_env.py || return 1
    python -m pytest hnav/tests/ -q || return 1
}

# ═══ T1 ══════════════════════════════════════════════════════════════════════
stage_t1_smoke() {
    hnav_gpu_guard "$EMBED_DEV" 17 || return 1
    python hnav/stage0/m1_geometry_calibration.py --subsets sh_6k --max-pairs 50
}

stage_t1() {
    hnav_gpu_guard "$EMBED_DEV" 17 || return 1
    python hnav/stage0/m1_geometry_calibration.py
}

# ═══ embed server lifecycle (m0 + t4 only) ═══════════════════════════════════
embed_server_is_fp32_normalized() {
    # Probe :8001: fp32-normalized embeddings have ||v|| = 1 ± ~1e-7; a bf16
    # server leaves norms at 1 ± ~2e-3 (the S1 signature). Threshold 5e-5.
    HNAV_PROBE_MODEL="$EMBED_MODEL" python - <<'PYEOF'
import json, os, sys, urllib.request
import numpy as np
req = urllib.request.Request(
    "http://localhost:8001/v1/embeddings",
    data=json.dumps({"model": os.environ["HNAV_PROBE_MODEL"],
                     "input": ["dtype probe alpha", "dtype probe beta"],
                     "encoding_format": "float"}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=60))
except Exception as e:  # noqa: BLE001
    print(f"probe request failed: {e}"); sys.exit(2)
dev = max(abs(float(np.linalg.norm(np.asarray(r["embedding"], dtype=np.float64))) - 1.0)
          for r in d["data"])
print(f"probe max |norm-1| = {dev:.2e}")
sys.exit(0 if dev < 5e-5 else 1)
PYEOF
}

start_embed_server() {
    # a previous crashed run may have left a server up — reuse it, but only
    # after checking it serves in the calibrated dtype (a leftover bf16 server
    # would silently reintroduce the S1 failure this script guards against)
    if curl -sf http://localhost:8001/v1/models -o /dev/null 2>/dev/null; then
        if embed_server_is_fp32_normalized >> "$PIPE/embed_server.log" 2>&1; then
            log "embed server already answering on :8001 and unit-norm to fp32 precision — reusing it"
            SERVER_PID=""
            return 0
        fi
        log "server on :8001 fails the fp32-normalization probe (likely bf16) — killing it and starting fresh"
        SERVER_PID=""
        stop_embed_server
    fi
    [ -x "$EMBED_VLLM_PY" ] || { log "no vllm python at $EMBED_VLLM_PY"; return 1; }
    hnav_gpu_guard "$EMBED_DEV" 20 || return 1
    log "starting embed server :8001 on GPU$EMBED_DEV (vllm_0.9.1 env) ..."
    local task
    for task in embed embedding; do
        # --dtype: serve in the calibrated dtype (see EMBED_DTYPE above).
        # --max-model-len 16384: fp32 weights are ~15.1 GiB of the 20.4 GiB
        #   budget; profiling at the checkpoint's native 40960 risks OOM (vLLM
        #   warns exactly this). No MAB input comes near 16384 tokens (chunks
        #   are 4096), so truncation behaviour is unchanged.
        CUDA_VISIBLE_DEVICES="$EMBED_DEV" HF_HOME="$HF_HOME" \
        nohup "$EMBED_VLLM_PY" -m vllm.entrypoints.openai.api_server \
            --model "$EMBED_MODEL" --task "$task" --port 8001 \
            --dtype "$EMBED_DTYPE" --max-model-len 16384 \
            --gpu-memory-utilization 0.85 \
            --served-model-name "$EMBED_MODEL" \
            > "$PIPE/embed_server.log" 2>&1 &
        SERVER_PID=$!
        if wait_http "http://localhost:8001/v1/models" 600; then
            log "embed server up (pid $SERVER_PID, --task $task)"
            return 0
        fi
        kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
        log "--task $task did not come up; trying next flag"
    done
    log "embed server failed to start. Log: $PIPE/embed_server.log"
    return 1
}

stop_embed_server() {
    if [ -n "$SERVER_PID" ]; then
        log "stopping embed server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
    else
        # reused (or orphaned) server: find it by its command line on :8001
        local orphan
        orphan="$(pgrep -f 'vllm\.entrypoints\.openai\.api_server.*--port 8001' | head -1 || true)"
        if [ -n "$orphan" ]; then
            log "stopping reused/orphaned embed server (pid $orphan)"
            kill "$orphan" 2>/dev/null
        fi
    fi
    wait_vram_free "$EMBED_DEV" 17 || log "!! GPU$EMBED_DEV VRAM did not drain; the next stage's guard will catch it"
}

# ═══ M0 ══════════════════════════════════════════════════════════════════════
stage_m0() {
    # deps probe — m0 imports the benchmark's retriever (langchain, faiss, ...)
    if ! (cd "$MAB" && python -c "import sys; sys.path.insert(0,'.'); from methods.embedding_retriever import TextRetriever") \
            > "$PIPE/m0_deps_probe.log" 2>&1; then
        mark m0 "SKIP" "benchmark-deps-missing (see m0_deps_probe.log; pip install -r $MAB/requirements.txt)"
        log "── m0: SKIP — benchmark deps missing. M0 will render NOT RUN (designed behaviour)."
        return 3   # special: skip, not fail
    fi
    python hnav/stage0/m0_live_fidelity.py --subsets sh_6k sh_32k --max-pairs 50 || return $?
    python hnav/stage0/m0_live_fidelity.py
}

# ═══ T4 ══════════════════════════════════════════════════════════════════════
stage_t4() {
    if ! (cd "$MAB" && python -c "import sys; sys.path.insert(0,'.'); import main") \
            > "$PIPE/t4_deps_probe.log" 2>&1; then
        mark t4 "SKIP" "benchmark-deps-missing (see t4_deps_probe.log; pip install -r $MAB/requirements.txt)"
        log "── t4: SKIP — cannot import benchmark main. Run-level neutrality stays an open item."
        return 3
    fi
    local sid
    sid="$(served_model)"
    [ -n "$sid" ] || { log "t4: :8000 not answering"; return 1; }
    case "$sid" in *deepseek*) log "t4: served id '$sid' contains 'deepseek' — would route to Ark SDK. Cannot run."; return 1;; esac

    local base=configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml
    local subset rc=0
    for subset in sh_6k sh_32k; do
        local ds="configs/data_conf/Conflict_Resolution/Factconsolidation_${subset}.yaml"
        local off_dir="./outputs/hnav_t4_${subset}_off" sha_dir="./outputs/hnav_t4_${subset}_shadow"
        local off_cfg="$PIPE/t4_${subset}_off.yaml" sha_cfg="$PIPE/t4_${subset}_shadow.yaml"

        sed -e "s|^model:.*|model: $sid|" -e "s|^output_dir:.*|output_dir: $off_dir|" "$MAB/$base" > "$off_cfg"
        sed -e "s|^model:.*|model: $sid|" -e "s|^output_dir:.*|output_dir: $sha_dir|" "$MAB/$base" > "$sha_cfg"

        # Fresh arms EVERY run (identically for both — S2 compares the arms
        # against each other). main.py saves results incrementally and its
        # load_existing_results() RESUMES from a leftover results.json, so a
        # crashed earlier attempt would splice old-era entries (computed under
        # different server conditions, gold answers list->str mangled by the
        # resume path) into a fresh arm — seen 2026-08-14: the off arm carried
        # 4 zombie-era entries and diff_neutrality would have fired a FALSE S2.
        rm -rf "$MAB/$off_dir" "$MAB/$sha_dir"

        # HNAV_DOTENV_NO_OVERRIDE=1 (both arms, identically — S2 compares the
        # arms against each other): stops main.py's load_dotenv(override=True)
        # from letting MAB/.env's OPENAI_BASE_URL=:8001 (the *embedding*
        # endpoint) clobber the shell's :8000, which sent LLM chat completions
        # to the embed server (choices=None crash, 2026-08-14).
        log "t4/$subset: arm off ..."
        (cd "$MAB" && CUDA_VISIBLE_DEVICES="" HNAV_MODE=off HNAV_DOTENV_NO_OVERRIDE=1 \
            OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
            python main.py --agent_config "$off_cfg" --dataset_config "$ds") || return 1
        log "t4/$subset: arm shadow ..."
        (cd "$MAB" && CUDA_VISIBLE_DEVICES="" HNAV_MODE=shadow HNAV_DOTENV_NO_OVERRIDE=1 \
            OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
            python main.py --agent_config "$sha_cfg" --dataset_config "$ds") || return 1

        # main.py writes to <output_dir>/<DatasetName>/*_results.json — one
        # level below output_dir, not into it. Search both levels.
        local off_json sha_json
        off_json="$(ls -t "$MAB/$off_dir"/*.json "$MAB/$off_dir"/*/*.json 2>/dev/null | head -1)"
        sha_json="$(ls -t "$MAB/$sha_dir"/*.json "$MAB/$sha_dir"/*/*.json 2>/dev/null | head -1)"
        [ -n "$off_json" ] && [ -n "$sha_json" ] || { log "t4/$subset: result JSONs not found"; return 1; }

        log "t4/$subset: comparing $off_json vs $sha_json"
        python hnav/deploy/diff_neutrality.py "$off_json" "$sha_json" || rc=$?
        [ "$rc" -ne 0 ] && return 42   # S2 marker
    done
    return 0
}

# ═══ M2 — plus the fallback-chunker tripwire ═════════════════════════════════
stage_m2() {
    hnav_gpu_guard "$EMBED_DEV" 17 || return 1
    python hnav/stage0/m2_retrieval_calibration.py || return 1
    python - <<'PYEOF'
import json, sys
rows = json.load(open("hnav/_out/m2_retrieval_calibration.json", encoding="utf-8"))
bad = [r["subset"] for r in rows if r.get("fallback_chunker")]
if bad:
    print(f"fallback_chunker=true on {bad}: nltk/punkt is missing — the chunking "
          f"is NOT the benchmark's and every read-side number is over wrong units.")
    sys.exit(1)
print("fallback_chunker=false everywhere — chunking is the benchmark's own.")
PYEOF
}

stage_m3() {
    hnav_gpu_guard "$EMBED_DEV" 17 || return 1
    curl -sf http://localhost:8000/v1/models -o /dev/null || { log "m3: LLM :8000 not answering"; return 1; }
    python hnav/stage0/m3_headroom.py
}

stage_m4() { python hnav/stage0/m4_marginal_diff_test.py; }

stage_report() {
    python hnav/stage0/report.py || return 1
    if python hnav/stage0/report.py --strict > "$PIPE/report_strict.log" 2>&1; then
        log "report --strict: COMPLETE — every measurement present."
    else
        log "report --strict: INCOMPLETE — missing measurements listed in $PIPE/report_strict.log"
        log "The GO/NO_GO gate must NOT be evaluated on a partial report."
    fi
}

# ═══════════════════════════════ main ════════════════════════════════════════
log "Stage-0 pipeline starting (pid $$). Status dir: $PIPE"
hnav_gpu_report

run_stage preflight stage_preflight
run_stage t1_smoke  stage_t1_smoke

# T1 full — gate S3 needs its exit code inspected, so not via run_stage
if done_already t1; then
    log "── t1: already done, skipping"
else
    log "── t1: starting (the kill switch)  →  $PIPE/t1.log"
    stage_t1 > "$PIPE/t1.log" 2>&1
    rc=$?
    if [ $rc -eq 2 ]; then
        halt t1 "GATE S3" "median whole_blob_sim < 0.70: conflict pairs are not near-duplicates; the geometry half is dead on this benchmark."
    elif [ $rc -ne 0 ]; then
        mark t1 "FAIL" "exit=$rc"; log "── t1: FAILED (exit $rc). See $PIPE/t1.log"; summary; exit 1
    fi
    mark t1 "PASS" "-"
    log "── t1: PASS — S3 did not fire. Geometry premise holds."
fi

run_stage t2 python hnav/stage0/m1b_grouping_ablation.py

# ── m0 + t4 share the embed-server window ────────────────────────────────────
need_server=0
done_already m0 || need_server=1
done_already t4 || need_server=1

if [ "$need_server" -eq 1 ]; then
    if start_embed_server; then
        if ! done_already m0; then
            log "── m0: starting  →  $PIPE/m0.log"
            stage_m0 >> "$PIPE/m0.log" 2>&1
            rc=$?
            if [ $rc -eq 2 ]; then
                stop_embed_server
                halt m0 "GATE S1" "live-index top-k agreement < 0.999: rank_self, margin, dH_*, churn are INVALID."
            elif [ $rc -eq 3 ]; then
                :  # SKIP already marked inside
            elif [ $rc -ne 0 ]; then
                mark m0 "FAIL" "exit=$rc"; log "── m0: FAILED. See $PIPE/m0.log"; stop_embed_server; summary; exit 1
            else
                mark m0 "PASS" "-"; log "── m0: PASS — replica is faithful on the live index."
            fi
        fi
        if ! done_already t4; then
            log "── t4: starting  →  $PIPE/t4.log"
            stage_t4 >> "$PIPE/t4.log" 2>&1
            rc=$?
            if [ $rc -eq 42 ]; then
                stop_embed_server
                halt t4 "GATE S2" "off vs shadow arms differ at temperature=0. Shadow instrumentation is not neutral."
            elif [ $rc -eq 3 ]; then
                :
            elif [ $rc -ne 0 ]; then
                mark t4 "FAIL" "exit=$rc"; log "── t4: FAILED. See $PIPE/t4.log"; stop_embed_server; summary; exit 1
            else
                mark t4 "PASS" "-"; log "── t4: PASS — shadow is byte-neutral on sh_6k and sh_32k."
            fi
        fi
        stop_embed_server
    else
        for s in m0 t4; do
            done_already "$s" || { mark "$s" "SKIP" "embed-server-unavailable (see $PIPE/embed_server.log)"; log "── $s: SKIP — no embed server."; }
        done
    fi
fi

run_stage m2 stage_m2
run_stage m3 stage_m3
run_stage m4 stage_m4
run_stage report stage_report

log "════════════════════════════════════════════════════════════"
log " Pipeline COMPLETE. This is the T8 HARD STOP."
log " STAGE0_REPORT.md is written. The GO/NO_GO gate is evaluated"
log " by a HUMAN against EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md §4."
log " Do not implement write/read policies. Do not set HNAV_MODE=live."
log "════════════════════════════════════════════════════════════"
summary
