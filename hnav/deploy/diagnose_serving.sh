#!/usr/bin/env bash
# Which serving variable is destroying this model's answers?        [E2E-5]
#
# gemma-3-4b-it scored 6-13/100 in a measured campaign - 4/26 on the UNIQUE
# stratum (single-fact retrieval, no conflict, nothing for H-Nav to do) where
# Phi-4-mini scores 26/26 - with looping output: "United States of United
# States of United States of United". KV-cache dtype is the obvious suspect,
# but it is not the only one: Gemma-3's interleaved sliding-window/global
# attention has known bad interactions with attention backend, prefix caching
# and specific vLLM/Transformers versions, and Google's reference inference is
# BF16.
#
# So this varies more than one axis, cheaply, before any measured cell is spent.
#
#   bash hnav/deploy/diagnose_serving.sh <models.d/*.env> [variant ...]
#
# A variant is  env:kv  (optionally  env:kv:"extra flags")  e.g.
#   legacy:fp8      the configuration the void campaign actually ran
#   legacy:auto     same stack, BF16 cache        -> isolates KV dtype
#   modern:auto     newer vLLM, BF16 cache        -> isolates the vLLM version
#   modern:fp8      newer vLLM, fp8 cache         -> completes the 2x2
#
# Each variant costs ~11 completions (one short prompt + a ten-question
# unique-stratum probe). The 2x2 is ~44 calls; one measured subset is 500.
#
# Reads nothing, decides nothing, rewrites no config: it prints a table and the
# operator pins the winner in models.d/. Writes kv_probe_<variant>.json.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONF="${1:?usage: diagnose_serving.sh <models.d/*.env> [env:kv[:extra] ...]}"
shift
VARIANTS=("$@")
[ "${#VARIANTS[@]}" -gt 0 ] || VARIANTS=(legacy:fp8 legacy:auto modern:auto modern:fp8)

# shellcheck disable=SC1090
source "$CONF"
: "${MODEL_KEY:?}" "${MODEL_PATH:?}" "${SERVED_NAME:?}"

OUTDIR="hnav/_out/campaign/${MODEL_KEY}"
PORT="${HNAV_STAGE1_CHAT_PORT:-8003}"
mkdir -p "$OUTDIR"
source hnav/deploy/_activate.sh || exit 2

if curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "REFUSED: something is already serving :$PORT. Stop it first."
    exit 3
fi

# `grep -c .` on EMPTY input prints "0" and exits 1, so `|| echo 0` appends a
# second "0" and the variable becomes "0\n0" - never equal to "0", so an idle
# GPU reads as busy forever. This aborted the first 2x2 run after 7.5 minutes
# with "GPU1 never freed" while GPU1 was sitting at 15 MiB. `wc -l` always
# exits 0, and -eq compares numerically.
gpu1_procs() {
    nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sed '/^$/d' | wc -l
}
wait_gpu_free() {
    for _ in $(seq 1 90); do
        [ "$(gpu1_procs)" -eq 0 ] && return 0
        sleep 5
    done
    return 1
}

for V in "${VARIANTS[@]}"; do
    ENV_="${V%%:*}"; REST="${V#*:}"
    DT="${REST%%:*}"
    EXTRA=""
    [ "$REST" != "$DT" ] && EXTRA="${REST#*:}"
    SLUG=$(printf '%s' "$V" | tr -c 'A-Za-z0-9._-' '_')

    echo
    echo "================ variant: $V ================"
    wait_gpu_free || { echo "  GPU1 never freed; aborting"; exit 4; }

    export HNAV_FORCE_VLLM_ENV="$ENV_" HNAV_FORCE_KV_DTYPE="$DT"
    [ -n "$EXTRA" ] && export HNAV_FORCE_EXTRA_SERVE_FLAGS="$EXTRA"

    setsid bash hnav/deploy/serve_campaign_model.sh "$CONF" \
        > "$OUTDIR/probe_${SLUG}_server.log" 2>&1 < /dev/null &
    SRV=$!
    ok=0
    for _ in $(seq 1 150); do
        kill -0 "$SRV" 2>/dev/null || break
        curl -sf --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 \
            && { ok=1; break; }
        sleep 10
    done
    if [ "$ok" != 1 ]; then
        echo "  SERVER DID NOT START for $V - that is itself a result"
        tail -25 "$OUTDIR/probe_${SLUG}_server.log" | sed 's/^/    /'
    else
        grep -m1 -E '^   (context|extra)' "$OUTDIR/probe_${SLUG}_server.log" | sed 's/^/  /'
        HNAV_PREFLIGHT_ALLOW_LOW_ACCURACY=1 \
        python hnav/deploy/preflight_model.py --probe-only \
            --model-key "$MODEL_KEY" --model-path "$MODEL_PATH" \
            --served-name "$SERVED_NAME" \
            --out "$OUTDIR/kv_probe_${SLUG}.json" 2>&1 \
            | grep -E '\[(ok |FAIL)\]|PREFLIGHT' | sed 's/^/  /'
    fi
    unset HNAV_FORCE_VLLM_ENV HNAV_FORCE_KV_DTYPE HNAV_FORCE_EXTRA_SERVE_FLAGS
    kill -TERM -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
    kill -KILL -"$SRV" 2>/dev/null
done

wait_gpu_free
echo
echo "================ comparison ================"
python - "$OUTDIR" "${VARIANTS[@]}" <<'PY'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
print(f"  {'variant':22s} {'unique-stratum':16s} {'degenerate':11s} sample")
for v in sys.argv[2:]:
    slug = re.sub(r'[^A-Za-z0-9._-]', '_', v)
    p = out / f"kv_probe_{slug}.json"
    if not p.exists():
        print(f"  {v:22s} {'SERVER FAILED':16s}")
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    c = r["checks"]
    s = c.get("answer_sanity", {})
    deg = c.get("no_degenerate", {})
    ex = next((x for x in r.get("sanity_probe", [])), {})
    print(f"  {v:22s} {str(s.get('correct'))+'/'+str(s.get('n')):16s} "
          f"{str(not deg.get('ok', True)):11s} "
          f"{ex.get('out','')!r} vs {ex.get('truth','')!r}")
print("\n  A working model of this class scores 9-10/10 on the unique stratum.")
PY
