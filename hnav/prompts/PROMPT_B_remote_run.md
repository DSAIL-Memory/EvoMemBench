# PROMPT B — Remote run (run this on ozonderlab2, after transfer)

> Use only after Prompt A is finished and `pytest hnav/tests/ -q` is green.
> Paste everything below the line into a coding agent with SSH access to the GPU box,
> or run the steps yourself.

---

You are executing Stage 0 of an H-Nav port into EvoMemBench on a GPU machine.

**Read `HNAV_AGENT_BRIEF.md` first — it is the authoritative spec.** The code is already
written and unit-tested on CPU. Your job is to make it run here, in order, and to **stop
at every gate**.

## Machine

- 2× RTX 4090 (24 GB each).
- **GPU0 is occupied** by a persistent vLLM server running `Qwen3-4B-Instruct-2507` on
  `localhost:8000`. **Do not touch it, do not restart it, do not allocate on it.**
- **GPU1 is free.** Everything you run goes there via `HNAV_EMBED_DEVICE=1`.
- Full internet. Plain SSH + tmux; no scheduler.

## Sequence

Stop and report at anything marked **GATE**. Do not exercise judgment past a gate.

### 1. Setup
```bash
bash hnav/deploy/setup_remote.sh          # venv, torch+cu124, deps, nltk/tiktoken, weights (~8GB)
cp hnav/deploy/.env.template .env         # then confirm the values below
```
Confirm in `.env`: `HNAV_MODE=off`, `HNAV_EMBED_DEVICE=1`, `HNAV_EMBED_DTYPE=float32`,
`HNAV_LLM_MODEL` matching what `:8000` actually serves.

### 2. Pre-flight — **GATE**
```bash
source .venv-hnav/bin/activate
python hnav/deploy/check_env.py
```
Must exit 0. It verifies data files, reproduces the committed conflict census
(11,037 keys / 7,197 conflicted on `sh_262k`), checks CUDA and free VRAM on GPU1,
confirms the weights are cached, and pings `:8000`.

If it reports insufficient VRAM for float32, set `HNAV_EMBED_DTYPE=float16` **and keep
it pinned for every later task** — dtype drift changes cosines and moves thresholds.
Record the change.

### 3. Smoke test M1
```bash
python hnav/stage0/m1_geometry_calibration.py --subsets sh_6k --max-pairs 50
```
~2 minutes. Confirms the embedder loads on GPU1 and the pipeline runs end to end.

### 4. T1 — full M1 — **GATE (this is the kill switch)**
```bash
bash hnav/deploy/run_t1.sh
tail -f hnav/_out/m1.log
```
Exit code **0** = gate passed. Exit code **2** = `STOP S3` fired.

**The number that decides the thesis:** median `whole_blob_sim` over conflict pairs.

- **≥ 0.70** → the near-duplicate premise holds. Continue to step 5.
- **< 0.70** → conflict pairs are not near-duplicates in this embedding space. The
  geometry half of the work is dead. **STOP. Report. Do not run T3 onward.** Read-side
  stale repair may survive, but a human must re-scope first.

Report the full summary table either way.

### 5. T2 — grouping ablation
```bash
python hnav/stage0/m1b_grouping_ablation.py
```
Reuses T1's embedding cache; no new GPU compute. Reports precision/recall/F1 of the
geometry grouper against the regex (`parse()`) grouper.

**This decides attribution and must be reported prominently.** The benchmark hands you
templated facts *and* explicit serial numbers, so a critic will say any downstream gain
comes from the metadata rather than the geometry. High F1 → the detector is validated
and can be carried to CrossEp-Know, where no templates or serials exist. Low F1 → any
later gain must be reported as metadata-attributable.

### 6. T3 — replica fidelity — **GATE**
```bash
pytest hnav/tests/test_replica_fidelity.py -q
```
Needs ≥99.9% exact top-k identity vs the native retriever; `NumpyCosineReplica` must be
100% modulo documented `np.argsort` ties. Below that, `rank_self`, `margin`, `dH_self`,
`dH_neighbor` and `churn` are all invalid — stop and report which signals die.

### 7. T4 — shadow neutrality — **GATE**
```bash
bash hnav/deploy/serve_embeddings.sh      # in tmux; :8001, GPU1
pytest hnav/tests/test_shadow_neutrality.py hnav/tests/test_leakage_audit.py -q
```
`HNAV_MODE=off` vs `shadow` must produce **byte-identical** model outputs and identical
per-question scores at `temperature=0`. Any difference is a bug, not sampling noise.
Fix before proceeding — this is not a warning-level failure.

### 8. T5 → T7
```bash
python hnav/stage0/m2_retrieval_calibration.py
python hnav/stage0/m3_headroom.py
python hnav/stage0/m4_marginal_diff_test.py
```
T6 (`m3_headroom`) makes real LLM calls against `:8000` — the largest compute item, on
the order of a couple of thousand calls. Run it in tmux.

### 9. T8 — report — **HARD STOP**
```bash
python hnav/stage0/report.py            # writes STAGE0_REPORT.md
```
Then **stop**. Do not implement `write_policy.py` / `read_policy.py`. Do not run live
arms. A human evaluates the GO/NO_GO gate in `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` §4.

Tag every NO_GO with which of the three verdicts it is — **benchmark** (class absent),
**detection** (class present, signal fails), or **policy** (signal works, repair fails).
These must never be conflated; they are different scientific conclusions.

## Operational notes

- Everything long-running goes in `tmux` or under `nohup`. SSH will drop.
- Embeddings cache to `hnav/_cache/emb/`. First M1 run pays ~26k embeddings; every later
  task is free. **Do not delete this cache.**
- If you OOM on GPU1, lower `HNAV_EMBED_BATCH` before changing dtype — dtype is pinned.
- Never set `HNAV_MODE=live` during Stage 0.
- Commit results and push to `claude/evomembench-hnav-analysis-nfwl9z` as you go.
