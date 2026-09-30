# T4 / S2 Adjudication Protocol — PRE-REGISTERED

> Registered by commit BEFORE any trial data collection (see git history: this
> file lands before any `t4_s2_trials_*` result is committed).
> Date: 2026-08-14. Author: auditor agent, executing the user's S2 decision
> (relayed 2026-08-14 afternoon): repeated trials to quantify the noise floor,
> pre-registered equivalence analysis (supporting evidence only), and a
> deterministic rerun as the definitive verdict.

## Background (already collected, NOT part of the trials)

GATE S2 fired 2026-08-14 12:36:02 on `sh_6k` with two clean arms
(off vs shadow: 2/100 outputs differ, `input_len` identical 100/100).
A 2-run A/A pilot (off vs off, no H-Nav code) differed at 5/100 outputs with
4 correctness flips (`stage0_results/t4_s2_evidence/`). The pilot suggests the
`:8000` vLLM substrate (0.9.1, continuous batching + prefix caching) is not
run-to-run deterministic at temperature=0. The pilot is evidence for the
hypothesis; the trials below quantify it properly.

## Part 1 — Repeated trials on :8000 (noise floor + equivalence)

### Design (fixed before running)
- Subset: `factconsolidation_sh_6k` only — the subset on which S2 fired.
  (`sh_32k` is separately blocked by a context-length risk near the server's
  `--max-model-len 32000`; documented as an open user question.)
- Server: the user's `:8000` (untouched), idle — trials run BEFORE m3.
  Embeddings via `:8001` fp32 (the pinned campaign dtype), started fresh.
- 1 warm-up off-run, DISCARDED (prefix-cache cold/warm asymmetry).
- Then 15 counted runs in this exact interleaved order (balances cache/state
  drift across groups):
  `O S O S O S O S O S O O O O O`  → N_off = 10, N_shadow = 5.
- Each run: fresh output dir (`rm -rf` first), `HNAV_MODE=off|shadow`,
  `HNAV_DOTENV_NO_OVERRIDE=1`, `OPENAI_BASE_URL=http://localhost:8000/v1`,
  `CUDA_VISIBLE_DEVICES=` (arms never touch a GPU directly), temperature 0.
  Identical agent config except `output_dir`.
- Driver: `stage0_results/t4_s2_trials_driver.sh` (committed with this file).

### Metrics
- Primary: per-run `substring_exact_match` (%) over the 100 questions
  (the benchmark's own deterministic evaluator).
- Secondary: pairwise output-mismatch rate = |{i : output_A[i] ≠ output_B[i]}| / 100.

### Pre-registered analyses (`stage0_results/t4_s2_trials_analysis.py`)
1. **Noise floor**: distribution (mean, max) of within-off pairwise mismatch
   rates (45 pairs); SD of run-level `substring_exact_match` within off runs.
2. **Equivalence (TOST)** on run-level `substring_exact_match`, off (n=10) vs
   shadow (n=5): two one-sided Welch t-tests against margin **δ = ±2.0
   percentage points**, α = 0.05. Equivalence declared iff BOTH p < 0.05.
   Margin rationale: the A/A pilot swung 4.0 points between two identical
   runs; δ = half that pilot swing. Limitation acknowledged: n = 10 vs 5
   gives limited power; a TOST failure from wide CIs alone is reported as
   "underpowered", not as evidence of an effect.
3. **Shadow effect on outputs**: Δ = mean(cross-group mismatch rate, 50
   pairs) − mean(within-group mismatch rate, 45 off/off + 10 shadow/shadow
   pairs). Two-sided permutation test (10,000 random relabelings of the 15
   runs, statistic recomputed each time), α = 0.05.

### Pre-registered decision rule
- These trials are **SUPPORTING EVIDENCE ONLY — never a PASS**.
  Non-significance alone is NOT a PASS.
- Supports substrate-noise attribution iff: TOST equivalence established AND
  permutation p ≥ 0.05 AND |Δ| does not exceed the within-group mean.
- Evidence AGAINST neutrality iff: permutation p < 0.05, or the off−shadow
  accuracy point difference exceeds δ with TOST failing. In that case: report;
  no fix; the deterministic rerun still runs and is still definitive.

## Part 2 — Definitive verdict: deterministic rerun on :8002

### Setup (GPU1 only; the user's :8000 / PIDs 52259/52520 are never touched)
- Chat server `:8002`: vLLM 0.9.1, model
  `/mnt/nvmes/nvme1/egekutlu/models/Qwen3-4B-Instruct-2507` (local weights,
  same checkpoint the user serves), `--served-model-name
  Qwen/Qwen3-4B-Instruct-2507`, `--enforce-eager`, `--max-num-seqs 1`,
  `--no-enable-prefix-caching`, `--max-model-len 16384`, bf16 (default),
  `--gpu-memory-utilization 0.55`.
- Embeddings `:8001` restarted **bf16**, `--gpu-memory-utilization 0.33`,
  `--max-model-len 16384` — DOCUMENTED DEVIATION for this test only
  (fp32 embed + chat do not fit GPU1 together; the S2 verdict needs the two
  arms to see *identical, deterministic* retrieval, not the campaign dtype;
  user authorized "document whatever embed config you use").
  The determinism proof below covers the full retrieval+generation path.
- Both arms identical env: `OPENAI_BASE_URL=http://localhost:8002/v1` (shell),
  embeddings at `:8001` via `MAB/.env` + `HNAV_DOTENV_NO_OVERRIDE=1`.

### Procedure
1. **Determinism proof (A/A)**: two consecutive `HNAV_MODE=off` runs on
   `:8002`. Required: all 100 `output` fields byte-identical AND
   `diff_neutrality.py` exit 0. If not identical: one pre-registered fallback
   (`VLLM_USE_V1=0` on the chat server) is tried once; if still not identical,
   verdict = CANNOT ADJUDICATE (no deterministic substrate available); report.
2. **The verdict pair**: one `off` + one `shadow` run on `:8002`,
   `diff_neutrality.py`:
   - exit 0 → **S2 PASS (definitive)**; the `:8000` noise is reported as a
     substrate finding.
   - exit 1 → **genuine S2 failure** → STOP, no fix, report.

## Scope and honesty notes
- `sh_6k` only; `sh_32k` S2 status remains OPEN (context-length +
  GPU1-memory constraints), listed for the user.
- Phase ordering deviation from the coordinator's schedule, with reasons:
  m2 and the trials could NOT run concurrently (the trials' retrieval needs
  the `:8001` embed server on GPU1; m2 needs the same GPU for the in-process
  embedder), and the `:8002` phase runs BEFORE m3 (m3 holds GPU1's 17 GB
  embedder for its whole duration). Order actually executed:
  m2 → trials (:8000) → :8002 verdict → m3 launch.
