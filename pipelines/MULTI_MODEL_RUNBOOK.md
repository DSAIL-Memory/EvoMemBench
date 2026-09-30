# Multi-model campaign runbook

Everything needed to run the frozen H-Nav arms against a **new answering
model**. Only the answering LLM varies; the embedder, thresholds, artifacts
and prepasses are frozen and LLM-independent.

> **Read this first — the campaign of 2026-08-30/31 found that four serving
> facts must be MEASURED per model, not inherited from the reference config.**
> Inheriting one of them (`--kv-cache-dtype fp8`) silently destroyed a model's
> entire run: gemma-3-4b scored 13/100 on sh_6k where the same weights, prompts
> and suppression plan score 89/100 under BF16. It passed every preflight check
> in force at the time, because none of them checked whether the answers were
> *right*.
>
> | must be measured per model | how | tool |
> | --- | --- | --- |
> | context window | tokenize the real longest prompt with the model's own tokenizer **and chat template** | `measure_prompt_tokens.py` |
> | which vLLM can load it | architecture registry check, not assumption | `setup_vllm_modern.sh` |
> | thinking mode | 10 output tokens is the whole budget; a thinking model scores ~0 | `preflight_model.py` |
> | **KV-cache dtype** | serve twice, vary only the dtype, probe | `diagnose_serving.sh` |
>
> Run `diagnose_serving.sh <models.d/*.env>` **before** any measured cell for a
> new model. ~11 completions per variant against 1,500 for one wasted arm.

## Per-model serving configurations actually used

| model | KV | vLLM | attention | extra |
| --- | --- | --- | --- | --- |
| Qwen3-4B (reference) | fp8 | 0.9.1 | default | frozen substrate, 65536 ctx |
| Phi-4-mini-instruct | fp8 | 0.9.1 | default | 49152 ctx |
| gemma-3-4b-it | **BF16** | 0.9.1 | default | fp8 measured at 0/10 vs 9/10 |
| gemma-4-E2B-it | **BF16** | 0.28.0 | TRITON_ATTN | `--language-model-only` |
| Qwen3.5-9B | **BF16** | 0.28.0 | TRITON_ATTN | `--language-model-only`, thinking off |

The dtype varies by model **by necessity**, and every within-model comparison
— which is where the campaign's endpoints live — holds it fixed across all
three arms. Record it per model; do not homogenise it back.

On a box with **no CUDA toolkit**, vLLM 0.28 JIT-compiles FlashInfer for both
attention *and* sampling. `serve_campaign_model.sh` sets
`VLLM_USE_FLASHINFER_SAMPLER=0` (numerically irrelevant at temperature 0, where
sampling is argmax); the attention backend is set per model because it is *not*
numerically irrelevant.

## The three arms to run per model

| arm | identity screen | semantic gate | why it is in the campaign |
| --- | --- | --- | --- |
| **`hnav_raw`** | parser same-key | NLI ≥ 0.90 | the shipped reference; the +19 result to replicate |
| **`hnav_idonly`** | parser same-key | **waived** | tests whether the semantic gate was the binding constraint; zero harm by construction |
| **`hnav_geo`** | geometry only | NLI ≥ 0.90 | the parser-free contrast — **report its void condition 4**, see below |

`hnav_abtt` is answer-identical to `hnav_raw` on the committed model (0
disagreements) and can be skipped unless a per-model geometry check is
wanted. `hnav_ces` and `hnav_fusion` are closed (superseded / failed gate).

> **Void-condition warning — now confirmed on five models.** `hnav_geo`
> produced **exactly 8** harmful suppressions and **524** superseded ones on
> sh_64k for *every* model tested (Qwen3-4B, Phi-4-mini, gemma-3-4b,
> gemma-4-E2B, Qwen3.5-9B), byte-identical. `hnav_abtt_noparser` gave 5 on the
> reference model. This is structural: suppression plans are computed from the
> retrieved page and the screen with **no LLM involved**, so harm cannot vary
> with the answering model, and only a `same_key`-based screen has zero harm by
> construction. Running `hnav_geo` on a further model cannot produce a new harm
> count — one demonstration is mathematically sufficient. Report the void;
> never quote the accuracy alone. `pipelines/_shared/runner.py` fails a subset
> on any run-voiding condition, so new runs surface it.

## Preconditions (verified 2026-08-30)

- All 7 arms' operating points match their pinned sha256.
  `pytest hnav/tests/test_pipelines.py` now asserts this for **every** arm
  (it previously covered only `hnav_raw`/`hnav_abtt`, and the extension
  immediately caught two arms whose worktree bytes had drifted to CRLF while
  `git diff` stayed empty — `detector_gap.freeze` now writes LF explicitly).
- **Answering-model context window ≥ ~52k tokens — not 48k.** The earlier
  "≈42.5k tokens" here was a chars/4 estimate, and it was wrong.
  `hnav/deploy/measure_prompt_tokens.py` counts the real longest sh_64k prompt
  (169,810 chars) under each model's own tokenizer *and* chat template:

  | model | tokens | chars/token |
  | --- | --- | --- |
  | Phi-4-mini-instruct | 43,321 | 3.92 |
  | gemma-3-4b-it | 48,224 | 3.52 |
  | gemma-4-E2B-it | 48,228 | 3.52 |
  | Qwen3.5-9B | 49,195 | 3.45 |
  | **Qwen3-4B-Instruct-2507 (reference)** | **49,623** | 3.42 |

  Three of the four new models exceed 48k, and so does the reference model —
  which is why the frozen substrate serves at 65536. A server sized from the
  old estimate would have run for hours and then died on the longest question.
  Per-model values live in `hnav/deploy/models.d/*.env`; set
  `HNAV_STAGE1_MAX_MODEL_LEN` only for the frozen `serve_stage1_chat.sh` path.
- **Serve the new model under its own name**: `HNAV_STAGE1_SERVED_NAME`
  (and `HNAV_STAGE1_MODEL` for the weights path). Without it vLLM advertises
  the reference model's name and every artifact mislabels the run.
- **Void condition 2 (`native_in_band`) is model-specific.** Its band
  (0.30, 0.50) was fixed from m3's sh_64k measurement for Qwen3-4B; a
  stronger model, or any model on sh_32k (native 0.53 today), leaves it
  without anything being wrong. The runner reports it as a **WARNING**, not a
  validity failure, and the band must be re-preregistered per model. Every
  other run-voiding condition still fails the subset.
- Prepasses required, **per subset**, already built on the box and reusable
  for every model:
  - `hnav_raw`, `hnav_idonly`, `hnav_geo` → `stage1_prepass_<subset>_benchmarkpage.json`
  - `hnav_abtt`, `hnav_abtt_noparser` → `…_benchmarkpage_abtt.json`
  - `hnav_ces`, `hnav_fusion` → `…_benchmarkpage_ces.json`
- Subsets are fixed at `sh_6k, sh_32k, sh_64k` (`ALLOWED_SUBSETS`; sh_262k
  permanently excluded).

## Per-model sequence

```bash
# ── on the box, repo root ───────────────────────────────────────────────
git pull
source hnav/deploy/_activate.sh

# 1. serve the answering model on GPU1 (frozen flags for Qwen3-4B; adapt
#    ONLY the model path/name and, if the model needs it, --max-model-len)
HNAV_STAGE1_MODEL=/path/to/new-model-weights HNAV_STAGE1_SERVED_NAME=<org>/<new-model-name> nohup bash hnav/deploy/serve_stage1_chat.sh > hnav/_out/pipeline/chat.log 2>&1 &
until curl -sf http://localhost:8003/v1/models >/dev/null; do sleep 10; done

# 2. pre-flight per arm — sends nothing, prints the budget and guards
python pipelines/hnav_raw/run.py     --llm-model <served-name> --dry-run
python pipelines/hnav_idonly/run.py  --llm-model <served-name> --dry-run
python pipelines/hnav_geo/run.py     --llm-model <served-name> --dry-run

# 3. one shot per arm (each writes results/<model-tag>_<date>/)
python pipelines/hnav_raw/run.py    --llm-model <served-name> --llm-base-url http://localhost:8003/v1
python pipelines/hnav_idonly/run.py --llm-model <served-name> --llm-base-url http://localhost:8003/v1
python pipelines/hnav_geo/run.py    --llm-model <served-name> --llm-base-url http://localhost:8003/v1
```

Long runs go under `nohup` via a committed driver script — never an inline
compound ssh command (`hnav/deploy/run_geo_arm.sh`, `run_idonly_arm.sh` are
the templates; copy one per arm).

## Cost per model

Each arm runs 5 internal arms (native, native_repeat, detector_suppress,
detector_demote_late, detector_anti) × 100 questions × 3 subsets =
**1,500 chat completions per pipeline arm**, so ~4,500 for the three-arm
set. Prompt sizes are the benchmark's own RAG prompts (~53k tokens on
sh_64k); the detector arm's prompt is *shorter* than native
(−0.31 % sh_64k, −2.87 % sh_6k). **No prepass rebuild, no embedding
recompute, no extra generative call.**

## Known sharp edges

- `--dry-run` still writes `results/<tag>/run_manifest.json`; use a throwaway
  `--tag` for pre-flights so a dry folder never blocks (or gets committed
  alongside) the real one.
- `--smoke-llm` through the runner writes production filenames into
  `results/<tag>/`; use a distinct `--tag` for smoke runs.
- `hnav_fusion` is NOT runnable — its frozen point is the vacuous zero-recall
  cell, so the positive-control guard aborts before the dry-run returns.
  Excluded by design.
- `hnav_raw`/`hnav_abtt` have no `results/` folder for the reference model
  (their numbers live in the older `stage0_results/` artifact format), so the
  first new model must run `hnav_raw` itself to get a comparable baseline.
- `e2e3_analysis.py`'s comparison baselines point at the Qwen3-4B artifacts;
  for a different model, compare that model's own arms to each other.

## After the runs

```bash
# fetch into the repo and analyse (locally)
scp -r egekutlu@<box>:.../pipelines/<arm>/results/<tag> pipelines/<arm>/results/
python -m hnav.geometry_filter.e2e3_analysis pipelines/hnav_geo/results/<tag>
```

Then, for the campaign-level claims, check both:

1. **Replication** — does the governance gain hold per model (native vs arm,
   McNemar exact, per stratum)?
2. **Structural vs interaction** — suppression plans are LLM-independent, so
   the parser-only question set should be *identical* across models if the
   parser's edge is structural. The committed sh_64k reference set is
   **{9, 11, 26, 35, 54, 69, 70, 91, 93}**. Same set ⇒ structural; scattered
   ⇒ model interaction. Either result is publishable.

## Discipline (unchanged)

One shot per model per subset; a void is reported, not re-rolled; the runner
refuses to overwrite an existing `results/<tag>/`. Never tune on sh_64k.
A new embedder invalidates every threshold — that is a new calibration
campaign, not a runner flag.
