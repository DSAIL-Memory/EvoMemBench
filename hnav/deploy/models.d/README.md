# `models.d/` — one frozen serving configuration per answering model

The multi-model campaign is a **controlled answering-model substitution**: the
memory store, retrieval, suppression plans, prompts, generation settings and
scoring are all frozen and LLM-independent, and the answering model is the only
experimental variable. These files are the *only* place a model-specific fact
is allowed to live, so the substitution stays auditable.

`run_multimodel_campaign.sh` runs them in filename order — smallest weights
first, so an OOM on the largest model costs nothing that already ran.

| # | key | weights | arch | env | why it is where it is |
| --- | --- | --- | --- | --- | --- |
| 01 | `phi4_mini` | 7.2 GB | `Phi3ForCausalLM` | legacy 0.9.1 | registered in the frozen env |
| 02 | `gemma3_4b` | 8.1 GB | `Gemma3ForConditionalGeneration` | legacy 0.9.1 | registered in the frozen env |
| 03 | `gemma4_e2b` | 9.6 GB | `Gemma4ForConditionalGeneration` | modern 0.28.0 | **not** registered in 0.9.1 |
| 04 | `qwen35_9b` | 19.3 GB | `Qwen3_5ForConditionalGeneration` | modern 0.28.0 | **not** registered in 0.9.1; memory-critical |

The env split was measured, not assumed — see `setup_vllm_modern.sh` and
`hnav/_out/campaign/vllm_modern.json`. The frozen `vllm_0.9.1` env is never
upgraded: it is where the reference `Qwen3-4B-Instruct-2507` result was
produced.

## The flags that must not vary, and why

Every model is served with the determinism-critical half of the frozen Stage-1
chat configuration (`serve_stage1_chat.sh`), because the campaign's A/A floor
(native vs `native_repeat`, which must be **0** discordant or the run is void)
depends on it:

    --max-num-seqs 1            no batch-composition dependence
    --no-enable-prefix-caching  the measured source of run-to-run coupling
    --enforce-eager             no cudagraph memory, no capture-shape effects
    temperature 0               set by the harness, not the server

Two things legitimately vary per model and are recorded in every artifact:

- **`MAX_MODEL_LEN`** — the benchmark's longest prompt is a fixed 169,810
  characters, but characters per token are a property of the tokenizer. The
  value here comes from `measure_prompt_tokens.py`, which counts that exact
  prompt under each model's own tokenizer *and* chat template.
- **`GPU_MEM_UTIL`** — the reference config reserved room on GPU1 for an
  embedding server. This campaign needs none: fact vectors are read from the
  committed disk cache (`detector_gap` installs a fail-on-miss embedder), so
  the whole card is available to the answering model.

`KV_CACHE_DTYPE` defaults to `fp8`, the reference precedent. If a model's
attention implementation refuses it, set it in that model's file — never let
the driver silently retry with different numerics, and never let two arms of
the same model run against different server configurations.

## Deliberately absent: tool-call parsers

`Conflict_Resolution` is plain question answering. The harness sends a system
message and a user message and never a `tools` array, so no tool-call parser is
enabled for any model — enabling one could only introduce a parse failure mode.
`preflight_model.py` records this as `tool_check: N/A` rather than leaving it
implicit.
