# Multi-model campaign — the plan, and what it is actually testing

Launched 2026-08-30 13:43 UTC, git `d34c10f`, branch
`claude/hnav-presentation-evidence`.
**Completed 2026-08-31 12:55 UTC.** Results: `MULTIMODEL_SUMMARY.md`
(generated from the artifacts, never transcribed).

## Outcome, against what this document predicted before the run

| prediction made here, before any model ran | outcome |
| --- | --- |
| `hnav_geo`'s 8 harmful suppressions "will recur identically on every model" | **Confirmed 5/5.** `n_suppressed_harmful = 8` and `n_suppressed_superseded = 524`, byte-identical on Qwen3-4B, Phi-4-mini, gemma-3-4b, gemma-4-E2B, Qwen3.5-9B. VOID by condition 4 every time. |
| the parser-only question set is identical across models if the edge is structural | **Confirmed, and it decomposes.** The set of sh_64k questions where `idonly` and `raw` prompts differ is byte-identical across models — 17 questions. Which of the 17 a model *converts* is model-dependent (0–2). |
| governance gain replicates per model | **Confirmed.** 15 of 15 model-arm cells positive. |

Two claims this document did **not** make, and one it should have avoided:

- The campaign was framed around a capability ladder with the expectation
  that weaker models benefit most. **That is false.** So is its opposite.
  See `TEZ_HIKAYESI.md` §4: the gain is `final − native` and native varies
  for its own reasons; what is monotone in model capability, at every
  context length, is the **post-H-Nav accuracy**, not the gain.
- Four serving facts had to be measured per model rather than inherited.
  The context window, the vLLM/architecture support matrix, thinking mode,
  and **the KV-cache dtype** — the last of which destroyed gemma-3-4b's
  first run (13/100 vs 89/100 on sh_6k, same weights and prompts).

## A comparability defect this campaign exposed in the older artifacts

The reference model's `hnav_raw` numbers for sh_6k and sh_32k (94 and 86,
quoted in `TEZ_HIKAYESI.md` before this campaign) came from runs with
`page_source=None` — the prepass page. Every `pipelines/` arm and every new
model uses `page_source=benchmark`. The native rows are the tell: 28 and 48
there, against 30 and 53 for the same model at benchmark page. Each delta was
internally valid; the *table* mixed two configurations across its columns.

sh_64k was never affected — the confirmatory run already used benchmark page,
which is why its native (45) agrees across both formats.
`hnav_raw × Qwen3-4B × {sh_6k, sh_32k}` at `page_source=benchmark` was then
measured as a new cell (94 and 83; note sh_32k differs from the old 86), and
all 50 cells in `MULTIMODEL_SUMMARY.md` are now one configuration.
`multimodel_summary.py` prints `page_source` per cell and warns on any mix,
so this cannot recur silently.

## The experiment

A **controlled answering-model substitution**. The memory store, retrieval
prepasses, embeddings, suppression plans, benchmark questions, prompts,
generation settings and scoring are all frozen and LLM-independent. The
answering model is the only variable.

Three arms per model, the same three the reference model ran:

| arm | identity screen | semantic gate | what it isolates |
| --- | --- | --- | --- |
| `hnav_raw` | parser `same_key` | NLI ≥ 0.90 | the shipped reference configuration |
| `hnav_idonly` | parser `same_key` | **waived** (0.0) | whether the semantic gate, not identity, was the binding constraint |
| `hnav_geo` | geometry only | NLI ≥ 0.90 | the fully parser-free contrast |

Each arm is 5 internal arms (`native`, `native_repeat`, `detector_suppress`,
`detector_demote_late`, `detector_anti`) × 100 questions × 3 subsets =
**1,500 completions**; 4,500 per model, ~18,000 for the campaign.

The reference model `Qwen3-4B-Instruct-2507` is **not** re-run. Its numbers are
committed; a second shot at a measured cell is a re-roll, not a replication.

## What the campaign can and cannot decide

Two claims are on the table and they need different evidence:

1. **Replication.** Does the governance gain hold per model (native vs arm,
   McNemar exact, per stratum, never pooled across subsets)? Each model is its
   own paired experiment, so this is answerable from one model's artifacts.

2. **Structural vs. interaction.** Suppression plans are computed from the
   retrieved page and the screen with **no LLM involved**, so they are
   identical across models. If the parser's residual edge is structural, the
   parser-only question set must be *identical* across models — the committed
   sh_64k reference set is **{9, 11, 26, 35, 54, 69, 70, 91, 93}**. Same set ⇒
   structural. Scattered ⇒ model interaction. Either result is publishable.

The same fact has a hard corollary: **`hnav_geo`'s 8 harmful suppressions on
sh_64k will recur identically on every model**, because harm is counted from
the suppression plan, not from the answers. Its runs will be void by
preregistered condition 4 every time. It is in the campaign as the parser-free
contrast, and its void is the result, not a failure of the run.

## Model order — smallest weights first

So that a failure on the largest costs nothing that already ran.

| # | key | weights | arch | vLLM | measured longest sh_64k prompt | served context |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | `phi4_mini` | 7.2 GB | `Phi3ForCausalLM` | 0.9.1 | 43,321 tok | 49,152 |
| 02 | `gemma3_4b` | 8.1 GB | `Gemma3ForConditionalGeneration` | 0.9.1 | 48,224 tok | 57,344 |
| 03 | `gemma4_e2b` | 9.6 GB | `Gemma4ForConditionalGeneration` | 0.28.0 | 48,228 tok | 57,344 |
| 04 | `qwen35_9b` | 19.3 GB | `Qwen3_5ForConditionalGeneration` | 0.28.0 | 49,195 tok | 57,344 |
| — | *reference* | 8.0 GB | `Qwen3ForCausalLM` | 0.9.1 | 49,623 tok | 65,536 |

## Four things that were measured rather than assumed

Each of these would have produced a plausible-looking wrong result.

**1. The context window.** The runbook said the longest prompt was ≈42.5k
tokens. That was `chars / 4`. Measured against each model's own tokenizer *and*
chat template it is 43.3k–49.6k, and three of the four new models exceed 48k.
Serving at the instructed 48000 would have run for hours and then died on the
longest question of sh_64k — the held-out subset, one shot only.
(`hnav/deploy/measure_prompt_tokens.py`; the server's own `prompt_tokens` for
Phi-4-mini came back 43,321, matching the offline count exactly.)

**2. Which environment can load which model.** `vllm_0.9.1` registers
`Phi3ForCausalLM` and `Gemma3ForConditionalGeneration` but not
`Gemma4ForConditionalGeneration` or `Qwen3_5ForConditionalGeneration`. A second
environment (vLLM 0.28.0, torch 2.13+cu130, Python 3.12) was built in its own
prefix. **The frozen 0.9.1 env is never upgraded** — it is where the reference
result was produced, and upgrading it would silently redefine the baseline.

**3. Thinking mode.** The harness allows the model `GENERATION_MAX_TOKENS = 10`
output tokens — the benchmark's own `generation_max_length`. A thinking model
spends all ten on reasoning and scores ≈0, which reads as a model result and is
really a serving-mode artifact. Qwen3.5-9B's chat template honours
`enable_thinking=false`, so it is served in its non-thinking instruct mode via
`--default-chat-template-kwargs`. The preflight fails any model that still
emits a reasoning marker.

**4. No embedding server is needed.** `detector_gap` reads fact vectors from
the committed disk cache behind a fail-on-miss embedder. The reference config
reserved 42% of GPU1 for an embedding server; this campaign does not, so the
whole card is available to the answering model — which is what makes a 19.3 GB
model on a 24 GiB card feasible at all.

## What is held fixed while the model varies

From the frozen Stage-1 substrate, because the A/A floor (native vs
`native_repeat`, which must be **0** discordant or the run is void) depends on
them:

    --max-num-seqs 1            no batch-composition dependence
    --no-enable-prefix-caching  the measured source of run-to-run coupling
    --enforce-eager             no cudagraph memory or capture-shape effects
    --kv-cache-dtype fp8        reference precedent
    temperature 0, max_tokens 10, fixed system message   (set by the harness)

All three arms of a model answer against **one** server instance, so the
within-model cross-arm comparison is not confounded by the serving config.

Deliberately absent: tool-call parsers. `Conflict_Resolution` is plain question
answering; the harness never sends a `tools` array. Enabling a parser could
only add a failure mode. The preflight records this as `tool_check: N/A`.

## The gate before the money is spent

`hnav/deploy/preflight_model.py`, seven checks, all hard:

| check | the silent failure it prevents |
| --- | --- |
| `one_shot` | a second shot at an already-measured cell |
| `served_name` | artifacts carrying the reference model's name |
| `prompt_fits` | dying on the longest question, hours in |
| `generates` | an endpoint that answers `/v1/models` but not `/chat/completions` |
| `no_reasoning` | a thinking model scoring 0 and reading as a capability result |
| `deterministic` | a non-zero A/A floor, which voids the run |
| `short_prompt` | a failure that only appears on small contexts |

## Validity policy per run

Every preregistered run-voiding condition fails the subset, with one exception:
condition 2 (`native_in_band`) whose band (0.30, 0.50) was fixed from m3's
sh_64k measurement for **one** model. A different model leaves that band
without anything being wrong, so it is reported as a WARNING and must be
re-preregistered per model. One shot per model per subset; a void is reported,
never re-rolled.

## Operational

```bash
# on the box, repo root
nohup bash hnav/deploy/run_multimodel_campaign.sh > hnav/_out/campaign/campaign.log 2>&1 &

tail -f hnav/_out/campaign/campaign_progress     # model-level
tail -f hnav/_out/campaign/<key>/progress        # arm-level
```

Detached via `setsid`, so it is independent of the ssh session. Per-model and
campaign-level `mkdir` locks (atomic, unlike check-then-act) prevent the
duplicate-launch collision that happened on 2026-08-30. A model whose
`campaign.done` sentinel exists is skipped, so re-running the orchestrator
resumes rather than repeats. A failing arm does not abort its model; a failing
model does not abort the campaign.

Runtime, from the reference model's measured pace (~2h05m per arm): ~6.5 h per
small model, more for the 9B; roughly 30 h in total.

## After the runs

```bash
python -m hnav.geometry_filter.campaign_analysis <model-tag> --subset sh_64k
```

which reports, per arm and per stratum, native vs detector with an exact
within-model McNemar, the cross-arm pairing with its native-agreement check,
and every void condition.
