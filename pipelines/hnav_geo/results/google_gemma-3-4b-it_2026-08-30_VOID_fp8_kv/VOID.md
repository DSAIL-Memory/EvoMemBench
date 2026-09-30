# VOID — this run measured a broken instrument, not a model

**Do not quote these numbers as gemma-3-4b-it's accuracy.** They are kept
because the failure itself is a result.

## What happened

The multi-model campaign inherited `--kv-cache-dtype fp8` from the frozen
Stage-1 substrate, where it was chosen for the reference `Qwen3-4B-Instruct-2507`
so an embedding server could share GPU1. gemma-3-4b-it does not tolerate it.

This run scored **6–13/100**, including **4/26 on the unique stratum** —
single-fact retrieval, no conflict present, nothing for H-Nav to do — where
Phi-4-mini scores 26/26 on the same questions. The outputs are the tell:

    "United States of United States of United States of United"
    " [1920. \n"                        (truth: shahnameh)
    "Answer: \n"                        27–54% of answers, by subset

## The measurement that condemned it

`hnav/deploy/diagnose_serving.sh` served the same weights with the same flags on
the same stack, varying only the cache dtype, and probed ten unique-stratum
questions:

| variant | unique-stratum | sample |
| --- | --- | --- |
| `legacy:fp8` | **0/10** | `' [1920. \n'` vs `shahnameh` |
| `legacy:auto` | **9/10** | `'Shahnameh'` ✓ |

The vLLM-version axis was tested too (`modern:auto`, `modern:fp8`) rather than
assuming the dtype: Gemma-3's interleaved sliding-window/global attention has
known bad interactions with attention backend, prefix caching and specific
vLLM/Transformers versions, so a dtype-only A/B could have cleared fp8 and left
the real cause untested. Those cells failed to start for an unrelated reason
(no CUDA toolkit for FlashInfer's JIT), and the legacy pair was already
decisive.

## Why it was not caught before 4,500 completions were spent

Every preflight check in force at the time passed. The model's output was
fluent, non-empty, deterministic, and free of reasoning markers. Nothing
verified that the answers were **right**.

Two checks were added in response, and they are what now stands between a
broken serving configuration and a measured cell:

- `no_degenerate` — an n-gram repeated three times in a ten-token answer;
- `answer_sanity` — a ten-question accuracy floor on the unique stratum, the
  questions any working model of this class answers near ceiling.

The floor is the load-bearing one. It later caught a second, unrelated failure
the marker-based check missed entirely: Qwen3.5 with thinking left on answers
`"Thinking Process:\n\n1.  **Analyze"` — no tag, empty `reasoning_content`, so
`no_reasoning` passed it, and only the 0/10 floor exposed it.

## The methodological point

A KV-cache dtype silently inherited from a *different* model's frozen
configuration destroyed one model's benchmark score while leaving every
surface property of its output looking healthy. Configuration provenance is
part of the experiment, and "it ran and produced plausible text" is not
evidence that an instrument works.

The re-run under BF16 lives in `google_gemma-3-4b-it_2026-08-31`.
See `hnav/deploy/models.d/02_gemma3_4b.env` and
`pipelines/MULTIMODEL_CAMPAIGN_PLAN.md`.
