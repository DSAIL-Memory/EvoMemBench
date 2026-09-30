<!-- GENERATED FILE - do not edit by hand.
     python -m hnav.geometry_filter.multimodel_summary --all --out pipelines/MULTIMODEL_SUMMARY.md
     (needs the hnav env; the tool imports the campaign mcnemar_exact_p rather
     than reimplementing it, which pulls numpy through the import chain.)
     Every number here is derived from the committed detector_gap artifacts.
     A hand-transcribed version of this table was wrong once - see the
     page_source note in MULTIMODEL_CAMPAIGN_PLAN.md. -->

# Multi-model comparison (generated from artifacts)

50 measured cells · mechanism `detector_suppress` · page_source `['benchmark']`

> **hnav_geo sh_64k n_suppressed_harmful is 8 on EVERY model - suppression plans contain no LLM, so harm cannot vary with the answering model. Void by condition 4 throughout.**

### sh_6k · all stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 30 | 94 (+64) | 95 (+65) | 77 (+47) | 84 (+54) | 54 (+24) |
| Qwen/Qwen3.5-9B | 39 | 98 (+59) | 99 (+60) | 81 (+42) | — | — |
| google/gemma-3-4b-it | 45 | 89 (+44) | 89 (+44) | 76 (+31) | — | — |
| google/gemma-4-E2B-it | 40 | 83 (+43) | 83 (+43) | 72 (+32) | — | — |
| microsoft/Phi-4-mini-instruct | 40 | 88 (+48) | 89 (+49) | 76 (+36) | — | — |

### sh_6k · conflicted stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 4 | 68 (+64) | 69 (+65) | 51 (+47) | 58 (+54) | 28 (+24) |
| Qwen/Qwen3.5-9B | 13 | 72 (+59) | 73 (+60) | 55 (+42) | — | — |
| google/gemma-3-4b-it | 22 | 65 (+43) | 65 (+43) | 53 (+31) | — | — |
| google/gemma-4-E2B-it | 16 | 59 (+43) | 59 (+43) | 48 (+32) | — | — |
| microsoft/Phi-4-mini-instruct | 14 | 62 (+48) | 63 (+49) | 50 (+36) | — | — |

### sh_6k · unique stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 26 | 26 (+0) | 26 (+0) | 26 (+0) | 26 (+0) | 26 (+0) |
| Qwen/Qwen3.5-9B | 26 | 26 (+0) | 26 (+0) | 26 (+0) | — | — |
| google/gemma-3-4b-it | 23 | 24 (+1) | 24 (+1) | 23 (+0) | — | — |
| google/gemma-4-E2B-it | 24 | 24 (+0) | 24 (+0) | 24 (+0) | — | — |
| microsoft/Phi-4-mini-instruct | 26 | 26 (+0) | 26 (+0) | 26 (+0) | — | — |

### sh_32k · all stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 53 | 83 (+30) | 85 (+32) | 77 (+24) | 78 (+25) | 72 (+19) |
| Qwen/Qwen3.5-9B | 61 | 91 (+30) | 92 (+31) | 86 (+25) | — | — |
| google/gemma-3-4b-it | 38 | 51 (+13) | 52 (+14) | 45 (+7) | — | — |
| google/gemma-4-E2B-it | 44 | 63 (+19) | 63 (+19) | 58 (+14) | — | — |
| microsoft/Phi-4-mini-instruct | 50 | 72 (+22) | 72 (+22) | 66 (+16) | — | — |

### sh_32k · conflicted stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 19 | 49 (+30) | 51 (+32) | 43 (+24) | 44 (+25) | 38 (+19) |
| Qwen/Qwen3.5-9B | 27 | 57 (+30) | 58 (+31) | 52 (+25) | — | — |
| google/gemma-3-4b-it | 11 | 22 (+11) | 23 (+12) | 18 (+7) | — | — |
| google/gemma-4-E2B-it | 17 | 36 (+19) | 36 (+19) | 31 (+14) | — | — |
| microsoft/Phi-4-mini-instruct | 16 | 39 (+23) | 39 (+23) | 33 (+17) | — | — |

### sh_32k · unique stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 34 | 34 (+0) | 34 (+0) | 34 (+0) | 34 (+0) | 34 (+0) |
| Qwen/Qwen3.5-9B | 34 | 34 (+0) | 34 (+0) | 34 (+0) | — | — |
| google/gemma-3-4b-it | 27 | 29 (+2) | 29 (+2) | 27 (+0) | — | — |
| google/gemma-4-E2B-it | 27 | 27 (+0) | 27 (+0) | 27 (+0) | — | — |
| microsoft/Phi-4-mini-instruct | 34 | 33 (-1) | 33 (-1) | 33 (-1) | — | — |

### sh_64k · all stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 45 | — | 66 (+21) | 56 (+11) **VOID** | 55 (+10) | 59 (+14) **VOID** |
| Qwen/Qwen3.5-9B | 51 | 67 (+16) | 69 (+18) | 62 (+11) **VOID** | — | — |
| google/gemma-3-4b-it | 33 | 38 (+5) | 38 (+5) | 36 (+3) **VOID** | — | — |
| google/gemma-4-E2B-it | 37 | 43 (+6) | 45 (+8) | 41 (+4) **VOID** | — | — |
| microsoft/Phi-4-mini-instruct | 46 | 57 (+11) | 57 (+11) | 52 (+6) **VOID** | — | — |

### sh_64k · conflicted stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 17 | — | 39 (+22) | 29 (+12) **VOID** | 28 (+11) | 31 (+14) **VOID** |
| Qwen/Qwen3.5-9B | 24 | 42 (+18) | 43 (+19) | 36 (+12) **VOID** | — | — |
| google/gemma-3-4b-it | 14 | 19 (+5) | 19 (+5) | 17 (+3) **VOID** | — | — |
| google/gemma-4-E2B-it | 21 | 29 (+8) | 31 (+10) | 25 (+4) **VOID** | — | — |
| microsoft/Phi-4-mini-instruct | 16 | 27 (+11) | 27 (+11) | 22 (+6) **VOID** | — | — |

### sh_64k · unique stratum

| model | native | hnav_raw | hnav_idonly | hnav_geo | hnav_ces | hnav_abtt_noparser |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 28 | — | 27 (-1) | 27 (-1) **VOID** | 27 (-1) | 28 (+0) **VOID** |
| Qwen/Qwen3.5-9B | 27 | 25 (-2) | 26 (-1) | 26 (-1) **VOID** | — | — |
| google/gemma-3-4b-it | 19 | 19 (+0) | 19 (+0) | 19 (+0) **VOID** | — | — |
| google/gemma-4-E2B-it | 16 | 14 (-2) | 14 (-2) | 16 (+0) **VOID** | — | — |
| microsoft/Phi-4-mini-instruct | 30 | 30 (+0) | 30 (+0) | 30 (+0) **VOID** | — | — |

## Provenance

| model | arm | subset | page_source | A/A | void | harmful |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | hnav_abtt_noparser | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_abtt_noparser | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 5 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_abtt_noparser | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_ces | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_ces | sh_64k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_ces | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_geo | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_geo | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 8 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_geo | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_idonly | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_idonly | sh_64k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_idonly | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_raw | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | hnav_raw | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_geo | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_geo | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 8 |
| Qwen/Qwen3.5-9B | hnav_geo | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_idonly | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_idonly | sh_64k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_idonly | sh_6k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_raw | sh_32k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_raw | sh_64k | benchmark | 0 | - | 0 |
| Qwen/Qwen3.5-9B | hnav_raw | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_geo | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_geo | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 8 |
| google/gemma-3-4b-it | hnav_geo | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_idonly | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_idonly | sh_64k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_idonly | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_raw | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_raw | sh_64k | benchmark | 0 | - | 0 |
| google/gemma-3-4b-it | hnav_raw | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_geo | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_geo | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 8 |
| google/gemma-4-E2B-it | hnav_geo | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_idonly | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_idonly | sh_64k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_idonly | sh_6k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_raw | sh_32k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_raw | sh_64k | benchmark | 0 | - | 0 |
| google/gemma-4-E2B-it | hnav_raw | sh_6k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_geo | sh_32k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_geo | sh_64k | benchmark | 0 | 4_no_harmful_suppression | 8 |
| microsoft/Phi-4-mini-instruct | hnav_geo | sh_6k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_idonly | sh_32k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_idonly | sh_64k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_idonly | sh_6k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_raw | sh_32k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_raw | sh_64k | benchmark | 0 | - | 0 |
| microsoft/Phi-4-mini-instruct | hnav_raw | sh_6k | benchmark | 0 | - | 0 |
