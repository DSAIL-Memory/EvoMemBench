# Threshold re-fit after the 512-token correction — DIFF REPORT  [T12/T13]

> Executed 2026-08-16 on `ozonderlab2` GPU1 per `hnav/deploy/REFIT_RUNBOOK.md`.
> **Calibration split only** (`sh_6k` + `sh_32k`); `sh_64k`/`sh_262k` untouched.
> Corrected embedder: fp32, `max_length=8192`, namespace
> `Qwen_Qwen3-Embedding-4B|float32|L8192`, `fallback_chunker: false` in every
> output. New artifacts: `stage0_results/refit_L8192/*_L8192.json`. The L512
> originals in `stage0_results/final/` are untouched — evidence is never deleted.
>
> Structured on the T12 supervisor triage: **A** invalid pending re-fit,
> **B** expected-unaffected-but-confirm, **C** structurally unreachable.

## 0. Preconditions verified

| check | result |
|---|---|
| `max_position_embeddings` (from `config.json`, not the model card) | **40960** ≥ 8192 ✓ (my earlier 32768 figure was conservative) |
| tokenizer `model_max_length` | 131072 |
| calibration chunks in **Qwen** tokens | sh_6k 2,455–4,885 · sh_32k 58–5,008 |
| chunks exceeding 8192 | **0** — L8192 never truncates the calibration split |
| `fallback_chunker` | `false` in m2 and m3 (the benchmark's own chunker) |

Two engineering defects had to be fixed before the re-fit could run at all; both
were verified numerically neutral before use (`BUILD_NOTES` §11, §11b):
attention-memory (GQA fused-kernel eligibility) and token-budget batching. The
runbook's premise was right — at L8192 the old code OOM'd on 9 of 11 chunks.

---

## 1. BUCKET A — invalid pending re-fit → what they became

### 1a. The headline claim: `NOT_DEGENERATE`

| subset | old (L512) | new (L8192) | verdict |
|---|---|---|---|
| sh_6k | NOT_DEGENERATE | **NOT_DEGENERATE** | **re-earned** |
| sh_32k | NOT_DEGENERATE | **NOT_DEGENERATE** | **re-earned** |

Pre-registered criterion (unchanged): degenerate iff `median H_raw < 0.1·ln(m)`
**and** `median H_z ≥ 5·median H_raw`. Neither subset comes close to it under
the corrected embedder:

| subset | median H_raw | median H_z | 0.1·ln(m) | ln(m) ceiling |
|---|---|---|---|---|
| sh_6k | 0.533566 → **0.548283** | 0.365334 → **0.365334** | 0.069315 | 0.693147 |
| sh_32k | 1.570876 → **1.889055** | 1.957277 → **2.037665** | 0.219722 | 2.197225 |

**Re-earned, not carried forward.** Scope note: this is the *calibration split*.
The committed "4/4" spanned all four subsets; `sh_64k`/`sh_262k` were not re-run
(they are confirmatory and out of scope here), so the defensible claim is now
**NOT_DEGENERATE 2/2 on the calibration split, re-verified after the truncation
correction**, with the other two subsets' L512 rows still formally provisional.

### 1b. Retrieval signals moved materially — sh_32k roughly halved

| subset | quantity | old p50 | new p50 | Δ |
|---|---|---|---|---|
| sh_6k | `nmargin` | 0.016788 | 0.014982 | −10.8% |
| sh_6k | `margin` | 1.234844 | 1.165947 | −5.6% |
| sh_6k | `H_raw` | 0.533566 | 0.548283 | +2.8% |
| sh_6k | `H_z` | 0.365334 | 0.365334 | **0.0%** (see §3) |
| sh_32k | `nmargin` | 0.010214 | **0.004876** | **−52.3%** |
| sh_32k | `margin` | 0.794147 | **0.365021** | **−54.0%** |
| sh_32k | `H_raw` | 1.570876 | 1.889055 | +20.3% |
| sh_32k | `H_z` | 1.957277 | 2.037665 | +4.1% |
| sh_32k | `eff_size` | 7.0800 | 7.6727 | +8.4% |

Reading: with whole chunks embedded rather than their first ~12%, the top
candidates on sh_32k are **much closer together** — margin and normalized margin
both halve, entropy rises. The truncated measurement made retrieval look more
decisive than it is. Tie rate stays 0.0 everywhere.

### 1c. Thresholds

**Pooled** (what `fit_thresholds` emits, and what the frozen constants copied):

| threshold | old (L512) | new (L8192) | Δ |
|---|---|---|---|
| `nmargin` p25 | 0.0047643914 | **0.0039060968** | **−18.0%** |
| `H_z` p75 | 1.9569327965 | **2.0362532742** | **+4.1%** |
| `r_min` p10 | 0.1923661662 | 0.1923663786 | +0.0000011% |

**Per subset** — the form the runbook requires, and the only defensible one:

| subset | `nmargin` p25 | `H_z` p75 | `r_min` p10 |
|---|---|---|---|
| sh_6k | 0.0085124131 | **0.3653338551** (the constant) | 0.7435854973 |
| sh_32k | 0.0019850782 | 2.0747014126 | 0.1768898278 |
| *pooled* | *0.0039060968* | *2.0362532742* | *0.1923663786* |

The two subsets' `nmargin` thresholds differ by **4.3×**, and neither equals the
pooled value.

### 1d. Read-side labels and coverage

| label | sh_6k old → new | sh_32k old → new |
|---|---|---|
| READ_AMBIGUOUS | 0.210 → **0.110** | 0.290 → **0.390** |
| READ_CONFLICT | 1.000 → 1.000 | 1.000 → 1.000 |
| READ_STALE | 1.000 → 1.000 | 1.000 → 1.000 |
| READ_DISTRACTOR | 0.110 → 0.110 | 0.110 → 0.110 |
| READ_HIGH_INTERFERENCE | — | 0.500 → 0.500 |
| READ_RELEVANT_BELOW_K | — | 1.000 → 1.000 |

The Faz A **"ambiguity fires 94/200"** figure, re-derived (mode `any`):

| thresholds applied | sh_6k | sh_32k | total |
|---|---|---|---|
| old (L512) constants on old signals | — | — | **94/200** (as reported in T11) |
| old constants on **new** signals | 14 | 92 | 106/200 |
| **new pooled** constants on new signals | 11 | 63 | **74/200** |
| **per-subset** thresholds on their own subset | 25 | 38 | 63/200 |

So the coverage figure moves 94 → **74/200** under the corrected embedder and
re-fit pooled thresholds. It is threshold-dependent in exactly the way §3
predicts, and on sh_6k the `H_z` half of the screen contributes **zero** under
every variant.

### 1e. Not re-fit — still provisional (declared, not hidden)

`m3`'s counterfactual columns — `accuracy_native`, `accuracy_repaired`,
`repair_helped`/`repair_harmed` — and **m4** could **not** be re-fit: they need
the LLM counterfactual arm, and the re-fit ran `--no-llm` per the runbook
(the arm costs hours of the same GPU that Thrust 2 is queued for). m4 aborted
cleanly with "no labeled candidates found" rather than emitting a stale number.

Old values, still carrying the L512 caveat: sh_6k `accuracy_native` 0.33,
helped 0 / harmed 0; sh_32k 0.47, helped 2 / harmed 1; m4 `delta_auc` 0.0674,
LRT p 0.3408, `h2_pass` False.

**However, m4's inputs are confirmed stable** — every feature it consumes is
identical old→new to 6 decimals (§2), so re-running it would change only
through the labels, which are LLM-derived and unchanged in kind. Recommend
running the LLM arm only if those specific numbers are load-bearing for the
thesis; otherwise report them as L512-era with the caveat.

---

## 2. BUCKET B — expected unaffected → **all CONFIRMED STABLE**

| quantity | old | new | verdict |
|---|---|---|---|
| **`r_min` p10 → `R_MIN_CAL`** (pooled) | 0.1923661662 | 0.1923663786 | **stable** (Δ 2.1e-07, 1.1e-06 relative) |
| `r_min` p10 sh_6k | 0.7435854631 | 0.7435854973 | stable (Δ 3.4e-08) |
| `r_min` p10 sh_32k | 0.1768898819 | 0.1768898278 | stable (Δ 5.4e-08) |
| M1 separation AUC (both subsets) | 1.000000 | 1.000000 | identical |
| M1 `gate_pass` (S3 kill switch) | True | True | identical |
| M1 median whole-blob sim | 0.9636 / 0.9638 | 0.9636 / 0.9638 | identical |
| M1 control sim p50 | 0.597665 / 0.605088 | 0.597665 / 0.605088 | identical |
| M1b best-F1 τ → `COS_PAIR_CAL` | 0.91 / 0.93 | 0.91 / 0.93 | identical |
| M1b precision | 0.8605 / 0.8474 | 0.8605 / 0.8474 | identical |
| M1b recall | 0.9250 / 0.8311 | 0.9250 / 0.8311 | identical |
| M1b F1 | 0.8916 / 0.8392 | 0.8916 / 0.8392 | identical |
| m3 write conflict rate | 0.3516 / 0.3615 | 0.3516 / 0.3615 | identical |
| m3 write critical-delta | 0.0000 / 0.0000 | 0.0000 / 0.0000 | identical |
| m3 write stale-supersede | 0.3516 / 0.3615 | 0.3516 / 0.3615 | identical |
| m3 write would-intervene | 0.0022 / 0.1039 | 0.0022 / 0.1039 | identical |
| m3 write after-vetoes | 0.0000 / 0.0165 | 0.0000 / 0.0165 | identical |
| `sim_max` p10/p50/p90 | — | — | identical to 6 dp |
| `whole_blob_sim`, `diff_sim`, `qr_residual` p10/p50/p90 | — | — | identical to 6 dp |

### **`R_MIN_CAL` verdict: CONFIRMED STABLE — no escalation.**

Movement is **1.1e-06 relative**, i.e. fp32 accumulation noise, on both the
pooled value and each subset independently. The premise the whole triage rests
on — *facts are far below 512 tokens, so fact-level signals were never
truncated* — is **upheld**. Nothing needs re-deriving beyond Bucket A.

---

## 3. BUCKET C — structurally unreachable, and now empirically demonstrated

**Defect (a): the `H_z` screen cannot fire on sh_6k, under any threshold.**

`sh_6k` has `n_chunks = 2`. Z-scoring two scores always yields `{+1, −1}`
regardless of their values, so `H_z ≡ 0.3653338550872077` **exactly** — verified
closed-form on three unrelated score pairs, and confirmed in both m2 runs where
sh_6k's `H_z` has `min = max = p50 = 0.3653338551`. The entropy ceiling is
`ln 2 = 0.693147`, far below the threshold (old 1.9569, new 2.0363).

The re-fit demonstrates this empirically. `H_z` fired on sh_6k:

| thresholds | H_z fires on sh_6k |
|---|---|
| old pooled (1.9569) | 0/100 |
| new pooled (2.0363) | 0/100 |
| **sh_6k's OWN p75 (0.3653338551)** | **0/100** |

The last row is the important one: **per-subject fitting does not rescue the
screen**. Because `H_z` is a constant, no strict inequality can select a proper
subset of sh_6k — the screen is vacuous there by arithmetic, not by calibration.
Half the calibration split contributes nothing but a constant to that screen.

**Defect (b): the pooled percentile is an artifact, now with hard evidence.**

`fit_thresholds` concatenates both subsets' read rows and takes one percentile.
Since all 100 sh_6k rows sit below sh_32k's entire range, the pooled p75 is
arithmetically **sh_32k's median**, not anyone's p75:

| | pooled `H_z` p75 | sh_32k `H_z` **p50** |
|---|---|---|
| old (L512) | 1.9569327965 | 1.9572768924 |
| new (L8192) | 2.0362532742 | 2.037665 |

The two agree to ~0.0004 in both eras — the pooled "75th percentile of the
calibration split" is a median of one subset, mislabelled.

**Position, as the runbook required me to state: a pooled percentile cannot be
justified here.** Pooling quantiles assumes exchangeable draws from one
population. These subsets are not: `n_chunks` differs 4.5× (2 vs 9), `H_z`'s
support differs in kind (a point mass vs a distribution), its median moves
0.365 → 2.038, and `r_min` p10 differs 4.2× (0.744 vs 0.177). The quantity is as
much a function of store size as of ambiguity. **Report and consume per-subset
thresholds.** If a single value is ever needed for an unfitted subset, derive it
from a declared scaling rule (e.g. in `ln m`), fixed before any confirmatory
use — never from a pooled percentile, and never described as "the calibration
threshold".

---

## 4. Verdict summary

| # | statement |
|---|---|
| 1 | **`NOT_DEGENERATE` re-earned** on both calibration subsets under the corrected embedder; the 4/4 claim narrows to 2/2 calibration + 2 still-provisional confirmatory subsets. |
| 2 | **`R_MIN_CAL` confirmed stable** (1.1e-06 relative). No escalation; the fact-level premise holds. |
| 3 | **`COS_PAIR_CAL` (0.92) confirmed** — M1b τ/P/R/F1 bit-identical. |
| 4 | **`NMARGIN_CAL` changed −18.0%** (pooled) and is 4.3× apart between subsets. |
| 5 | **`H_Z_CAL` changed +4.1%** (pooled) and is *vacuous on sh_6k under any value*. |
| 6 | **M1's S3 kill switch, m3's whole write side, and all m4 inputs: confirmed identical.** |
| 7 | **m3 read accuracy columns and m4 itself: NOT re-fit** (need the LLM arm) — still L512-era. |
| 8 | The corrected signals make sh_32k retrieval look **substantially more ambiguous** (margin −54%). |

## 5. Decision taken — option 3: mark superseded, do not change live values

**Decided by the coordinator, 2026-08-16.** `NMARGIN_CAL` and `H_Z_CAL` keep
their L512-era values; the corrected per-subset values are recorded alongside
them in `hnav/core/read_gate.py` (`REFIT_PER_SUBSET`) and in
`stage0_results/refit_L8192/per_subset_thresholds.json`.

Rationale on the record:

1. **Reproducibility wins.** `hnav/stage1/detector_gap.py`'s committed
   artifacts — including the pre-registered, audited, one-shot held-out
   confirmatory run — were produced under these exact constants. Swapping them
   would make an audited result unreproducible from the code that produced it,
   and the constants are **inert in the shipped configuration** anyway
   (`ambiguity_mode="none"`).
2. **The replacement would re-freeze the documented defect.** Writing the new
   *pooled* scalars in would perpetuate exactly what §3 disproves.

Enforcement (`hnav/tests/test_threshold_provenance.py`, 12 tests): the live
values are pinned to the L512 numbers and asserted **not** to equal the L8192
pooled ones; the superseded annotation's markers must be present, so deleting
the label fails the suite; `REFIT_PER_SUBSET` must equal the artifact exactly;
the two eras must remain distinct records; `R_MIN_CAL`'s 1.1e-06 stability is
pinned with an ESCALATE message; and the sh_6k `H_z` vacuity is pinned against
the artifact's `min == max == p75` plus the `ln 2` ceiling. Verified to fail on
all four decay modes (live swap 3 fail, annotation deleted 1, per-subset drift
1, eras aliased 2).

**Rule going forward:** the live constants reproduce L512-era artifacts only.
Any new analysis uses `REFIT_PER_SUBSET`, per subset, never pooled.

## 6. Closed — LLM arms deliberately not re-run

`m3`'s counterfactual columns and `m4` remain L512-era **LLM labels with
confirmed-identical inputs** (§1e, §2). Coordinator decision 2026-08-16: not
re-run, because the probe and detector-gap results have superseded m3's repair
story with better measurements, and hours of contended GPU for numbers nothing
now rests on is not a good trade. If a specific one later proves load-bearing,
that one gets re-run.
