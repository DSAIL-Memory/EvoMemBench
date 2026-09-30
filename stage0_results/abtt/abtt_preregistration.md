# Pre-registration — ABTT geometry, sh_64k REPLICATION

**Status: registered, not yet run.** Committed before any answer on sh_64k was
graded in the whitened space. The commit timestamp of this file, relative to the
commit of the result artifact, is the evidence of ordering — a reviewer should
check it.

**This is a REPLICATION on a spent arena, not a fresh confirmatory shot.** The
sh_64k one-shot registered in `stage1_preregistration_v2.md` was fired and is
reported in `HNAV_FINAL_REPORT.md`. Re-using the arena is a deliberate,
declared choice (user decision, 2026-08-21): the value of the ABTT contrast is
that it pairs question-for-question against that result, and no other subset
gives that. Nothing here may be presented as an independent confirmation.

---

## 1. The claim, scoped

> On `sh_64k`, conflicted stratum, retrieval harness, benchmark page source:
> does applying ABTT whitening **before** the cosine screen change H-Nav's
> answer accuracy relative to the shipped raw-geometry detector?

Inert in this configuration and not claimed: `ambiguity_mode="none"` (so
`nmargin` and `H_z` never fire), chunk-level reranking, the write path,
`sh_262k`, and every arena other than MemoryAgentBench `Conflict_Resolution`.

## 2. Arms

Each `detector_gap` run emits 500 calls: `native` (100), `detector_suppress`
(100), `detector_demote_late` (100), `detector_anti` (100), `native_repeat`
(100, the A/A pair). Two runs, 1,000 calls total, both against the **frozen**
`:8003` substrate (`serve_stage1_chat.sh`, flags unchanged).

| arm | geometry | operating point | source |
|---|---|---|---|
| **A0** | — | `HNAV_MODE=off` | the `native` arm inside each run |
| **A1** | raw | `stage0_results/stage1_operating_point.json` (cos_pair 0.90, r_min 0.44) | re-run, not reused |
| **A2** | ABTT D=128 | `stage0_results/abtt/abtt_operating_point.json` (cos_pair 0.30, r_min 0.9539) | new |

**A1 is re-run in the same session as A2** rather than reusing
`detector_gap_confirmatory_sh64k.json`. It doubles the cost and removes any
argument that the two arms sat on different substrates. The headline contrast
is A2 − A1, so that contrast must not also carry a substrate difference.

**Consistency check, free:** the two runs produce two independent `native`
arms over identical prompts. They must agree. Disagreement is a substrate
instability and voids the comparison (see §5).

## 3. The whitening artifact

`stage0_results/abtt/abtt_whitening_D128.json`, fingerprint
`3fdacc1fcc479efb…`, `frozen_global` regime, D = 128, **fit on `sh_6k` +
`sh_32k` only** (`fit_subsets` recorded in the artifact; the fitter exits 2 on
any held-out subset). ABTT is unsupervised, so pooling held-out text would not
be leakage in the gold sense — it is refused anyway, because the transform
applied to sh_64k must not be partly derived from sh_64k.

D = 128 and `frozen_global` were selected on the calibration split from
detection quality alone (`G1_GATE_REPORT.md`); the operating point
`cos_pair = 0.30` was then selected by the unchanged `SELECTION_RULE` — require
`pair_filter`, require `n_suppressed_harmful == 0`, maximise
`pair_recall_pool` — with no LLM, no accuracy and no gold answer in the
objective. Both were frozen and committed before this document.

## 4. Success criteria, as computable expressions

**The study is deliberately registered as estimation, not as a superiority
test, and it is underpowered for superiority.** Calibration measured
`question_recall_conflicted` at 135/139 (ABTT) vs 133/139 (raw) — a +1.44 point
detection difference. Scaled to 66 conflicted sh_64k questions that predicts a
difference of roughly **+1 question**, which no test on n = 66 can resolve.
Registering a superiority threshold we expect to miss, and then reporting the
miss as if it were informative, would be theatre. So:

**Primary outcome (registered):** the point estimate and 95% confidence
interval of the conflicted-stratum accuracy difference **A2 − A1**, paired by
question, with the McNemar exact p reported alongside as a descriptive
statistic.

**Superiority** is claimed only if, on the conflicted stratum:
```
net_discordant(A2 over A1) >= +5  AND  mcnemar_exact_p < 0.01
```
**Non-inferiority** is claimed if:
```
net_discordant(A1 over A2) <= 3   (A2 no worse than A1 by more than 3 questions)
```
**Both arms must clear the no-H-Nav baseline:** A1 and A2 conflicted accuracy
each `> A0`, reproducing the direction of the committed 17/66 → 37/66 result.

Token cost of A2 must be `<= 0` relative to A0, as for A1.

## 5. Harm criteria and void conditions

Harm classes counted separately, stratified: `gold_cut`,
`malformed_generation`, `refusal_after_edit`, `information_loss`. The shipped
result was voided from "protective" by exactly one `refusal_after_edit`; that
outcome remains possible here and would be reported the same way.

**Void the RUN** (the comparison does not count at all):
1. Any `page_edit_errors > 0`, `mismatches > 0`, or containment violation
   (`named_ids ⊄ page_ids`) in either arm. *Pre-flight: 0 / 0 / 0 in both.*
2. Positive control does not fire 100/100 in either arm. *Pre-flight: OK, both.*
3. The two `native` arms disagree on any question.
4. A/A discordance > 0 in either run (the frozen server measured 0/0 before).
5. Wrong page source, wrong harness, or a prepass whose stamped
   `geometry_space` / `whitening_fingerprint` does not match the arm.
6. Any embedding cache miss on sh_64k. *Pre-flight: 4,680 hits / 0 misses.*
7. `max_pair_cosine_error_vs_prepass > 1e-6` in either arm. *Pre-flight:
   3.63e-07 (raw), 6.66e-16 (ABTT).*
8. Any change to `:8003`'s flags between the two runs.

**Void the PROTECTIVE claim only:** non-conflicted-stratum regression, as in the
original registration.

## 6. Falsifiable side-predictions

Computed from the parse and the frozen operating points **before any call is
sent**, and recorded here so they can be missed:

- **P1 — suppression counts.** ABTT suppresses fewer facts than raw on sh_64k:
  **719 vs 735**. *(Already measured in pre-flight; stated for the record, not
  falsifiable now.)*
- **P2 — the effect size.** The conflicted-stratum difference A2 − A1 lies in
  **[−2, +3] questions of 66**. This is the real prediction and it is the one
  most likely to be wrong; a difference outside that band falsifies the claim
  that calibration detection metrics extrapolate to held-out accuracy.
- **P3 — no new harm.** ABTT introduces no harm class that raw does not also
  produce. Calibration measured `n_conflicted_gold_cut = 2` for **both** arms,
  so ABTT's 16 fewer suppressions are not expected to come from the gold-fact
  region.
- **P4 — direction of the recall/coverage split.** Despite suppressing fewer
  facts, A2's conflicted-question coverage is `>=` A1's, because the facts it
  drops are ones no conflicted question depended on. If A2 both suppresses less
  *and* covers fewer conflicted questions, the calibration finding did not
  transfer.

## 7. Analysis

`hnav/stage1/detector_gap.py` at the commit of this document, unmodified
between registration and reporting. Grading is the benchmark's own
`substring_exact_match`. Strata come from `stage0_results/question_strata.json`,
which is parse- and gold-derived and model-independent.

**One shot per arm.** No re-runs because a number is unwelcome. If the run
voids, the void is reported and the cause diagnosed; it is not reinterpreted.

## 8. What may not be claimed from this result, whatever it shows

- Not an independent confirmation of the shipped sh_64k result — the arena is
  spent and this is a declared replication.
- Not a statement about `sh_262k`, CrossEp-Know, or any other encoder.
- Not a portability result: G1 explicitly failed to establish threshold
  transfer (one usable direction, band-normalised spread slightly favouring
  raw).
- A null here bounds ABTT's value **in this pipeline**, whose precision comes
  from the regex `pair_filter` and NLI rather than from cosine. It says nothing
  about arenas with no parse to fall back on, where G1 measured the largest
  whitening gains (recall at precision 1.000: 0.0750 → 0.5125 on sh_6k,
  0.0072 → 0.2910 on sh_32k).
