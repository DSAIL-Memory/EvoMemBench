# ABTT before the cosine screen — result

**Question.** H-Nav thresholds raw cosine in a space measured to be strongly
anisotropic (unrelated facts sit at cos ≈ 0.604). Does correcting the anisotropy
*before* the cosine screen, with thresholds re-fitted in the corrected space,
improve conflict detection and answer accuracy?

**Answer, in one line.** ABTT fixes the geometry completely and improves the
detector's precision/recall trade-off substantially, but it does **not** change
answer accuracy on this benchmark: **64/100 overall and 37/66 conflicted in both
arms, with not a single question differing.**

Pre-registered at `abtt_preregistration.md` (commit `dd4439b`,
2026-08-21T13:26:16+03:00); analysis code committed before results (`132a532`);
results below. Commit order is checkable.

---

## 1. The headline table

`sh_64k`, retrieval harness, benchmark page source, frozen `:8003` substrate,
`detector_suppress` mechanism.

| stratum | n | A0 native | A1 raw | A2 ABTT D=128 |
|---|---|---|---|---|
| overall | 100 | 45/100 | **64/100** | **64/100** |
| conflicted | 66 | 17/66 | **37/66** | **37/66** |
| non-conflicted | 34 | 28/34 | 27/34 | 27/34 |

Primary outcome, paired on the conflicted stratum:

```
A2 - A1:  n=66   A1-only correct = 0   A2-only correct = 0
          net = +0    95% CI [+0.0000, +0.0000]    McNemar exact p = 1
```

Registered **superiority** (net ≥ +5 and p < 0.01): **not met** — as the
registration predicted. Registered **non-inferiority** (net ≥ −3): **met**.
Both arms clear A0. Token cost −0.3067% (A1) and −0.2988% (A2), both ≤ 0.

## 2. Why this null is trustworthy

**A1 reproduces the committed campaign exactly.** Re-running the raw arm six
days after the original gave **500/500 identical graded outcomes and zero
differing model outputs** — byte-identical generated strings on every arm. The
frozen substrate is deterministic across sessions, so the A2−A1 contrast carries
the geometry difference and nothing else.

**The A/A floor is a true 0/0** in both runs. There is no run-to-run noise to
hide an effect in. A null here is an exact null, not a power failure at the
measurement level. What it cannot do is generalise beyond 66 conflicted
questions on one subset and one model.

**Every guard passed in both arms:** page-edit mismatches 0, containment
violations 0, page-edit errors 0, positive control fired 100/100, embedding
cache 4,680 hits / 0 misses, and the T12 cosine-reproduction guard measured
3.63e-07 (raw) and **6.66e-16** (ABTT) against a 1e-6 tolerance. `run_void =
False` in both.

## 3. The null is not "nothing happened"

The two arms genuinely behaved differently — the intervention reached the page
and then stopped mattering:

| | value |
|---|---|
| questions whose suppression plan differed | **12 / 100** (10 of them conflicted) |
| facts suppressed | raw **735**, ABTT **719** (−16) |
| questions whose answer text changed | **1** (q25) |
| questions whose *correctness* changed | **0** |

So ABTT edited 12% of pages differently and cut 16 fewer facts, and the model
produced a different string exactly once — without flipping a single answer from
wrong to right or right to wrong. **The facts the two geometries disagree about
are facts no question depended on.**

## 4. Side-predictions: 4/4 hit

| | prediction | outcome |
|---|---|---|
| P1 | ABTT suppresses fewer facts than raw | **HIT** — 719 < 735 |
| P2 | conflicted net in [−2, +3] of 66 | **HIT** — net = 0 |
| P3 | ABTT introduces no harm class raw does not | **HIT** — both arms: exactly one `refusal_after_edit`, on the same question (q77), same output |
| P4 | A2 conflicted coverage ≥ A1 | **HIT** — 37 = 37 |

P2 was the falsifiable one and it held: calibration detection metrics
(`question_recall_conflicted` 135/139 vs 133/139, predicting ≈ +1 question) did
extrapolate to held-out accuracy, in the sense that the predicted effect was too
small to appear and it did not appear.

The protective claim is void in **both** arms for the same single
`refusal_after_edit` on q77 — identical native output ("John Milton") and
identical post-edit refusal. ABTT neither caused nor fixed the known defect.

## 5. What ABTT did change — measured, and large

None of this reached accuracy, but all of it is real (`G1_GATE_REPORT.md`,
`m6_abtt_*.json`):

| property | raw | ABTT |
|---|---|---|
| anisotropy (unrelated-pair mean cos), sh_6k / sh_32k | 0.6024 / 0.6026 | ≈ 0.000 |
| candidate-pair floor | 0.5815 / 0.6130 | 0.083 / 0.081 |
| p10–p90 band width (screened pairs) | 0.1037 | 0.5402 |
| screen precision at equal recall, sh_32k | 5.3% (24,232 pairs → 1,282 true) | **51.3%** (2,421 pairs → 1,242 true) |
| recall at precision 1.000, sh_6k / sh_32k | 0.0750 / 0.0072 | **0.5125 / 0.2910** |

Mean-centering alone (D = 0) removes essentially all of the anisotropy; the
principal-direction removal buys the ranking improvements on top.

## 6. Why the improvement does not convert here

**The shipped pipeline does not buy its precision from cosine.** It runs a
deliberately loose cosine screen and gets precision from the regex `pair_filter`
(parsed subject+relation identity) and bidirectional NLI. Those stages already
deliver `pair_precision = 1.000` and `n_suppressed_harmful = 0` on raw geometry,
so a cleaner cosine screen has nothing left to contribute: raw's screen already
reaches ~99.5% of the reachable true pairs inside the pool, and the downstream
stages remove its false positives for free.

Put plainly: **ABTT improves the stage that was not the bottleneck.**

A second, independent finding from the same campaign reinforces this. Whitening
the *query* vector as well as the facts (`--whiten-scope all`) actively hurts:
reachable true-supersession pairs fall 1,443 → 1,048 on sh_6k (−27%), because
`select_pool` then builds a worse pool. ABTT helps symmetric fact-vs-fact
comparison and hurts asymmetric question-vs-fact retrieval. The shipped scope
(`pairs`) confines it to the comparison it helps.

## 7. What may and may not be claimed

**May be claimed.**
- On MemoryAgentBench `Conflict_Resolution` `sh_64k`, with this encoder and this
  answering model, ABTT-whitened geometry is *exactly as accurate* as raw
  geometry — 0 of 66 conflicted questions differ — at equal precision, equal
  harm and equal token cost, despite a materially different intervention.
- The anisotropy of `Qwen3-Embedding-4B` is real, large (≈ 0.60), and fully
  removable by mean-centering; the geometric gains from removing it are large
  on detection-quality metrics.
- In this pipeline those gains are absorbed by the regex identity screen and the
  NLI verifier and do not reach accuracy.

**May not be claimed.**
- Not an independent confirmation of the shipped sh_64k result — declared
  replication on a spent arena.
- Nothing about `sh_262k`, CrossEp-Know, other encoders or other answering
  models. The generalisation limit here is n = 66 conflicted questions on one
  subset, *not* measurement noise (the A/A floor is 0).
- Not a portability result. G1 explicitly failed to establish threshold
  transfer: one usable direction, and band-normalised threshold spread slightly
  favoured raw.
- **Not evidence that ABTT is useless.** It is evidence that it is redundant
  *with a regex identity screen available*. The largest measured gains
  (recall at precision 1.000, 6.8× and 40×) live precisely in the regime where
  cosine must carry precision alone — which is the situation in any arena
  without a parseable fact template. That case is untested here.

## 8. The most useful follow-up

Drop the regex `pair_filter` and re-run both geometries. If whitened geometry
holds precision without the parse and raw does not, that is a direct answer to
the standing criticism that H-Nav's gain comes from the benchmark's templates
and serial numbers rather than from geometry — the criticism M1b exists to
address. It costs one more pair of arms and needs no new held-out data.

## 9. Artifacts

| file | what |
|---|---|
| `abtt_preregistration.md` | registered before grading (`dd4439b`) |
| `abtt_arm_analysis.py` | registered analysis (`132a532`), self-tested against the committed artifact |
| `abtt_arm_A1_raw_sh64k.json` | arm A1, raw geometry |
| `abtt_arm_A2_abtt_sh64k.json` | arm A2, ABTT D=128 |
| `abtt_operating_point.json` | whitened operating point, frozen on detection quality only |
| `abtt_whitening_D128.json` | μ and C, `frozen_global`, fit on sh_6k+sh_32k, fingerprint `3fdacc1fcc479efb…` |
| `G1_GATE_REPORT.md`, `m6_abtt_*.json` | offline geometry evidence |

The shipped `stage0_results/stage1_operating_point.json` was **not modified** by
this campaign.
