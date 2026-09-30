# QDA conflict scorer — campaign report (2026-08-29)

Code: `hnav/qda_filter/` (preprocess / spectrum / fit / score / calibrate /
adapt / eval / run_all). Raw results: the JSONs in this directory, each with
full provenance (git commit, dataset sha256, embedding namespace, seeds).
Preregistration: `PREREG.md` — gates G1–G6 and every rule were fixed before
Stage-2 numbers existed; the three whitening/rank fixes forced by the smoke
diagnostics are in its Addendum A with their triggers.

**The headline is a null result. V2 — the unquantized QDA core this campaign
exists to test — does NOT beat CES.** Balanced sh_64k AUROC 0.8695
[0.8574, 0.8811] vs CES 0.9756; paired delta −0.106 [−0.118, −0.095]; band
AUROC 0.797 vs 0.969. G2 fails decisively, in every cosine stratum (mean
within-bin delta −0.32 over the 14 populated bins), and no variant closes
the gap on the mandated arena. The existing pipeline keeps CES + ABTT-cosine;
nothing changes. This extends the repo's earlier §11 verdict (LDA − RCESP =
−0.114): covariance-aware extensions of the subspace methods lose on this
data, and this campaign now shows *why* (§2, §8).

**Fit discipline.** Fit split = sh_6k + sh_32k. Negatives partitioned
(seed 0) 20% conformal-calibration / half A (whitening) / half B (null
pool) before any fitting. sh_64k scored once, frozen. Discovery
(`discovery.json`): serial order is derivable for BOTH classes (fact ids are
serials; `fact_a` earlier on all 54,569 records), so `μ̂0` was estimated
rather than zeroed; sh_262k is absent from the gold dataset by construction.

## 1. What broke, and the fixes (Addendum A — read before trusting anything)

The prescribed `Σ0^{−1/2}` whitening is not computable as stated on this
data, and the smoke gates caught it exactly as designed:

- Difference vectors of a finite fact set span ≤ n_facts − 1 dims and the
  audited pair graph is FRAGMENTED: half A (3,893 pairs) spans only 1,379
  of 2,432 dims, and half B's difference energy is mostly orthogonal to it
  (median whitened variance 0.0095). Inverting small eigenvalues amplified
  un-estimated directions ~440× and made V2 come out *inverted* (0.34).
- Fix B: cap whitening amplification at the Ledoit-Wolf target scale μ —
  only the 454 well-estimated above-bulk directions are damped (max
  damping 4.9×); no direction is amplified. No new tuning knob.
- Fix C: parallel analysis on trace-normalized spectra — pooled permutation
  draws are ~80% negatives whose whitened variance dominates gold's at
  every index, so raw-eigenvalue envelopes test total variance, not shape.

After the fixes the machinery demonstrably works: V1 (the quantized pooled
core) reaches 0.9337 balanced sh_64k — +0.061 over the pooled raw-space CES
(0.8725), i.e. the μ-capped whitening genuinely helps a pooled subspace
detector. See G1 below for why the gate still fails.

## 2. Spectrum (`spectrum.json`)

LW shrinkage 0.0866 on half A (n = 3,893); 454 of 2,432 eigenvalues above
the target scale μ = 6.04e-5. Whitened gold spectrum (n1 = 989, nontrivial
rank 988): top eigenvalues 4.87, 4.67, 4.32, … 1.92 at index 64 — a long,
flat, distributed slope, not a spiked spectrum. Trace-normalized parallel
analysis: the top fraction 0.0277 is ~3× its null envelope 0.0088, and
**k_obj hit the preregistered cap of 64** with every index still
significant. **k_subj = 0** — no significant variance-deficit tail
(with n1 = 989 < N′, the sample bottom cannot resolve one). σ1² = 0.346.

Two structural readings, both load-bearing for the conclusion:

- The half-B whitened sanity spectrum (median 0.0095, top 3.47, bottom
  ~1e-17) says whitening CANNOT isotropize the negative class here: held-out
  negatives live largely outside the eigenbasis estimated from half A. The
  QDA premise — "Σ0 known, gold structure read off in whitened coordinates"
  — fails at the estimation step, not the modeling step.
- Bootstrap stability (50 resamples, W0 fixed): median largest principal
  angle of U_obj vs the point estimate is **81° at k=1, 87° at k=5, 89° at
  k=64**. Individual eigendirections of the gold covariance are essentially
  unidentified; only the aggregate subspace energy carries signal. A
  weighting scheme keyed to per-direction eigenvalues (the (1 − 1/λ_i)
  weights) is therefore fitting noise on top of the usable subspace signal —
  which is exactly what the V1-vs-V2 comparison shows (quantized 0.934 >
  weighted 0.870).

## 3. Ordered term (`ordered_term.json`)

The old→new direction is real, in BOTH classes: ‖μ̂1‖² = 3.07 vs sign-flip
null mean 0.50 (q95 0.55), p = 1/2001; negatives ‖μ̂0‖² = 0.97 vs null 0.095,
p = 1/2001. Serial order is geometrically visible in these embeddings even
for non-conflicting pairs — a systematic later-minus-earlier drift. G3's
sign-flip condition passes and V3 (held-out) edged V2 (0.8702 vs 0.8695), so
the ordered term stays in the score — but it is worth +0.0007 AUROC: the
mean edit direction adds almost nothing once the subspace energy is known.

## 4. Headline table (sh_64k, everything frozen; `eval.json`)

Balanced = the mandated cosine-matched eval set (1,681/1,681; band = cos
0.87–0.97, 998/1,644). Hard = confirmatory gold vs 39,215 hard negatives.
Tail TPRs at FPR 1e-4 (achievable: n0 = 39,215 ≥ 1e4), seen/unseen
transitions vs the full hard-negative pool.

| method | balanced | band | hard AUROC | hard AUPRC | inv-win | tail seen 1e-4 | tail unseen 1e-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V0 = CES | **0.9756** | **0.9690** | 0.9810 | 0.847 | **0.980** | **0.835** | 0.208 |
| V1 quantized core | 0.9337 | 0.9281 | 0.9361 | 0.611 | 0.960 | 0.274 | 0.009 |
| V2 QDA core | 0.8695 | 0.7972 | 0.9228 | 0.576 | 0.559 | 0.444 | 0.083 |
| V3 = V2 + ordered | 0.8702 | 0.7984 | 0.9232 | 0.578 | 0.560 | 0.448 | 0.083 |
| V4 = V3 + β·norm | 0.8909 | 0.8362 | 0.9757 | 0.881 | 0.496 | 0.597 | 0.355 |
| V5 logistic recal | 0.9105 | 0.8676 | 0.9797 | **0.907** | 0.604 | 0.625 | **0.366** |
| ABTT cosine | 0.9648 | 0.9516 | **0.9991** | 0.979 | 0.927 | 0.507 | 0.404 |
| campaign cosine | 0.8930 | 0.8498 | 0.9920 | 0.916 | 0.000 | 0.372 | 0.214 |

The one genuinely new positive: **the norm term.** log‖d_t‖ (the
ABTT-complement difference length) carries enough signal that V4/V5 beat CES
on hard-task AUPRC (0.881/0.907 vs 0.847) and on the unseen-transition tail
(TPR@1e-4 0.355/0.366 vs CES 0.208; ABTT-cos 0.404) — at matched cosine,
conflicts have systematically *smaller* residual difference norms than
subject swaps (β < 0). It is cheap, label-light, and tail-complementary in
the same direction the nuisance report identified — a candidate third input
for any future fusion attempt. It does not rescue the mandated arena
(V5 balanced 0.9105 still loses to CES and ABTT-cosine).

## 5. Gate verdicts

- **G1 — FAIL** (|0.9337 − 0.9756| = 0.042 > 0.010), after the documented
  fix attempts the gate prescribes. The recorded discrepancy analysis: the
  residual gap is *relation identity*, not broken machinery — the
  apples-to-apples pooled CES scores 0.8725, so the pooled whitened
  quantization V1 is +0.061 *above* its raw-space counterpart. Per the
  gate's fallback, V2's numbers are reported with an "unverified by G1"
  flag; given G2's decisive failure this changes no decision.
- **G2 — FAIL.** The null result above. Reported with CI; no harm to the
  existing pipeline.
- **G3 — PASS** (p = 1/2001 and V3 ≥ V2 held-out), worth +0.0007.
- **G4** — 1e-4 rows are achievable (n0 = 39,215); conformal claims stop at
  α ≥ 1/(1,946+1) = 5.14e-4.
- **G5 — nominal PASS, treated as VACUOUS; Stage 4b is NOT adopted.** No
  relation reached the preregistered n ≥ max(50, 3·k_obj) = 192 gold pairs,
  so the mixture holds ZERO per-relation covariances and reduces to the
  pooled model plus the ordered mean inside a logsumexp over gate
  probabilities. Its "wins" over V2 (+9.1e-5 balanced, +1.4e-4 band,
  +1.8e-4 relation-disjoint) clear the CI half-widths only because the two
  scores are numerically near-identical — the deltas are the ordered term
  wearing a different coat, not relation gating. Adopting it would add a
  36-class midpoint classifier for +0.0001 AUROC. Recorded in
  `relation_gate.json`; the gate arithmetic is preserved, the adoption
  claim is explicitly declined in favor of V3, which contains the same
  information.
- **G6 — nuisance adaptation kept** (trivially: +8.1e-5, CI excludes harm;
  scale ratio 0.863, μ_t norm 0.011 — a near-no-op, as the PREREG's own
  formula predicted for difference vectors). **Coherence fusion REJECTED**:
  see §7.

## 6. Transfer and calibration

**Relation-disjoint (cal 2-fold, whole pipeline refit per fold):** fold 0 →
0.953; fold 1 → **0.020** with k_obj = 0 — the trace-normalized parallel
analysis found no significant spectrum on fold 1's 437 edits and the score
degenerated to a signed norm. The pooled QDA core does not transfer
reliably across relation splits; CES's relation-aware route remains the only
stable one.

**Conformal thresholds (`calibration.json`,** n0_cal = 1,946 fit-side-naive
negatives): V2 −243.5 / −192.6 / −155.9 at α = 0.1 / 0.01 / 0.001 (the
α = 0.001 row already uses the maximum order statistic; the guarantee floor
is α = 5.14e-4). V4: 3278.7 / 4245.3 / 5895.2.

**χ² is unusable at the tail:** KS = 0.628 against χ²₆₄; empirical
1−1e-2 / 1−1e-3 quantiles exceed theory 2.8× / 5.1×. The Gaussian null does
not hold for object-subspace energy (the whitened negatives are far from
isotropic — §2), so parametric p-values below the conformal floor would be
anticonservative by half an order of magnitude. Conformal or nothing.

## 7. Label-free adaptation (`adaptation.json`)

- **Prevalence EM: biased low, badly.** With f0/f1 frozen from sh_32k,
  known-π sh_64k mixtures give π̂ = 0.0003 / 0.001 / 0.033 / 0.208 for
  π = 0.01 / 0.05 / 0.2 / 0.5 — the score distributions shift across
  subsets enough that the frozen-KDE mixture model underestimates
  prevalence 3–30×. Do not use these π̂ (or their Bayes thresholds)
  operationally.
- **Local coherence is INVERTED** (standalone AUROC 0.347; stable across
  seen 0.342 / unseen 0.356 / relation folds 0.354, 0.429). The additive
  model predicted conflicts sit in coherent neighborhoods; empirically the
  hard negatives do — a midpoint neighborhood of subject swaps around one
  fact family shares near-parallel subject-difference directions (and
  shared-fact pairs correlate mechanically), while gold edits are lonely in
  their neighborhoods. Rank-average fusion with V2 costs −0.21 to −0.26
  AUROC on every slice → G6 rejects it. The *signal* is real but its sign
  and framing belong to a different detector (a "crowd veto"), left for
  future work.

## 8. Scientific conclusion

1. **Does the unquantized QDA score beat CES?** No — −0.106 balanced,
   negative in every cosine bin, worse in the band. G2 null result.
2. **Why?** Two estimation failures, both now measured: (a) the negative
   difference distribution is not whitenable from this data — its sample
   eigenbasis does not transfer even across a random half-split of the same
   subsets (half-B median whitened variance 0.0095); (b) the gold
   covariance's individual eigendirections are unidentified (bootstrap
   principal angles ~81–89°), so eigenvalue-keyed weights add noise where
   CES's flat ±1 weights add none. The estimable content of both
   covariances was already captured by ABTT (top common directions) plus
   low-rank subspace energies (CES) — the full-covariance refinement has
   nothing solid left to estimate.
3. **Is serial order geometrically visible?** Yes — p = 1/2001 in both
   classes, a genuinely new fact about these embeddings — but its projected
   score contribution is +0.0007: effectively redundant given subspace
   energy.
4. **Anything worth keeping?** The norm term: log‖d_t‖ makes V4/V5 the best
   hard-task AUPRC in the repo (0.907 vs CES 0.847) and beats CES on the
   unseen-transition tail (0.366 vs 0.208 TPR@1e-4). As a third fusion
   input it is the concrete candidate this campaign leaves behind; as a
   standalone screen it loses the mandated arena. Plus the negative
   calibration knowledge: χ² tails are anticonservative ~3–5× here, and
   frozen-KDE prevalence EM under-reads π by 3–30× across subsets.
5. **Verdict for the pipeline:** keep CES + ABTT-cosine unchanged. The
   `weights.npz` artifact and `score_pairs` API are committed and
   fingerprinted should the norm-term fusion ever be pursued; the conformal
   V2/V4 thresholds above are the only calibrated operating points.

## 9. Pipeline permutation null (merged into `eval.json`)

200 label-shuffle repeats, the WHOLE pipeline refit each time (halving, LW,
whitening, trace-normalized rank selection, weights); inner
parallel-analysis at 50 perms and the null hard task on a fixed seed-7
subsample of 4,000 negatives — the preregistered compute deviations.

- Real balanced sh_64k AUROC 0.8687 (in-harness recompute; headline 0.8695)
  vs null max 0.8209 / q95 0.8190 / mean 0.8160 → **p = 1/201 = 0.005**,
  the add-one minimum. Hard-subsample: real 0.9242 vs null max 0.8932 →
  p = 1/201.
- Two structural reads. First, the learned spectrum is real: 199 of 200
  shuffled refits selected k_obj = 0 (mean 0.115) — rank selection
  correctly finds nothing when the labels are noise — and no shuffled
  pipeline reaches the real AUROC. Second, the null does NOT sit at 0.5: a
  label-shuffled fit degenerates to a signed whitened-norm score, and that
  alone gets ~0.816 balanced AUROC. That elevated floor is an independent,
  label-free confirmation of the §4 norm finding, and it bounds what the
  covariance structure itself contributes: about +0.05 AUROC over the
  norm-only floor — real (p = 0.005), but far short of CES's relation-aware
  subspaces.

## 10. Runtime and memory

Full campaign (everything but the null): 469 s wall on the local CPU box,
peak ~2.5 GB (54,569 × 2,432 float32 differences + whitened copy). The
null: ~60–90 s per repeat. Embedding cache: all 4,499 fact vectors were
already local; no GPU, no LLM, no network. `weights.npz` is 25 MB
(float32, fingerprinted; loading is verified against the manifest by
`hnav/tests/test_qda_filter.py`, which also pins k_obj = 64, k_subj = 0,
the G1/G2 headline AUROCs, p_signflip, and the conformal thresholds).
