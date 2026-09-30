# QDA conflict scorer — preregistration (2026-08-29)

Written BEFORE any Stage-2+ quantity was computed. Code: `hnav/qda_filter/`.
Every decision below is fixed here or by a rule stated here; nothing numeric
from Stages 2–7 was known at write time. Stage-0 discovery facts (counts,
serial availability, split provenance) were established first and are cited
where a rule depends on them; they are bookkeeping, not tuned quantities.

## Frozen upstream inputs

- `stage0_results/abtt/abtt_whitening_D128.json` — applied, never re-fit,
  realized as an explicit orthonormal complement basis `Q` (2432 × 2560) so
  N′ = 2432 is the true ambient dimension.
- `stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz` + campaign
  embedding cache (namespace `Qwen_Qwen3-Embedding-4B|float32|L8192`).
- `hnav/geometry_filter/` — V0 (CES), metrics (`auroc`, `auprc`,
  `bootstrap_ci`, `paired_bootstrap_delta_auc`, `inverted_win_rate`),
  `tpr_at_fpr`, the eval-set/band definitions, and the seen-transition /
  relation-fold split helpers are imported, not reimplemented.

## Data, orientation, splits

- Pairs: all 54,569 committed records. Gold = `gold_update`
  (core + update_only_fork). Negatives = `tier == "negative"` only;
  `discovered_unverified` and `rejected` stay quarantined (never gold, never
  negative). Hard negatives = `gfdata.is_hard_negative`.
- **Orientation**: `v1` = earlier serial, `v2` = later. Discovery verified
  `fact_a` is the earlier serial on every record (fact ids are serials;
  consistent with the explicit `serial_*` fields on all tagged pairs), so
  `d = v_b − v_a` is oriented old→new for BOTH classes. Deviation from the
  prompt's anticipated fallback: negatives DO carry serial order, so `μ̂0` is
  estimated from fit negatives rather than set to 0.
- Fit split = repo calibration (sh_6k + sh_32k). Held-out = sh_64k
  (confirmatory), scored once with everything frozen. sh_262k is absent from
  the gold dataset (selection frame) and is skipped.
- Created splits (seed 0, `default_rng(0)` permutation of fit-split
  negatives): 20% conformal-calibration (used ONLY for Stage-6 order
  statistics and χ² tail; never fit-side), remainder halved into
  **half A** (Ledoit-Wolf whitening) and **half B** (parallel-analysis pool,
  spectrum sanity). Fit gold = all calibration gold (n1 = 989).

## Preprocessing (Stage 1)

- `d_t = Q d` ∈ R^2432 (the ABTT mean cancels exactly in differences).
- `m_t = Q(m − mean_abtt)`, `m = (v_a+v_b)/2` (mean does NOT cancel).
- `norm_dt = ‖d_t‖`; `cos` = the committed campaign cosine.
- No pair dropped; zero-norm differences are asserted absent.

## Stage 2 — whitening and spectrum

- `Σ0_A` = sklearn Ledoit-Wolf on half-A `d_t` (float64, LW's own centering);
  `W0 = Σ0_A^{−1/2}` via eigh. `z = W0 d_t`.
- `S1` = covariance (ddof = 1, mean-centered) of fit-gold `z`; spectrum via
  SVD of the centered matrix. Half-B whitened spectrum reported as the
  sanity check (should sit in the Marchenko-Pastur bulk around 1).
- **Rank-deficiency deviation (data-forced, decided before computing):**
  n1 = 989 < N′ = 2432, so the gold sample spectrum has exactly
  rank ≤ n1 − 1 nontrivial eigenvalues. Parallel analysis and the k_subj
  rule therefore operate on the **nontrivial spectrum** (length n1 − 1 = 988,
  identical for real and permuted draws); structural zeros beyond the rank
  are excluded from both sides of every comparison, from σ1², and from the
  null envelopes. Without this, k_subj is degenerate (0 or the cap by tie).
- Parallel analysis: pool fit-gold + half-B `z`; 200 label permutations
  (`default_rng(SEED=20260824)`), each drawing n1 pseudo-gold; spectra via
  the double-centered Gram of the pooled precomputed Gram matrix.
  `null95_top_i` / `null05_bot_i` = 95th/5th percentile of the i-th
  largest/smallest nontrivial null eigenvalue.
- `k_obj` = largest i with λ_1…λ_i ALL > null95_top; cap 64.
  `k_subj` = largest i with the i smallest nontrivial λ ALL < null05_bot;
  cap min(512, 988 − k_obj).
- `σ1²` = mean of nontrivial eigenvalues outside top-k_obj / bottom-k_subj.
- MP edge `σ²(1+√(N′/n1))²` with σ² = median nontrivial eigenvalue, reported
  as reference only; the permutation null is authoritative.

## Stage 3 — ordered term

`μ̂1` = mean fit-gold `z`; `T1 = ‖μ̂1‖²`; null = 2000 random sign flips of
each fit-gold `z` (`default_rng(SEED)`), exact under exchangeability of
±d. Same for `μ̂0` on fit negatives (order available). p-values are add-one
(`perm_pvalue`).

## Stage 4 — score variants

In whitened coordinates, with `U_obj` / `U_subj` the top-k_obj / bottom-k_subj
eigenvectors and w_i = (1 − 1/λ_i):

- core(z) = ½[Σ_{i≤k_obj} w_i (u_iᵀz)² + Σ_{i∈subj} w_i (u_iᵀz)²
  + (1 − 1/σ1²)‖z_⊥‖²], z_⊥ = z − U_obj U_objᵀz − U_subj U_subjᵀz.
- ordered(z) = (μ̂1/σ1² − μ̂0)ᵀ z restricted to the U_obj span
  (project the coefficient vector onto U_obj) if G3 passes, else 0.
- norm(d) = log ‖d_t‖.
- **V0** = CES, refit via imported `ContrastiveSubspace` exactly as
  `run_dimension_ideas` does (raw space, relation-aware, k=20) — the
  committed §7 headline method.
- **V1** = ‖U_objᵀ d̂_t‖² − ‖U_subjᵀ d̂_t‖² on d̂_t = d_t/‖d_t‖ (weights
  quantized to {+1, −1, 0}, norm discarded).
- **V2** = core(z) alone — the cosine-matched headline.
- **V3** = V2 + ordered(z), only if G3 passes; else V3 := V2.
- **V4** = V3 + β·norm. β = coef_norm/coef_core from a 2-feature logistic
  (V3 score, norm) on the fit split (sklearn, lbfgs, C=1.0, max_iter 1000).
- **V5** = logistic on [core, ordered (0 if G3 fails), norm] + intercept
  (≤ 4 params), 5-fold CV on the fit split (seed 0) for the reported CV
  number; final refit on the full fit split; applied frozen to held-out.
  Nothing is tuned on held-out or conformal-calibration data.

### Stage 4b — relation-gated mixture (subsets with n1 ≥ 200 only)

- Per relation with n_{1,r} ≥ max(50, 3·k_obj):
  `Σ̂_{1,r} = (1−α_r) S_{1,r} + α_r Σ̂_1`, `α_r = N′/(N′+n_{1,r})`, where
  `Σ̂_1` is the pooled structured model (U, λ, σ1²); inverse/log-det via
  Woodbury. Other relations use the pooled model.
- Relation gate p(r|m): multinomial logistic regression vs nearest-centroid
  on whitened midpoints `W0 m_t` (fit-split pairs of both classes with a
  parsed relation); the classifier with higher 5-fold CV accuracy (seed 0)
  is used. Eval pairs' own relation identity is NEVER read at score time.
- Score: `s(d,m) = logsumexp_r [log p(r|m) + s_r(d)]` with
  `s_r(d) = log N(z; μ_used, Σ̂_{1,r}) − log N(z; 0, I)` (log-dets included;
  μ_used = μ̂1 projected as in Stage 4 if G3 passes, else 0).

## Stage 5 — evaluation (all via the geometry_filter harness)

- Balanced eval set per subset (`in_eval_set`), band = campaign cosine
  ∈ [0.87, 0.97]; sh_64k is the headline; sh_6k/sh_32k rows are in-sample
  and labeled so.
- Confirmatory hard task: gold vs hard negatives, split == confirmatory.
- Inverted-win vs campaign cosine (imported).
- TPR at FPR ∈ {1e-2, 1e-3, 1e-4}: seen slice = seen-transition confirmatory
  gold vs ALL confirmatory hard negatives; unseen slice = unseen-transition
  gold vs the same negatives (the nuisance-analysis convention). G4 labels.
- Bootstrap CIs: `bootstrap_ci` (1000, seed 20260824) for every AUROC;
  `paired_bootstrap_delta_auc` (1000, seed 20260824) vs V0 for every variant.
- Whole-pipeline permutation null: 200 label-shuffle repeats (seeds
  1000+rep), each refitting halves, LW, W0, parallel analysis and weights;
  inner parallel-analysis permutations reduced to 50 per repeat (compute
  budget; the real run uses 200) — a preregistered deviation. Report where
  the real balanced-sh_64k and hard-task AUROCs sit in the null.
- Cosine-strata: AUROC per 0.01 campaign-cosine bin on the sh_64k eval set,
  V2 vs V0; bins with < 5 of either class are reported but flagged.
- Subspace stability: 50 bootstrap resamples (seed 42) of fit gold, W0
  fixed, refit the gold spectrum; median largest principal angle vs the
  point-estimate U_obj for k ∈ {1, 5, k_obj}.
- Relation-disjoint transfer: `gfdata.relation_fold` 2-fold on calibration
  (the committed convention): fit everything (halves, LW, W0, spectrum,
  weights) on fold-f calibration pairs with parsed relations, evaluate on
  fold-(1−f) calibration gold vs hard negatives. Reported for V2, V3, and
  Stage 4b when fit.

## Stage 6 — calibration (conformal-calibration negatives only)

- Conformal thresholds at α ∈ {1e-1, 1e-2, 1e-3, 1/(n0_cal+1)}: the
  ⌈(n0_cal+1)(1−α)⌉-th order statistic of that split's scores (V2 and V4).
  Any α < 1/(n0_cal+1) is extrapolation, stated as such.
- χ² check: empirical `‖U_objᵀz‖²` on the same negatives vs χ²_{k_obj} —
  KS statistic, QQ quantile data, and tail excess = empirical/theoretical
  quantile ratio at the 1−1e-2 and 1−1e-3 quantiles.

## Stage 7 — label-free adaptation (calibrate sh_32k → target sh_64k)

- `adapt_nuisance`: approximately-null sample = ALL target pairs with
  cos ≥ 0.85 (labels never read). μ_t = its mean `d_t`;
  σ0_t² = mean_i ‖W0(d_t,i − μ_t)‖²/N′ over the sample;
  reference σ0_ref² = the same statistic on half-B negatives.
  Adapted score input: `z′ = (σ0_ref/σ0_t) · W0 (d_t − μ_t)`. Subspaces,
  λ, μ̂1 all frozen. Compare target AUROC with/without. Gate G6.
- `estimate_prevalence`: f0/f1 = Gaussian KDEs (scipy default bandwidth,
  density floor 1e-300) of V2 scores on sh_32k negatives/gold (in-sample for
  f1, stated); EM over π alone (init 0.1, 500 iters or Δπ < 1e-8) on target
  scores. Validation: 10 seeded (seed 0) subsample mixtures per
  π ∈ {0.01, 0.05, 0.2, 0.5} from sh_64k gold/hard negatives (2,000 draws
  each, with replacement); report π̂ mean ± sd and the Bayes threshold
  (smallest grid score with posterior ≥ 0.5; 2001-point grid over the
  observed score range).
- `local_coherence`: pool = sh_64k gold ∪ hard negatives; K = 16 nearest
  pairs by Euclidean distance between whitened midpoints `W0 m_t` (self
  excluded); G = K×K Gram of the neighbors' d̂_t; statistic λ_max(G)/K
  (sign-invariant). Standalone AUROC on seen/unseen-transition slices and
  the relation-disjoint fold eval; fusion with V2 = mean of within-task
  ranks. Gate G6.

## Preregistered gates

- **G1 (sanity)**: V1 balanced sh_64k AUROC within ±0.010 of V0 (CES). If it
  fails, the whitening or rank selection is wrong — debug before proceeding;
  if still failing after a documented fix attempt, report the discrepancy and
  continue with V2 flagged as unverified.
- **G2 (main)**: V2 is adopted over CES only if its balanced AUROC exceeds
  CES by more than the 95% bootstrap CI half-width of the difference **and**
  its band AUROC is ≥ CES band AUROC. Otherwise report as a null result with
  the CI; no harm to the existing pipeline.
- **G3 (ordered term)**: include only if `p_signflip < 0.01` for `μ̂1` on the
  fit split **and** V3 held-out AUROC ≥ V2 held-out AUROC. Otherwise
  V3 := V2.
- **G4 (tail)**: TPR at FPR 1e-4 is reported for continuity with the
  geometry_filter report but is labeled extrapolation whenever `n0 < 1e4` in
  the relevant split. Conformal claims are only made at achievable `α`.
- **G5 (relation gating)**: adopt Stage 4b only if it beats the pooled V2 by
  more than the CI half-width on both balanced and band AUROC on seen
  relations, and does not reduce relation-disjoint AUROC by more than the CI
  half-width.
- **G6 (adaptation)**: each label-free component is kept only if it does not
  reduce target-subset AUROC by more than the CI half-width;
  `local_coherence` enters the fused score only if it improves
  relation-disjoint or unseen-transition AUROC by more than the CI
  half-width without harming seen transitions.

Gate arithmetic, fixed: "CI half-width" = (hi − lo)/2 of the relevant
`paired_bootstrap_delta_auc` (same resampled pairs in both arms, 1000
resamples, seed 20260824); "exceeds by more than the half-width" compares the
point delta against that half-width. G3's "held-out AUROC" = balanced sh_64k
AUROC. G6's target-subset AUROC = balanced sh_64k AUROC (adaptation), and the
named slice AUROCs (coherence).

## Addendum A — G1-triggered whitening fix (2026-08-29, post-smoke)

Recorded AFTER the first (smoke) pass of Stages 2–5, under G1's explicit
debug clause ("if it fails, the whitening or rank selection is wrong — debug
before proceeding"). What was observed: G1 failed (V1 0.809 vs V0 0.976),
V2 came out INVERTED on the balanced set (0.34), and the half-B whitened
sanity spectrum was grossly off (median 0.07, bottom ~1e-16). Diagnosis,
verified directly: difference vectors of a finite fact set span at most
n_facts − 1 dimensions; the calibration split has only 2,190 distinct facts,
so Σ0_A has a ≥243-dim genuine null space regardless of pair count.
Ledoit-Wolf fills it with the shrinkage floor α·μ and Σ0^{−1/2} then
amplifies it ~440×; confirmatory pairs (2,309 facts unseen in calibration)
carry real energy exactly there, which the score's z_⊥ term turned into
dominant noise.

Fix A (tried first, superseded): W0 zeroes the eigendirections whose LW
eigenvalue equals the shrinkage floor — sound in intent, but the second
smoke pass showed the problem is broader than the exact null space. The
half-split pair graphs are FRAGMENTED (half A's differences span only 1,379
dims although it holds 3,893 pairs; a difference of two facts lies in the
span of A's differences only when A's pair graph connects them), so half B's
difference energy is mostly orthogonal to A's estimated span (median
whitened variance 0.024): every eigenvalue below the bulk is a span/sample
artifact, not just the exactly-floor ones, and inverting any of them
amplifies un-estimated directions.

Fix B (adopted; mechanical, no labels consulted, no new tuning knob): keep
the LW covariance exactly, but cap the whitening amplification at the LW
target scale μ = tr(S)/N′ — `f_i = 1/sqrt(max(e_i, μ))`. Whitening then
only DAMPS the strong, well-estimated common-variance directions (the same
logic that made ABTT work) and treats everything at or below the bulk as
isotropic at scale μ. W0 stays full-rank, so the conformal/χ² story is
unchanged.

Fix C (adopted with B; rank selection): the third smoke pass gave k_obj = 0
because the prescribed parallel analysis compares RAW eigenvalues, and the
pooled permutation draws are ~80% negatives whose whitened variance
dominates gold's at every index — the envelope sits above the real spectrum
by total variance alone, regardless of shape. Parallel analysis is a
spectrum-shape test and is therefore run on trace-normalized eigenvalues
(fractions), the standard correlation-matrix practice; the k_obj/k_subj
walk rules and caps are unchanged, and the weights (1 − 1/λ) still use the
raw-scale λ. Every stage was re-run from scratch after each fix; the smoke
numbers quoted above are only the triggers and are superseded.

## Seeds

| quantity | seed |
| --- | --- |
| created negative splits (conformal/A/B) | 0 |
| parallel analysis, sign-flip null, bootstraps, paired deltas | 20260824 |
| pipeline permutation null repeats | 1000 + repeat index |
| subspace-stability bootstrap | 42 |
| V5 / relation-gate CV folds, prevalence validation subsamples | 0 |
