# Repository analysis — geometry-filter experiment series (2026-08-26)

Written before the experiment implementation, per protocol: the pipeline as it
actually exists in code, what is reusable, what was missing, and the risks that
shaped the experiment design.

## A. The real pipeline

```
raw data          In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench, dataset
                  Conflict_Resolution, subsets factconsolidation_sh_{6k,32k,64k}.
                  Numbered fact lists; later serial supersedes same subject+relation.
fact parsing      hnav/labeling/conflict_analysis.py::parse  (validated 99.5%+,
                  hard invariant: import, never rewrite). Emits (relation
                  template, subject, object) triples — S/R/O IS explicitly
                  available for 99.86% of audited pairs (77/54,569 unparsed).
pair construction hnav/labeling/export_conflict_pairs.py (2,682 parser-tagged
                  conflicts) and export_audit_candidates.py (all 87,102 pairs at
                  campaign-embedding cos >= 0.80 — the selection frame).
labeling          THREE independent sources, deliberately kept apart:
                  (1) parser same-key rule; (2) GPT-5-mini judge with per-pair
                  alignment flags (audit_runner.py, 54,569 verdicts);
                  (3) benchmark serial structure (gold-is-latest).
                  build_gold_conflict_dataset.py joins them into tiers:
                  core 2,388 / update_only_fork 282 / rejected 12 /
                  discovered_unverified 105 (quarantined) / negative 51,782.
                  gold_update is a *convention-based* label (later serial
                  supersedes same key); gold_strict is the logical-incompatibility
                  label — the dataset already separates "slot changed" from
                  "values cannot coexist", exactly the distinction §19 of the
                  experiment brief demands.
embedding         Qwen/Qwen3-Embedding-4B, float32, L8192, unit-normalized,
                  dim 2560. Cache: hnav/_cache/emb/, key sha256(namespace||text).
                  Verified: 100% of the 4,499 distinct gold-dataset facts hit.
geometric layer   hnav/core/geometry.py  — sim_max, QR-residual novelty,
                  ABTTWhitening (mean + top-D principal directions removed,
                  renormalize; refuses to fit under 200 rows; persisted with a
                  sha256 fingerprint), TauPolicy.
                  hnav/core/diff_geometry.py — whole_blob_sim / diff_sim /
                  diff_novelty on parsed object spans (span-level, not
                  difference-vector-level).
decision layer    hnav/core/read_gate.py — cosine-edge grouping + leave-one-out
                  QR residual + bidirectional NLI. NOTE: on an isolated pair the
                  geometric stage reduces to a monotone function of pairwise
                  cosine (LOO residual of a 2-set = sqrt(1 - cos^2)), so "the
                  existing H-Nav geometry" baseline for pairwise conflict
                  detection IS cosine. There is no committed geometry score that
                  is not cosine-derived at the pair level.
evaluation        benchmark: substring_exact_match. For the filter: the balanced
                  cosine-matched eval set inside the gold dataset, with the
                  mandated cosine_only_auc baseline (0.96 / 0.91 / 0.89 for
                  sh_6k / sh_32k / sh_64k) and the ~0.87–0.97 overlap band rule.
splits            calibration = sh_6k + sh_32k; confirmatory = sh_64k. Frozen
                  campaign-wide; every record carries `split`.
```

## B. Reusable components (used as-is)

- `stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz` — pairs,
  S/R/O metadata, dual labels, tiers, balanced eval flags, campaign cosine.
- `hnav/_cache/emb/` — campaign embeddings (no re-embedding anywhere).
- `stage0_results/abtt/abtt_whitening_D128.json` — frozen ABTT (D=128, fit on
  the 2,765 calibration facts only), loaded via
  `hnav.core.geometry.ABTTWhitening.from_dict` with fingerprint verification.
  Its mean vector doubles as the calibration-fit mean for the "centered" space.
- Prior evidence: the scratchpad difference-vector analysis (same-transition
  cross-subject cos 0.86, reversed transitions −0.72, low-rank per-relation
  spectra, global mean direction ~8× the random null) — the observations the
  null-control experiment here re-tests properly.
- ABTT A/B campaign (`pipelines/README.md`): ABTT whitening improved the
  read-time conflict detector 17/66 → 37/66 — prior evidence that removing
  dominant directions did NOT destroy conflict-relevant geometry.

## C. What was missing (implemented in `hnav/geometry_filter/`)

RCED / RCESP, sign-invariant scoring for unordered pairs, the shuffled-transition
and random-vector nulls with bootstrap/permutation machinery, the slot-probe,
the cosine-inverted hard-negative comparison, relation/transition/subject-
disjoint evaluations, and machine-readable experiment logging with provenance
(git commit, dataset sha256, embedding namespace, seed). No sign-fingerprint
experiments existed anywhere; none are introduced as a method (per brief §12).

## D. Risks identified and how the design answers them

| risk | status |
| --- | --- |
| train/test leakage | all fitting (mu_r, U_r, k, thresholds, LDA) on calibration only; confirmatory scored once, frozen. Calibration metrics labeled `in_sample`. |
| transition reuse between fit and eval | 686/1,302 confirmatory gold transitions also occur in calibration → explicit transition-disjoint eval + transition-deduped refit ablation. |
| relation leakage | all 37 relations appear in both splits → cal→conf is NOT relation-disjoint; a dedicated 2-fold relation-disjoint protocol runs within calibration, global variants only. |
| template leakage | positives share the relation template with hard negatives (same relation, different subject) by construction of the hard-negative class — the probe's relation-disjoint arm measures the residual template effect. |
| parser labels mistaken for ground truth | conflict labels are the tiered dual labels (judge + parser + convention); slot labels use the parser only for the slot question, where string comparison is exact. gold_update is called what it is: a convention label. |
| class imbalance | balanced eval set for the headline; AUPRC everywhere; hard-negative task reported with explicit n_pos/n_neg. |
| duplicated pairs | pair_ids are unique (asserted in test_gold_conflict_dataset); facts repeat across pairs by design (bootstrap is over pairs — stated limitation: pair-level resampling treats shared facts as independent). |
| directional pair reversal | detection scores are sign-invariant (`\|d_hat·mu\|`, subspace norm). Signed probe features get a deterministic random sign — kept as a designed chance-level control. |
| normalization inconsistency | one convention: every space re-normalizes to unit vectors; mean-centering *without* renormalization is a no-op on differences (asserted in tests). |
| cached embeddings with different preprocessing | single cache namespace, asserted against the ABTT artifact's recorded namespace. |
| cosine-matched eval still carries cosine signal | acknowledged in the dataset summary; the decisive statistic here is the cosine-inverted win rate, where cosine scores 0 by construction. |
