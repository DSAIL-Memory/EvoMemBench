# E2E benchmark — geometric identity screens vs the parser pipeline (2026-08-27)

The question this campaign answers: **can the geometry filters (ABTT-cosine, CES) replace
H-Nav's raw-cosine + parser screen in the full pipeline, measured on overall answer accuracy?**
Protocol identical to the committed campaign: screen → bidirectional NLI (0.90, untouched) →
`detector_suppress` → frozen Qwen3-4B-Instruct-2507 on `:8003` → `substring_exact_match`;
one shot per arm per subset; sh_64k held out from every fitting step. Artifacts:
`pipelines/{hnav_ces,hnav_abtt_noparser}/results/Qwen_Qwen3-4B-Instruct-2507_2026-08-27/`,
paired analysis in `e2e_comparison.json` (recomputed from per-question records). All six new
subset runs pass every guard (containment, page-edit integrity, positive control, A/A floor 0).

## Held-out result (sh_64k — the only arena that counts)

| arm | screen | overall /100 | conflicted /66 | net vs native | paired vs parser arm |
| --- | --- | --- | --- | --- | --- |
| native (no detector) | — | 45 | 17 | — | — |
| **hnav_raw** (committed) | raw cos 0.90 + **parser** | **64** | **37** | +20 (p=1.9e-06) | reference |
| **hnav_abtt** (committed) | ABTT cos 0.30 + **parser** | **64** | **37** | +20 | ±0 (identical answers) |
| **hnav_abtt_noparser** (new) | ABTT cos 0.80, no identity screen | 59 | 31 | +14 (p=1.2e-04) | −5, p=0.125 (ns) |
| **hnav_ces** (new) | raw cos 0.80 + CES τ 0.40 (parser relation only) | 55 | 28 | +11 (p=0.0034) | −9, p=0.0039 |

Native outputs are byte-identical across all four sh_64k artifacts (0 disagreements), so the
cross-arm McNemar comparisons are exactly paired.

Calibration side (sh_6k / sh_32k; CES partly in-sample — its artifact and τ were fit there):
hnav_ces conflicted 4→58/74 and 19→44/65; hnav_abtt_noparser 4→28/74 and 19→38/65; the
committed raw parser arm 2→68/74 and 13→52/65. (Calibration baselines come from an earlier
session whose native flags differ by ≤2 questions — only the sh_64k comparison is strictly
paired.)

## Operating points (calibration-only; both preregistered grids amended upward — recorded
verbatim in each `pipeline.json` and the ops' provenance)

| arm | frozen screen | pair precision | pool recall | harmful |
| --- | --- | --- | --- | --- |
| hnav_ces | cos 0.80 + CES τ 0.40 | 1.000 | 0.734 | 0 |
| hnav_abtt_noparser | ABTT-cos 0.80 | 1.000 | 0.444 | 0 |

The original CES τ grid ({−0.05…0.10}) had **no** zero-harm cell with nonzero recall: at the
benchmark's pool prevalence (~0.8% true pairs vs 4% in the gold-dataset hard task), τ 0 gives
precision 0.084 with 2,865 harmful suppressions. The amendment to τ 0.40 — and the noparser
grid's extension to ABTT-cos 0.80 — happened after inspecting calibration detection metrics
only; sh_64k was never touched.

## What the result means

1. **Every arm beats native decisively** — the pipeline's value does not depend on the parser.
2. **The parser screen is still the best identity evidence end-to-end** (37/66). Removing it
   costs 6 conflicted questions with the best cosine screen (31, ns at n=66) and 9 with CES
   (28, significant). The Faz A prediction ("NLI rubber-stamps different-subject pairs; the
   identity screen carries the precision") survives even against the strongest geometry.
3. **CES's calibration advantage did not transfer.** It kept 73% of true pairs at zero harm on
   calibration (vs 44% for pure ABTT-cos) yet answered *fewer* sh_64k questions (28 vs 31).
   This is the transition-boundedness measured at pair level (`REPORT.md` §4/§7: unseen-
   transition AUROC 0.768 for the object-edit side) plus an in-sample τ: on sh_64k, roughly
   half the gold transitions are unseen, and the τ=0.40 operating point tuned on calibration
   pools sheds exactly those. ABTT-cosine, which encodes no transition memory, transferred
   its calibration recall essentially unchanged.
4. **Between the two geometry screens, whitened cosine ≥ CES end-to-end** (net +4 for
   noparser head-to-head, ns) — consistent with the pair-level verdict that ABTT-cosine is
   the strongest transferable pairwise signal, and now confirmed where it matters.
5. Honest labels: `hnav_ces` is a *partial* parser removal (relation template still comes from
   the parser; the parser-free CES-global variant failed its pair-level gate at 0.8725 and was
   never promoted). `hnav_abtt_noparser` is fully parser-free.

**Bottom line for the thesis:** the geometry filters can carry the pipeline without the parser
at a measurable but modest accuracy cost (64 → 59 overall on held-out data for ABTT-cosine;
64 → 55 for CES); they do not beat the parser pipeline. The strongest use of the geometry
evidence remains what the pair-level report concluded: ABTT whitening *inside* the screen
(committed hnav_abtt) and CES as complementary analysis — not as a replacement identity
screen at benchmark prevalence.

## Addendum (2026-08-28): the fusion arm

The CES + ABTT-cosine fusion screen (experiment 5's tail-complementarity result) **FAILS the
preregistered zero-harm rule** — no non-vacuous operating point exists (calibration harm
plateaus at ~305; the vacuous zero-harm cell is pinned in `pipelines/hnav_fusion`). A
user-directed **exploratory relaxed-harm** analysis (thresholds predefined on calibration,
four one-shot sh_64k runs) measured a completely flat curve at **61/100** overall
(+16 vs native, −3 vs the parser arms, best of all geometry arms) — details, protocol and
caveats in `FUSION_TRADEOFF.md`. Final ranking with all evidence: parser 64 >
fusion (relaxed, exploratory) 61 > abtt_noparser 59 > hnav_ces 55 > native 45.

## Reproducibility

Operating points: `ces_operating_point.json` (sha256 `1b96d100…`),
`abtt_noparser_operating_point.json` (`c656a826…`), pinned in the pipeline specs in the same
commit. CES artifact `ces_subspaces_k20.json` (fingerprint `34e3abc1…`, fit on sh_6k+sh_32k
gold only). Prepasses: `_ces` suffix, raw space, cos_loose 0.80 (sh_64k: 86,753 loose pairs,
93,466 directed NLI scores, 0 embedding-cache misses). Answering model, harness, grader,
mechanism: identical to the committed campaign.
