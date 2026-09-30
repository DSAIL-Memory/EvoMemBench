# Fusion trade-off — EXPLORATORY, relaxed-harm (2026-08-27/28)

**This is NOT a preregistered result.** The preregistered outcome of the fusion arm is a
**FAIL**: under the campaign's zero-harm selection rule the fusion screen has no non-vacuous
operating point — calibration harmful suppressions plateau at ~305 across the whole tau grid
(517/359/321/312/305 at tau 0/2/4/6/8 while recall falls only 0.93→0.87), an irreducible
high-ABTT-cosine cross-key core that the NLI verifies even at 0.99. That FAIL is pinned in
`pipelines/hnav_fusion/pipeline.json` and its vacuous operating point is the frozen artifact.

What follows is the user-directed exploratory question (2026-08-27): *if a known amount of
calibration harm is accepted, does the fusion screen give a net gain in downstream overall
accuracy?* Protocol: four thresholds predefined from the calibration grid only (no held-out
selection anywhere); one sh_64k run per threshold via the unchanged pipeline (`detector_gap
--confirmatory`, `--operating-point` override announced in-run; frozen Qwen3-4B on `:8003`;
same NLI, same `detector_suppress`, same grader). Artifacts:
`fusion_tradeoff/fusion_tradeoff_tau{0,2,4,6}_sh64k.json` (+ `tradeoff_summary.json`).
Native answers are byte-identical across all four runs and across the committed arms, so
every comparison below is exactly question-paired. All guards clean in all four runs
(containment / page-edit / positive control / A/A floor 0).

## The trade-off curve (held-out sh_64k)

| τ (logit) | cal recall | cal harmful | sh_64k suppressed | **overall /100** | Δ vs native | conflicted /66 | unique /34 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.931 | 517 | 1,039 | **61** | +16 (p=8.6e-4) | 33 | 28 |
| 2 | 0.928 | 359 | 844 | **61** | +16 | 34 | 27 |
| 4 | 0.911 | 321 | 775 | **61** | +16 | 34 | 27 |
| 6 | 0.882 | 312 | 695 | **61** | +16 | 34 | 27 |

Reference points (same questions, same model): native 45 · **hnav_raw/hnav_abtt (parser) 64**
· abtt_noparser 59 · hnav_ces 55.

Paired McNemar, fusion τ=2 (the pre-named candidate) vs the others on sh_64k:
vs parser arm **−3** (0/3 discordant, p=0.25); vs abtt_noparser **+2** (6/4, p=0.75);
vs hnav_ces **+6** (9/3, p=0.15).

## What the curve says

1. **The answer to the exploratory question is yes**: accepting calibration harm, the fusion
   screen yields 61/100 — the best geometric arm measured (vs 59 parser-free ABTT, 55 CES),
   +16 over native (significant), and within 3 questions of the parser pipeline (n.s. at
   n=100, but consistently one-sided: the parser arm answers 3 questions fusion misses,
   fusion answers none the parser misses).
2. **The curve is completely flat.** From 1,039 suppressions (τ=0) down to 695 (τ=6),
   overall accuracy never moves; the unique stratum loses at most one question anywhere on
   the curve. The extra suppressions the looser thresholds make — including the ones the
   calibration harm counter flags — are overwhelmingly *answer-neutral* on held-out data.
3. **The zero-harm calibration proxy is far more conservative than realized cost.** 312–517
   flagged harmful suppressions on calibration translate to ≤1 lost unique-stratum answer on
   sh_64k. The proxy protects against information loss per fact; the benchmark only pays for
   information loss on asked questions. This gap is why the preregistered rule and the
   exploratory result diverge — and the rule is not being changed retroactively: the FAIL
   stands as the preregistered result, this curve stands as exploratory evidence.
4. **Ranking with all evidence in:** parser 64 > fusion (relaxed) 61 > abtt_noparser 59 >
   hnav_ces 55 > native 45. Geometry closes most of the parser gap only when allowed to
   trade calibration-harm guarantees for recall; under the do-no-harm rule the parser's
   exact key predicate remains unbeaten.

## Provenance

Runs 2026-08-27T15:28–19:58Z, `hnav/deploy/run_fusion_tradeoff.sh`, operating points
`fusion_exploratory_tau{0,2,4,6}_op.json` (each stamped `EXPLORATORY_RELAXED_HARM`, thresholds
from the calibration selection grid of commit `7ad4581`), fusion artifact `2ffb0c85…`
(pins CES `34e3abc1…` + ABTT whitening `3fdacc1f…`).
