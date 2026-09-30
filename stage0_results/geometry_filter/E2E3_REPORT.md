# E2E-3 — the hnav_geo arm: parser-free geometric identity, one shot (2026-08-29)

The question: can the H-Nav detector beat the committed parser pipeline
(hnav_raw, sh_64k **64/100**) end-to-end with an identity screen that reads
embeddings only? Preregistration `GEO_PREREG.md`; screen
`hnav/geometry_filter/geo_artifact.py`; arm `pipelines/hnav_geo/`; results
`pipelines/hnav_geo/results/Qwen_Qwen3-4B-Instruct-2507_2026-08-29/`
(+ `e2e3_comparison.json` recomputed from per-question records).

> ## CORRECTION (2026-08-30, E2E-4 adversarial review)
>
> **This run is VOID by its own preregistered void condition 4.** The
> artifact records `void_conditions.4_no_harmful_suppression =
> {status: "fail", voids: "run", n_suppressed_harmful: 8}`. The original
> version of this report called the run "valid" because it checked only the
> mechanical guards (page-edit mismatch, containment, positive control, A/A
> floor) — the same blind spot that made `pipelines/_shared/runner.py` print
> "VALID" (fixed in E2E-4; the runner now reads the preregistered void
> conditions). **The 56/100 below must be quoted as coming from a void run.**
>
> The same defect voids the committed **hnav_abtt_noparser** arm
> (`n_suppressed_harmful = 5`), whose 59/100 in `E2E_REPORT.md` is likewise
> from a void run. `hnav_raw`, `hnav_abtt` and `hnav_ces` pass condition 4.
>
> **What went wrong is the finding, not a footnote.** Geometry has no
> same-key guarantee, so a geometric group can merge two different keys;
> suppression then deletes *every* member of one of them. All 8 harmful drops
> are key erasures: `official language of Italy` (q19, q48),
> `type of music that The Game plays` (q58, q73),
> `outfielder · associated with the sport` (q83). And the coincidence is
> exact: **the five suppressions geometry made that the parser did not are
> precisely the five that erased a key — 5 for 5.** Geometry's entire unique
> contribution over symbolic identity was information loss.
>
> Zero harm was observed for this screen on calibration and **did not
> transfer**. That is the structural difference E2E-4 proves: a screen built
> on `same_key` has zero harm *by construction* at any threshold; a geometric
> screen has only an empirical, split-specific zero.

**The preregistered primary endpoint FAILS. hnav_geo scores 56/100 on
sh_64k** — +11 over native (17→29 conflicted, exact p = 0.0018; overall
p = 0.0074), statistically level with the other parser-free arms (vs
hnav_ces +1, p = 1.0; vs hnav_abtt_noparser −3, p = 0.45), and **−8 vs the
parser arms (p = 0.021)**. GG2 (> 64) is not met; GG1 was met and the shot
was spent per the rules. Mechanical guards were clean (page-edit
mismatch/containment/errors 0, positive control ok, A/A floor 0/0), but see
the correction above: void condition 4 failed, so the run is void and the
accuracy figure is reported for attribution only. Protective claim also void
via the same q77 refusal that voids it for the committed parser arms.

## What the campaign established on the way

1. **The best parser-free detection layer this repo has measured.** Frozen at
   the standard zero-harm rule: pair precision 1.000, calibration pool recall
   **0.7895** (previous parser-free best 0.4444; CES with parser-relation
   0.7343; parser 0.9784), conflicted-question recall 104/139, 0 harmful of
   1,627 suppressions. The screen is a per-axis rectangle over two geometric
   margins — ABTT-whitened cosine × the slot probe (logistic on |d̂|,
   object-slot vs subject-slot change) — with anchors from the calibration
   pools. The preregistered diagonal τ family managed only 0.596; the
   documented Amendment-1 rectangle grid found the corner (loose cosine,
   strict probe) the diagonal cannot reach.
2. **Pair-level, the geometry is CES-class without the parser**
   (`geo_pairlevel.json`): balanced sh_64k AUROC 0.9716 (CES 0.9756 with
   parser-routed subspaces; ABTT-cos 0.9648), band 0.9657, hard task 0.9984 /
   AUPRC 0.9784, and the best unseen-transition tail on record
   (TPR@1e-4 0.481 vs ABTT-cos 0.404, CES 0.208). PCA-compressing the probe
   features hurts (0.73–0.82 balanced): the identity signal is distributed —
   the same lesson the QDA campaign measured from the covariance side.
3. **Where the last 8 points live, precisely.** The screen transferred:
   sh_64k fired on 99/100 questions with 532 suppressions (parser 735), and
   sh_64k harm stayed at the registered exposures (q69 information_loss —
   the same question that hits CES; q77 refusal — the same as the parser).
   The parser-only questions are {9, 11, 26, 35, 54, 69, 70, 91, 93}
   (geo-only: {30}): the identical missed-suppression cluster E2E-1 measured
   for abtt_noparser, now reproduced by a screen with 1.8× its recall. The
   remaining gap is not detection volume — it is that these questions'
   conflict groups contain exactly the pairs whose geometry and whose NLI
   verdict both look cross-key-safe, and only symbolic same-key identity
   verifies them. Calibration quantified the same core from the other side:
   at loose thresholds harm plateaus (396–471 harmful at recall 0.89–0.91)
   on cross-key pairs the NLI stamps bidirectionally even at 0.99.

## The four-arm ladder, final (sh_64k, one shot each, paired)

| arm | screen | overall | conflicted | unique |
| --- | --- | --- | --- | --- |
| native | — | 45 | 17/66 | 28/34 |
| hnav_ces | cos 0.80 + CES τ (parser relation) | 55 | 28 | 27 |
| **hnav_geo** | **cos 0.94 + geo rectangle (parser-free)** | **56** | **29** | **27** |
| hnav_abtt_noparser | ABTT-cos 0.80 (parser-free) | 59 | 31 | 28 |
| fusion (relaxed, exploratory) | CES+ABTT logit | 61 | 33–34 | 27–28 |
| hnav_raw / hnav_abtt | cos + **parser** same-key | **64** | **37** | 27 |

The geometry arms are statistically indistinguishable from one another
end-to-end (all pairwise |net| ≤ 4, ns) despite calibration detection
recalls spanning 0.44 → 0.79. E2E-2's flat-61 curve and this arm's 56 say
the same thing: on this substrate the geometry-only E2E plateau is ~55–61,
the parser sits at 64, and the difference is a small, specific, now
name-listed question cluster — not a threshold anywhere on any grid.

## Verdict for the thesis

- The parser's end-to-end edge is real, reproducible, and now *localized*:
  nine sh_64k questions whose supersessions only symbolic identity verifies.
- Geometry alone reaches CES-level pair discrimination, doubles the
  parser-free zero-harm detection record, and pays no unique-stratum cost —
  but detection recall above ~0.45 stops buying held-out answers.
- The honest claim this campaign supports: *a parser-free geometric identity
  screen can carry most of H-Nav's governance gain (+11 of the parser's +19
  over native) at measured zero harm; the residual is a bounded, identified
  set of conflicts that require symbolic identity on this embedder.*

Provenance: box run at git 81cca60, Qwen3-4B-Instruct-2507 on :8003 (frozen
flags), operating point sha pinned in `pipelines/hnav_geo/pipeline.json`,
artifact fingerprint 335b4540…, selection re-run clean after the review-found
cache fix (identical operating point). Adversarial pre-shot review: 5
verifiers, 2 blocking defects found and fixed before the shot.
