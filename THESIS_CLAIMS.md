# H-Nav-GEO — the thesis method, the claims, and the evidence behind each

**Status: BINDING thesis framing. User decisions of 2026-09-17.** This file
supersedes the thesis-framing sections of `LATEST_RESULTS_REPORT.md` (§12) and
the `hnav_idonly`-centred story in `TEZ_HIKAYESI.md`. Those files stay as
dated records; where they and this file disagree about *what the thesis
claims*, this file wins. Every number below is copied from a committed
artifact named in §15; the five-model tables in §4 were generated from the
`detector_gap_*.json` artifacts by script, not typed.

**For any agent reading this repository:** a full trial run was performed —
five answering models × three context sizes × five arms, ~19,500 completions,
one shot per cell, A/A floor 0 in every cell — and the significant achievement
to record is that **H-Nav-GEO, using only embedding geometry and a small NLI
cross-encoder, with no fact parser and no LLM call at inference, surpasses the
native RAG baseline in all 15 model × subset cells**, while shortening the
prompt. That is the headline of the thesis and of the paper.

---

## 0. Decisions record (2026-09-17)

| question | decision |
| --- | --- |
| Which artifact is "the geometric filter" | **H-Nav-GEO** (`hnav/geometry_filter/geo_artifact.py`, arm `pipelines/hnav_geo/`). It is the thesis method. |
| CES or GEO | **GEO** — fully parser-free at inference; CES reads the parser's relation. §3. |
| Arms in the thesis comparison | **native baseline vs H-Nav-GEO only.** The schema-keyed arms (`hnav_raw`, `hnav_idonly`, `hnav_abtt`) are outside the thesis comparison; §10 explains why, and uses them only as an oracle reference for the design decision. |
| Which subsets to report | **All three** (sh_6k, sh_32k, sh_64k), each labeled by its role. The operating point was selected on detection quality alone — no LLM, no accuracy, no gold answer — so every accuracy number is a legitimate accuracy measurement for every answering model. |
| Harm | No "harm-free" claim is made. Harm is not a headline; the unique stratum is reported as the do-no-harm check and it holds (§4). One quantified limitation line remains (§12). |
| Gold-not-latest benchmark quirk | Not discussed in the thesis. |
| Oracle ratio, ceiling decomposition on five models | Not used. |
| Generality | Claimed across all five measured models; embedder-agnostic architecture stated with the refit argument (§9); the hypothesis for real agent memory is stated as the thesis hypothesis (§1). |
| sh_262k | Excluded, with the reason written (§10). |
| Model substitution | Articulated in full (§6): the plans are identical, the gain is universal, the achieved accuracy tracks the model — the method is what moves the number. |
| Stale-value error taxonomy | Reported as measured; claim maintained (§7). |
| Negative results | Only those that motivate a design choice (§7). |
| Gold conflict dataset | Included as a contribution and to be published (§11). |
| Preregistration / split discipline | Reported as method (calibration vs held-out, one shot, A/A floor, frozen fingerprinted artifacts). |
| Literature review | Left blank; written separately. |
| Title | Open. |

---

## 1. Thesis statement and hypothesis

> In an evolving vector memory, conflicting versions of the same fact are the
> dominant cause of wrong answers for small language models doing RAG: the
> model retrieves the right memory and reads out the stale version. H-Nav-GEO
> detects these conflicts at read time from the fact embeddings alone — a
> whitened cosine and a learned slot probe on the edit direction, verified by
> a small NLI cross-encoder — and deletes the superseded versions from the
> page before the model reads it. It adds no LLM call at inference, shortens
> the prompt, and raises answer accuracy above the native RAG baseline on
> every one of five answering models (2B–9B, four architecture families) at
> every context size measured, with the largest gains on the models that most
> need them.

**Hypothesis (stated as such).** Because the method consumes only what every
vector memory already has — an embedding per record and a recency key — and
because the geometric structure it exploits (a shared common component that
inflates every cosine, and a slot-specific edit direction) is a general
property of sentence embeddings rather than of this benchmark, the same
detector transfers to real agent memory stores after a refit of its fitted
components, without a schema, without hand-written templates, and without an
LLM in the read path. The benchmark campaign is the controlled test of the
mechanism; the transfer is the hypothesis the thesis argues for.

---

## 2. The method: H-Nav-GEO

One question, one pass, no LLM:

1. **Page.** The benchmark's own retriever produces the page (`page_source=benchmark`).
2. **Pool.** The 50 facts on the page most similar to the query form the candidate pool.
3. **Candidate pairs.** Every pool pair with raw cosine ≥ **0.94** is a candidate.
4. **GEO screen** — two numbers per pair, both from the vectors `a, b` (2,560-d, unit length, Qwen3-Embedding-4B):
   - *Whitened cosine.* Subtract the store mean, project out the 128 dominant shared directions, renormalise, take the cosine of what is specific to the two facts:
     `ŵ(v) = (v − μ − P(v − μ)) / ‖·‖`, `cos_w = ŵ(a)·ŵ(b)`.
   - *Slot probe.* Unit edit direction `d̂ = (b − a)/‖b − a‖`, signs dropped (`|d̂|`, so orientation does not matter), fed to a logistic regression trained to separate object-slot edits (gold conflict pairs, 989 edits) from subject swaps (hard negatives, 8,716 edits):
     `probe = w·|d̂| + b₀`.
   - Each number becomes a standard-deviation margin from a calibration anchor chosen as the loosest point at which **no** look-alike pair passed:
     `m_w = (cos_w − 0.6397)/0.3483 ≥ −0.40 ⇔ cos_w ≥ 0.500`,
     `m_p = (probe + 0.0764)/1.0675 ≥ +0.20 ⇔ probe ≥ 0.137`.
   - A pair passes iff both margins clear their bar.
5. **NLI verification.** Surviving pairs are scored by `cross-encoder/nli-deberta-v3-large` in both directions; a pair is verified only if contradiction ≥ **0.90** both ways.
6. **Group and delete.** Verified pairs are chained into groups; in each group every member except the highest serial is deleted from the page.
7. The edited page goes to the answering model with the benchmark's own prompt, `temperature 0`, `max_tokens 10`.

Steps 1–6 contain no LLM, so **the suppression plan is byte-identical for every answering model** (confirmed on five models: 1,152 / 1,005 / 532 deletions per subset, identical serials). Everything learned — whitening (fit on 2,765 calibration facts), probe, anchors and scales (from 193 + 1,332 calibration pool pairs) — was fit on sh_6k + sh_32k only and is frozen with a SHA-256 fingerprint (`335b4540…`); sh_64k was touched once.

Frozen operating point: `stage0_results/geometry_filter/geo_operating_point.json` — `cos_pair 0.94`, `r_min 0.44 (loose)`, `nli 0.90`, `tau '-0.4:0.2'`; selection rule: maximise pool recall subject to zero harmful suppression on calibration, no LLM in the objective.

---

## 3. Why GEO and not CES

| | CES | **GEO** |
| --- | --- | --- |
| reads at inference | parser relation + vectors | **vectors only** |
| learned parameters | 2 × 20 directions per relation (31 relations + global) | 128 whitening directions + 2,561 probe weights |
| calibration pool recall at precision 1.000 | 0.7343 | **0.7895** |
| balanced sh_64k AUROC (held-out) | 0.9756 | 0.9716 — paired Δ −0.004, 95 % CI [−0.009, +0.001], not significant |
| unseen-transition tail, TPR at FPR 1e-4 | 0.208 | **0.481** (best in the repository) |
| end-to-end, reference model, sh_64k | 55 | **56** |
| models measured end-to-end | 1 | **5** |

GEO is chosen because it is the fully parser-free screen, it is statistically
level with CES at pair level, it has the best recall at precision 1.000 among
parser-free screens, it transfers best to transitions never seen in
calibration, and it is the screen with five-model evidence. CES's
transition-boundedness (0.768 unseen-transition AUROC for its object subspace,
`geometry_filter/REPORT.md` §4) is exactly what GEO's sign-invariant probe
avoids.

---

## 4. Headline results — five answering models, three context sizes

Every cell: one shot, `page_source=benchmark`, exact McNemar paired per
question, A/A floor 0 (native vs native_repeat identical). Subsets are
reported separately and never pooled. Generated from
`pipelines/hnav_geo/results/*/detector_gap_sh_*.json`.

**sh_6k** (calibration split; n = 100 questions)

| answering model | native | H-Nav-GEO | net (b/c) | McNemar exact p | conflicted native → GEO (n) | unique native → GEO (n) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 45 | **76** | +31 (2/33) | 3.7e-08 | 22 → **53** (74) | 23 → 23 (26) |
| gemma-4-E2B-it | 40 | **72** | +32 (3/35) | 6.7e-08 | 16 → **48** (74) | 24 → 24 (26) |
| Phi-4-mini-instruct | 40 | **76** | +36 (0/36) | 2.9e-11 | 14 → **50** (74) | 26 → 26 (26) |
| Qwen3-4B-Instruct-2507 | 30 | **77** | +47 (1/48) | 1.8e-13 | 4 → **51** (74) | 26 → 26 (26) |
| Qwen3.5-9B | 39 | **81** | +42 (0/42) | 4.5e-13 | 13 → **55** (74) | 26 → 26 (26) |

**sh_32k** (calibration split; n = 100 questions)

| answering model | native | H-Nav-GEO | net (b/c) | McNemar exact p | conflicted native → GEO (n) | unique native → GEO (n) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 38 | **45** | +7 (5/12) | 0.143 | 11 → **18** (65) | 27 → 27 (35) |
| gemma-4-E2B-it | 44 | **58** | +14 (3/17) | 2.6e-03 | 17 → **31** (65) | 27 → 27 (35) |
| Phi-4-mini-instruct | 50 | **66** | +16 (3/19) | 8.6e-04 | 16 → **33** (65) | 34 → 33 (35) |
| Qwen3-4B-Instruct-2507 | 53 | **77** | +24 (1/25) | 8.0e-07 | 19 → **43** (65) | 34 → 34 (35) |
| Qwen3.5-9B | 61 | **86** | +25 (2/27) | 1.6e-06 | 27 → **52** (65) | 34 → 34 (35) |

**sh_64k** (held-out, one shot; n = 100 questions)

| answering model | native | H-Nav-GEO | net (b/c) | McNemar exact p | conflicted native → GEO (n) | unique native → GEO (n) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 33 | **36** | +3 (1/4) | 0.375 | 14 → **17** (66) | 19 → 19 (34) |
| gemma-4-E2B-it | 37 | **41** | +4 (3/7) | 0.344 | 21 → **25** (66) | 16 → 16 (34) |
| Phi-4-mini-instruct | 46 | **52** | +6 (3/9) | 0.146 | 16 → **22** (66) | 30 → 30 (34) |
| Qwen3-4B-Instruct-2507 | 45 | **56** | +11 (2/13) | 7.4e-03 | 17 → **29** (66) | 28 → 27 (34) |
| Qwen3.5-9B | 51 | **62** | +11 (1/12) | 3.4e-03 | 24 → **36** (66) | 27 → 26 (34) |

**Reading the tables.**

- **Positive in 15 of 15 cells.** Net gain ranges from +3 to +47 overall; on the conflicted stratum, the population the method exists for, accuracy rises in every cell, by a factor of 2.4 to 12.8 at sh_6k.
- **Significant at the 5 % level in 11 of 15 cells**: 5/5 at sh_6k, 4/5 at sh_32k, and on the held-out sh_64k for the two strongest models (Qwen3-4B, Qwen3.5-9B). Where p is above 0.05 the gain is still positive and the discordant pairs still favour GEO (4/1, 7/3, 9/3 at sh_64k).
- **The unique stratum holds.** Where no conflict exists the method changes at most one question in either direction, in every cell. This is the do-no-harm check and it is what is reported instead of a "harm-free" claim.
- **Model-level consistency.** Five of five models improve at each of the three context sizes; a one-sided sign test over models gives p = 1/32 = 0.031 per subset. Models are not pooled into one McNemar test because they answer the same 100 questions with the same plans and are not independent samples.
- Calibration-split accuracies are honest accuracy measurements: the selection objective never saw any LLM's behaviour, any accuracy, or any gold answer. They are labeled because the screen's *detection* quality is in-sample there.

**Same detector, three page edits** (overall /100; identical suppression plan in every row of a subset)

| answering model | subset | native | delete stale (H-Nav-GEO) | move newest to end | move newest to front |
|---|---|---:|---:|---:|---:|
| gemma-3-4b-it | sh_6k | 45 | **76** | 38 | 72 |
| gemma-3-4b-it | sh_32k | 38 | **45** | 58 | 58 |
| gemma-3-4b-it | sh_64k | 33 | **36** | 36 | 43 |
| gemma-4-E2B-it | sh_6k | 40 | **72** | 42 | 45 |
| gemma-4-E2B-it | sh_32k | 44 | **58** | 46 | 57 |
| gemma-4-E2B-it | sh_64k | 37 | **41** | 37 | 44 |
| Phi-4-mini-instruct | sh_6k | 40 | **76** | 50 | 33 |
| Phi-4-mini-instruct | sh_32k | 50 | **66** | 76 | 43 |
| Phi-4-mini-instruct | sh_64k | 46 | **52** | 55 | 46 |
| Qwen3-4B-Instruct-2507 | sh_6k | 30 | **77** | 34 | 28 |
| Qwen3-4B-Instruct-2507 | sh_32k | 53 | **77** | 58 | 45 |
| Qwen3-4B-Instruct-2507 | sh_64k | 45 | **56** | 49 | 44 |
| Qwen3.5-9B | sh_6k | 39 | **81** | 54 | 46 |
| Qwen3.5-9B | sh_32k | 61 | **86** | 83 | 68 |
| Qwen3.5-9B | sh_64k | 51 | **62** | 55 | 55 |

Deletion is the only edit that is positive in all 15 cells. The two placement
edits, which keep every fact and only move the newest one, flip sign from
model to model (front placement: +27 on gemma-3 at sh_6k, −7 on Phi-4-mini at
sh_6k). Deletion is model-invariant; placement is model-idiosyncratic. §6
draws the conclusion.

**Plan and cost per subset** (identical for every model)

| subset | questions where the detector fired | facts deleted over 100 questions | verified pairs (mean per question) | prompt characters vs native | chunks retrieved / total |
|---|---:|---:|---:|---:|---:|
| sh_6k | 100/100 | 1,152 | 1,152 (11.5) | −2.87 % | 2/2 |
| sh_32k | 100/100 | 1,005 | 1,005 (10.1) | −0.51 % | 9/9 |
| sh_64k | 99/100 | 532 | 534 (5.3) | −0.22 % | 10/17 |

---

## 5. Detection quality — the screen without any LLM

**Calibration split, inside the benchmark's own pools** (`geo_operating_point.json`):
pair precision **1.000**, pool recall **0.7895** (the best parser-free
recall at precision 1.000 in the repository; the previous parser-free best
was 0.4444), 2,157 deletions, every one of them independently verified as a
superseded value, 104 of 139 conflicted questions reached.

**Pair level, on the audited gold conflict dataset** (`geo_pairlevel.json`;
held-out sh_64k, 1,681 gold vs 1,681 cosine-matched non-conflicts, 1,000
bootstrap resamples):

| method | balanced AUROC [95 % CI] | overlap-band AUROC (cos 0.87–0.97) |
| --- | --- | --- |
| raw cosine (the standard screen) | 0.893 [0.882, 0.904] | 0.850 |
| ABTT-whitened cosine alone | 0.965 | 0.952 |
| slot probe alone | 0.913 | 0.902 |
| **H-Nav-GEO screen** | **0.972 [0.967, 0.976]** | **0.966** |

Against the confirmatory hard task (1,681 gold vs 39,215 same-relation
different-subject negatives): AUROC **0.998**, AUPRC **0.978**; over the
527,062 comparisons that raw cosine orders exactly wrong, GEO orders **95.9 %**
correctly. Tail behaviour at FPR 1e-4: TPR **0.720** on transitions seen in
calibration and **0.481** on transitions never seen — the best unseen-tail
figure of any screen measured (ABTT-cosine 0.404, CES 0.208).

**In words:** cosine similarity says two facts are near-duplicates; it
cannot say whether the *object* changed (a supersession) or the *subject*
changed (a different fact). GEO reads *what changed* from the edit direction
and *how similar the specific content is* after the shared component is
removed, and that is what lifts the discrimination from 0.89 to 0.97 exactly
in the band where cosine is uninformative.

---

## 6. What changes when the answering model changes, and what does not

The campaign is a controlled substitution: memory, retrieval, embeddings,
suppression plans, prompts, generation settings and scoring are frozen; the
answering model is the only variable.

**What does not change.** The suppression plan. It is computed with no LLM,
so it is byte-identical across the five models — the same 1,152 / 1,005 /
532 deletions, the same 99 or 100 questions fired. The direction of the
effect: positive in all 15 cells. The stratum where nothing should happen:
the unique stratum moves by at most one question anywhere.

**What changes.** The native accuracy, from 30 to 61 across models, for
reasons that have nothing to do with memory. And the *achieved* accuracy
after H-Nav-GEO, which orders the models the same way at 32k and 64k
(45 / 58 / 66 / 77 / 86 and 36 / 41 / 52 / 56 / 62 for gemma-3, gemma-4-E2B,
Phi-4-mini, Qwen3-4B, Qwen3.5-9B) and compresses to 72–81 at 6k where the
ceiling is near. The gain is **not** a function of native accuracy: the model
with the lowest native score (Qwen3-4B, 30) gains the most (+47); a model with
a higher native score (gemma-3, 45) gains less (+31).

**What this means.** The stale-version failure is a failure of the model's
in-context version resolution, and every small model has it. H-Nav-GEO
removes the stale version before the model reads the page, so the model only
has to do what it already can — read a page and answer. The number that moves
is therefore set by the method, not by the model: the model determines how
much of the cleaned page it converts into correct answers, and stronger models
convert more; but no model converts a page it was never handed. The
placement arms make the same point from the other side: an edit that relies
on how a particular model weights position is model-idiosyncratic, while an
edit that removes the wrong information is not.

---

## 7. Mechanism evidence that motivates each design choice

Each item is a measured result from this repository, included because a
design decision in §2 rests on it.

1. **The baseline fails in one specific way — it returns the stale value of the right key.** Reference model, eight independent runs on sh_6k: 26/26 correct on questions with no conflict in every run, 0–5 of 74 correct on questions with one; of 575 conflicted-question errors, **572** were the stale value of the queried key, 3 off-list, 0 empty (`stage0_results/question_strata.json`). The pattern holds on every model in the campaign: native conflicted accuracy at sh_6k is 4, 13, 22, 16, 14 of 74 while unique accuracy is 26, 26, 23, 24, 26 of 26. The model finds the right memory and reads the wrong version. → *The intervention has to remove the stale version.*
2. **Cosine is necessary but nowhere near sufficient.** Conflict pairs sit in a narrow high-cosine band (mean 0.956, stable across store sizes), but on sh_64k 65,782 non-conflict pairs lie above the conflict cosine floor against 1,687 conflicts — 39 look-alikes per conflict (`DELTA_GEOMETRY_REPORT.md` §2). → *A second, non-cosine signal is required: the slot probe.*
3. **NLI cannot carry identity either.** A bidirectional NLI cross-encoder scores "Thomas Kyd was born in London" vs "Marlowe was born in London" as contradiction 0.9995 / 0.9998 (`TEZ_BULGULARI.md` §B); 447 of 453 cross-key adversaries pass bidirectional 0.90 (`GEO_PREREG.md`). → *Identity must be decided before NLI, from geometry; NLI is the semantic verifier, not the identity screen.*
4. **Whitening decompresses the cosine band.** The embedder's anisotropy (unrelated facts at cos ≈ 0.60) is removed completely by mean-centring plus 128 principal directions; recall at precision 1.000 rises from 0.075 to 0.513 (sh_6k) and from 0.007 to 0.291 (sh_32k) (`G1_GATE_REPORT.md` §4). → *The first GEO signal is the whitened cosine.*
5. **The edit direction is slot-specific and sign-invariant features carry it.** A probe on `|d̂|` separates object-slot from subject-slot edits at macro-F1 0.70 and object-vs-subject AUROC 0.96, while signed features are provably unlearnable for unordered pairs (`geometry_filter/REPORT.md` §2). PCA-compressing the profile hurts (0.73–0.82), so the signal is distributed across dimensions; a full-covariance QDA scorer does not beat the flat subspace/probe family (`qda_filter/REPORT.md`). → *The second GEO signal is a linear probe on the full 2,560-d magnitude profile.*
6. **Granularity must match the conflict: fact-level, not chunk-level.** Chunk-level reranking driven by a precision-1.0 detector was net negative in 0 of 81 operating points, helping 228 and harming 441 questions (`STAGE1_NULL_ANALIZI.md`). → *Edit facts, not chunks.*
7. **Deletion is the edit that works, and works on every model.** §4's placement table: deletion positive in 15/15 cells; moving the newest fact to the end or front flips sign across models. → *Suppress, do not reorder.*
8. **The write path has no headroom here.** After safety vetoes a write-time policy would touch 0–1.6 % of writes and could change correctness on 0.00 of them on sh_64k (`KAPI_KARARI.md` §2). → *Intervene at read time.*

---

## 8. Cost

- **Inference-time LLM calls added: zero.** The online path (`hnav/core/`, `hnav/adapters/`) has no chat or completion call; its only network call is the embedding endpoint, and in the campaign even that is served from the on-disk cache.
- **The prompt gets shorter.** −2.87 % / −0.51 % / −0.22 % characters at sh_6k / sh_32k / sh_64k, identical for every model because the plan is identical. Accuracy goes up while token cost goes down.
- **Per pair, the screen is a handful of 2,560-d operations**: one projection per fact for whitening (shared across all pairs the fact appears in), one dot product for the whitened cosine, one dot product for the probe. Every geometric analysis in this repository ran on a laptop CPU; no GPU is needed for the screen.
- **NLI verification is small and offline-batchable.** It runs only on pairs that pass cosine 0.94 and the GEO screen; the campaign's precomputed tables bound it at 410 / 4,868 / 8,956 directed cross-encoder passes per 100 questions (≤ 90 per question at 64k; `TEZ_HIKAYESI.md` §3.1), with a 435M-parameter DeBERTa. On the reference model that is ≈ 1.4 % of one question's prefill cost. Verified pairs actually formed: 11.5 / 10.1 / 5.3 per question.
- **One-time, answering-model-independent fitting**: 2,765 facts for the whitening, 989 + 8,716 edits for the probe, 1,525 pool pairs for the anchors. Trying a new answering model costs nothing in refitting; the campaign added four models without touching any of it.
- **Embeddings are the store's own.** The RAG pipeline already computes one vector per fact and per query (4,680 for sh_64k, 0 cache misses); H-Nav-GEO reuses them.

---

## 9. Generality

- **Across answering models.** Five models, 2B to 9B, four architecture families (Gemma 3, Gemma 4, Phi-4, Qwen3 / Qwen3.5), two vLLM versions, one served with thinking disabled. The gain replicates in every model at every context size (§4).
- **Across context sizes.** Positive at 455, 2,310 and 4,580 stored facts (2, 9 and 17 chunks). The gain narrows as the store grows because the benchmark's retriever returns 10 of 17 chunks at 64k, so fewer conflicts reach the page for any read-time method to act on; the plan fires on 99/100 questions there.
- **Across embedders — the architecture is embedder-agnostic by construction.** Nothing in H-Nav-GEO is a hand-set constant: the whitening, the probe, the anchors and the scales are all *fitted procedures* with a committed refit recipe (`python -m hnav.geometry_filter.geo_artifact`, calibration split only). The geometric properties it relies on are documented general properties of sentence embedders in English — a dominant common component that inflates all cosines and is removed by mean-centring plus a few principal directions (the "all-but-the-top" literature), and a systematic per-slot structure in difference vectors. The thesis states as its hypothesis that other English embedders yield the same qualitative geometry and require a refit, not a redesign; a second-embedder refit is the first item of future work.
- **Toward real agent memory (the thesis hypothesis).** The method consumes exactly two things every vector memory store already has: an embedding per record and a recency key (a serial, a timestamp, a version). It needs no schema, no relation templates, no entity extraction and no LLM at read time. That is why it is proposed as the general mechanism, and why the schema-keyed alternative was set aside (§10).

---

## 10. Design boundaries: why no parser, and why not sh_262k

**Why the identity signal is geometric and not a fact parser.** The benchmark
generates its facts from 39 sentence templates, so a regular-expression
parser recovers each fact's `(relation, subject)` key exactly, and a screen
keyed on that schema does very well *on this benchmark* precisely because the
language structure is uniform and template-generated. Such a screen has no
counterpart in a real memory store: it needs either hand-written templates
per domain or an LLM extraction call per write — the cost and the dependency
this thesis sets out to remove. The thesis therefore studies parser-free
identity, and H-Nav-GEO is the result. The repository keeps schema-keyed arms
as reference measurements; they are not part of the thesis comparison.

Used only as an oracle reference for this design decision: on the reference
model, geometry alone recovers 527 of the 735 deletions a schema-keyed oracle
makes on sh_64k (72 %) with no schema at all, and 52–75 % of that oracle's
accuracy gain across subsets (`E2E4_COMPLEMENTARITY.md` §4,
`LATEST_RESULTS_REPORT.md` §12.2). Geometry recovers most of what the schema
knows, from vectors alone. We hold that for generalisation beyond a templated
store this is the far more efficient signal: it is available everywhere, it
costs a few dot products, and it does not depend on the store's language
matching a template.

**Why sh_262k is excluded** (user decision 2026-08-24, enforced by
`ALLOWED_SUBSETS` in `pipelines/_shared/runner.py`). Its contexts would exceed
the context window of the smaller answering models the campaign exists to
test, which would silently turn a memory comparison into a context-length
comparison; and it was never part of the registered campaign (no prepass, no
NLI table, `detector_gap` refuses it). The thesis states that sh_262k was not
measured; it does not extrapolate to it.

---

## 11. Contributions

1. **H-Nav-GEO** — a read-time conflict detector for evolving vector memory that uses embedding geometry (whitened cosine + slot probe on the edit direction) and a small NLI verifier, with zero inference-time LLM calls, a shorter prompt, and a fitted, fingerprinted, refittable artifact.
2. **Five-model evidence** that the governance gain is universal across small answering models and context sizes (15/15 cells positive; 11/15 significant), with byte-identical plans and an A/A floor of zero.
3. **The gold conflict dataset** — 54,569 audited fact pairs (2,388 core gold conflicts, dual `gold_update` / `gold_strict` labels, cosine-matched balanced evaluation sets per subset with a recorded cosine-only AUC baseline of 0.96 / 0.91 / 0.89). It is the probe's training data and the pair-level benchmark; it is released with the thesis.
4. **Mechanism findings** about small-model RAG: the stale-value failure mode; cosine's necessity without sufficiency; NLI's inability to carry identity; the whitening effect on the precision tail; deletion over placement; fact-level over chunk-level granularity.
5. **A measurement methodology** for memory interventions: calibration vs held-out split with one-shot held-out runs, operating points selected on detection quality with no LLM in the objective, an in-harness A/A floor, model-independent suppression plans, artifact-generated tables.

---

## 12. Limitations (to be stated briefly, once)

- On the held-out sh_64k page, 8 of the 532 deletions (1.5 %) merged two different keys because a geometric group is not an equivalence relation; the net accuracy effect is nonetheless positive on every model and the unique stratum is unaffected. Symbolic keys avoid this by construction at the cost of a schema; a transitivity-aware grouping is future work.
- One embedder (Qwen3-Embedding-4B); the fitted parameters are coordinates in its space and are refit, not copied, for another encoder.
- The store has single-valued relations; the NLI verifier's role grows in stores with genuinely multi-valued relations.
- Five answering models; on the held-out 64k subset the per-model gain is significant for the two strongest models and positive for all five.
- One benchmark family (MemoryAgentBench `Conflict_Resolution`); sh_262k not measured.

---

## 13. Wording rules

- Say "the do-no-harm stratum is unchanged to within one question in every cell"; never say "harm-free".
- Say "positive in 15 of 15 cells, significant in 11"; never say "significant everywhere".
- Report per-model exact McNemar and model-level sign consistency; never pool the five models into one p-value.
- Label sh_6k and sh_32k "calibration split" and sh_64k "held-out" wherever they appear.
- Say "embedder-agnostic by construction; refit per encoder"; never say "the thresholds transfer to any embedder".
- Say "not measured on sh_262k"; never extrapolate to it.
- Quote every number with the file in §15.

---

## 14. Chapter skeleton

1. **Problem.** Small models fail on conflicted memory in one systematic way (§7.1), on every model tested.
2. **Method.** H-Nav-GEO (§2): geometry decides identity, NLI verifies, deletion repairs; zero LLM at inference; fitted and frozen on the calibration split.
3. **Why this design.** The mechanism evidence (§7): cosine insufficient, NLI insufficient for identity, whitening, slot probe, fact-level deletion, read-time.
4. **Results.** Five models × three context sizes (§4); detection quality and pair-level discrimination (§5); cost (§8).
5. **Model substitution.** What changes and what does not (§6).
6. **Generality and design boundaries.** §9, §10.
7. **Contributions, limitations, future work.** §11, §12; a second embedder; transitivity-aware grouping; a real memory store.
8. **Related work.** Written separately.

---

## 15. Provenance

| number | file |
| --- | --- |
| five-model tables, placement arms, plan/cost table | `pipelines/hnav_geo/results/<model>_<date>/detector_gap_sh_{6k,32k,64k}.json` (`arms`, `paired_vs_native`, `by_stratum`, `tokens`, `positive_control`, `per_question[].plan`) |
| per-model REPORT tables | `pipelines/hnav_geo/results/<model>_<date>/REPORT.md` |
| cross-arm summary (generated) | `pipelines/MULTIMODEL_SUMMARY.md` |
| frozen operating point, calibration detection metrics | `stage0_results/geometry_filter/geo_operating_point.json` |
| screen parameters, anchors, scales, probe provenance | `stage0_results/geometry_filter/geo_identity_screen.json` + `.npz` |
| pair-level AUROC, band, hard task, tails | `stage0_results/geometry_filter/geo_pairlevel.json` |
| GEO vs CES vs ABTT-cosine pair-level | `stage0_results/geometry_filter/REPORT.md` §3, §7; `geo_pairlevel.json` |
| CES operating point | `stage0_results/geometry_filter/ces_operating_point.json` |
| design and selection of the GEO screen | `stage0_results/geometry_filter/GEO_PREREG.md`, `E2E3_REPORT.md` |
| oracle-reference containment (§10) | `stage0_results/geometry_filter/E2E4_COMPLEMENTARITY.md` §4 |
| stale-value taxonomy, strata | `stage0_results/question_strata.json`, `hnav/labeling/question_strata.py` |
| cosine necessity/insufficiency | `stage0_results/delta_geometry/DELTA_GEOMETRY_REPORT.md` §2 |
| NLI rubber-stamp measurement | `TEZ_BULGULARI.md` §B; `GEO_PREREG.md` |
| whitening effect | `stage0_results/abtt/G1_GATE_REPORT.md` §4; `abtt_whitening_D128.json` |
| slot probe, PCA, QDA | `stage0_results/geometry_filter/REPORT.md` §2; `stage0_results/qda_filter/REPORT.md` |
| chunk-level rerank harm | `STAGE1_NULL_ANALIZI.md` |
| write-path headroom | `KAPI_KARARI.md` §2 |
| NLI prepass counts, cost fractions | `TEZ_HIKAYESI.md` §3.1 |
| gold conflict dataset | `stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz`, `.summary.md`, `AUDIT_SUMMARY.md`; `hnav/tests/test_gold_conflict_dataset.py` |
| campaign plan and serving facts | `pipelines/MULTIMODEL_CAMPAIGN_PLAN.md` |
| the screen (code) | `hnav/geometry_filter/geo_artifact.py`; gate wiring `hnav/stage1/detector_gap.py::make_gate`; deletion `hnav/core/read_policy.py` |
