# H-Nav — Final Program Report

**Project:** H-Nav, a memory-governance layer for LLM agents
**Substrate:** EvoMemBench / MemoryAgentBench (`Conflict_Resolution`), Qwen3-4B-Instruct-2507
**Period covered:** 2026-08-14 → 2026-08-16
**Companion documents:** `HNAV_HOW_IT_WORKS.md` (how the system works),
`TEZ_BULGULARI.md` (evidence ledger, Turkish), `KAPI_KARARI.md` (Stage-0 gate decision, Turkish, superseded in part)

---

## 0. Executive summary

We set out to test whether a governance layer over an agent's memory can make
the agent answer better. The honest short version:

**What we found first — and it reframed everything.** On the canonical
conflict-resolution benchmark, the model answers **perfectly** (26/26, in every
one of eight independent runs) when a question has no conflicting memory, and
**almost never** (0–5 of 74) when it does. Of ~575 conflicted-question errors,
**572 returned the stale value of the correct key**. The benchmark's headline
accuracy is therefore carried almost entirely by questions that contain no
conflict, and the model ignores an explicit "higher serial = newer" instruction
roughly 95% of the time.

**What we built.** After measuring and discarding two earlier designs, the
surviving system is narrow: at read time, detect superseded facts inside the
retrieved page — geometry, then a parsed subject-identity screen, then
bidirectional NLI — and **delete them**. Nothing else.

**What it achieved.** On held-out data, in a single pre-registered shot:

| | baseline | H-Nav |
|---|---|---|
| conflicted-question accuracy | 17/66 (25.8%) | **37/66 (56.1%)** |
| overall accuracy | 0.450 | **0.640** |
| token cost | — | **−0.31%** (cheaper) |
| conflicted questions harmed | — | **0** |

McNemar exact **p = 1.9 × 10⁻⁶**, against an A/A noise floor measured at exactly
**0/0 discordant pairs**. Detector precision **1.000** — 735 of 735 deleted
facts independently verified as genuinely superseded.

**What it did not achieve.** One non-conflicted question regressed: the model
*refused to answer* after the edit even though the fact it needed was still on
the page. That voids the pre-registered safety criterion. The registered
conclusion, written before the data existed, therefore stands: **effective, but
not yet safe.**

**What we ruled out, and reported as negative:** write-side intervention
(headroom ≈ 0), chunk-level reranking (actively harmful), — in the secondary
arena — the conversion of measured redundancy into accuracy (unproven and
currently unmeasurable), and **ABTT anisotropy correction applied before the
cosine screen** (fixes the geometry completely, changes no answer; §8b).

---

## 1. Why this project exists

H-Nav is a governance and navigation layer that sits **on top of** a memory
backend. It does not replace vector memory, the retriever, or the model. It
supervises how they interact, detecting problematic memory states — conflicts,
stale facts, redundancy, retrieval interference — and, where justified,
intervening.

Its design has two paths:

- **Write path** — inspect a candidate memory against what is already stored and
  decide whether to write, merge, update, ignore or block it.
- **Read path** — inspect the retrieved memory set before it is used, and repair
  it when confidence is high enough.

The core commitment, and the one this program actually tested, is *selectivity*:
**detect first, estimate how much room there is to help, and only then act.**

A prior attempt on a different substrate (BFCL) had returned a null result. That
history shaped the method here: rather than build an intervention and hope,
Stage 0 was designed to **measure whether intervention was even possible** before
any policy code was written.

---

## 2. Method: how this program was run

Three disciplines did most of the work of making the final number believable.

### 2.1 Gates that stop the pipeline

Stage 0 shipped with hard gates. If a validity check failed, the pipeline halted
and a human decided — no proceeding on the agent's judgment. **Two fired**, and
both caught real problems (§3.2, §3.3).

### 2.2 Split discipline and pre-registration

`sh_6k` + `sh_32k` were the only data any threshold, operating point or design
choice was ever fitted on. `sh_64k` was touched **exactly once**. `sh_262k` was
never used. The confirmatory run's design, success criterion, harm criterion,
analysis code, void conditions and a falsifiable side-prediction were all
committed **before** the run — commit timestamps prove the ordering
(19:52:24 registered → 22:00:02 fired).

An earlier pre-registration was **formally withdrawn** with cause rather than
quietly amended, and is retained in the repository as evidence.

### 2.3 Independent adversarial audit

Every deliverable passed through a supervisor whose job was to *refute* it: it
re-derived numbers with its own implementations, re-verified deletions against
ground truth, attacked classifiers with independently-written alternatives, and
checked commit ordering rather than trusting claims.

Across eleven audits it found five real defects — a void condition whose prose
read weaker than its code, a question mis-classification, a cross-context state
leak, an internal contradiction in the void-scope rules, and a latent code path
that could have fitted thresholds on held-out data. **All five were fixed rather
than argued away.**

---

## 3. Stage 0 — validating the instruments

Stage 0 asked one question: *are our measurements trustworthy, and is there
anything to win?* Ten stages, all completed.

### 3.1 What was validated

| component | result |
|---|---|
| **Geometry premise** — do conflicting facts sit close in embedding space? | median similarity **0.964** vs **0.60** for controls; **AUC ≥ 0.9999** on all four subsets |
| **Geometric grouping** vs a regex oracle | best F1 **0.892** (sh_6k) → **0.757** (sh_262k), precision 0.83–0.90 |
| **Replica fidelity** — does our shadow index reproduce the benchmark's ranking? | top-1, top-k and Kendall τ all **1.0000**, max score error ≤ 4.5×10⁻⁵, 400/400 pairs |
| **Signal degeneracy** | **NOT_DEGENERATE** — refutes the prior BFCL degeneracy finding (re-earned on calibration after the truncation fix; held-out subsets remain formally pre-fix) |
| **Shadow neutrality** — does enabling H-Nav without acting change anything? | established by pre-registered statistical equivalence (see §3.3) |
| **Fact parser coverage** | 99.44–99.65% |

### 3.2 Gate S1 fired — and caught a real precision fault

The pipeline halted 16 seconds into the replica-fidelity check: top-k agreement
had collapsed to **0.24**.

Root cause: the embedding server had been started without a dtype flag and
defaulted to the checkpoint's **bfloat16**, while the campaign was pinned to
float32. Normalized vectors came back with norms of 0.998–1.002 instead of
1±10⁻⁷, and at that error scale the ordering of near-duplicate chunks becomes
unstable. With float32 restored, agreement went to **1.0000** on all four
subsets.

**As a finding:** serving an embedder in reduced precision can silently destroy
retrieval fidelity while every component still "works." The gate caught it; a
system without that gate would have produced plausible, wrong numbers.

### 3.3 Gate S2 fired — and revealed the substrate is nondeterministic

Shadow mode was required to be byte-identical to off. It differed on 2 of 100
outputs. Before treating that as a defect, we ran the control: **two identical
baseline runs**, no H-Nav code at all.

They differed on **5 of 100**, with a 4-point exact-match swing (26.0 vs 30.0).

The evaluation substrate is not run-to-run deterministic at temperature 0 —
continuous batching and prefix caching reorder floating-point work. Byte
identity was unsatisfiable for *any* code, H-Nav or not.

Resolution (user decision, pre-registered): 10 baseline + 5 shadow runs, TOST
equivalence against a margin fixed in advance. Off↔shadow difference (2.42%)
came in **below** the baseline's own noise floor (3.04%), and equivalence was
established (p = 0.0008 / 0.017). We also attempted a deterministic server to
settle it definitively and **failed** — reported as such, with the mechanism
documented.

**Later refinement:** the noise is confined **entirely** to conflicted
questions. Across 28 run-pairs, **zero** flips landed on a non-conflicted
question. That made the non-conflicted stratum a zero-noise control for
everything that followed.

### 3.4 The Stage-0 verdict

A differentiated verdict rather than a binary one:

- **Detection instruments: GO.** All five components validated.
- **Write-path intervention: NO_GO.** After safety vetoes, H-Nav would touch
  0–1.6% of writes, and on the confirmation subset **0.00** of those could
  change answer correctness. `write_policy.py` is now permanently forbidden by a
  test. *This is the arena's answer, not the design's:* in single-hop fact
  consolidation, retrieval already finds the newest fact, so there is nothing
  for a write policy to fix.
- **Read-path intervention: CONDITIONAL.** Signal present, but an offline repair
  experiment helped at one scale and net-harmed at the largest.
- **A pre-registered secondary test (H2) FAILED** — direction positive, but the
  registered conjunction failed on its likelihood-ratio prong (p = 0.341).
  Reported as a failure.

---

## 4. Stage 1, first attempt — a null that was not a null

The first read-path design reordered **chunks**, promoting the chunk containing
the newest fact. We tested 162 gate configurations with real LLM grading on
calibration data. The objective returned: *no feasible operating point*.

Rather than accept or reinterpret that, we retrieved the raw per-cell data and
classified all 162 cells. The summary statistic had been hiding three very
different worlds:

| | |
|---|---|
| cells with no order change (a harness no-op) | **0** — every cell reordered and re-graded 26–115 of 200 questions |
| cells net-negative | **129** |
| cells net-positive | 21 |
| **with the subject screen ON** (precision 1.000) | **0 of 81** cells net-positive; helped 228, harmed **441** |

So the intervention was not inert and not noise-limited — it was **systematically
harmful, about twice as often as it helped**. The 21 apparently-positive cells
all had the subject screen *off*, i.e. they were reordering on ~86% false
conflicts; the pre-registered quality criterion correctly rejected them.

**Two lessons, both reportable:**

1. **Granularity must match the conflict.** A chunk carries ~230–260 facts;
   moving one to fix a single conflict scrambles hundreds of unrelated facts.
2. **A sloppy detector manufactures apparent gains.** Gains that vanish when the
   detector is made precise were never gains. Any memory-intervention result
   reported without its detector's precision is untrustworthy.

We also recorded a design error of our own: the harm cap (≤2% of 200 = 4
questions) sat **below** the measured noise floor (~6.6 expected flips), so no
intervention could have passed it.

---

## 5. The finding that changed the project

While the box was offline we stratified every question by whether its queried
key actually had two competing values.

| | non-conflicted | conflicted |
|---|---|---|
| sh_6k (n = 26 / 74) | **26/26 correct in all 8 runs** | **0–5 / 74** |
| error taxonomy (575 errors) | — | **572 stale value**, 3 off-list, **0 empty** |

Implied conflicted-only accuracy: **9.5%** (sh_6k measured), with bounds of
[0.185, 0.723], [0.152, 0.667], [0.000, 0.263] for the larger subsets — reported
as *bounds*, not point estimates, because the premise behind extrapolation is
measured only on sh_6k and is provably false on sh_262k.

**Why this reframed everything:**

- The benchmark's headline number mostly measures conflict-*free* retrieval.
- The failure is **systematic and single-mode** — not confusion, not refusal:
  the model reads the right memory slot and takes the wrong version.
- Therefore the headroom is **large** (71 of 100 questions wrong at sh_6k for
  one reason), and chunk reranking had simply been the wrong tool.

This finding is independent of every embedding-related defect in the program —
it computes no embeddings at all — and it survived two independently-written
classifiers. It is the most robust result we have.

---

## 6. Measuring the ceiling, then closing the gap

### 6.1 The oracle probe

Before building anything, we measured what a *perfect* intervention could
achieve, using gold answers to identify the target (a ceiling, not a system).
Four arms plus an A/A floor, calibration split only:

| arm | sh_6k conflicted | sh_32k conflicted |
|---|---|---|
| baseline | 4/74 (5.4%) | 7/65 (10.8%) |
| A/A repeat | 4/74 — **0/0 discordant** | 7/65 — **0/0 discordant** |
| **suppress the stale fact** | **66/74 (89.2%)** | **53/65 (81.5%)** |
| move newest to the **end** | 20/74 (27.0%) | 33/65 (50.8%) |
| move newest to the **front** | 1/74 (1.4%) | 4/65 (6.2%) |

This settled a question that had been open since §5: was the model **anchoring on
the stale fact's presence**, or **overriding context with world knowledge**?
Deletion lifting accuracy to ~85% answered it — the model reads context; it just
loses to whatever appears in a particular position. The front/end asymmetry
identified the mechanism as **late-position anchoring**, which also explains
retroactively why §4's upward reranking harmed.

### 6.2 From oracle to a real, gold-free detector

A shippable policy may use only detector output. We built the fact-level
mechanisms and measured the gap:

| | detector achieved | oracle ceiling | ratio |
|---|---|---|---|
| sh_6k | +61 | +62 | **0.984** |
| sh_32k | +44 | +46 | **0.957** |

The entire cost of replacing gold with the detector across 1,000 graded calls
was **two flips**. Detector precision was **1.000** over 2,673 verified pairs,
with **zero** facts deleted that carried a key's current value.

The operating point was frozen on **detection quality alone** — no LLM, no gold,
no accuracy — before any arm was graded, and commit ordering proves it.

### 6.3 Testing in the deployed setting

Two constraints forced a design change before the confirmatory run, and both
turned out to strengthen the result:

1. A whole-context prompt at sh_64k is **75,886 tokens** against a 65,536 limit
   — physically impossible.
2. At sh_64k the deployed system never sees the whole context anyway: 17 chunks
   exist, 10 are retrieved.

So the confirmatory run used the **retrieval path** — the setting the system
actually occupies. On calibration, that harness performed **equal or better**
(conflicted 68/74 and 52/65), and the one inconsistent control resolved: `anti`
had helped on one subset in the whole-context layout, but in the deployed layout
it is harmful on all three (−1, −6, −2). The position hypothesis is better
supported in the setting that matters than in the one where it was found.

**A subtle trap caught here:** re-embedding the chunks with a corrected encoder
still reproduces the benchmark's retrieved page on only **26 of 100** questions
(3/100 with the old vectors). With 17 tightly-clustered chunks, a perturbation of
3×10⁻⁵ reshuffles the top-10 boundary. **Retrieval-page membership is not
reproducible by re-encoding — only by reading the index.** The harness was
changed to read the benchmark's own vectors.

---

## 7. The confirmatory result

**Held-out `sh_64k`, one pre-registered shot, 100 questions × 5 arms, 500 calls.**

| arm | overall | non-conflicted | conflicted | McNemar b/c | net | exact p | tokens |
|---|---|---|---|---|---|---|---|
| baseline | 0.450 | 28/34 | 17/66 | — | — | — | 0 |
| A/A repeat | 0.450 | 28/34 | 17/66 | 0/0 | 0 | 1.0 | 0 |
| **suppression** | **0.640** | 27/34 | **37/66** | **0/20** | **+20** | **1.9×10⁻⁶** | **−0.31%** |
| placement | 0.480 | 28/34 | 20/66 | 2/5 | +3 | 0.45 | 0 |
| anti | 0.430 | 28/34 | 15/66 | 3/1 | −2 | 0.63 | 0 |

**Primary criterion (registered in advance): MET.** Net ≥ +10 ✓ (+20);
p < 0.01 ✓ (1.9×10⁻⁶); token cost ≤ 0 ✓ (−0.31%).

**Protective criterion: VOIDED**, by one question. On q77 the model went from
answering "John Milton" to *"the knowledge pool does not contain any information
about…"* — with the gold fact **still on the page**. Not a deletion error; a
refusal induced by the edit. Under the registered rule (deliberately strict),
one such case voids the safety claim.

**The falsifiable side-prediction: MISSED, and reported as missed.** We predicted
exactly 2 gold-fact deletions. Observed: 1 deletion, **0** accuracy losses. And
the detail is a caveat rather than reassurance — on the one real deletion the
page was left containing only the *wrong* value, and the model answered correctly
anyway from world knowledge. So *"zero accuracy loss from gold cuts" is evidence
about the evaluator, not evidence that gold cuts are safe.*

**Void conditions: 1–4 and 6–8 all passed** (zero edit mismatches, baseline
inside its pre-fixed band, A/A floor 0, all 735 suppressions verified superseded,
correct page source, zero containment violations, positive controls fired). The
run is **not** void; the shot is spent.

**Effect size shrank exactly as predicted, for the predicted reason:** +62/+66 at
sh_6k, +44/+38 at sh_32k, +20 here — with only 735 suppressions versus 1,416 and
1,257, because the 50-fact pool and 10-of-17 retrieval bound what the detector
can see. That prediction was written two hours before the shot.

---

## 8. The secondary arena — Cross-Episode Knowledge

A parallel effort asked whether the *write*-side cascade (which had no headroom
in the primary arena) has any on a genuine memory-API substrate.

**First fix:** the secondary adapter was emitting **zero** write-side signals —
geometry, diff and effect were hardcoded to null. Repairing it exposed two
latent bugs: a mixed embedding-space crash (1024-d backend vectors against
2560-d H-Nav vectors) and a missing context identifier.

**What we measured** (120 contexts, 7,879 write events, cluster-first statistics,
a frozen 48/72-cluster split):

| axis | calibration | held out | control |
|---|---|---|---|
| byte-identical duplicates | 0.117 | 0.072 | — |
| **pairwise near-duplication (cos ≥ 0.95)** | **21.8% within-context** | — | **0.04% across contexts** (8 of 20,000 pairs) — a **~545× separation** |
| bidirectional contradiction | 0.0129 | — | — |
| **critical delta (fact-level, untruncated)** | **0.0000** | **0.0000** | — |

Compare the primary arena, where the duplicate rate is **0.000 everywhere**. The
substrates are structurally different.

**Two pieces of self-correction shaped this result.** First, the hypothesis that
the near-duplicate mass was a harness artifact proved *half* wrong: it explains
the exact duplicates (11.6% are verbatim substrings of a shared context block)
but not the near-duplicates — organic chunks with almost no shared content still
reach 0.95 similarity at 85%, and 62% of neighbours come from a different sample
in the same context. Second, the headline was demoted by its own author: 86% is a
**nearest-neighbour** statistic, and with a 21.8% pairwise rate over stores of
30–250 records, the predicted nearest-neighbour rate under independence is
**0.944** — *above* the observed 0.863. **The 86% is therefore not part of the
claim at all**: it is what the pairwise rate and the store size already predict,
and carries no additional evidence of redundancy. All reporting leads with 21.8%.

**Verdict:** *Redundancy headroom is real and measured; its conversion to accuracy
is unproven and currently unmeasurable.* Three reasons for refusing to assert a
sign: the direction is unpredictable (this program already produced a
compensator that helped at one scale and harmed at another); the measurement
chain requires credentials we do not have; and "redundancy exists" is not
"redundancy harms" — high self-rank and low novelty are equally consistent with a
**coherent** store.

*Note on the contradiction figure:* 85% of NLI inputs were truncated at the
model's own 512-position limit, and truncation can only *remove* text, so 0.0129
is a **lower bound whose true value can only be higher**. The
redundancy-not-conflict characterisation therefore rests on the untruncated
fact-level `critical delta = 0.0000`, with NLI as corroboration.

---

## 8b. Anisotropy and ABTT whitening — a measured null

The detector thresholds raw cosine in a space we measured to be strongly
anisotropic: two facts with nothing in common sit at cosine **≈ 0.604**, and no
candidate pair at sh_262k falls below **0.65**, so roughly the bottom three
quarters of the nominal cosine range is never used (§3;
`presentation_evidence/GEOMETRY_AND_ANISOTROPY.md`). The standard remedy —
All-But-The-Top (Mu & Viswanath, ICLR 2018) — had been implemented and tested
since Stage 0 but never armed. In August 2026 it was armed and fired.

**Design.** The whitening parameters (mean and 128 principal directions) were
fitted **offline, once, on the calibration split only** and shipped as a frozen
artifact with a sha256 fingerprint. That dissolves the standing objection that
ABTT cannot run at decision time because the 50-fact read pool is below
`min_fit_n = 200`: a pre-fitted whitener has nothing left to estimate. Whitening
was confined to the **fact–fact geometry the gate decides on**; whitening the
query as well was measured to be actively harmful, costing 27% of the reachable
true-supersession pairs because `select_pool` then builds a worse pool. ABTT
helps symmetric fact–fact comparison and hurts asymmetric question–fact
retrieval. Thresholds were re-fitted from scratch in the corrected space on
detection quality alone (`cos_pair` 0.90 → 0.30), and the run was
pre-registered before grading.

**What it changed — substantially:**

| property | raw | ABTT |
|---|---|---|
| anisotropy (unrelated-pair mean cos) | 0.6024 / 0.6026 | ≈ 0.000 |
| candidate-pair floor | 0.5815 / 0.6130 | 0.083 / 0.081 |
| screen precision at equal recall (sh_32k) | 5.3% | **51.3%** |
| recall at precision 1.000 (sh_6k / sh_32k) | 0.0750 / 0.0072 | **0.5125 / 0.2910** |

**What it changed in accuracy — nothing.** On held-out sh_64k, conflicted
stratum: **37/66 in both arms. Not one question differs** (95% CI [0, 0],
McNemar p = 1), at equal harm and equal token cost.

The null is exact rather than underpowered: the raw arm reproduced the
confirmatory campaign of §7 with **500/500 identical graded outcomes and zero
differing model outputs**, and the A/A floor was a true 0/0. Nor is it inert —
the two arms produced different suppression plans on **12 of 100 questions** and
differed by 16 suppressed facts, changing the model's output text exactly once
and its correctness never.

**Interpretation.** ABTT improves the stage that was not the bottleneck. This
pipeline buys its precision from the parsed subject+relation screen and
bidirectional NLI, not from cosine, so a cleaner cosine screen has nothing left
to contribute. That bounds the mechanism rather than dismissing it: the measured
gains live precisely where cosine must carry precision *alone*, which is the
situation in any arena without a parseable fact template. The decisive follow-up
is to remove the regex screen and re-run both geometries.

Full campaign, pre-registration and artifacts: `stage0_results/abtt/`.

---

## 9. Methodological findings (independent of whether H-Nav works)

These are reportable results in their own right.

1. **The benchmark's headline metric is dominated by conflict-free questions**
   (§5). Anyone citing `Conflict_Resolution` accuracy is largely measuring
   something other than conflict resolution.
2. **Explicit precedence instructions are ~95% ineffective** on this model at
   this scale, and the failure is systematic (the stale value, not confusion).
3. **NLI alone cannot verify memory conflicts.** Bidirectional contradiction
   false-verifies **33–93%** of candidate pairs; the dominant class is
   same-template/different-subject ("Kyd was born in London" vs "Marlowe was born
   in London" scores contradiction at 0.9995 in both directions). A parsed
   subject-identity screen drives this to **0.000 at precision 1.000**.
4. **Reduced-precision embedding serving silently destroys retrieval fidelity**
   (§3.2): top-k agreement 1.0000 → 0.24 under bf16.
5. **vLLM at temperature 0 is not run-to-run deterministic** (§3.3), with a
   4-point accuracy swing between identical runs — and the noise is confined to
   the conflicted stratum.
6. **Retrieval-page membership is not reproducible by re-encoding** when chunks
   cluster tightly (§6.3): 26/100 agreement even with a corrected encoder.
7. **Content-addressed caches must encode every parameter that shapes the
   content.** We hit this three times — an embedding namespace missing its
   truncation length, a persisted NLI table missing its engine config, and a
   near-miss on page source. In each case a *correction* would have been
   invisible: the cache would have returned the old values and the fix would have
   measured as "no change."
8. **A pooled percentile across non-exchangeable subsets describes neither.** Our
   own frozen entropy threshold landed on one subset's median in two independent
   eras, and was arithmetically unreachable on another (2 chunks force the
   statistic to a constant).
9. **Intervention granularity must match conflict granularity** (§4): a
   precision-1.000 detector still harmed twice as often as it helped when acting
   on chunks of ~250 facts.

---

## 10. What may and may not be claimed

### May be claimed

> On the held-out `factconsolidation_sh_64k` subset, in a single pre-registered
> confirmatory run (100 questions × 5 arms, 500 calls, frozen substrate,
> benchmark-retrieved top-10 page), fact-level suppression of detector-verified
> superseded facts raised conflicted-stratum accuracy from **17/66 to 37/66** —
> **+20 net discordant pairs, McNemar exact p = 1.9 × 10⁻⁶, with zero conflicted
> questions harmed** — against an A/A noise floor measured at exactly 0/0, and at
> **no token cost (−0.31%)**. Suppression precision was **1.000** (735/735
> deleted facts independently verified as superseded). The gate was gold-free:
> the operating point was frozen on detection quality alone before any arm was
> graded. **The pre-registered protective criterion was not met** — one
> non-conflicted question regressed, where the model declined to answer although
> the fact it needed was still on the page. The mechanism is therefore
> **effective but not yet safe**, and is not recommended for deployment on
> traffic containing non-conflicted queries until that mechanism is eliminated.

### May **not** be claimed

- **Not "H-Nav improves accuracy," unqualified.** The protective claim voided,
  and that belongs in the same sentence as the improvement — never a later
  paragraph.
- **Not "effective" without scope.** Effective *on the conflicted stratum of this
  arena, at this scale.*
- **No oracle ratio at sh_64k.** The 98.4% / 95.7% ceiling-recovery figures are
  calibration-only and cross-harness.
- **No forecast from calibration.** +62/+66 and +44/+38 must never be presented
  as transferring; the pre-registration explicitly banned it.
- **Not "H-Nav's Stage-0 gate improves accuracy."** The entropy/margin
  precondition is inert in the shipped configuration; the detector's precision
  carries the result, and the precondition layer is *untested*, not validated.
- **No generalization.** One arena, one subset, one scale, one model, one shot.
- **Not "safe at scale."** The pool cap and incomplete retrieval bound both
  benefit and exposure in ways measured only here.
- **In the secondary arena:** redundancy is measured; its accuracy benefit is
  not, and the recommended follow-up would test *deduplication at retrieval* —
  strictly weaker than "the write cascade works."

---

## 11. Limitations

- **Single model, single arena.** Qwen3-4B-Instruct-2507 on one dataset family.
- **The safety failure is understood but unfixed.** "Refusal after edit" has a
  mechanism and a count, not a solution.
- **The shipped configuration disables a Stage-0 component**, for a stated
  reason (its inputs were contaminated by the truncation defect and re-fitting
  in time was impossible). It fires on every question as a result.
- **Held-out subsets carry pre-fix LLM labels.** The threshold re-fit covered the
  calibration split; `sh_64k`/`sh_262k` signal-era labels remain from before the
  truncation correction, with inputs confirmed identical.
- **`sh_262k` was never tested.** It has the highest exposure to the known
  failure mode and an earlier compensator net-harmed at that scale.
- **The evaluator can be satisfied by a wrong page** (§7), which limits what any
  accuracy number on this benchmark proves.
- **Secondary-arena runs are incomplete** — one backend measured; the LLM-mediated
  path is blocked on credentials.

---

## 12. What would come next

In order of value per unit of effort:

1. **Eliminate refusal-after-edit.** It is the single thing standing between
   "effective" and "effective and safe." A likely direction: leave a minimal
   marker where a fact was removed so the page does not read as truncated.
2. **A replicate at sh_64k** (N > 1), so a single malformed generation cannot
   decide the safety criterion.
3. **A second model.** The mechanism is a claim about how models read context;
   one model cannot support it.
4. **`sh_262k`**, declared exploratory, to test whether the effect survives where
   retrieval is most lossy.
5. **The secondary arena's accuracy question** — a token-matched retrieval-level
   A/B, which needs only a judge model and the credentials.

---

## 13. Closing note on how this was produced

The work was carried out by a team of specialized agents under an orchestrator,
with an independent supervisor auditing every deliverable adversarially. That
structure is worth describing in the thesis because it is visible in the results:

- Two safety gates fired and **both caught real faults** that would otherwise
  have produced plausible, wrong numbers.
- A calibration null was **investigated rather than accepted**, and turned out to
  be a directional finding.
- An agent **refused to spend the one-shot budget** when the registered design
  proved physically impossible, and escalated instead of substituting a design
  nobody had approved.
- An agent **downgraded its own committed claim** from "4/4" to "2/2" when only
  half had been re-verified — a change nobody would have checked.
- An agent **argued its own headline down by three-quarters** after showing the
  statistic was inflated by store size.
- A missing provenance field was **recorded as missing** rather than back-filled
  into a published artifact.
- The supervisor **corrected its own earlier note** when a builder showed its
  proposed fix would not have worked.

The final numbers survived adversarial re-derivation from raw artifacts every
time it was attempted. The claims that did not survive — write-path headroom,
chunk reranking, the redundancy-to-accuracy conversion — were reported as
negative results rather than reframed. **That ratio is the reason the positive
result is worth believing.**

---

*All figures in this report trace to committed artifacts under
`stage0_results/`. Raw measurement JSONs, pre-registrations (including the
withdrawn one), audit records and the full commit history are in the repository
on branch `claude/evomembench-hnav-analysis-nfwl9z`.*
