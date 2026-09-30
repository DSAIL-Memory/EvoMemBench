# How close does the real H-Nav detector come to the oracle ceiling?

*A strict separation of two conditions that are easy to conflate: **oracle stale
deletion** (which uses gold labels — an intervention ceiling, not a system) and
**H-Nav detector suppression** (which sees no gold — the deployable method).
Every number below is read live from a committed artifact; the extraction
scripts are `presentation_evidence/_scripts/make_oracle_gap_charts.py` and the
ad-hoc joins recorded in this file.*

Written 2026-08-19, entirely offline.

---

## 1. Executive conclusion

Where an oracle ceiling was actually measured — the calibration split, under the
whole-context harness that both arms share — **H-Nav's gold-free detector
recovers essentially all of it: 100.0% of the available gain on sh_6k (66/74 vs
the oracle's 66/74) and 95.7% on sh_32k (51/65 vs 53/65)**. The cost of
replacing gold labels with the real detector across 139 conflicted questions is
**two questions**. Detection quality is not the bottleneck: suppression
precision is **1.000** (2,673/2,673 on calibration, 735/735 on the held-out run)
with zero facts deleted that carried a key's current value. On the held-out
sh_64k confirmatory run the detector still produced a large, significant gain
(**17/66 → 37/66**, McNemar exact *p* = 1.9 × 10⁻⁶, 0 conflicted questions
harmed, tokens −0.31%), but **no oracle arm exists there**, so no ceiling ratio
can be quoted for that subset — and its lower absolute level is driven
overwhelmingly by questions the detector *never acted on* (25 of its 29
remaining errors), not by the intervention failing. The honest headline is
therefore: **the detector is close to the ceiling wherever the conflicting facts
are actually on the page; the residual gap is dominated by coverage, not by
detection precision, and one non-conflicted question regressed, so the
pre-registered safety claim is void.**

---

## 2. Which artifacts support which comparison

The single most important structural fact: **there are two different prompt
harnesses**, and only one of them supports an oracle-vs-detector ratio.

| Artifact | Arms | Prompt shape | Oracle comparison? |
|---|---|---|---|
| `stage0_results/stage1/stale_suppression_probe_sh6k.json` | native, native_repeat, **oracle_suppress**, oracle_recency, anti | one Memory block, whole context | — (is the oracle) |
| `stage0_results/stage1/stale_suppression_probe_sh32k.json` | same | one Memory block, whole context | — (is the oracle) |
| `stage0_results/stage1/detector_gap_sh6k.json` | native, native_repeat, **detector_suppress**, detector_demote_late, detector_anti | one Memory block, whole context | ✅ **VALID** |
| `stage0_results/stage1/detector_gap_sh32k.json` | same | one Memory block, whole context | ✅ **VALID** |
| `stage0_results/stage1/detector_gap_retrieval_sh6k.json` | same | rank-ordered top-k chunk page | ⚠️ cross-harness |
| `stage0_results/stage1/detector_gap_retrieval_sh32k.json` | same | rank-ordered top-k chunk page | ⚠️ cross-harness |
| `stage0_results/stage1/detector_gap_confirmatory_sh64k.json` | same | benchmark's own retrieved top-10 page | ❌ **NO ORACLE ARM AT ALL** |

The artifacts say this themselves, verbatim:

- `detector_gap_sh6k.json` → `harness.identical_to_oracle_probe`:
  > "same prompt shape, same system message, same grader, same frozen :8003
  > substrate — the headline is a RATIO against the oracle arms and a ratio taken
  > across harnesses is meaningless"

- `detector_gap_retrieval_sh6k.json` → `detector_vs_oracle.sh_6k.harness_caveat`:
  > "This run uses the RETRIEVAL-PATH harness; the oracle probe is whole-context.
  > The ratios below therefore compare across harnesses and are NOT the
  > detector/oracle ratio of the confirmatory design"

- `detector_gap_confirmatory_sh64k.json` → `corrections[0].items[4]`:
  > "NO ORACLE-CEILING RATIO EXISTS FOR sh_64k. detector_vs_oracle is empty
  > because no oracle arm was ever run there — the whole-context probe does not
  > fit the window. The 0.984 / 0.957 ratios are calibration-only and
  > cross-harness, and may not be quoted for this subset."

Confirmed mechanically: the confirmatory artifact's `detector_vs_oracle` block
is literally `{}`.

**Why the comparison on the calibration split is trustworthy.** The oracle and
detector runs are two *separate* LLM passes. The artifacts record that their
`native` arms agree exactly — `native_cross_run.identical: true`, 29/100 vs
29/100 on sh_6k and 42/100 vs 42/100 on sh_32k — and an independent per-question
recount confirms the native output string is identical on **100 of 100**
questions in both subsets. The A/A floor within each run is **0 discordant pairs
of 100**. So any difference between the oracle and detector arms is the
intervention, not run-to-run noise.

---

## 3. The comparison, exactly

### 3.1 Valid oracle comparison — calibration split, whole-context harness

Source: `results[0].by_stratum.conflicted.arms` in each artifact.

| | **sh_6k** | **sh_32k** |
|---|---|---|
| conflicted questions (n) | 74 | 65 |
| Native | 4/74 = **5.41%** | 7/65 = **10.77%** |
| Oracle stale suppression (ceiling) | 66/74 = **89.19%** | 53/65 = **81.54%** |
| H-Nav detector suppression | 66/74 = **89.19%** | 51/65 = **78.46%** |
| Oracle gain over native | +83.78 pp | +70.77 pp |
| H-Nav gain over native | +83.78 pp | +67.69 pp |
| **Gap (oracle − H-Nav)** | **0.00 pp** | **3.08 pp** |
| **Captured oracle gain** | **100.00%** | **95.65%** |

Arithmetic shown: sh_6k (89.19 − 5.41) ÷ (89.19 − 5.41) = 1.0000; sh_32k
(78.46 − 10.77) ÷ (81.54 − 10.77) = 67.69 ÷ 70.77 = 0.9565.

These reproduce the artifacts' own `detector_vs_oracle.by_mechanism.
detector_suppress.conflicted_gain_ratio` values of **1.0** and **0.9565217…**
exactly. The report's headline figures **0.984 / 0.957** are the *overall*
net-discordant-pair ratios across all 100 questions (`net_ratio`: 61/62 and
44/46), not the conflicted-stratum ratios — both are correct, they answer
slightly different questions, and they should not be swapped.

**A caveat that matters for sh_6k.** 100% is an aggregate identity, not a
question-by-question identity. Paired on the same 74 questions: **62 correct
under both**, **4 oracle-only** (indices 0, 7, 26, 60), **4 detector-only**
(indices 30, 41, 52, 86). McNemar exact *p* = 1.000 — statistically
indistinguishable, but the two arms do not fix the same set. On sh_32k: 48 both,
5 oracle-only (8, 9, 23, 32, 87), 3 detector-only (31, 57, 92), exact
*p* = 0.727. (Indices independently re-derived here match
`presentation_evidence/data/item12_detector_vs_oracle.json`
→ `per_question_cross_run` exactly.)

### 3.2 Cross-harness runs — flagged, not compared to the oracle

The retrieval-harness calibration runs are reported for completeness only. Their
`native` arms differ from the oracle probe's (28 vs 29 on sh_6k; 48 vs 42 on
sh_32k → `native_cross_run.identical: false`), so a ratio against the oracle is
not apples-to-apples.

| retrieval harness | native | detector_suppress |
|---|---|---|
| sh_6k (retrieval complete, 2/2 chunks) | 2/74 = 2.70% | 68/74 = **91.89%** |
| sh_32k (retrieval complete, 9/9 chunks) | 13/65 = 20.00% | 52/65 = **80.00%** |

### 3.3 Held-out confirmatory run — no ceiling exists

`detector_gap_confirmatory_sh64k.json`, `page_source: "benchmark"`,
`retrieval_complete: false` (10 of 17 chunks on the page).

| | sh_64k (held out) |
|---|---|
| conflicted questions | 66 |
| Native | 17/66 = **25.76%** |
| H-Nav detector suppression | 37/66 = **56.06%** |
| Gain | **+30.30 pp**; McNemar b=0 / c=20, exact *p* = 1.9073 × 10⁻⁶ |
| Oracle | **NOT MEASURED — no arm was run** |
| Captured oracle gain | **UNDEFINED** |

---

## 4. Detection quality vs downstream utility

These are two different questions and the artifacts answer them separately.

**A. Did H-Nav correctly identify stale facts?** (detection quality — no LLM
involved; measured on the frozen operating point,
`stage0_results/stage1_operating_point.json`)

| metric | calibration (sh_6k + sh_32k) | held-out sh_64k |
|---|---|---|
| facts proposed for suppression | 2,673 | 735 |
| of those, genuinely superseded | 2,673 | 735 |
| **suppression precision (`fact_precision`)** | **1.000** | **1.000** |
| incorrect suppressions (`n_suppressed_harmful`) | **0** | **0** |
| suppressions of a key's current value (`n_suppressed_same_value`) | **0** | **0** |
| conflict-pair recall in pool (`pair_recall_pool`) | **0.9784** (2,673 of 2,732) | not recorded |
| question-level recall, conflicted (`question_recall_conflicted`) | **0.9568** (133 of 139) | not recorded |
| gold-cut predictions (`n_conflicted_gold_cut`) | 2 (both sh_32k) | 2 predicted → 1 executed → **0 accuracy losses** |

**B. Did deleting them improve the answer?** (downstream utility — the LLM arms)

| run | fixed (c) | harmed (b) | McNemar exact *p* | prompt tokens |
|---|---|---|---|---|
| sh_6k whole-context, conflicted | 62 | 0 | 4.34 × 10⁻¹⁹ | −3.478% |
| sh_32k whole-context, conflicted | 45 | 1 | 1.34 × 10⁻¹² | −0.631% |
| sh_6k retrieval, conflicted | 66 | 0 | 2.71 × 10⁻²⁰ | −3.477% |
| sh_32k retrieval, conflicted | 40 | 1 | 3.82 × 10⁻¹¹ | −0.630% |
| **sh_64k confirmatory, conflicted** | **20** | **0** | **1.91 × 10⁻⁶** | **−0.307%** |

Every run's A/A floor is `b=0, c=0` over 100 questions, so none of these flips
are sampling noise.

**Why the distinction matters here.** Precision is 1.000 everywhere, yet the
answer accuracy after suppression ranges from 56% to 92%. Perfect precision
guarantees H-Nav never deletes something it shouldn't; it guarantees nothing
about whether the *right* thing was on the page to delete, or whether the model
changes its mind once it is gone. Both of those are where the remaining gap
lives.

---

## 5. Where the remaining gap comes from

Method: for every conflicted question, the queried key's full fact group is
recovered with the repository's validated offline parser
(`hnav/labeling/conflict_analysis.py::analyze`, imported, not reimplemented) and
intersected with the detector's `plan.suppress_serials`. This asks a precise
question — *did the detector act on the conflict that the question is about?*

| | sh_6k (whole-ctx) | sh_32k (whole-ctx) | sh_64k (held out) |
|---|---|---|---|
| conflicted questions | 74 | 65 | 66 |
| acted on the queried key → **fixed** | 62 | 45 | **20** |
| acted → still wrong | 6 | 9 | **4** |
| **never acted on the queried key → still wrong** | **2** | **5** | **25** |
| never acted → already right | 0 | 1 | 13 |
| acted → already right | 4 | 5 | 4 |

Consistency check: detector-correct = 62+0+4 = **66** ✅, 45+1+5 = **51** ✅,
20+13+4 = **37** ✅ — the decomposition reproduces the artifacts' headline counts
exactly.

### The four causes, separated

**(1) Imperfect stale-fact detection — small on calibration, and never a
precision problem.** On sh_6k the entire 4-question shortfall against the oracle
is: 2 detection misses (q26, q60 — the stale serial was simply not in the
detector's plan) and 2 questions where the detector *did* delete the right stale
fact and the answer still did not become correct (q0: stale 223 suppressed,
native 'ice hockey' → detector 'association football', gold 'pesäpallo'; q7:
stale 31 suppressed, native 'American football' → detector 'cricket', gold 'muay
thai'). On sh_32k, 3 of the 5 are detection misses (q23, q32, q87).

**(2) A rule mismatch on the questions where gold is *not* the newest fact.**
sh_32k q8 and q9 are the only two conflicted questions in the calibration split
with `gold_is_latest: false`. The oracle deletes every non-gold value and gets
them right; H-Nav applies the *stated recency rule* — keep the highest serial —
and therefore suppresses the **gold** fact (serial 707 on q8, 1291 on q9) and
gets them wrong. This is not a detector defect; it is the benchmark's own rule
being violated by 2 of 139 calibration items, and H-Nav following the rule. It
accounts for 2 of the 3.08 pp sh_32k gap. On the held-out run the same situation
arose twice and cost nothing: per `corrections[0].items[2]`, q18's gold was never
suppressed and native was already wrong, while q20's gold *was* suppressed and
the arm answered correctly anyway — 2 predicted gold-cuts, 1 deletion, **0
accuracy losses**.

**(3) LLM behaviour that no suppression can fix — this is what caps the oracle
itself.** The oracle fails on 8/74 (sh_6k) and 12/65 (sh_32k) conflicted
questions. By the arm's own definition ("delete every non-expected-value fact of
the queried key"), after the edit the *only* value of that key remaining on the
page is the gold one — so an answer naming a different value cannot have been
read off the page. **19 of these 20 name the real-world-true value** instead of
the counterfactual gold: Germany → 'Europe' (gold 'africa'), France → 'Paris'
(gold 'harare'), Microsoft HQ → 'Redmond' (gold 'beverly hills'), Apple CEO →
'Tim Cook' (gold 'vijay mallya'), US head of state → 'Donald Trump' (gold
'connachta' — and note 'Donald Trump' is also the answer printed in the prompt's
own one-shot example). *Deduction from the arm definition, not speculation:* the
model is overriding the page with parametric knowledge. The single exception is
sh_32k q67 ('Sue Grafton | is a citizen of', gold 'ukraine'), where the oracle
arm emitted 'United Kingdom' — neither the gold nor the real-world value
('United States of America', which is what native answered); this one is
unexplained by the artifact and no mechanism should be asserted for it.
**Parametric override is the true ceiling on the whole intervention family, and
it binds the oracle and H-Nav identically.**

**(4) Coverage — and it is the dominant term on the held-out run.** Of the 29
conflicted questions still wrong after detector suppression on sh_64k, **25
(86%) are questions where the detector never touched the queried key's conflict
at all**; only 4 are cases where it acted and the answer did not flip. Compare
the same figure on calibration: 2/74 and 5/65. The structural difference is
visible in the artifacts:

- sh_64k retrieval is **incomplete**: `n_chunks_total: 17`, `n_chunks_on_page: 10`,
  `retrieval_complete: false`. The page carries 2,560–2,823 facts (44 distinct
  page sizes across questions) out of a 4,580-fact store — roughly 56–62%.
- The detector verifies **7.3 conflict groups per question** on sh_64k versus
  **14.2** (sh_6k) and **12.6** (sh_32k), despite the store being larger.
- Both calibration retrieval runs have `retrieval_complete: true` (2/2 and 9/9
  chunks) and reach 91.9% and 80.0%.

**What cannot be concluded from these artifacts.** Whether each specific missed
pair was absent from the page or present-but-screened-out is **not verifiable
here** — per-question detector pools were not recorded. The artifact is explicit
about this and withdrew an earlier attribution of exactly this shape
(`corrections[0].items[3]`: *"The pool-cap attribution offered for the missed
prediction is WITHDRAWN: per-question pools were not recorded in this artifact,
so it was not verifiable."*). What *is* certain from the detector's design is
that a contradiction can only be verified when **both** members of a pair are on
the page. Any sharper attribution requires the re-run that now records `n_pool`.

**(5) Intervention side effects — one, and it voided the safety claim.** On
sh_64k, one non-conflicted question regressed: index 77, class
`refusal_after_edit` — native answered `"John Milton"`, the edited arm answered
`"The provided knowledge pool does not contain any information about"`, with
`gold_cut: false` (the fact it needed was still on the page). Pre-registration
void condition 5 fires: `protective_claim_void: true`, while
`run_void: false` — the accuracy result stands, the safety claim does not. On
calibration the same side effect appears once (sh_32k b=1, the gold-not-latest
q8) and never on sh_6k.

### Verdict on the cause of the gap

- **Calibration split (ceiling measured, conflicts fully visible):** the gap is
  ~0–3 pp and is *not* caused by detection precision (1.000) or by intervention
  side effects. It is a **combination of (a) a handful of recall misses, (b) two
  gold-is-not-latest items where H-Nav correctly follows the stated rule and the
  benchmark does not, and (c) LLM parametric override, which caps the oracle too.**
- **Held-out sh_64k:** the dominant term is unambiguously **coverage** — 25 of 29
  residual errors are questions the detector never acted on, under demonstrably
  incomplete retrieval. Detection precision and intervention safety are not the
  binding constraints on the gain; they are the binding constraint on
  *deployability*, via the single refusal.

---

## 6. Thesis interpretation

**Does oracle deletion show that stale conflict is causally important?** Yes,
about as cleanly as this design allows. Deleting the stale fact moves
conflicted-stratum accuracy from 5.4% → 89.2% (sh_6k) and 10.8% → 81.5%
(sh_32k), paired, one edit per question, against an A/A floor of exactly zero
discordant pairs. The presence of the superseded sentence — not retrieval
failure, since both facts are on the page — is what makes the model wrong.

**How much of that opportunity does the real system recover?** Where the ceiling
was measured: **100.0%** and **95.7%** of the available gain, i.e. the price of
removing gold labels was **2 questions out of 139**. Where the ceiling was not
measured (sh_64k), H-Nav still delivers **+30.3 pp** with *p* = 1.9 × 10⁻⁶ and
zero conflicted questions harmed — but the fraction of the ceiling that
represents is **unknown and must not be asserted**.

**Detection or intervention?** The evidence says **detection is the finished
part and the intervention is the contribution that still carries risk.** The
detector's precision is 1.000 on 3,408 verified deletions across all runs, and
its recall is high enough that swapping it in for gold costs almost nothing. The
intervention, by contrast, is where the mechanism ranking was discovered
(suppress ≫ demote: on sh_6k the demote arm captures only 18.75% of the oracle
recency gain, and on sh_32k 34.6%) and where the one safety failure lives. A
fair one-line framing: *H-Nav's contribution is a gold-free detector accurate
enough to substitute for an oracle, plus the demonstration that fact-level
suppression — not reranking, not demotion — is the mechanism that converts that
detection into accuracy.*

**Does H-Nav "solve" the problem?** No, and three separate facts forbid the word:
the protective criterion voided on one non-conflicted question; even a perfect
oracle leaves 8–12 conflicted questions per subset unfixed because the model
overrides the page with world knowledge; and on the one held-out subset the
system never acted on 25 of 66 conflicted questions.

**Strongest defensible claim.**

> On the calibration split, under a harness identical to the oracle probe,
> H-Nav's gold-free detector recovers **100% (sh_6k) and 95.7% (sh_32k)** of the
> accuracy gain available to a gold-labelled oracle that deletes stale facts —
> at suppression precision 1.000 over 2,673 verified deletions, with thresholds
> frozen on detection quality alone before any answer was graded. On held-out
> sh_64k, in a single pre-registered confirmatory run on the benchmark's own
> retrieved page, the same frozen detector raised conflicted-stratum accuracy
> from 17/66 to 37/66 (McNemar exact *p* = 1.9 × 10⁻⁶, zero conflicted questions
> harmed, tokens −0.31%). **No oracle ceiling exists at sh_64k, so the recovery
> ratios may not be transferred to it. The pre-registered protective criterion
> was not met — one non-conflicted question regressed into a refusal — so the
> mechanism is effective but not yet safe**, and the remaining shortfall at scale
> is driven mainly by conflicts the detector never acted on under incomplete
> retrieval (25 of 29 residual errors), not by mis-detection.

---

## 7. Figures

- **`figures/fig16_oracle_vs_detector.png`** — grouped bars, conflicted-stratum
  accuracy. Left panel: the valid comparison (sh_6k, sh_32k) with all three bars,
  the oracle hatched and explicitly labelled a ceiling. Right panel, divided off
  and annotated: sh_64k with only two bars, because no oracle arm exists.
- **`figures/fig17_captured_oracle_gain.png`** — fraction of the oracle gain
  captured, with the arithmetic printed inside each bar; sh_64k is absent by
  necessity.

**Slide caption.** Deleting the stale fact with gold knowledge lifts
conflicted-question accuracy from 5.4% to 89.2% (sh_6k) and 10.8% to 81.5%
(sh_32k) — proof that the superseded sentence itself causes the error. Running
the *same* experiment with H-Nav's gold-free detector recovers 100% and 95.7% of
that gain, so replacing the oracle with the real system costs two questions out
of 139. On held-out sh_64k the detector still gains +30.3 pp (*p* = 1.9 × 10⁻⁶,
no conflicted question harmed), but no oracle was run there, so the ceiling —
and the recovery ratio — is unmeasured at that scale.
