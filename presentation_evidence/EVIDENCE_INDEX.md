# Presentation Evidence Index

Built offline on 2026-08-18 from committed artifacts only — no GPU, no LLM, no
network. Every number below was read from a JSON artifact (or a source file) by
the scripts in `presentation_evidence/_scripts/` and re-derived where it is a
sum, rate, or ratio. Where a report and an artifact disagree, the artifact's
value is used and the discrepancy is flagged.

Directory contents:

- `EVIDENCE_INDEX.md` — this file, one section per item.
- `figures/` — 13 charts, each as 300-dpi PNG **and** SVG.
- `data/` — extracted numbers per item (`.csv` / `.json` / `.txt`).
- `_scripts/` — the extraction and chart scripts (reproducibility; every number
  in this file is printed by one of them).

---

## Item 1 — Old and new fact really are in the same retrieved context

**Claim:** the problem is not that retrieval fails to find the right fact; the
old and the new fact arrive *together*, on the same page the model reads.

**Source:**

- Dataset: `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json`,
  entry index **4** (`metadata.qa_pair_ids[0] = "factconsolidation_sh_6k_no0"`), field `context`:
  - serial **91**: `Nobuhiro Watsuki is famous for Rurouni Kenshin.`
  - serial **259**: `Nobuhiro Watsuki is famous for The Fairly OddParents.`
  - `questions[1] = "What is Nobuhiro Watsuki famous for?"`, `answers[1] = ["The Fairly OddParents"]`
- Model behaviour: `stage0_results/stage1/stale_suppression_probe_sh6k.json`
  → `results[0].per_question[]` where `index == 1`:
  `arms.native.output = "Rurouni Kenshin"` (wrong),
  `arms.oracle_suppress.output = "The Fairly OddParents"` (right),
  `plan = {gold_serials: [259], stale_serials: [91], latest: 259, gold_is_latest: true}`.

**Two more examples** (same file, same filter — conflicted, native wrong,
oracle-suppress right, exactly one gold + one stale serial; all 62 candidates in
`data/item01_examples.json`):

| index | question | old (stale) fact | new (gold) fact | native says | after suppressing the old fact |
|---|---|---|---|---|---|
| 19 | Who is the chief executive officer of Microsoft? | #85 `The chief executive officer of Microsoft is Satya Nadella.` | #188 `The chief executive officer of Microsoft is Steve Jobs.` | `Satya Nadella` ✗ | `Steve Jobs` ✓ |
| 29 | Which city did Oscar Wilde die in? | #128 `Oscar Wilde died in the city of Paris.` | #270 `Oscar Wilde died in the city of Guangzhou.` | `Paris` ✗ | `Guangzhou` ✓ |

(Gold answers from `Conflict_Resolution.json` entry 4, `answers[19]` and `answers[29]`.
Note these benchmark "facts" are deliberately counterfactual — the prompt
instructs the model to answer *only* from the knowledge pool, which is exactly
why these examples also show the model is not just using world knowledge when
it errs: `Satya Nadella` and `Paris` are both *in the pool*, at lower serials.)

**Reproduced page:** `data/item01_page_excerpt.txt` — prompt head, the lines
around serials 91 and 259, and the prompt tail with the question. Reproduced
with `hnav/stage1/stale_suppression_probe.py::render_context` and verified
**byte-identical** to the dataset's `context` field (455 facts, 26,157 chars;
full prompt 26,959 chars). Screenshot the excerpt file, or the dataset JSON at
entry 4 directly.

**Limitation to state:** no artifact stores the *raw retrieved page* of the
benchmark runs — run files keep `query` and `input_len` but not the memory
blocks. For sh_6k this is immaterial: the store is 2 chunks and both are always
retrieved (`retrieve_num = 10`), so the page *is* the whole context and is
reproduced byte-exactly above. For sh_64k the page cannot be reproduced
offline: the confirmatory prepass lives on the GPU box
(`detector_gap_confirmatory_sh64k.json` → `detector_inputs.prepass` =
`/mnt/nvmes/nvme1/egekutlu/EvoMemBench/hnav/_out/stage1_prepass_sh_64k_benchmarkpage.json`).

**Figure:** NO CHART — text exhibit. Screenshot `data/item01_page_excerpt.txt`
(or the two dataset lines directly). A chart of two sentences would be fake.

**Caveat:** the dataset's facts are synthetic counterfactuals (Steve Jobs at
Microsoft, Wilde dying in Guangzhou). Say so before the advisor spots it; it is
by design (the pool must beat world knowledge, per the prompt instruction).

---

## Item 2 — Stale-fact dominance: 572 of 575 errors

**Claim:** the model is not picking a wrong entity or relation. It finds the
correct memory slot and reads the superseded value out of it.

**Source:** `stage0_results/question_strata.json` →
`aggregate.errors_total = {"stale_value": 572, "off_list": 3, "empty": 0}`.

**Recomputed** by summing `runs[].strata.{unique,conflicted,ambiguous,unmatched}.errors`
over all 8 runs: 572 + 3 + 0 = **575** — matches the summary field exactly.
The unique stratum contributes **zero** errors of any class in all 8 runs
(it is at 26/26 everywhere), so all 575 errors are conflicted-stratum errors.

**Error-class definitions** (quoted verbatim from `question_strata.json` → `definitions`):

> `stale_value`: "incorrect output contains a non-expected value OF THE SAME KEY"
> `off_list`: "incorrect output contains no value of the queried key"
> `empty`: "output is empty after the evaluator's normalization"

**Numbers** (per-run conflicted stratum, n = 74 each; full table in
`data/item02_error_classes.csv`):

| run | correct | stale_value | off_list | empty |
|---|---|---|---|---|
| detA | 1 | 72 | 1 | 0 |
| detB | 0 | 73 | 1 | 0 |
| detC | 5 | 69 | 0 | 0 |
| detD | 1 | 72 | 1 | 0 |
| off | 3 | 71 | 0 | 0 |
| offA | 0 | 74 | 0 | 0 |
| offB | 4 | 70 | 0 | 0 |
| shadow | 3 | 71 | 0 | 0 |

**Figure:** `fig02a_error_classes_per_run.(png|svg)` (per-run stacked — the one
for the deck) and `fig02b_error_classes_total.(png|svg)` (aggregate bar).

**Caveat:** all 8 runs are **sh_6k** — the 572/575 figure is one subset, eight
runs, not eight subsets. Also `stale_value` is substring-based classification;
the artifact records the classifier definitions precisely so it is defined, not
eyeballed, but it is still a string-matching rule.

---

## Item 3 — Fine without conflict, collapses with conflict

**Claim:** the failure is not general language understanding and not retrieval;
the collapse is conditional on conflict.

**Source:** `stage0_results/question_strata.json` → `runs[]` (8 entries) and
`aggregate`: `unique_accuracy_min = unique_accuracy_max = 1.0`;
`conflicted_accuracy_min = 0.0`, `max = 0.06756…`. Raw runs:
`stage0_results/t4_s2_evidence/sh_6k_{off,offA,offB,detA,detB,detC,detD,shadow}_results.json`;
stratum membership from `subsets[0].indices` (26 unique / 74 conflicted).

**Independent recount:** run `sh_6k_off` was re-graded from its raw `data`
rows against `subsets[0].indices`: unique 26/26, conflicted 3/74 — matches the
artifact. Overall accuracy re-derived for every run as
(unique_correct + conflicted_correct)/100 and asserted equal to
`accuracy_overall` (script `_scripts/extract_main.py`).

**Numbers** (full CSV: `data/item03_strata_accuracy.csv`):

| run | unique | unique acc | conflicted | conflicted acc | overall |
|---|---|---|---|---|---|
| detA | 26/26 | 1.000 | 1/74 | 0.014 | 0.27 |
| detB | 26/26 | 1.000 | 0/74 | 0.000 | 0.26 |
| detC | 26/26 | 1.000 | 5/74 | 0.068 | 0.31 |
| detD | 26/26 | 1.000 | 1/74 | 0.014 | 0.27 |
| off | 26/26 | 1.000 | 3/74 | 0.041 | 0.29 |
| offA | 26/26 | 1.000 | 0/74 | 0.000 | 0.26 |
| offB | 26/26 | 1.000 | 4/74 | 0.054 | 0.30 |
| shadow | 26/26 | 1.000 | 3/74 | 0.041 | 0.29 |

**Figure:** `fig03_strata_collapse.(png|svg)` — grouped bars, ceiling line at 1.0.

**Caveat:** one subset (sh_6k), one model. The unique stratum is only 26
questions; 26/26 × 8 runs is strong but narrow. `question_strata.json` was
generated with uncommitted producer code at generation time
(`producer_uncommitted_at_generation: true`) — the raw run files it reads are
committed, and the recount above reproduces its numbers from them.

---

## Item 4 — The prompt already states the recency rule

**Claim:** the model is *told* the rule explicitly and still takes the stale fact.

**Source:** `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/utils/templates.py`,
`BASE_TEMPLATES['factconsolidation']['query']` — line **40** `'long_context_agent'`,
line **41** `'rag_agent'` (near-identical), line **42** `'agentic_memory_agent'`
(archival-memory variant). Load-bearing sentences, verbatim from line 41:

> "Each fact in the knowledge pool is provided with a serial number at the
> beginning, and the newer fact has larger serial number."

> "You need to solve the conflicts of facts in the knowledge pool by finding
> the newest fact with larger serial number."

And the world-knowledge pre-emption, same line:

> "You should give a very concise answer without saying other words for the
> question **only** from the knowledge pool you have memorized rather than the
> real facts in real world."

**Which template the runs used:** the `rag_agent` template. The oracle/detector
probes copy it verbatim into `hnav/stage1/calibrate_read_policy.py:92-101`
(comment at lines 86-90: "the benchmark's own prompt shape, verbatim …
the Conflict_Resolution 'rag_agent' query template"), and every probe artifact
records `harness.prompt_source = "hnav.stage1.calibrate_read_policy (imported
verbatim)"` with `prompt_shape = "RAGSystem: 'Memory 1:\n<whole context>\n' +
templated query"`. The t4_s2 evidence runs are benchmark RAG-agent runs of the
same dataset family. The reproduced prompt tail in `data/item01_page_excerpt.txt`
shows the template around a real question.

**Figure:** NO CHART — screenshot `utils/templates.py:41` with the two
sentences highlighted (a quote is not a chart).

**Caveat:** the instruction sentence is grammatically awkward ("the newer fact
has larger serial number") — it is the benchmark's wording, not ours. Quote it
exactly; do not clean it up on the slide.

---

## Item 5 — Position / ordering experiment

**Claim:** the physical position of a fact inside the prompt changes the
answer. We are **not** claiming the model has no notion of the recency rule —
we are claiming that despite the explicit instruction, positional competition
exerts a strong causal effect on the response.

**Sources:**
- `stage0_results/stage1/stale_suppression_probe_sh6k.json` / `..._sh32k.json`
  → `results[0].by_stratum.conflicted.{arms, paired_vs_native}`
- `stage0_results/stage1/detector_gap_confirmatory_sh64k.json` — held-out
  mirror (arms `detector_demote_late` = newest→END, `detector_anti` = newest→FRONT),
  **different harness** (rank-ordered multi-block page vs one whole-context block).
- NEW/OLD/OTHER taxonomy: `data/item05_taxonomy.json`, produced by
  `position_taxonomy.py`. **Provenance:** this script is committed on branch
  `origin/claude/repo-analysis-advisor-doc-z2fra2` (commit `50dd955`), not on
  the current branch. It was materialized with
  `git show 50dd955:hnav/stage1/position_taxonomy.py` (copy kept in
  `_scripts/position_taxonomy.py`) and re-run offline — it reads only committed
  artifacts. Background doc: `HNAV_POSITION_VS_RECENCY.md` on the same branch
  (copy in `_scripts/`).

**Numbers — conflicted-stratum accuracy by placement arm:**

| subset | baseline (serial order) | newest → END | newest → FRONT | McNemar newest→END vs native |
|---|---|---|---|---|
| sh_6k (n=74, oracle probe) | 4/74 = 5.4% | 20/74 = 27.0% | 1/74 = 1.4% | b=1, c=17, net +16, p = 1.4e-4 |
| sh_32k (n=65, oracle probe) | 7/65 = 10.8% | 33/65 = 50.8% | 4/65 = 6.2% | b=2, c=28, net +26, p = 8.7e-7 |
| sh_64k (n=66, held out, detector harness) | 17/66 = 25.8% | 20/66 = 30.3% | 15/66 = 22.7% | b=2, c=5, net +3, p = 0.45 |

(McNemar values from `results[0].by_stratum.conflicted.paired_vs_native` of
each file — the "newest → END" arm is `oracle_recency` on the probes and
`detector_demote_late` on the confirmatory run; all in `data/item05_arms.json`.)

**NEW / OLD / OTHER taxonomy** (which value the model actually names;
A/A repeat row = 0 changed answers on every subset — the noise floor is
exactly zero):

- sh_6k: native NEW 4 / OLD 70; newest→END NEW **20** / OLD 54; newest→FRONT NEW **1** / OLD 72.
- sh_32k: native NEW 7 / OLD 58; newest→END NEW **33** / OLD 32; newest→FRONT NEW **4** / OLD 61.
- sh_64k (held out): native NEW 17 / OLD 44 / OTHER 5; newest→END NEW 20; newest→FRONT NEW 15.

**Three clean paired examples** (sh_6k, native vs oracle_recency disagree; full
set in `data/item05_paired_examples.json`):

| index | subject | native (serial order) | newest fact moved to END | gold |
|---|---|---|---|---|
| 12 | Robert Parish | `center` ✗ | `quarterback` ✓ | quarterback |
| 29 | Oscar Wilde | `Paris` ✗ | `Guangzhou` ✓ | Guangzhou |
| 58 | Hermione Granger | `Emma Watson` ✗ | `Kylie Minogue` ✓ | Kylie Minogue |

Counter-direction case worth volunteering: index 80 (Lisa Leslie) — native was
*right* and newest→END made it wrong (the single `b_native_only` on sh_6k).

**Figures:** `fig05a_position_arms.(png|svg)` (accuracy by placement arm ×
subset) and `fig05b_taxonomy.(png|svg)` (NEW/OLD/OTHER stacked, A/A row included).

**Caveat:** sh_64k's placement arms are the *detector's* demote/anti edits on a
retrieved multi-block page — a different harness with a much weaker, non-significant
effect (p = 0.45). Label it as the attenuated held-out mirror, never pool it
with the oracle-probe rows. And the taxonomy script came from a sibling branch —
say so if asked (provenance above).

---

## Item 6 — Does geometry actually separate conflicts?

**Claim:** conflicting facts sit far closer in embedding space than random pairs.

**Source:** `stage0_results/final/m1_geometry_calibration.json` (list of 4, one
per subset). Values read (p50, with AUC and pair counts):

| subset | conflict `whole_blob_sim` p50 | control p50 | `diff_sim` p50 | AUC | n conflict / control | gate_pass |
|---|---|---|---|---|---|---|
| sh_6k | 0.9636 | 0.5977 | 0.6872 | 1.0000 | 160 / 159 | true |
| sh_32k | 0.9638 | 0.6051 | 0.7125 | 1.0000 | 835 / 830 | true |
| sh_64k | 0.9638 | 0.6021 | 0.7165 | 1.0000 | 1,687 / 1,685 | true |
| sh_262k | 0.9641 | 0.6109 | 0.7239 | 0.9999 | 7,197 / 7,197 | true |

Full p10/p90 and means per series are in `data/item06_geometry_percentiles.csv`;
model/dtype are recorded in the artifact per subset.

**Limitation (state on the slide):** this artifact stores **summary percentiles
only** — `{mean, p10, p50, p90}` per series. The raw per-pair similarity arrays
were not saved, so a histogram or ROC curve **cannot** be drawn offline; the
AUC is quoted from the artifact, not re-derived. Re-drawing a distribution
would require re-running `hnav/stage0/m1_geometry_calibration.py` with the
embedder on the GPU box.

**Figure:** `fig06_geometry_percentiles.(png|svg)` — p10–p50–p90 interval plot,
three series × four subsets, AUC annotated, explicitly titled as a percentile
summary.

**Caveat:** AUC 1.0 on three subsets sounds too good; it is separation of
*conflict pairs vs random control pairs*, which is an easy discrimination —
the hard problem (which same-key pair is a true supersession) is items 7/10.
Don't let the slide imply AUC 1.0 solves the task.

---

## Item 7 — Geometry-only grouping ablation

**Claim:** conflict structure is genuinely present in the embedding geometry —
it is not the fact template and serial metadata doing the work.

**Source:** `stage0_results/final/m1b_grouping_ablation.json` — per subset:
`best_f1`, `equal_coverage`, `recall_ceiling_from_knn`, counts, and a full
50-point `pr_curve` sweep (tau from 0.5 up). F1 at the best point re-derived
from its own precision/recall and asserted equal.

| subset | best F1 | at tau | precision / recall at best | equal-coverage F1 | knn recall ceiling | truth pairs |
|---|---|---|---|---|---|---|
| sh_6k | 0.8916 | 0.91 | 0.861 / 0.925 | 0.8750 | 1.0 | 160 |
| sh_32k | 0.8392 | 0.93 | 0.847 / 0.831 | 0.8371 | 1.0 | 835 |
| sh_64k | 0.8211 | 0.94 | 0.896 / 0.758 | 0.8222 | 1.0 | 1,687 |
| sh_262k | 0.7569 | 0.95 | 0.829 / 0.696 | 0.7602 | 1.0 | 7,197 |

(Best-point precision/recall in `data/item07_pr_curves.csv`; summary in
`data/item07_summary.json`.)

**The artifact's own `interpretation` field, verbatim** (identical in all four
entries — written into the output deliberately; use this wording):

> "Geometry that recovers the regex grouping *without parsing* is what licenses
> applying H-Nav to CrossEp-Know, where no templates and no serial numbers
> exist. High F1 -> the detector is validated. Low F1 -> any downstream gain is
> attributable to the metadata, not to geometry, and must be reported as such."

**Figures:** `fig07a_pr_curve.(png|svg)` (PR sweep, best-F1 points marked) and
`fig07b_f1_vs_tau.(png|svg)`.

**Caveat:** F1 declines with store size (0.89 → 0.76 across 455 → 18,332
facts); the recall ceiling of 1.0 is the *kNN candidate* ceiling, not achieved
F1. Truth pairs come from the regex parser (item 8), so this is
geometry-vs-parser agreement, not geometry-vs-human agreement.

---

## Item 8 — How the parser works (pointer only)

**Claim:** relation, subject and value are not extracted by another LLM; a
validated deterministic regex parser runs over the benchmark's templated facts.

**Source:** `hnav/labeling/conflict_analysis.py::parse`, **lines 53–68**
(prefix/suffix relation tables begin at line 12; `parse` returns
`(relation_key, subject, object) or None`). Do-not-rewrite rule: `CLAUDE.md`
hard invariants.

**Worked example — actually executed** (`_scripts/extract_dataset_items.py`):

```
parse("Nobuhiro Watsuki is famous for Rurouni Kenshin.")
  -> relation_key = '| is famous for '   subject = 'Nobuhiro Watsuki'   object = 'Rurouni Kenshin'
```

**Coverage** (read from both Stage-0 artifacts; per-subset):

| subset | `parse_coverage` (m1_geometry_calibration.json) | `parse_coverage_pct` (m1b_grouping_ablation.json) |
|---|---|---|
| sh_6k | 99.56 | 99.56 |
| sh_32k | 99.65 | 99.65 |
| sh_64k | 99.63 | 99.63 |
| sh_262k | 99.47 | 99.47 |

(Also in `data/item08_parser.json`.)

**Figure:** NO CHART — method slide. Screenshot `conflict_analysis.py:53-68`.

**Caveat:** coverage is high because the benchmark's facts are templated; the
parser is validated *for this arena only* and would not transfer to free text.

---

## Item 9 — Span residual (rationale only)

**Source locations, confirmed:**

- QR-residual function: `hnav/core/geometry.py:149` (`def qr_residual(...)`).
- Thresholds: `stage0_results/stage1_operating_point.json` → `thresholds` =
  `{cos_pair: 0.9, r_min: 0.44, nmargin: 0.00476..., H_z: 1.9569...,
  ambiguity_mode: "none", nli_contradiction: 0.9}` (`r_min_label: "loose"`).

**Numerical consistency check, computed:** `sqrt(1 − 0.44²) = 0.8980` — a pair
that clears the residual screen `r ≥ 0.44` has in-span cosine ≤ 0.898, i.e.
just under the `cos_pair = 0.90` screen. The two screens meet at essentially
the same geometric point; neither one overrides the other.
(`data/item09_thresholds.json`.)

**Figure:** NO CHART.

**Caveat:** `ambiguity_mode: "none"` means the frozen Stage-0 `nmargin`/`H_z`
precondition is **disabled** at the shipped operating point (the artifact's own
`ambiguity_note` says so). If the advisor asks what H_z does in the shipped
configuration: nothing — it fires on every question (see final report §10:
"the precondition layer is untested, not validated").

---

## Item 10 — NLI alone is not safe

**Claim:** bidirectional NLI on its own false-verifies a large fraction of
pairs; adding the parsed subject-identity screen drives that to exactly zero.

**Source:** `stage0_results/stage1/stage1_calibration.json` → `cells` (162
objects = 81 with `pair_filter: true` + 81 with `pair_filter: false`).

**Recomputed per cell** as `(n_fv_diff_key + n_fv_same_object) / n_verified`
(asserted equal to the stored `false_verified_rate` in all 162 cells; every
cell has `n_verified > 0`):

- `pair_filter == false` (screen OFF): FV rate range **0.3150 – 0.9409** over 81 cells.
- `pair_filter == true` (screen ON): FV rate **exactly 0.0 in all 81 cells**.

Grid (`provenance.grid`): `cos_pair {0.90, 0.92, 0.94}` × `r_min {frozen,
loose, off}` × `ambiguity_mode {all, any, none}` × `nli_contradiction {0.5,
0.9, 0.99}` × `pair_filter {true, false}` = 162 cells. NLI model
(`provenance.nli_model`): `cross-encoder/nli-deberta-v3-large`.

**Precision at the shipped operating point** (`stage0_results/stage1_operating_point.json`
→ `metrics`): `pair_precision = 1.0`, `fact_precision = 1.0`, `tp = 2673`,
`fp = 0`, `n_questions = 200` — **calibration split (sh_6k + sh_32k), not held-out**.

**Kyd/Marlowe example — provenance is markdown only:** `TEZ_BULGULARI.md`
lines 265–267: *"Thomas Kyd was born in the city of London."* vs *"Marlowe was
born in the city of London."* → contradiction **0.99949 / 0.99983** in the two
directions. There is **no JSON artifact** behind these two scores; they were
measured on the box and recorded in the markdown. Quote with that provenance.
(Note: it is 0.99949 in one direction and 0.99983 in the other — not "0.99949
in both directions".)

**Figure:** `fig10_nli_false_verification.(png|svg)` — per-cell FV-rate strip,
screen OFF vs screen ON; the ON group collapsing to a single point at zero is
the message. Data: `data/item10_11_cells.csv`, `data/item10_summary.json`.

**Caveat:** the zero is *on this arena's templated facts*, where the parser is
near-perfect (item 8); the screen inherits the parser's domain. And 2673/0 is
calibration — the held-out precision figure is item 14's 735/735, a different
number from a different subset. Never merge them.

---

## Item 11 — Why chunk-level intervention was rejected

**Claim:** conflict is fact-level; a chunk-level intervention is too coarse —
moving a ~230–260-fact chunk to fix one conflict displaces hundreds of
unrelated facts. Intervention granularity should match conflict granularity.

**Source:** same file as item 10 — `stage1_calibration.json` → `cells[].{helped, harmed, pair_filter}`.
Narrative: `STAGE1_NULL_ANALIZI.md`.

**Recomputed aggregates:**

| group | helped (Σ) | harmed (Σ) | net-positive cells |
|---|---|---|---|
| subject screen ON (81 cells) | 228 | 441 | **0 of 81** |
| subject screen OFF (81 cells) | 354 | 426 | 21 of 81 |

Every one of the 21 net-positive cells has `pair_filter == false` — i.e. only
detectors that false-verify 31–94% of pairs (item 10) ever look net-positive
at chunk granularity. Verified: net-positive ∧ `pair_filter == true` count = 0.

**Figure:** `fig11_helped_vs_harmed.(png|svg)` — helped × harmed scatter, all
162 cells, marker area = number of coincident cells, y = x diagonal drawn.
Every screen-ON point sits on or above the diagonal.

**Caveat:** "helped/harmed" counts changed *questions* under chunk moves on the
calibration split; the ~230–260 facts-per-chunk figure is the arena's chunking
(store sizes / 2 and /9 chunks), not stored in this artifact — don't put a
per-chunk fact count on the slide from this file.

---

## Item 12 — How close does the detector get to the oracle?

**Claim:** on calibration data, the ground-truth-free detector recovers roughly
96–98% of the perfect intervention's net gain.

**Source:** `stage0_results/stage1/detector_gap_sh6k.json` and `..._sh32k.json`
→ `detector_vs_oracle.<subset>.by_mechanism.detector_suppress`:

| subset | oracle net (vs native) | detector net | ratio (recomputed) | native cross-run identical? |
|---|---|---|---|---|
| sh_6k | +62 | +61 | 61/62 = **0.984** | yes — 100/100 questions, output-identical |
| sh_32k | +46 | +44 | 44/46 = **0.957** | yes — 100/100 |

Harness comparability, artifact's own words
(`harness.identical_to_oracle_probe`): "same prompt shape, same system message,
same grader, same frozen :8003 substrate - the headline is a RATIO against the
oracle arms and a ratio taken across harnesses is meaningless."

**Do not use** `detector_gap_retrieval_sh{6,32}k.json` for the headline: those
carry `harness_match: false` and this caveat, verbatim: "This run uses the
RETRIEVAL-PATH harness; the oracle probe is whole-context. The ratios below
therefore compare across harnesses and are NOT the detector/oracle ratio of the
confirmatory design…".

**Missed cases (derived, cross-run by question index):** conflicted questions
where the oracle fixed it and the detector did not — sh_6k: indices
{0, 7, 26, 60}; sh_32k: {8, 9, 23, 32, 87}. Detector-right/oracle-wrong —
sh_6k: {30, 41, 52, 86}; sh_32k: {31, 57, 92}. These compare two separate LLM
passes (native outputs were verified identical question-by-question, 100/100 on
both subsets). In `data/item12_detector_vs_oracle.json`.

**Hard limit on this claim** — the confirmatory artifact's own correction
(`detector_gap_confirmatory_sh64k.json` → `corrections[0].items[4]`, verbatim):

> "NO ORACLE-CEILING RATIO EXISTS FOR sh_64k. detector_vs_oracle is empty
> because no oracle arm was ever run there - the whole-context probe does not
> fit the window. The 0.984 / 0.957 ratios are calibration-only and
> cross-harness, and may not be quoted for this subset."

**Figure:** `fig12_detector_vs_oracle.(png|svg)` — oracle vs detector net gain,
ratios printed.

**Caveat:** say "calibration only" out loud. Note the correction labels the
ratios "cross-harness" relative to the sh_64k confirmatory design (the gap runs
match the *oracle probe's* whole-context harness, not the confirmatory
retrieval-page harness) — so the ratio may not be projected onto item 13's
result at all.

---

## Item 13 — Final held-out result

**Claim:** on held-out sh_64k, in one pre-registered confirmatory run,
fact-level suppression of detector-verified superseded facts raised
conflicted-stratum accuracy 25.8% → 56.1% (+20 net, McNemar exact
p = 1.9×10⁻⁶) at −0.31% prompt tokens, with 0 conflicted questions harmed —
**and the pre-registered protective criterion was voided by one non-conflicted
question**. Effective, but not yet safe.

**Source:** `stage0_results/stage1/detector_gap_confirmatory_sh64k.json`
(preregistration: `stage0_results/stage1_preregistration_v2.md`).

The four headline numbers, each at its exact key path:

- **25.8% → 56.1%**: `results[0].by_stratum.conflicted.arms.native` = 17/66
  (0.2576) → `.detector_suppress` = 37/66 (0.5606).
- **+20**: `results[0].by_stratum.conflicted.paired_vs_native.detector_suppress`
  = `{b_native_only: 0, c_arm_only: 20, net: 20}` — b = 0 is the "0 conflicted
  questions harmed" statement.
- **p = 1.9×10⁻⁶**: same object, `p_exact = 1.9073486328125e-06`.
- **−0.31% tokens**: `results[0].tokens.detector_suppress.delta_pct = -0.30673…`
  (prompt chars 15,852,510 → 15,803,885 = −48,625).

All five arms, all strata (recomputed: unique + conflicted = overall in every arm):

| arm | overall (n=100) | non-conflicted (n=34) | conflicted (n=66) |
|---|---|---|---|
| native | 45 | 28 (82.4%) | 17 (25.8%) |
| native_repeat (A/A) | 45 | 28 | 17 |
| **detector_suppress** | **64** | **27 (79.4%)** | **37 (56.1%)** |
| detector_demote_late | 48 | 28 | 20 (30.3%) |
| detector_anti | 43 | 28 | 15 (22.7%) |

A/A floor: `paired_vs_native.native_repeat` = b 0 / c 0 — exactly zero noise.

**The failure, in the same breath:** `results[0].void_conditions.5_protected_stratum`
→ `status: "fail"`, `voiding_questions: [77]`, `counts.refusal_after_edit: 1`.
Question 77 (unique stratum): native answered `John Milton`; the suppress arm
answered "The provided knowledge pool does not contain any information about…"
although the needed fact (serial 2558) was still on the page
(`results[0].harm.detector_suppress.harms[0]`, `gold_cut: false`). The
artifact's own note on VC5: "the ONLY condition that leaves the run and the
accuracy result standing; the shot is still spent." Verdict object:
`run_void: false`, `protective_claim_void: true`, `shot_spent: true`.
Registered conclusion: **effective, but not yet safe.**

All 8 void conditions and their statuses are in `data/item13_summary.json`;
the 100-row question-level answer table is `data/item13_per_question.csv`.

**Figure:** `fig13_confirmatory_arms.(png|svg)` — 5 arms × 3 strata, McNemar
annotated on the suppression bar, VC5 failure named in the caption.

**Caveat:** one run, one subset, one scale, one model, one shot — the
pre-registration bans forecasting from calibration and the final report bans
generalizing (§10). The −0.31% token delta is measured in characters and
converted (`approx_prompt_tokens` = chars/4); call it a character-based
approximation if pressed.

---

## Item 14 — 735/735 deletion precision

**Claim:** every fact the detector suppressed on the held-out run was
independently verified against benchmark ground truth as superseded — 735 of
735. This answers "are you deleting the right things?", not "did answers improve?".

**Source:** `detector_gap_confirmatory_sh64k.json` →
`results[0].void_conditions.4_no_harmful_suppression.observed` =
`{"n_suppressed_harmful": 0, "n_suppressed_superseded": 735, "n_suppressed_same_value": 0}`.
Cross-check: `void_conditions.8_guards_and_positive_control.observed.positive_control.n_facts_suppressed`
= 735 as well.

**Independent recount (mine, not the artifact's):**
Σ `len(per_question[i].plan.suppress_serials)` over all 100 questions = **735**
(counted with multiplicity across question-pages). Each suppressed serial was
joined to the full sh_64k context (4,580 facts; 17 unparsed by the validated
parser, none of them suppressed), keyed with `conflict_analysis.parse`, and
compared with its key's highest serial:

- not the key's latest serial (superseded): **735 / 735**
- was the key's latest: **0**
- parse failures among suppressed facts: **0**

Per-deletion table: `data/item14_deletions.csv` (735 rows: question index,
serial, fact text, its value, the key's latest serial and latest value, and an
is-latest flag).

**The 200 counter is not a failure:** `positive_control.n_fact_edits_applied = 200`
accumulates across the two editing arms (suppress + demote_late), 100 per arm —
the artifact's `counter_note` says exactly this. Do not read 200 against VC8's
"expected 100".

**One nuance you must volunteer** (from the artifact's `corrections[0].items[2]`
and confirmed by my recount): exactly **1** of the 735 suppressed facts carried
the queried question's *gold* value — q20, gold serial 2374 ("Europe"), whose
key's latest serial is 2468 ("Asia"), i.e. gold-is-not-latest. The fact was
superseded by the serial rule (so it counts in the 735 legitimately), the gold
was cut, and the suppressed arm **answered correctly anyway**. So "735/735
genuinely stale" is precise under the benchmark's serial rule; "no gold fact
was ever deleted" would be false.

**Figure:** NO CHART — it is a precision figure (735 identical outcomes has no
distribution). Screenshot the `4_no_harmful_suppression` object in the JSON;
hand the advisor `data/item14_deletions.csv` if they want the list.

---

## Item 15 — Is the effect just serving nondeterminism?

**Claim:** the serving stack is not fully deterministic, so small single-run
differences are uninterpretable — but the held-out effect is an order of
magnitude larger than the measured noise floor.

**Source:** `stage0_results/t4_s2_trials_summary.json`
(protocol: `stage0_results/t4_s2_protocol.md`, raw runs: `stage0_results/t4_s2_evidence/`):

- `off_sem_per_run` (10 runs): 26, 27, 28, 27, 26, 26, 27, 26, 27, 31 (% accuracy);
  mean recomputed 27.1 = artifact's `off_sem_mean`.
- `shadow_sem_per_run` (5 runs): 26, 27, 26, 28, 26; mean recomputed 26.6.
- `noise_floor`: `within_off_mismatch_mean = 0.0304` (max 0.09),
  `within_shadow_mismatch_mean = 0.022`, `cross_mismatch_mean = 0.0242`;
  pairs 45 / 10 / 50.
- `tost`: diff 0.5, p_lower 8.4e-4, p_upper 0.0166, `equivalent: true` (±2 pts).
- `permutation`: observed_delta −0.0047, p_two_sided 0.475, 10,000 reps, seed 20260814.

**The comparison to make:** off↔shadow disagreement (**2.42%**) sits *below*
the baseline's own off↔off floor (**3.04%**) — shadow mode is
indistinguishable from off at the run level. Against that floor, the held-out
conflicted-stratum effect is **+30.3 accuracy points** (25.8 → 56.1, item 13);
overall +19 points (45 → 64).

**Figure:** `fig15_noise_floor_vs_effect.(png|svg)` — off and shadow run dots
with the noise band, and the held-out effect drawn on the same accuracy axis
(clearly labelled: different subset and stratum, drawn for scale).

**Caveat:** the noise runs are sh_6k overall accuracy; the effect is sh_64k
conflicted-stratum — same axis, different distributions, and the chart says so.
The artifact's own `decision_rule_result.note`: "SUPPORTING EVIDENCE ONLY — the
definitive verdict is the :8002 deterministic rerun (protocol Part 2)". Also
n_shadow = 5 vs n_off = 10; TOST equivalence is at a ±2-point margin, not proof
of byte-identity (that is the separate S2 mechanism claim).

---

# Numbers to state carefully

1. **"0 of 162 configurations were net-beneficial" is wrong; use the screen-on
   form.** Verified from `stage1_calibration.json`: with the subject screen ON,
   **0 of 81** cells are net-positive (helped 228 / harmed 441). Across all 162
   cells, **21** are net-positive — and every one of the 21 has
   `pair_filter == false`, i.e. a detector false-verifying 31–94% of its pairs.
   Correct sentence: *"with a trustworthy detector, 0 of 81 chunk-level
   configurations were net-beneficial."*

2. **735 and 2,673 are different things.** `stage1_operating_point.json →
   metrics` (tp 2,673, fp 0, fact_precision 1.0, n_questions 200) is the
   **calibration** operating-point selection on sh_6k + sh_32k. **735** is the
   held-out sh_64k suppression count (item 14). Never present 2,673 as held-out
   and never merge them into one precision claim.

3. **Item 6 is a percentile summary, not a distribution.** The chart is titled
   accordingly; no histogram or ROC exists offline, and the AUC is quoted from
   the artifact, not re-derived.

4. **The oracle-recovery ratios (0.984 / 0.957) are calibration-only.** The
   confirmatory artifact's corrections state no oracle ceiling exists for
   sh_64k and the ratios may not be quoted for that subset (item 12).

5. **The Kyd/Marlowe contradiction scores are 0.99949 / 0.99983** (two
   directions, two values), provenance markdown-only (`TEZ_BULGULARI.md:265-267`)
   — not "0.99949 in both directions", and not machine-extracted.

6. **The improvement and the voided protective criterion belong in the same
   sentence** (final report §10): +20 conflicted questions, p = 1.9×10⁻⁶, zero
   conflicted harmed — and one non-conflicted question regressed by refusal,
   voiding the protective claim. Effective, not yet safe. No generalization
   beyond one arena, one subset, one scale, one model, one shot.

7. **sh_6k conflicted native accuracy differs by harness**: 4/74 (5.4%) in the
   whole-context oracle probe (item 5) vs 0–6.8% across the benchmark runs
   (item 3) vs 29% overall in the detector-gap run. Quote each number with its
   harness; never average across them.

# What is missing (offline)

- **Raw per-pair similarity distributions for item 6** — only percentiles were
  saved; a histogram/ROC needs an embedder re-run on the GPU box.
- **The raw retrieved page for sh_64k (item 1)** — the confirmatory prepass
  JSON lives on the box (`/mnt/nvmes/nvme1/egekutlu/EvoMemBench/hnav/_out/
  stage1_prepass_sh_64k_benchmarkpage.json`); only sh_6k's page is reproducible
  offline (and was, byte-exactly).
- **An oracle ceiling for sh_64k (item 12)** — never measured; the
  whole-context probe does not fit the window. NOT IN REPOSITORY, and per the
  artifact's corrections it may not be extrapolated.
- **sh_32k McNemar exact figures for item 5's table** were read from
  `stale_suppression_probe_sh32k.json` programmatically; the per-question JSON
  for sh_32k paired examples is extracted but examples were drawn from sh_6k
  (richer, and the Nobuhiro entry lives there).
- Everything else requested was produced.
