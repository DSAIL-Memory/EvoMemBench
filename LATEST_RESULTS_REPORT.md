# H-Nav — what worked in the latest tests, and how the two last filters work

> **Thesis framing decided 2026-09-17 — see `THESIS_CLAIMS.md` (binding).** The
> thesis method is **H-Nav-GEO** (§3.4, §4 below): parser-free, no LLM at
> inference, geometry + NLI. The full five-model trial in §8 is the thesis
> evidence, and its headline is that H-Nav-GEO surpasses the native baseline in
> all 15 model × subset cells. The recommendation in §12 of this report (build
> the story around the schema-keyed arm) is superseded by that decision; §12 is
> kept as a dated record. Measured numbers and void records below are unchanged.


Written 2026-09-03 from the committed artifacts on branch
`claude/hnav-presentation-evidence` (HEAD `c93572b`). Every number below has a
file next to it. Nothing was re-run on the GPU box for this report; the only
thing executed locally was the unit-test suite (§1).

Contents

1. Test suite status
2. The experiment in plain words
3. The three screens in simple mathematics — `hnav_ces`, `hnav_geo`, `hnav_idonly`
4. How the last geometric filter (`hnav_geo`) works, in full
5. How the last symbolic filter (`hnav_idonly`) works, in full
6. Detection quality of every screen on the calibration split (no LLM)
7. Reference model results — all arms, Qwen3-4B
8. Multi-model results — five answering models, three arms, three context sizes
9. What worked, what did not, and what was corrected
10. What must not be claimed
11. Where every number comes from
12. Is the parser the contribution? — how to tell the thesis story

---

## 1. Test suite status

```
.venv-hnav-local\Scripts\python.exe -m pytest hnav/tests/ -q
642 passed, 1 skipped, 24 warnings in 34.72s      (2026-09-03, Windows, no GPU)
```

The suite includes the invariant guards that make the research numbers
trustworthy, and all of them pass:

| guard | what it protects |
| --- | --- |
| `test_leakage_audit.py` | no gold answer / question key can be read by online code (`hnav/core`, `hnav/adapters`) |
| `test_no_raw_entropy_in_policy.py` | `H_raw` never feeds a decision; no `write_policy.py` exists |
| `test_no_torch_at_import.py` | importing any `hnav` module stays free of torch/transformers |
| `test_gold_conflict_dataset.py` | the gold conflict dataset's tier counts, labels and AUC re-derive from the raw audit files |
| E2E-4 structural zero-harm test | a `same_key`-only screen cannot drop a key's newest value, at any threshold (see §5) |

The working tree has two files marked modified (`geo_operating_point.json`,
`idonly_operating_point.json`); `git diff` shows the change is line endings
only (LF→CRLF on Windows checkout). No content differs from the commit.

---

## 2. The experiment in plain words

**The benchmark.** MemoryAgentBench `Conflict_Resolution`, single-hop subsets
`sh_6k`, `sh_32k`, `sh_64k` (100 questions each). Each context is a numbered
list of facts. Later facts *overwrite* earlier ones about the same thing:
fact 210 might say "X was born in London" and fact 980 "X was born in Paris".
The correct answer is always the **latest** version (highest serial number).
Scoring is exact substring match — deterministic, offline, free.

**The problem.** A small language model doing RAG retrieves the right facts
but answers with the **old** version. This is systematic, not random: of 575
wrong answers on conflicted questions across 8 committed runs, 572 gave the
stale value of the correct key. On `sh_6k` the reference model gets 26/26 of
the questions where no conflict exists, and 4/74 where one does.

**What H-Nav does about it.** At read time, before the model sees the
retrieved page, a detector looks for groups of facts that are *versions of the
same thing*, keeps the newest member of each group, and **deletes the older
members from the page**. Nothing is added; the prompt gets shorter
(−0.3 % to −3.5 % characters). No extra LLM call is made at inference. The
detector uses only fact embeddings, an optional symbolic key parsed from the
fact text, and (in some arms) a small NLI cross-encoder run once offline.

**The arms.** Every arm uses the same page-editing machinery; they differ in
the **identity screen** — how a pair of facts is certified as "same thing":

| arm | identity screen | NLI contradiction gate | parser used at inference? |
| --- | --- | --- | --- |
| `hnav_raw` (reference) | parser `same_key` | yes, ≥ 0.90 both directions | yes |
| `hnav_idonly` | parser `same_key` | **waived** (0.0) | yes |
| `hnav_geo` | geometry only (§3, §4) | yes, ≥ 0.90 | **no** |
| `hnav_ces` | parser relation + geometric subject (§3) | yes | partly |
| `hnav_abtt_noparser` | ABTT-whitened cosine only | yes | no |

**The design rules that make the numbers mean something.**

- Thresholds are fitted on `sh_6k` + `sh_32k` **only**, from detection
  quality (no LLM, no accuracy, no gold answer). `sh_64k` is held out and gets
  **one shot** per model per arm. A void run is reported, never re-rolled.
- Questions are split into a **conflicted** stratum (the primary endpoint —
  66/100 on sh_64k) and a **unique** stratum (no conflict exists; the
  do-no-harm check — the detector should change nothing there).
- Every run includes an **A/A floor** (`native` vs `native_repeat`, the same
  prompt sent twice). It must be 0 discordant questions or the run is void.
  It was 0 in all 50 measured cells.
- Statistics are paired per question (exact McNemar), reported per subset,
  never pooled across subsets.
- **Pre-registered void conditions.** The one that mattered most is
  condition 4: a run is void if **any** suppressed fact was *harmful* — i.e.
  deleting it left the page saying something different about that key than
  the corpus does (the key's newest value gone, or the key erased entirely).

---

## 3. The three screens in simple mathematics — `hnav_ces`, `hnav_geo`, `hnav_idonly`

This section is the short version. §4 and §5 give the full pipelines; here
each screen is reduced to the one or two formulas that actually decide.

### 3.1 What every screen shares

Every fact `i` on the page has one embedding vector `v_i` with 2,560 numbers
and length 1 (Qwen3-Embedding-4B). Every screen runs the same four steps and
differs only in step 3:

```
1. pool      : the 50 facts on the page most similar to the question
2. candidate : a pair (a, b) from the pool is a candidate if  cos(a, b) = v_a · v_b  ≥  c
3. identity  : is (a, b) two versions of the SAME thing?      ← the screen
4. suppress  : join accepted pairs into groups; in each group keep the
               fact with the highest serial number, delete the others
```

The screens are asked one question. Two facts with a high cosine can be a
**version pair** or a **look-alike pair**:

```
version pair    "Kyd was born in the city of London"  →  "Kyd was born in the city of Paris"
                same subject, same relation, the OBJECT changed          ⇒ old one should go

look-alike pair "Kyd was born in the city of London"  →  "Marlowe was born in the city of London"
                same relation, same object, the SUBJECT changed          ⇒ both must stay
```

Both pairs have cosine well above 0.9 — they share almost every word. Cosine
alone cannot separate them; that is the entire difficulty. The two geometric
screens look not at *how similar* the facts are but at *what changed* between
them. They use the **edit direction**:

```
d̂ = (v_b − v_a) / ‖v_b − v_a‖          a unit vector: the direction you move in
                                         embedding space when a turns into b
```

The hypothesis is that an "object changed" move and a "subject changed" move
point in recognisably different directions. Both geometric screens are trained
on the same material from the audited gold conflict dataset, calibration
split only: **989 version-pair edits** (positives) and **8,716 look-alike
edits** (hard negatives).

### 3.2 `hnav_idonly` — symbolic identity, one equality test

Each fact is passed through the template parser
(`hnav/labeling/conflict_analysis.py::parse`, 39 hand-written relation
templates such as `"{s} was born in the city of {o}"`). It returns a key or
nothing:

```
key(f) = (relation, subject)     e.g. ("was born in the city of", "Kyd")
       = ∅                       if no template matches
```

The screen is then a single equality:

```
accept(a, b)  ⇔  cos(a, b) ≥ 0.90   and   key(a) ≠ ∅   and   key(a) = key(b)
```

No NLI, no learned parameters, no thresholds beyond the cosine. In the
example, `key(Kyd/London) = key(Kyd/Paris)` → accept; `key(Kyd/London) ≠
key(Marlowe/London)` → reject.

**Why it can never delete the wrong fact.** "`=` on keys" is an equivalence
relation: it is *transitive* (if a=b and b=c then a=c). So a group formed by
chaining accepted pairs always has exactly one key, and keeping the highest
serial in the group keeps that key's newest value. Harm is zero at any
threshold — a property of `=`, not of the data.

Calibration pool recall 0.9952 (misses only the 0.5 % of true pairs with
cosine below 0.90). Held-out `sh_64k`: **66/100**, 0 harmful.

### 3.3 `hnav_ces` — contrastive edit subspaces

CES asks: *in which directions does a version edit move, and in which does a
look-alike edit move?* Offline, stack the 989 positive edit directions as rows
of a matrix `D_obj` and the 8,716 negative ones as rows of `D_subj`. Take the
top **k = 20** principal directions of each (SVD):

```
U_obj  = the 20 directions that best explain object-changed edits   (2560 × 20)
U_subj = the 20 directions that best explain subject-changed edits  (2560 × 20)
```

This is done once per relation template (31 relations with ≥ 5 examples of
both kinds) plus one global pair as fallback. At inference the score of a pair
is how much of the edit's energy lands in each subspace:

```
score(a, b) = ‖U_objᵀ d̂‖²  −  ‖U_subjᵀ d̂‖²
```

Because `d̂` has length 1 and the columns of `U` are orthonormal, each term is
a number between 0 and 1: *the fraction of the edit that lies inside that
subspace*. The score is the difference. A version edit lands mostly in
`U_obj` → score near +1; a look-alike edit lands in `U_subj` → score negative.

```
accept(a, b)  ⇔  cos(a, b) ≥ 0.80   and   score(a, b) > 0.40   and   NLI ≥ 0.90 both ways
```

`0.40` means "at least 40 percentage points more of the edit lives in the
object subspace than in the subject subspace".

**Where the parser is still used.** CES picks `U_obj, U_subj` for the pair's
relation, and the relation comes from `key(a)[0] = key(b)[0]` (the parser's
relation half). The *subject* half is deliberately not read — replacing the
symbolic subject test with geometry is the experiment. If the relation is
unknown, the global subspaces are used (pair-level AUROC 0.87 vs 0.98
relation-aware).

Calibration pool recall 0.7343, 0 harmful. Held-out `sh_64k`: **55/100**,
passed the harm rule (0 harmful).

### 3.4 `hnav_geo` — parser-free identity, two numbers per pair

GEO reads only `v_a` and `v_b`; no parser field is touched at inference. It
computes two numbers and requires both to be large enough.

**Number 1 — whitened cosine.** All facts in the store look alike (short
English sentences of one template family), and that shared component inflates
every cosine. Whitening removes it: subtract the store mean `μ`, project out
the 128 strongest shared directions `P`, renormalise:

```
ŵ(v)        = (v − μ − P(v − μ)) / ‖ · ‖
cos_w(a, b) = ŵ(v_a) · ŵ(v_b)               similarity in what is SPECIFIC to a and b
```

**Number 2 — the slot probe.** Take the edit direction `d̂`, drop the signs
(`|d̂|`, so it does not matter which fact is older), and feed the 2,560
per-axis magnitudes to a logistic regression trained on the same 989 vs
8,716 edits:

```
probe(a, b) = w · |d̂| + b        > 0 ⇒ "looks like an object-slot edit"
```

**Decision.** Each number is turned into a margin in standard-deviation units
from an anchor (the anchors are the loosest thresholds at which *no*
look-alike pair passed on calibration; the scales are the feature standard
deviations there), and both margins must clear a bar:

```
m_w = (cos_w − 0.6397) / 0.3483  ≥ −0.40        ⇔  cos_w ≥ 0.500
m_p = (probe + 0.0764) / 1.0675  ≥ +0.20        ⇔  probe ≥ 0.137
accept(a, b)  ⇔  cos(a, b) ≥ 0.94  and  m_w ≥ −0.40  and  m_p ≥ +0.20  and  NLI ≥ 0.90 both ways
```

Calibration pool recall 0.7895 (best parser-free result on record), 0
harmful. Held-out `sh_64k`: **56/100**, but **8 harmful suppressions → void**.

**Why it can delete the wrong fact and `hnav_idonly` cannot.** "Similar
enough" is *not* transitive: `a ≈ b` and `b ≈ c` do not imply `a ≈ c`, and
they certainly do not imply that `a` and `c` are about the same thing. But
step 4 chains accepted pairs into one group regardless. On `sh_64k` the
screen accepted a look-alike pair it had never seen the like of on
calibration; the group then held two keys, kept a single serial, and the
other key's newest fact was deleted — its whole key vanished from the page
(`official language of Italy`, `type of music that The Game plays`,
`outfielder · sport`). In the toy example: if GEO accepts
`(Kyd/London, Marlowe/London)` even once, the group
`{Kyd/London, Kyd/Paris, Marlowe/London}` keeps only the highest serial and
Marlowe's fact is gone. Equality cannot do this; a learned similarity can,
and a threshold tuned to zero false positives on one split does not carry
that guarantee to the next.

### 3.5 Side by side

| | `hnav_idonly` | `hnav_ces` | `hnav_geo` |
| --- | --- | --- | --- |
| identity test | `key(a) = key(b)` | `‖U_objᵀd̂‖² − ‖U_subjᵀd̂‖² > 0.40` | `cos_w ≥ 0.500` and `probe ≥ 0.137` |
| what it reads | parser key | parser *relation* + `v_a, v_b` | `v_a, v_b` only |
| learned parameters | none | 2 × 20 directions × 32 (31 relations + global) | 128 whitening directions + 2,561 probe weights |
| cosine pre-filter | 0.90 | 0.80 | 0.94 |
| NLI gate | off | ≥ 0.90 | ≥ 0.90 |
| transitive? (harm-free by construction) | **yes** | no | no |
| calibration pool recall | 0.9952 | 0.7343 | 0.7895 |
| `sh_64k` reference model (native 45) | **66** | 55 | 56 (void, 8 harmful) |

---

## 4. How the last geometric filter (`hnav_geo`) works, in full

Goal: decide "are these two facts two versions of the *same thing*?" using
**only their embedding vectors** — no text parsing at all. Code:
`hnav/geometry_filter/geo_artifact.py::GeoIdentityScreen`; frozen operating
point `stage0_results/geometry_filter/geo_operating_point.json`.

### 4.1 The whole pipeline for one question

1. The benchmark's own retriever produces the page for the question
   (`page_source=benchmark`).
2. From the facts on that page, keep the **50 most query-similar** as the
   candidate pool (`select_pool`).
3. Form every pair in the pool whose **raw cosine similarity ≥ 0.94**
   (`cos_pair`).
4. Apply the **GEO screen** (below) to each such pair. Pairs that fail are
   dropped before any NLI is spent on them.
5. Surviving pairs are scored by a DeBERTa-v3 NLI cross-encoder in **both
   directions**; the pair is a *verified conflict* only if the contradiction
   score is ≥ 0.90 both ways (A contradicts B **and** B contradicts A).
6. Verified pairs are joined into groups (connected components). In each
   group the fact with the **highest serial** is kept; every other member is
   **deleted from the page** (`detector_suppress`).
7. The edited page is sent to the answering model with the benchmark's own
   prompt, `temperature 0`, `max_tokens 10`.

Steps 1–6 contain no LLM, so the **suppression plan is identical for every
answering model** — this was later confirmed byte-for-byte on five models.

### 4.2 The GEO screen itself — two numbers per pair

For two facts with unit-length embeddings **a** and **b** (2,560 dimensions,
Qwen3-Embedding-4B):

**Signal 1 — whitened cosine `cos_w`.** Plain cosine between fact
embeddings is dominated by "common" directions that every sentence in the
store shares (they all look like short factual sentences). ABTT ("all but
the top") whitening subtracts the store mean and projects out the top 128
principal directions, then renormalises. The cosine measured *after* that
tells you how similar two facts are in what is *specific* to them. The
whitening is the committed Stage-1 artifact (`abtt_whitening_D128.json`),
copied into the screen with its fingerprint pinned.

**Signal 2 — the slot probe `probe`.** Take the difference vector
`d = b − a`, normalise it to unit length, then take the **absolute value of
every coordinate** (so it does not matter which fact is "first"). This gives
a 2,560-number profile of *which embedding axes moved* when you go from one
fact to the other. A logistic regression on that profile was trained to
separate:

- **positives**: gold conflict pairs from the audited gold dataset — same
  subject, same relation, the *object* changed ("X born in London" →
  "X born in Paris"); 989 edits;
- **hard negatives**: pairs that look similar but where the *subject*
  changed ("Kyd born in London" → "Marlowe born in London"); 8,716 edits.

The intuition: when only the object slot changes, the difference concentrates
on a recognisable set of axes; when the subject changes, a different set
moves. The probe outputs a logit — high means "looks like an object-slot
edit of the same subject".

Both signals are fitted on the calibration split only (`sh_6k` + `sh_32k`);
the fitter refuses any other data.

### 4.3 Turning two numbers into a yes/no

Each signal is turned into a **margin in standard-deviation units** relative
to an anchor:

```
m_w = (cos_w − T_w) / s_w          T_w = 0.6397   s_w = 0.3483
m_p = (probe − T_p) / s_p          T_p = −0.0764  s_p = 1.0675
```

The anchors `(T_w, T_p)` are the *joint zero-false-positive point* found by
sweeping thresholds over the calibration pool pairs that already pass the NLI
gate: the loosest pair of thresholds at which **no cross-key pair** gets
through. The scales are the feature standard deviations over those same
pairs (193 pairs on sh_6k, 1,332 on sh_32k).

A pair passes the screen if **both margins clear a per-axis threshold**:

```
pass  ⇔  m_w ≥ tw  and  m_p ≥ tp        frozen: tw = −0.40,  tp = +0.20
```

In raw units that is `cos_w ≥ 0.500` and `probe ≥ 0.137`. The shape matters:
the whitened cosine is allowed to be **loose** (0.4 SD below its anchor) while
the probe must be **strict** (0.2 SD above its anchor). The originally
pre-registered grid only allowed a single `tau` applied to both axes
(diagonal); that reached pool recall 0.596. The documented, calibration-only
Amendment 1 opened a per-axis rectangle and found this corner, pool recall
0.790 at zero harmful suppressions on calibration.

### 4.4 What it achieved, and where it broke

- **Best parser-free detector this repo has measured.** Calibration: pair
  precision 1.000, pool recall **0.7895** (previous parser-free best 0.4444),
  104/139 conflicted questions hit, 0 harmful of 2,157 suppressions.
- **Pair-level, it is as good as the parser-assisted CES screen**
  (`geo_pairlevel.json`): balanced sh_64k AUROC 0.9716 (CES 0.9756, ABTT
  cosine 0.9648), and the best unseen-transition tail on record
  (TPR@1e-4 = 0.481).
- **End to end on the held-out set: 56/100** on the reference model
  (+11 over native 45, p = 0.0074), but **−8 vs the parser arm** (64,
  p = 0.021). The pre-registered primary endpoint (> 64) failed.
- **The run is VOID by condition 4.** On `sh_64k` the screen made **8
  harmful suppressions** (524 correct ones). All eight are *key erasures*: a
  geometric group merged two different keys — `official language of Italy`,
  `type of music that The Game plays`, `outfielder · associated with the
  sport` — and suppression kept only the merged group's newest serial, so a
  whole key vanished from the page. Zero harm on calibration **did not
  transfer** to held-out data.
- The decisive detail: the five suppressions geometry made that the parser
  did *not* make are **exactly** the five that erased a key — 5 for 5.
  Geometry's entire unique contribution over symbolic identity was
  information loss. On the questions the parser alone answers, geo's plans
  are strict subsets of the parser's (0 geo-only serials).
- Because the plan has no LLM in it, the same 8 harmful suppressions
  recurred **on every one of the five models** (§8). That is the result, not
  a run failure.

---

## 5. How the last symbolic filter (`hnav_idonly`) works, in full

Goal: test whether the **NLI gate**, not identity, was what held the parser
arm back. Code path: `hnav/stage1/detector_gap.py::make_gate` with
`pair_filter == "idonly"` → `MABAdapter.same_key_pair`; frozen operating point
`stage0_results/stage1/idonly_operating_point.json`; pre-registration
`stage0_results/stage1/IDONLY_PREREG.md`.

### 5.1 The pipeline for one question

Same steps as §4.1, with two changes:

- Step 3: pairs with **raw cosine ≥ 0.90** (the parser arm's threshold).
- Step 4: the identity screen is **`same_key`**: both facts must parse
  (`hnav/labeling/conflict_analysis.py::parse`, validated at 99.5 %+
  coverage), and their parsed `(relation, subject)` key must be **identical**.
  An unparseable fact is *rejected*, not waved through.
- Step 5: the NLI gate is **switched off** (`nli_contradiction = 0.0`). A
  same-key pair is a verified conflict as soon as it passes cosine + key
  equality.
- Step 6 is unchanged: group, keep the highest serial, delete the rest.

That is the entire difference from the shipped `hnav_raw` arm.

**It is not "parser only".** The parser is the *identity* check, but the
pairs it ever gets to see are chosen by embedding geometry, and geometry can
still veto a group afterwards. What actually runs, in order
(`hnav/core/read_gate.py::ReadGate.decide`):

| step | what it uses | idonly setting |
| --- | --- | --- |
| candidate pool | embeddings — 50 facts on the page most similar to the query | as every arm |
| cosine screen | embeddings — pair cosine ≥ `cos_pair` | 0.90 |
| identity screen | **parser** — both parse, same `(relation, subject)` key | `same_key` |
| span-residual check | embeddings — a group survives only if some member's leave-one-out QR residual < `r_min` | 0.44 ("loose") |
| NLI contradiction gate | cross-encoder | **off** (0.0) — the only change vs `hnav_raw` |
| suppress | serials — keep the highest, delete the rest | unchanged |

The cosine and residual steps can only *remove* pairs, so they never put two
different keys into one group — geometry here limits recall (the 0.5 % of
true pairs still missed are same-key pairs under cosine 0.90) but can never
create harm. The zero-harm guarantee (§5.3) rests on the `same_key` step
alone.

The selection
grid deliberately included the shipped NLI value (0.90) with a tie-break that
*prefers the stricter gate*, so the arm could only choose 0.0 if waiving the
gate actually bought recall. It did: pool recall 0.9784 → **0.9952**.

### 5.2 Why this arm was built (the diagnosis)

E2E-4 decomposed the parser arm's 29 conflicted failures on `sh_64k`:

| cause | n | fixable by page editing? |
| --- | ---: | --- |
| retrieval miss — the gold fact never reaches the page | **22** | no |
| parametric-knowledge failure — page already correct, model answers from its weights | 5 | no |
| detection miss | **2** | yes |

Both detection misses were same-key pairs the **NLI** rejected: q23
(`sport of racing` → `Australian rules football`, contradiction 0.854, just
under 0.90) and q98 (`The Kinks was founded in the city of London` →
`… of England`, contradiction 0.0002 — the NLI is factually right that London
is in England and *wrong for this store*, whose rule is "later serial wins").
Calibration showed this is systematic: 2.2 % of true supersession pairs
(13/536 at cos ≥ 0.90) are blocked by the NLI gate. The pre-registration
therefore predicted, **by name**, that `hnav_idonly` would recover exactly
{q23, q98} and land at 65–66.

### 5.3 Why harm is zero *by construction*

`same_key` is an equivalence relation on `(relation, subject)`, so a verified
group can never contain two different keys. Suppression keeps each group's
highest serial, which is therefore the key's newest value in the corpus. And
`same_key` requires both facts to parse, so the "unparsed drop" channel is
empty. Hence `n_suppressed_harmful = 0` **at any threshold**, on any split —
a structural guarantee the geometric screen does not have. This is encoded as
a unit test and was observed in every grid cell during selection and in all
15 held-out model cells.

### 5.4 What it achieved

Reference model, `sh_64k`, one shot: **66/100** (native 45; parser arm 64).
Conflicted 17 → **39**/66, unique 27/34. The arm recovered **exactly
{23, 98}** and lost no question the parser arm had — the named prediction
landed exactly. Condition 4: **0 harmful** of 753 suppressed facts on
held-out data. Calibration: sh_6k 95 (parser 94), sh_32k 85 (parser 86).

Honest statistics: +2 over the parser arm on two discordant pairs is
p = 0.5 — not significant on its own. The within-model effect vs native is
+21 overall (p = 5.7e-06) and +22 conflicted (p = 4.8e-07). The result is the
mechanism confirmation, not the +2.

---

## 6. Detection quality of every screen (calibration split, no LLM)

All fitted on `sh_6k` + `sh_32k`, 200 questions, 2,732 true supersession
pairs in the pools. "Harmful" is counted by `classify_drops` (§2).

| screen | parser at inference | cos_pair | NLI | pool recall | precision | conflicted questions hit | harmful |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hnav_idonly` | yes | 0.90 | off | **0.9952** | 1.000 | 137/139 | 0 |
| `hnav_raw` (parser) | yes | 0.90 | 0.90 | 0.9784 | 1.000 | 133/139 | 0 |
| `hnav_geo` | **no** | 0.94 | 0.90 | 0.7895 | 1.000 | 104/139 | 0 |
| `hnav_ces` | relation only | 0.80 | 0.90 | 0.7343 | 1.000 | — | 0 |
| `hnav_abtt_noparser` | no | 0.80 | 0.90 | 0.4444 | 1.000 | — | 0 |

Zero harm on calibration was true of every screen. Only the two `same_key`
screens kept it on `sh_64k` (`hnav_geo`: 8 harmful; `hnav_abtt_noparser`:
5 harmful — both runs void by condition 4).

---

## 7. Reference model — all arms, Qwen3-4B-Instruct-2507

Held-out `sh_64k`, one shot each, paired against the same native answers
(all at `page_source=benchmark`):

| arm | overall | conflicted /66 | unique /34 | p vs native (overall) | condition 4 |
| --- | ---: | ---: | ---: | ---: | --- |
| native | 45 | 17 | 28 | — | — |
| `hnav_idonly` | **66** | **39** | 27 | 5.7e-06 | pass (0 harmful) |
| `hnav_raw` / `hnav_abtt` (parser) | 64 | 37 | 27 | 2.1e-05 | pass |
| fusion (exploratory, relaxed harm) | 61 | 33–34 | 27–28 | — | failed the zero-harm rule |
| `hnav_abtt_noparser` | 59 | 31 | 28 | 1.2e-04 | **VOID** (5 harmful) |
| `hnav_geo` | 56 | 29 | 27 | 7.4e-03 | **VOID** (8 harmful) |
| `hnav_ces` | 55 | 28 | 27 | 1.3e-02 | pass |

Other mechanisms on the parser arm, same run: `demote_late` (move the newest
fact to the end instead of deleting the old ones) 48; `anti` (move it to the
front) 43 — placement moves the number in opposite directions, so deletion is
the mechanism that works.

Ceiling: 22 retrieval misses + 5 parametric failures are unreachable by any
page edit, so the most a suppression-only detector can score on this
substrate is 44/66 + 28/34 = **72/100**. `hnav_idonly` at 66 is within that
budget with the two detection misses recovered.

Calibration subsets (for continuity, not endpoints):

| arm | sh_6k | sh_32k |
| --- | ---: | ---: |
| native | 30 | 53 |
| `hnav_idonly` | 95 | 85 |
| `hnav_raw` | 94 | 83 |
| `hnav_ces` | 84 | 78 |
| `hnav_geo` | 77 | 77 |
| `hnav_abtt_noparser` | 54 | 72 |

---

## 8. Multi-model results — five answering models, three arms

Campaign E2E-5, run 2026-08-30 13:43 → 2026-08-31 12:55 UTC on the box,
~19,500 completions. The **only** thing that changed between models was the
answering LLM; memory store, retrieval, embeddings, suppression plans,
prompts, generation settings and scoring were frozen. Each cell is one shot,
`page_source=benchmark`, A/A floor 0 everywhere. Table generated from
artifacts by `hnav/geometry_filter/multimodel_summary.py`
(`pipelines/MULTIMODEL_SUMMARY.md`) — never hand-typed, after a hand-typed
version was once wrong.

Models (smallest weights first): Phi-4-mini-instruct (7.2 GB, vLLM 0.9.1),
gemma-3-4b-it (8.1 GB, 0.9.1), gemma-4-E2B-it (9.6 GB, 0.28.0),
Qwen3.5-9B (19.3 GB, 0.28.0, thinking off), plus the reference
Qwen3-4B-Instruct-2507 (8.0 GB, not re-run).

### 8.1 Overall accuracy /100 (native → arm, gain)

**sh_6k**

| model | native | `hnav_raw` | `hnav_idonly` | `hnav_geo` |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 30 | 94 (+64) | **95 (+65)** | 77 (+47) |
| Qwen3.5-9B | 39 | 98 (+59) | **99 (+60)** | 81 (+42) |
| gemma-3-4b-it | 45 | 89 (+44) | **89 (+44)** | 76 (+31) |
| gemma-4-E2B-it | 40 | 83 (+43) | **83 (+43)** | 72 (+32) |
| Phi-4-mini-instruct | 40 | 88 (+48) | **89 (+49)** | 76 (+36) |

**sh_32k**

| model | native | `hnav_raw` | `hnav_idonly` | `hnav_geo` |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 53 | 83 (+30) | **85 (+32)** | 77 (+24) |
| Qwen3.5-9B | 61 | 91 (+30) | **92 (+31)** | 86 (+25) |
| gemma-3-4b-it | 38 | 51 (+13) | **52 (+14)** | 45 (+7) |
| gemma-4-E2B-it | 44 | 63 (+19) | **63 (+19)** | 58 (+14) |
| Phi-4-mini-instruct | 50 | 72 (+22) | **72 (+22)** | 66 (+16) |

**sh_64k — held out, one shot** (`hnav_geo` VOID on every row: 8 harmful
suppressions, identical plan)

| model | native | `hnav_raw` | `hnav_idonly` | `hnav_geo` |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | 45 | 64 (+19)¹ | **66 (+21)** | 56 (+11) VOID |
| Qwen3.5-9B | 51 | 67 (+16) | **69 (+18)** | 62 (+11) VOID |
| gemma-3-4b-it | 33 | 38 (+5) | **38 (+5)** | 36 (+3) VOID |
| gemma-4-E2B-it | 37 | 43 (+6) | **45 (+8)** | 41 (+4) VOID |
| Phi-4-mini-instruct | 46 | 57 (+11) | **57 (+11)** | 52 (+6) VOID |

¹ From the committed Stage-1 artifact `abtt_arm_A1_raw_sh64k.json` (same
configuration); the generated summary leaves this cell blank because it lives
outside `pipelines/`.

### 8.2 Held-out `sh_64k` by stratum, with exact McNemar p-values

Conflicted stratum = primary endpoint (n = 66). Unique = do-no-harm (n = 34).

| model | arm | conflicted native → arm | p | unique native → arm | overall p |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | `hnav_idonly` | 17 → **39** | 4.8e-07 | 28 → 27 | 5.7e-06 |
| Qwen3-4B | `hnav_raw` | 17 → 37 | 1.9e-06 | 28 → 27 | 2.1e-05 |
| Qwen3-4B | `hnav_geo` (void) | 17 → 29 | 1.8e-03 | 28 → 27 | 7.4e-03 |
| Qwen3.5-9B | `hnav_idonly` | 24 → **43** | 3.8e-06 | 27 → 26 | 4.0e-05 |
| Qwen3.5-9B | `hnav_raw` | 24 → 42 | 7.6e-06 | 27 → 25 | 4.0e-04 |
| Qwen3.5-9B | `hnav_geo` (void) | 24 → 36 | 4.9e-04 | 27 → 26 | 3.4e-03 |
| Phi-4-mini | `hnav_idonly` | 16 → **27** | 1.9e-02 | 30 → 30 | 1.9e-02 |
| Phi-4-mini | `hnav_raw` | 16 → 27 | 1.9e-02 | 30 → 30 | 1.9e-02 |
| Phi-4-mini | `hnav_geo` (void) | 16 → 22 | 0.146 | 30 → 30 | 0.146 |
| gemma-4-E2B | `hnav_idonly` | 21 → **31** | 6.3e-03 | 16 → 14 | 0.077 |
| gemma-4-E2B | `hnav_raw` | 21 → 29 | 2.1e-02 | 16 → 14 | 0.180 |
| gemma-4-E2B | `hnav_geo` (void) | 21 → 25 | 0.125 | 16 → 16 | 0.344 |
| gemma-3-4b | `hnav_idonly` | 14 → **19** | 0.125 | 19 → 19 | 0.180 |
| gemma-3-4b | `hnav_raw` | 14 → 19 | 0.0625 | 19 → 19 | 0.125 |
| gemma-3-4b | `hnav_geo` (void) | 14 → 17 | 0.25 | 19 → 19 | 0.375 |

Reading this honestly: on the held-out set the conflicted-stratum gain is
significant for `hnav_idonly` on four of five models; gemma-3-4b's +5 is not.
The unique stratum never moves by more than 2 questions in either direction,
and on gemma-4-E2B the −2 (16 → 14) has p = 0.625.

### 8.3 Calibration subsets by stratum (`hnav_idonly`)

| model | sh_6k conflicted /74 | sh_6k unique /26 | sh_32k conflicted /65 | sh_32k unique /35 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 4 → 69 (p 5e-20) | 26 → 26 | 19 → 51 (p 4e-09) | 34 → 34 |
| Qwen3.5-9B | 13 → 73 (p 2e-18) | 26 → 26 | 27 → 58 (p 8e-09) | 34 → 34 |
| gemma-3-4b | 22 → 65 (p 2e-13) | 23 → 24 | 11 → 23 (p 0.012) | 27 → 29 |
| gemma-4-E2B | 16 → 59 (p 2e-11) | 24 → 24 | 17 → 36 (p 2e-05) | 27 → 27 |
| Phi-4-mini | 14 → 63 (p 4e-15) | 26 → 26 | 16 → 39 (p 6e-06) | 34 → 33 |

### 8.4 Prompt-length change (the "cheap" claim)

Suppression only deletes, so the prompt shrinks. Identical across models
because the plan is identical: `hnav_idonly` −3.52 % (sh_6k), −0.64 %
(sh_32k), −0.31 % (sh_64k, −48.6 k characters); `hnav_geo` −2.87 % / −0.51 %
/ −0.22 %. Accuracy goes up while token cost goes down.

---

## 9. What worked, what did not, and what was corrected

### Worked

1. **The governance gain replicates.** 15 of 15 model × arm cells for the
   two parser-based arms are positive on `sh_64k`; every calibration cell is
   positive too. Range +5 … +65 overall.
2. **`hnav_idonly` ≥ `hnav_raw` on every model and every subset** (0–2
   points). Waiving the NLI gate never cost a question. The arm passed its
   pre-registered endpoint (66 > 64) and its named prediction ({q23, q98})
   landed exactly on the reference model.
3. **Zero harm held out-of-sample for the `same_key` screens** — 0 harmful
   suppressions on every held-out run, as the structural argument predicted.
4. **Suppression plans are model-independent — measured, not assumed.**
   `hnav_geo` produced `n_suppressed_harmful = 8`, `n_suppressed_superseded
   = 524` on all five models, byte-identical. Likewise the set of `sh_64k`
   questions where `hnav_idonly` and `hnav_raw` prompts differ is the same
   17 questions on every model: {5, 8, 15, 20, 23, 38, 49, 50, 52, 55, 58,
   59, 73, 82, 85, 88, 98}.
5. **Which of those 17 convert is model-dependent** — Qwen3-4B {23, 98},
   Phi-4-mini gains {98} loses {55}, gemma-4-E2B {23, 59}, Qwen3.5-9B
   {82, 98}. One degradation in 264 model-question opportunities.
6. **The instrument checks fired when they should have.** The A/A floor was
   0 in all 50 cells; page-edit mismatch, containment and positive control
   were clean everywhere; the preflight gate (`preflight_model.py`, seven
   hard checks) caught context-window, thinking-mode and endpoint problems
   before any shot was spent.
7. **Post-H-Nav accuracy is monotone in model capability** at every
   context length: sh_6k 83/89/89/95/99, sh_32k 52/63/72/85/92,
   sh_64k 38/45/57/66/69 (ascending model strength). The spread between
   models is much larger on the long contexts (40 and 31 points) than on
   sh_6k (16 points), where the ceiling compresses everyone.

### Did not work

1. **`hnav_geo` cannot replace the parser.** 56 vs 64 on the reference model
   (p = 0.021), below the parser arms on all five models, and void by
   condition 4 on every one of them (8 key erasures). Geometry's only
   additions over symbolic identity were the harmful ones (5/5).
2. **Geometry and symbolic identity are not complementary.** Geo's
   suppressions are a 99 % subset of the parser's (527 shared / 208
   parser-only / 5 geo-only). The oracle "either arm correct" union is
   65/100 vs 64, and that +1 (q30) is answering noise. The pre-registered
   hybrid arm was therefore not built.
3. **`hnav_abtt_noparser` is also void** (5 harmful) — its committed 59/100
   was originally reported as valid because the runner checked only the
   mechanical guards, not the pre-registered void conditions. Fixed in E2E-4.
4. **Fusion** (CES + ABTT logit) failed the zero-harm rule; relaxed-harm
   exploratory variants sat flat at 61.

### Corrected during the campaign (recorded, not hidden)

- **gemma-3-4b's first run was VOID** because the campaign inherited
  `--kv-cache-dtype fp8` from the reference model. Same weights, same
  prompts, same plan: **13/100** at fp8 vs **89/100** at BF16 on sh_6k; the
  fp8 output was `"United States of United States of …"`. A dtype A/B
  (0/10 vs 9/10 on unique-stratum questions) condemned it. The run is kept
  under `*_VOID_fp8_kv/` with a `VOID.md`. Lesson: the serving configuration
  is part of the experiment; "it produced fluent text" is not a check.
- **The reference model's old `hnav_raw` sh_6k/sh_32k numbers (94, 86) came
  from a different page configuration** (`page_source=None`). They were
  re-measured at `page_source=benchmark` (94, 83) so all 50 cells are one
  configuration; the summary generator now warns on any mix.
- **Two capability claims were retracted** (`TEZ_HIKAYESI.md` §4): "the gain
  is largest for weak models" (false — Qwen3-4B starts at 30 and gains +65,
  gemma-3 starts at 45 and gains +44) and "the gain rises with model
  strength" (also false in general — on sh_6k the gain anti-correlates with
  native because the ceiling is compressed). What is monotone is the
  post-H-Nav accuracy, not the gain. Framing: H-Nav *converts latent
  capability into accuracy cheaply*; it does not *rescue* weak models.
- **Four serving facts must be measured per model, not inherited:** the
  context window (the chars/4 estimate said ≈42.5k tokens; measured with
  each model's own tokenizer and chat template it is 43.3k–49.6k, so serving
  at 48k would have died on the longest sh_64k question hours in), the
  vLLM version/architecture support
  matrix, thinking mode (a thinking model spends its 10 output tokens on
  reasoning and scores ~0), and the KV-cache dtype.

---

## 10. What must not be claimed

- "Works on every model" — it was measured on **five** 2–9B English
  instruction models, four architectures, two vLLM versions. That is five,
  not all.
- "The gain scales with weakness / with strength" — both refuted (§9).
- "Works with any embedder" — every threshold is a coordinate in the
  Qwen3-Embedding-4B space; the G1 probe measured that they do not transfer.
- "Semantic verification is unnecessary" — shown only for a store with
  **single-valued relations** where a later serial supersedes the same key.
  With genuinely multi-valued relations the NLI gate would do real work.
- "Generalises to sh_262k" — excluded by design (context windows would turn
  a memory comparison into a context-length comparison).
- "Geometry can replace the parser" — it cannot: 64 → 56, p = 0.021, and
  void by harm on all five models.
- Do not pool the four new models into one p-value: they answer the same
  100 questions with the same plans and are not independent samples. The
  defensible statement is direction and consistency (5/5 models
  `idonly ≥ raw`).

---

## 11. Where every number comes from

| number | file |
| --- | --- |
| test suite 642 passed / 1 skipped | local run 2026-09-03, `pytest hnav/tests/ -q` |
| multi-model tables (§8.1) | `pipelines/MULTIMODEL_SUMMARY.md` (generated) |
| per-model p-values (§8.2, §8.3) | `pipelines/<arm>/results/<model>_<date>/REPORT.md` and `detector_gap_sh_*.json` |
| campaign plan, predictions vs outcome | `pipelines/MULTIMODEL_CAMPAIGN_PLAN.md` |
| reference parser arm sh_64k 64/100 | `stage0_results/abtt/abtt_arm_A1_raw_sh64k.json` |
| `hnav_idonly` 66/100 and the {23, 98} prediction | commit `4e01e47`; `pipelines/hnav_idonly/results/Qwen_Qwen3-4B-Instruct-2507_2026-08-30/` |
| `hnav_geo` design, selection, gates | `stage0_results/geometry_filter/GEO_PREREG.md` |
| `hnav_geo` result and correction (void) | `stage0_results/geometry_filter/E2E3_REPORT.md` |
| complementarity analysis, ceiling decomposition | `stage0_results/geometry_filter/E2E4_COMPLEMENTARITY.md` |
| `hnav_idonly` pre-registration | `stage0_results/stage1/IDONLY_PREREG.md` |
| GEO screen parameters (anchors, scales, probe) | `stage0_results/geometry_filter/geo_identity_screen.json` + `.npz` |
| GEO / idonly / parser operating points | `stage0_results/geometry_filter/geo_operating_point.json`, `stage0_results/stage1/idonly_operating_point.json`, `stage0_results/stage1_operating_point.json` |
| GEO pair-level AUROC | `stage0_results/geometry_filter/geo_pairlevel.json` |
| gemma-3 fp8 void | `pipelines/hnav_*/results/google_gemma-3-4b-it_2026-08-30_VOID_fp8_kv/VOID.md` |
| thesis framing and retracted claims | `TEZ_HIKAYESI.md` §4, §7, §8 |
| harm rule and void conditions (code) | `hnav/stage1/detector_gap.py::classify_drops`, `::void_condition_report` |
| the two screens (code) | `hnav/geometry_filter/geo_artifact.py`, `hnav/adapters/mab_adapter.py::same_key_pair`, `hnav/stage1/detector_gap.py::make_gate` |
| CES screen (code, artifact) | `hnav/geometry_filter/ces_artifact.py`, `stage0_results/geometry_filter/ces_subspaces_k20.json` + `.npz`, `ces_operating_point.json` |
| the parser (39 templates) | `hnav/labeling/conflict_analysis.py:18-68` |

---

## 12. Is the parser the contribution? — how to tell the thesis story

This section answers a question the results force: the best arm,
`hnav_idonly`, is three lines of logic around a template parser. Does that
undermine the thesis? Short answer: **it undermines one framing of the thesis
and strengthens another.** The rest of this section is the argument, with
the numbers that carry it.

### 12.1 Say the uncomfortable thing first, and say it precisely

What the parser is: 39 hand-written relation templates
(`conflict_analysis.py:18-48`) — 21 of the form `"{s} MID {o}"`, 18 of the
form `"PRE {s} MID {o}"`. They are the benchmark's own fact generator run
backwards. Coverage 99.5 %+, precision effectively 1.0, because the data is
synthetic and template-generated. On this benchmark the parser is not an
approximation of the store's schema; it **is** the store's schema.

What `hnav_idonly` is, then: *"if two retrieved facts share a schema key and
are similar, keep the later serial"* — which is the benchmark's own
supersession rule applied with the benchmark's own schema. A reader will say:
**"you reverse-engineered the generator; of course it works."** That reader
is right, and the thesis must say it before they do. Anything else reads as
concealment once they open `conflict_analysis.py`.

Two things are fair to add, and both are true:

- The benchmark's native RAG pipeline does not do this, none of the six
  vendored memory systems do it, and the answering models cannot do it
  in-context: 2/74 on `sh_6k` conflicted questions with the facts *on the
  page*. "Obvious in hindsight" is not "already done".
- The geometric screens are not independent of the parser either. Their
  training positives come from the gold conflict dataset, whose labels were
  seeded by parser keys and then judge-audited. `hnav_geo` and `hnav_ces` are
  *learned approximations of the parser's decision*, trained under its
  supervision. This should be stated plainly; it is why "parser vs geometry"
  is really "exact schema vs learned proxy for the schema".

### 12.2 How much of the gain is actually the parser? The numbers

Decompose the reference model's gain into the part a **parser-free** screen
reaches (`hnav_geo`) and the part only the schema reaches:

| subset | native | `hnav_geo` (no parser) | `hnav_idonly` (schema) | parser-free share of the gain |
| --- | ---: | ---: | ---: | ---: |
| sh_6k | 30 | 77 (+47) | 95 (+65) | 72 % |
| sh_32k | 53 | 77 (+24) | 85 (+32) | 75 % |
| sh_64k (held out) | 45 | 56 (+11) | 66 (+21) | 52 % |

Across the five models on `sh_64k` the parser-free share is 50–61 % in every
row (§8.1: 11/21, 11/18, 3/5, 4/8, 6/11). So the honest statement is **not** "the gain comes from the
parser". It is:

> Roughly half to three quarters of the gain is reached with no parser at
> all. What the schema buys is the *remaining* quarter-to-half **and the
> zero-harm guarantee**. Geometry gets most of the accuracy and none of the
> safety.

And the safety is not a technicality. A memory governor that silently erases
a key (8 times per 100 questions, on every model) is not deployable, however
good its average accuracy. The void verdict is the result.

### 12.3 What the thesis found that the parser cannot explain

If the parser were the contribution, every finding would be about the parser.
Most are not:

1. **The diagnosis is about the LLM, not the store.** 572/575 conflicted
   errors are the *stale value of the correct key*; 26/26 vs 4/74 on unique
   vs conflicted questions. This holds on five models. It says small models
   cannot do version resolution in-context even when both versions are on
   the page. No parser involved.
2. **The mechanism is deletion, not placement.** Same detector, three edits:
   delete the old versions 66, move the newest to the end 48, move it to the
   front 43. Placement moves the number in *opposite* directions; only
   deletion works. A finding about how small models read a page.
3. **Semantic verification was the weak link, not the safeguard.** The NLI
   gate blocked 2.2 % of true supersessions on calibration and exactly the
   two remaining detection misses on `sh_64k`; waiving it cost nothing in
   15/15 cells. The component that *looked* like the intelligent part was
   removing recall. Counter-intuitive and clean.
4. **The ceiling is retrieval, not governance.** Of the 29 residual
   conflicted errors, 22 are the gold fact never reaching the page and 5 are
   the model answering from its weights. Governance sits at 66 of an
   absolute 72. The next lever is somewhere else, and the thesis proves it.
5. **The geometry negative result has a mechanism.** Pair-level, learned
   geometry is as good as the schema-assisted screen (AUROC 0.972 vs 0.976).
   End-to-end it is unsafe, and the reason is a one-line argument: grouping
   requires an *equivalence relation*, similarity is not transitive, and a
   threshold tuned to zero false positives on one split carried no guarantee
   to the next — 5 of 5 of geometry's unique suppressions were key erasures,
   reproduced byte-identically on five models. That is the most transferable
   finding in the thesis: it applies to any embedding-only memory
   deduplication or consolidation scheme, of which there are many.
6. **The instrument.** Pre-registration with named predictions ({q23, q98}
   landed exactly), void conditions that fired on the author's own preferred
   arm, A/A floors, one-shot held-out, a retracted claim on record, and the
   fp8 case. Plus the audited gold conflict dataset (2,388 core pairs, dual
   labels, cosine-matched negatives) as a reusable artifact.

None of these six is "the parser works".

### 12.4 The reframe: the parser is a write-time schema, and real systems have one

The parser here is exact because the benchmark is synthetic. But the *kind*
of signal it provides — a `(subject, relation)` key per memory — is not
exotic. Production memory systems (mem0, MemOS, graph memories such as
Zep/Graphiti) already extract entities or triples **when they write** a
memory, with one LLM call at write time. The thesis' parser is the
benchmark-exact stand-in for that write-time extraction.

So the realistic question is not "can a memory system parse facts?" — they
already do — but **"given a write-time key, what does read-time supersession
buy, and what happens when the key is absent?"** The thesis answers both:
+21 held-out / +65 short-context at zero harm and zero inference-time LLM
cost with the key; 50–75 % of that gain and a structural harm channel
without it. The parser↔geometry gap is a *bracket* on what an imperfect
extractor would give.

State the limit of the structural proof in the same breath: it assumes keys
are **correct**. A noisy extractor that *splits* one entity into two keys
only loses recall (safe). One that *merges* two entities into one key
produces exactly geometry's failure. The guarantee is "harm-free given
correct identity", and the geometry result shows what happens when identity
is merely probable.

### 12.5 The recommended story

Move the thesis from *"a geometric governance layer for memory"* to
*"supersession governance in evolving memory for small-model RAG — and what
the identity signal has to be"*. Geometry becomes the studied-and-rejected
alternative, not the promise. Chapter skeleton:

1. **Problem.** Small models fail on conflicted memory systematically;
   stale-version errors (§12.3 item 1). Five models.
2. **The oracle.** With the store's schema, read-time supersession by
   deletion: what it achieves, the 72-point ceiling, why it is harm-free,
   why it is free at inference and shortens the prompt. (`hnav_idonly`)
3. **Ablating the oracle.** NLI gate off > on; deletion > demotion >
   promotion; cosine prefilter. Each a clean, paired, per-subset result.
4. **Can the schema be dropped?** Relation-only (`hnav_ces`), nothing
   (`hnav_geo`). Pair-level yes; end-to-end most of the gain, none of the
   safety; the non-transitivity argument; the pre-registered void.
5. **Generality.** Five answering models; plans are model-independent
   (measured); which recovered questions convert is model-dependent.
6. **Limits.** Embedder-specific thresholds; single-valued relations only;
   five models, not all; the schema is exact here and would not be in
   deployment; sh_262k excluded by design.

In this structure the parser is introduced in chapter 2 as *the oracle
identity signal, exact on this benchmark*, and the reader never discovers it
by surprise.

### 12.6 On the disappointment

The disappointing reading is: *"I built a geometry method and a three-line
baseline beat it."* The accurate reading is: *"I built the oracle, measured
how far learned geometry gets toward it (50–75 % of the gain), pre-registered
a held-out test, and found precisely why the remainder cannot be trusted to
geometry — with the failure reproduced on five models."* A negative result
with a mechanism and a pre-registration is worth more to a committee than a
marginal positive that nobody can explain; the version of this thesis where
`hnav_geo` scraped 65 with 8 quiet key erasures would have been *weaker*, not
stronger, and it would have been wrong.

It becomes a real problem only if the text (a) claims geometry as the
contribution, (b) leaves the reader to discover that the parser inverts the
generator, or (c) reports `hnav_geo`'s 56 without the void. Do none of these
and the parser is a strength: it is how the ceiling was measured.

### 12.7 The one experiment that would close the objection

The "you inverted the generator" objection dies if the result survives a
**realistic extractor**. Replace `key(f)` with a key produced by an LLM
extraction pass — one offline call per fact at write time, e.g. a small model
prompted to emit `(subject, relation)` — and run `hnav_idonly` on those keys.
Everything else (pool, cosine, grouping, harm counter, void conditions) is
already built; only the `key()` function changes. If it lands near 60+ on
`sh_64k` with harm at or near zero, the parser is demonstrably a stand-in and
not a trick; if harm appears, §12.4's "merge" caveat is measured rather than
argued. Either outcome is publishable. It needs a fresh pre-registration and
one held-out shot per model, and the extraction cost must be declared as a
write-time cost (it does not touch the "zero inference-time LLM calls" claim,
but it is not free).
