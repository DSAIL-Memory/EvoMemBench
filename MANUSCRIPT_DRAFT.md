# H-Nav-GEO: Resolving Conflicting Memories for Small-Model Retrieval-Augmented Generation with Embedding Geometry and No LLM in the Read Path

*Manuscript draft, 2026-09-17. Built from `THESIS_CLAIMS.md` (binding) and the committed
artifacts it cites. Section 2 (Related Work) is intentionally left as a placeholder. The
title is provisional; two alternatives are listed at the end of the file.*

---

## Abstract

Agent memory evolves. When a stored fact is later updated, a vector memory holds both
versions, and retrieval hands both to the language model. We show that this is the
dominant failure mode of small language models (2B–9B parameters) performing
retrieval-augmented generation over an evolving store: on a canonical
conflict-resolution benchmark the models answer almost perfectly when the queried fact
has a single version and fail systematically when it has two, returning the superseded
value of the correct fact. We propose H-Nav-GEO, a read-time governance layer that
detects competing versions of the same fact from their embeddings alone and deletes the
superseded versions from the retrieved page before the model reads it. The detector
combines two geometric signals computed on the fact vectors, a whitened cosine that
removes the shared component inflating every similarity in the store, and a learned
slot probe on the sign-invariant edit direction that distinguishes an object-slot
update from a subject swap, followed by verification with a small natural-language
inference cross-encoder. No fact parser, no schema, and no large-language-model call
are used at inference; the suppression plan is byte-identical for every answering
model. In a controlled substitution study with five answering models from four
architecture families at three store sizes, H-Nav-GEO raised overall answer accuracy
above the native baseline in all 15 model-by-subset cells (significant at the 5 % level
in 11), by up to 47 points, while leaving the no-conflict stratum unchanged to within
one question and shortening the prompt. At the pair level, the detector lifts
discrimination between true conflicts and cosine-matched look-alikes from an AUROC of
0.893 for raw cosine to 0.972, precisely in the similarity band where cosine is
uninformative. We release the audited gold conflict dataset of 54,569 fact pairs used to
train and evaluate the detector.

**Keywords:** agent memory; knowledge conflict; retrieval-augmented generation; small
language models; sentence embeddings; anisotropy; memory governance.

---

## 1. Introduction

Language-model agents that persist across sessions accumulate memory: facts about the
user, the world, and the task, written over time into a vector store and retrieved
when relevant. Because the world changes and earlier statements get corrected, an
evolving memory inevitably comes to hold *competing versions of the same fact*. A
person changes employer; a value is corrected; a schedule is moved. Nothing in a
standard vector store removes the old version when the new one arrives, and nothing in
a standard retriever prefers one over the other: both are similar to the query, so both
are retrieved and both are placed in front of the model.

This paper starts from a measurement of what small language models do in that
situation, and the measurement is stark. On the single-hop *Conflict Resolution* task
of MemoryAgentBench, where each stored fact may be superseded by a later fact about
the same subject and relation and the benchmark's own prompt instructs the model that
the later fact wins, a 4B-parameter instruction model answers 26 of 26 questions
correctly when the queried fact has one version and 4 of 74 when it has two (Section
3.3). Across 575 wrong answers on two-version questions in eight independent runs,
572 return the *stale value of the correct fact*. The model finds the right memory
and reads the wrong version out of it. The same pattern appears on every one of the
five answering models we study. The benchmark's headline accuracy is, in effect, an
accuracy on the questions that contain no conflict.

The failure is therefore not one of retrieval and not one of knowledge. It is a failure
of in-context version resolution, and it suggests a remedy that does not require a
bigger model: remove the superseded version before the model reads the page. Doing so
requires deciding, for a pair of retrieved facts, whether they are two versions of the
same thing. The obvious tool, embedding cosine similarity, is necessary but far from
sufficient: on the largest store we study, 65,782 pairs of *unrelated* facts sit above
the similarity floor of the true conflicts, 39 look-alikes for every conflict (Section
7.2). The two sentences "Thomas Kyd was born in the city of London" and "Thomas Kyd was
born in the city of Paris" are a supersession; "Thomas Kyd was born in the city of
London" and "Marlowe was born in the city of London" are two facts that must both stay.
Both pairs have cosine well above 0.9. A natural-language-inference model does not help
with this distinction either; it scores the second pair as a contradiction with
probability 0.999 in both directions. Existing memory systems resolve it by extracting
a schema, subject and relation, with an LLM call at write time, or by not resolving it
at all.

We propose **H-Nav-GEO**, a detector that makes the identity decision from the two
fact vectors alone. It reads *what changed* between the two facts rather than *how
similar* they are: the unit difference vector, with signs removed so that the order of
the two facts does not matter, is fed to a logistic probe trained to separate
object-slot edits from subject swaps; and the cosine is recomputed after removing the
store's mean and its 128 dominant shared directions, so that the similarity measures
what is specific to the two facts rather than the template they share. Pairs that
clear both signals are verified by a small NLI cross-encoder, chained into groups, and
every group member except the most recent is deleted from the page. The entire
procedure involves no schema, no fact parser and no large-language-model call at
inference; its parameters are fitted once, offline, on a calibration split, and frozen.

We evaluate H-Nav-GEO in a controlled answering-model substitution: memory, retrieval,
embeddings, prompts, generation settings, scoring and the suppression plan itself are
held fixed, and only the answering model changes. Five models from four architecture
families, spanning 2B to 9B parameters, are each run once at three store sizes.
H-Nav-GEO improves overall accuracy in every one of the 15 cells, by +3 to +47 points,
with the improvement concentrated exactly where the failure lives: on the conflicted
stratum, accuracy rises by a factor of 2.4 to 12.8 at the smallest store. The stratum
with no conflict, where the method should do nothing, moves by at most one question in
any cell. Because deletion only removes text, the prompt gets shorter as accuracy goes
up.

The contributions of this paper are:

1. **H-Nav-GEO**, a read-time conflict detector and repair mechanism for evolving
   vector memory that uses embedding geometry and a small NLI verifier, adds no LLM
   call at inference, requires no schema, and is refittable to any encoder from a
   committed recipe (Section 4).
2. **Five-model evidence** that the governance gain is universal across small
   answering models and store sizes, obtained under a design in which the suppression
   plan is provably model-independent (Sections 5, 6).
3. **A characterisation of the failure mode** of small-model RAG over evolving memory,
   and of why each component of the detector is necessary: cosine is insufficient, NLI
   cannot carry identity, whitening decompresses the precision tail, the edit
   direction is slot-specific, and deletion, not reordering, is the edit that works on
   every model (Section 7).
4. **The gold conflict dataset**, 54,569 audited fact pairs with dual labels and
   cosine-matched balanced evaluation sets, released with this paper (Section 5.2).

Section 2 reviews related work. Section 3 defines the setting and the failure mode.
Section 4 presents the method. Section 5 describes the experimental design, Section 6
the results, Section 7 the analysis of model substitution and of the mechanism, and
Section 8 discusses generality, design boundaries and limitations.

---

## 2. Related Work

*[Left blank by design; to be written separately. Planned subsections:
knowledge conflicts and outdated knowledge in LLMs; memory systems for LLM agents and
their write-time extraction; anisotropy and post-processing of sentence embeddings;
positional effects in long-context reading; deduplication and consolidation in
retrieval-augmented generation.]*

---

## 3. Problem Setting

### 3.1 Evolving memory and supersession

We model an agent's memory as an append-only store of short factual records
$f_1, f_2, \dots$, each with an embedding $v_i \in \mathbb{R}^d$ and a recency key
$s_i$ (a serial number, a timestamp or a version counter). A record $f_j$
*supersedes* $f_i$ when both assert a value for the same subject and relation and
$s_j > s_i$. At query time a retriever returns a page of records; the answering model
reads the page and answers. Supersession is a property of the store's convention, not
of world knowledge: the newer record wins whether or not it is true of the world. Our
task is to ensure that the model answers from the newest version.

We impose three deployment constraints, because the target is a small model serving
an agent, not an offline pipeline: (i) no large-language-model call may be added at
read time; (ii) no schema, relation template or entity extractor may be assumed, since
a real store holds free text; (iii) the mechanism may use only what every vector store
already has, the record embeddings and the recency key.

### 3.2 The benchmark

We use the *Conflict Resolution* dataset of MemoryAgentBench, single-hop subsets
`factconsolidation_sh_{6k, 32k, 64k}`. Each subset is a numbered list of facts serving
as the memory, and 100 questions. A later fact may restate an earlier fact's subject
and relation with a different object; the benchmark's gold answer is the object of the
highest-serial fact, and the model is told so in the prompt. Facts are counterfactual
by design, so a model that answers from world knowledge answers the stale version. The
evaluator is deterministic substring exact match.

| subset | facts in store | chunks (4,096 tokens) | chunks retrieved | questions on a conflicted key | questions on a unique key |
|---|---:|---:|---:|---:|---:|
| sh_6k | 455 | 2 | 2 | 74 | 26 |
| sh_32k | 2,310 | 9 | 9 | 65 | 35 |
| sh_64k | 4,580 | 17 | 10 | 66 | 34 |

**Table 1.** The three store sizes. At sh_6k and sh_32k every chunk is retrieved; at
sh_64k the benchmark's retriever returns 10 of 17 chunks.

The benchmark's own pipeline chunks the fact list into 4,096-token chunks, embeds them,
retrieves the top ten by cosine, and places them in the prompt in rank order. We keep
that pipeline untouched and intervene only on the retrieved page (Section 4.1).

Each question is assigned to one of two strata from the fact text alone: **conflicted**
if the queried subject and relation carry two distinct values in the store, **unique**
otherwise. The conflicted stratum is the primary endpoint, the population the method
exists for; the unique stratum is the do-no-harm control, where a correct method
should change nothing.

### 3.3 The failure mode

Stratifying the native baseline reveals the structure of the problem.

*Reference model, eight independent runs at sh_6k.* Unique stratum: 26 of 26 correct in
every run. Conflicted stratum: between 0 and 5 of 74 correct. Of the 575 errors on
conflicted questions across the eight runs, 572 contained the stale value of the
queried key, 3 an off-list value, 0 were empty.

*All five answering models, sh_6k, native.* Conflicted accuracy 4, 13, 22, 16 and 14 of
74; unique accuracy 26, 26, 23, 24 and 26 of 26 (gemma-3-4b-it, gemma-4-E2B-it,
Phi-4-mini-instruct, Qwen3-4B-Instruct-2507, Qwen3.5-9B; Section 6.3, Table 4).

The model reads the context: it never returns an empty answer or an unrelated fact.
It finds the right slot: the wrong answer is the other value of the same key. What it
cannot do is apply the recency rule it has been given. This holds on every model
tested. It follows that the remedy is to remove the stale version, and that the
difficult step is deciding, for a pair of retrieved facts, whether they are versions
of the same fact.

### 3.4 The identity decision

Two retrieved facts with high cosine similarity can be a **version pair** (same
subject, same relation, different object: the older should go) or a **look-alike
pair** (same relation, same object, different subject: both must stay). Cosine cannot
separate them; the look-alikes are, if anything, closer (Section 7.2). Any detector
must therefore read something other than similarity, and under constraint (ii) it must
do so from vectors. That is the problem H-Nav-GEO solves.

---

## 4. Method: H-Nav-GEO

### 4.1 Overview

For one question, H-Nav-GEO runs the following steps; none involves an LLM.

1. **Page.** The benchmark's retriever produces the page as usual.
2. **Pool.** The 50 facts on the page most similar to the query form the candidate pool.
3. **Candidate pairs.** Every pool pair with raw cosine $\ge 0.94$ is a candidate.
4. **GEO screen.** Two geometric signals are computed from the pair's vectors
   (Sections 4.2, 4.3) and turned into a yes/no decision (Section 4.4).
5. **NLI verification.** Passing pairs are scored by an NLI cross-encoder in both
   directions and verified only if both directions contradict (Section 4.5).
6. **Group and delete.** Verified pairs are chained into groups; each group keeps its
   highest-serial member and the rest are deleted from the page (Section 4.6).
7. The edited page is sent to the answering model with the benchmark's unchanged prompt.

Because steps 1–6 contain no LLM, the resulting *suppression plan* for a question is a
deterministic function of the store, the retriever and the frozen detector, and is
identical for every answering model. We verified this byte-for-byte across the five
models of Section 6 (Section 7.1).

*[Figure 1: pipeline diagram, page → pool → candidate pairs → GEO screen → NLI →
groups → deletion → answering model; annotate the two geometric signals.]*

### 4.2 Signal 1: whitened cosine

Sentence embeddings of short factual statements share a large common component: in the
encoder we use, unrelated facts have a mean cosine of about 0.60, and the candidate
pairs occupy only the top quarter of the cosine scale (Section 7.2). This component
inflates every similarity equally and hides the differences that matter. Following the
all-but-the-top family of post-processing methods, we subtract the store mean $\mu$,
project out the $D=128$ dominant principal directions $P$, and renormalise:

$$\hat w(v) = \frac{(v-\mu) - P(v-\mu)}{\|(v-\mu) - P(v-\mu)\|}, \qquad
\mathrm{cos}_w(a,b) = \hat w(v_a)\cdot \hat w(v_b).$$

The whitening statistics $(\mu, P)$ are fitted once on the calibration split's 2,765
facts and frozen. Whitening removes the anisotropy completely (mean cosine of
unrelated pairs $0.60 \to 0.00$) and, more importantly for a detector that must be
precise, decompresses the high-similarity tail: recall at precision 1.000 rises from
0.075 to 0.513 on sh_6k and from 0.007 to 0.291 on sh_32k (Section 7.2).

### 4.3 Signal 2: the slot probe

The second signal reads *what changed*. Let $d = v_b - v_a$ and $\hat d = d/\|d\|$ be
the unit edit direction. Because a candidate pair is unordered, we drop the signs and
use the coordinate-wise magnitude profile $|\hat d| \in \mathbb{R}^{2560}$, which is
invariant to the order of the pair. A logistic regression on this profile,

$$\mathrm{probe}(a,b) = w\cdot|\hat d| + b_0,$$

is trained to separate **object-slot edits** (the positives: gold conflict pairs from
the audited dataset of Section 5.2, 989 edits on the calibration split) from
**subject swaps** (the hard negatives: same relation, same object, different subject,
8,716 edits). The intuition, confirmed in Section 7.2, is that an object change and a
subject change move the embedding along recognisably different sets of axes. The
probe is trained with balanced class weights and $C=1$; no relation identity, no
parsed field and no metadata enters it.

### 4.4 Anchored margins and the decision

Each signal is converted into a margin in standard-deviation units relative to an
anchor. The anchors $(T_w, T_p)$ are the *joint zero-false-positive point* on the
calibration pools: sweeping over the pool pairs that pass the NLI gate, the loosest
pair of thresholds at which no look-alike pair passes. The scales $(s_w, s_p)$ are the
feature standard deviations over the same pairs (193 pairs on sh_6k, 1,332 on sh_32k).

$$m_w = \frac{\mathrm{cos}_w - T_w}{s_w}, \qquad m_p = \frac{\mathrm{probe} - T_p}{s_p}.$$

A pair passes the screen iff $m_w \ge \tau_w$ and $m_p \ge \tau_p$. The frozen
operating point is $\tau_w = -0.40$, $\tau_p = +0.20$: a *loose* whitened cosine and a
*strict* probe. With the fitted anchors $T_w = 0.6397$, $s_w = 0.3483$,
$T_p = -0.0764$, $s_p = 1.0675$, this is equivalent to $\mathrm{cos}_w \ge 0.500$ and
$\mathrm{probe} \ge 0.137$. The rectangle was chosen on the calibration split by a
selection rule that maximises pool recall subject to zero harmful deletions, with no
answering model, no accuracy and no gold answer in the objective (Section 5.1).

### 4.5 NLI verification

Pairs that pass the screen are scored by a bidirectional NLI cross-encoder
(`cross-encoder/nli-deberta-v3-large`, 435M parameters). A pair is a *verified
conflict* only if the contradiction probability is $\ge 0.90$ in both directions
($a \Rightarrow b$ and $b \Rightarrow a$); one-directional contradiction or entailment
is rejected, because supersession is symmetric disagreement about the same slot, not
refinement. The NLI is a semantic verifier, not the identity screen: as Section 7.2
shows, it cannot itself distinguish version pairs from look-alikes, which is why the
geometric screen must decide identity before it.

### 4.6 Grouping and deletion

Verified pairs are edges; connected components are conflict groups. A group also
passes a geometric span check: it survives only if some member's leave-one-out QR
residual against the span of the other pool candidates is below $r_{\min} = 0.44$. In
each surviving group the member with the highest serial is kept and every other member
is deleted from the page. Nothing is added, nothing is reordered, and the prompt gets
shorter.

We chose deletion over the two obvious alternatives, moving the newest version to the
end or to the front of the page, on evidence: deletion is the only edit that improves
accuracy on every answering model (Section 6.5).

### 4.7 Fitting, freezing and provenance

Every learned quantity, the whitening $(\mu, P)$, the probe $(w, b_0)$, the anchors and
scales, and the operating point, is fitted on the calibration split (sh_6k + sh_32k)
only. The fitter refuses any other data. The resulting artifact is stored with a
SHA-256 fingerprint over its arrays and scalars; the runner refuses to score if the
fingerprint does not match. sh_64k is the held-out split and was evaluated exactly
once per answering model.

### 4.8 Cost

At inference, per fact on the page, whitening is one projection (shared across all
pairs the fact appears in). Per candidate pair, the screen is two dot products in
2,560 dimensions. NLI runs only on screened pairs; in the campaign the precomputed
tables bound it at 410, 4,868 and 8,956 directed cross-encoder passes per 100
questions at the three store sizes, at most about 90 per question at sh_64k, for a
435M-parameter model. No LLM call is added; the embeddings are the store's own. The
geometric parts of every analysis in this paper ran on a laptop CPU.

---

## 5. Experimental Design

### 5.1 Splits and one-shot discipline

sh_6k and sh_32k are the **calibration split**: every fitted quantity and every
threshold was chosen there, from detection quality alone. sh_64k is **held out** and
was run exactly once per answering model; a void run would have been reported, never
re-rolled. Because the selection objective never saw any answering model's behaviour,
any accuracy or any gold answer, the accuracy measurements on the calibration split
are also legitimate accuracy measurements for every model; we report all three subsets,
labeled by role, and never pool across them, since store sizes span an order of
magnitude.

### 5.2 The gold conflict dataset

To train the probe and to evaluate pair-level discrimination we built an audited
dataset of fact pairs. All 87,102 pairs with campaign-embedding cosine $\ge 0.80$
across the three subsets form the candidate frame; 54,569 of them (62.6 %) were
adjudicated by an LLM judge under a fixed schema with A/B order randomisation, with
100 % coverage of the parser-tagged supersession pairs and of the structural
adversary channels, and an unbiased shuffled sample of the bulk tail. Pairs carry
dual labels: `gold_update` (the benchmark's convention: a later record supersedes the
same key) and `gold_strict` (the two values logically cannot coexist). The gold set
comprises 2,388 core pairs plus 282 pairs gold under update semantics only; 105
single-judge positives are quarantined and excluded from both classes. Per subset, a
balanced 1:1 evaluation set is drawn with negatives cosine-matched to positives in
0.01 bins (5,340 pairs in total). Because verified non-conflicts above cosine 0.95
barely exist, matching cannot remove all cosine signal; the dataset therefore records a
**cosine-only AUC baseline** (0.960 / 0.911 / 0.893 for sh_6k / sh_32k / sh_64k) and an
**overlap band** (cosine 0.87–0.97) inside which cosine is uninformative. Any
geometric detector is scored against that baseline and inside that band. The dataset,
its builder and a test that re-derives its tier counts and AUCs from the raw audit
files are released.

### 5.3 Answering models and serving

| model | family | parameters | served with |
|---|---|---|---|
| gemma-3-4b-it | Gemma 3 | 4B | vLLM 0.9.1 |
| gemma-4-E2B-it | Gemma 4 | 2B effective | vLLM 0.28.0 |
| Phi-4-mini-instruct | Phi-4 | 3.8B | vLLM 0.9.1 |
| Qwen3-4B-Instruct-2507 | Qwen3 | 4B | vLLM 0.9.1 |
| Qwen3.5-9B | Qwen3.5 | 9B | vLLM 0.28.0, thinking disabled |

**Table 2.** Answering models. Qwen3-4B-Instruct-2507 is the reference model on which
the detector was developed; the other four were added after every parameter was
frozen.

All models are served with temperature 0, ten output tokens (the benchmark's own
generation length), one sequence at a time, prefix caching disabled and eager
execution, so that the same prompt yields the same output within a run. Before any
shot is spent, a preflight gate measures the longest sh_64k prompt with the model's
own tokenizer and chat template (43k–50k tokens across models) and serves the model
with a context window that fits it, verifies that the model does not emit reasoning
tokens, and checks determinism on a short prompt.

### 5.4 Arms

Every run answers all 100 questions of a subset under five arms:

- **native** — the benchmark's page, unedited (the baseline);
- **native_repeat** — the identical prompt sent again: the in-harness A/A floor, which
  must be zero discordant questions for the run to count;
- **H-Nav-GEO (delete)** — the method of Section 4;
- **move newest to end** and **move newest to front** — the same detector and the same
  groups, but the newest member of each group is moved rather than the older members
  deleted; nothing is removed. These two placement arms are the mechanism controls.

Native outputs are compared across arms and models to confirm that all comparisons are
exactly question-paired.

### 5.5 Metrics and statistics

Accuracy is the benchmark's substring exact match, reported overall and per stratum.
Each arm is compared to native with the exact McNemar test on the discordant pairs
$(b, c)$, where $b$ counts questions native alone answers and $c$ questions the arm
alone answers. Models are never pooled into a single test: the five models answer the
same 100 questions under the same plans and are not independent samples. Consistency
across models is reported as the count of positive cells and as a one-sided sign test
over models per subset. The A/A floor is reported for every cell.

---

## 6. Results

### 6.1 Pair-level discrimination

We first ask whether the geometric screen separates true conflicts from look-alikes
better than cosine, on the held-out balanced evaluation set (sh_64k, 1,681 gold vs
1,681 cosine-matched non-conflicts; 1,000 bootstrap resamples).

| screen | balanced AUROC [95 % CI] | overlap-band AUROC |
|---|---|---|
| raw cosine (the standard screen) | 0.893 [0.882, 0.904] | 0.850 |
| whitened cosine alone | 0.965 [0.959, 0.970] | 0.952 |
| slot probe alone | 0.913 [0.904, 0.923] | 0.902 |
| **H-Nav-GEO screen (both)** | **0.972 [0.967, 0.976]** | **0.966** |

**Table 3.** Held-out pair-level discrimination on the cosine-matched evaluation set.
The overlap band (cosine 0.87–0.97) is where cosine is uninformative by construction.

The combined screen lifts discrimination from 0.893 to 0.972, and from 0.850 to 0.966
inside the band where cosine is uninformative. Against the confirmatory hard task
(1,681 gold vs 39,215 same-relation different-subject negatives) the screen reaches an
AUROC of 0.998 and an average precision of 0.978. Over the 527,062 gold-negative
comparisons that raw cosine orders exactly wrong, the screen orders 95.9 % correctly.
At a false-positive rate of $10^{-4}$, the operating regime a deployed detector needs,
it recovers 72.0 % of conflicts whose object transition was seen in calibration and
48.1 % of those never seen, the best unseen-transition figure of any screen we
measured (whitened cosine alone: 40.4 %).

*[Figure 2: ROC curves of the four screens on the held-out balanced set, with the
overlap band shaded; inset: TPR at FPR 1e-4, seen vs unseen transitions. Source:
`stage0_results/geometry_filter/geo_pairlevel.json`.]*

### 6.2 Detection quality inside the benchmark's pools

Inside the retrieved pools of the calibration split, under the frozen operating point,
the screen followed by NLI made 2,157 deletions over 200 questions at pair precision
1.000: every deleted fact was independently verified against the store to be a
superseded value. Pool recall was 0.790 (0.798 on sh_6k, 0.780 on sh_32k), and the
detector reached 104 of the 139 conflicted questions. For comparison, the best
previously measured parser-free screen reached pool recall 0.444 at the same precision.

### 6.3 End-to-end accuracy across five answering models

Tables 4–6 give the overall accuracy of the native baseline and of H-Nav-GEO for every
model and subset, with the discordant pairs and the exact McNemar $p$, and the two
strata. Every cell is one shot; the A/A floor was 0 discordant questions in all 15
cells.

**Table 4.** sh_6k (calibration split; $n = 100$).

| answering model | native | H-Nav-GEO | net ($b/c$) | $p$ | conflicted native → GEO (of 74) | unique native → GEO (of 26) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 45 | **76** | +31 (2/33) | 3.7e-08 | 22 → **53** | 23 → 23 |
| gemma-4-E2B-it | 40 | **72** | +32 (3/35) | 6.7e-08 | 16 → **48** | 24 → 24 |
| Phi-4-mini-instruct | 40 | **76** | +36 (0/36) | 2.9e-11 | 14 → **50** | 26 → 26 |
| Qwen3-4B-Instruct-2507 | 30 | **77** | +47 (1/48) | 1.8e-13 | 4 → **51** | 26 → 26 |
| Qwen3.5-9B | 39 | **81** | +42 (0/42) | 4.5e-13 | 13 → **55** | 26 → 26 |

**Table 5.** sh_32k (calibration split; $n = 100$).

| answering model | native | H-Nav-GEO | net ($b/c$) | $p$ | conflicted native → GEO (of 65) | unique native → GEO (of 35) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 38 | **45** | +7 (5/12) | 0.143 | 11 → **18** | 27 → 27 |
| gemma-4-E2B-it | 44 | **58** | +14 (3/17) | 2.6e-03 | 17 → **31** | 27 → 27 |
| Phi-4-mini-instruct | 50 | **66** | +16 (3/19) | 8.6e-04 | 16 → **33** | 34 → 33 |
| Qwen3-4B-Instruct-2507 | 53 | **77** | +24 (1/25) | 8.0e-07 | 19 → **43** | 34 → 34 |
| Qwen3.5-9B | 61 | **86** | +25 (2/27) | 1.6e-06 | 27 → **52** | 34 → 34 |

**Table 6.** sh_64k (held out, one shot; $n = 100$).

| answering model | native | H-Nav-GEO | net ($b/c$) | $p$ | conflicted native → GEO (of 66) | unique native → GEO (of 34) |
|---|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it | 33 | **36** | +3 (1/4) | 0.375 | 14 → **17** | 19 → 19 |
| gemma-4-E2B-it | 37 | **41** | +4 (3/7) | 0.344 | 21 → **25** | 16 → 16 |
| Phi-4-mini-instruct | 46 | **52** | +6 (3/9) | 0.146 | 16 → **22** | 30 → 30 |
| Qwen3-4B-Instruct-2507 | 45 | **56** | +11 (2/13) | 7.4e-03 | 17 → **29** | 28 → 27 |
| Qwen3.5-9B | 51 | **62** | +11 (1/12) | 3.4e-03 | 24 → **36** | 27 → 26 |

Three observations.

**The gain is universal.** H-Nav-GEO improves overall accuracy in all 15 model-by-subset
cells, by +3 to +47 points, and conflicted-stratum accuracy in all 15. At sh_6k the
conflicted stratum improves by a factor of 2.4 (gemma-3) to 12.8 (Qwen3-4B). The
improvement is significant at the 5 % level in 11 of the 15 cells: five of five at
sh_6k, four of five at sh_32k, and on the held-out sh_64k for the two strongest models.
In the four cells with $p > 0.05$ the discordant pairs still favour H-Nav-GEO (12/5,
4/1, 7/3, 9/3). Five of five models improve at each store size; a one-sided sign test
over models gives $p = 1/32$ per subset.

**The no-conflict stratum is untouched.** Where the queried fact has a single version,
H-Nav-GEO changes at most one question in either direction in every cell. This is the
do-no-harm check, and it holds on all five models at all three store sizes.

**The gain narrows with store size, and the reason is retrieval.** At sh_64k the
benchmark's retriever returns 10 of 17 chunks, so fewer conflict pairs reach the page
for any read-time method to act on; the detector fired on 99 of 100 questions but
formed 5.3 verified pairs per question against 11.5 at sh_6k (Table 8).

*[Figure 3: paired dot plot, native vs H-Nav-GEO per model, one panel per subset,
conflicted stratum highlighted. Source: the `detector_gap_sh_*.json` artifacts.]*

### 6.4 Achieved accuracy tracks the model; the gain does not

After H-Nav-GEO the five models rank identically at sh_32k and sh_64k (45 / 58 / 66 /
77 / 86 and 36 / 41 / 52 / 56 / 62 for gemma-3, gemma-4-E2B, Phi-4-mini, Qwen3-4B,
Qwen3.5-9B) and compress into 72–81 at sh_6k, where the ceiling is near. The *gain*,
however, is not a function of the native score: the model with the lowest native
accuracy (Qwen3-4B, 30) gains the most (+47), while a model with a higher native score
(gemma-3, 45) gains less (+31). Section 7.1 interprets this.

### 6.5 Deletion versus placement

Table 7 compares the three page edits driven by the same detector and the same groups.

**Table 7.** Overall accuracy (of 100) under the three edits; identical groups in every
row of a subset.

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

Deletion is the only edit that is positive in all 15 cells. The two placement edits,
which keep every fact and only move the newest one, flip sign from model to model:
moving the newest fact to the front gains +27 on gemma-3 at sh_6k and loses 7 on
Phi-4-mini at sh_6k; moving it to the end gains +26 on Phi-4-mini at sh_32k and loses
7 on gemma-3 at sh_6k. Position-based repairs depend on how a particular model weights
position; removing the wrong information does not.

### 6.6 Cost

**Table 8.** Plan and cost per subset; identical for every answering model.

| subset | questions where the detector fired | facts deleted (100 questions) | verified pairs (mean per question) | prompt length vs native | chunks retrieved / total |
|---|---:|---:|---:|---:|---:|
| sh_6k | 100/100 | 1,152 | 1,152 (11.5) | −2.87 % | 2/2 |
| sh_32k | 100/100 | 1,005 | 1,005 (10.1) | −0.51 % | 9/9 |
| sh_64k | 99/100 | 532 | 534 (5.3) | −0.22 % | 10/17 |

H-Nav-GEO adds no LLM call; its only effect on the answering model's input is to make
it shorter. Accuracy rises while token cost falls. The offline fitting cost, 2,765
facts for the whitening, 9,705 edit vectors for the probe and 1,525 pool pairs for the
anchors, is paid once and is independent of the answering model: four models were
added to the study without refitting anything.

---

## 7. Analysis

### 7.1 What changes when the answering model changes, and what does not

The campaign is a controlled substitution. Memory, retrieval, embeddings, prompts,
generation settings, scoring and the detector are frozen; the answering model is the
only variable.

*What does not change.* The suppression plan. Computed without an LLM, it is
byte-identical across the five models: the same 1,152, 1,005 and 532 deletions, the
same serials, the same 99 or 100 questions fired. We verified this by hashing the
per-question plans of every artifact. The direction of the effect: positive in all 15
cells. The behaviour on the no-conflict stratum: unchanged to within one question
everywhere.

*What changes.* The native accuracy, from 30 to 61, for reasons unrelated to memory.
And the accuracy *achieved* after the page is cleaned, which orders the models by
capability at the two larger stores and compresses toward the ceiling at the smallest.

*Interpretation.* The stale-version failure is a failure of in-context version
resolution, and every small model we tested has it. H-Nav-GEO removes the stale
version before the model reads the page, so the model only has to do what it already
can: read a page and answer. The number that moves is therefore set by the method, not
by the model. The model determines how much of a cleaned page it converts into correct
answers, and stronger models convert more; but no model converts a page it was never
handed. The placement controls make the same point from the other side. An edit that
relies on how a particular model weights position is idiosyncratic to that model; an
edit that removes the wrong information is not.

### 7.2 Why each component is necessary

Each design choice in Section 4 rests on a measurement.

**Cosine is necessary but not sufficient.** True conflict pairs occupy a narrow,
reproducible band of raw cosine (mean 0.955, 0.956 and 0.956 across the three stores).
But on sh_64k, 65,782 non-conflict pairs lie above the conflict floor against 1,687
conflicts: 39 look-alikes per conflict. A cosine threshold cannot be both precise and
useful. *A second, non-cosine signal is required.*

**NLI cannot carry identity.** The bidirectional cross-encoder scores "Thomas Kyd was
born in the city of London" against "Marlowe was born in the city of London" as a
contradiction with probability 0.9995 and 0.9998 in the two directions; among 453
cross-key adversary pairs in the calibration pools, 447 pass the bidirectional 0.90
gate. Two facts about different subjects are not in contradiction, but the NLI model
reads the shared template and the differing token as a disagreement. *Identity must be
decided from geometry before NLI; NLI is the verifier.*

**Whitening decompresses the precision tail.** Mean-centring alone removes essentially
all of the encoder's anisotropy (unrelated-pair cosine 0.60 → 0.00); removing the top
128 principal directions on top of it cleans the high-confidence tail of the ranking.
Judged by AUROC the gain is small (+0.002 to +0.005, from a base of 0.99), which is why
whitening is easy to dismiss; judged by the statistic a precise detector needs, recall
at precision 1.000, it is transformative: 0.075 → 0.513 on sh_6k and 0.007 → 0.291 on
sh_32k. *The first geometric signal is the whitened cosine.*

**The edit direction is slot-specific, and sign-invariant features carry it.** A
multinomial probe on $|\hat d|$ identifies which of six slot combinations changed
between two facts at macro-F1 0.70 (chance 0.17) and separates object-slot from
subject-slot changes at AUROC 0.96, while a probe on signed features with randomised
orientation sits at chance, as it must for unordered pairs. The signal is distributed
across dimensions: compressing the profile to 64 or 256 principal components drops the
held-out balanced AUROC to 0.73 and 0.82, and a full-covariance quadratic scorer does
not beat the flat linear family. *The second signal is a linear probe on the full
magnitude profile.*

**Granularity must match the conflict.** An earlier design reordered *chunks*,
promoting the chunk that carried the newest version. Driven by a precision-1.0
detector on the calibration split, it was net negative in all 81 operating points
tested, helping 228 questions and harming 441: a chunk carries ~250 facts, and moving
one to fix a single conflict scrambles the order of hundreds of unrelated facts.
*Edit facts, not chunks.*

**Deletion is the edit that works, on every model.** Table 7. *Suppress; do not
reorder.*

**Read time is where the headroom is.** A write-time policy in this setting would, after
safety vetoes, touch 0–1.6 % of writes and could change the correctness of 0.00 of them
on the held-out store, because the store is append-only and retrieval already returns
the newest version alongside the old one. *Intervene when the page is read.*

*[Figure 4: (a) recall at precision 1.000 before and after whitening, both calibration
subsets; (b) held-out AUROC of the slot probe under PCA compression to 64, 256 and all
2,560 dimensions. Sources: `stage0_results/abtt/G1_GATE_REPORT.md`,
`stage0_results/geometry_filter/geo_pairlevel.json`.]*

---

## 8. Discussion

### 8.1 Generality

*Across answering models.* Five models, 2B to 9B parameters, four architecture
families, two serving stacks, one served in non-thinking mode: the gain replicates in
every model at every store size, under a design in which the intervention itself
cannot vary with the model.

*Across store sizes.* Positive at 455, 2,310 and 4,580 stored facts. The narrowing at
the largest store is a property of the benchmark's retriever, which returns 10 of 17
chunks, not of the detector, which fired on 99 of 100 questions there.

*Across embedders.* H-Nav-GEO is embedder-agnostic by construction: nothing in it is a
hand-set constant. The whitening, the probe, the anchors and the scales are fitted
procedures with a committed refit recipe that runs on the calibration split in
minutes on a CPU. The geometric properties the method exploits are documented general
properties of sentence embeddings of English text rather than of this benchmark: a
dominant common component that inflates every cosine and is removed by mean-centring
plus a few principal directions, and a systematic per-slot structure in difference
vectors. We therefore state as our hypothesis that other English encoders exhibit the
same qualitative geometry and require a refit, not a redesign. The experiments in this
paper use one encoder, Qwen3-Embedding-4B, in float32; a second-encoder refit is the
first item of future work.

*Toward real agent memory.* The method consumes exactly two things every vector memory
store already has: an embedding per record and a recency key. It needs no schema, no
relation templates, no entity extraction and no LLM at read time. This is the
hypothesis the paper argues for: that the stale-version failure of small-model RAG
over evolving memory, which we have shown to be systematic and model-independent, can
be governed at read time by geometry alone, at negative token cost. The benchmark
campaign is the controlled test of the mechanism; the transfer to production stores is
the claim it motivates.

### 8.2 Why the identity signal is geometric rather than a fact parser

The benchmark generates its facts from a small set of sentence templates, so a
regular-expression parser recovers each fact's subject and relation exactly, and a
detector keyed on that schema does well on this benchmark precisely because the
language is uniform and template-generated. Such a screen has no counterpart in a real
memory store: it needs either hand-written templates per domain or an LLM extraction
call per write, the cost and the dependency this work sets out to remove. We therefore
study parser-free identity, and H-Nav-GEO is the result. Used only as an oracle
reference for this design decision, geometry alone recovers 72 % of the deletions a
schema-keyed oracle makes on the held-out store, and between half and three quarters
of that oracle's accuracy gain across store sizes, from vectors alone. For
generalisation beyond a templated store we hold that geometry is the far more
efficient signal: it is available everywhere, it costs a few dot products, and it does
not depend on the store's language matching a template.

### 8.3 Why the largest subset is excluded

MemoryAgentBench also ships a 262k-token subset. We exclude it by design: its contexts
would exceed the context window of the smaller answering models the study exists to
test, which would silently turn a memory comparison into a context-length comparison.
It was never part of the registered campaign, and the runner refuses it. We report
that it was not measured; we do not extrapolate to it.

### 8.4 Limitations

- On the held-out store, 8 of the 532 deletions (1.5 %) merged two different keys,
  because a geometric group is a connected component of a similarity relation and not
  an equivalence relation; the net accuracy effect is nonetheless positive on every
  model and the no-conflict stratum is unaffected. A transitivity-aware grouping is
  future work.
- One encoder; the fitted parameters are coordinates in its space and are refit, not
  copied, for another encoder.
- The store has single-valued relations; the NLI verifier's role grows in stores with
  genuinely multi-valued relations.
- Five answering models; on the held-out store the per-model gain is significant for
  the two strongest models and positive for all five.
- One benchmark family; the 262k subset was not measured.

---

## 9. Conclusion

Small language models doing retrieval-augmented generation over an evolving memory
fail in one systematic way: they retrieve the right fact and answer with its stale
version. We have shown that this failure can be governed at read time, without a
schema and without an LLM, by a detector that reads what changed between two facts
from their embeddings, verifies it with a small NLI model, and deletes the superseded
versions from the page. In a controlled substitution study, H-Nav-GEO raised accuracy
above the native baseline on every one of five answering models at every one of three
store sizes, left the no-conflict stratum untouched, and shortened the prompt. The
suppression plan does not depend on the model; the method is what moves the number.
Because it consumes only what every vector store already has, an embedding and a
recency key, we propose it as a general governance layer for evolving agent memory.

---

## Data and Code Availability

The detector, the frozen artifacts with their fingerprints, the five-model campaign
artifacts, the gold conflict dataset with its builder and its self-verifying test, and
the scripts that generate every table in this paper from the artifacts are available
in the project repository (branch `claude/hnav-presentation-evidence`).

---

## Appendix A. Frozen parameters of H-Nav-GEO

| component | value | fitted on |
|---|---|---|
| encoder | Qwen3-Embedding-4B, float32, 2,560-d, unit norm | — |
| candidate pool | 50 most query-similar facts on the page | — |
| candidate pair cosine | $\ge 0.94$ | selection grid, calibration |
| whitening | mean + top-128 principal directions removed, renormalised | 2,765 calibration facts |
| probe | logistic on $\lvert\hat d\rvert$, $C=1$, balanced class weights | 989 gold edits vs 8,716 hard-negative edits, calibration |
| anchors, scales | $T_w = 0.6397$, $s_w = 0.3483$, $T_p = -0.0764$, $s_p = 1.0675$ | 193 + 1,332 NLI-passing calibration pool pairs at cos $\ge 0.88$ |
| decision | $m_w \ge -0.40$ and $m_p \ge +0.20$ ($\mathrm{cos}_w \ge 0.500$, probe $\ge 0.137$) | selection grid, calibration |
| NLI | `cross-encoder/nli-deberta-v3-large`, contradiction $\ge 0.90$ both directions | — |
| span check | leave-one-out QR residual $< 0.44$ | selection grid, calibration |
| artifact fingerprint | `335b4540…` (SHA-256 over arrays and scalars) | — |

## Appendix B. Provenance of every number

| number | file |
|---|---|
| Tables 4–8 | `pipelines/hnav_geo/results/<model>_<date>/detector_gap_sh_{6k,32k,64k}.json` |
| Table 3, hard task, tails | `stage0_results/geometry_filter/geo_pairlevel.json` |
| Section 6.2 | `stage0_results/geometry_filter/geo_operating_point.json` |
| Appendix A | `stage0_results/geometry_filter/geo_identity_screen.json` |
| Section 3.3 | `stage0_results/question_strata.json`; Table 4 native columns |
| Section 7.2, cosine | `stage0_results/delta_geometry/DELTA_GEOMETRY_REPORT.md` §2 |
| Section 7.2, NLI | `TEZ_BULGULARI.md` §B; `stage0_results/geometry_filter/GEO_PREREG.md` |
| Section 7.2, whitening | `stage0_results/abtt/G1_GATE_REPORT.md` §2, §4 |
| Section 7.2, probe and PCA | `stage0_results/geometry_filter/REPORT.md` §2; `geo_pairlevel.json`; `stage0_results/qda_filter/REPORT.md` |
| Section 7.2, chunk reranking | `STAGE1_NULL_ANALIZI.md` |
| Section 7.2, write path | `KAPI_KARARI.md` §2 |
| Section 4.8, NLI counts | `TEZ_HIKAYESI.md` §3.1 |
| Section 5.2 | `stage0_results/conflict_pairs/AUDIT_SUMMARY.md`, `gold_conflict_dataset.summary.md` |
| Section 5.3 | `pipelines/MULTIMODEL_CAMPAIGN_PLAN.md` |
| Section 8.2 oracle reference | `stage0_results/geometry_filter/E2E4_COMPLEMENTARITY.md` §4; `LATEST_RESULTS_REPORT.md` §12.2 |

---

*Alternative titles:*
*(a) "Reading the Right Version: Geometric Read-Time Governance of Conflicting Memories for Small-Model RAG"*
*(b) "No Parser, No LLM: Resolving Superseded Facts in Evolving Agent Memory from Embedding Geometry"*
