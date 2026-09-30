# How H-Nav Works — A Step-by-Step Guide

*Version: 2026-08-16. Describes the system as it exists after the Stage-1
confirmatory run. Every number cited here is traceable to a committed artifact;
see `TEZ_BULGULARI.md` for the evidence ledger.*

---

## 1. The problem in one paragraph

An LLM agent with memory accumulates facts over time. Some of those facts
**supersede** earlier ones — a person changes jobs, a value is corrected, a
record is updated. When the agent is later asked a question, the retrieval
system hands it a page of remembered text that may contain **both the old and
the new version**. The question this project asks is simple:

> Does the model reliably use the newer fact? And if not, can a governance layer
> fix that without breaking anything else?

The answer we measured is: **no, it does not** — and **yes, a very specific kind
of intervention fixes most of it.**

---

## 2. The benchmark: what it is and what it actually tests

### 2.1 Where we measure

We measure inside **MemoryAgentBench**, a benchmark suite that evaluates agent
memory across four quadrants (in-episode vs cross-episode × knowledge vs
execution). We use one quadrant and one dataset:

```
EvoMemBench/
└── In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/
    └── dataset: Conflict_Resolution
        └── subsets: factconsolidation_sh_6k
                     factconsolidation_sh_32k     ← our tuning ("calibration") data
                     factconsolidation_sh_64k     ← held out for the final test
                     factconsolidation_sh_262k    ← held out, not used
```

`sh` means *single-hop*: each question maps to exactly one fact. (There are also
`mh` multi-hop subsets; we excluded them because the question→fact mapping is
not 1:1, which makes ground truth ambiguous.)

The four subsets are the **same task at four store sizes**:

| subset | facts in the store | chunks | questions |
|---|---|---|---|
| sh_6k | 455 | 2 | 100 |
| sh_32k | 2,310 | 9 | 100 |
| sh_64k | 4,580 | 17 | 100 |
| sh_262k | 18,332 | 67 | 100 |

### 2.2 What one task looks like

Each subset gives the model a **numbered list of facts** as its memory:

```
0. Thomas Kyd was born in the city of London.
1. The chairperson of Fatah is Mahmoud Abbas.
2. Amy Winehouse died in the city of Camden Town.
...
91. Nobuhiro Watsuki is famous for Rurouni Kenshin.
...
259. Nobuhiro Watsuki is famous for The Fairly OddParents.
...
454. ...
```

The rule the benchmark encodes is: **a higher serial number means a newer
fact**. When two facts share the same subject and relation but give different
values, the higher serial wins.

So in the example above, facts 91 and 259 are about the same subject
(`Nobuhiro Watsuki`) and the same relation (`is famous for`), but disagree.
Fact 259 is newer, so it is the truth *for this benchmark*.

### 2.3 A worked question

> **Question:** Based on the provided Knowledge Pool, what is Nobuhiro Watsuki
> famous for?
>
> **Gold answer:** `The Fairly OddParents`   ← fact 259, the newer one
>
> **What the base model answers:** `Rurouni Kenshin`   ← fact 91, the *stale* one

Notice what makes this hard and interesting: `Rurouni Kenshin` is the
**real-world truth**. The benchmark deliberately injects counterfactual updates,
so a model that leans on world knowledge instead of reading the context will
answer the *old* fact. This is by design — it is how the benchmark separates
"the model read the memory" from "the model already knew."

### 2.4 Two kinds of question — and why the distinction matters enormously

Not every question involves a conflict. We classified all 400 questions:

| subset | questions on a **non-conflicted** key | questions on a **conflicted** key |
|---|---|---|
| sh_6k | 26 | 74 |
| sh_32k | 35 | 65 |
| sh_64k | 34 | 66 |
| sh_262k | 22 | 76 (+2 unmatched) |

This split turned out to be the single most important measurement in the whole
project:

| | non-conflicted questions | conflicted questions |
|---|---|---|
| model's accuracy (sh_6k, 8 independent runs) | **26/26 — every run** | **0–5 / 74** |

The model answers *perfectly* when there is no conflict, and **almost never**
answers correctly when there is one. And of ~575 conflicted-question errors
across those runs, **572 were the stale value of the correct key** — not a
refusal, not a hallucination, not an unrelated fact. The model finds the right
memory slot and reads the wrong version out of it.

**Consequence:** the benchmark's headline accuracy (0.29–0.45) is carried almost
entirely by the questions that contain no conflict. On the questions the
benchmark exists to test, base accuracy is roughly 5–26%.

### 2.5 How the benchmark scores an answer

The evaluator is `substring_exact_match`: the gold string must appear in the
model's output. It is deterministic, offline and free — no LLM judge, no
subjective grading. The model is capped at **10 output tokens**, so answers are
short phrases like `The Fairly OddParents`.

---

## 3. What goes into the model, and what comes out

### 3.1 The prompt, anatomically

Every question produces one prompt with three parts:

```
┌── SYSTEM MESSAGE ─────────────────────────────────────────────┐
│ (the benchmark's standard system prompt)                      │
└───────────────────────────────────────────────────────────────┘
┌── MEMORY BLOCKS ──────────────────────────────────────────────┐
│ Memory 1:                                                     │
│ 0. Thomas Kyd was born in the city of London. 1. The chair... │
│                                                               │
│ Memory 2:                                                     │
│ 228. ... 229. ... (the next chunk of facts)                   │
│  ... up to 10 blocks (top_k = 10) ...                         │
└───────────────────────────────────────────────────────────────┘
┌── QUERY TEMPLATE (verbatim from the benchmark) ───────────────┐
│ "Pretend you are a knowledge management system. Each fact in  │
│  the knowledge pool is provided with a serial number at the   │
│  beginning, and the newer fact has larger serial number.      │
│  You need to solve the conflicts of facts in the knowledge    │
│  pool by finding the newest fact with larger serial number... │
│  ...                                                          │
│  Now Answer the Question: Based on the provided Knowledge     │
│  Pool, {question}                                             │
│  Answer:"                                                     │
└───────────────────────────────────────────────────────────────┘
```

**Read that query template again.** The benchmark *explicitly tells the model
the rule*: higher serial = newer, resolve conflicts by taking the newest. And
the model still fails 95% of conflicted questions. That is not a prompt-design
oversight; it is a genuine limitation of the model's ability to follow an
explicit precedence rule inside a long context.

**Input size:** ~6,700 tokens at sh_6k, up to ~50,000 at sh_64k.
**Output:** at most 10 tokens, temperature 0.

### 3.2 How memory gets from the store into the prompt

```
455 facts
   │  chunk_text_into_sentences()   ← the benchmark's own chunker
   ▼
2 chunks of ~4,096 tokens each      (sh_64k: 17 chunks; sh_262k: 67)
   │  embed each chunk, build a FAISS index
   ▼
retrieve top-10 chunks by similarity to the question
   │
   ▼
those chunks become "Memory 1 … Memory N" in the prompt
```

At sh_6k and sh_32k there are only 2 and 9 chunks, so **everything is always
retrieved** — retrieval is lossless and the model sees the entire store. At
sh_64k only 10 of 17 chunks fit, so ~41% of the store never reaches the model.
This distinction matters later.

---

## 4. Baseline vs H-Nav, side by side

**Baseline (`HNAV_MODE=off`)** — H-Nav is inert, byte-for-byte:

```
question → retrieve top-10 chunks → build prompt → LLM → answer
```

**With H-Nav (`HNAV_MODE=live`)** — one extra step, between retrieval and the
prompt:

```
question → retrieve top-10 chunks → ★ H-NAV READ GATE ★ → build prompt → LLM → answer
                                    (edits the page)
```

H-Nav **never changes what is retrieved**. It only edits what is inside the
retrieved page before the model sees it.

---

## 5. The H-Nav pipeline, step by step

We follow the Nobuhiro Watsuki example all the way through.

### Step 1 — Explode the page into individual facts

The retrieved chunks are big blobs of ~230–260 facts each. H-Nav parses them
back into individually addressable facts using a validated parser (99.5%+
coverage):

```
chunk "Memory 1"  →  (0,  "Thomas Kyd was born in the city of London.")
                     (1,  "The chairperson of Fatah is Mahmoud Abbas.")
                     ...
                     (91, "Nobuhiro Watsuki is famous for Rurouni Kenshin.")
                     ...
                     (259,"Nobuhiro Watsuki is famous for The Fairly OddParents.")
```

**Why facts and not chunks?** Because we measured chunk-level operation and it
was *actively harmful* — see §6.1.

### Step 2 — Geometric filter: which pairs even look like a conflict?

Every fact is embedded into a vector. Two cheap geometric signals narrow the
field:

**(a) Pair cosine similarity** — how close two facts are in meaning-space.
Conflicting facts are near-identical except for one value, so they sit very
close together:

```
"Nobuhiro Watsuki is famous for Rurouni Kenshin."
"Nobuhiro Watsuki is famous for The Fairly OddParents."
                                        cosine ≈ 0.97   ← candidate ✓

"Nobuhiro Watsuki is famous for Rurouni Kenshin."
"Amy Winehouse died in the city of Camden Town."
                                        cosine ≈ 0.55   ← not a candidate ✗
```

The threshold (0.90) was chosen on calibration data only. For scale: across
the whole benchmark, conflicting pairs have a median similarity of **0.964**
against **0.60** for random pairs, and separate with **AUC ≥ 0.9999**. The
geometric premise is extremely strong here.

**(b) Span residual** — "how much of this fact is *new* relative to the
others?" Formally we take one member out of a group and ask how well it can be
rebuilt from the span of the remaining ones. A near-restatement rebuilds almost
perfectly (residual ≈ 0); genuinely new information does not (residual ≈ 1).
This screens out groups that merely look similar.

After this step we have **conflict candidates** — but they are not yet trusted.

> **The mathematics behind this step is written out in full in §11.** If you want
> to know what "span residual" actually computes, why we remove principal
> components before measuring similarity, or why one of our thresholds turned
> out to be arithmetically unreachable, read that section.

### Step 3 — Subject-identity screen: is it really the same thing?

This step exists because of a failure we measured, and it is the single most
transferable finding in the project.

Consider these two facts:

```
"Thomas Kyd was born in the city of London."
"Marlowe was born in the city of London."
```

Same template, same relation, same value — **different people**. These do not
conflict at all. Both can be true.

H-Nav parses each fact into `(relation, subject, value)` and **requires the
relation and subject to match**:

```
(born in the city of, Thomas Kyd, London)
(born in the city of, Marlowe,    London)
        ↑ subjects differ → REJECT, not a conflict
```

Unparseable pairs are **rejected**, not trusted — the conservative direction.

### Step 4 — Bidirectional NLI: does a language model agree they contradict?

Surviving pairs are scored by a natural-language-inference cross-encoder
(`nli-deberta-v3-large`) in **both directions**:

```
premise: "Nobuhiro Watsuki is famous for Rurouni Kenshin."
hypothesis: "Nobuhiro Watsuki is famous for The Fairly OddParents."
        → contradiction 0.99+   ✓

premise and hypothesis swapped
        → contradiction 0.99+   ✓

Both directions agree → VERIFIED CONFLICT
```

If only one direction says contradiction, the pair is rejected. (A one-way
contradiction usually means one statement is more specific than the other, not
that they conflict.)

**Why both the screen and the NLI?** Because either alone is insufficient:

| configuration | false-verification rate |
|---|---|
| bidirectional NLI **alone** | **33–93%** |
| NLI **+ subject-identity screen** | **0.000** (2,673 verified pairs, precision 1.000) |

The Kyd/Marlowe pair scores contradiction at **0.9995 in both directions**.
NLI alone would happily "verify" it. The parsed screen is what makes the
detector safe.

### Step 5 — Identify the newest member

Within each verified group, the fact with the **highest serial number** is the
current one:

```
group: { fact 91  → "Rurouni Kenshin"
         fact 259 → "The Fairly OddParents" }   ← LATEST (higher serial)
```

If the newest member cannot be identified unambiguously (ties, unparseable
serials), H-Nav **does nothing to that group**. It refuses to guess.

### Step 6 — Suppress the stale fact

H-Nav deletes the superseded fact **from the page text**, byte-exactly:

```
BEFORE (what the model would have seen):
  ... 90. ... 91. Nobuhiro Watsuki is famous for Rurouni Kenshin. 92. ...
  ... 258. ... 259. Nobuhiro Watsuki is famous for The Fairly OddParents. 260. ...

AFTER (what the model actually sees):
  ... 90. ... 92. ...
  ... 258. ... 259. Nobuhiro Watsuki is famous for The Fairly OddParents. 260. ...
```

Rules that make this safe:

- **No renumbering.** Serial 91 simply disappears; the gap is left alone. The
  benchmark's precedence rule depends on serials, so rewriting them would
  corrupt the very signal the model is told to use.
- **No reflow, no reformatting.** Every other byte is identical.
- **Same page, same number of blocks.** Only content inside a block changes.
- **Fewer tokens, never more** — measured at −0.31% to −3.48%.
- **Any irregularity → fall back to the original page**, and count it. A silent
  fallback would look like a null result, so the counter is a hard failure
  condition, not a warning.

### Step 7 — The model answers

```
Question: What is Nobuhiro Watsuki famous for?
Page now contains only: 259. Nobuhiro Watsuki is famous for The Fairly OddParents.
Model answers: "The Fairly OddParents"   ✓ correct
```

---

## 6. Why it is built this way — every design choice has a measurement

This is the part that matters most for a thesis: **nothing in the final design
is there by intuition.** Three earlier designs were built, measured, and
discarded.

### 6.1 Why facts, not chunks

Our first read-path design reordered **chunks** — promoting the chunk containing
the newest fact toward the top of the page. We tested 162 different
configurations of it.

| result | |
|---|---|
| configurations with net benefit | **0 of 162** |
| total questions helped | 228 |
| total questions harmed | **441** |

It was not neutral — it was **twice as harmful as helpful**, with a detector
operating at precision 1.000. The reason is granularity: **a chunk carries
~230–260 facts.** Moving one chunk to fix a single conflict scrambles the
relative order of hundreds of unrelated facts. The effective signal-to-noise
ratio is about 1:250.

### 6.2 Why deletion, not reordering

Once we operated at fact level, we tested three interventions against each
other on calibration data (an "oracle" version, using gold answers to identify
the target — measuring the **ceiling**, not a shippable system):

| intervention | conflicted accuracy sh_6k | sh_32k |
|---|---|---|
| do nothing (baseline) | 4/74 (5.4%) | 7/65 (10.8%) |
| **delete the stale fact** | **66/74 (89.2%)** | **53/65 (81.5%)** |
| move the newest fact to the **end** | 20/74 (27.0%) | 33/65 (50.8%) |
| move the newest fact to the **front** | 1/74 (1.4%) | 4/65 (6.2%) |

Three things fall out of this table:

1. **Deletion is dramatically effective** — the failure is *repairable*.
2. **Position matters, and the model anchors on late-appearing text.** Moving
   the newest fact to the end helps *without deleting anything*. Moving it to
   the front hurts.
3. **This retroactively explains §6.1's failure.** The chunk design promoted the
   newest fact *upward* — i.e. away from the position that helps.

### 6.3 What is switched off, and why we say so

The operating point disables H-Nav's original "ambiguity precondition"
(entropy/margin screens from Stage 0). Two honest reasons:

- Those screens are the **only** inputs contaminated by an embedding-truncation
  defect we found and fixed (see §7.3), and re-fitting them was not possible in
  time for the confirmatory run.
- Keeping them collapsed the detector's recall from 0.957 to 0.403.

**The consequence must be stated plainly:** with the screen off, H-Nav evaluates
*every* question (though it only acts on verified conflicts). So the shipped
system is **"a fact-level conflict detector applied unconditionally"**, not
"H-Nav's validated Stage-0 gate." What carries the result is the detector's
precision, not the gate.

---

## 7. How performance is measured — the experimental protocol

### 7.1 The split discipline

```
sh_6k  + sh_32k   →  CALIBRATION.  All tuning happens here. Thresholds,
                     operating points, design choices.
sh_64k            →  HELD OUT.  Touched exactly once, for the final test.
sh_262k           →  HELD OUT.  Not used at all.
```

Every script refuses held-out subsets by default; the confirmatory run required
an explicit flag that itself refuses any configuration other than the
pre-registered one.

### 7.2 Pre-registration

Before the final run, we committed a document specifying:

- the exact claim being tested,
- the arms, the number of questions, the analysis code,
- the success criterion (**net ≥ +10 discordant pairs, exact p < 0.01, token
  cost ≤ 0**),
- the harm criterion and what would void it,
- a **falsifiable side-prediction** (exactly 2 questions where the gold fact
  would be deleted, derived from the parse with no model involved),
- and eight "void conditions" that would invalidate the run.

Commit timestamps prove the pre-registration (19:52:24) preceded the run
(22:00:02). No optional stopping; one shot.

### 7.3 Controls that make the numbers trustworthy

| control | why it exists | result |
|---|---|---|
| **A/A floor** — grade the untouched baseline twice | the server is not perfectly deterministic; effects must clear its noise | **0/0 discordant** on every run |
| **`anti` arm** — do the *opposite* of the intervention | if position causes the effect, the reverse must hurt | −1, −6, −2 across three subsets |
| **stratified reporting** — conflicted vs non-conflicted | the non-conflicted stratum is natively ~100% with *zero* observed noise, so any drop there is signal | see §8 |
| **precision audit** — every deleted fact checked against ground truth | deleting a current value would be silent damage | **735/735 correct, 0 errors** |

A measured detail worth knowing: identical baseline runs of this benchmark
differ on ~3.3% of questions (a 4-point accuracy swing), and that noise is
confined **entirely** to the conflicted stratum. Any single-run claim smaller
than that band is not interpretable.

---

## 8. The result

**Held-out sh_64k, single pre-registered run, 100 questions × 5 arms:**

| arm | overall | non-conflicted | conflicted | vs baseline |
|---|---|---|---|---|
| baseline | 0.450 | 28/34 | 17/66 (25.8%) | — |
| baseline repeated (A/A) | 0.450 | 28/34 | 17/66 | 0/0 discordant |
| **H-Nav suppression** | **0.640** | 27/34 | **37/66 (56.1%)** | **+20, p = 1.9×10⁻⁶** |
| placement (newest last) | 0.480 | 28/34 | 20/66 | +3, n.s. |
| anti (newest first) | 0.430 | 28/34 | 15/66 | −2, n.s. |

- **Accuracy up:** conflicted-stratum accuracy more than doubled.
- **Tokens down:** −0.31%. The intervention is *cheaper* than the baseline.
- **Zero conflicted questions harmed** — the McNemar b-cell is literally 0.
- **One non-conflicted question regressed**, which voids the pre-registered
  safety criterion (see §9).
- **Detector recovered 96–98% of the oracle ceiling** on calibration — i.e.
  running without gold answers costs almost nothing.

---

## 9. What H-Nav does *not* do

Being precise about the boundaries is what makes the rest credible.

- **It does not govern writes.** We measured write-side intervention headroom in
  this arena and it was ~0 (0.00 could-change-correctness on held-out data).
  `write_policy.py` is permanently forbidden by a test.
- **It is not proven safe.** On the held-out run, one non-conflicted question
  regressed: the model *refused to answer* after the edit, even though the fact
  it needed was still on the page. That failure mode ("refusal after edit") is
  understood as a mechanism but not eliminated. The registered conclusion is
  therefore: **effective, but not yet safe** — not recommended for deployment on
  traffic containing non-conflicted queries until that is fixed.
- **It does not generalize (yet).** One benchmark, one model (Qwen3-4B-Instruct),
  one scale, one shot.
- **It does not archive or canonicalize.** Those parts of the original design had
  no substrate to be measured against in this repository.
- **It does not fix a store the retriever cannot reach.** At sh_64k, 41% of the
  store never enters the page; H-Nav only edits what was retrieved.

---

## 10. One-page summary

| | |
|---|---|
| **Where** | MemoryAgentBench → `Conflict_Resolution` → `factconsolidation_sh_*` |
| **What the benchmark tests** | can the model use the *newest* of two conflicting memories, when told the rule explicitly |
| **Model input** | system message + up to 10 retrieved memory blocks + templated question (~6.7k–50k tokens) |
| **Model output** | ≤10 tokens, scored by exact substring match |
| **The failure** | the model answers the **stale** value on ~95% of conflicted questions |
| **H-Nav's action** | detect superseded facts in the retrieved page and delete them — nothing else |
| **How it detects** | fact-level: cosine + span residual → parsed subject-identity screen → bidirectional NLI |
| **Detector precision** | 1.000 (735/735 deletions correct on held-out data) |
| **Result** | conflicted accuracy 25.8% → 56.1% held out, p = 1.9×10⁻⁶, tokens −0.31% |
| **Caveat** | one non-conflicted regression voids the safety criterion: *effective but not yet safe* |

---

## 11. The geometry in detail — what mathematics is actually used

This section expands §5 Step 2. Everything here is linear algebra on unit
vectors plus one information-theoretic family; there is no learning, no training
and no gradient anywhere in H-Nav's detector. Source: `hnav/core/geometry.py`,
`hnav/core/read_gate.py`, `hnav/core/retrieval_signals.py`.

### 11.1 The space we work in

Each fact is turned into a vector by the embedding model:

$$v \in \mathbb{R}^{d}, \qquad d = 2560 \quad (\text{Qwen3-Embedding-4B})$$

and **L2-normalized**, so every fact lives on the unit sphere:

$$\lVert v \rVert_2 = 1$$

Normalization is what makes the rest cheap: for unit vectors the cosine
similarity is just the dot product,

$$\cos(u,v) \;=\; \frac{u \cdot v}{\lVert u\rVert\,\lVert v\rVert} \;=\; u \cdot v$$

so a similarity search is a matrix–vector product, and "angle" and "distance"
become interchangeable:

$$\lVert u - v\rVert_2^{2} \;=\; 2 - 2\cos(u,v)$$

That identity is not decoration — it is why H-Nav's replica of the benchmark's
index can rank by dot product while the benchmark's FAISS index ranks by squared
L2 distance and still produce **identical orderings** (verified: Kendall τ =
1.0000, max score error ≤ 4.5×10⁻⁵). It is also why the precision incident in
`HNAV_FINAL_REPORT.md` §3.2 was so damaging: the equivalence holds **only for
exact unit vectors**, and bf16 rounding pushed norms to 0.998–1.002, which is
enough to swap neighbours whose scores differ by less than that error.

### 11.2 Signal 1 — `sim_max`: nearest-neighbour similarity

For a candidate fact $v$ against a bank of stored facts $B \in \mathbb{R}^{m \times d}$:

$$\text{sim\_max}(v) = \max_{i} \; (B v)_i , \qquad
\text{argmax\_id} = \arg\max_{i} \; (B v)_i$$

One matrix–vector product. This answers "what is the most similar thing already
in memory, and which one is it?"

### 11.3 ABTT whitening — removing the directions that carry no information

**The problem.** Sentence embeddings are not isotropic. A few directions are
shared by almost every sentence in a corpus (they encode things like "this is an
English declarative sentence"), and because they are common to everything they
carry *no discriminative information* — yet they dominate the dot product. The
practical symptom is that unrelated sentences all sit at cosine ≈ 0.6–0.8 instead
of near 0.

**The fix** (Mu & Viswanath's *all-but-the-top*), implemented in
`ABTTWhitening`:

1. **Estimate the mean** over the store's $N$ vectors:
   $$\mu = \frac{1}{N}\sum_{i=1}^{N} v_i$$
2. **Center**: $\tilde{M} = M - \mathbf{1}\mu^{\top}$
3. **Find the dominant directions** by economy SVD:
   $$\tilde{M} = U\Sigma V^{\top}, \qquad C = V^{\top}_{1:D} \;\in\; \mathbb{R}^{D \times d}, \quad D = 3$$
   The top $D$ right singular vectors *are* the principal directions of the
   centered cloud.
4. **Project them out and renormalize**, for any vector $v$:
   $$v' = (v - \mu) - \big((v-\mu)C^{\top}\big)C, \qquad
     \hat{v} = \frac{v'}{\lVert v'\rVert_2}$$

The middle expression is the projection onto the orthogonal complement of
$\operatorname{span}(C)$: $(I - C^{\top}C)(v-\mu)$.

**The refusal rule — a deliberate design constraint.** $\mu$ and $C$ are
*estimates*. On a small store they are estimates of noise. So `fit()`
**refuses** below `min_fit_n = 200` records and the module falls back to raw
cosine, **recording which path it took** (`whitened`, `whitening_refused`) so the
fallback rate is reportable instead of invisible. On `sh_6k` (455 facts) it fits;
on early cross-episode contexts (<20 chunks) it does not.

**Measured effect** (same-key separation AUC, raw → whitened):
sh_6k **0.936 → 0.955**, sh_32k **0.988 → 0.991**. Real but modest — which is
itself informative: in this arena the raw geometry was already strong enough
that whitening was not load-bearing.

### 11.4 Signal 2 — the QR residual: novelty as an orthogonal projection

**The question:** how much of a new fact is *not already explained* by what is
stored?

Formally: let $B$ be the bank. The stored facts span a subspace
$\mathcal{S} = \operatorname{span}(B) \subseteq \mathbb{R}^{d}$. Decompose $v$
into the part inside that subspace and the part orthogonal to it:

$$v = P_{\mathcal{S}}\,v \;+\; (I - P_{\mathcal{S}})\,v$$

The **residual** is the norm of the second term:

$$r(v) = \big\lVert v - P_{\mathcal{S}}\,v \big\rVert_2$$

To compute $P_{\mathcal{S}}$ we need an **orthonormal basis** for $\mathcal{S}$,
which is exactly what QR decomposition provides:

$$B^{\top} = QR \quad\Longrightarrow\quad
  Q^{\top}Q = I, \quad \operatorname{span}(Q) = \operatorname{span}(B^{\top})$$
$$P_{\mathcal{S}} = QQ^{\top}, \qquad
  r(v) = \big\lVert v - QQ^{\top}v \big\rVert_2$$

Because $v$ is a unit vector and $P$ is an orthogonal projection, Pythagoras
bounds the result neatly:

$$r(v) \in [0, 1], \qquad r^2 = 1 - \lVert P_{\mathcal{S}}v\rVert^2$$

- $r = 0$ → the fact is a **linear combination of things already stored**:
  nothing new.
- $r = 1$ → the fact is **orthogonal to the entire store**: entirely new.

**An honesty detail that is stated in the return value, not buried.** QR on a
bank of 18,332 vectors is expensive, so the basis is subsampled to
`max_basis = 512` columns (fixed seed). A *smaller* span can only leave *more*
of $v$ unexplained, so the reported residual is a rigorous **upper bound** on
true novelty — and the flag `qr_basis_subsampled` travels with the number so no
reader mistakes a bound for a point estimate.

Cost: $O(d\,m^2)$ for the decomposition, $O(dm)$ per query afterwards.

### 11.5 The grouping test — leave-one-out span residual

This is the screen that decides whether a set of similar-looking facts is really
*one slot restated* rather than several distinct facts that happen to look alike.

For a candidate set $\{v_1,\dots,v_n\}$, compute for each member its residual
against **all the others**:

$$r_i = \big\lVert v_i - P_{\operatorname{span}(V \setminus v_i)}\, v_i \big\rVert_2$$

A tentative group survives only if

$$\min_i r_i \;<\; r_{\min}$$

In words: **at least one member must be almost entirely reconstructible from the
rest.** That is the signature of a restatement of the same slot — which is what a
supersession is — rather than a cluster of merely-similar facts.

**A closed form worth knowing, because it links the two thresholds.** For a
two-member group whose other candidates are unrelated (near-orthogonal), the
projection of the unit vector $v_a$ onto $\operatorname{span}\{v_b\}$ has norm
$|\cos|$, so:

$$r_a = \sqrt{1 - \cos^2(v_a, v_b)}$$

Invert it and a residual threshold *is* a cosine threshold:

$$r_{\min} \;\Longleftrightarrow\; \cos > \sqrt{1 - r_{\min}^{2}}$$

| $r_{\min}$ | implied pair cosine | where it comes from |
|---|---|---|
| 0.1924 | **> 0.9813** | the frozen Stage-0 calibration value |
| **0.44** | **> 0.8980** | the Stage-1 operating point |

This is why the operating point pairs `r_min = 0.44` with `cos_pair = 0.90`:
$\sqrt{1-0.44^2} = 0.898$, so the two screens **agree** instead of one silently
overriding the other. The original 0.1924 was far stricter than the cosine screen
it accompanied — it implied ≈ 0.981, well above the median true-conflict
similarity of 0.964, which is precisely why the Stage-0 defaults were
precision-first to the point of rarely firing.

### 11.6 From pairwise scores to groups — a graph problem

Once pairs are scored, grouping is pure graph theory:

- **Nodes** = candidate facts.
- **Edges** = pairs passing the cosine screen.
- **Tentative groups** = connected components of size ≥ 2.
- Each tentative group is then tested by §11.5, and each surviving edge is
  verified by the subject screen and bidirectional NLI (Steps 3–4).
- **Final groups** = connected components of the **verified** edges only.

Components are computed deterministically (members ascending), so the same input
always produces the same grouping — a requirement for the byte-neutrality
guarantees in §5 Step 6.

### 11.7 The adaptive threshold `tau_t`

`sim_max` drifts upward with store size: in a bigger store, the nearest
neighbour of *anything* is closer. A fixed threshold therefore silently tightens
as a run proceeds. So the write-side threshold is adaptive:

$$\tau_t = \operatorname{clip}\!\Big(\bar{s}_t + z\,\sigma_t,\; \text{lo},\, \text{hi}\Big),
\qquad z = 1,\; \text{lo} = 0.50,\; \text{hi} = 0.99$$

with $\bar{s}_t$ and $\sigma_t$ the running mean and standard deviation of the
`sim_max` values observed *so far*, maintained from streaming sums
($\sum s$, $\sum s^2$). Below `min_n = 50` observations there is nothing to adapt
to and a fixed `init_tau = 0.85` is used.

**The subtlety that makes it honest:** it is stateful and order-dependent *by
design*. It models what an online policy would have known at time $t$, so
`update()` must be called in observation order and never with a future
candidate's value. This is the same write-time-versus-read-time visibility rule
that runs through the whole codebase.

### 11.8 The retrieval-side family — entropy, margin, and a threshold that could never fire

These signals describe the *ranking* rather than individual facts. They are
**inert in the shipped configuration** (§6.3), but they matter because one of
them produced a genuine methodological finding.

Given ranked scores $s_1 \ge s_2 \ge \dots \ge s_n$ over the top $m = 50$:

$$\text{margin} = s_1 - s_2, \qquad \text{nmargin} = \frac{s_1 - s_2}{|s_1|}$$

For the entropy family the scores are first **z-scored**, then softmaxed, then
Shannon entropy is taken in nats:

$$z_i = \frac{s_i - \bar{s}}{\sigma}, \qquad
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
H_z = -\sum_i p_i \log p_i$$

with $\text{eff\_size} = e^{H_z}$ read as "how many neighbours are effectively
competing." A variant $H_{vn}$ (von Neumann entropy of the top-$m$ Gram matrix)
is computed when vectors are supplied.

Why z-score first? Because raw cosine scores have an arbitrary scale, and a
softmax over them measures the *scale* as much as the *shape*. A parallel raw
version $H_{raw}$ is logged **but is forbidden from ever feeding a decision** —
enforced by an AST scan in the test suite, not by convention.

**The finding.** Entropy over $n$ items is bounded by $\log n$. At `sh_6k` the
store is **2 chunks**, so:

$$H_z \le \log 2 = 0.693$$

Worse, with $n = 2$ the z-scored vector is *always* $[+1, -1]$ regardless of the
underlying scores, so the entropy collapses to a **constant**:

$$H_z \equiv 0.36533385508\ldots \quad \text{for every question at } n=2$$

The frozen threshold inherited from Stage 0 was $H_z > 1.9569$ — **above the
mathematical ceiling**. It could never fire on `sh_6k`, under that threshold, a
re-fitted one, or even `sh_6k`'s own 75th percentile. And because it had been fit
on *pooled* calibration data, it landed on `sh_32k`'s median instead — making it
effectively a **store-size detector rather than an ambiguity detector**. That is
the concrete reason the shipped operating point disables this screen, and the
reason the codebase now requires per-subset reporting.

### 11.9 The cheap paths that run first

Two O(1) checks precede all vector arithmetic:

- **Exact duplicate:** MD5 of the exact string. If the text has been seen
  verbatim, no geometry is needed.
- **Verbatim value extraction:** a deliberately crude regex for capitalized
  phrases and numbers, used to ask "does this candidate introduce a value the
  store has never seen?" It is tuned for **recall, not precision**, because the
  error costs are asymmetric — a false positive costs a suppression that would
  have happened anyway; a false negative destroys an answer.

### 11.10 Summary of the mathematics

| signal | mathematics | cost | what it answers |
|---|---|---|---|
| `sim_max` | dot product on unit vectors | $O(md)$ | is anything in memory this similar? |
| ABTT whitening | mean removal + top-3 SVD projection + renormalize | $O(Nd^2)$ once | remove directions that carry no information |
| QR residual | orthogonal projection via QR, $\lVert v - QQ^{\top}v\rVert$ | $O(dm^2)$ | how much of this is genuinely new? |
| LOO span residual | the same, member-vs-rest | $O(n \cdot dn^2)$ per group | is this one slot restated, or several facts? |
| grouping | connected components | $O(V+E)$ | which facts belong together? |
| `tau_t` | running mean + $z\sigma$, clipped | $O(1)$ streaming | adapt to a growing store |
| margin / `nmargin` | $s_1-s_2$, normalized | $O(1)$ | is the top result decisive? |
| $H_z$ / eff\_size | softmax over z-scores → Shannon entropy | $O(m)$ | how many neighbours compete? |

**The overall design principle:** cheap and exact first (hash), then linear
algebra (cosine, projections), then — only for the few pairs that survive — the
expensive neural check (NLI). Nothing in the geometric stage is learned or
tuned during a run; every threshold is fitted once on the calibration split and
frozen.
