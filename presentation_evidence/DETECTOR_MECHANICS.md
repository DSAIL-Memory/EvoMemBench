# How the detector actually works

*Parsing → geometric filtering → NLI verification → the suppression decision;
and what "held-out accuracy" means.*

This document answers five questions in order:

1. **How does the parser work, and how does it decide that two facts conflict?**
2. **What happens after a conflict candidate is found — the geometric filter and
   the NLI stage, written out mathematically.**
3. **How is the final decision made?**
4. **How was the detector's accuracy at finding conflicts actually verified —
   against what reference, and which parts of that check are independent of the
   detector?** (§4.5)
5. **What exactly is "held-out accuracy", and what is it for?**

Every number cited names the file it came from. Where a number exists only in
prose with no artifact behind it, that is said explicitly rather than glossed
over.

---

## 0. One distinction that has to come first

The word "conflict" is used in two completely different places in this
repository, and confusing them is the single easiest way to misread the results.

| | **Offline labelling** | **Online detection** |
|---|---|---|
| Where | `hnav/labeling/` (offline tier) | `hnav/core/read_gate.py` + `hnav/adapters/mab_adapter.py` |
| Input | the *entire* fact context, plus the questions and gold answers | only the retrieved page, no questions, no answers |
| Method | template parse → group by `(relation, subject)` → "conflicted = more than one distinct object" | cosine screen → identity screen → bidirectional NLI |
| Purpose | build the ground truth *against which the detector is scored* | make the actual runtime decision |
| May see gold? | **yes** — that is what the offline tier is for | **never** (enforced by the AST scan in `hnav/tests/test_leakage_audit.py`) |

The parser appears in **both** columns, but doing **different jobs**:

- Offline it is the **conflict oracle**: it decides which facts truly conflict.
- Online it is only an **identity screen**: it supplies the `(relation, subject)`
  key so the gate can check "are these two sentences even talking about the same
  slot?" — and nothing more. It never gets to say "these conflict." That verdict
  is reserved for the NLI.

`hnav/adapters/mab_adapter.py:53` imports it for the online role:

```python
from hnav.labeling.conflict_analysis import parse as parse_fact
```

This import is legal because `parse` reads **only the fact text**. It never
touches `questions` or `answers`. It is the one function that crosses the tier
boundary, and it crosses in the safe direction.

---

## 1. The parser

### 1.1 Step one — split the context into numbered facts

The benchmark's context is a numbered list:

```
0. Thomas Kyd was born in the city of London.
1. The chairperson of Sony Corporation is Kenichiro Yoshida.
...
306. Thomas Kyd was born in the city of Leeds.
```

Offline (`hnav/labeling/conflict_analysis.py:78`) the split is one regex:

```python
facts = re.findall(r"^\s*(\d+)\.\s+(.*)$", item["context"], re.M)
```

Online it is not that simple, and this is a real trap the repository documents
loudly. The benchmark does not hand H-Nav the raw context — it hands it the
output of `chunk_text_into_sentences`, which **joins sentences with spaces**. A
line-anchored regex (`^\s*(\d+)\.`) therefore has nothing to anchor to and
matches **zero** facts. So the adapter carries two patterns
(`hnav/adapters/mab_adapter.py:62-71`):

```python
FACT_RE        = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)                                   # line-anchored
FACT_RE_INLINE = re.compile(r"(?:(?<=\s)|(?<=\A))(\d+)\.\s+(.+?)(?=\s+\d+\.\s|\Z)", re.S)   # inline fallback
```

`explode_facts` tries the line-anchored form first, then falls back to the inline
form when it finds more facts. `hnav/tests/test_chunking_and_facts.py` asserts
that the line-anchored form **fails** on a real chunk, so the fallback cannot
later be deleted as apparently-redundant.

The serial number is preserved exactly. It is load-bearing: the prompt states its
precedence rule in terms of serial order, so renumbering would change the task.

### 1.2 Step two — parse a fact into `(relation, subject, object)`

This is a **template-induction** parser, not an NLP model. The dataset is
synthetic and generated from a fixed set of relation templates, so the templates
can simply be enumerated. There are two families
(`hnav/labeling/conflict_analysis.py:16-52`):

**Suffix family** — `{subject} MID {object}.` — 21 templates:

```
" is affiliated with the religion of ", " is associated with the sport of ",
" was created in the country of ", " was written in the language of ",
" is located in the continent of ", " was born in the city of ",
" was founded in the city of ", " speaks the language of ",
" died in the city of ", " worked in the city of ", " plays the position of ",
" works in the field of ", " is a citizen of ", " was performed by ",
" was developed by ", " was founded by ", " was created by ",
" is employed by ", " is married to ", " is famous for ", "'s child is ",
```

**Prefix family** — `PRE {subject} MID {object}.` — 18 templates:

```
("The name of the current head of the ", " government is "),
("The headquarters of ",                 " is located in the city of "),
("The chief executive officer of ",      " is "),
("The capital of ",                      " is "),
...
```

The algorithm (`conflict_analysis.py:53-68`):

```
strip the trailing period
for each PREFIX template (pre, mid), longest first:
    if the text starts with `pre`:
        rest = text[len(pre):]
        i = rest.find(mid)
        if i > 0:  return ( pre+"|"+mid , rest[:i] , rest[i+len(mid):] )
for each SUFFIX template mid, longest first:
    i = text.find(mid)
    if i > 0:  return ( "|"+mid , text[:i] , text[i+len(mid):] )
return None
```

Three details that matter:

- **Longest-first ordering** (`SUFFIX_RELS.sort(key=len, reverse=True)`,
  `PREFIX_RELS.sort(key=lambda x: -(len(x[0])+len(x[1])))`). Without it, the
  shorter `" is "` inside `"The chief executive officer of X is Y"` would match
  before the correct longer template and split the sentence in the wrong place.
- **Prefix family is tried before suffix family**, for the same reason.
- **`i > 0`, not `i >= 0`.** A match at position 0 would mean an empty subject,
  which is not a fact.

**Measured coverage** (`presentation_evidence/data/item08_parser.json`, from the
M1 run):

| subset | parse coverage |
|---|---|
| sh_6k | 99.56% |
| sh_32k | 99.65% |
| sh_64k | 99.63% |
| sh_262k | 99.47% |

Worked example, verbatim from that artifact:

```
input : "Nobuhiro Watsuki is famous for Rurouni Kenshin."
output: relation_key = "| is famous for "
        subject      = "Nobuhiro Watsuki"
        object       = "Rurouni Kenshin"
```

> **Note.** `CLAUDE.md` forbids rewriting this function: *"Do not rewrite
> `hnav/labeling/conflict_analysis.py::parse`. Validated at 99.5%+ coverage;
> import it."* Every consumer — the offline labeller, the adapter, the
> calibration harness — imports the same object, so the online identity screen
> and the offline ground truth cannot drift apart.

### 1.3 Step three — grouping, and the offline definition of "conflict"

Offline (`conflict_analysis.py:80-88`):

```python
groups = defaultdict(list)                      # (relation, subject) -> [(serial, text, object)]
for num, txt, rel, subj, obj in parsed:
    groups[(rel, subj)].append((num, txt, obj))

conflicts = {k: v for k, v in groups.items() if len({o for _, _, o in v}) > 1}
```

So the offline definition is exactly:

> A key `(relation, subject)` is **conflicted** iff the facts sharing that key
> carry **more than one distinct object string**.

Note what this deliberately excludes: a key whose facts all repeat the *same*
object is **not** conflicted, however many times it is restated. Duplication is
not conflict. Only disagreement is.

Under this definition (`python3 hnav/labeling/conflict_analysis.py`, the T0
reproduction):

```
sh_262k:  11,037 keys   7,197 conflicted   (65.2%)
```

and `hnav/labeling/gold_rule.py` establishes the answer rule: 77% of sh_262k
questions target a conflicted key, and 73 of 77 sampled have gold = the
highest-serial value. That is the "latest wins" rule the whole system is built
on — and its 4/77 exceptions are why the system can never claim to be perfectly
aligned with gold (see §4.3).

### 1.4 What the parser is *not* allowed to do online

Online, `fact_key` (`mab_adapter.py:88-96`) returns only the key and the object,
and the gate uses only the **key**:

```python
@staticmethod
def same_key_pair(a: MemoryRecord, b: MemoryRecord) -> bool:
    ka = a.metadata.get("key")
    return ka is not None and ka == b.metadata.get("key")
```

The online path never runs the "more than one distinct object" test. It could —
the objects are right there — and it was a deliberate design choice not to,
because that test only works on this synthetic template family. The whole point
of routing the disagreement verdict through NLI is that NLI transfers to
substrates where no templates and no serial numbers exist.

Note also: **`ka is not None`**. A fact that fails to parse is **rejected**, not
waved through. The docstring gives the reason — the NLI has been *measured* to
rubber-stamp same-template/different-subject pairs, so absence of identity
evidence must not default to trust. With 99.5%+ coverage, the recall cost of
this fail-closed choice is bounded and is reported in the calibration artifact.

---

## 2. Stage 1 — geometric filtering

Entry point: `ReadGate.decide` (`hnav/core/read_gate.py:556-635`). It mutates
nothing — not the records, not the ranking, not the store.

### 2.0 Before anything: the candidate pool

`select_pool` (`mab_adapter.py:321-344`) builds the pool from the retrieved page:

- drop facts with no vector,
- if more than `cap = top_m = 50` remain and a query vector exists, keep the 50
  most query-similar,
- restore original `(chunk-rank, serial)` order so the pool is deterministic.

The cap exists because the next stage is **quadratic**: a 64k page holds ~2,900
facts, and the leave-one-out residual is O(n) QR decompositions over an n×d
matrix. It is also the dominant reason the held-out effect is smaller than the
calibration effect (§5.4).

### 2.1 The cosine screen — building edges

All candidates are unit-normalised embeddings. Stack them into `M ∈ ℝ^{n×d}` and
form the full Gram matrix in one shot:

```
S = M Mᵀ          S_ij = ⟨v_i, v_j⟩ = cos(v_i, v_j)
```

For every unordered pair `(i, j)`, keep an edge iff

```
    S_ij  ≥  cos_pair                    (frozen value: 0.90)
```

**Why a threshold and not top-k:** conflict group sizes are not fixed. A subject
may be superseded once or twenty times. A cosine threshold adapts; a fixed k
does not.

**Why 0.90:** it is the loosest value on the calibration grid
(`stage0_results/stage1_operating_point.json → provenance.grid.cos_pair =
[0.9, 0.92, 0.94]`), selected by the rule in §4.2. The prior Stage-0 default was
`COS_PAIR_CAL = 0.92` (`read_gate.py:150`), the mean of M1b's best-F1 taus on the
calibration split (0.91 and 0.93 — see `item07_summary.json`). Stage 1 loosened
it because precision was already saturated by the two later screens.

Cost: `S` is one matrix multiply; with the pool capped at 50 that is 1,225 pairs.

### 2.2 Tentative groups — connected components

Edges define a graph. Its connected components of size ≥ 2 are the *tentative*
groups (`_components`, `read_gate.py:481-499`), computed by union-find with path
compression and union-by-min-root, so the output is deterministic and members
come out ascending.

Transitivity is doing real work here. If A–B and B–C both clear 0.90 but A–C is
0.88, all three still land in one group. That is correct behaviour for a
supersession chain, where the first and last values of a long chain can be quite
far apart in embedding space.

### 2.3 The leave-one-out span residual — the mathematics

This is the part worth writing out properly.

**The question it answers.** Pairwise cosine asks "is `v_i` close to *one* other
vector?" The span residual asks a stronger question: **"is `v_i` already
explained by everything else in the pool taken together?"** A fact that is a
genuine restatement-with-a-changed-value of material already present should be
almost entirely reconstructible from the rest. A fact introducing genuinely new
content should not be.

**Setup.** For candidate `i`, let

```
B_i  =  M with row i deleted        B_i ∈ ℝ^{(n−1)×d}
```

Let `V_i = span(rows of B_i) ⊆ ℝ^d`, and let `P_i` be the orthogonal projector
onto `V_i`. The residual is

```
    r_i  =  ‖ v_i − P_i v_i ‖₂
```

**Computing `P_i` by QR** (`qr_residual`, `hnav/core/geometry.py:159-179`):

```python
q, _ = np.linalg.qr(b.T)          # b.T is d×(n−1); q has orthonormal columns
recon = (v @ q) @ q.T             # = P v
return np.linalg.norm(v - recon, axis=1), subsampled
```

`np.linalg.qr` on `B_iᵀ ∈ ℝ^{d×(n−1)}` yields `Q ∈ ℝ^{d×k}` with `QᵀQ = I_k`,
whose columns span the column space of `B_iᵀ` — which is exactly the *row* space
of `B_i`, i.e. `V_i`. Then

```
    P_i  =  Q Qᵀ
```

is the orthogonal projector onto `V_i` (idempotent: `P² = QQᵀQQᵀ = QQᵀ = P`;
symmetric: `Pᵀ = P`). QR is used instead of forming a pseudo-inverse because it
is numerically stable and needs no matrix inversion.

**The closed form.** Because `P_i` is an *orthogonal* projector, `v_i − P_i v_i ⊥
P_i v_i`, so Pythagoras applies:

```
    ‖v_i‖²  =  ‖P_i v_i‖²  +  ‖v_i − P_i v_i‖²
```

All vectors are unit-normalised, so `‖v_i‖ = 1` and

```
    r_i  =  √( 1 − ‖P_i v_i‖² )
```

Define the **in-span cosine** — the cosine of the angle between `v_i` and the
subspace `V_i`:

```
    c_i  :=  ‖P_i v_i‖  =  max_{u ∈ V_i, ‖u‖=1} ⟨v_i, u⟩
```

Then the identity is simply

```
    r_i  =  √( 1 − c_i² )              r ∈ [0, 1]
```

with the interpretation:

| `r_i` | meaning |
|---|---|
| 0 | `v_i` lies **exactly inside** the span of the others — fully redundant |
| 1 | `v_i` is **orthogonal** to everything else — fully novel |

(`hnav/tests/test_geometry.py::test_qr_residual_is_zero_inside_the_span_and_one_outside`
asserts exactly these two endpoints against a closed-form construction.)

**The two-member special case.** Suppose the group is just `{i, j}` and every
other pool member is roughly orthogonal to `v_i`. Then `P_i v_i ≈ ⟨v_i, v_j⟩ v_j`,
so `c_i ≈ |cos(v_i, v_j)|` and

```
    r_i  ≈  √( 1 − cos²(v_i, v_j) )  =  sin( angle between the pair )
```

This is the bridge between the two thresholds, and it is the reason the frozen
operating point looks the way it does.

**The group rule** (`read_gate.py:617-621`):

```python
loo = {}
for i in sorted({i for g in tentative for i in g}):
    r, _ = qr_residual(mat[i], np.delete(mat, i, axis=0))
    loo[i] = float(r[0])

geo_groups = [g for g in tentative if min(loo[i] for i in g) < thr.r_min]
```

A group survives iff **at least one** member is well-explained by the rest:

```
    min_{i ∈ g}  r_i   <   r_min
```

`min`, not `max` or `mean` — a supersession group only needs *one* member to be
redundant given the others for the group to be worth verifying. Requiring all of
them would reject exactly the chains we are hunting.

Note the residual is taken against the whole pool minus `i`, not just the group
minus `i`. This is stricter: a fact that another *non-group* fact already
explains does not get credit for novelty.

**The frozen value, and an honest disclosure.** The operating point uses
`r_min = 0.44`, labelled `"loose"`
(`stage0_results/stage1_operating_point.json → thresholds.r_min`,
`r_min_label: "loose"`). Its definition, verbatim from
`hnav/stage1/calibrate_read_policy.py:112-113`:

```python
R_LOOSE = 0.44          # sqrt(1-0.90^2)=0.436: pass-through for pairs
                        # the loosest cos screen admits
```

Work the arithmetic in both directions:

```
r < 0.44        ⟺   c_i > √(1 − 0.44²)  =  √0.8064  =  0.8980
cos ≥ 0.90      ⟹   r ≈ √(1 − 0.81)     =  0.4359   <  0.44   ✓
```

**Therefore: at the frozen operating point the residual screen rejects nothing
that the cosine screen already admitted.** The two screens meet at essentially
the same geometric point (`presentation_evidence/data/item09_thresholds.json`
records `sqrt(1-r_min^2) = 0.8980` and calls this out). This is declared, not
hidden: the residual machinery is present, tested, and *inert at this setting*.
The tighter alternative on the grid, `R_MIN_CAL = 0.19236616622633881`
(`read_gate.py:113`), implies pair cosine ≳ 0.981 and was rejected by the
selection rule because it cost recall while precision was already 1.000.

For the presentation, the correct statement is: **the geometric stage is
carried by the cosine screen; the residual screen is a tested safety rail that
is not binding at this operating point.**

### 2.4 The ambiguity precondition — present, and switched off

There is a fourth geometric gate that is *disabled* in the shipped configuration,
and this too is declared rather than quietly dropped.

`_ambiguity` (`read_gate.py:521-552`) can require that the retrieval ranking was
*ambiguous* before any conflict machinery runs at all:

```
    nmargin  <  0.004764391389658354        (normalised top-1/top-2 score margin)
    H_z      >  1.9569327964981853          (z-scored retrieval entropy)
```

Both are frozen Stage-0 values from `m3_headroom.json`, fit on sh_6k + sh_32k
only. `ambiguity_mode` selects `all` (both must fire), `any` (either), or `none`
(skip the precondition entirely). The operating point uses **`"none"`**.

The artifact states the reason in full
(`stage0_results/stage1_operating_point.json → ambiguity_note`):

> `ambiguity_mode='none'` disables the frozen Stage-0 nmargin/H_z precondition.
> Declared, not incidental: those two signals are the only gate input computed
> from CHUNK embeddings, which were truncated at 512 of ~4096 tokens (T12) and
> are not yet re-fit; they are also the dominant recall bottleneck (question-level
> recall collapses from 0.97 to 0.16 with the screen on); and the volume-limiting
> role they played is now carried by the identity screen plus bidirectional NLI
> at the measured precision recorded here.

A 0.97 → 0.16 recall collapse is not a tuning nuisance; it is the screen
answering a different question than the one being asked. It stays in the code,
switched off, with a written instruction that a campaign run after the
embeddings are re-fit must revisit it.

One hard rule is enforced right here (`read_gate.py:532-534`): the raw-score
entropy `H_raw` may never reach a policy. Signals arrive only through
`RetrievalSignalSet.for_policy()`, and any name in `POLICY_FORBIDDEN` that
appears raises immediately. `hnav/tests/test_no_raw_entropy_in_policy.py`
AST-scans this module to keep it that way.

### 2.5 The identity screen — between geometry and NLI

Placed inside the cosine loop, *after* the cosine test and *before* any NLI is
spent (`read_gate.py:583-590`):

```python
for i, j in all_pairs:
    if thr.cos_pair is not None and sims[i, j] < thr.cos_pair:
        continue
    if self.pair_filter is not None and not self.pair_filter(recs[i], recs[j]):
        dec.n_pairs_filter_rejected += 1
        continue
    edges.append((i, j))
```

`pair_filter` is `MABAdapter.same_key_pair` — parsed `(relation, subject)`
equality, §1.4. The ordering is deliberate on two counts: NLI is by far the most
expensive stage, and — as §3.3 shows with measurements — it is precisely the
stage that cannot be trusted to enforce subject identity on its own.

Architecturally this is the seam that keeps `hnav/core/` benchmark-agnostic: the
core sees an opaque `Callable[[MemoryRecord, MemoryRecord], bool]`. All knowledge
of relation templates lives in the adapter. `pair_filter=None` reproduces the
pre-mitigation behaviour exactly, which is what made the measurement in §3.3
possible.

---

## 3. Stage 2 — bidirectional NLI

### 3.1 What NLI is here

**Natural Language Inference**: given an ordered pair (premise `p`, hypothesis
`h`), classify their logical relation as one of

- **entailment** — `p` being true makes `h` true,
- **neutral** — `p` says nothing about `h`,
- **contradiction** — `p` and `h` cannot both be true.

The model is a **cross-encoder**: `cross-encoder/nli-deberta-v3-large`
(`read_gate.py:154`). Both sentences enter the transformer *together* in one
sequence, so attention runs across them. That is what makes it able to judge a
relation between two texts — unlike a bi-encoder, which embeds each separately
and can only compare the resulting points.

Given the pair, the head emits three logits, softmaxed to `(e, n, c)` with
`e + n + c = 1`.

**The label order is verified, never assumed** (`read_gate.py:340-359`): the
constructor reads the checkpoint's own `id2label`, builds an index, and raises if
the three labels are not exactly `{entailment, neutral, contradiction}`. A
differently-ordered NLI head would otherwise silently swap entailment and
contradiction — and every downstream decision would invert while every test still
passed.

### 3.2 The bidirectional criterion

For every within-group pair `(i, j)` the gate scores **both orderings**
(`read_gate.py:623-631`):

```
    c_ij  =  contradiction( premise = text_i , hypothesis = text_j )
    c_ji  =  contradiction( premise = text_j , hypothesis = text_i )
```

The pair is a **verified conflict** iff

```
    c_ij  ≥  τ_nli    AND    c_ji  ≥  τ_nli          τ_nli = 0.90
```

equivalently

```
    min( c_ij , c_ji )  ≥  τ_nli
```

Both directions are pushed through in one batch — the code builds a flat
`directed` list of `2k` ordered pairs and reads them back as
`nli_scores[2k], nli_scores[2k+1]`.

**Why both directions.** NLI is *not* a symmetric relation. Entailment
especially is directional: "Kyd was born in London in 1558" entails "Kyd was born
in London", but not conversely. Requiring symmetry is therefore a filter on the
*kind* of relation, and it is chosen to match the phenomenon being hunted. The
module docstring (`read_gate.py:22-30`) puts it exactly:

> One-directional contradiction — e.g. entailment one way — is rejected:
> supersession in this arena is symmetric disagreement about the same slot, not
> refinement.

Supersession — "the capital is Paris" then later "the capital is Lyon" — is a
*mutual* incompatibility. Refinement, specialisation and elaboration are not.
The bidirectional requirement is how the gate separates *replacement* from
*addition of detail*, which is the distinction that decides whether deleting the
older sentence is safe.

`τ_nli = 0.90` is the middle rung of the grid `[0.5, 0.9, 0.99]`. The Stage-0
default was `NLI_CONTRA_DEFAULT = 0.5` (`read_gate.py:152`) — plain probability
majority. Stage 1 raised it.

### 3.3 Why NLI is useful — and why it is not sufficient

**What NLI supplies that geometry cannot.**

Cosine similarity measures *topical proximity*. It has no notion of truth,
agreement or disagreement. Consider three pairs, all sharing the same subject and
relation:

| pair | cosine | what the system should do |
|---|---|---|
| "X is a citizen of France." / "X is a citizen of France." | ~1.00 | nothing — duplicate, no conflict |
| "X is a citizen of France." / "X is a citizen of Germany." | very high | **suppress the older one** |
| "X is a citizen of France." / "X was born in Lyon." | high | nothing — compatible facts |

Cosine **ranks the harmless duplicate as the most similar pair of the three.**
Geometry alone would act most aggressively on exactly the case where acting is
pointless, and it has no principled way to separate rows 2 and 3. The
disagreement of *values* is a truth-functional property of the sentences, and
embedding proximity simply does not encode it.

NLI does. That is its job, and it is the only stage in the pipeline that can do
it.

**The measured evidence that it does the job.** At the frozen operating point on
the 200-question calibration split
(`stage0_results/stage1_operating_point.json → metrics`):

| quantity | value | reading |
|---|---|---|
| `n_suppressed` | 2,673 | facts the detector deleted |
| `n_suppressed_superseded` | 2,673 | independently confirmed genuinely superseded |
| **`n_suppressed_same_value`** | **0** | **not one duplicate-value deletion** |
| `n_suppressed_harmful` | 0 | none carried a value still needed |
| `fp` | 0 | zero false-positive pairs |
| `pair_precision` | 1.000 | |
| `fact_precision` | 1.000 | |

Zero out of 2,673 is the row-1 case above never once firing. That is what the NLI
stage bought.

**And now the counter-evidence, which is just as important.** The Faz A audit
ran the identical grid with the identity screen *off*
(`presentation_evidence/data/item10_summary.json`):

| configuration | cells | false-verification rate |
|---|---|---|
| `pair_filter = false` | 81 | **min 31.50%, max 94.09%** |
| `pair_filter = true` | 81 | **min 0.000, max 0.000** |

where the false-verification rate is
`(n_fv_diff_key + n_fv_same_object) / n_verified` per cell.

So across every one of 81 threshold combinations, **bidirectional NLI on its own
falsely verified between a third and 94% of the pairs it accepted.** The dominant
failure class is same-template/different-subject:

```
"Thomas Kyd was born in the city of London."
"Christopher Marlowe was born in the city of London."
→ contradiction 0.99949 and 0.99983 in the two directions
```

Two people can perfectly well both be born in London. The cross-encoder, faced
with two sentences of near-identical surface form differing in one noun phrase,
reports near-certain contradiction in both directions. It is pattern-matching
the *shape* of a contradiction, not evaluating one.

> **Provenance flag, stated because the repository states it.**
> `item10_summary.json` records:
> *"kyd_marlowe_provenance: TEZ_BULGULARI.md lines 265-267 ONLY; scores 0.99949 /
> 0.99983 in the two directions; no JSON artifact behind it."*
> The **0.99949 / 0.99983 pair is an illustrative example with no committed JSON
> behind it** and should be presented as an illustration. The **31.50%–94.09%
> range and the 0.000 collapse are backed by the 162-cell artifact** and can be
> quoted as measurements. (`HNAV_FINAL_REPORT.md:424` renders the range as
> "33–93%"; the artifact's exact figures are 31.50% and 94.09%.)

**Therefore the three screens are complementary, and none is redundant:**

| screen | what it can decide | what it is blind to |
|---|---|---|
| cosine | "are these two texts about similar material?" — and cheaply reduces O(n²) pairs to a handful | agreement vs disagreement; subject identity |
| identity (parsed key) | "is this the *same slot* of the *same entity*?" | whether the values actually differ |
| bidirectional NLI | "do these two values *mutually exclude* each other?" | subject identity (measured to fail at 31–94%) |

The pipeline is ordered cheapest-first and each stage covers the next one's blind
spot. Remove any one and the measured precision of 1.000 does not survive.

---

## 4. The final decision

### 4.1 From verified pairs to a suppression list

```
verified edges  →  connected components (union-find, size ≥ 2)  →  ConflictGroup
```

Note this is a **second** components pass, over *verified* edges only
(`read_gate.py:634-636`). Cosine-only edges that failed the identity screen or
the NLI are gone by now, so a tentative group can shatter into several final
groups, or vanish entirely.

For each final group, `_finalize` (`read_gate.py:639-657`) names the survivor:

```python
key_fn = latest_key or (lambda r: r.version)
vals = [key_fn(recs[i]) for i in members]

if any(v is None for v in vals):
    note = "recency key missing for at least one member"        # latest_id = None
else:
    mx = max(vals)
    if vals.count(mx) > 1:
        note = "recency key tied at the maximum"                # latest_id = None
    else:
        latest_id = recs[members[vals.index(mx)]].id

stale = [m for m in member_ids if m != latest_id] if latest_id else []
```

In words:

```
    survivor  =  argmax over group members of the recency key
    stale     =  every other member of the group
```

Here the recency key is the benchmark's **serial number**, supplied by the
adapter through the `latest_key` callable — the core never imports it, per the
layering rule.

**Two fail-closed branches.** If any member's key is missing, or if the maximum
is *tied*, `latest_id` stays `None` and `stale` is **empty** — nothing is
suppressed and a note is recorded. The gate declines rather than guesses. A tie
at the maximum genuinely does not determine a winner, and picking one would be
fabricating precedence.

### 4.2 How the thresholds got their values

`stage0_results/stage1_operating_point.json → selection_rule`, verbatim:

```json
"require":   ["pair_filter is True", "n_suppressed_harmful == 0"],
"maximise":  "pair_recall_pool",
"tie_break": ["higher cos_pair", "higher nli_contradiction",
              "tighter r_min (frozen<loose<off)",
              "stricter ambiguity_mode (all<any<none)"],
"fit_on":    "detection quality only - no LLM, no accuracy, no gold answer",
"split":     "sh_6k + sh_32k (calibration) ONLY"
```

The grid is 3 × 3 × 3 × 3 × 2 = 162 cells. `pair_filter = True` is a
**requirement, not a preference** — §3.3 is why. The selected cell:

```
cos_pair = 0.90    r_min = 0.44 ("loose")    ambiguity_mode = "none"
nli_contradiction = 0.90                     pair_filter = True
```

**The `fit_on` line is the load-bearing one.** The thresholds were chosen by
looking at *detection* quality — did the detector identify the superseded facts —
with **no LLM in the loop, no answer graded, and no gold answer consulted**.
Answer accuracy was measured only afterwards, with the thresholds already frozen.
That is what makes the accuracy numbers in §5 an *out-of-sample* result rather
than a curve fit.

### 4.3 What the frozen detector achieves on the calibration split

Detection quality (`stage0_results/stage1_operating_point.json`):

| metric | pooled (200 q) | sh_6k | sh_32k |
|---|---|---|---|
| `pair_precision` | **1.000** | 1.000 | 1.000 |
| `fact_precision` | **1.000** | 1.000 | 1.000 |
| `pair_recall_pool` = tp/gt_pool | 0.9784 (2673/2732) | 0.9806 (1416/1444) | 0.9759 (1257/1288) |
| `question_recall_conflicted` | 0.9568 (133/139) | 0.9730 (72/74) | 0.9385 (61/65) |
| `n_conflicted_gold_cut` | 2 | 0 | 2 |

The two `gold_cut` cases are worth understanding, because they are a **rule
mismatch, not a defect**. On sh_32k q8 and q9 the gold answer is *not* the
highest-serial value (`gold_is_latest: false`). H-Nav follows the precedence rule
the prompt itself states, so it suppresses the gold-carrying serial (707 and
1291). The gold-based oracle, which knows the answer, does not. The detector is
behaving exactly as specified; the specification and the gold label disagree on
those two items. This is the 4/77 exception rate from `gold_rule.py` (§1.3)
showing up in practice.

Downstream answer accuracy on the conflicted stratum, whole-context harness
(`stage0_results/stage1/detector_gap_sh{6,32}k.json` and
`stale_suppression_probe_sh{6,32}k.json`, `results[0].by_stratum.conflicted.arms`):

| subset | native | oracle (gold-based ceiling) | **H-Nav detector** | share of oracle gain captured |
|---|---|---|---|---|
| sh_6k | 4/74 (5.4%) | 66/74 (89.2%) | **66/74 (89.2%)** | **100.0%** |
| sh_32k | 7/65 (10.8%) | 53/65 (81.5%) | **51/65 (78.5%)** | **95.7%** |

See `presentation_evidence/ORACLE_VS_DETECTOR_ANALYSIS.md` for the full treatment,
including why these two subsets are the *only* ones where an oracle ratio exists.

**§4.5 answers the obvious follow-up: how was a precision of 1.000 verified, and
against what?** It is the section to read before quoting any of the numbers
above, because two of the four metrics are less independent than they look.

### 4.4 Applying the decision

The gate names facts; the page is a list of chunk texts. The splice lives in the
adapter (`mab_adapter.py:122+`) because it needs the benchmark's serial numbering
and its sentence-joining chunker. Its contract, enforced by
`test_read_policy_facts.py` against hand-built pages whose expected output is
written out in full:

- surviving facts keep their **original serials** — no renumbering, because the
  prompt states its rule in terms of serial order;
- the page is otherwise byte-identical.

One earlier design was measured and **discarded**: doing the same thing at
*chunk* granularity. A precision-1.000 detector acting on chunks of ~250 facts
**harmed twice as often as it helped** — 228 helped against 441 harmed across the
grid (`presentation_evidence/data/item11_summary.json`), because deleting a chunk
to remove one stale fact also removes ~249 innocent ones. Hence
`HNAV_FINAL_REPORT.md:445`, methodological finding 9: *intervention granularity
must match conflict granularity.*

### 4.5 How the detection numbers were verified

§4.3 reports precision 1.000, pair recall 0.978 and question recall 0.957. A
precision of exactly 1.000 should invite suspicion, not applause, so this section
sets out **what those numbers were measured against, which parts of the
measurement are independent of the detector, and which parts are not.**

#### 4.5.1 The ground truth is built from fact text, not from gold answers

`fact_table` (`hnav/stage1/detector_gap.py:200-221`) constructs the reference
labels. Its docstring states the constraint:

> Everything the detector's ground truth needs, decided by the validated parser
> **without reading a single answer**.

It works on `key_members(item)` (`hnav/labeling/question_strata.py:107-121`),
which parses the **raw dataset `context` field** — the complete fact list, not
the retrieved page. From that it derives:

```
superseded  =  { fact f : ∃ a later fact of the SAME key with a DIFFERENT object }
latest[key] =  highest serial carrying that key
latest_obj  =  that fact's object
```

So the reference standard is the benchmark's **own stated precedence rule**
("higher serial = newer, latest wins"), applied mechanically to the full corpus.
It is not the gold answer key. Three consequences worth being explicit about:

- The detector is scored on **whether it correctly identifies superseded facts**,
  not on whether it agrees with the answer key. Those two things coincide on
  ~95% of conflicted questions and diverge on the rest (§1.3, §4.3).
- The ground truth sees **the whole context**; the detector sees only a 50-fact
  pool drawn from a retrieved page. Recall is therefore measured against
  something strictly larger than what the detector could possibly reach — which
  is deliberate, and is why recall falls at scale while precision does not (§5.4).
- Because no answers are read, this scoring could run in the **offline tier with
  no LLM at all**. That is exactly how the operating point was chosen (§4.2).

#### 4.5.2 The four metrics, and what each actually tests

```
pair_precision            = tp / (tp + fp)
pair_recall_pool          = tp / gt_pool
fact_precision            = n_suppressed_superseded / n_suppressed
question_recall_conflicted= n_conflicted_hit / n_conflicted
```
(`finish_metrics`, `detector_gap.py:471-484`)

**`tp` / `fp` — the pair test** (`score_decision`, `detector_gap.py:435-441`).
For every pair the gate marked `verified`:

```python
true_pair = bool(ra and rb and ra[2] is not None and ra[2] == rb[2]
                 and ra[3] != rb[3])
m["tp" if true_pair else "fp"] += 1
```

A verified pair counts as a true positive iff **same key AND different object**.

**`gt_pool`** is the denominator: `gt_pairs(pool_ids, by_id)` enumerates *all*
same-key/different-object pairs that existed among the 50 pool candidates,
whether the detector found them or not. So recall is measured against an
exhaustive enumeration, not against a sample.

**`fact_precision` — the page-effect test** (`classify_drops`,
`detector_gap.py:379-415`). This one is materially stronger than the pair test,
and its docstring explains why:

> The criterion is what the edit does to the **PAGE**, not what one fact looks
> like in isolation: for every key the drop set touches, take the members that
> survive and ask whether the key's newest surviving value is still the key's
> newest value. If it is not — or if the key loses every member — then every fact
> dropped from that key counts as harmful, because the page now says something
> different about that key than the corpus does.

Two fail-safe choices in that function are worth naming:

- **Unparseable dropped facts count as harmful**, not as unknown
  (`m["n_suppressed_harmful"] += unknown`). Absence of evidence is scored against
  the detector.
- **The check is per-key, not per-fact.** The docstring says what that prevents:
  a single-member key would otherwise be waved through as a "duplicate
  restatement", because its only fact *is* its own latest value — right up to the
  moment deleting it erases the key entirely.

**`n_conflicted_hit` — the queried-key test** (`detector_gap.py:462-468`). For a
conflicted question, a hit requires **both**: some non-latest member of the
queried key was cut, **and** the latest member was not.

#### 4.5.3 The independence audit — where the check is circular, and where it is not

This is the part that matters for defending the number.

**`pair_precision = 1.000` is partly true by construction, and the size of that
"partly" is exactly one half of the criterion.**

The TP criterion has two clauses: `same key` and `different object`. When
`pair_filter = True`, a pair only becomes an edge if `same_key_pair` already
found the two parsed keys equal (§2.5). Both the screen and the judge call the
same `parse`. **So the `same key` clause is effectively decided before the judge
is consulted, and the only way a verified pair can realistically be scored `fp`
is if the two facts carry the *same object*.**

(Strictly, one narrow channel remains: the screen parses the *chunked page text*
and the judge parses the *raw `context` field* (§4.5.4), so a fact whose two
renderings parsed to different keys could pass the screen and still fail the TP
test. `fp = 0` means that channel never fired either — but it is a check on
parse stability across renderings, not on subject identity.)

Stated precisely, therefore:

> `fp = 0` over 2,673 verified pairs means: **the NLI stage never once verified a
> pair whose two facts carried the same value.** It is *not* independent evidence
> that the pairs shared a subject.

That is still a real, non-trivial measurement — it is the §3.3 duplicate case
(`n_suppressed_same_value = 0`) and it is the thing NLI was added to do. But it
is not a validation of the identity screen, and it should not be presented as one.

**The non-circular version of the same test exists, was run, and is reported.**
With `pair_filter = False` the `same key` clause is no longer guaranteed, and the
identical scoring function measures **31.50%–94.09% false verification across all
81 cells** (§3.3). That is the honest measurement of NLI against a truth standard
it does not share a component with — and it is the number that looks bad. The
repository ran the experiment that could have embarrassed it and published the
result; that is the strongest thing that can be said for the precision figure.

**Where the check *is* independent:**

| metric | independent of the detector? | why |
|---|---|---|
| `pair_precision` | **partly** — only the "different object" clause | the "same key" clause is enforced by the same parser that judges it |
| `pair_recall_pool` | **yes** | denominator enumerated exhaustively from the corpus; the detector had no say in it |
| `fact_precision` | **yes** | tests the *effect of the edit on the page* against the *full corpus*, using surviving-member logic the gate never runs |
| `question_recall_conflicted` | **yes, and gold-derived** | see below |

**The gold-derived counters are quarantined.** `score_decision`'s docstring is
explicit:

> Everything except the two `n_conflicted_*` counters is parse-derived: it uses
> fact text and serial order only. The conflicted-stratum counters use the
> question→key assignment from `question_strata`, which is **gold-derived** —
> they are reported for **ATTRIBUTION** and are **never used to choose an
> operating point**.

`question_strata.py:200-240` does read `item["answers"]` — it needs them to map a
question to its key and to identify `target_serial`. So `n_conflicted`,
`n_conflicted_hit` and `n_conflicted_gold_cut` are the one place gold enters, and
the selection rule (§4.2) maximises `pair_recall_pool` instead, which does not.
This is the leakage boundary from §0 holding under load.

#### 4.5.4 The residual failure mode both sides share

**The parser is a single point of failure for detector and judge alike.** If
`parse` mis-parses a fact, the detector's identity screen and the ground truth's
`by_id` inherit the *same* error, and the mistake is invisible to the metric.

What bounds this:

- **Measured coverage 99.44–99.65%** across all four subsets
  (`item08_parser.json`), so at most ~0.5% of facts are unparsed.
- Unparsed facts are **excluded from ground truth** (`if p is None: continue`)
  **and rejected by the identity screen** (`ka is not None`) **and counted as
  harmful if dropped**. All three treatments fail in the safe direction.
- There is one small genuine asymmetry: ground truth parses the **raw `context`
  field** with the line-anchored regex, while the detector parses the
  **chunked page text** through the inline fallback (§1.1). Same function, two
  renderings of the same text. Agreement is not formally proven, though the
  99.5% coverage figures are computed on the raw field and the chunk-level tests
  (`test_chunking_and_facts.py`) pin the fallback's behaviour.

What is *not* claimed: that the parser is correct on text outside this synthetic
template family. It is not, and that is the whole reason the disagreement verdict
was routed through NLI rather than through object comparison (§1.4).

#### 4.5.5 The independent recomputation on the held-out run

For the one run that carries the headline claim, the suppression set was
re-derived **from scratch, outside the run**
(`presentation_evidence/data/item14_summary.json`). Its stated method:

> sum of `len(per_question[i].plan.suppress_serials)`; each serial joined to the
> full sh_64k context, keyed by the validated parser, and compared to the key's
> highest serial in the full context

| quantity | artifact (VC4, in-run) | recomputed (out-of-run) |
|---|---|---|
| facts suppressed | 735 | 735 |
| genuinely superseded | 735 | 735 (`n_not_key_latest = 735`) |
| suppressed fact *was* its key's latest | — | **0** |
| same-value deletions | 0 | — |
| harmful | 0 | — |
| parse failures among the suppressed | — | **0** |
| carrying the queried question's gold value | — | **1** |

The two paths agree. Note this is a **recomputation, not a second opinion** — it
uses the same parser and the same rule, so it verifies the bookkeeping (that the
run's counters match what its own `suppress_serials` imply) rather than the rule.
It is exactly what caught the gold-cut mechanism being reported backwards: the
same artifact carries the correction that q18's gold was **never** suppressed and
q20's **was**, giving 2 predicted gold-cuts → 1 deletion → 0 accuracy losses.

Alongside it, the pre-registered void conditions provide the run-level checks:
**VC4** requires `n_suppressed_harmful == 0`, **VC8** requires the positive
control to fire on every question (100/100, `n_fact_edits_applied_per_arm = 100`).
Both passed. `run_void: false`.

#### 4.5.6 The accuracy numbers are verified differently

Detection quality and answer accuracy are separate measurements with separate
verification, and §4.3 mixes both in one table.

The accuracy figures (5.4% → 89.2%, 17/66 → 37/66) come from the benchmark's own
`substring_exact_match` grader — deterministic, offline, no LLM judge — with
three additional controls:

- **McNemar exact, paired.** Every question is compared against *itself* under
  the other arm, so per-question difficulty cancels. The reported `b`/`c` are the
  discordant counts, not a difference of two independent proportions.
- **An A/A arm.** `native_repeat` runs the byte-identical prompt a second time and
  measures the noise floor directly: **0/0 discordant pairs** on sh_64k. Without
  this the +20 could be vLLM nondeterminism (`HNAV_FINAL_REPORT.md` §9 finding 5
  records a 4-point swing between identical runs).
- **A cross-run identity check.** `detector_gap_sh{6,32}k.json` record
  `native_cross_run.identical: true` — the baseline arm reproduced the earlier
  probe's outputs on 100/100 questions in both subsets, which is what licenses
  putting oracle and detector numbers in the same table at all.

#### 4.5.7 The one-paragraph answer

> Detection was scored against a reference standard built by applying the
> benchmark's own "latest serial wins" rule to the **complete fact corpus** with
> the validated parser, reading **no gold answers**. `fact_precision = 1.000`
> (2,673/2,673 on calibration, 735/735 held out) is the strongest of the four
> figures: it tests the *effect of the edit on the page* against the full corpus,
> using surviving-member logic the gate never runs, with unparseable deletions
> counted as failures. `pair_precision = 1.000` is weaker than it looks — with
> the identity screen on, the "same key" half of the criterion is enforced by the
> same parser that judges it, so `fp = 0` means only that NLI never verified a
> same-value pair. The non-circular version of that test was run and reported:
> without the identity screen the same scorer measures **31.5%–94.1%** false
> verification. `question_recall_conflicted` is the one gold-derived metric and is
> quarantined to attribution — the operating point was selected on
> `pair_recall_pool`, which is not. The shared residual risk is the parser itself,
> bounded at 99.5% coverage with all three failure paths (ground truth, identity
> screen, harm accounting) failing safe.

---

## 5. What "held-out accuracy" means

### 5.1 The definition

The four single-hop subsets are split once, in advance, and the split is a hard
invariant of the project (`CLAUDE.md`):

```
CALIBRATION  (may be tuned on)  :  sh_6k , sh_32k
HELD OUT     (may NOT be)       :  sh_64k , sh_262k
```

**Held-out accuracy is the answer accuracy measured on a subset that was never
looked at while any threshold, any design choice, or any grid cell was being
selected.**

The rule is enforced in code, not just in prose:

- `hnav/stage0/m3_headroom.py` refuses to fit thresholds without a calibration
  subset;
- `hnav/stage0/m4_marginal_diff_test.py` refuses a non-calibration split outright;
- `hnav/stage1/detector_gap.py:1632` refuses to select a cell on a confirmatory
  subset — *"would tune on HELD-OUT data"*;
- `stage0_results/stage1_operating_point.json → provenance` records
  `fit_subsets: ["sh_32k","sh_6k"]` and
  `confirmatory_refused: ["sh_64k","sh_262k"]`.

Additional conditions on top of the split:

- **One shot.** The held-out run is *pre-registered*: primary criterion, protective
  criterion, void conditions VC1–VC8, and a falsifiable side-prediction are all
  written down **before** the run. Afterwards `shot_spent: true` is recorded in
  the artifact. There is no second attempt, and no re-analysis with different
  cuts.
- **sh_262k was never run at all.** It remains completely untouched.

### 5.2 Why it exists — the failure it prevents

Any system with tunable thresholds can be made to look good on data used to tune
it. With a 162-cell grid, five thresholds and 200 questions, picking the cell with
the best accuracy would be a near-guarantee of an inflated number — the selection
itself fits the noise. The resulting figure would describe *the search*, not *the
method*.

Held-out accuracy closes that loop by construction: the thresholds are frozen
first, the subset is opened once afterwards, and whatever comes out is the
reported result — favourable or not.

This project layers a second protection on top. Even on the calibration split the
thresholds were fit on **detection quality only** — `fit_on: "detection quality
only - no LLM, no accuracy, no gold answer"` (§4.2). So the calibration accuracy
figures in §4.3 are *already* out-of-sample with respect to accuracy. The
held-out subset then tests something further: whether the method survives a
**change of scale** — 455 facts and 2 chunks at sh_6k versus 4,580 facts and 17
chunks at sh_64k, with the retrieval page covering only 10 of those 17.

### 5.3 The held-out result

**`factconsolidation_sh_64k`, one pre-registered shot, 100 questions × 5 arms,
500 LLM calls** (`stage0_results/stage1/detector_gap_confirmatory_sh64k.json`,
summarised in `HNAV_FINAL_REPORT.md` §7):

| arm | overall | non-conflicted | conflicted | McNemar b/c | net | exact p | tokens |
|---|---|---|---|---|---|---|---|
| baseline | 0.450 | 28/34 | 17/66 | — | — | — | 0 |
| A/A repeat | 0.450 | 28/34 | 17/66 | 0/0 | 0 | 1.0 | 0 |
| **suppression** | **0.640** | 27/34 | **37/66** | **0/20** | **+20** | **1.9×10⁻⁶** | **−0.31%** |
| placement | 0.480 | 28/34 | 20/66 | 2/5 | +3 | 0.45 | 0 |
| anti | 0.430 | 28/34 | 15/66 | 3/1 | −2 | 0.63 | 0 |

Reading the row that matters:

- **Conflicted accuracy 17/66 → 37/66**, McNemar exact **p = 1.9 × 10⁻⁶**.
- **b = 0**: not one conflicted question that baseline got right was broken.
- The **A/A repeat arm** — the same prompt run twice — produced **0/0 discordant
  pairs**. The noise floor is exactly zero, so the +20 is not run-to-run variance.
- **Cheaper**: −0.31% tokens, because deleting facts shortens the prompt.
- **Suppression precision 1.000**: all 735 deleted facts independently verified as
  genuinely superseded (`item14_summary.json`, recomputed by joining each
  suppressed serial back to the full sh_64k context).

**Primary criterion: MET.** Net ≥ +10 ✓ (+20); p < 0.01 ✓; token cost ≤ 0 ✓.

**Protective criterion: VOIDED, by one question.** On q77 the model went from
answering "John Milton" to *"The provided knowledge pool does not contain any
information about…"* — **with the gold fact still on the page** (`gold_cut:
false`). Not a deletion error: a refusal induced by the edit. Under the
deliberately strict registered rule, one such case voids the safety claim.
The artifact records `protective_claim_void: true`, `run_void: false`,
`shot_spent: true`.

**The falsifiable side-prediction: MISSED, and reported as missed.** Two gold-fact
deletions were predicted; observed were 1 deletion and 0 accuracy losses. And the
detail is a caveat, not reassurance — on the one real deletion the page was left
containing only the *wrong* value and the model answered correctly anyway from
world knowledge. So *"zero accuracy loss from gold cuts" is evidence about the
evaluator, not evidence that gold cuts are safe.*

### 5.4 Reading it correctly

**Why held-out is lower than calibration** (89.2% at sh_6k → 56.1% here). The
cause was written down *before* the run and it is **coverage, not detector
quality**: 735 suppressions here against 1,416 and 1,257 on calibration, because
the 50-fact pool cap and the 10-of-17 retrieval bound what the detector can even
see. Facts that never reach the page cannot be suppressed. Precision stayed at
1.000; recall fell.

**What the held-out number does buy.** It is a genuine out-of-sample estimate at a
scale never used for tuning, with a pre-registered criterion, a measured zero
noise floor, and a stated failure.

**What it does not buy** (`HNAV_FINAL_REPORT.md` §10–11):

- **One subset, one arena, one model, one shot.** No generalisation claim.
- The ratios 100% / 95.7% from §4.3 are **calibration-only and cross-harness**.
  There is **no oracle arm at sh_64k** — `detector_vs_oracle` is empty by design,
  because a whole-context prompt at that scale is 75,886 tokens against a 65,536
  limit. The ceiling at this scale is **unmeasured**, and those ratios may not be
  carried over.
- **The mechanism is effective but not yet safe.** Any statement of the +20 must
  name the voided protective criterion in the same breath, and the method is not
  recommended for deployment on traffic containing non-conflicted queries until
  the refusal mechanism is eliminated.

---

## Appendix A — one fact through the whole pipeline

Source: `presentation_evidence/data/data_journey_sh6k_q1.md`, sh_6k question
index 1.

**Two facts in the context, same key:**

```
  91. Nobuhiro Watsuki is famous for Rurouni Kenshin.
 259. Nobuhiro Watsuki is famous for The Fairly OddParents.
```

**Question:** "What is Nobuhiro Watsuki famous for?"  **Gold:** `The Fairly OddParents`

| stage | what happens |
|---|---|
| parse | both → `("\| is famous for ", "Nobuhiro Watsuki")`, objects `Rurouni Kenshin` / `The Fairly OddParents` |
| cosine screen | same subject, same relation template, differing only in the object → cosine above 0.90 → edge kept |
| identity screen | keys are equal → passes; NLI is allowed to run |
| LOO residual | `r = √(1−c²)` under 0.44 at this cosine → group survives (inert at this operating point, §2.3) |
| bidirectional NLI | both orderings must clear 0.90 contradiction → verified |
| final decision | `max(91, 259) = 259` survives; **fact 91 is stale** |
| splice | fact 91 deleted; 259 keeps its original serial |

**Measured outcome** (`stale_suppression_probe_sh6k.json → per_question[1]`):

```
native           -> 'Rurouni Kenshin'          correct = False
native_repeat    -> 'Rurouni Kenshin'          correct = False    (A/A floor: identical)
oracle_suppress  -> 'The Fairly OddParents'    correct = True
oracle_recency   -> 'Rurouni Kenshin'          correct = False
anti             -> 'Rurouni Kenshin'          correct = False
```

Deleting one sentence flips the answer. Moving it to the end does not. That
contrast is the whole finding in miniature: the failure is not about ordering, it
is about the stale value being *present at all*.

> The per-pair cosine and NLI scores for this specific question are **not stored**
> in any committed artifact — only the prompt hashes and the arm outcomes are.
> The stage-by-stage table above describes what the code does to this input; the
> intermediate scores are reconstructible on the GPU box, not quotable from the
> repository.

---

## Appendix B — provenance

| claim | source |
|---|---|
| parser templates, grouping, conflict definition | `hnav/labeling/conflict_analysis.py:16-88` |
| parse coverage 99.44–99.65% | `presentation_evidence/data/item08_parser.json` |
| inline-fallback regex and its test | `hnav/adapters/mab_adapter.py:62-71`; `hnav/tests/test_chunking_and_facts.py` |
| identity screen | `hnav/adapters/mab_adapter.py:712-721` |
| pool cap = 50 | `hnav/adapters/mab_adapter.py:321-344, 723-730` |
| cosine screen, components, LOO residual, NLI, finalisation | `hnav/core/read_gate.py:556-657` |
| `qr_residual` implementation | `hnav/core/geometry.py:159-179` |
| `r = √(1−c²)` endpoints | `hnav/tests/test_geometry.py::test_qr_residual_is_zero_inside_the_span_and_one_outside` |
| `R_LOOSE = 0.44` and its pass-through comment | `hnav/stage1/calibrate_read_policy.py:112-113` |
| `R_MIN_CAL`, `COS_PAIR_CAL`, `NLI_CONTRA_DEFAULT`, NLI model | `hnav/core/read_gate.py:113, 150, 152, 154` |
| `√(1−0.44²) = 0.8980` | `presentation_evidence/data/item09_thresholds.json` |
| frozen operating point, selection rule, detection metrics, ambiguity note | `stage0_results/stage1_operating_point.json` |
| ground-truth table built from fact text, no answers read | `hnav/stage1/detector_gap.py:200-221` (`fact_table`) |
| `key_members` parses the raw `context` field | `hnav/labeling/question_strata.py:107-121` |
| exhaustive true-pair enumeration (recall denominator) | `hnav/stage1/detector_gap.py:224-243` (`gt_pairs`) |
| superseded / same-value / harmful split, per-key page-effect test | `hnav/stage1/detector_gap.py:379-415` (`classify_drops`) |
| tp/fp criterion, conflicted-hit counters, gold-derived quarantine note | `hnav/stage1/detector_gap.py:419-468` (`score_decision`) |
| the four metric formulas | `hnav/stage1/detector_gap.py:471-484` (`finish_metrics`) |
| `question_strata` reads `item["answers"]` | `hnav/labeling/question_strata.py:200-240` |
| out-of-run recomputation of the 735 suppressions; gold-cut correction | `presentation_evidence/data/item14_summary.json` |
| VC4 (`n_suppressed_harmful == 0`), VC8 (positive control) | `hnav/stage1/detector_gap.py:1133-1137`; `detector_gap_confirmatory_sh64k.json → void_conditions` |
| baseline reproduced 100/100 across runs | `stage0_results/stage1/detector_gap_sh{6,32}k.json → native_cross_run.identical` |
| NLI false-verification 31.50–94.09% → 0.000 | `presentation_evidence/data/item10_summary.json` |
| Kyd/Marlowe scores — **prose only, no JSON artifact** | `TEZ_BULGULARI.md:265-267`, flagged in `item10_summary.json` |
| chunk-granularity harm 228 helped / 441 harmed | `presentation_evidence/data/item11_summary.json` |
| calibration conflicted accuracy | `stage0_results/stage1/detector_gap_sh{6,32}k.json`, `stale_suppression_probe_sh{6,32}k.json` → `results[0].by_stratum.conflicted.arms` |
| held-out result, void conditions, q77 | `stage0_results/stage1/detector_gap_confirmatory_sh64k.json`; `HNAV_FINAL_REPORT.md` §7 |
| 735/735 suppressions verified superseded | `presentation_evidence/data/item14_summary.json` |
| calibration/held-out split enforcement | `CLAUDE.md`; `hnav/stage1/detector_gap.py:1632`; `stage0_results/stage1_operating_point.json → provenance` |
| what may and may not be claimed | `HNAV_FINAL_REPORT.md` §10–11 |
| oracle-vs-detector separation | `presentation_evidence/ORACLE_VS_DETECTOR_ANALYSIS.md` |
| worked example | `presentation_evidence/data/data_journey_sh6k_q1.md` |
