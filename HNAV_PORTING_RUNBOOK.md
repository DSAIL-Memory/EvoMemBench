# H-Nav Porting & Replication Runbook

**Audience: an agent (or engineer) who has been handed this repository plus a
new model, and must (a) run H-Nav on it, (b) prove or disprove that H-Nav helps
that model too, or (c) compare several models on the same task.**

You do not need to have seen the previous work. Everything you must know is
here or is cited by path. Read §0 and §1 before touching anything.

> Companion documents: `HNAV_HOW_IT_WORKS.md` (what the system does),
> `HNAV_FINAL_REPORT.md` (what was found and why), `CLAUDE.md` (hard invariants).

---

## 0. The first thing to understand: there are THREE models

Most porting mistakes come from thinking "the model" is one thing. It is three,
they fail differently, and **which one you are swapping decides most of your
work**.

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ANSWERING LLM        reads the memory page, produces the    │
│                          answer that gets scored                │
│                          (was: Qwen3-4B-Instruct-2507)          │
├─────────────────────────────────────────────────────────────────┤
│  2. EMBEDDING MODEL      turns facts and chunks into vectors;   │
│                          drives retrieval AND H-Nav's geometry  │
│                          (was: Qwen3-Embedding-4B, fp32)        │
├─────────────────────────────────────────────────────────────────┤
│  3. NLI CROSS-ENCODER    confirms two facts really contradict   │
│                          (was: cross-encoder/nli-deberta-v3-large)│
└─────────────────────────────────────────────────────────────────┘
```

**Decision table — find your row before planning anything:**

| What you are swapping | What must be redone | Roughly |
|---|---|---|
| **Answering LLM only** (the common case) | A/A noise floor, baseline accuracy per stratum, the oracle probe, the detector-gap run, a new pre-registration + confirmatory run. **The detector itself is unchanged** — no threshold re-fit, no cache invalidation. | 1 day |
| **Embedding model** | *Everything geometric*: cache namespace changes, replica fidelity (M0) must be re-verified, all thresholds re-fit on the calibration split, then everything in the row above. | 2–3 days |
| **NLI model** | The subject-screen + NLI precision measurement, and the operating point's NLI threshold. Nothing else. | 2–3 hours |
| **Benchmark/dataset** | Effectively a new project. See §8. | weeks |

**Why the answering-LLM case is cheap:** H-Nav's detector never sees the
answering model. It reads facts, embeds them, screens them, and deletes. Swapping
the answering LLM changes *whether the intervention helps*, not *whether the
detector is correct*.

---

## 1. Non-negotiable rules

Violating any of these invalidates the result — not the code, the **research**.
They are enforced by tests; do not disable a test to make a run pass.

1. **Never tune on held-out data.** `sh_6k` + `sh_32k` are the only subsets any
   threshold, operating point or design choice may touch. `sh_64k` and
   `sh_262k` are held out. Scripts refuse them by default; the refusal is the
   feature.
2. **Pre-register before you measure.** Commit the design, criteria, analysis
   code and void conditions *before* the confirmatory run. Commit timestamps are
   your proof of ordering — a reviewer will check them.
3. **One shot.** No re-running a confirmatory experiment because you did not like
   the number. If it voids, diagnose and re-register; do not reinterpret.
4. **`hnav/core/write_policy.py` must never exist.** The write path was measured
   as NO_GO and the ban is permanent (a test enforces it).
5. **`HNAV_MODE=off` must stay a byte-identical no-op.** If a change makes the
   off path differ at all, the change is wrong.
6. **Report stratified, never pooled.** Store sizes span 455 → 18,332 facts, and
   pooling across them produces statistics that describe no subset (we proved
   this the hard way; see `HNAV_FINAL_REPORT.md` §9.8).
7. **Never delete `hnav/_cache/emb/`**, and never copy it between machines unless
   the embedding model *and* dtype *and* max_length match exactly.
8. **Every claim needs its raw artifact committed**, not just its summary.

---

## 2. Step 0 — Prove the baseline works before you change anything

Do not swap a model into a system you have not verified.

```bash
# 1. Full test suite. No GPU, no network needed.
pytest hnav/tests/ -q                 # expect ~486 passed, 1-2 skipped

# 2. Reproduce a committed offline measurement from raw data
python3 hnav/labeling/conflict_analysis.py
#    expect sh_262k: 11,037 keys / 7,197 conflicted (65.2%)

# 3. Environment gate (needs the GPU box)
python hnav/deploy/check_env.py       # must exit 0
```

If any of these fail, **stop and fix that first**. A port built on a broken
baseline produces numbers nobody can interpret.

---

## 3. Step 1 — Swap the answering model

### 3.1 Configure it

Everything is read from the repo-root `.env`, and `os.environ` wins over the
file. The relevant knobs:

```bash
HNAV_LLM_BASE_URL=http://localhost:8003/v1   # OpenAI-compatible endpoint
# the model name is taken from the endpoint's /v1/models
```

Serve the new model with **the same generation settings the benchmark uses**:
temperature 0, `max_tokens` 10 (the benchmark's `generation_max_length`), and a
context window that fits the largest prompt you intend to send (see §3.2).

`hnav/deploy/serve_stage1_chat.sh` is the reference launcher. Keep its
determinism-favouring flags (`--enforce-eager`, `--max-num-seqs 1`, prefix
caching off) — they do not make the server deterministic (see §5.1) but they
reduce variance, and **the frozen substrate must not change between your arms**.

### 3.2 Check the prompts fit — before you run anything

This has bitten this project twice. Measure, do not assume:

```bash
python hnav/stage1/detector_gap.py --dry-run --subsets sh_6k --harness retrieval \
       --page-source benchmark
# prints exact call count and prompt-token totals, sends nothing
```

Known sizes: **~6.7k tokens** at sh_6k, **~50k** at sh_64k on the retrieval path.
A whole-context prompt at sh_64k is **75,886 tokens** and does not fit a 65k
window — this is why the confirmatory run uses the retrieval path.

**If your new model has a smaller context window than the prompts require, you
cannot simply truncate.** Truncation changes what the model sees and silently
invalidates every comparison. Either serve a larger window or restrict the
subsets you claim.

### 3.3 What you do NOT need to redo

- The detector, its thresholds, its cache. Unchanged.
- The question stratification (§4.1) — it is derived from the dataset, not the
  model.
- Replica fidelity (M0) — that validates the retrieval replica, not the LLM.

---

## 4. Step 2 — Establish the new model's baselines

**This is the step people skip and it is the one that decides whether your final
number means anything.**

### 4.1 Get the question strata (free, no model, no GPU)

```bash
python hnav/labeling/question_strata.py
# writes stage0_results/question_strata.json
```

This classifies every question as **conflicted** (its key has ≥2 competing
values) or **non-conflicted**. It is model-independent — reuse it as is.

Why it matters: on the original model, non-conflicted questions were answered
**26/26 in every run** while conflicted ones were **0–5/74**. If you report only
overall accuracy you will be reporting mostly the conflict-free stratum. **All
your readouts must be stratified.**

### 4.2 Measure the A/A noise floor — mandatory, model-specific

Run the untouched baseline **twice** and count how many answers differ.

```bash
HNAV_LLM_BASE_URL=<your endpoint> \
  python hnav/stage1/stale_suppression_probe.py --subsets sh_6k
# the native / native_repeat arms are the A/A pair
```

- On the original substrate this was **0/0 discordant** on the frozen server,
  but **~3.3% per question** (a 4-point accuracy swing) on the default one.
- **Your model will have a different floor.** Measure it. Any effect you later
  claim must clear it.
- Useful property to check: on the original model, noise was confined
  **entirely** to the conflicted stratum — the non-conflicted stratum had zero
  flips across 28 run-pairs, which made it a free zero-noise control. Verify
  whether that holds for your model; if it does, use it the same way.

### 4.3 Measure baseline accuracy per stratum

From the same run. Record overall, conflicted and non-conflicted separately.

**Sanity check before proceeding:** does your model also fail conflicted
questions? If it answers them correctly at a high rate, **there is no headroom
for H-Nav on this model** — and that is a publishable finding, not a failure.
Report it and stop; do not go hunting for a configuration that shows an effect.

---

## 5. Step 3 — Measure the ceiling before building anything

Never implement an intervention before measuring whether it *could* work.

### 5.1 The oracle probe

```bash
HNAV_LLM_BASE_URL=<your endpoint> \
  python hnav/stage1/stale_suppression_probe.py --subsets sh_6k sh_32k
# ~938 calls; calibration split only (it refuses held-out subsets)
```

Five arms: `native`, `native_repeat` (the A/A floor), `oracle_suppress`,
`oracle_recency`, `anti`. These use gold answers to identify the target — they
measure the **ceiling**, not a shippable system.

**Read the result like this:**

| pattern | meaning | what to do |
|---|---|---|
| suppress ≫ native, anti < native | the model anchors on stale facts, and position matters | proceed to §6 |
| suppress ≈ native | the model overrides context with world knowledge, or already resolves conflicts | **stop.** Report as a negative finding — this is a real result about the model |
| everything ≈ native, including anti | the model may be ignoring the page entirely | check the harness before concluding anything |
| effects inside the A/A floor | you have no measurable signal at this N | increase N or stop |

For reference, the original model gave: native 5.4% conflicted → suppress
**89.2%**, recency 27.0%, anti 1.4%.

---

## 6. Step 4 — Close the oracle-to-detector gap

A shippable policy may use **only detector output** — no gold, no answers, no
future facts.

```bash
# freeze the operating point on DETECTION QUALITY ONLY (no LLM, no gold, no accuracy)
python hnav/stage1/detector_gap.py --select --subsets sh_6k sh_32k
# then measure
HNAV_LLM_BASE_URL=<your endpoint> \
  python hnav/stage1/detector_gap.py --subsets sh_6k --harness retrieval \
         --page-source benchmark
```

**The rule that makes this credible:** the operating point must be frozen
*before* any arm is graded, and selected on detection metrics alone. Commit it,
then run. A reviewer will check the commit order.

Report **detector-achieved ÷ oracle-achieved**. On the original model this was
0.984 and 0.957 — i.e. dropping gold cost almost nothing. If your ratio is much
lower, the gap is in *detection*, not in the mechanism; report detector
precision/recall so a reader can tell which.

---

## 7. Step 5 — Pre-register, then take one shot

### 7.1 What the pre-registration must contain

Copy `stage0_results/stage1_preregistration_v2.md` as your template. It must
state, before any held-out data is touched:

1. **The claim, correctly scoped** — which stratum, which subset, which harness,
   and what is *inert* in your configuration.
2. **Arms and N**, with a power argument against **your measured A/A floor**
   (§4.2), not the one in this document.
3. **Success criterion as a computable expression.** Ours: conflicted net ≥ +10
   discordant pairs AND exact p < 0.01 AND token cost ≤ 0. Use McNemar on paired
   per-question outcomes, not a bare accuracy difference.
4. **Harm criterion**, stratified, with the noise floor inside it. Name the harm
   classes you will count separately (we use `gold_cut`,
   `malformed_generation`, `refusal_after_edit`, `information_loss`).
5. **Void conditions** — states in which the run does not count *at all*, as
   distinct from a run that counts and fails. Ours: edit mismatches > 0, baseline
   outside its pre-declared band, A/A floor non-zero, any harmful suppression,
   wrong page source, containment violations, positive controls not firing.
6. **At least one falsifiable side-prediction** derived without the model (ours:
   the exact number of questions whose gold fact would be deleted, computed from
   the parse). Predictions you can miss are what make the rest credible — ours
   missed, and we reported it as a miss.
7. **The analysis code, committed alongside.** No optional stopping.

### 7.2 Fire it

```bash
HNAV_LLM_BASE_URL=<your endpoint> \
  python hnav/stage1/detector_gap.py --confirmatory --subsets sh_64k \
         --harness retrieval --page-source benchmark
```

`--confirmatory` refuses every configuration except the registered one, and
refuses `--select` so nothing can fit thresholds on held-out data.

Then report **whatever comes out**, including a void.

---

## 8. Proving H-Nav generalizes across models

If your goal is the stronger claim — *"this method helps models in general"* —
one model is not enough and a second model is barely enough. Design it as a
single pre-registered multi-model study, not a series of one-offs.

### 8.1 What to hold constant

Everything except the answering model: same benchmark, same subsets, same
splits, same detector, same operating point *policy* (see §8.2), same prompt
templates, same evaluator, same harness, same page source.

### 8.2 The one decision that will be questioned

**Per-model operating points, or one shared point?**

| choice | claim it supports | cost |
|---|---|---|
| **Re-freeze per model** on that model's calibration split | "the *method* generalizes" | one calibration pass per model |
| **One shared operating point**, frozen once | "these *thresholds* transfer" | none, but a weaker result if it underperforms |

Recommended: **re-freeze per model, and report both.** Then state which claim
you are making. Do not silently re-tune per model and then claim threshold
transfer.

**Do not reach for isotropy correction to make thresholds transfer.** It is the
obvious idea and we tested it. ABTT (all-but-the-top) removes this encoder's
anisotropy completely — unrelated-pair mean cosine 0.604 → ~0.000 — and it does
improve detection quality a lot (recall at precision 1.000 rises 6.8× and 40×
on the two calibration subsets). But it did **not** buy threshold portability:
with only one usable transfer direction and a band-normalised threshold spread
that slightly favoured raw, we explicitly declined to claim it. And on held-out
sh_64k it changed no answer at all — 37/66 conflicted in both arms, not one
question different. See `stage0_results/abtt/ABTT_REPORT.md`.

Two transferable lessons if you try it anyway: fit the whitening **offline on
the calibration split** and ship (mean, components) as a fingerprinted constant
— that removes the "cannot fit on a small decision pool" problem entirely — and
whiten **only the fact-fact comparison**. Whitening the query vector too cost
27% of the reachable true pairs, because the pool selector then ranks a question
against facts in a space that has had the shared "factual English" directions
removed. ABTT helps symmetric comparison and hurts asymmetric retrieval.

### 8.3 Analysis for a multi-model study

- **Stratify by model and by question stratum.** Never pool across models — the
  same mistake as pooling across store sizes.
- **Pair within model** (McNemar on the same questions), then combine across
  models with a random-effects meta-analysis or, more simply, report each model
  and a sign/consistency statement ("k of n models improved, all p < …").
- **Pre-register the combination rule.** Deciding after seeing the models is the
  most common way this kind of study goes wrong.
- **Report the failures.** A model where H-Nav does nothing is *more* informative
  than a fourth where it works — it bounds the mechanism.

### 8.4 What a strong multi-model result looks like

> Across N models spanning M parameter scales and F families, fact-level
> suppression of detector-verified superseded memory improved conflicted-stratum
> accuracy in k of N (per-model McNemar p < 0.01 in j of N), with token cost ≤ 0
> in all cases, and no model showing conflicted-stratum harm. The effect size
> correlated with baseline conflicted accuracy: models that were already good at
> supersession had less to gain.

That last clause is worth measuring explicitly — it is the most likely *shape* of
the generalization result, and it makes the mechanism claim testable rather than
anecdotal.

---

## 9. Comparing the same outputs across models

If the goal is a comparison table rather than a causal claim, the protocol is
simpler but the traps are sharper.

### 9.1 Freeze the inputs, byte for byte

Generate the prompts **once**, store them, and send the identical strings to
every model. Do not regenerate per model — retrieval, chunking or a page-order
difference will silently change the input and you will be comparing two things
at once.

```
prompts.jsonl   ← one record per (subset, question, arm), with the exact string
                  and a sha256 of it
```

Then assert the hash before every model's run. Any mismatch voids that run.

### 9.2 Hold generation settings identical

Temperature 0, same `max_tokens`, same stop conditions, same system message. If a
model needs a different chat template, that is a **declared deviation** — note it
next to that model's numbers.

### 9.3 Measure each model's own noise floor

Run each model's baseline twice. **Do not assume a shared floor.** A 2-point
difference between two models means nothing if one of them has a 4-point A/A
swing.

### 9.4 Score identically

Use the benchmark's own `substring_exact_match`. Do not introduce an LLM judge
for a cross-model comparison unless you also measure the judge's own variance —
it becomes a fourth model in your pipeline.

### 9.5 Report

Per model: overall, conflicted, non-conflicted, A/A floor, token cost, harm
classes. **Show the A/A floor in the same table as the effect.** A reader must be
able to see at a glance whether a difference clears the noise.

---

## 10. Traps this project actually hit — check each one

Every item below cost real time or nearly produced a wrong number. They are
listed so you do not rediscover them.

| trap | symptom | guard |
|---|---|---|
| **Embedder served in reduced precision** | retrieval fidelity collapses (top-k agreement 1.000 → 0.24) while everything "works" | pin dtype explicitly; verify vector norms are 1±1e-7 |
| **Content-addressed cache missing a parameter** | you fix something and the fix measures as "no change" — the cache returns the old vectors | the cache key must encode model, dtype **and** max_length; same for any persisted NLI/score table |
| **Silent tokenizer truncation** | signals computed on the first ~12% of each chunk | measure the real token lengths; assert `max_length` covers them |
| **Server nondeterminism at temperature 0** | your effect is inside the run-to-run noise | always run the A/A arm; never claim an effect you have not compared to it |
| **Page selection not reproducible by re-encoding** | your arms use a different page than the benchmark would | read the page from the benchmark's own index, not from re-embedded vectors |
| **Pool built from the wrong page** | the policy names a fact absent from the page → edit fails → silent fallback to baseline → **looks exactly like a null result** | assert `named_ids ⊆ page_ids`; make the fallback counter a void condition |
| **NLI alone as a conflict verifier** | 33–93% false verifications; same-template/different-subject pairs score contradiction at 0.999 | require parsed subject+relation identity as a screen |
| **Fixing the stage that is not the bottleneck** | a large, real improvement in one component (ABTT: screen precision 5.3% → 51.3%) moves the end metric by exactly zero | measure where precision actually comes from before optimising a component; ours came from the regex screen and NLI, not from cosine |
| **Judging a detector by AUC** | AUC is dominated by the easy bulk and moved only +0.002 where recall-at-precision-1.000 moved 6.8× | report the statistic your operating point is selected on, not the one that is conventional |
| **Harm cap below the noise floor** | no intervention can pass, however good | set harm criteria above the measured A/A floor |
| **Pooled percentile across unlike subsets** | a threshold that describes no subset and is unreachable on one | fit and report per subset |
| **Chunk-granularity intervention** | helps a little, harms twice as much | act at the granularity of the conflict (facts) |
| **`git stash` in a shared checkout** | another agent's uncommitted work disappears | stage by explicit path; never use repo-global git commands |
| **Reading source without `encoding="utf-8"`** | invariant tests fail as decode errors on non-UTF-8 locales | always pass the encoding |

---

## 11. Definition of done

You have a defensible port when all of these are true:

- [ ] Baseline suite green **before** and **after** your changes
- [ ] Question strata computed and **all readouts stratified**
- [ ] The new model's **A/A floor measured**, and every claimed effect clears it
- [ ] Oracle ceiling measured **before** any policy work
- [ ] Operating point frozen on **detection metrics only**, committed **before**
      grading, with commit order provable
- [ ] Pre-registration committed **before** the held-out run, containing success
      criteria, harm classes, void conditions, a falsifiable side-prediction and
      the analysis code
- [ ] Exactly **one** confirmatory run; result reported **as it came out**,
      including void conditions and missed predictions
- [ ] Every deleted/edited item verified against ground truth (we report
      precision over every suppression: 735/735)
- [ ] Raw artifacts committed, not just summaries
- [ ] A written statement of **what may not be claimed** from your result

---

## 12. If you get stuck

- The system's behaviour is described in `HNAV_HOW_IT_WORKS.md`.
- Every past decision and its evidence is in `HNAV_FINAL_REPORT.md`.
- Hard invariants and the layering rules are in `CLAUDE.md`.
- Prior measurements, with raw JSON, are under `stage0_results/`.
- Two pre-registrations exist — one **withdrawn** (with its reasons) and one
  fired. Read both; the withdrawn one is the better lesson.

**And the single most important habit:** when a result surprises you, assume your
harness is wrong before you assume the finding is real. In this project that
instinct caught a precision bug, a truncation bug, three cache-invalidation bugs,
a page-selection mismatch and a false null — and every one of them would
otherwise have become a published number.
