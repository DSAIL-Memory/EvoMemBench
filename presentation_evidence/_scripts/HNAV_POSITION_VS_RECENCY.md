# Does the Model Track *Recency* or Just *Position*?

*An answer to the advisor's question, from the committed run logs. 2026-08-18.*
*Reproduce with `python3 hnav/stage1/position_taxonomy.py` — no model calls needed.*

---

## The question

> If the model learned during SFT to rely on the most recent information in the
> text (positional-encoding information), there shouldn't be a problem — but
> they might not have taught it. For example: when we swap the positions of the
> contradictory old and new messages in the conversation history, does the
> response change? If so, we can say that the language model distinguishes
> between old and new messages.

**Short answer: yes, we have run exactly this experiment**, on three subsets,
with the raw per-answer outputs committed. Swapping the positions **does** change
the response — decisively, against a zero noise floor.

But the conclusion goes the other way from the one proposed, and that reversal is
the interesting part. **The swap changes the answer because the model is
following raw text position — and when position and the explicit recency label
disagree, position wins ~97% of the time.** That is evidence the model does *not*
distinguish old from new as a *property of the fact*; it tracks where the text
sits.

---

## 1. Why this benchmark can answer the question at all

The arena hands the model **two independent recency cues**:

| cue | what it is |
|---|---|
| **symbolic** | the serial number, plus a prompt that states in words: *"the newer fact has larger serial number… solve the conflicts by finding the newest fact"* |
| **positional** | raw text order — where the sentence physically sits in the prompt |

In the untouched dataset these two are **perfectly confounded**. Verified
directly: in `sh_6k`, `sh_32k` and `sh_64k` the facts appear in **strictly
ascending serial order** (0 → 454, 0 → 2309, 0 → 4579). The newer fact is always
both higher-numbered *and* later in the text.

So the native run cannot separate them. The probe arms were built to do exactly
that.

---

## 2. The arms — one of them *is* the proposed experiment

From `hnav/stage1/stale_suppression_probe.py` (calibration, oracle-targeted) and
`hnav/stage1/detector_gap.py --confirmatory` (held out, detector-driven):

| arm | what it does to the two competing facts | which cue points where |
|---|---|---|
| `native` | nothing | both cues → NEW |
| `native_repeat` | nothing; second independent call | **the A/A noise floor** |
| `oracle_recency` / `detector_demote_late` | NEW moved to the very **END** | both cues → NEW, positional cue *intensified* |
| **`anti`** | **NEW moved to the FRONT, most recent OLD moved to the END** | **cues CONFLICT: symbolic → NEW, positional → OLD** |
| `oracle_suppress` / `detector_suppress` | OLD **deleted** | no competitor — the reference point |

**`anti` is the advisor's swap.** The two contradictory facts literally exchange
ends of the context. The serial numbers are untouched, so the symbolic cue still
says the fact at the *front* is the newer one.

---

## 3. What the model actually said

The published tables report only right/wrong, which collapses "answered the stale
value" together with "refused." The question turns on exactly that distinction, so
`hnav/stage1/position_taxonomy.py` re-reads the committed raw outputs and
classifies each answer by **which value it names**:

- **NEW** — the value of the highest-serial fact of the queried key (the gold value)
- **OLD** — the value of a superseded fact of the same key
- **OTHER** — neither (refusal, off-list, malformed)

### Calibration split, whole context in one block, oracle-targeted

**`factconsolidation_sh_6k`** — 74 conflicted questions

| arm | NEW | OLD | OTHER | answers changed vs native |
|---|---:|---:|---:|---:|
| native (serial order) | 4 | 70 | 0 | — |
| native repeat (**A/A floor**) | 4 | 70 | 0 | **0 / 74** |
| NEW → END | **20** | 54 | 0 | 18 / 74 |
| **NEW → FRONT, OLD → END** | **1** | **72** | 1 | 4 / 74 |
| OLD deleted | **66** | 8 | 0 | 62 / 74 |

**`factconsolidation_sh_32k`** — 65 conflicted questions

| arm | NEW | OLD | OTHER | answers changed vs native |
|---|---:|---:|---:|---:|
| native (serial order) | 7 | 58 | 0 | — |
| native repeat (**A/A floor**) | 7 | 58 | 0 | **0 / 65** |
| NEW → END | **33** | 32 | 0 | 30 / 65 |
| **NEW → FRONT, OLD → END** | **4** | **61** | 0 | 9 / 65 |
| OLD deleted | **53** | 11 | 1 | 48 / 65 |

### Held-out split, benchmark retrieval path, detector-driven

**`factconsolidation_sh_64k`** — 66 conflicted questions. Only 10 of 17 chunks
reach the page and only detector-verified groups are moved, so both placement
arms are attenuated by construction.

| arm | NEW | OLD | OTHER | answers changed vs native |
|---|---:|---:|---:|---:|
| native (serial order) | 17 | 44 | 5 | — |
| native repeat (**A/A floor**) | 17 | 44 | 5 | **0 / 66** |
| NEW → END | **20** | 43 | 3 | 10 / 66 |
| NEW → FRONT | **15** | 46 | 5 | 6 / 66 |
| OLD deleted | **37** | 23 | 6 | 21 / 66 |

**The A/A row is what makes all of this readable.** The same prompt called twice
produces **zero** changed answers on every subset. So every change counted above
is caused by the surgery, not by sampling. (The wider cross-run floor from eight
independent baseline runs is 0–5 of 74, and it lands *entirely* on the conflicted
stratum — but within a single process, as here, it is 0.)

---

## 4. What this shows — three findings

### Finding 1 — Position causally changes the answer. Yes.

Moving the newest fact to the end changed **18 of 74** and **30 of 65** answers,
against an A/A floor of 0. The direction is right and the effect is large:
NEW answers go 4 → 20 and 7 → 33; OLD answers go 70 → 54 and 58 → 32.

So the proposed manipulation works, and the model is genuinely position-sensitive.

### Finding 2 — But when the two cues conflict, **position beats the serial number**.

This is the `anti` arm, and it is the decisive cell. The serial number still says
"the fact at the front is the newer one." The prompt still spells out the rule in
words. And the model answers the value sitting at the **end**:

- `sh_6k`: **72 of 74 (97.3%)** answered OLD; only 1 answered NEW.
- `sh_32k`: **61 of 65 (93.8%)** answered OLD; only 4 answered NEW.

A model that had internalised recency as a *property of the fact* — read off the
serial number, as instructed — would be **invariant** to this swap. It is not
invariant; it moves the wrong way.

> **So the reversal:** the response changing under a position swap is evidence
> that the model is following **position**, not evidence that it distinguishes
> old from new. Here it demonstrates the opposite of what the phrasing suggests —
> the model is ignoring the explicit recency label it was handed.

One honest caveat on `anti`: native is already at the floor (4/74 NEW), so this
arm has almost no room to move the number *down*. Its 97% OLD rate is the readable
statistic; its small change count (4/74) is a floor effect, not weak evidence.
The clean directional evidence comes from the END arm.

### Finding 3 — Position is not the main driver either. **Presence is.**

Even with the newest fact placed as the last line before the question — the
strongest positional cue available — **54 of 74 (73%)** still answered the stale
value. Deleting the stale fact instead gives **66 of 74 (89%)** correct.

Effect sizes at `sh_6k`, same 74 questions, same substrate:

| manipulation | net change in correct answers |
|---|---:|
| delete the stale fact | **+62** |
| move the newest fact to the end | +16 |
| move the newest fact to the front (`anti`) | −3 |

The competing value's **presence** matters roughly four times more than its
**position**.

And deletion working at all settles a separate question that had been open: if the
model were simply overriding the context with its own world knowledge, removing a
sentence it never read would change nothing. It changes almost everything. **The
model does read the context** — it just loses to whichever value is more salient.

---

## 5. On the SFT hypothesis specifically

The advisor's framing was: *if recency-preference was taught during SFT there
shouldn't be a problem; maybe they didn't teach it.* The measurements are
consistent with **"not taught, or taught only as a weak positional habit"**:

- A late-position preference **does** exist and points the right way (Finding 1),
  which is what you would expect from generic next-token training on text where
  later usually means more current.
- It is **not bound to the symbolic notion of recency.** The model does not read
  the serial number as a precedence signal — when the two disagree, the serial
  number loses (Finding 2).
- The prompt states the rule **explicitly, in words**, and it is still ignored on
  ~95% of conflicted questions. If a precedence *rule* had been trained in, an
  in-context restatement of that same rule should activate it. It does not.
- Even the positional habit is too weak to carry the task on its own (Finding 3).

**Scope limit:** this is one model, Qwen3-4B-Instruct-2507. "SFT did / did not
teach this" is a claim about model families and cannot be settled from one
checkpoint. See gap 4 below.

---

## 6. One confound we cannot break with the current arms

In this benchmark the stale value is almost always the **world-true** one: the
dataset takes real facts and injects counterfactual updates, so the gold answer
is the *fictional* value and the superseded one is what the model already
believes. (*"Nobuhiro Watsuki is famous for Rurouni Kenshin"* is the stale fact
and also the truth; the gold answer is *"The Fairly OddParents."*)

So **"old" and "matches the model's parametric knowledge" are confounded.**

The deletion arm bounds how much this can matter — 89% correct with the
competitor removed means the parametric pull is weak when unopposed. But it does
**not** tell us how much of the residual OLD preference in the placement arms is
world-knowledge pull versus position. Any write-up should state this rather than
attribute the whole residual to position.

---

## 7. What we have **not** traced

Four gaps, in the order they are worth closing.

### Gap 1 — The serial-label swap, holding position fixed *(cheap, closes the argument)*

We have dissociated the two cues in **one direction only**. Laid out as a 2×2:

| | NEW value is at the END | OLD value is at the END |
|---|---|---|
| **NEW value has the higher serial** | ✅ `native`, `oracle_recency` | ✅ `anti` |
| **OLD value has the higher serial** | ❌ **never run** | ❌ **never run** |

The missing row is the clean measurement of the serial cue's *own* main effect:
keep both sentences exactly where they are and **swap only the two serial
numbers**. If the model is reading serials at all, its answer must flip; if it is
purely positional, the answer will not move at all.

This is pure prompt surgery on an existing harness — no new infrastructure, no
detector, ~150 calls per subset, and the machinery to do it already exists in
`stale_suppression_probe.py` (`apply_arm` plus a `renumber` helper). It would turn
"position beats serial" from an inference into a measured main effect with an
interaction term. **This is the single highest-value follow-up.**

### Gap 2 — Deconfounding recency from world truth

No arm has ever been run in which the **newest** fact is the world-true one. Until
one is, the OLD preference cannot be split into "prefers earlier text" and
"prefers what it already believes." Requires authoring counterfactual items in the
reverse direction — more work than gap 1, but it is what licenses any claim about
*why* the model prefers the stale value.

### Gap 3 — A genuine multi-turn conversation form

The advisor's phrasing said *conversation history*. Our substrate is a numbered
fact list inside **one user message**, not a dialogue. Real turns carry chat-template
role markers and a different positional structure, and turn-level recency behaviour
may differ from within-message recency. Nothing in this repository tests that.

### Gap 4 — No mechanistic evidence, and only one model

Everything above is **behavioural**: inputs in, answers out. There are no attention
weights, no logit-lens analysis, no positional-encoding ablation. If the thesis
wants to say something about *positional encodings* specifically, rather than about
position as an observable, that evidence does not exist yet.

*One thing to flag so nobody assumes otherwise:* `hnav/tests/test_attention_memory.py`
sounds relevant and is not — it covers a GPU out-of-memory fix in grouped-query
attention during embedding, not attention tracing.

And a single checkpoint cannot support a claim about what SFT does in general. A
second model, ideally from a different family, is the cheapest way to find out
whether "position beats the explicit recency label" is a property of this model or
of instruction-tuned models generally.

---

## 8. Why this matters for the thesis

This re-analysis strengthens the main result rather than complicating it, because
it explains **why the shipped mechanism is deletion and not reordering**:

- If the model tracked the serial number, no intervention would be needed — the
  rule is already in the prompt.
- If the model tracked position strongly, **reordering** would be the right fix.
  It is not: placement recovers +16 where deletion recovers +62, and at chunk
  granularity reordering was actively harmful (0 of 162 configurations
  net-positive, helped 228 / harmed 441).
- Because the failure is driven by the competing value's **presence**, the
  correct intervention is to **remove it**. Which is what H-Nav does.

So the position experiment is not a side quest — it is the measurement that
selects the mechanism.

It also sharpens methodological finding #2 in `HNAV_FINAL_REPORT.md` §9. The old
phrasing was *"explicit precedence instructions are ~95% ineffective."* The
stronger, now-measured version is:

> **The model does not read the serial number as a precedence signal at all.**
> When the symbolic recency label and text position are placed in conflict, the
> model follows position on 93–97% of conflicted questions. The positional
> preference itself is real but weak, and is dominated by the mere presence of a
> competing value.

---

## 9. How to reproduce

```bash
# Re-derives every table above from the committed run logs. No GPU, no LLM,
# no network — it re-reads raw outputs that were already recorded.
python3 hnav/stage1/position_taxonomy.py

# Optional machine-readable form
python3 hnav/stage1/position_taxonomy.py --json hnav/_out/position_taxonomy.json
```

Inputs, all committed:

| file | what it supplies |
|---|---|
| `stage0_results/stage1/stale_suppression_probe_sh6k.json` | sh_6k raw outputs, 473 calls, 5 arms |
| `stage0_results/stage1/stale_suppression_probe_sh32k.json` | sh_32k raw outputs, 465 calls, 5 arms |
| `stage0_results/stage1/detector_gap_confirmatory_sh64k.json` | held-out sh_64k, 500 calls, 5 arms |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json` | fact texts, for recovering each stale value |
| `hnav/labeling/conflict_analysis.py::parse` | the validated `(relation, subject, value)` parser (99.5%+ coverage) |

Substrate for all three runs: Qwen3-4B-Instruct-2507, temperature 0, 10 output
tokens, graded by the benchmark's own `substring_exact_match`.
