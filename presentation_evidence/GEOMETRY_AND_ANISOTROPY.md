# Geometric filtering, anisotropy, and ABTT whitening

*What is actually applied to text entering the H-Nav pipeline, what the embedding
space's anisotropy was measured to be, and why the whitening machinery is built,
tested, armed once — and still switched off.*

Companion to `presentation_evidence/DETECTOR_MECHANICS.md` §2, which covers the
filter's decision logic. This document covers the *space* the filter operates in.

---

## 1. What is applied to incoming data

### 1.1 The embedding pipeline

Text entering H-Nav is turned into a vector by `HFEmbedder`
(`hnav/core/embedding.py:252-338`). Four steps, chosen to mirror the benchmark's
own retriever exactly:

| step | detail | code |
|---|---|---|
| tokenise + truncate | `max_length = 8192` (config default `embed_max_length`) | `embedding.py:326`, `config.py:60` |
| forward pass | `Qwen/Qwen3-Embedding-4B`, `float32`, 2560-d | `config.py`, `m1_geometry_calibration.json` |
| **mean pooling** over the attention mask | `(last * mask).sum(1) / mask.sum(1)` | `embedding.py:335-336` |
| **L2 normalisation** | `F.normalize(pooled, p=2, dim=1)` | `embedding.py:337` |

The module docstring states the constraint this exists to satisfy:

> `HFEmbedder`'s pooling mirrors `Qwen3Embedding4BEmbeddings` at
> `MemoryAgentBench/methods/embedding_retriever.py:58` — mean pooling over the
> attention mask, then L2 normalization — **so H-Nav's geometry is the geometry
> the benchmark itself retrieves in.**

Two consequences worth stating explicitly:

- **Every vector is a unit vector.** This is what makes `M Mᵀ` a cosine matrix and
  what makes the residual identity `r = √(1 − c²)` exact (`DETECTOR_MECHANICS.md`
  §2.3).
- **No centring, no whitening, no dimensionality reduction is applied.** The
  vectors that reach the cosine screen are the raw normalised mean-pooled
  outputs. This is the fact that makes anisotropy load-bearing.

Vectors are cached under `sha256(model|dtype|L{max_length}||text)`
(`cache_key`, `embedding.py:87-102`). `max_length` is in the namespace because it
changes the vector — the T12 defect, where 512-token truncation silently produced
different geometry, was born from its absence.

### 1.2 Then, and only then, the geometric filter

`ReadGate.decide` receives already-embedded records and applies, in order: the
pairwise cosine screen (`≥ 0.90`), union-find grouping, and the leave-one-out QR
span residual (`< 0.44`). No transformation of the vectors happens inside the
gate. See `DETECTOR_MECHANICS.md` §2 for the mechanics.

---

## 2. Anisotropy — and why it decides the threshold

**Anisotropy** is the empirical fact that contextual embeddings do not spread
over the unit sphere. They occupy a narrow cone, dominated by a few directions
shared by *every* input. The practical symptom: **two completely unrelated texts
have high cosine similarity.** For an isotropic space, random unit vectors in
2560 dimensions would have cosine ≈ 0 (concentrating with spread ≈ 1/√d ≈ 0.02).

This matters here for one reason: **the entire pipeline is thresholded on raw
cosine.** If unrelated facts sit at 0.60 rather than 0.00, then a threshold of
0.5 admits everything and the "similarity" screen filters nothing.

### 2.1 Measured: the primary arena

`hnav/stage0/m1_geometry_calibration.py` embedded, per subset, all *true*
conflict pairs and an equal-sized **random control** sample of non-conflicting
pairs. Source: `stage0_results/final/m1_geometry_calibration.json`, tabulated in
`presentation_evidence/data/item06_geometry_percentiles.csv`. All four subsets,
`Qwen3-Embedding-4B`, float32:

| subset | **control (unrelated) pairs** mean cos | p10 | p90 | conflict pairs mean cos | separation AUC |
|---|---|---|---|---|---|
| sh_6k | **0.6048** | 0.5312 | 0.6894 | 0.9547 | 1.0000 |
| sh_32k | **0.6062** | 0.5269 | 0.6846 | 0.9557 | 1.0000 |
| sh_64k | **0.6037** | 0.5187 | 0.6848 | 0.9557 | 1.0000 |
| sh_262k | **0.6125** | 0.5316 | 0.6934 | 0.9570 | 0.9999 |

**The anisotropy value is ≈ 0.604–0.613.** Two facts with nothing in common —
different subject, different relation, different object — sit at cosine 0.60, not
0.00. The nominal cosine range `[−1, 1]` is compressed to a working range of
roughly `[0.52, 1.00]`; the bottom 76% of the scale is never used.

### 2.2 Measured independently: the candidate-pair floor

A second, sharper view comes from the M1b grouping ablation
(`stage0_results/final/m1b_grouping_ablation.json`), which sweeps τ over every
kNN candidate pair. The τ at which the *first* pair falls below threshold is a
direct read of the distribution's floor:

| subset | candidate pairs | **all pairs still pass at τ =** | pairs surviving τ=0.90 |
|---|---|---|---|
| sh_6k | 3,144 | **0.58** | 190 (6.0%) |
| sh_32k | 16,770 | **0.61** | 1,363 (8.1%) |
| sh_64k | 33,755 | **0.63** | 3,201 (9.5%) |
| sh_262k | 135,875 | **0.65** | 24,372 (17.9%) |

Every one of 135,875 candidate pairs at sh_262k has cosine **≥ 0.65**. A
threshold below that value is not a filter — it is the identity function.

Note the floor **rises with store size** (0.58 → 0.65). This is the drift the
`TauPolicy` docstring warns about (`geometry.py:110-120`): in a larger store the
nearest neighbour of anything is closer, so *a constant threshold silently
tightens as the run proceeds*.

### 2.3 Measured on a second, non-synthetic corpus

The Cross-Episode control (`stage0_results/crossep/m5b_crossep_control.json`,
20,000 sampled pairs each, real dialogue chunks — not templated facts):

| population | mean cos | min | p05 | p95 | frac ≥ 0.95 |
|---|---|---|---|---|---|
| **across contexts** (unrelated conversations) | **0.7860** | 0.5260 | 0.6909 | 0.8653 | **0.0004** |
| within context | 0.9162 | 0.5978 | 0.8326 | 0.9713 | 0.2180 |

Unrelated chunks from *different conversations* average **0.786**. This corpus is
even more anisotropic than the synthetic fact corpus, which rules out the
template family as the cause — the anisotropy is a property of the encoder, not
of the data.

### 2.4 What this justifies

- **`cos_pair = 0.90` is not conservatism; it is the floor of the usable band.**
  At 0.85 precision is 0.436 (sh_6k) and 0.104 (sh_262k); at 0.90 it is 0.805 and
  0.282 (`m1b_grouping_ablation.json` PR curves).
- **Anisotropy does not destroy the signal here.** Conflict pairs sit at 0.955
  against a control mean of 0.604 — a 0.35 gap with *no overlap* between control
  p90 (0.689) and conflict p10 (0.913). Separation AUC is **1.0000** on three
  subsets and 0.9999 on sh_262k. The cone is narrow, but the two populations
  occupy different parts of it.
- **This is exactly why the repository forbids reusing the prior BFCL port's
  thresholds** (`CLAUDE.md`): a different encoder has a different cone, so a
  numeric threshold does not transfer.

---

## 3. ABTT whitening

### 3.1 What it does

**All-But-The-Top** (Mu & Viswanath) is the standard remedy for anisotropy:
remove the common mean and the few dominant directions that carry no
discriminative information but dominate cosine.
`ABTTWhitening` (`hnav/core/geometry.py:55-105`):

```
1.  μ = mean over the store's vectors
2.  centre:      X' = X − μ
3.  SVD:         X' = U Σ Vᵀ,  take the top D right singular vectors  C = V[:D]
4.  project out: v ↦ (v − μ) − ((v − μ) Cᵀ) C
5.  renormalise: v ↦ v / ‖v‖
```

Step 4 is `I − CᵀC` applied to the centred vector — an orthogonal projection onto
the complement of the dominant subspace. Defaults: `whiten_components = 3`,
`whiten_min_fit_n = 200` (`config.py:107-108`).

### 3.2 The refusal rule

`fit()` returns `refused = True` and falls back to raw cosine when the store has
fewer than `min_fit_n = 200` rows (`geometry.py:79-82`). The docstring:

> **Whitening must refuse to fit on a small store.** ABTT estimates a mean and a
> few principal directions; on the 6k subsets (455 facts) and on early
> CrossEp-Know contexts (<20 chunks) those estimates are noise.

`transform()` is a pass-through when unfitted, and `GeometrySignals` carries both
`whitened` and `whitening_refused` **so the fallback rate is reportable rather
than invisible** — a design rule worth transferring.

There is an architectural consequence: the read gate's candidate pool is capped
at **50 facts** (`select_pool`, `mab_adapter.py:321-344`). 50 < 200, so **ABTT
would refuse at gate time even if it were wired in.** Whitening is fittable at
corpus level (455 / 2,310 facts) but not at decision level.

### 3.3 Measured: does whitening actually help?

The A/B was run and committed — `stage1_calibration.json → provenance.abtt_ab`,
fitted on the **full fact matrix** per subset (`calibrate_read_policy.py:292-295`,
`whitening_fitted: true` in both):

| subset | pairs | true supersession | AUC cos **raw** | AUC cos **whitened** | AUC r_min **raw** | AUC r_min **whitened** |
|---|---|---|---|---|---|---|
| sh_6k | 4,501 | 1,437 | 0.9361 | **0.9546** (+0.019) | **0.3523** | **0.4513** (+0.099) |
| sh_32k | 24,232 | 1,282 | 0.9876 | **0.9908** (+0.003) | 0.6288 | **0.6946** (+0.066) |

Three readings, and the second is the important one:

**(a) Whitening helps cosine, marginally.** +0.019 AUC at sh_6k, +0.003 at
sh_32k. Real but small, and shrinking as the store grows — consistent with §2.1,
where the anisotropy offset is nearly constant across subsets and therefore
largely cancels in a *pairwise ranking* comparison. Anisotropy shifts all cosines
up together; AUC is invariant to a monotone shift.

**(b) The `r_min` signal is anti-predictive at sh_6k — AUC 0.3523, well below
0.5** — and whitening lifts it only to 0.4513, *still below chance*. Within the
loose-screen population, true supersession pairs tend to have **higher**
leave-one-out residuals than non-supersession pairs. This is independent
corroboration of `DETECTOR_MECHANICS.md` §2.3: the residual screen was frozen at
a pass-through value (`r_min = 0.44 ≈ √(1−0.90²)`) not by oversight but because
tightening it would have removed true positives. The selection rule's outcome and
this AUC agree.

**(c) None of it feeds a decision.** The artifact's own `note`:

> `"logged only - no decision consumes whitened values (V1 A/B evidence)."`

The separation is structural, not merely a disabled flag: **`GeometryModule` — the
only class that can hold an `ABTTWhitening` — is never constructed in
`hnav/adapters/mab_adapter.py` at all.** The primary arena's read path passes
record vectors straight to `ReadGate.decide`. Grep confirms every construction
site is elsewhere: `stage0/m3_headroom.py:150` (Stage-0 write-side measurement),
`stage0/crossep_m5_write_headroom.py:265` (secondary arena, commented *"whitening:
refused below 200 anyway"*), and `adapters/clbench_adapter.py:129` (secondary
arena). The shipped operating point (`stage1_operating_point.json`) contains no
whitening parameter. **Every number in the thesis was produced on raw, unwhitened
cosine.**

> **Comparability caveat.** The §3.3 AUCs and the §2.1 AUC of 1.0000 are *not* the
> same measurement. §2.1 separates conflict pairs from **random** pairs; §3.3
> separates true supersession from other pairs **that already passed the loose
> cosine screen** — a far harder, pre-filtered population. Do not present 1.0000
> and 0.9361 as a contradiction or as a degradation.

### 3.4 Why it was not turned on — and what happened when it was

The four reasons below were the *a priori* case for leaving ABTT off. Three of
them have since been tested directly (§3.5); reason 2 turned out to be right for
the wrong reason, and reason 1 turned out to be avoidable.

1. ~~**It cannot run where the decision is made**~~ (§3.2): the 50-fact pool is
   below `min_fit_n`. **Superseded.** This conflates the *fit basis* with the
   *decision pool*. `MABAdapter.facts` holds the whole store (455–18,332 facts)
   and `select_pool` only selects 50 from it, so a per-store fit is available
   online; and fitting **offline**, once, removes the question entirely — a
   pre-fitted whitener has nothing left to estimate at decision time.
2. **The gain is small and the pipeline does not need it.** Half right. The gain
   on *detection quality* is not small — recall at precision 1.000 rises 6.8× on
   sh_6k and 40× on sh_32k (§3.5). But the conclusion holds: the pipeline does
   not need it, because it buys precision from the regex `pair_filter` and NLI
   rather than from cosine.
3. **Turning it on would change every threshold.** Confirmed, and paid: the
   whitened operating point was re-fitted from scratch on the calibration split
   (`cos_pair` 0.90 → 0.30, `r_min` 0.44 → 0.9539).
4. **It is kept, tested and logged** so the decision is reversible and the
   evidence for reversing it is on disk. This is what made §3.5 possible.

### 3.5 Measured: ABTT armed before the cosine screen

Full campaign in `stage0_results/abtt/` (`ABTT_REPORT.md`, pre-registered at
commit `dd4439b` before grading). ABTT was fitted **offline** on the calibration
split (`frozen_global`, D = 128, fingerprint `3fdacc1fcc479efb…`), applied to the
fact–fact geometry the gate decides on, thresholds re-fitted on detection quality
alone, and fired once on held-out `sh_64k`.

**What it changed — a lot:**

| property | raw | ABTT |
|---|---|---|
| anisotropy (unrelated-pair mean cos) | 0.6024 / 0.6026 | ≈ 0.000 |
| candidate-pair floor | 0.5815 / 0.6130 | 0.083 / 0.081 |
| screen precision at equal recall (sh_32k) | 5.3% | **51.3%** |
| recall at precision 1.000 (sh_6k / sh_32k) | 0.0750 / 0.0072 | **0.5125 / 0.2910** |

Mean-centering alone (D = 0) removes essentially all of the anisotropy; the
principal directions buy the ranking gains on top.

**What it changed in accuracy — nothing:**

| stratum | A0 native | raw | ABTT |
|---|---|---|---|
| overall | 45/100 | 64/100 | 64/100 |
| conflicted | 17/66 | 37/66 | 37/66 |

Paired on the conflicted stratum: **net = 0, not one question differs**
(95% CI [0, 0], McNemar p = 1), at equal harm and equal token cost. The A/A floor
was a true 0/0 and the raw arm reproduced the committed campaign with 500/500
identical outcomes and zero differing model outputs, so this is an exact null
rather than a measurement failure.

It is not, however, an inert null: the two arms produced **different suppression
plans on 12 of 100 questions** and differed by 16 suppressed facts, changing the
model's output text exactly once and its correctness never. The facts the two
geometries disagree about are facts no question depended on.

**The conclusion, stated carefully.** ABTT improves the stage that was not the
bottleneck. Its large gains live in the regime where cosine must carry precision
*alone*; here the parsed subject+relation screen already supplies that precision
for free. This bounds the mechanism rather than dismissing it — the untested and
interesting case is an arena with no parseable template, where that screen does
not exist.

One further measured caveat: whitening the **query** vector as well as the facts
is actively harmful — reachable true-supersession pairs fall 1,443 → 1,048 on
sh_6k (−27%) because `select_pool` then builds a worse pool. ABTT helps
symmetric fact–fact comparison and hurts asymmetric question–fact retrieval.

---

## 4. Summary

| question | answer | evidence |
|---|---|---|
| What is applied to incoming text? | tokenise (8192) → mean-pool over mask → **L2 normalise**. No centring, no whitening. | `embedding.py:326-337` |
| Is the space anisotropic? | **Yes, strongly.** | below |
| By how much, primary arena? | unrelated pairs mean cos **0.604–0.613**; no candidate pair below **0.58–0.65** | `m1_geometry_calibration.json`; `m1b_grouping_ablation.json` |
| By how much, secondary arena? | across-context mean cos **0.786** (min 0.526) | `m5b_crossep_control.json` |
| Does it break the filter? | No — conflict 0.955 vs control 0.604, separation AUC **1.0000** | `item06_geometry_percentiles.csv` |
| What does it force? | a high absolute threshold (`cos_pair = 0.90`) and an adaptive `tau_t`; no threshold transfer between encoders | `read_gate.py`; `geometry.py:110-120` |
| Is ABTT implemented? | Yes, tested, with an explicit refusal below 200 rows | `geometry.py:54-113` |
| Does it help detection? | **Yes, substantially.** Recall at precision 1.000: 0.0750 → 0.5125 (sh_6k), 0.0072 → 0.2910 (sh_32k) | `stage0_results/abtt/G1_GATE_REPORT.md` |
| Does it help accuracy? | **No.** sh_64k conflicted 37/66 in both arms; net 0, not one question differs | `stage0_results/abtt/ABTT_REPORT.md` |
| Is it used? | **No** — and now for a measured reason, not a presumed one: armed and fired on sh_64k, it changed no answer. | `ABTT_REPORT.md`; `stage1_operating_point.json` unchanged |
