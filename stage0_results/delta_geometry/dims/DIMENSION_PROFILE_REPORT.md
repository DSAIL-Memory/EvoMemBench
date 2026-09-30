# Can conflict be seen in all dimensions at once? — M7b per-coordinate profile

**Question.** Cosine similarity reduces two 2560-dimensional vectors to one
number, `Σ aᵢbᵢ`. Two pairs with the same cosine can differ in every
coordinate. If we look at the difference vector `Δ = v_later − v_earlier`
coordinate by coordinate — its direction and size in every dimension — is
there a conflict pattern that the single cosine number throws away?

**Answer in four lines.** Yes, and it is measurable: a coordinate *sign*
pattern fitted on half the conflict pairs picks the held-out conflict over its
cosine-matched control **~83%** of the time (sh_64k, raw), and still **85%** on
the pairs where the control's cosine is *higher* than the conflict's — where
cosine is wrong by definition. But the information is not in *which*
coordinates are active (the energy profiles of conflict and control Δ are the
same, r = 0.895); it is in the **signs**, spread over hundreds of coordinates —
i.e. a direction. What that direction encodes is *which slot of the fact
changed*: the high-cosine competitors of a conflict are overwhelmingly "same
relation, different subject", and Δ points differently when the subject
changes than when the object changes. Cosine measures *how much* changed; the
coordinates say *what* changed. ABTT weakens all of this.

Exploratory: nothing is trained beyond a ranking of coordinates and a sign per
coordinate, fitted on one half and scored on the other. No threshold is derived.

---

## 1. Method

**Pairs and controls** are M7's (`DELTA_GEOMETRY_REPORT.md` §1): conflict =
same relation, same subject, different object, oriented earlier → later serial;
the decisive control is `cos_matched`, a 1:1 nearest-cosine non-conflict partner
within 0.02 drawn from the exhaustive pair space, and every comparison below is
**paired** — conflict *j* against its own control *j*.

**Layer 1 — the literal request (fig A).** Ten conflict pairs and their ten
matched controls, unit Δ drawn coordinate by coordinate as a 20 × 2560 heatmap,
the two ten-pair means, their difference, and a histogram of all coordinate
values. This is a picture. Ten vectors in 2560 dimensions always look patterned.

**Layer 2 — the population (figs B–E).** Per coordinate *i*, over all pairs:

| statistic | what it asks | null |
|---|---|---|
| energy `E[Δ̂ᵢ²]` | is coordinate *i* more *active* for conflicts? | equal profiles |
| sign consistency `|mean sign(Δ̂ᵢ)|` | do conflicts agree on the *direction* of coordinate *i*? | `0.8/√m` |
| `z` of the signed mean | same, as a t-statistic | \|z\| < 4 |

Per pair: concentration of Δ̂ over coordinates — effective number of
coordinates `1/ΣΔ̂ᵢ⁴`, energy in the top-10, `‖Δ̂‖₁` — none of which is a
function of the cosine, tested with a paired sign-flip permutation.

**Held-out test (fig E).** On half the conflicts, rank coordinates by |z| and
record the consensus sign; on the other half, score `s(Δ̂) = Σ_{top-k} signᵢ·Δ̂ᵢ`
and ask how often the held-out conflict outscores its own matched control.
Repeated over 20 random splits, and over 20 **relation-disjoint** splits (no
relation template appears on both sides), so a coordinate that merely encodes a
template cannot pass. A second family ranks coordinates by *energy* contrast and
scores by energy in them — the "which dimensions are active" hypothesis, tested
the same way.

**The comparison that had to be corrected.** I first wrote that cosine is "0.5
by construction" inside the matched set. It is not: non-conflict pairs are scarce
at high cosine, so the matched control lands *below* its conflict by ~0.012 on
average, and cosine still ranks the conflict first ~89–92% of the time. So every
held-out pair is also binned by its residual gap; the bin at **gap ≤ 0** — the
control is *more* similar than the conflict — is where cosine is wrong, and it is
the clean test.

Vectors and whitening are the campaign's own (`Qwen3-Embedding-4B` fp32
L8192; `abtt_whitening_D128.json`, fingerprint `3fdacc1f…`, fitted on
sh_6k + sh_32k only).

---

## 2. The ten pairs (fig A)

Nothing. The heatmap rows are indistinguishable noise for conflict and control
alike; the ten-pair means look like noise; the coordinate-value histograms lie
on top of each other (fig A, bottom). The ten conflict pairs on sh_6k are
ordinary object swaps — *Hines Ward plays the position of wide receiver →
cornerback*, *CEO of Apple is Tim Cook → Vijay Mallya* — and their matched
controls are things like *head of state in United Kingdom is Elizabeth II →
head of state in Soviet Union is Elizabeth II*.

That last example is the whole story, and it took the population to see it.

---

## 3. Which dimensions? — none in particular (fig B)

The **energy** profiles — how much of a unit Δ each coordinate carries, averaged
over pairs — are the same for conflicts and their matched controls:

| sh_64k, raw, n = 848 each | conflict | matched control |
|---|---|---|
| effective number of coordinates in the mean profile | 2227 / 2560 | 2322 / 2560 |
| energy share of the top 64 coordinates | 0.065 | 0.058 |
| Pearson *r* between the two profiles | **0.895** | |

There is no set of "conflict dimensions" that light up. Both kinds of Δ spread
over essentially all coordinates, and they spread over the *same* coordinates.
The per-pair concentration statistics agree: conflict Δ are marginally *more*
concentrated on sh_64k raw (effective coordinates 747 vs 780, paired z = −8.5;
top-10 share 0.052 vs 0.047, z = +8.9), but the effect is 4% and vanishes under
ABTT (z = −1.8) and at sh_6k (z = −0.1). Not a usable signal.

The **sign** profiles are where conflicts and controls part:

| sh_64k, raw, n = 848 each | conflict | matched control |
|---|---|---|
| coordinates with sign consistency above 3σ | **975** | 111 |
| coordinates with \|z\| > 4 | **1359** (all 1687) | 37 |
| strongest coordinate: \|mean sign\| | 0.491 (74% agree) | 0.172 (59%) |

Nearly a thousand coordinates carry a sign that conflict Δ agree on beyond 3σ,
against a hundred for the controls. On sh_6k with n = 50 the same ordering
holds (65 vs 12; strongest coordinate 0.76 vs 0.56). That is the per-coordinate
face of M7's shared direction: a direction *is* a pattern of signs across
coordinates. What the coordinate view adds is that the direction is **broad** —
it is not carried by a handful of axes (fig B, col 2: the curve decays slowly
over hundreds of coordinates).

---

## 4. Does it hold on pairs cosine cannot separate? — yes (fig E)

Held-out paired accuracy, sign pattern, random splits, sh_64k raw:

| coordinates kept *k* | 1 | 4 | 16 | 64 | 256 | 1024 | all 2560 |
|---|---|---|---|---|---|---|---|
| paired accuracy | 0.64 | 0.70 | 0.77 | 0.80 | 0.83 | 0.83 | 0.83 |
| cosine, same pairs | | | | | | | 0.89 |

Sixteen coordinates already reach 0.77; the curve saturates by ~256. The
relation-disjoint split lands slightly lower (0.74 / 0.80 / 0.80 at
k = 16 / 256 / all) — so the pattern is **mostly not a relation-template
artefact**, unlike the bulk of M7's directional effect.

Now the clean comparison, binned by the residual cosine gap (sh_64k, raw,
k = 256, pair-evaluations over 20 splits):

| residual gap  conflict − control | n | sign-pattern accuracy | what cosine says |
|---|---|---|---|
| **≤ 0 (control more similar)** | 958 | **0.847** | wrong (0.00) |
| (0, 0.005] | 1371 | 0.789 | right, barely |
| (0.005, 0.01] | 823 | 0.820 | right |
| (0.01, 0.02] | 5372 | 0.830 | right |

The pattern's accuracy is **flat across the gap** — it does not care what cosine
thinks. On the 958 evaluations where cosine ranks the control above the
conflict, the coordinate pattern still picks the conflict 85% of the time; on
the relation-disjoint split, 81%. That is information cosine does not contain.

Energy-based selection (which coordinates are *active*) reaches the same
plateau only by k ≈ 1024 and is undefined at k = d (total energy is 1 for every
unit vector); it is the weaker family, consistent with §3.

---

## 5. What the direction encodes: which slot changed

The non-conflict pairs that reach conflict-level cosine are not a random
sample. Of the 65,782 non-conflict pairs above the conflict cosine floor on
sh_64k:

| kind | share |
|---|---|
| same relation, **different subject**, different object | 53.3% |
| same relation, **different subject**, same object | 37.2% |
| different relation | 9.5% |

**90% of a conflict's high-cosine competitors are the mirror image of a
conflict**: the template and often the object are shared, and the *subject* is
what differs. A conflict changes the object slot; its look-alikes change the
subject slot. Both are one-slot edits of the same template, so both have
cosine ≈ 0.95 — cosine measures how much text changed, and one slot is one
slot. But `v(…Cook) − v(…Mallya)` and `v(UK…) − v(Soviet Union…)` point in
different directions in the embedding, and the coordinate sign pattern reads
that difference.

This reframes the result: the "conflict direction" is, operationally, an
**object-slot-change direction as opposed to a subject-slot-change direction**.
It is a real geometric distinction, it is exactly the distinction the read-time
detector has to make after the cosine screen, and it is what the regex
`pair_filter` currently makes with string matching on parsed subjects.

---

## 6. Raw vs ABTT

ABTT weakens every per-coordinate statistic, as it weakened M7's:

| sh_64k | raw | ABTT D=128 |
|---|---|---|
| coordinates with conflict sign consistency > 3σ (n-matched) | 975 | 363 |
| held-out paired acc, k = 16 / 256 / all | 0.77 / 0.83 / 0.83 | 0.69 / 0.77 / 0.78 |
| … relation-disjoint | 0.74 / 0.80 / 0.80 | 0.68 / 0.74 / 0.75 |
| gap ≤ 0 bin, k = 256 | 0.847 | 0.765 |
| concentration contrast (paired z) | −8.5 | −1.8 |

Two mechanisms, both visible in the data:

1. The difference operator already cancels the mean ABTT subtracts, so
   whitening cannot *add* Δ structure; what the principal-direction removal does
   is delete 128 directions, and some of the sign pattern lived there.
2. **ABTT changes who the competitors are.** Under whitening the matched
   controls are no longer "same relation, different subject" (133 of 599) but
   "different relation, same subject" (242) and "different everything" (208):
   removing common directions collapses the template similarity, so
   same-template pairs fall out of the conflict cosine range and
   subject-sharing pairs take their place. The slot-change distinction of §5
   is then being asked to separate a *different* kind of pair, and does it
   less well.

The anatomy of the cosine itself (fig D) shows why the raw space is more
informative here, not less: the 0.605 that unrelated facts share is spread
over the coordinates — the top 16 supply 16%, the top 256 supply 54% — it is
not a rogue-dimension effect; and conflict-pair similarity is spread even
wider (top 256: 39%). There is no small set of coordinates to remove that would
clean the cosine without also removing signal.

---

## 7. sh_6k vs sh_64k

Same ordering everywhere; sh_6k is simply underpowered — 50 matched pairs, so
each held-out half has ~25, and the gap-≤ 0 bin has ~2 pairs per split (39
evaluations in 20 splits). Its numbers (0.85 at k = 256; 0.85 in the gap-≤ 0
bin) agree with sh_64k but should be read as consistent, not as confirmation.
sh_6k is also a strict subset of sh_64k (M7 §7).

---

## 8. What is and is not supported

**Supported.**
- Conflict Δ carry a consistent *sign* pattern across ~1000 coordinates that
  cosine-matched controls do not; it generalises to held-out pairs and to
  held-out relations.
- That pattern separates conflicts from their look-alikes **where cosine
  cannot** (85% on the gap-≤ 0 bin) and its accuracy is independent of the
  residual cosine gap. Looking at all coordinates does carry information the
  single cosine number discards.
- The distinction it draws is *object-slot change vs subject-slot change*, the
  exact post-screen decision the detector needs.

**Not supported.**
- "Conflict dimensions." No coordinates are preferentially active; the energy
  profiles are identical. The information is directional, and a direction is
  not a subset of axes.
- A detector. 0.83 paired accuracy on a *matched* set is a statement about
  information content, not about a deployable threshold; on the full 39 : 1
  non-conflict population this was not measured and must not be extrapolated.
  The shipped pipeline's `pair_filter` + NLI already reach precision 1.000 on
  this decision.
- Anything about the ten pictured pairs. Fig A is a demonstration that ten
  examples cannot show this; the population can.
- Generality beyond this encoder and this templated benchmark. The
  slot-change reading depends on facts being one-slot edits of a template.

---

## 9. Figures and artifacts

| file | shows |
|---|---|
| `figA_ten_pairs_sh_6k.png`, `figA_ten_pairs_sh_64k.png` | the ten pairs, coordinate by coordinate, raw and ABTT |
| `figB_population_profiles.png` | per-coordinate energy (identical), sign consistency and \|z\| (not), n-matched |
| `figC_concentration.png` | per-pair spread over coordinates, paired with permutation z |
| `figD_cosine_anatomy.png` | which coordinates the cosine itself comes from |
| `figE_heldout_coordinates.png` | held-out separation vs coordinates kept; random and relation-disjoint splits |
| `m7b_dimension_profile.json` | every statistic including the gap-stratified table |

```bash
python hnav/stage0/m7b_dimension_profile.py --subsets sh_6k sh_64k   # ~55 s, CPU, cached vectors
```
