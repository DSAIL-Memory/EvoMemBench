# Do supersession updates have a geometry? — M7 difference-vector analysis

**Question.** For a supersession pair (two facts sharing a `(relation, subject)`
key, disagreeing about the object, the later serial superseding the earlier),
take the directed difference

```
Δ = v_later − v_earlier
```

Do these Δ occupy a characteristic **region**, **magnitude range** or
**direction** of the embedding space, compared with non-conflict pairs matched
on the things that would otherwise explain the difference? Exploratory only —
**no classifier is trained and no threshold is derived**, which is why running
this on held-out `sh_64k` is safe.

**Answer in three lines.** Yes to magnitude, but it is one statistic, not two,
and it does not separate conflicts from non-conflicts. Yes to direction: conflict
Δ share a genuine *signed* direction that survives cosine matching — the single
strongest result here. But that direction is mostly **one direction per relation
template**, so it is a property of this benchmark's phrasing at least as much as
of factual updating. ABTT whitening makes all of it weaker, never stronger,
exactly as the algebra predicts.

---

## 1. Method

**Pairs.** Conflict pairs come from `conflict_analysis.parse` (the validated
99.5%+ parser, imported not re-derived). Every group has exactly two facts in
these subsets, so each conflicted key yields one pair, oriented
earlier-serial → later-serial.

**Controls.** A conflict pair is *same relation, same subject, different
object*. Each control varies one factor so a difference can be attributed:

| control | definition | isolates |
|---|---|---|
| `same_relation` | same relation, different subject | subject identity |
| `same_subject` | same subject, different relation | the relation template |
| `cos_matched` | any non-conflict pair, 1:1 nearest-matched to the conflict cosine (caliper 0.02, no reuse) | **similarity level — the decisive control** |
| `conflict_matched` | the conflict pairs that a `cos_matched` partner was found for | the only fair partner for `cos_matched` |
| `random` | uniform non-conflict pairs | the reference a naive analysis stops at |

Controls are drawn from the **exhaustive** pair space (10.4 M pairs at sh_64k),
not a sample: only a few hundred non-conflict pairs reach the conflict cosine
range, so a sampled pool would silently fail to match and the "matched" control
would quietly be a lower-cosine control. Every control is oriented
earlier → later like the conflicts, and eligibility is `¬same_key`, which
excludes conflicts and same-key duplicates in one stroke.

**Statistics.** On unit Δ̂ = Δ/‖Δ‖:

- `resultant` R = ‖mean Δ̂‖ — the shared-direction magnitude. Isotropic null
  `1/√m`.
- pairwise `align` = mean cos(Δ̂ᵢ, Δ̂ⱼ). Isotropic null: mean 0, sd `1/√d`.
- `participation_ratio` (Σλ)²/Σλ² — effective dimensionality of the Δ̂ cloud.
- **held-out subspace energy**: fit a rank-k subspace on *half* the conflict Δ̂,
  measure captured energy on the other half and on the controls. Baseline for a
  random subspace is exactly `k/d`. No classifier, and the fit half is disjoint.

**The null is a sign-flip permutation**: each Δ is randomly re-oriented while
norms, pair identities and axis structure are held fixed. A statistic that
survives it reflects agreement about *which way* an update points, not merely a
shared axis. All `z` below are against this null.

### The one identity that governs everything

For unit vectors, `‖Δ‖² = 2(1 − cos)`. **‖Δ‖ and the pair cosine are the same
statistic.** Reporting "conflict Δ are short" and "conflict pairs are similar"
as two findings would count one fact twice. Both are shown in fig 1; only one is
a degree of freedom.

### Provenance

Vectors: `Qwen/Qwen3-Embedding-4B`, float32, `max_length` 8192, namespace
`Qwen_Qwen3-Embedding-4B|float32|L8192`, pulled from the campaign cache on
ozonderlab2 — the same vectors every committed H-Nav number is built from.
Whitening is the **committed campaign artifact**
`stage0_results/abtt/abtt_whitening_D128.json` (`frozen_global`, D = 128,
n_fit = 2765, fingerprint `3fdacc1f…` verified against
`abtt_operating_point.json`), fitted on `sh_6k`+`sh_32k` only — never on
`sh_64k`. It was loaded, not refitted: refitting here would drop the facts this
script's parser rejects and so silently produce a *different* space.

---

## 2. Magnitude — a real region, but not a separating one

`sh_64k`, raw space (sh_6k and sh_32k agree to within 0.001 on every cosine):

| pair set | mean cos | mean ‖Δ‖ |
|---|---|---|
| **conflict** | **0.9557** | **0.284** |
| same_subject | 0.7960 | 0.638 |
| same_relation | 0.7142 | 0.756 |
| random | 0.6052 | 0.888 |

The conflict cosine is astonishingly stable across a 10× change in store size:
**0.9547 / 0.9557 / 0.9557** for sh_6k / sh_32k / sh_64k. Supersession pairs
occupy a narrow, reproducible band.

The sharpest way to state the magnitude result is as a **matching failure**:

| subset | eligible non-conflict pairs | above the conflict *minimum* | matched at caliper 0.02 | unmatched conflicts sit at cos |
|---|---|---|---|---|
| sh_6k | 102,218 | 306 (0.299%) | 50 / 160 | mean 0.9672 |
| sh_32k | 2,647,616 | 6,716 (0.254%) | 383 / 835 | mean 0.9738 |
| sh_64k | 10,406,516 | 65,782 (0.632%) | 848 / 1687 | mean 0.9748 |

**Half the conflict pairs in sh_64k have no non-conflict partner within 0.02
cosine anywhere among 10.4 million pairs.** That is a statement about a region,
not a mean.

**The honest counterweight, which forbids reading this as detection:** those
65,782 non-conflict pairs above the conflict floor outnumber the 1,687 conflicts
by **39 : 1**. High cosine is close to necessary and nowhere near sufficient.
This is the same fact the ABTT campaign ran into from the other side — the
shipped pipeline buys its precision from the regex `pair_filter` and NLI, not
from cosine — and M7 is a geometric restatement of why.

---

## 3. Direction — the real finding, and it survives matching

`sh_64k`, raw. `conflict_matched` and `cos_matched` are the n-matched,
cosine-matched pair; everything else differs in n:

| pair set | n | R | R·√m | z (sign-flip) | mean align |
|---|---|---|---|---|---|
| conflict | 1687 | 0.1545 | 6.34 | **+79.8** | +0.0233 |
| **conflict_matched** | **848** | — | **4.22** | **+53.3** | **+0.0200** |
| **cos_matched** | **848** | — | **1.65** | **+10.9** | **+0.0020** |
| same_relation | 1687 | — | 1.26 | +4.3 | +0.0002 |
| same_subject | 1012 | — | 1.61 | +4.3 | +0.0018 |
| random | 1687 | — | 1.17 | +2.2 | −0.0002 |

Two things make this a real effect rather than an artefact:

**R does not decay like the null.** If there were no common direction, R would
fall as `1/√m`. Across sh_6k → sh_32k → sh_64k the null falls
0.0791 → 0.0346 → 0.0243 (3.3×) while the observed R holds at
**0.1895 → 0.1604 → 0.1545**. A stable non-zero R is the signature of a genuine
mean direction; the growing z is just the null shrinking under it.

**It survives cosine matching.** At identical n and matched cosine,
`conflict_matched` reaches R·√m = 4.22 (z = +53.3) against `cos_matched`'s 1.65
(z = +10.9), and mean alignment is **10× higher** (0.0200 vs 0.0020). The
directional structure is not a repackaging of "conflict pairs are similar".

**It is about orientation, not just an axis.** The null re-orients each Δ at
random; conflict survives it at z ≈ +53 while every control sits below +11. The
`random` control is oriented earlier → later exactly like the conflicts and shows
**nothing** (z = +2.2, align −0.0002), which rules out a serial-position or
"later facts drift" artefact.

**But calibrate the size.** R = 0.155 means the average unit Δ has only ~15% of
its length along the shared direction — about 2.4% of its energy. The leading
principal component of the conflict Δ̂ cloud carries **5.7%** of the variance and
the participation ratio is ~102 of 2560 dimensions. This is a **weak, broad
tendency**, not a compact cluster. Fig 4 shows why the picture still looks
striking: the conflict Δ form several discrete lobes outside the control blob —
but that plane is fitted to the conflict Δ themselves, the projection maximally
favourable to them, and the lobes turn out to be relations.

---

## 4. Most of the direction is one direction per relation template

Splitting conflict-Δ alignment by whether the two pairs share a relation
(sh_64k raw): **within relation +0.0888, across relation +0.0198** — a 4.5×
gap, and the same gap appears in all three subsets. The top templates are far
above the pooled mean:

| relation | within-relation mean align |
|---|---|
| `… speaks the language of …` | +0.304 |
| `… is located in the continent of …` | +0.227 |
| `… was written in the language of …` | +0.222 |
| `… is affiliated with the religion of …` | +0.219 |

So "the conflict direction" is largely ~37 relation-specific directions. This is
the finding that most limits how far the result generalises: these are synthetic
templated facts, and a per-template update direction is what you would expect
from templated text regardless of whether the update is a *conflict*.

**The residue is small but consistently non-zero.** Across-relation alignment is
+0.0198, still ~10× the cosine-matched control's +0.0020, in all three subsets
and in both spaces. There is a relation-independent component; it is real and it
is weak. Whether it reflects "supersession" or some other shared property of
object-swaps is not resolved here.

---

## 5. A characteristic subspace — low-rank, generalising, and it runs out

Subspace fitted on half the conflict Δ̂, scored on the disjoint half, as a
multiple of the random-subspace baseline `k/d` (sh_64k raw):

| k | conflict (held out) | same_relation | random | cos_matched |
|---|---|---|---|---|
| 1 | **135×** | 68× | 39× | 25× |
| 16 | 44× | 27× | 20× | 14× |
| 256 | 7× | 6× | 6× | 5× |

The conflict direction generalises to conflict pairs the subspace never saw —
that is what makes it structure rather than memorisation. But the advantage is
concentrated in the first few directions and has essentially dissolved by
k ≈ 256. There is a characteristic *low-rank* subspace, not a characteristic
region in any strong sense.

---

## 6. Raw vs ABTT-whitened

**Stated before measuring** (module docstring): ABTT subtracts a common mean μ,
and `(v_l − μ) − (v_e − μ) = v_l − v_e`. The difference operator is *already* a
mean-centering, so the part of ABTT that dominates its effect on cosine geometry
cancels exactly in Δ. Whitening should therefore do little for Δ geometry.

It does less than little — it consistently subtracts (sh_64k):

| statistic | raw | ABTT D=128 |
|---|---|---|
| conflict R | 0.1545 | 0.1196 (−23%) |
| conflict mean align | +0.0233 | +0.0138 (−41%) |
| conflict_matched R·√m vs cos_matched | 4.22 vs 1.65 | 2.49 vs 1.21 |
| held-out k=1 energy | 135× | 51× |
| participation ratio | 102 | 232 |
| within / across relation align | 0.0888 / 0.0198 | 0.0719 / 0.0106 |

Whitening spreads the Δ cloud out (PR 102 → 232) and weakens every measure of
shared direction. It reveals no structure the raw space did not already show.

Three honest qualifications:

1. **Significance does not fall.** The sign-flip z actually rises
   (+79.8 → +90.4) because the permutation null shrinks along with the signal.
   ABTT reduces the *magnitude* of the shared component, not its detectability
   against its own null.
2. **The controls get noisier.** After whitening, `random` moves from z = +2.2 to
   +7.0 and `same_relation` from +4.3 to +9.0. The conflict-vs-control contrast
   narrows from both ends, so the raw space gives the cleaner separation.
3. **What ABTT removes is the relation-independent part.** Within-relation
   alignment falls 19%, across-relation falls 46%. After whitening the residual
   shared direction is *more* purely a relation-template effect, not less — the
   opposite of what you would want if you were hoping whitening would expose a
   general "update" direction.

The matched-fraction column is **not** comparable across spaces: a fixed 0.02
caliper is relatively tighter in ABTT, where the conflict cosine distribution is
much wider. It is comparable across subsets within a space.

---

## 7. sh_6k vs sh_64k — and a nesting caveat that had to be measured

Every qualitative finding holds in all three subsets. Quantitatively:

| | sh_6k | sh_32k | sh_64k |
|---|---|---|---|
| conflict pairs | 160 | 835 | 1687 |
| conflict cos | 0.9547 | 0.9557 | 0.9557 |
| R | 0.1895 | 0.1604 | 0.1545 |
| mean align | +0.0299 | +0.0247 | +0.0233 |
| conflict_matched vs cos_matched (R·√m) | 1.52 vs 1.11 | 3.03 vs 1.50 | 4.22 vs 1.65 |
| within / across relation | 0.086 / 0.027 | 0.086 / 0.022 | 0.089 / 0.020 |

sh_6k is the weakest evidence, not because the effect is smaller but because only
**50** conflict pairs could be cosine-matched there; the matched contrast is
1.52 vs 1.11, suggestive rather than conclusive. sh_64k is where the matched
comparison has enough pairs (848) to be convincing.

**The caveat.** These are *not* independent replications: **454 of sh_6k's 455
facts are also in sh_64k**, and 97% of sh_32k is. Agreement across subsets is
largely re-measurement on nested corpora.

**So it was measured directly.** Restricting sh_64k to the 2,327 facts that occur
in neither sh_6k nor sh_32k (820 conflict pairs, none of which the whitener was
fitted on):

| | full sh_64k | novel facts only |
|---|---|---|
| conflict cos | 0.9557 | 0.9559 |
| R | 0.1545 | 0.1514 |
| mean align | +0.0233 | +0.0216 |
| sign-flip z | +79.8 | +46.4 |
| conflict_matched vs cos_matched (R·√m) | 4.22 vs 1.65 | 2.53 vs 1.74 |
| within / across relation | 0.089 / 0.020 | 0.090 / 0.018 |
| held-out k=1 energy | 135× | 145× |

The effect is unchanged on facts none of the calibration subsets contain. The
nesting is a real limitation of the *cross-subset* comparison, but it is not
what is producing the result.

---

## 8. Figures

| file | what it shows |
|---|---|
| `fig1_magnitude.png` | cosine and ‖Δ‖ densities per pair set. Two panels per space, deliberately, to make visible that they are the same statistic reflected. |
| `fig2_alignment.png` | full pairwise-alignment distributions on a log density axis, dotted line at each set's mean, grey band = ±2/√d. The means differ; the distributions overlap almost completely — the effect is a shift of a broad distribution, not a separated mode. |
| `fig3_spectrum.png` | cumulative variance of the Δ̂ cloud against the isotropic diagonal. |
| `fig4_projection.png` | controls projected into the plane fitted to the **conflict** Δ̂ — the most favourable projection available. Discrete red lobes = relation clusters. |
| `fig5_heldout_energy.png` | held-out subspace energy ÷ `k/d`, log-log. The k=1 advantage and its dissolution by k≈256 are both visible. |
| `fig6_relation_split.png` | within- vs across-relation alignment against the cosine-matched control, plus the top relations carrying it. |
| `fig7_effect_summary.png` | **the summary figure**: every statistic on the sign-flip-null z scale, all sets, both spaces, all three subsets. |

Reading fig 7: z grows with n, so only the adjacent `conflict_matched` /
`cos_matched` pair is an n-matched comparison. The others are indicative.

---

## 9. What is *not* supported

- **Not a detector, and not evidence one would work.** Nothing was trained and no
  threshold was fitted. 65,782 non-conflict pairs at sh_64k sit above the
  conflict cosine floor — 39 per conflict.
- **Not a compact cluster.** R ≈ 0.155, top PC = 5.7% of variance, participation
  ratio ~102/2560. "Characteristic direction" here means a measurable shift in a
  broad distribution.
- **Probably not a property of factual updating in general.** The dominant
  component is per-relation-template, on synthetic templated text. The
  relation-independent residue (align ≈ 0.020 vs 0.002 for matched controls) is
  the only part with a claim to generality, and it is small.
- **One encoder, one benchmark family.** `Qwen3-Embedding-4B` on MemoryAgentBench
  `Conflict_Resolution`. Nothing here transfers to CrossEp-Know or another
  embedder without measurement.
- **The three subsets are nested** (§7). The novel-facts slice is the honest
  out-of-sample evidence; the cross-subset table is not three independent shots.
- **`‖Δ‖` is not independent evidence** from the cosine, ever.
- **ABTT is not shown to be useless** — only that it does not help *this*
  geometry, for a reason that was stated algebraically before it was measured.

---

## 10. Reproduce

```bash
python hnav/stage0/m7_delta_geometry.py --subsets sh_6k sh_32k sh_64k
python hnav/stage0/m7_delta_geometry.py --subsets sh_64k --novel-vs sh_6k sh_32k \
       --out-dir stage0_results/delta_geometry/novel
pytest hnav/tests/test_delta_geometry.py -q      # 15 tests, closed-form checks
```

~50 s on CPU with a warm embedding cache; no LLM, no GPU, no gold answers read.

| artifact | contents |
|---|---|
| `m7_delta_geometry.json` | every statistic, all sets × spaces × subsets |
| `novel/m7_delta_geometry.json` | the novel-facts-only sh_64k slice |
| `fig1..fig7*.png` | figures above |
| `hnav/stage0/m7_delta_geometry.py` | the measurement |
| `hnav/tests/test_delta_geometry.py` | null-regime and planted-signal tests |

Nothing in `stage0_results/stage1_operating_point.json` or the shipped
thresholds was read or modified by this analysis.
