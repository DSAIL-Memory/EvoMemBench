# The conflict fingerprint — reading guide for `fp*.png`

Drawn from the same vectors and controls as `m7b_dimension_profile.json`
(`hnav/stage0/m7c_fingerprint_viz.py`, ~50 s). The **fingerprint** of a set of
unit difference vectors Δ̂ = (v_later − v_earlier)/‖·‖ is the per-coordinate
z-score of the signed mean: `zᵢ > 0` means the later fact is consistently
*higher* than the earlier one in coordinate *i*. Every panel compares conflicts
with their cosine-matched non-conflict controls at the **same n**, so a
coordinate that lights up does so for conflict-ness, not for similarity.

## fp1 — what the fingerprint is (`fp1_fingerprint_sh_64k_{raw,abtt}.png`)

- **Top row (Manhattan plot).** |z| for every one of the 2560 coordinates. Red =
  later fact higher, blue = lower, grey = the matched control in the same
  coordinate. 842 coordinates clear |z| = 4 for conflicts; 37 for controls.
- **Middle row.** The 40 strongest coordinates. The control bar is inside the
  grey ±4 band in every one of them.
- **Bottom left.** Same coordinates, read per pair: the fraction of pairs whose
  later fact is higher. The strongest coordinate (#8) goes up in 73% of
  conflicts and 53% of controls — a *tendency*, never a rule.
- **Bottom right.** Cumulative share of fingerprint energy Σz²: the top 16
  coordinates hold 7%, the top 256 hold 35%. It is spread, not localised.

**An observation the tables could not show:** the strongest coordinates sit at
low indices. 62% of the top 16 are among coordinates 0–127 (5% expected by
chance; median index 58); mean |z| there is 5.1 vs 3.2 elsewhere, while the
control is flat (1.33 vs 1.31). This is not a variance artefact — coordinates
0–127 hold 8% of raw-vector variance, and the rank correlation between |z| and
per-coordinate variance is 0.04. Why this encoder's early coordinates carry the
object-slot direction is not known; it is reported, not explained.

## fp2 — is it the same fingerprint? (`fp2_reproducibility.png`)

Per-coordinate z, one point per coordinate.

| comparison | r | top-256 overlap | same sign on overlap |
|---|---|---|---|
| sh_6k vs sh_64k (raw) | **+0.906** | 52% | 100% |
| raw vs ABTT (sh_64k) | +0.541 | 17% | 100% |
| conflict vs matched control | +0.237 | 5% | — |

The fingerprint reproduces across a 10× store change almost coordinate for
coordinate (sh_6k is nested in sh_64k, but only 160 of 1687 pairs are shared).
ABTT keeps the signs but moves the weight to different coordinates. The
residual r = 0.24 against the control is real and expected: most matched
controls share the *relation template* with a conflict, so a little of the
template direction leaks into both.

## fp3 — what it does to a pair (`fp3_pair_scores.png`)

Each held-out pair projected onto a fingerprint fitted on the other half of the
conflicts. With 16 coordinates the conflict out-scores its own matched control
in 77% of pairs; with 256, 83% (raw). ABTT: 69% / 77%. The distributions overlap
heavily — this is a shift, not a separation.

## fp4 — one fingerprint or one per template? (`fp4_by_relation_sh_64k.png`)

The 60 strongest coordinates, re-computed inside each relation template. A
column with the same colour top to bottom is a coordinate every template uses
the same way; a patchy column belongs to a few templates. Most columns are
consistent in *sign* but vary in strength — three templates (*citizen of*,
*religion of*, *continent of*) carry much of the amplitude, and *was educated*
barely participates. So the fingerprint is shared in direction and
template-weighted in magnitude, matching M7b's relation-disjoint result
(0.80 vs 0.83).

## What not to read into it

No coordinate is a "conflict detector": the best one is a 73/53 split. The
information is the *joint sign pattern* over hundreds of coordinates, and what
it encodes is *which slot of the fact changed* (object vs subject), on
one-slot template edits from one encoder.
