# GATE G1 — ABTT offline geometry, calibration split

**Verdict: PASS.** Whitening before the cosine screen substantially improves the
decision-relevant readout on both calibration subsets. Winning cell:
**`frozen_global`, D = 128** (`D = 64` is the conservative alternative).

Artifacts: `m6_abtt_geometry.json` (mechanism M3 + transfer),
`m6_abtt_vectors.json` (D ≤ 16), `m6_abtt_dsweep.json` (D 16–192).
Script: `hnav/stage0/m6_abtt_geometry.py`. Tests: `hnav/tests/test_abtt_geometry.py`.
Subsets: `sh_6k` + `sh_32k` only. **`sh_64k` and `sh_262k` were not read.**

---

## 1. Correctness checks before any claim

The vector path independently reproduces two committed numbers:

| quantity | committed | M6 | source |
|---|---|---|---|
| candidate-pair floor, sh_6k | 0.58 | **0.5815** | `m1b_grouping_ablation.json` |
| candidate-pair floor, sh_32k | 0.61 | **0.6130** | `m1b_grouping_ablation.json` |
| anisotropy (unrelated-pair mean cos) | 0.6048 / 0.6062 | **0.6024 / 0.6026** | `m1_geometry_calibration.json` |

The prepass path reproduces the committed `abtt_ab` AUCs **exactly**
(0.9361 → 0.9546 and 0.9876 → 0.9908; r_min 0.3523 → 0.4513 and 0.6288 → 0.6946),
which validates the supersession label logic against the original implementation.

## 2. ABTT does what the literature says

Anisotropy is removed completely, and **mean-centering alone (D = 0) does
essentially all of it**:

| subset | raw | D=0 | D=3 | D=128 |
|---|---|---|---|---|
| sh_6k | 0.6024 | −0.0024 | −0.0011 | +0.0002 |
| sh_32k | 0.6026 | −0.0010 | −0.0012 | −0.0009 |

The band is restored with it: the candidate-pair floor drops from 0.5815 → 0.062
(sh_6k) and 0.6130 → 0.081 (sh_32k). The documented "bottom 76% of the cosine
scale is never used" is a raw-space artifact and it disappears.

## 3. AUC barely moves — and that is why this was left off before

Grouping AUC gains are **+0.002 to +0.005**, because raw is already 0.9964 /
0.9917. Judged on AUC, ABTT is not worth turning on. That is the reading the
committed A/B produced, and on its own terms it is correct.

## 4. But AUC is the wrong readout, and the right one moves a lot

The shipped operating point is selected under `n_suppressed_harmful == 0` —
precision 1.000 — then maximises recall. So **recall-at-precision** is the
decision-relevant statistic. Raw vs `frozen_global | D=128`:

| bar | sh_6k raw | sh_6k ABTT | sh_32k raw | sh_32k ABTT |
|---|---|---|---|---|
| P ≥ 0.90 | 0.8625 | 0.9187 | 0.7856 | 0.8419 |
| P ≥ 0.95 | 0.4437 | **0.8438** | 0.6611 | 0.6946 |
| P ≥ 0.99 | 0.0750 | **0.7188** | 0.0072 | **0.5210** |
| P ≥ 1.00 | 0.0750 | **0.5125** | 0.0072 | **0.2910** |

Consistent and monotone in the bar: **the tighter the precision requirement, the
larger the gain** — modest (+0.056) at P ≥ 0.90, transformative (9.6× and 72×) at
P ≥ 0.99. This is exactly the predicted mechanism: anisotropy crowds unrelated
pairs into the same high-cosine band as true conflicts, contaminating the
high-confidence tail of the ranking. Whitening decompresses the band and cleans
that tail. AUC is dominated by the easy bulk and cannot see it.

## 5. Regime and D

- **`pool_level` refuses in every cell** (`n_fit = 50 < 200`). The documented
  objection is now *evidenced* rather than asserted — but it is also irrelevant,
  because the fit basis need not be the decision pool (§7).
- **`frozen_global` ≥ `per_store`**, and degrades far more gracefully at high D:
  on sh_6k, `per_store` collapses past its optimum (AUC 0.9988 at D=48 → 0.9435
  at D=192) while `frozen_global` holds (0.9983 → 0.9963). The global fit uses
  2,765 rows instead of 455, so its directions are better estimated. It also
  decouples estimation quality from store size, which is the property a shipped
  constant needs.
- **Optimal D is store-size dependent under `per_store`** (≈48 at sh_6k, ≈128 at
  sh_32k) and stable under `frozen_global` (128 on both). Another reason to
  prefer the global fit.
- D = 128 of 2560 dimensions is 5% of the space, and it is an interior optimum on
  both subsets (both fall back at D = 192), so the grid brackets it.

## 6. What did NOT replicate

- **Threshold transfer is not supported by this evidence.** Whitening halves the
  degradation in the one direction that works (0.632 → 0.368), but the reverse
  direction admits zero pairs in *both* spaces, and band-normalised threshold
  spread slightly favours raw (0.089 vs 0.105). n = 1 usable direction is not a
  portability result. Do not claim one.
- **Inside the raw-screened pool at the shipped `cos_pair = 0.90`, whitening
  changes nothing** (recall is already 1.0000 there; equal-coverage ΔP = −0.003 /
  −0.0003). The committed prepass population is censored by the raw screen, so it
  structurally cannot show the gain in §4 — which is why the original A/B saw
  only +0.019.

## 7. The load-bearing caveat for Phase 3

The shipped pipeline **does not buy precision from cosine.** It runs a
deliberately loose cosine screen and gets precision from the regex
`pair_filter` (parsed subject+relation identity) plus bidirectional NLI. It
therefore never operates in the regime where §4's gain exists.

Two consequences, and they point in opposite directions:

1. **Turning ABTT on may change nothing end-to-end** in the primary arena, because
   the identity screen is already doing the work the geometry would newly be able
   to do. Phase 3's G3 must be judged on `pair_recall_pool` at precision 1.000,
   not on §4.
2. **It is the strongest available answer to the standing criticism of the
   thesis** — that any gain comes from templates and serial numbers rather than
   geometry (the reason M1b exists). If whitened geometry can carry precision
   without the regex, that criticism weakens substantially, and the result
   transfers to arenas with no parse to fall back on. CrossEp-Know is exactly
   such an arena and its anisotropy is worse (0.786 across contexts).

`min_fit_n = 200` is not a barrier: `MABAdapter.facts` holds the full store
(455 / 2,310 / 4,580 / 18,332 rows) and `select_pool` only *selects* 50 from it,
so a `per_store` or `frozen_global` fit is available online. Under
`frozen_global` the question does not arise at all — μ and C are constants.

## 8. Recommended next step

Proceed to Phase 3 (whitened prepass + operating-point re-selection on detection
quality only, calibration split, no LLM), carrying **`frozen_global`, D = 128**
with `D = 64` as a declared fallback. Phase 2's Stage-0 re-derivation is
completeness work and is not on the critical path.

Given §7, running **Phase 6 (CrossEp) before Phase 5** is worth considering: it
is cheap, nothing is frozen there, and it is where the §4 mechanism has room to
convert into an effect.
