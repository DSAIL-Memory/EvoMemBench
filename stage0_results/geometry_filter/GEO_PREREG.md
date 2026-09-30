# GEO identity screen — preregistration (E2E-3, 2026-08-29)

Question: can a PARSER-FREE geometric identity screen carry the full H-Nav
pipeline to a higher held-out overall accuracy than the committed parser arm
(hnav_raw, sh_64k 64/100)? Written before `detector_gap --select` was run for
this arm; the screen's construction and anchors were designed from
calibration-only inspection (gold-dataset calibration split + calibration
pool pairs), the committed precedent of the CES/noparser grid amendments
(E2E_REPORT.md §"Operating points").

## The screen (frozen form)

`hnav/geometry_filter/geo_artifact.py::GeoIdentityScreen` — at inference it
reads the two fact VECTORS only (no parser field, no metadata):

    score(a,b) = min( (cos_w(a,b) − T_w)/s_w , (probe(a,b) − T_p)/s_p )
    pair_filter(tau): score >= tau

with `probe` = the slot probe (logistic on |d_hat|, fit on gold-dataset
calibration gold vs hard-negative edits — the committed slot-probe recipe,
REPORT.md §2) and `cos_w` = cosine under the frozen committed ABTT D=128
whitening (copied into the artifact, source fingerprint pinned). Anchors
(T_w, T_p) = the joint zero-false-positive staircase point over NLI-passing
calibration pool pairs; scales = feature std over the same pairs. All fit
material is sh_6k + sh_32k; sh_64k untouched.

Design evidence (calibration-only, recorded before selection): joint
pair-level zero-FP recall of the anchor conjunction ≈ 0.71; single-axis
references cos_w ≈ 0.33–0.55, probe ≈ 0.25–0.39; the NLI backstop kills
almost none of the cross-key adversaries (447 of 453 pass bidirectional 0.9)
— the screen carries the entire identity burden, and the key-level harm rule
(not pair-level FP) decides how much slack negative tau buys.

## Selection (identical machinery to every committed arm)

- `detector_gap.py --select --subsets sh_6k sh_32k --page-source benchmark
  --pair-screen geo --geo-artifact stage0_results/geometry_filter/geo_identity_screen.json`
- Grid: the shipped raw axes (cos_pair {0.90, 0.92, 0.94}, r_min
  {frozen, loose, off}, ambiguity {all, any, none}, NLI {0.5, 0.9, 0.99})
  × GEO_TAU_GRID = (−0.75, −0.50, −0.25, −0.10, 0.0, 0.10, 0.25),
  tau in anchored-margin units (0 = the pair-level zero-FP anchor; negative
  explores the harm-rule slack; positive is stricter).
- Rule: `selection_rule_for("geo")` — n_suppressed_harmful == 0 hard,
  maximize pair_recall_pool, tie-breaks (higher tau first) as shipped.
  A screen that cannot reach zero harm gets a null result, not an operating
  point (the fusion precedent).

## Amendment 1 (2026-08-29, calibration-only — the CES-grid precedent)

Round-1 selection (preregistered diagonal grid) chose tau +0.10: pool recall
0.5955, precision 1.000, 0 harmful; looser diagonal taus hit the harm
plateau (tau −0.75: recall 0.914, 471 harmful — the same irreducible
NLI-rubber-stamped core E2E-2 measured). The diagonal family couples the two
axes through one anchor, but the calibration staircases place the best
zero-FP corner at a LOOSE whitened-cos with a STRICT probe (margin units
tw ≈ −0.55, tp ≈ +0.74) — unreachable by any diagonal tau. Amendment, after
inspecting calibration detection metrics only (sh_64k untouched): the tau
axis is extended to per-axis rectangles 'tw:tp',
tw ∈ {−1.0, −0.75, −0.55, −0.4, −0.25, 0} × tp ∈ {0.2, 0.4, 0.74, 1.0},
plus the round-1 winner '0.1:0.1' for continuity; base axes unchanged.
Selection rule unchanged (zero harm hard, max pool recall; rectangle
tie-break ranks by tw+tp, stricter first).

## Gates

- **GG1 (wet-run gate):** the arm is run wet only if the selected operating
  point's calibration pool recall EXCEEDS 0.4444 — the best committed
  parser-free point (abtt_noparser). Otherwise: null result, no LLM spent.
- **GG2 (primary endpoint, sh_64k, one shot):** overall accuracy > 64/100
  (the committed hnav_raw/hnav_abtt result), paired per-question McNemar
  against the committed parser-arm records. Secondary: conflicted ≥ 37/66;
  unique ≥ 27/34; exact-p reported for all.
- **GG3 (do-no-harm):** the standard void conditions and harm classes apply
  unchanged (guards, positive control, A/A floor, protective claim).
- One shot per subset; a void is reported, not re-rolled. sh_262k excluded.
- Calibration accuracy runs (sh_6k, sh_32k) are reported for continuity but
  are not endpoints.

## Comparison set (same protocol, committed artifacts)

native 45; hnav_raw/hnav_abtt 64 (conflicted 37, unique 27); hnav_abtt_noparser
59 (31, 28); hnav_ces 55 (28, 27); fusion relaxed-harm exploratory 61 flat.
Answering model frozen: Qwen3-4B-Instruct-2507 at :8003, all server flags
unchanged.

## Also preregistered: the pair-level companion report

The GEO score (continuous, tau-free) is evaluated on the gold conflict
dataset exactly as CES was (balanced per-subset AUROC vs the cosine-only
baseline, 0.87–0.97 band, confirmatory hard task, inverted-win, tail TPRs,
seen/unseen transitions) and reported alongside the committed CES rows —
plus the PCA investigation (PCA-compressed probe variants, PCA routing) with
its calibration evidence.
