# Next goal — a geometry+NLI conflict detector, benchmarked like H-Nav

> **STATUS (2026-08-27): DONE.** Executed as `pipelines/hnav_ces` (relation-aware CES —
> the parser-free global variant failed its pair-level gate) and `pipelines/hnav_abtt_noparser`.
> Held-out sh_64k: parser arms 64/100 · noparser 59/100 · CES 55/100 (native 45).
> Full results and interpretation: `E2E_REPORT.md`. The rest of this file is the
> original plan, kept for provenance.

*(User directive, 2026-08-26: if a truly promising method emerges from the
geometry-filter experiments — "like hnav but without hnav's cosine and
parser" — build a conflict detector by running NLI on top of it, benchmark it
exactly the way H-Nav was benchmarked, and compare the results.)*

## What the experiments say the candidate is

Two screens survived the evidence in `REPORT.md`; they answer the directive
differently and both should be carried into the benchmark:

1. **CES screen (contrastive edit subspace — experiment 4, `REPORT.md` §7)**
   — the genuinely *non-cosine* method and the strongest one measured on the
   mandated arena: per relation, object-edit subspace minus subject-edit
   subspace energy, `||U_obj_r^T d̂||² − ||U_subj_r^T d̂||²` (k=20 each, fit
   on 989 calibration gold edits + 8,716 calibration hard negatives). It
   **beats ABTT-cosine on the balanced sh_64k set (0.9756 vs 0.9648) and the
   0.87–0.97 band (0.9690 vs 0.9516)**, orders 98.0% of the comparisons raw
   cosine gets exactly wrong, transfers to unseen transitions (0.963) and
   subjects (0.974), and has an interpretable natural threshold at 0.
   Caveats to design around: it takes relation identity from the parser
   (a global-subspace fallback exists but is unmeasured as a headline
   number), and it superseded RCESP, which stays as ablation evidence only.
2. **ABTT(D=128)-cosine screen** — still cosine-family, but parser-free and
   the strongest, most transferable pairwise signal measured (0.999 hard-task
   AUROC, 0.952 band AUROC, robust frozen operating point P0.85/R0.98). Its
   NLI pipeline **already exists and is benchmarked** (`pipelines/hnav_abtt`),
   which is precisely what makes the comparison meaningful.

## The pipeline to build: `pipelines/hnav_ces/`

Same shape as the two committed pipelines (`pipelines/_shared/runner.py`
conventions), so results are comparable row-for-row:

- **Stage 1 (replaces the cosine screen and the parser key-check):** CES
  pair score with the natural threshold 0 (or a frozen calibration-fit
  operating point if 0 proves miscalibrated on calibration — decided before
  touching sh_64k). Ship the fitted subspaces as a committed artifact
  (`stage0_results/geometry_filter/ces_subspaces_k20.json`, sha256
  fingerprint, same pattern as `abtt_whitening_D128.json` /
  `ABTTWhitening.to_dict`). Two arms:
  - `ces_global` — fully parser-free, global object/subject subspaces (the
    honest "no cosine, no parser" arm; measure it at the pair level first);
  - `ces` with relation identity — the upper-bound arm; if relation identity
    must come from the parser, say so in the arm name rather than hiding it.
- **Stage 2 (unchanged):** the existing bidirectional-NLI verification from
  `hnav/core/read_gate.py` — a pair is a verified conflict only if
  contradiction clears the threshold in both directions. Reuse the committed
  NLI model and thresholds; do not retune them, or the comparison stops being
  about the screen.
- **Operating point:** fit on sh_6k + sh_32k only (best-F1 or the
  coverage-balanced criterion the other pipelines used), freeze, commit as
  `rcesp_operating_point.json`. Never touched after the first confirmatory run.

## Benchmark protocol (identical to the committed runs)

- Subsets: `sh_6k + sh_32k + sh_64k`, never sh_262k (`ALLOWED_SUBSETS` in
  `pipelines/_shared/runner.py`; user decision 2026-08-24).
- One shot per answering model, same models as the existing series; embedder
  frozen (Qwen3-Embedding-4B float32 L8192, the campaign cache).
- Report stratified per subset, never pooled; same question-level scoring as
  the committed arms so the numbers line up with `pipelines/README.md`
  (hnav_raw 17/66 → hnav_abtt 37/66 on the recorded comparison).
- Compare four arms: `hnav_raw` (committed), `hnav_abtt` (committed),
  `hnav_ces_global` (new, parser-free), `hnav_ces` (new, relation-aware
  upper bound). Paired per-question comparison, McNemar or paired bootstrap.

## Honest expectations, written down before running

- At the pair level CES beats ABTT-cosine on the balanced set and the
  overlap band but trails it on the aggregate hard task (0.981 vs 0.999) —
  so `hnav_ces` vs `hnav_abtt` is a genuinely open comparison, decided by
  which regime the benchmark's real retrieval pools resemble. The scientific
  value of the run is (a) whether a *non-cosine* geometric screen can carry
  an end-to-end benchmark, and (b) whether CES's profile spends the NLI
  budget better (NLI wastes calls on screen false positives). A negative
  result is reportable per the decision rules.
- If a combined screen is tried (ABTT-cos OR CES above their thresholds),
  it is a fifth arm, preregistered as such — not a post-hoc rescue.

## Guard rails that still bind

- Read-path only. `hnav/core/write_policy.py` must not exist (T8 NO_GO is
  permanent). `read_policy.py`/`read_gate.py` stay barred from `H_raw`.
- The subspace artifact is calibration-fit; sh_64k stays confirmatory.
- No BFCL suites (`In-Episode-Execution/`, `Cross-Episode-Execution/`).
- New `.gz` files under `stage0_results/` need `!name.gz` gitignore lines.
