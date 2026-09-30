# hnav_geo — the fully parser-free geometric identity screen (E2E-3)

The question this arm answers: can the H-Nav read-time detector beat the
committed parser pipeline (hnav_raw, sh_64k **64/100**) end-to-end with an
identity screen that reads **embeddings only** — no parser field consulted at
inference anywhere in the detection path?

The screen (`hnav/geometry_filter/geo_artifact.py`):

    score(a,b) = min( (cos_w − T_w)/s_w , (probe − T_p)/s_p )

- `probe` — the slot probe: logistic regression on `|d_hat|`, the absolute
  axis profile of the unit difference vector. It answers "did the OBJECT
  slot change (candidate supersession) or the SUBJECT slot (the cross-key
  adversary the NLI rubber-stamps)?" — fit on the gold conflict dataset's
  calibration split (989 gold vs 8,716 hard-negative edits).
- `cos_w` — cosine under the frozen committed ABTT D=128 whitening (copied
  into the artifact; source fingerprint pinned).
- anchors/scales from calibration pool pairs; `pair_filter(tau)` accepts a
  float (diagonal family) or a `'tw:tp'` rectangle (per-axis offsets — the
  Amendment-1 grid; the diagonal couples the axes and misses the best
  zero-harm corner, a loose whitened-cos with a strict probe).

**Frozen operating point** (clean-cache selection 2026-08-29, sha pinned in
`pipeline.json`): `cos_pair 0.94, geo tau '-0.4:0.2', NLI 0.90, r_min 0.44`
— pair precision **1.000**, pool recall **0.7895**, **0 harmful**,
conflicted-question recall 104/139. Best committed parser-free point before
this arm: 0.4444 (abtt_noparser); CES with parser-relation: 0.7343.

Pair-level evidence (`stage0_results/geometry_filter/geo_pairlevel.json`):
balanced sh_64k AUROC 0.9716 (CES 0.9756, ABTT-cos 0.9648, raw cos 0.8930),
band 0.9657; confirmatory hard task 0.9984 / AUPRC 0.9784 (ABTT-cos level,
far above CES 0.8466); best unseen-transition tail in the repo
(TPR@FPR1e-4 = 0.481 vs ABTT-cos 0.404, CES 0.208) with seen-tail 0.720.

Preregistration: `stage0_results/geometry_filter/GEO_PREREG.md` (grids,
selection rule, wet-run gate GG1: pool recall must exceed the best committed
parser-free point 0.4444, endpoint GG2: sh_64k overall > 64/100 paired vs the
committed parser-arm records). Selection ran on sh_6k + sh_32k only;
sh_64k untouched until the single wet shot.

## Run against a new answering model

```bash
python pipelines/hnav_geo/run.py --llm-model <served-name> --dry-run   # always first
python pipelines/hnav_geo/run.py --llm-model <served-name> --llm-base-url http://localhost:8003/v1
```

Same frozen substrate as every arm: Qwen3-Embedding-4B fp32 L8192 prepasses
(base benchmarkpage, cos_loose 0.90 — the parser arm's own prepasses, reused),
`cross-encoder/nli-deberta-v3-large` bidirectional NLI replay, subsets
sh_6k + sh_32k + sh_64k, one shot per model, voids reported not re-rolled.

## Multi-model outcome (campaign E2E-5, 2026-08-30/31) — the thesis method

This arm is **the thesis method** (`THESIS_CLAIMS.md`, user decisions
2026-09-17). Five answering models, three context sizes, one shot per cell,
A/A floor 0 in every cell, suppression plans byte-identical across models
(1,152 / 1,005 / 532 deletions per subset). Overall accuracy /100,
native → H-Nav-GEO (net, exact McNemar p); tables generated from
`results/*/detector_gap_sh_*.json`:

| model | sh_6k | sh_32k | sh_64k (held-out) |
| --- | --- | --- | --- |
| gemma-3-4b-it | 45 → **76** (+31, 3.7e-08) | 38 → **45** (+7, 0.14) | 33 → **36** (+3, 0.38) |
| gemma-4-E2B-it | 40 → **72** (+32, 6.7e-08) | 44 → **58** (+14, 2.6e-03) | 37 → **41** (+4, 0.34) |
| Phi-4-mini-instruct | 40 → **76** (+36, 2.9e-11) | 50 → **66** (+16, 8.6e-04) | 46 → **52** (+6, 0.15) |
| Qwen3-4B-Instruct-2507 | 30 → **77** (+47, 1.8e-13) | 53 → **77** (+24, 8.0e-07) | 45 → **56** (+11, 7.4e-03) |
| Qwen3.5-9B | 39 → **81** (+42, 4.5e-13) | 61 → **86** (+25, 1.6e-06) | 51 → **62** (+11, 3.4e-03) |

**Positive in 15 of 15 cells, significant at 5 % in 11 of 15**, with the
unique (no-conflict) stratum unchanged to within one question everywhere and
the prompt shorter by 2.87 % / 0.51 % / 0.22 %. Deletion is the only page edit
that is positive on every model; the placement arms (`demote_late`, `anti`)
flip sign from model to model. Full claims, wording rules and provenance:
`THESIS_CLAIMS.md`.

