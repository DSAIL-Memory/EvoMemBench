# hnav_raw — Qwen/Qwen3-4B-Instruct-2507

Run 2026-08-31T11:56:50.215133+00:00 · git `247ac2cf8bfa` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 30 | 94 | +64 | 1.084e-19 |
| sh_6k | calibration split | conflicted | 74 | 4 | 68 | +64 | 1.084e-19 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 53 | 83 | +30 | 1.537e-08 |
| sh_32k | calibration split | conflicted | 65 | 19 | 49 | +30 | 1.537e-08 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.4768250588238345% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6301580283964697% · harm {"n_harmed": 1, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 9, "stratum": "conflicted", "class": "gold_cut", "native_output": "Rome", "arm_output": "Watertown", "target_serial": 1291, "gold_cut": true}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.53, "band": [0.3, 0.5], "unique_native": 0.9714285714285714, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.