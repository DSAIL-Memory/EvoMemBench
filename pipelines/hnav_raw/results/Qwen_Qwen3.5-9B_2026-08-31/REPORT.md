# hnav_raw — Qwen/Qwen3.5-9B

Run 2026-08-31T03:34:35.117916+00:00 · git `b511ffad0815` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 39 | 98 | +59 | 3.469e-18 |
| sh_6k | calibration split | conflicted | 74 | 13 | 72 | +59 | 3.469e-18 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 61 | 91 | +30 | 1.537e-08 |
| sh_32k | calibration split | conflicted | 65 | 27 | 57 | +30 | 1.537e-08 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 51 | 67 | +16 | 0.0004025 |
| sh_64k | held-out | conflicted | 66 | 24 | 42 | +18 | 7.629e-06 |
| sh_64k | held-out | unique | 34 | 27 | 25 | -2 | 0.5 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.4768250588238345% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6301580283964697% · harm {"n_harmed": 1, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Washington, D.C.", "target_serial": 707, "gold_cut": true}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.61, "band": [0.3, 0.5], "unique_native": 0.9714285714285714, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.3067337601427156% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}, "by_stratum": {"unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}}, "protective_claim_void": true, "voiding_questions": [82, 87], "harms": [{"index": 82, "stratum": "unique", "class": "information_loss", "native_output": "South America", "arm_output": "Antarctica", "target_serial": 4049, "gold_cut": false}, {"index": 87, "stratum": "unique", "class": "information_loss", "native_output": "Lady Gaga", "arm_output": "Madonna", "target_serial": 350, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.51, "band": [0.3, 0.5], "unique_native": 0.7941176470588235, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.