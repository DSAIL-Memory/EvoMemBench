# hnav_idonly — google/gemma-3-4b-it

Run 2026-08-30T18:01:34.247734+00:00 · git `d34c10fcd5f7` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 11 | 13 | +2 | 0.625 |
| sh_6k | calibration split | conflicted | 74 | 7 | 7 | +0 | 1 |
| sh_6k | calibration split | unique | 26 | 4 | 6 | +2 | 0.5 |
| sh_32k | calibration split | all | 100 | 6 | 6 | +0 | 1 |
| sh_32k | calibration split | conflicted | 65 | 0 | 0 | +0 | 1 |
| sh_32k | calibration split | unique | 35 | 6 | 6 | +0 | 1 |
| sh_64k | held-out | all | 100 | 8 | 7 | -1 | 1 |
| sh_64k | held-out | conflicted | 66 | 1 | 1 | +0 | 1 |
| sh_64k | held-out | unique | 34 | 7 | 6 | -1 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.5247960952910904% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 51, "stratum": "conflicted", "class": "information_loss", "native_output": "Cricket. \n", "arm_output": "Association Football.", "target_serial": 279, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.11, "band": [0.3, 0.5], "unique_native": 0.15384615384615385, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6423438869658674% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.06, "band": [0.3, 0.5], "unique_native": 0.17142857142857143, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.31396289925065496% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [80], "harms": [{"index": 80, "stratum": "unique", "class": "information_loss", "native_output": " Persian.", "arm_output": "F. 1.\n", "target_serial": 3594, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.08, "band": [0.3, 0.5], "unique_native": 0.20588235294117646, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.