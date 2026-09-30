# hnav_idonly — Qwen/Qwen3.5-9B

Run 2026-08-31T05:31:42.219035+00:00 · git `b511ffad0815` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 39 | 99 | +60 | 1.735e-18 |
| sh_6k | calibration split | conflicted | 74 | 13 | 73 | +60 | 1.735e-18 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 61 | 92 | +31 | 7.916e-09 |
| sh_32k | calibration split | conflicted | 65 | 27 | 58 | +31 | 7.916e-09 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 51 | 69 | +18 | 4.005e-05 |
| sh_64k | held-out | conflicted | 66 | 24 | 43 | +19 | 3.815e-06 |
| sh_64k | held-out | unique | 34 | 27 | 26 | -1 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.5247960952910904% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6423438869658674% · harm {"n_harmed": 1, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Washington, D.C.", "target_serial": 707, "gold_cut": true}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.61, "band": [0.3, 0.5], "unique_native": 0.9714285714285714, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.31396289925065496% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [87], "harms": [{"index": 87, "stratum": "unique", "class": "information_loss", "native_output": "Lady Gaga", "arm_output": "Madonna", "target_serial": 350, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.51, "band": [0.3, 0.5], "unique_native": 0.7941176470588235, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.