# hnav_geo — Qwen/Qwen3.5-9B

Run 2026-08-31T07:28:50.740745+00:00 · git `b511ffad0815` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 39 | 81 | +42 | 4.547e-13 |
| sh_6k | calibration split | conflicted | 74 | 13 | 55 | +42 | 4.547e-13 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 61 | 86 | +25 | 1.624e-06 |
| sh_32k | calibration split | conflicted | 65 | 27 | 52 | +25 | 1.624e-06 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 51 | 62 | +11 | 0.003418 |
| sh_64k | held-out | conflicted | 66 | 24 | 36 | +12 | 0.0004883 |
| sh_64k | held-out | unique | 34 | 27 | 26 | -1 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -2.869587363821015% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.5085686228619366% · harm {"n_harmed": 2, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Washington, D.C.", "target_serial": 707, "gold_cut": true}, {"index": 42, "stratum": "conflicted", "class": "information_loss", "native_output": "superhero", "arm_output": "rabbi", "target_serial": 1568, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.61, "band": [0.3, 0.5], "unique_native": 0.9714285714285714, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VOID: preregistered void condition 4_no_harmful_suppression: {"n_suppressed_harmful": 8, "n_suppressed_superseded": 524, "n_suppressed_same_value": 0} · A/A discordant 0 · token Δ -0.22338418332491194% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [82], "harms": [{"index": 82, "stratum": "unique", "class": "information_loss", "native_output": "South America", "arm_output": "Antarctica", "target_serial": 4049, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.51, "band": [0.3, 0.5], "unique_native": 0.7941176470588235, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.