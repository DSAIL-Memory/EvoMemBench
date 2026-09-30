# hnav_geo — google/gemma-3-4b-it

Run 2026-08-31T11:08:05.092958+00:00 · git `247ac2cf8bfa` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 45 | 76 | +31 | 3.673e-08 |
| sh_6k | calibration split | conflicted | 74 | 22 | 53 | +31 | 3.673e-08 |
| sh_6k | calibration split | unique | 26 | 23 | 23 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 38 | 45 | +7 | 0.1435 |
| sh_32k | calibration split | conflicted | 65 | 11 | 18 | +7 | 0.1185 |
| sh_32k | calibration split | unique | 35 | 27 | 27 | +0 | 1 |
| sh_64k | held-out | all | 100 | 33 | 36 | +3 | 0.375 |
| sh_64k | held-out | conflicted | 66 | 14 | 17 | +3 | 0.25 |
| sh_64k | held-out | unique | 34 | 19 | 19 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -2.869587363821015% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 36, "stratum": "conflicted", "class": "information_loss", "native_output": "Madonna", "arm_output": "The Beatles", "target_serial": 364, "gold_cut": false}, {"index": 55, "stratum": "conflicted", "class": "information_loss", "native_output": "Czech Republic", "arm_output": "United States of America", "target_serial": 299, "gold_cut": false}]}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.5085686228619366% · harm {"n_harmed": 5, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 4}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 3}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [7], "harms": [{"index": 7, "stratum": "unique", "class": "information_loss", "native_output": "Montanus of Phrygia", "arm_output": "Adalbert of Prague", "target_serial": 202, "gold_cut": false}, {"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Saint Petersburg", "target_serial": 707, "gold_cut": true}, {"index": 18, "stratum": "conflicted", "class": "information_loss", "native_output": "Charles Darwin", "arm_output": "Stephenie Meyer", "target_serial": 2203, "gold_cut": false}, {"index": 65, "stratum": "conflicted", "class": "information_loss", "native_output": "Catholic Church", "arm_output": "Montanism", "target_serial": 2237, "gold_cut": false}, {"index": 92, "stratum": "conflicted", "class": "information_loss", "native_output": "Blue Origin", "arm_output": "Aston Martin Lagonda", "target_serial": 1856, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.38, "band": [0.3, 0.5], "unique_native": 0.7714285714285715, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VOID: preregistered void condition 4_no_harmful_suppression: {"n_suppressed_harmful": 8, "n_suppressed_superseded": 524, "n_suppressed_same_value": 0} · A/A discordant 0 · token Δ -0.22338418332491194% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [49], "harms": [{"index": 49, "stratum": "unique", "class": "information_loss", "native_output": "The author of The Thorn Birds", "arm_output": "The author of The Song of Hiawatha", "target_serial": 3815, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.33, "band": [0.3, 0.5], "unique_native": 0.5588235294117647, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.