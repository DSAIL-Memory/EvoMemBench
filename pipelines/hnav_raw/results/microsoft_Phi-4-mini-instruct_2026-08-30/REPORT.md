# hnav_raw — microsoft/Phi-4-mini-instruct

Run 2026-08-30T13:47:18.969994+00:00 · git `d34c10fcd5f7` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 40 | 88 | +48 | 7.105e-15 |
| sh_6k | calibration split | conflicted | 74 | 14 | 62 | +48 | 7.105e-15 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 50 | 72 | +22 | 2.744e-05 |
| sh_32k | calibration split | conflicted | 65 | 16 | 39 | +23 | 5.648e-06 |
| sh_32k | calibration split | unique | 35 | 34 | 33 | -1 | 1 |
| sh_64k | held-out | all | 100 | 46 | 57 | +11 | 0.01921 |
| sh_64k | held-out | conflicted | 66 | 16 | 27 | +11 | 0.01921 |
| sh_64k | held-out | unique | 34 | 30 | 30 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.4768250588238345% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6301580283964697% · harm {"n_harmed": 3, "counts": {"gold_cut": 2, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 2, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [12], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Washington, D.C.", "target_serial": 707, "gold_cut": true}, {"index": 9, "stratum": "conflicted", "class": "gold_cut", "native_output": "Rome", "arm_output": "Watertown", "target_serial": 1291, "gold_cut": true}, {"index": 12, "stratum": "unique", "class": "information_loss", "native_output": "Apple Inc.", "arm_output": "Microsoft", "target_serial": 775, "gold_cut": false}]}
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.3067337601427156% · harm {"n_harmed": 4, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 3}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 3}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 9, "stratum": "conflicted", "class": "information_loss", "native_output": "Hiroshige", "arm_output": "Andrew Stanton", "target_serial": 2786, "gold_cut": false}, {"index": 11, "stratum": "conflicted", "class": "information_loss", "native_output": "Mahidol Adulyadej", "arm_output": "Clementine Churchill, Baroness Spencer-Church", "target_serial": 3548, "gold_cut": false}, {"index": 20, "stratum": "conflicted", "class": "gold_cut", "native_output": "Europe", "arm_output": "Oceania", "target_serial": 2374, "gold_cut": true}, {"index": 86, "stratum": "conflicted", "class": "information_loss", "native_output": "Baseball", "arm_output": "Association Football", "target_serial": 4536, "gold_cut": false}]}

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.