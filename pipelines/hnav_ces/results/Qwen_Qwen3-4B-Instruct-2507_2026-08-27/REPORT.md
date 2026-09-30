# hnav_ces — Qwen/Qwen3-4B-Instruct-2507

Run 2026-08-26T23:42:34.871626+00:00 · git `c7d12f869060` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 30 | 84 | +54 | 1.11e-16 |
| sh_6k | calibration split | conflicted | 74 | 4 | 58 | +54 | 1.11e-16 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 53 | 78 | +25 | 4.172e-07 |
| sh_32k | calibration split | conflicted | 65 | 19 | 44 | +25 | 4.172e-07 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 45 | 55 | +10 | 0.01294 |
| sh_64k | held-out | conflicted | 66 | 17 | 28 | +11 | 0.003418 |
| sh_64k | held-out | unique | 34 | 28 | 27 | -1 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -2.6242084686303158% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.4789806307414946% · harm {"n_harmed": 1, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 9, "stratum": "conflicted", "class": "gold_cut", "native_output": "Rome", "arm_output": "Watertown", "target_serial": 1291, "gold_cut": true}]}
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.16644051951394448% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 1, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 1, "information_loss": 0}}, "protective_claim_void": true, "voiding_questions": [77], "harms": [{"index": 69, "stratum": "conflicted", "class": "information_loss", "native_output": "California State University, Long Beach", "arm_output": "University of California, Berkeley", "target_serial": 4505, "gold_cut": false}, {"index": 77, "stratum": "unique", "class": "refusal_after_edit", "native_output": "John Milton", "arm_output": "The provided knowledge pool does not contain any information about", "target_serial": 2558, "gold_cut": false}]}

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.