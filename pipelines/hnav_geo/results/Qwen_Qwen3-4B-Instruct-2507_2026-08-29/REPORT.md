# hnav_geo — Qwen/Qwen3-4B-Instruct-2507

Run 2026-08-29T16:15:09.087625+00:00 · git `81cca60fe701` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 30 | 77 | +47 | 1.776e-13 |
| sh_6k | calibration split | conflicted | 74 | 4 | 51 | +47 | 1.776e-13 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 53 | 77 | +24 | 8.047e-07 |
| sh_32k | calibration split | conflicted | 65 | 19 | 43 | +24 | 8.047e-07 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 45 | 56 | +11 | 0.007385 |
| sh_64k | held-out | conflicted | 66 | 17 | 29 | +12 | 0.001831 |
| sh_64k | held-out | unique | 34 | 28 | 27 | -1 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -2.869587363821015% · harm {"n_harmed": 1, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 1, "stratum": "conflicted", "class": "information_loss", "native_output": "The Fairly OddParents", "arm_output": "Rurouni Kenshin", "target_serial": 259, "gold_cut": false}]}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.5085686228619366% · harm {"n_harmed": 1, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 9, "stratum": "conflicted", "class": "gold_cut", "native_output": "Rome", "arm_output": "Watertown", "target_serial": 1291, "gold_cut": true}]}
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.22338418332491194% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 1, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 1, "information_loss": 0}}, "protective_claim_void": true, "voiding_questions": [77], "harms": [{"index": 69, "stratum": "conflicted", "class": "information_loss", "native_output": "California State University, Long Beach", "arm_output": "University of California, Berkeley", "target_serial": 4505, "gold_cut": false}, {"index": 77, "stratum": "unique", "class": "refusal_after_edit", "native_output": "John Milton", "arm_output": "The provided knowledge pool does not contain any information about", "target_serial": 2558, "gold_cut": false}]}

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.