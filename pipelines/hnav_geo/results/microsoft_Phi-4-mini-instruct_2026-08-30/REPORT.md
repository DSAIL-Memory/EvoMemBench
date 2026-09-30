# hnav_geo — microsoft/Phi-4-mini-instruct

Run 2026-08-30T16:03:15.716908+00:00 · git `d34c10fcd5f7` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 40 | 76 | +36 | 2.91e-11 |
| sh_6k | calibration split | conflicted | 74 | 14 | 50 | +36 | 2.91e-11 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 50 | 66 | +16 | 0.0008554 |
| sh_32k | calibration split | conflicted | 65 | 16 | 33 | +17 | 0.0002213 |
| sh_32k | calibration split | unique | 35 | 34 | 33 | -1 | 1 |
| sh_64k | held-out | all | 100 | 46 | 52 | +6 | 0.146 |
| sh_64k | held-out | conflicted | 66 | 16 | 22 | +6 | 0.146 |
| sh_64k | held-out | unique | 34 | 30 | 30 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -2.869587363821015% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.5085686228619366% · harm {"n_harmed": 3, "counts": {"gold_cut": 2, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 2, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [12], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Washington, D.C.", "target_serial": 707, "gold_cut": true}, {"index": 9, "stratum": "conflicted", "class": "gold_cut", "native_output": "Rome", "arm_output": "Watertown", "target_serial": 1291, "gold_cut": true}, {"index": 12, "stratum": "unique", "class": "information_loss", "native_output": "Apple Inc.", "arm_output": "Microsoft", "target_serial": 775, "gold_cut": false}]}
- `sh_64k`: VOID: preregistered void condition 4_no_harmful_suppression: {"n_suppressed_harmful": 8, "n_suppressed_superseded": 524, "n_suppressed_same_value": 0} · A/A discordant 0 · token Δ -0.22338418332491194% · harm {"n_harmed": 3, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 9, "stratum": "conflicted", "class": "information_loss", "native_output": "Hiroshige", "arm_output": "Andrew Stanton", "target_serial": 2786, "gold_cut": false}, {"index": 20, "stratum": "conflicted", "class": "gold_cut", "native_output": "Europe", "arm_output": "Oceania", "target_serial": 2374, "gold_cut": true}, {"index": 86, "stratum": "conflicted", "class": "information_loss", "native_output": "Baseball", "arm_output": "Association Football", "target_serial": 4536, "gold_cut": false}]}

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.