# hnav_abtt_noparser — Qwen/Qwen3-4B-Instruct-2507

Run 2026-08-27T01:47:01.478738+00:00 · git `c7d12f869060` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 30 | 54 | +24 | 3.032e-06 |
| sh_6k | calibration split | conflicted | 74 | 4 | 28 | +24 | 3.032e-06 |
| sh_6k | calibration split | unique | 26 | 26 | 26 | +0 | 1 |
| sh_32k | calibration split | all | 100 | 53 | 72 | +19 | 6.604e-05 |
| sh_32k | calibration split | conflicted | 65 | 19 | 38 | +19 | 6.604e-05 |
| sh_32k | calibration split | unique | 35 | 34 | 34 | +0 | 1 |
| sh_64k | held-out | all | 100 | 45 | 59 | +14 | 0.0001221 |
| sh_64k | held-out | conflicted | 66 | 17 | 31 | +14 | 0.0001221 |
| sh_64k | held-out | unique | 34 | 28 | 28 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -1.5433402002846381% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 1, "stratum": "conflicted", "class": "information_loss", "native_output": "The Fairly OddParents", "arm_output": "Rurouni Kenshin", "target_serial": 259, "gold_cut": false}, {"index": 70, "stratum": "conflicted", "class": "information_loss", "native_output": "Canadair", "arm_output": "Westinghouse Electric", "target_serial": 356, "gold_cut": false}]}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.27241032135673204% · harm {"n_harmed": 2, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}, "by_stratum": {"conflicted": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 2}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 46, "stratum": "conflicted", "class": "information_loss", "native_output": "Terrance Dicks", "arm_output": "Rumiko Takahashi", "target_serial": 2120, "gold_cut": false}, {"index": 74, "stratum": "conflicted", "class": "information_loss", "native_output": "J-pop", "arm_output": "Jazz", "target_serial": 595, "gold_cut": false}]}
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.15005194760955837% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.