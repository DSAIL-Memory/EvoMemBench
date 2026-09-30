# hnav_idonly — google/gemma-3-4b-it

Run 2026-08-31T10:20:18.463797+00:00 · git `247ac2cf8bfa` · endpoint `http://localhost:8003/v1` · mechanism `detector_suppress`

| subset | role | stratum | n | native | H-Nav | net | McNemar p |
|---|---|---|---|---|---|---|---|
| sh_6k | calibration split | all | 100 | 45 | 89 | +44 | 1.137e-13 |
| sh_6k | calibration split | conflicted | 74 | 22 | 65 | +43 | 2.274e-13 |
| sh_6k | calibration split | unique | 26 | 23 | 24 | +1 | 1 |
| sh_32k | calibration split | all | 100 | 38 | 52 | +14 | 0.004344 |
| sh_32k | calibration split | conflicted | 65 | 11 | 23 | +12 | 0.01182 |
| sh_32k | calibration split | unique | 35 | 27 | 29 | +2 | 0.5 |
| sh_64k | held-out | all | 100 | 33 | 38 | +5 | 0.1797 |
| sh_64k | held-out | conflicted | 66 | 14 | 19 | +5 | 0.125 |
| sh_64k | held-out | unique | 34 | 19 | 19 | +0 | 1 |

## Guards

- `sh_6k`: VALID · A/A discordant 0 · token Δ -3.5247960952910904% · harm {"n_harmed": 0, "counts": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "by_stratum": {}, "protective_claim_void": false, "voiding_questions": [], "harms": []}
- `sh_32k`: VALID · A/A discordant 0 · token Δ -0.6423438869658674% · harm {"n_harmed": 4, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 3}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 3}}, "protective_claim_void": false, "voiding_questions": [], "harms": [{"index": 8, "stratum": "conflicted", "class": "gold_cut", "native_output": "London", "arm_output": "Saint Petersburg", "target_serial": 707, "gold_cut": true}, {"index": 18, "stratum": "conflicted", "class": "information_loss", "native_output": "Charles Darwin", "arm_output": "Stephenie Meyer", "target_serial": 2203, "gold_cut": false}, {"index": 65, "stratum": "conflicted", "class": "information_loss", "native_output": "Catholic Church", "arm_output": "Montanism", "target_serial": 2237, "gold_cut": false}, {"index": 92, "stratum": "conflicted", "class": "information_loss", "native_output": "Blue Origin", "arm_output": "General Motors", "target_serial": 1856, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.38, "band": [0.3, 0.5], "unique_native": 0.7714285714285715, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure
- `sh_64k`: VALID · A/A discordant 0 · token Δ -0.31396289925065496% · harm {"n_harmed": 2, "counts": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}, "by_stratum": {"conflicted": {"gold_cut": 1, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 0}, "unique": {"gold_cut": 0, "malformed_generation": 0, "refusal_after_edit": 0, "information_loss": 1}}, "protective_claim_void": true, "voiding_questions": [49], "harms": [{"index": 20, "stratum": "conflicted", "class": "gold_cut", "native_output": "Europe", "arm_output": "Asia", "target_serial": 2374, "gold_cut": true}, {"index": 49, "stratum": "unique", "class": "information_loss", "native_output": "The author of The Thorn Birds", "arm_output": "The author of The Song of Hiawatha", "target_serial": 3815, "gold_cut": false}]}
    - WARNING: void condition 2_native_in_band is out of band ({"native_overall": 0.33, "band": [0.3, 0.5], "unique_native": 0.5588235294117647, "unique_floor": 0.8}) - the band was preregistered for Qwen3-4B on sh_64k and must be re-preregistered for this model/subset; NOT treated as a validity failure

Strata from `stage0_results/question_strata.json` (parse-derived, model-independent). The conflicted stratum is the primary endpoint; the unique stratum is the do-no-harm check. Subsets are reported separately and are never pooled. One shot per model per subset: a void is reported, not re-rolled.