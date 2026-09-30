# Provenance of the numbers in `hnav_geo_manuscript.tex`

Every number in the manuscript comes from a committed artifact or from
`paper/derive_revision_numbers.py`, which recomputes the numbers added in the
revision from committed artifacts (stdlib only, about two seconds). The
manuscript itself no longer names files; this table replaces its former
Appendix B.

| manuscript location | number(s) | source |
| --- | --- | --- |
| Table 1 (stores), Section 3.2 overlap | 455 / 2,310 / 4,580 facts; sh_6k inside sh_32k; 2,244 shared with sh_64k | `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json`; `derive_revision_numbers.py` §Store overlap |
| Tables 5–7, Figure 4 (end-to-end) | accuracies, b/c, exact McNemar p, strata | `pipelines/hnav_geo/results/<model>_<date>/detector_gap_sh_{6k,32k,64k}.json` (`arms`, `paired_vs_native`, `by_stratum`) |
| Section 6.3 Holm correction; Appendix B flips | 11 of 15 survive Holm; unique-stratum b/c | `derive_revision_numbers.py` §McNemar |
| Table 7B, novel-fact rates | 24 seen / 42 novel; per-model counts | `derive_revision_numbers.py` §sh_64k retrieval misses and novel-fact slice |
| Section 6.3 retrieval misses | 22 of 66; 22/49 … 22/30 of residual errors | same section of the script |
| Table 2, Section 3.3 | error composition at sh_6k; clean-page control; 283/301 superseded | `derive_revision_numbers.py` §Table 2 |
| Section 3.3 eight earlier runs | 26/26 unique; 0–5/74 conflicted; 572/575 stale | `stage0_results/question_strata.json` (`runs`, `aggregate`); run files `stage0_results/t4_s2_evidence/sh_6k_*_results.json` |
| Section 5.4 run-to-run variation | 1/100 and 7/100 differing answers; 26–31 overall | `DURUM_RAPORU_STAGE0.md` (determinism checks detA–D); `question_strata.json` |
| Table 3 (models, KV cache, vLLM) | per-model serving | `pipelines/MULTI_MODEL_RUNBOOK.md` (per-model serving table) |
| Section 5.2 gemma-3 rerun | 4/26 unique under fp8; 0/10 vs 9/10 probe | `pipelines/hnav_geo/results/google_gemma-3-4b-it_2026-08-30_VOID_fp8_kv/VOID.md` |
| Section 5.2 preregistered endpoint | > 64 on sh_64k; GEO 56 | `stage0_results/geometry_filter/GEO_PREREG.md` (GG2) |
| Section 5.2 void flag | condition 4, 8 harmful on every sh_64k cell | `detector_gap_sh_64k.json` → `void_conditions`; `pipelines/MULTIMODEL_SUMMARY.md` |
| Section 5.1 development history | 162-cell chunk sweep; five earlier sh_64k arms | `STAGE1_NULL_ANALIZI.md`; comparison set in `GEO_PREREG.md` |
| Section 5.3 dataset | 87,102 / 54,569; tiers 2,388 / 282 / 12 / 105 / 51,782; coverage slices; judge settings | `stage0_results/conflict_pairs/AUDIT_SUMMARY.md`, `gold_conflict_dataset.summary.json` |
| Section 5.3 balanced sets, cosine-only AUC, band | 160 / 829 / 1,681; 0.960 / 0.911 / 0.893; 0.87–0.97 | `gold_conflict_dataset.summary.json` |
| Section 4.3 ABTT | 2,765 rows; D = 128; recall at precision 1.000 and ≥ 0.99; AUROC gain 0.002–0.005 | `stage0_results/abtt/G1_GATE_REPORT.md` §§2, 4, 5; `m6_abtt_dsweep.json` |
| Section 4.4 probe | 989 / 8,716 edits; C = 1; balanced | `stage0_results/geometry_filter/geo_identity_screen.json` (`provenance.probe`) |
| Section 4.4 negatives sharing the object | 5,124 of 8,716 | gold dataset, `parser` fields of calibration hard negatives |
| Section 4.5 anchors, scales, rule, raw thresholds | T_w, s_w, T_p, s_p; 193 + 1,332 pairs at cos ≥ 0.88 | `geo_identity_screen.json` (`scalars`, `provenance.anchors`); `hnav/geometry_filter/geo_artifact.py` |
| Section 4.5 operating point, grid | τ = (−0.40, +0.20); cos grid 0.90/0.92/0.94; NLI grid | `stage0_results/geometry_filter/geo_operating_point.json`; `hnav/stage1/calibrate_read_policy.py` (`COS_GRID`, `NLI_GRID`) |
| Section 4.5 recall ceiling | 26.3 / 23.5 / 24.2 % below 0.94 | `derive_revision_numbers.py` §Recall ceiling |
| Section 4.6–4.7 gate order, NLI on within-component pairs, span check | stages 1a, 1b, 2 | `hnav/core/read_gate.py` (`decide`) |
| Section 4.2 pooling | mean pooling, mirrors the benchmark retriever | `hnav/core/embedding.py` docstring; `MemoryAgentBench/methods/embedding_retriever.py` |
| Table 4, Figure 3, Section 6.1 | AUROCs, CIs, band, hard task, TPR tails, 95.9 % | `stage0_results/geometry_filter/geo_pairlevel.json` |
| Section 6.1 tail intervals | Wilson intervals on 752/1,045 and 306/636 | `derive_revision_numbers.py` §Tail and harm intervals |
| Section 6.2 pool detection | 2,157 deletions, precision 1.000, recall 0.790, 104/139 | `geo_operating_point.json` (`metrics`, `by_subset`) |
| Section 6.2 NLI without identity screen | 33–39 % cross-key at cos 0.94 | `TEZ_BULGULARI.md` §B |
| Section 6.2 screen without probe | pool recall 0.444 | `GEO_PREREG.md` (GG1); `abtt_noparser_operating_point.json` |
| Section 6.2 key erasures | 8 of 532; three keys; five questions | `stage0_results/geometry_filter/E2E4_COMPLEMENTARITY.md` §4; `detector_gap_sh_64k.json` |
| Table 8 (reference detectors) | oracle and probe-free accuracies | `pipelines/MULTIMODEL_SUMMARY.md` (`hnav_idonly`, `hnav_abtt_noparser`) |
| Section 6.4 plan containment | 527 / 532 / 735 | `E2E4_COMPLEMENTARITY.md` §4 |
| Table 9, Section 6.5 placement | three edits; per-question diagnosis | `detector_gap_*.json` (`detector_demote_late`, `detector_anti`); `derive_revision_numbers.py` §Placement |
| Table 10 (cost) | fired, deletions, pairs, prompt change | `detector_gap_*.json` (`tokens`, `per_question[].plan`) |
| Section 6.6 NLI passes | 410 / 4,868 / 8,956 directed | `TEZ_HIKAYESI.md` §3.1; counting code `hnav/stage1/calibrate_read_policy.py` (`directed_seen`) |
| Section 7.1, Table 11 | convergence 48–55; governed small vs native 9B | `detector_gap_*.json`; `derive_revision_numbers.py` §Governed small model |
| Section 7.2 cosine band | 0.955 / 0.956 / 0.956; 65,782 vs 1,687 | `stage0_results/delta_geometry/DELTA_GEOMETRY_REPORT.md` §2 |
| Section 7.2 NLI | 0.9995 / 0.9998; 447 of 453 | `TEZ_BULGULARI.md` §B; `GEO_PREREG.md` |
| Section 7.2 slot probe | macro-F1 0.70; relation-disjoint 0.21; AUROC 0.96; PCA 0.73 / 0.82 | `stage0_results/geometry_filter/REPORT.md` §2; `geo_pairlevel.json`; `stage0_results/qda_filter/REPORT.md` |
| Section 7.2 chunk reranking | 0 of 81 net positive; 228 vs 441 | `STAGE1_NULL_ANALIZI.md` §3 |
| Figure 7B, Section 7.3 | 54/48/16 kept; 20/15/27 untouched; 0/0/22 not in pool; 0/2/1 expected answer deleted | `detector_gap_*.json` (`pool`, `plan.demote_serials`, `plan.suppress_serials`); recipe in `paper/FIGURE_PACKAGE_BRIEF.md` §4.8 |
| Section 7.3 expected answer superseded | three questions, serials | `derive_revision_numbers.py` §Expected answer deleted |
| Section 8.3 oracle share | 50–81 % | Table 8 arithmetic |
| Section 8.4 write-time gate | earlier gate acts on almost no writes | `KAPI_KARARI.md` §2 |
| Appendix A | frozen parameters, fingerprint | `geo_identity_screen.json`, `geo_operating_point.json` |
| Section 2, 8.2 external scores | GPT-4o 60, GPT-5-mini 78; DeepSeek-V3.2 65, GPT-5-mini 77, Gemini-3-Flash 96; memory systems 18–54 | the cited papers; **not in the repository and not re-checked here** |
