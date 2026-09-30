# Figure package brief — H-Nav-GEO paper (instantiated master prompt)

Written 2026-09-17 from a full audit of branch `claude/hnav-presentation-evidence` at
`ca0d2c3`. This document instantiates the "MASTER PROMPT FOR CLAUDE DESIGN" for this
repository: it names the exact artifact, key and number behind every figure, states what is
and is not supported by committed data, and gives the execution plan. Where the master
prompt and this document differ on data availability, this document wins, because it was
checked against the files.

Everything below was verified by reading the artifacts. No number in this brief is from
memory. Where a number is quoted, the file and key are next to it so the plotting script
reads it live rather than hard-coding it.

---

## 0. How to use this brief

1. Read §1 (binding rules) and §2 (repository map) once.
2. §3 is the data-availability verdict per figure. Two figures (3 and 6) need inputs that
   are not in the repository; §3 says exactly what to obtain and what to ship if it cannot
   be obtained.
3. §4 is the per-figure specification: source paths, fields, numbers, visual form, caption
   draft, placement in the LaTeX manuscript.
4. §5 lists the manuscript edits, including four factual corrections the audit found.
5. §6 is the execution plan and the verification gate.
6. §7 is the house style with the semantic colour map.
7. Appendix A is the key-path cheat sheet for the plotting script.

Deliver: `paper/figures/final/fig{1..8}_<slug>.{pdf,svg,png}`,
`paper/figures/scripts/extract_data.py` + `make_figures.py` + `verify_numbers.py`,
`paper/figures/data/fig{1..8}.json` (every plotted number with its source path and key),
and the edited `paper/hnav_geo_draft.tex`.

---

## 1. Binding rules (from the repository, not negotiable)

These come from `CLAUDE.md` ("Thesis framing — binding, user decisions 2026-09-17") and
`THESIS_CLAIMS.md` §13 (wording rules). They apply to every caption, label and legend.

- **H-Nav-GEO is the method.** Arm name in the artifacts: `detector_suppress` inside
  `pipelines/hnav_geo/results/…`. The schema-keyed arms (`hnav_raw`, `hnav_idonly`,
  `hnav_abtt`) and the other screens (`hnav_ces`, `hnav_abtt_noparser`) are reference
  measurements only. They may appear as clearly labelled rejected alternatives or
  contextual references, never as the comparison the figure is about.
- **The comparison is native baseline vs H-Nav-GEO.** "Native" is the same answering
  model on the benchmark's own unedited page (`arms.native`), not a different model.
- **Never write "harm-free".** Say: "the do-no-harm (unique) stratum is unchanged to
  within one question in every cell."
- **Never write "significant everywhere".** Say: "positive in 15 of 15 cells, significant
  at the 5 % level in 11."
- **Never pool the five models into one p-value.** Per-cell exact McNemar only; consistency
  across models is a count of positive cells or a sign test.
- **Always label sh_6k and sh_32k "calibration split" and sh_64k "held-out, one shot".**
- **Never extrapolate to sh_262k**; it was not measured (excluded by design).
- **Never say thresholds transfer to another embedder**; they are refit, not copied.
- **Parser-free at inference.** The GEO screen reads only the two fact vectors
  (`geo_identity_screen.json → provenance.no_parser_at_inference`). Slot vocabulary
  (subject / relation / object) may be used for exposition only, with the mandated note.
  The gold dataset carries offline `parser.*` label fields; they are labels, not a runtime
  component, and must not be drawn as a pipeline step.
- **No multilingual claim.** The store is English; nothing in the repository tests another
  language.
- **The one limitation line stays** (`THESIS_CLAIMS.md` §12, `.tex` §8.5): on held-out
  sh_64k, 8 of 532 deletions merged two different keys. Harm is not a headline, but a
  figure that shows sh_64k accuracy must not hide it: one clause in the caption pointing
  to the Limitations section is the right weight.
- **Committed artifacts, void records and measured numbers are never edited to fit the
  framing.** The framing selects and orders them.

Additional constraints from the master prompt that the audit confirms are satisfiable:
no fabricated numbers; every plotted value must be re-derived from an artifact by the
script; external literature appears only in Figure 6B and only as "contextual comparison,
not head-to-head".

---

## 2. Repository map

### 2.1 Manuscripts

| file | role | notes |
| --- | --- | --- |
| `paper/hnav_geo_draft.tex` | **the manuscript to update** (achemso, `journal=jcisd8`, `manuscript=article`) | 533 lines. `\graphicspath{{./figures/}{./}}` (L20) points at `paper/figures/`, **which does not exist yet**. Zero `\includegraphics`; four `\placeholderfigure{}` boxes (L155, L268, L323, L424). Body-local `\code{}` macro defined after `\begin{document}`. |
| `MANUSCRIPT_DRAFT.md` | Markdown twin, 2026-09-17 | Same numbers, but **table numbering is one behind the .tex from the placement table onward** (the .md has no `tab:scale`). Do not carry cross-references from the .md into the .tex. |
| `THESIS_CLAIMS.md` | binding claims ledger | §4 has the cleanest plotting tables; §13 wording rules; §15 provenance table. |
| `LATEST_RESULTS_REPORT.md`, `TEZ_HIKAYESI.md` | dated records | Superseded on framing; still the source for some cost numbers (§3.1 of `TEZ_HIKAYESI.md`: NLI passes 410 / 4,868 / 8,956). |

Table labels in the .tex and their float numbers: `tab:stores` (1), `tab:models` (2),
`tab:pairlevel` (3), `tab:sh6k` (4), `tab:sh32k` (5), `tab:sh64k` (6), `tab:scale` (7),
`tab:placement` (8), `tab:cost` (9), `tab:frozen` (App. A), `tab:provenance` (App. B).

### 2.2 Artifact directories that back the figures

| directory | what is in it | backs |
| --- | --- | --- |
| `pipelines/hnav_geo/results/<model>_<date>/` | `detector_gap_sh_{6k,32k,64k}.json` (≈360–390 KB each), `run_manifest.json`, `REPORT.md`; reference dir also has `e2e3_comparison.json` | Figs 2 (inset), 4, 5, 6A, 8A, 8B |
| `stage0_results/geometry_filter/` | `geo_pairlevel.json`, `geo_operating_point.json`, `geo_identity_screen.{json,npz}`, `slot_probe.json`, `conflict_detection.json`, `dimension_ideas.json`, `nuisance_analysis.json`, `null_baselines.json`, `E2E4_COMPLEMENTARITY.md`, `GEO_PREREG.md`, `REPORT.md` | Figs 2, 3, 7 |
| `stage0_results/abtt/` | `m6_abtt_dsweep.json`, `m6_abtt_geometry.json`, `G1_GATE_REPORT.md`, `ABTT_REPORT.md`, `abtt_whitening_D128.json` | Fig 7A |
| `stage0_results/delta_geometry/` | `m7_delta_geometry.json`, `DELTA_GEOMETRY_REPORT.md` | Fig 1 inset |
| `stage0_results/conflict_pairs/` | `gold_conflict_dataset.jsonl.gz` (54,569 records), `gold_conflict_dataset.summary.json` | Figs 1, 3 |
| `stage0_results/question_strata.json` | per-question stratum indices per subset | Figs 4, 5, 8 |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json` | the benchmark data (3.29 MB, present, not an LFS pointer); list of 8 entries; **entry 4 is sh_6k**, 5 is sh_32k, 6 is sh_64k; `context` is `"<serial>. <fact>\n"` lines; `questions[i]`/`answers[i]` align with `per_question[i].index` | Figs 1, 2 (fact texts by serial) |
| `presentation_evidence/` | evidence pack dated 2026-08-18/19 for the **parser-era** detector | House style only (`_scripts/make_charts.py`). **Do not reuse its figures**: they depict a detector that parses at inference. |

### 2.3 The six result directories

Five valid models plus one void run. `run_manifest.json → llm_model` is the model key.

| directory | `llm_model` | use |
| --- | --- | --- |
| `Qwen_Qwen3-4B-Instruct-2507_2026-08-29` | Qwen/Qwen3-4B-Instruct-2507 | reference model |
| `Qwen_Qwen3.5-9B_2026-08-31` | Qwen/Qwen3.5-9B | yes |
| `google_gemma-3-4b-it_2026-08-31` | google/gemma-3-4b-it | yes (BF16 re-run) |
| `google_gemma-4-E2B-it_2026-08-31` | google/gemma-4-E2B-it | yes |
| `microsoft_Phi-4-mini-instruct_2026-08-30` | microsoft/Phi-4-mini-instruct | yes |
| `google_gemma-3-4b-it_2026-08-30_VOID_fp8_kv` | google/gemma-3-4b-it | **never** (broken serving config, `VOID.md`) |

Skip any directory whose name contains `_VOID`, exactly as
`hnav/geometry_filter/multimodel_summary.py` does.

---

## 3. Data-availability verdict per figure

| figure | verdict | what is missing | ship if missing |
| --- | --- | --- | --- |
| 1 Conflict anatomy | **fully supported** | — | — |
| 2 Pipeline | **fully supported** (worked-example inset verified) | — | — |
| 3 ROC + decision space | **A: scalars only; B: needs recomputation** | per-pair scores; `geo_pairlevel.json` stores no ROC curve points and no per-pair table; the embedding cache `hnav/_cache/emb/` (4,499 `.npy`) is gitignored and absent | 3A as an AUROC dot plot with 95 % CIs plus the TPR-at-FPR inset (all scalars committed); 3B only if scores are recomputed (§3.1), otherwise omit 3B rather than draw a schematic with fake points |
| 4 End-to-end 5 models | **fully supported** | — | — |
| 5 Stratified | **fully supported** | — | — |
| 6 Scale context | **A supported; B needs external transcription** | the MemoryAgentBench Table 3 values are recorded nowhere in the repository; arxiv.org and its mirrors were unreachable from the audit sandbox | 6A alone; 6B only after the design team transcribes Table 3 into a committed JSON with citation (§3.2) |
| 7 Mechanism | **fully supported** | — | — |
| 8 Deletion vs placement + funnel | **A fully supported; B supported for 4 of 6 stages** | per-pair screen scores and NLI verdicts (the "prepass" tables live only on the GPU box under `hnav/_out/`, not committed), so "cosine-threshold miss vs GEO-screen miss vs NLI veto" cannot be separated | the four-stage funnel that is exactly reconstructible (§4.8) |

### 3.1 Recomputing per-pair scores for Figure 3 (and the ROC curves)

The computation is one function call; the inputs are the problem.

Present in the repository:
- labels: `stage0_results/conflict_pairs/gold_conflict_dataset.jsonl.gz`
  (sha256 `f9be7cae6323149bf6a95cdc3cf52a40d907c29b6c7341a8c6e32bf6655979e7`),
  fields `pair_id, subset, split, tier, in_eval_set, gold_update, gold_strict, fact_a,
  fact_b, fact_a_id, fact_b_id, cosine_similarity, parser{…}, judge{…}`;
- weights: `stage0_results/geometry_filter/geo_identity_screen.npz`
  (`probe_w (2560,)`, `abtt_mean (2560,)`, `abtt_components (128, 2560)`) and
  `geo_identity_screen.json` scalars `probe_b, T_w, T_p, s_w, s_p`, fingerprint
  `335b4540d61cf6b79a7ead27cc7b107b84b5646a3a7618e53abfc943e89ccb2a` (the loader refuses a
  mismatch);
- code: `hnav/geometry_filter/data.py` (`load_records`, `fact_matrix`, `build_spaces`,
  `PairView`, `is_hard_negative`), `hnav/geometry_filter/geo_artifact.py`
  (`GeoIdentityScreen.load`, `.margins_pair(va, vb) -> (m_w, m_p)`),
  `hnav/geometry_filter/run_geo_pairlevel.py` (the four lines that compute
  `cos_w`, `probe`, `geo` for every record, then discard them).

Absent:
- `hnav/_cache/emb/`: 4,499 files named
  `sha256("Qwen_Qwen3-Embedding-4B|float32|L8192" + "||" + fact_text).hexdigest() + ".npy"`,
  float32, 2,560-d. Provenance in `audit_candidates_cos080.summary.json` names the
  original locations (the author's workstation and the `ozonderlab2` box). **Obtain the
  cache from the authors; do not re-embed** unless model, dtype (float32) and
  `max_length` 8192 are matched exactly, because a dtype change moves every cosine and
  every threshold.
- `numpy` is not installed in the audit sandbox; the design environment needs numpy and
  matplotlib (scikit-learn only for the PCA rows, which are already stored as scalars).

Recipe (write it as `paper/figures/scripts/compute_pair_scores.py`):

```
records = data.load_records()
index, V = data.fact_matrix(records)            # needs hnav/_cache/emb/
V_abtt = data.build_spaces(V)["abtt"]
pv = data.PairView(records, index)
art, _ = GeoIdentityScreen.load(ARTIFACT_JSON)  # fingerprint-checked
D = V[pv.ib] - V[pv.ia]; Dh = D / norm(D)
probe = abs(Dh) @ art.probe_w + art.probe_b
cos_w = pv.cos(V_abtt)
geo   = minimum((cos_w - art.T_w)/art.s_w, (probe - art.T_p)/art.s_p)
```

Write `paper/figures/data/fig3_pair_scores.csv` with columns
`pair_id, subset, split, tier, in_eval_set, gold_update, is_hard_negative, cosine_raw,
cos_w, probe, m_w, m_p, geo`. Plain CSV, not `.gz`: the repo-root `*.gz` ignore rule
would swallow a gzipped file. The eval-set rows (5,340) plus the confirmatory hard task
(1,681 gold vs 39,215 hard negatives) are enough; the whole 54,569-row table is fine if
under ~10 MB.

Decision rectangle in raw units (from `geo_identity_screen.json` scalars and
`geo_operating_point.json → ces.tau = "-0.4:0.2"`):
`cos_w ≥ T_w − 0.40·s_w = 0.6396504 − 0.40·0.3483433 = 0.5003` and
`probe ≥ T_p + 0.20·s_p = −0.0763662 + 0.20·1.0675459 = 0.1372`.
Anchors: `T_w = 0.6396504004901709`, `s_w = 0.3483432583731584`,
`T_p = −0.07636616045305258`, `s_p = 1.0675458709838188`, `probe_b = 0.0616016760468483`.

### 3.2 External reference values for Figure 6B

The .tex (§1, §2.3, §6.4, §8.2, §9, Table 7) cites "GPT-4o 60 % and GPT-5-mini 78 % FC-SH
from MemoryAgentBench" and "65 % DeepSeek-V3.2, 77 % GPT-5-mini, 96 % Gemini-3-Flash from
EvoMemBench". **None of these values is recorded in any repository file**; they exist only
in the pasted .tex. The master prompt instead names GPT-4o-mini as the reference backbone
of arXiv 2507.05257 Table 3. The audit sandbox could not reach arxiv.org, huggingface.co or
alphaxiv.org (egress blocked), so nothing was verified.

Required before drawing 6B:
1. Open arXiv 2507.05257 (use the version the paper cites, v4) and transcribe Table 3
   ("Overall Performance Comparison") into `paper/figures/data/external_reference.json`:
   `{source, version, table, retrieved_on, column_used, backbone_note, rows: [{method,
   backbone, fc_sh, fc_mh}]}`, with the note about GPT-4o-mini being the backbone when no
   model is specified quoted verbatim.
2. Record which context lengths the FC-SH column aggregates. The .tex asserts it includes
   262K; confirm or correct the manuscript accordingly.
3. Reconcile with the .tex: if Table 3 has GPT-4o and GPT-5-mini rows at 60 and 78, the
   manuscript stands; if not, the manuscript sentences in §1, §2.3, §6.4, §8.2, §9 and the
   abstract must change to what the table says. The EvoMemBench numbers (65 / 77 / 96)
   are outside the master prompt's allowed source; either verify them from arXiv
   2605.18421 the same way or drop them from 6B (they can stay in prose if verified).
4. Label the panel "contextual comparison, not head-to-head; different experimental
   context and source (MemoryAgentBench Table 3)".

If step 1 cannot be completed, ship Figure 6 as panel A only and leave Table 7's external
column as text.

---

## 4. Figure specifications

Numbering below is the manuscript numbering. All source paths are relative to the repo
root. "Reference model" = `Qwen_Qwen3-4B-Instruct-2507_2026-08-29`.

### 4.1 Figure 1 — Conflict anatomy and parser-free motivation

**Placement:** §3.4 "The identity decision" (after `.tex` L132–134), label `fig:anatomy`.
Referenced from §1 and §3.4.

**Example blocks (all real sh_6k facts, serials from `Conflict_Resolution.json` entry 4):**

| block | serial | sentence | evidence |
| --- | --- | --- | --- |
| unique | 2 | `Amy Winehouse died in the city of Camden Town.` | only fact with this (subject, relation) in sh_6k; visible in `presentation_evidence/data/item01_page_excerpt.txt` |
| true conflict, older | 0 | `Thomas Kyd was born in the city of London.` | `gold_conflict_dataset.jsonl.gz` `pair_id "sh_6k:0-306"`, tier `core`, `gold_update` and `gold_strict` true, `in_eval_set` true, `cosine_similarity` **0.984333** |
| true conflict, newest | 306 | `Thomas Kyd was born in the city of Leeds.` | same record; `parser.superseding_serial` 306 |
| look-alike, must stay | 342 | `Thomas Arne was born in the city of London.` | `pair_id "sh_6k:0-342"`, tier `negative`, `gold_update` false, `in_eval_set` true, `parser.same_relation` true, `same_object` true, `same_subject` false, `cosine_similarity` **0.834235** |

Bonus for the same panel: serial 366 `Thomas Arne was born in the city of Bengaluru.`
(342 → 366 is a second version pair); cross pairs 0–366 cosine 0.805523
(`sh_6k:0-366`) and 306–342 cosine 0.822113 (`sh_6k:306-342`), both tier `negative`.
Four sentences, two version pairs, one look-alike, all artifact-backed.

Alternative true-conflict pair if a different subject is preferred: serial 91
`Nobuhiro Watsuki is famous for Rurouni Kenshin.` → 259 `… The Fairly OddParents.`,
`sh_6k:91-259`, cosine 0.911612. (Do not use this question as a GEO success story: in the
GEO artifact q1 is the single `information_loss` harm at sh_6k.)

**Corrections this figure forces in the manuscript (see §5):** the .tex writes the Kyd
supersession as London → *Paris* (the store says Leeds) and pairs it with *Marlowe*, who
does not occur in sh_6k, sh_32k or sh_64k, nor in any of the 54,569 gold pairs. The .tex
also claims "both pairs have cosine well above 0.9"; the real look-alike sits at 0.834.
The figure must show the real pairs and the real cosines.

**Slot decomposition:** colour Subject / Relation / Object in each sentence. Mandated
note (verbatim): *"Semantic slots are shown for exposition only; H-Nav-GEO does not parse
facts into these fields at inference time."* The contrast to state: true conflict = same
subject + same relation + different object; look-alike = different subject + same
relation + same object.

**Cosine inset (committed data, no embeddings needed):**
- From `gold_conflict_dataset.jsonl.gz`, eval-set rows (`in_eval_set == true`), per
  subset, positives (`gold_update`) vs negatives. sh_64k: positives p10 / p50 / p90 =
  0.9166 / 0.9639 / 0.9839; negatives 0.8928 / 0.9094 / 0.9360, max 0.9934. sh_6k:
  0.9129 / 0.9637 / 0.9843 vs 0.8347 / 0.8592 / 0.9080. sh_32k: 0.9153 / 0.9638 / 0.9842
  vs 0.8821 / 0.8991 / 0.9324. Recompute from the file; the summary twin is
  `gold_conflict_dataset.summary.json → per_subset.<s>.eval`.
- Cosine-only AUC on the balanced sets: 0.9598 / 0.9109 / 0.893
  (`summary.json → per_subset.<s>.eval.cosine_only_auc`).
- Overlap band 0.87–0.97 (`conflict_detection.json → provenance.band`); pairs inside the
  band on sh_64k: 998 positives / 1,644 negatives (`conflict_detection.json →
  spaces.raw.balanced_eval.sh_64k.band_n_pos / band_n_neg`).
- The look-alike count: `m7_delta_geometry.json → subsets.sh_64k.raw.pair_space`:
  `n_same_key 1687`, `conflict_cos_min 0.8011`, `n_eligible_above_conflict_min 65782`,
  `nonconflict_cos_max 0.9934`. Same for sh_6k (306 above the floor vs 160) and sh_32k
  (6,716 vs 835), `DELTA_GEOMETRY_REPORT.md` §2.
- Class means: `m7_delta_geometry.json → subsets.sh_64k.raw.sets.<set>.cos.mean`:
  conflict 0.9557, cos_matched 0.9246, same_subject 0.7960, same_relation 0.7142,
  random 0.6052. Ready 7-number summaries (n, mean, sd, min, p10, p50, p90, max) exist for
  every set; a quantile strip or a small violin from these summaries is honest, a
  full-resolution histogram is not (no raw arrays).

**Rationale box (text only):** real memory is free text; a template or schema parser
needs hand-written templates per domain or an LLM extraction call per write; H-Nav-GEO
therefore uses only what every vector store has, the embedding and the recency key
(`.tex` §3.1 constraints i–iii, §8.3).

**Caption draft:** "Conflict anatomy in an evolving store. (A) Three kinds of retrieved
fact from the sh_6k store: a single-version fact; a version pair, where a later serial
restates the same subject and relation with a new object (cosine 0.984); and a look-alike
pair, where a different subject shares the relation and object (cosine 0.834). Semantic
slots are shown for exposition only; H-Nav-GEO does not parse facts into these fields at
inference time. (B) Raw cosine of true conflicts and of cosine-matched non-conflicts on
the held-out balanced set: the distributions overlap in the 0.87–0.97 band, and on
sh_64k 65,782 non-conflict pairs lie above the lowest conflict cosine (39 per conflict).
Sources: `gold_conflict_dataset.jsonl.gz`, `m7_delta_geometry.json`."

### 4.2 Figure 2 — H-Nav-GEO pipeline

**Placement:** replaces the `\placeholderfigure` at `.tex` L155 (§4.1), label
`fig:pipeline`.

**Stages and frozen constants** (all from `geo_operating_point.json` and
`geo_identity_screen.json`; also `pipelines/hnav_geo/pipeline.json →
frozen_thresholds_for_reference`):

1. Retrieved page (benchmark retriever, untouched; `page_source = "benchmark"`).
2. Query-similar pool: 50 facts (`per_question[].n_pool == 50`).
3. Candidate pairs: raw cosine ≥ 0.94 (`thresholds.cos_pair`).
4. GEO screen: whitened cosine (ABTT D = 128, mean + 128 directions removed) and slot
   probe (logistic on |d̂|, 2,560 magnitudes); rectangle `m_w ≥ −0.40`, `m_p ≥ +0.20`,
   i.e. `cos_w ≥ 0.500`, `probe ≥ 0.137`.
5. NLI verification: `cross-encoder/nli-deberta-v3-large`, contradiction ≥ 0.90 in both
   directions (`thresholds.nli_contradiction`).
6. Conflict groups: connected components; span check `r_min < 0.44`
   (`thresholds.r_min`, label "loose").
7. Delete stale versions: keep the highest serial per group (`plan.demote_serials` are
   the kept newest carriers; `plan.suppress_serials` are the deletions).
8. Cleaned page → answering model (benchmark prompt, temperature 0, 10 output tokens).

One-line purposes as the master prompt lists them. Emphasis cues: "no LLM in steps 1–7"
is accurate (the .tex says steps 1–6 with the model as step 7; use the manuscript's
numbering); "deterministic suppression plan, byte-identical across the five models"
(verified: per-subset plan hashes are identical in all six result directories).

**Worked-example inset (artifact-verified, not cherry-picked: it is one of 48 sh_6k
questions that flip native-wrong → GEO-correct on the reference model):**
`pipelines/hnav_geo/results/Qwen_Qwen3-4B-Instruct-2507_2026-08-29/detector_gap_sh_6k.json
→ results[0].per_question[12]`.
- Question (`Conflict_Resolution.json` entry 4, `questions[12]`): *"What position does
  Robert Parish play?"*; `key = ["| plays the position of ", "Robert Parish"]`;
  `stratum = "conflicted"`; `truths = ["quarterback"]`; `target_serial = 334`.
- Store: serial 330 `Robert Parish plays the position of center.` (stale) → serial 334
  `Robert Parish plays the position of quarterback.` (newest).
- `arms.native.output = "center"` (wrong, the stale value); `arms.native_repeat.output =
  "center"`; `arms.detector_suppress.output = "quarterback"` (correct);
  `arms.detector_demote_late.output = "center"`; `arms.detector_anti.output = "center"`.
- Plan: `n_groups 15`, `n_pairs_verified 15`; `suppress_serials` includes 330;
  `demote_serials` includes 334; page 454 → 439 facts (`arms.native.n_facts`,
  `arms.detector_suppress.n_facts`, as stored); prompt 26,970 → 26,021 characters
  (`arms.<arm>.prompt_chars`) for the "prompt gets shorter" annotation. All five values
  were re-read from the record during the audit.
- Independent cross-check: `presentation_evidence/data/item05_paired_examples.json`
  entry `sh_6k / index 12` (oracle probe) has `gold_serials [334]`, `stale_serials [330]`.

**Caption draft:** "H-Nav-GEO at read time. The benchmark's retriever produces the page;
the 50 facts most similar to the query form the pool; pairs above raw cosine 0.94 enter
the GEO screen, which reads only the two vectors: a whitened cosine (store mean and 128
shared directions removed) and a slot probe on the sign-invariant edit direction. Pairs
that clear both anchored margins are verified by a bidirectional NLI cross-encoder,
chained into groups, and every member except the highest serial is deleted. No LLM runs
before the answering model; the suppression plan is a deterministic function of the store
and the frozen detector and was byte-identical across the five answering models. Inset:
sh_6k question 12, where the reference model answers the stale value 'center' natively
and 'quarterback' after serial 330 is deleted."

### 4.3 Figure 3 — Pair-level discrimination

**Placement:** replaces the `\placeholderfigure` at `.tex` L268 (§6.1), label
`fig:pairlevel`. Two panels.

**Panel A (committed scalars; ROC *curves* only if §3.1 is done).**
Source `stage0_results/geometry_filter/geo_pairlevel.json`. Screen keys: raw cosine =
`campaign_cos`, whitened cosine = `abtt_cos`, slot probe = `probe`, H-Nav-GEO combined =
`geo`. (`ces` is the rejected schema-assisted alternative; `probe_pca64/256` belong to
Figure 7B.)

Held-out balanced set sh_64k (`balanced.sh_64k`, 1,681 vs 1,681, `in_sample_for_fit =
false`), `auroc` with `auroc_ci95` (1,000 bootstrap, seed 20260824) and `band_auroc`
(overlap band 0.87–0.97):

| screen | AUROC [95 % CI] | band AUROC |
| --- | --- | --- |
| raw cosine | 0.8930 [0.8816, 0.9037] | 0.8498 |
| whitened cosine | 0.9648 [0.9594, 0.9697] | 0.9516 |
| slot probe | 0.9131 [0.9038, 0.9228] | 0.9020 |
| **H-Nav-GEO** | **0.9716 [0.9668, 0.9761]** | **0.9657** |

CIs exist for sh_64k only; sh_6k / sh_32k are in-sample for the fit and have point values
only (`geo` 0.9927 / 0.9867). Show sh_64k as the main panel; the calibration values may
appear as small grey ticks labelled "in-sample".

Inset, hard confirmatory task (`hard_confirmatory`, 1,681 gold vs 39,215 same-relation
different-subject negatives), TPR at FPR 10⁻⁴, seen vs unseen object transitions:
geo 0.7196 / 0.4811; whitened cosine 0.5072 / 0.4041; probe 0.2555 / 0.0739; raw cosine
0.3722 / 0.2138 (keys `tpr_seen_fpr_0.0001`, `tpr_unseen_fpr_0.0001`). Also available at
10⁻³. Hard-task AUROC / AP for geo: 0.9984 / 0.9784; inverted-order win rate vs raw
cosine 0.9592 over 527,062 comparisons (`inverted_vs_campaign_cos`).

If per-pair scores are recomputed, draw the four ROC curves from
`fig3_pair_scores.csv` and verify that the recomputed AUROCs equal the stored ones to
four decimals before using them (this is the correctness gate for the recomputation).
Confidence bands: only if computed by bootstrap over the recomputed scores; the file
stores no bands.

**Panel B (needs §3.1).** x = whitened cosine `cos_w`, y = slot-probe logit `probe`,
positives = `gold_update` (colour: true conflict), hard negatives =
`is_hard_negative` (tier `negative`, both parse, same relation, different subject),
drawn on the sh_64k eval set or the hard task. Decision rectangle at `cos_w = 0.5003`,
`probe = 0.1372`; anchors `(T_w, T_p) = (0.6397, −0.0764)` marked. Annotations:
"cosine-alone overlap", "geometry separates", "operating point". Use alpha or 2-D
density; 40,896 hard-task points need thinning or hexbin. If scores are not available,
omit Panel B and say so in the caption; do not draw illustrative points.

**Caption draft:** "Pair-level discrimination between true conflicts and cosine-matched
look-alikes on the held-out sh_64k evaluation set (1,681 vs 1,681). (A) AUROC with 95 %
bootstrap intervals for raw cosine, whitened cosine, the slot probe and the combined
H-Nav-GEO screen; the inset gives the true-positive rate at a false-positive rate of
10⁻⁴ on the hard task (1,681 gold vs 39,215 same-relation different-subject negatives),
split by whether the object transition was seen in calibration. (B) The screen's
decision space: whitened cosine against slot-probe logit, with the frozen operating
rectangle. Source: `geo_pairlevel.json`; scores recomputed from the frozen artifact
`geo_identity_screen.npz` and the campaign embeddings."

### 4.4 Figure 4 — End-to-end accuracy across five answering models

**Placement:** replaces the `\placeholderfigure` at `.tex` L323 (§6.3), label `fig:e2e`.

**Source:** each valid `pipelines/hnav_geo/results/<dir>/detector_gap_sh_{6k,32k,64k}.json`.
Compute overall accuracy from `results[0].per_question[].arms.native.correct` and
`.arms.detector_suppress.correct` (this is what `multimodel_summary.py` does; do not read
`results[0].arms` blindly, though they agree). Paired statistics:
`results[0].paired_vs_native.detector_suppress → {b_native_only, c_arm_only, net,
p_exact}`. A/A floor: `results[0].aa_floor` (0 discordant in all 15 cells).

Expected values (must equal `.tex` Tables 4–6; the verify script asserts this):

| model | sh_6k native → GEO | sh_32k | sh_64k (held-out) |
| --- | --- | --- | --- |
| gemma-3-4b-it | 45 → 76 (+31, p 3.7e-08) | 38 → 45 (+7, p 0.143) | 33 → 36 (+3, p 0.375) |
| gemma-4-E2B-it | 40 → 72 (+32, p 6.7e-08) | 44 → 58 (+14, p 2.6e-03) | 37 → 41 (+4, p 0.344) |
| Phi-4-mini-instruct | 40 → 76 (+36, p 2.9e-11) | 50 → 66 (+16, p 8.6e-04) | 46 → 52 (+6, p 0.146) |
| Qwen3-4B-Instruct-2507 | 30 → 77 (+47, p 1.8e-13) | 53 → 77 (+24, p 8.0e-07) | 45 → 56 (+11, p 7.4e-03) |
| Qwen3.5-9B | 39 → 81 (+42, p 4.5e-13) | 61 → 86 (+25, p 1.6e-06) | 51 → 62 (+11, p 3.4e-03) |

**Form:** three aligned panels (sh_6k, sh_32k, sh_64k), one paired dot / slope line per
model, native → GEO, with the net gain and a significance marker (p < 0.05) per cell.
Model order by parameter count or by native score; keep the same order in Figures 5, 6
and 8.

**Caption draft:** "Overall accuracy (of 100) of the native baseline and H-Nav-GEO for five
answering models at three store sizes; sh_6k and sh_32k are the calibration split, sh_64k
is held out and was run once per model. Every cell is question-paired; the A/A floor
(native vs native repeat) was 0 discordant questions in all 15 cells. H-Nav-GEO is
positive in 15 of 15 cells and significant at the 5 % level in 11 (exact McNemar).
On sh_64k, 8 of the 532 deletions merged two different keys (Section 8.5). Source:
`detector_gap_sh_*.json`."

### 4.5 Figure 5 — Stratified accuracy: conflicted vs unique

**Placement:** §6.3, after Figure 4 and the "Three observations" paragraph, label
`fig:strata`.

**Source:** same files; `results[0].by_stratum.{conflicted,unique}.arms.{native,
detector_suppress}` and `.paired_vs_native.detector_suppress.p_exact`, or recompute from
`per_question[].stratum`. Stratum sizes: conflicted 74 / 65 / 66, unique 26 / 35 / 34
(`results[0].strata_counts`, identical to `stage0_results/question_strata.json →
subsets[].counts`). Strata are parse-derived offline and model-independent.

Expected values (from `.tex` Tables 4–6): conflicted native → GEO, sh_6k: 22→53, 16→48,
14→50, 4→51, 13→55; sh_32k: 11→18, 17→31, 16→33, 19→43, 27→52; sh_64k: 14→17, 21→25,
16→22, 17→29, 24→36. Unique: unchanged in every cell except Phi-4-mini sh_32k 34→33,
Qwen3-4B sh_64k 28→27, Qwen3.5-9B sh_64k 27→26.

**Form:** dumbbells per model, two rows (conflicted, unique) × three columns (subsets),
x = correct count with the stratum size on the axis; or a small-multiple grid. Native and
GEO keep their colours; strata are separated by panel, not by colour.

**Caption draft:** "The gain is concentrated where the failure lives. Conflicted stratum
(top; n = 74 / 65 / 66) and unique stratum (bottom; n = 26 / 35 / 34), native vs
H-Nav-GEO, per model and store size. The do-no-harm (unique) stratum is unchanged to
within one question in every cell. Source: `detector_gap_sh_*.json`,
`question_strata.json`."

### 4.6 Figure 6 — Small-model gains with contextual frontier reference

**Placement:** §6.4 next to Table 7 (`tab:scale`), label `fig:scale`.

**Panel A (supported):** per-model mean over sh_6k, sh_32k, sh_64k of the overall
accuracies in Figure 4. Native: 38.7 / 40.3 / 45.3 / 42.7 / 50.3; H-Nav-GEO: 52.3 / 57.0 /
64.7 / 70.0 / 76.3 (gemma-3, gemma-4-E2B, Phi-4-mini, Qwen3-4B, Qwen3.5-9B). Compute as
arithmetic means of the three cells; label the axis "mean over the evaluated 6K–64K
conditions" and say in the caption that this is a descriptive mean, not a benchmark
aggregate.

**Panel B (only after §3.2):** the same governed means against a shaded zone or marker set
derived from the transcribed Table 3 values, labelled "contextual comparison, not
head-to-head; different experimental context and source". No parity or superiority
wording anywhere in the figure.

**Caption draft (A only):** "Descriptive mean accuracy over the three evaluated single-hop
conditions, native vs H-Nav-GEO, per answering model. (B, if present) The governed means
placed against the FC-SH values reported in MemoryAgentBench Table 3 (arXiv 2507.05257):
a contextual comparison, not head-to-head; the published aggregates use different
context lengths and inference pipelines."

### 4.7 Figure 7 — Mechanism diagnostics

**Placement:** replaces the `\placeholderfigure` at `.tex` L424 (§7.2), label
`fig:mechanism`.

**Panel A — recall at precision 1.000, raw vs whitened, calibration subsets.**
Source `stage0_results/abtt/m6_abtt_dsweep.json`:
`subsets.<s>.raw.grouping.recall_at_precision.recall_at_precision_1` vs
`subsets.<s>.regimes["frozen_global|D=128"].grouping.recall_at_precision.recall_at_precision_1`.
sh_6k 0.0750 → 0.5125; sh_32k 0.00719 → 0.2910. Companion bars at P ≥ 0.99
(0.0750 → 0.7188; 0.0072 → 0.5210), P ≥ 0.95 (0.4437 → 0.8438; 0.6611 → 0.6946) and
P ≥ 0.90 (0.8625 → 0.9187; 0.7856 → 0.8419) are in the same keys (`…_0.99`, `…_0.95`,
`…_0.9`). Optional D-sweep line (frozen_global, `recall_at_precision_1`): sh_6k D = 16 …
192 → 0.344, 0.438, 0.406, 0.475, 0.494, 0.481, **0.513**, 0.488; sh_32k → 0.080, 0.131,
0.187, 0.231, 0.249, 0.260, **0.291**, 0.272 (`config.d_grid = [16,24,32,48,64,96,128,192]`).
AUROC context (why whitening is easy to dismiss): raw grouping AUC 0.9964 / 0.9917,
delta at D = 128 +0.0015 / +0.0041 (`auc_delta_vs_raw`).

**Panel B — slot-probe AUROC under PCA compression.**
Source `geo_pairlevel.json → balanced.sh_64k.methods.{probe_pca64, probe_pca256, probe}`:
0.7263 [0.7095, 0.7441], 0.8208 [0.8075, 0.8351], 0.9131 [0.9038, 0.9228] for 64, 256, all
2,560 dimensions. Calibration values (no CI): sh_6k 0.7007 / 0.8754 / 0.9879, sh_32k
0.7191 / 0.8412 / 0.9697. `pca_note` confirms the PCA was calibration-fit on |d̂| before
the probe.

**Optional mini-panel C (choose one, all committed):**
- Anisotropy removal: random-pair mean cosine 0.6024 → +0.0002 (sh_6k), 0.6026 → −0.0009
  (sh_32k), candidate-pair floor 0.5815 → 0.0615 and 0.6130 → 0.0810
  (`m6_abtt_dsweep.json → subsets.<s>.{raw, regimes[...]}.anisotropy_random_pairs.mean`,
  `.grouping.candidate_floor_min_cos`).
- Sign invariance: object-vs-subject AUROC of the probe on |d̂| 0.9622 vs a signed-feature
  probe with randomised orientation 0.4953; six-class macro-F1 0.6999 vs 0.2223, chance
  ≈ 0.17 (`stage0_results/geometry_filter/slot_probe.json →
  spaces.raw.{abs,signed}.{object_vs_subject_auroc.auroc, cal_to_conf.macro_f1}`).

**Caption draft:** "Why each geometric component is necessary. (A) Recall at precision
1.000 for candidate-pair ranking on the calibration subsets, raw cosine vs whitened
cosine (mean and 128 shared directions removed): the AUROC gain is small (+0.002 to
+0.004) but the precise tail decompresses from 0.075 to 0.513 (sh_6k) and from 0.007 to
0.291 (sh_32k). (B) Held-out balanced AUROC of the slot probe when its 2,560-dimensional
magnitude profile is compressed to 64 or 256 principal components: the signal is
distributed across dimensions. Sources: `m6_abtt_dsweep.json`, `geo_pairlevel.json`."

### 4.8 Figure 8 — Deletion vs placement, and where the remaining errors come from

**Placement:** §6.5 (`sec:placement`) after Table 8, label `fig:placement`; Part B is
cross-referenced from §7.2 ("Read time is where the headroom is") and §8.5.

**Part A — three page edits, same detector, same groups.**
Source: `results[0].arms.{native, detector_suppress, detector_demote_late,
detector_anti}.correct` per file. Arm semantics (from the artifacts' own `arms`
descriptions): `detector_suppress` = delete stale (H-Nav-GEO); `detector_demote_late` =
move newest to end; `detector_anti` = move newest to front. Expected values = `.tex`
Table 8 (15 rows), e.g. gemma-3 sh_6k 45 / 76 / 38 / 72; Phi-4-mini sh_32k 50 / 66 / 76 /
43; Qwen3.5-9B sh_32k 61 / 86 / 83 / 68. Form: a 5 × 3 grid of small grouped bars, or a
heatmap of net vs native for the three edits, with the sign visible. The conclusion
(deletion is positive in 15 of 15 cells; the placement edits flip sign between models)
goes in the caption, not as a slogan in the chart.

**Part B — "where gain is lost" funnel, exactly reconstructible for four stages.**
Per conflicted question, from `per_question[]`:
1. `("fact:%d" % target_serial) in pool` → gold fact reached the 50-fact pool
   (its complement is the retrieval miss; at sh_6k and sh_32k the page is the whole
   store, at sh_64k 10 of 17 chunks);
2. `target_serial in plan.demote_serials` → gold fact was the kept newest member of a
   verified group;
3. `target_serial in plan.suppress_serials` → gold fact itself deleted (gold cut);
4. neither → the queried key was never touched by the detector;
5. `arms.detector_suppress.correct` → outcome, crossed with 1–4.

Reference model, conflicted stratum (recomputed during the audit; the script must
reproduce these):

| subset | n | gold in pool | kept newest | gold deleted | key untouched | GEO correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sh_6k | 74 | 74 | 54 | 0 | 20 | 51 |
| sh_32k | 65 | 65 | 48 | 2 | 15 | 43 |
| sh_64k | 66 | 44 | 16 | 1 | 49 (22 of them not in pool) | 29 |

The sh_64k row reproduces the independent decomposition in
`stage0_results/geometry_filter/E2E4_COMPLEMENTARITY.md` §6: 66 − 44 = 22 retrieval
misses; the 37 wrong answers split 22 retrieval + 15 detection misses for the GEO arm.
Draw the funnel as a stacked or waterfall bar per subset with the outcome (correct /
wrong) crossed against the stage, for all five models or for the reference model with the
other four as thin replicates (the plan is identical across models, so stages 1–4 are
identical; only stage 5 differs).

What cannot be separated from committed data and must not be estimated: within "key
untouched", the share due to the 0.94 cosine pre-filter vs the GEO rectangle vs the NLI
veto vs the span check. Those per-pair verdicts are in the uncommitted prepass tables on
the GPU box. State this in the caption.

The 8 merged-key deletions on sh_64k (questions {19, 48, 58, 73, 83}; keys `official
language of Italy`, `type of music that The Game plays`, `outfielder · associated with
the sport`; `E2E4_COMPLEMENTARITY.md` §4) are a plan-level limitation, stated once in
§8.5; they are not a funnel stage and do not go in this figure.

**Caption draft:** "(A) Overall accuracy under three page edits driven by the same
detector and the same conflict groups: deleting the stale versions, moving the newest
version to the end of the page, and moving it to the front. Deletion is the only edit
that is positive in all 15 model-by-subset cells; the two placement edits change sign
from model to model. (B) Where the remaining conflicted-stratum errors come from, per
subset (reference model): whether the gold fact reached the candidate pool, whether the
detector kept it as the newest member of a verified group, left its key untouched, or
deleted it, crossed with the answer outcome. At sh_64k, 22 of the 37 remaining errors are
questions whose gold fact was never retrieved. The split of untouched keys between the
cosine pre-filter, the geometric screen and the NLI gate is not recoverable from the
committed artifacts. Sources: `detector_gap_sh_*.json`."

---

## 5. Manuscript edits (`paper/hnav_geo_draft.tex`)

Line numbers refer to commit `ca0d2c3`.

### 5.1 Figures

1. Create `paper/figures/final/`; keep `\graphicspath{{./figures/}{./}}` and reference
   files as `final/fig1_anatomy.pdf` etc., or change the path to `{./figures/final/}`.
2. Replace the four `\placeholderfigure{…}{…}` calls (L155, L268, L323, L424) with
   `\begin{figure}[H]\centering\includegraphics[width=\linewidth]{…}\caption{…}\label{…}\end{figure}`
   using the captions in §4. Remove the `\placeholderfigure` macro (L44–46) once unused.
3. Insert Figures 1, 5, 6, 8 at the placements in §4 and add `\ref{fig:…}` sentences:
   §3.4 → `fig:anatomy`; §6.3 "Three observations" → `fig:strata`; §6.4 → `fig:scale`;
   §6.5 → `fig:placement`; §7.2 last paragraph → `fig:placement`(B).
4. Add the figure files to `tab:provenance` (App. B): one row per figure naming its
   `paper/figures/data/figN.json`.
5. PDF for `\includegraphics`; SVG and PNG alongside for editing and preview. achemso
   `manuscript=article` is single-column, so `width=\linewidth` is right.

### 5.2 Factual corrections the audit found (fix these; do not leave them for review)

1. **L57 (§1):** "Thomas Kyd was born in the city of Paris" → the store's superseding
   fact is *Leeds* (serial 306). Replace Paris with Leeds.
2. **L57 (§1):** "Marlowe was born in the city of London" is not a fact in any evaluated
   subset or in any gold pair. Replace with the real look-alike *Thomas Arne was born in
   the city of London* (serial 342), or keep Marlowe explicitly as a constructed
   illustration. The NLI scores quoted for Kyd/Marlowe at L412 (0.9995 / 0.9998) exist
   only as prose in `TEZ_BULGULARI.md` L265–267 ("measured on the box"), with no JSON
   artifact; if kept, say so or replace them with the machine-backed statistic: 447 of
   453 cross-key adversary pairs pass the bidirectional 0.90 gate
   (`stage0_results/geometry_filter/GEO_PREREG.md` L30).
3. **L57 (§1):** "Both pairs have cosine well above 0.9" is false for the real look-alike
   (0.834). Reword to the distribution claim: cosine-matched non-conflicts reach 0.99 and
   65,782 non-conflict pairs sit above the lowest conflict cosine on sh_64k.
4. **§8.3 / L446:** "between half and three quarters" is consistent with all three
   recorded ranges (72 / 75 / 52 % per subset; 50–61 % across five models on sh_64k;
   `THESIS_CLAIMS.md` §10 says 52–75 %). Keep the .tex wording; do not "tidy" it to one
   of the others.
5. **External values (§1, §2.3, §6.4, §8.2, §9, abstract, Table 7):** unverifiable from the
   repository; resolve per §3.2 before submission.

### 5.3 Consistency items

- §3.3 mixes two harnesses: the eight-run 0–5 / 74 figure comes from the whole-context
  probe runs (`question_strata.json → runs[]`, `aggregate.errors_total.stale_value 572`),
  the 4 / 74 figure from the campaign artifact. Both are correct; the text already
  attributes them; the figures use only the campaign artifacts.
- Deletion counts: 2,157 (GEO, calibration pools, `geo_operating_point.json → metrics.tp`)
  is the number for §6.2. Never substitute 2,673 or 735, which belong to the parser-era
  detector.
- Model names in figures exactly as `run_manifest.json → llm_model` minus the org prefix.

---

## 6. Execution plan

Work from the repo root on the branch `claude/hnav-presentation-evidence`. Python ≥ 3.10
with `numpy`, `matplotlib`; `scikit-learn` optional; no torch needed unless embeddings
must be regenerated (avoid; see §3.1).

1. **Directories.** `paper/figures/{final,scripts,data,intermediate}`. Add a
   `.gitignore` exception if any `.gz` is produced (root rule `*.gz` swallows it).
2. **`extract_data.py`.** One function per figure. Reads the artifacts in §4, writes
   `paper/figures/data/figN.json` where every number carries `{value, source_file,
   source_key}`. Skips `_VOID` directories. Refuses to run if
   `geo_identity_screen.json → fingerprint` ≠ `335b4540…ccb2a` or
   `run_manifest.json → operating_point_sha256` differs between model directories
   (all six are `9c3cb668706ddd3e820f99ef5cba8047f3138709cc48aa3fab901011aa616038`).
3. **`verify_numbers.py`.** Recomputes Tables 3–9 of the .tex from the artifacts and
   diffs them against the values parsed from `paper/hnav_geo_draft.tex`. Also asserts the
   Figure 8B funnel table above, the per-subset plan hashes are identical across models,
   and `aa_floor` is 0 everywhere. Exit non-zero on any mismatch. Run before every
   figure export.
4. **Figures 1, 2, 4, 5, 6A, 7, 8** from committed data. Export PDF + SVG + PNG (300 dpi).
5. **Figure 3.** Obtain `hnav/_cache/emb/` from the authors (4,499 `.npy`, namespace
   `Qwen_Qwen3-Embedding-4B|float32|L8192`). Run `compute_pair_scores.py`; assert the
   recomputed sh_64k balanced AUROCs match `geo_pairlevel.json` to four decimals; commit
   `fig3_pair_scores.csv`; draw ROC curves and Panel B. Until then, ship 3A as the
   AUROC dot plot.
6. **Figure 6B.** Transcribe arXiv 2507.05257 Table 3 per §3.2; commit
   `external_reference.json`; reconcile the manuscript's external sentences.
7. **Manuscript.** Apply §5. Compile with `achemso` installed (not available in the audit
   sandbox); check every `\ref{fig:…}` resolves and no `\placeholderfigure` remains.
8. **Commit** on the branch with a `Paper:` prefix, figures and scripts together, and
   list in the commit message which of §3's gaps were closed.

Verification gate before hand-back: `verify_numbers.py` passes; every figure's JSON lists
its sources; no figure contains a number absent from an artifact or from
`external_reference.json`; captions obey §1 wording rules (grep for "harm-free",
"significant everywhere", "transfer").

---

## 7. House style

Reuse the rcParams of `presentation_evidence/_scripts/make_charts.py` (L19–28): sans
font (`DejaVu Sans` fallback), y-grid only, top/right/left spines off, `legend.frameon
False`, `svg.fonttype "none"` so SVG text stays editable, `bbox_inches="tight"`, 300 dpi.
Its `save()` writes PNG + SVG; add PDF. Its categorical palette (`#2a78d6, #eb6834,
#1baf7a, #eda100, #e87ba4`) was validated only for direct-labelled bars; the semantic map
below uses the Okabe–Ito set, which is colour-blind safe by construction. Validate the
final choice with the team's own contrast check.

Semantic colour map (one dictionary in `make_figures.py`, used by every figure):

| role | colour | used in |
| --- | --- | --- |
| Native | `#0072B2` blue | 4, 5, 6, 8A |
| H-Nav-GEO / governed / cleaned page | `#D55E00` vermilion | 2, 4, 5, 6, 8 |
| true conflict / positive pair | `#CC79A7` reddish purple | 1, 3 |
| look-alike / hard negative / non-conflict | `#56B4E9` sky blue | 1, 3 |
| stale fact | `#999999` grey, strikethrough in text panels | 1, 2, 8 |
| newest fact | `#009E73` bluish green | 1, 2, 8 |
| move-newest-to-end / move-newest-to-front (placement controls) | `#E69F00` amber / `#F0E442` yellow | 8A only |
| conflicted vs unique stratum | encoded by panel, not colour; marker shape differs | 5 |
| subject / relation / object (exposition only) | light tints of amber / sky blue / purple as text background, never as data marks | 1, 2 |
| whitened vs raw | raw = grey `#999999`, whitened = vermilion | 7A |

Typography: one family, 8–9 pt at final size, panel labels (A), (B), (C) bold at top-left,
axis titles in sentence case. Terminology, verbatim and consistent: parser-free,
whitened cosine, slot probe, deterministic suppression plan, contextual comparison,
stale, newest, conflict group, cleaned page, calibration split, held-out.

No 3-D, no gradients, no drop shadows, no titles inside the chart that restate the
caption. Every panel must read at 85 mm width (single-column reduction) even though the
manuscript mode is single-column.

---

## Appendix A — key-path cheat sheet

`detector_gap_sh_<subset>.json` (one per model × subset):

```
harness.llm_model                                   model id
operating_point.thresholds.{cos_pair, r_min, nli_contradiction}
operating_point.ces.tau                             "-0.4:0.2"
operating_point.by_subset.<s>.{pair_recall_pool, n_suppressed, n_suppressed_harmful}
results[0].subset, .n_facts, .strata_counts, .n_chunks_on_page, .n_chunks_total
results[0].arms.<arm>.{n, correct, accuracy}
results[0].paired_vs_native.<arm>.{b_native_only, c_arm_only, net, p_exact}
results[0].aa_floor.{b_native_only, c_arm_only}
results[0].by_stratum.<conflicted|unique>.{arms, paired_vs_native}
results[0].tokens.<arm>.{prompt_chars, delta_chars_vs_native, delta_pct}
results[0].positive_control.{n_questions_policy_fired, n_facts_suppressed}
results[0].void_conditions.<k>.{status, observed}; .verdict.run_void
results[0].per_question[i].{index, stratum, key, truths, target_serial, n_pool, pool,
                            plan{suppress_serials, demote_serials, n_groups,
                                 n_pairs_verified},
                            arms.<arm>{output, correct, n_facts, prompt_chars,
                                       prompt_sha, identical_to_native}}
arms: native | native_repeat | detector_suppress | detector_demote_late | detector_anti
```

`geo_pairlevel.json`: `balanced.<s>.methods.<m>.{auroc, band_auroc, auroc_ci95{lo,hi}}`,
`hard_confirmatory.methods.<m>.{auroc, auprc, tpr_seen_fpr_0.0001,
tpr_unseen_fpr_0.0001, inverted_vs_campaign_cos.win_rate}`; methods `campaign_cos,
abtt_cos, probe, geo, ces, probe_pca64, probe_pca256`.

`geo_identity_screen.json`: `scalars.{probe_b, T_w, T_p, s_w, s_p}`, `fingerprint`,
`provenance.anchors.n_pool_pairs{sh_6k 193, sh_32k 1332}`,
`provenance.probe.{n_pos_edits 989, n_neg_edits 8716}`.

`m6_abtt_dsweep.json`: `subsets.<s>.raw.grouping.recall_at_precision.recall_at_precision_{0.9,0.95,0.99,1}`,
`subsets.<s>.regimes["frozen_global|D=<D>"].grouping.{auc, recall_at_precision{…}, candidate_floor_min_cos}`,
`.anisotropy_random_pairs.mean`, `.auc_delta_vs_raw`.

`m7_delta_geometry.json`: `subsets.<s>.raw.pair_space.{n_same_key, conflict_cos_min,
n_eligible_above_conflict_min, nonconflict_cos_max}`,
`subsets.<s>.raw.sets.<conflict|conflict_matched|cos_matched|same_relation|same_subject|random>.cos.{n,mean,sd,min,p10,p50,p90,max}`.

`gold_conflict_dataset.jsonl.gz` record: `pair_id, subset, split, tier, in_eval_set,
gold_update, gold_strict, fact_a, fact_b, fact_a_id, fact_b_id, cosine_similarity,
parser.{same_key, same_relation, same_subject, same_object, both_parse,
superseding_serial}`; hard negative = `tier == "negative" and parser.both_parse and
parser.same_relation and not parser.same_subject`.

`question_strata.json`: `subsets[].{subset, counts{unique, conflicted}, indices{unique[],
conflicted[]}}`; `aggregate.errors_total.stale_value = 572`.

`Conflict_Resolution.json`: list; entry `k` with `metadata.qa_pair_ids[0]` starting
`factconsolidation_sh_6k` (k = 4), `sh_32k` (5), `sh_64k` (6); `context` lines
`"<serial>. <fact>"`; `questions[i]`, `answers[i]`. Parse with
`hnav.adapters.mab_adapter.explode_facts`.
