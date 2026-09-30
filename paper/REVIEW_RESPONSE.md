# Response to the internal review of the first draft

Revised manuscript: `paper/hnav_geo_manuscript.tex`. Numbers added in the
revision are recomputed by `paper/derive_revision_numbers.py`; every number is
traced in `paper/PROVENANCE.md`. Items the authors must still settle are
printed in red in the PDF as `[CONFIRM: …]`.

## 1. Errors and internal inconsistencies

| # | review point | status in the revision |
| --- | --- | --- |
| 1.1 | Abstract says "reference 14B model" | The draft supplied for revision already said 4B. Abstract rewritten anyway (item 3). |
| 1.2 | Per-model numbers in 3.3 in the wrong order | Confirmed against the artifacts. Section 3.3 is now Table 2, in the same model order as every other table. |
| 1.3 | Look-alike 0–342 (cos 0.834) never reaches the 0.94 screen | Confirmed. Captions now say so. **Figure 2 still shows 0–342**; see "Figures to regenerate". |
| 1.4 | NLI quote is for Kyd/Marlowe, not the pictured pair | Confirmed; the Marlowe sentence is a constructed probe, not a store fact. The text now says so. No NLI score for 0–342 is committed. |
| 1.5 | Precision 1.000 vs "gold record deleted" | Reconciled from the artifacts: all three cases (two at sh_32k, one at sh_64k) are questions whose expected answer is itself superseded by a later fact of the same key (e.g. Pemberton Billing, London → Washington, D.C.; expected answer London). Deleting is correct under the stated rule. Section 7.3 and the Figure 7 caption say this. "Independently verified" now reads "according to the parsed fact table". |
| 1.6 | NLI counts vs deletion counts | Confirmed: 410/4,868/8,956 are directed passes over distinct text pairs, cached across the 100 questions. Relabelled; "at most about 90" removed. Also corrected: NLI runs on every pair inside a screened component, not only on screened pairs (`read_gate.py`). |
| 1.7 | "Embeddings are the store's own" | Removed. Section 4.2 states that per-fact embeddings are an extra computation because the benchmark embeds 4,096-token chunks. |
| 1.8 | Figures say "held-out" | Text and captions say "frozen transfer". **The graphics still need relabelling.** |
| 1.9 | "Nested" is wrong; 2,765 has duplicates | Confirmed: sh_6k ⊂ sh_32k; 2,244 of 2,310 sh_32k facts are in sh_64k. "Overlap" used throughout; 2,765 rows = 2,310 distinct facts with sh_6k counted twice. |
| 1.10 | 455 vs 454; 1,681 vs 1,687; three thresholds | 1,687 = template-tagged pairs, 1,681 = those minus the 6 the judge rejected (stated). The 0.80 / 0.88 / 0.94 roles are explained in Section 4.5. **Figure 2 shows 454; the store has 455.** |
| 1.11 | Figure 3 is not ROC curves; panel B unlabelled | Caption rewritten as AUROC point estimates with intervals. **Label panel B in the graphic and extend the caption** (comment in the source marks the spot). |
| 1.12 | Placement beats deletion in six cases, not two | All six listed; a per-question diagnosis explains the gemma-3 case (Section 6.5). The combined delete-and-move arm is named as the next experiment; it was not run. |
| 1.13 | Placeholders | Related work written (78 references). Artifact path moved to `PROVENANCE.md`. The broken `\code{…_{6k, 32k, 64k}}` replaced by `\sub{6k}` etc. |
| 1.14 | Mean pooling | Verified as deliberate: it mirrors the benchmark retriever's own Qwen3-Embedding class. Justified in Section 4.2. |
| 1.15 | vLLM 0.28.0; external GPT-4o/GPT-5-mini numbers | "0.28.0" is what the repository records; confirm against `pip freeze` on the GPU box. External numbers could not be checked (arXiv is unreachable from this environment); they are cited as reported. |

## 2. Debatable or overstated claims

| # | point | change |
| --- | --- | --- |
| 2.1 | "Not a failure of knowledge" | Softened. New within-question control (Table 2): on 54 sh_6k questions where deletion leaves only the counterfactual value, models give the superseded value 0–6 times instead of 33–50, and answer correctly 43–54 times. The benchmark cannot separate position from prior preference; the text says so. Error taxonomy now covers all five models (283/301 superseded). |
| 2.2 | "Method, not model, sets the number"; gain vs native | Removed. Section 7.1 reports the convergence of governed conflicted accuracy to 48–55/74 at sh_6k and the perfect inverse ranking of gain on native conflicted accuracy. |
| 2.3 | Scale-gap framing | Reduced to one paragraph (Section 8.2) that names Gemini-3-Flash 96 % and the 262K caveat and draws no conclusion. Replaced as evidence by an in-harness comparison: governed Qwen3-4B ≥ native Qwen3.5-9B in all three cells (p = 4e-10, 0.007, 0.40); governed Phi-4-mini ≥ native 9B in all three (significant at sh_6k only). |
| 2.4 | +47 is the most in-sample number | Development history stated in Section 5.1, including the calibration accuracy sweeps and five earlier sh_64k runs with the reference model. Abstract leads with the range and the sh_64k range. |
| 2.5 | "Headline accuracy is accuracy on no-conflict questions" | Removed. |
| 2.6 | Isotropy claim; ABTT is not whitening | "Whitening" replaced by "ABTT cosine" everywhere in the text; Mu & Viswanath cited; the near-zero centred mean is stated to be expected by construction and not evidence of isotropy. **Figures still say "whitened".** |
| 2.7 | "Universal", "rules out memorisation", counts vs rates | Removed. Novel/seen slices reported as rates; gemma-4 is the exception and is named. Relation-disjoint collapse of the probe (macro-F1 0.21) is now in Sections 7.2 and 8.6. |
| 2.8 | Write-time argument | Rewritten (Section 8.4): the earlier write-path result concerns a different gate; write-time supersession was not evaluated; the case for read time is reversibility. |
| 2.9 | Negative token cost | Dropped. Latency stated as not measured. |
| 2.10 | Multi-valued / time-scoped relations | Now a limitation with the "speaks Turkish / speaks English" example. |

## 3. Abstract

Rewritten along the reviewer's lines (about 250 words): names the benchmark,
drops "canonical" and "FC-SH", gives 3–47 and 11/15 after Holm, the sh_64k
range, and no scale comparison. "Answers almost perfectly with one version"
is gone; the unique-stratum claim is the net change of at most one question.

## 4. Writing

Aphorisms and slogans removed; the "not held-out" caveat is stated once in
Section 5.1 and carried in captions only where a reader needs it. H-Nav and
GEO are defined; key, version pair, look-alike pair, cross-key pair, object
transition and the dual labels are defined at first use. The six slot classes
are listed (relation-only is absent). The ABTT formula uses (I − PPᵀ). The
anchor rule is defined exactly and the thresholds are given in raw form. The
span check is shown to be non-binding (any pool partner at cos ≥ 0.94 bounds
the residual by 0.341 < 0.44).

## 5. What referees will flag

| # | point | change |
| --- | --- | --- |
| 5.1 | Four references | Seventy-eight, covering RAG, knowledge conflicts, agent memory, knowledge editing, anisotropy and post-processing, probing, NLI, and statistics. **Check every entry before submission**; four 2026 arXiv entries were kept from the draft unverified. |
| 5.2 | No baselines | Table 8 adds the schema-keyed oracle for all five models and the probe-free screen for the reference model; GEO recovers 50–81 % of the oracle's gain. Missing end-to-end baselines are listed as not run. |
| 5.3 | External validity | Stated as the main limitation; paraphrased store, LongMemEval knowledge-update and a second encoder named as tests. |
| 5.4 | Run-to-run variance | Explained: the eight runs used different server settings; two identical-setting runs differed on 1/100 (vLLM V1) and 7/100 (V0) answers; campaign native 30 lies in the 26–31 range. |
| 5.5 | Annotation | Label definitions, judge settings and coverage slices given; no human validation (limitation); licence marked `[CONFIRM]`. |
| 5.6 | Pair-level statistics | Unpaired-interval caveat stated; no paired GEO-vs-ABTT test exists in the artifacts, so none is claimed. Tail TPR intervals added (69.2–74.6 %, 44.3–52.0 %) with the four-negative caveat. Recall ceiling of the 0.94 threshold added (24–26 % of version pairs below it). Rule of three (0.14 %) vs 8/532 = 1.5 % (0.7–2.9 %) stated. |
| 5.7 | Only sh_64k exercises retrieval | Stated in the introduction and at Table 1. |
| 5.8 | Compliance | File names removed from captions and appendix; data availability uses the GitHub URL; CRediT, funding, competing interests and generative-AI statements added as `[CONFIRM]` blocks; "four architecture families" → "three families". Template kept as achemso so the file still builds; switch at submission. |
| 5.9 | Title | "Read-Time Removal of Superseded Facts in Agent Memory Using Embedding Geometry". |

## Additions from the repository that the review did not raise

- **Preregistration.** GEO's preregistered primary endpoint (> 64 on sh_64k,
  reference model) was not met (56), and every sh_64k GEO cell carries the
  campaign's void flag (condition 4, 8 harmful deletions). Both are disclosed
  in Section 5.2 and the limitations.
- **gemma-3 rerun.** The first gemma-3 run (fp8 KV cache) was voided and all
  three subsets re-run with BF16; the draft's "never re-rolled" was untrue and
  is gone. KV-cache type per model is in Table 3.
- **Unique stratum is a weak harm test.** Its facts are real-world facts, so a
  model can answer them without the record. The plan-level harm count is named
  as the direct measure.
- **Hard negatives** are same-relation, different-subject pairs (5,124 of
  8,716 also share the object), not "same object".
- **Retrieval misses** are 45–73 % of residual sh_64k errors across models,
  not "most" for every model.

## Figures to regenerate (graphics, not LaTeX)

1. Figures 3, 4 and 6: replace "held-out" with "frozen transfer".
2. Figures 2, 3 and 6: replace "whitened cosine" with "ABTT cosine" wherever it appears.
3. Figure 2: store size 455, not 454. Replace the 0–342 look-alike with one
   above cosine 0.94, and show its probe margin. A clean sh_6k candidate from
   the gold set: serials 79/365, "The name of the current head of state in
   United Kingdom / Soviet Union is Elizabeth II." (raw cosine 0.952). Its
   probe margin must be computed on the GPU box.
4. Figure 3: label panel B and describe it in the caption.
