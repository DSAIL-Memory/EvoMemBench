# Stage-1 Confirmatory Campaign — PRE-REGISTRATION  ·  **WITHDRAWN**

> # ⛔ WITHDRAWN — 2026-08-15
>
> **This pre-registration is withdrawn and must not be executed.** The
> document is retained in full, unedited below this banner, because evidence
> is never deleted: it is the record of what was planned, on what basis, and
> why the plan did not survive contact with the data. Superseded by
> **`STAGE1_PLAN.md` REVISION R2** (commit `d9a56e9`).
>
> It was never registered — it never left DRAFT, no operating point was ever
> frozen, and **no `sh_64k` inference was ever run**. Withdrawal therefore
> costs no confirmatory evidence and burns no shot at the confirmatory subset.
>
> ## Why it is withdrawn — three independent reasons, any one sufficient
>
> **1. The mechanism is refuted.** Stratifying sh_6k's 100 questions by
> whether the queried key is conflicted (`hnav/labeling/question_strata.py`,
> `stage0_results/question_strata.json`, verified twice independently across
> all EIGHT committed runs): unique-key questions are **26/26 correct in every
> run**; conflicted-key questions are **0–5 of 74**. Of 575 conflicted-question
> errors, **572 emit the STALE value of the correct key** (3 off-list, 0
> empty). So the failure is not a retrieval-ordering failure at all — the
> right key is found, and the model then applies the stale value despite a
> prompt that explicitly states the larger serial is newer. Chunk-level UPWARD
> rerank — the sole mechanism this document tests — cannot reach that lever.
>
> **2. Retrieval is already complete on the calibration split.** sh_6k has 2
> chunks and sh_32k has 9, both ≤ `top_k` 10: *every* chunk is retrieved for
> *every* question. There is no "missing superseder" to promote, which is why
> the 162-cell calibration returned no operating point with net > 0. The
> objective was not underpowered by accident — it was measuring a mechanism
> with no room to act.
>
> **3. The success criterion was not defensible against the measured noise.**
> The +3.0-point bar was set from the sh_6k off-run SD of 1.52 measured on a
> DIFFERENT substrate (`:8000`, prefix caching on). The per-question noise
> floor has since been measured at **3.3%/question**, and this document's
> A/A section was never filled in — the frozen-substrate floor was never
> measured, because the box became unreachable first. Running a 7+7 campaign
> against an unmeasured floor, on a mechanism now known to be inert, would
> produce a null that proves nothing about H-Nav and consumes the one-shot
> subset doing it.
>
> ## Additional invalidating defect (independent of the above)
>
> Every threshold this document would have frozen was fit on embeddings
> truncated at 512 tokens against 4096-token chunks — the T12 defect
> (`hnav/BUILD_NOTES.md` §10). Even had the mechanism been sound, the
> operating point would have been derived from ~12% of each chunk. Re-fit is
> specified in `hnav/deploy/REFIT_RUNBOOK.md` and is calibration-split only.
>
> ## What replaces it
>
> Per `STAGE1_PLAN.md` R2, in order: **oracle probe first**
> (`hnav/stage1/stale_suppression_probe.py`, already written, not run) to
> measure the ceiling of the expanded mechanism set (stale-record
> **suppression** and **placement/recency**, both user-authorized); then
> mechanism selection on what the probe actually proves; then implementation;
> then a **NEW pre-registration**; then the campaign. The new document must
> set harm caps and a power calculation **above the measured 3.3%/question
> noise floor**, and must not inherit this one's +3.0-point bar or its
> truncated thresholds.
>
> Nothing below this banner has been altered.

---

> **STATUS (as written, superseded): `DRAFT — NOT REGISTERED. CAMPAIGN BLOCKED.`** The graded Faz B
> calibration COMPLETED on the box (2026-08-15, log:
> `hnav/_out/pipeline/stage1_evaluate.log`) and its pre-registered objective
> returned **"NO feasible operating point with net > 0 — REPORT AND STOP"**:
> after ~470 LLM-graded native-vs-reranked comparisons on sh_6k+sh_32k, no
> grid cell delivered positive net help within the harm caps. Cell-level
> verification (`hnav/_out/stage1_calibration.json`, box) is queued behind a
> box outage. Until a human decision on that verdict, `{{...}}` slots stay
> unfilled, no operating point is frozen, and `stage1_campaign_driver.sh`
> refuses to start (it requires this file committed as REGISTERED plus a
> committed operating point — neither exists).
>
> ## Timing provenance — objective frozen before grading (verified 2026-08-16)
>
> Recorded here so an external reviewer finds it without asking, since the
> credibility of a pre-registered objective rests entirely on it having been
> fixed before any outcome was observed. The audit question was whether the
> calibration's graded phase could have started before the objective was
> committed. It could not:
>
> | event | time (+0300) | evidence |
> |---|---|---|
> | objective committed (`3bd59ef`, in the harness docstring) | **07:23:42** | `git log -1 --format=%ad 3bd59ef` |
> | `:8003` chat server started | 07:32:45 | `hnav/_out/pipeline/chat_stage1.log` |
> | **first graded request** | **07:33:48** | first `Received request` in that log |
> | evaluate finished | 08:21:48 | log mtime; artifact `provenance.date` agrees (05:21:48 UTC) |
>
> The entire graded phase (469 LLM calls) postdates the committed objective by
> ~10 minutes. **No restatement is required.** One clarification for
> completeness: an earlier `--no-llm` grid pass did run before that commit, but
> it produces no helped/harmed outcomes — only coverage and false-verified
> counts — so no outcome data existed at the time the objective was frozen.
>
> Registered by commit BEFORE any sh_64k inference. Precedent and discipline:
> `stage0_results/t4_s2_protocol.md` (S2). Binding sources:
> `STAGE1_PLAN.md` §0 (user decisions, 2026-08-15), `KAPI_KARARI.md` §3/§6
> (T8 split verdict), the Faz A supervisor audit (Notes 1–2).
> Author: Agent B (T11). Date: 2026-08-15.

## 1. Claim under test

On `factconsolidation_sh_64k` (the CONFIRMATORY subset — untouched by every
calibration step), enabling H-Nav's read-path rerank (`HNAV_MODE=live`)
improves answer accuracy over the untouched benchmark (`HNAV_MODE=off`) on the
frozen Stage-1 substrate, with bounded harm and no token cost.

The intervention is EXACTLY: within the retrieved top-10 chunk page, promote
the chunk carrying a gate-verified conflict group's LATEST fact above that
group's stale-carrier chunks. Same chunk set, same count, order only
(`hnav/core/read_policy.py`; the seam returns the native page on any
irregularity). Nothing else changes; the write path observes only.

## 2. Frozen inputs (all committed before the campaign)

| Input | Where frozen |
|---|---|
| Gate operating point | `stage0_results/stage1_operating_point.json` == `hnav/core/read_policy.stage1_thresholds()` (equality enforced by `hnav/tests/test_threshold_provenance.py`) |
| Substrate, chat | `hnav/deploy/serve_stage1_chat.sh` — :8003, GPU1, Qwen3-4B-Instruct-2507 local bf16 weights, `--max-model-len 65536 --kv-cache-dtype fp8 --enforce-eager --max-num-seqs 1 --no-enable-prefix-caching --gpu-memory-utilization 0.58` |
| Substrate, embeddings | `hnav/deploy/serve_stage1_embed.sh` — :8001, GPU1, Qwen3-Embedding-4B **bfloat16** 0.33 (declared deviation; identical for both arms; gate geometry stays fp32 via the cache-first non-persisting embedder) |
| Live-arm NLI | `cross-encoder/nli-deberta-v3-large`, CPU (`HNAV_NLI_DEVICE=cpu`; arms run with `CUDA_VISIBLE_DEVICES=` empty) |
| Agent config | `configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml` (temperature 0, retrieve_num 10) |
| Driver | `stage0_results/stage1_campaign_driver.sh` (refuses to start without this file and the operating point committed; live-stack preflight) |
| Analysis code | `stage0_results/stage1_campaign_analysis.py` (this document, implemented verbatim) |

## 3. Design

- **N = 7 off + 7 live**, fixed interleaved order `W O L O L O L O L O L O L O L`
  (`W` = off warmup, DISCARDED). No optional stopping: all 14 counted runs
  execute, the analysis runs once on the full set, and whatever comes out is
  reported.
- Each run: fresh output dir, `HNAV_MODE=off|live`, `HNAV_DOTENV_NO_OVERRIDE=1`,
  `OPENAI_BASE_URL=http://localhost:8003/v1`, per-run `HNAV_RUN_ID` so every
  live run leaves an audit trail (`hnav/_out/audit/stage1c_live_<i>.read.jsonl`).
- Grading: the benchmark's own deterministic `substring_exact_match`
  (transcription pinned by `hnav/tests/test_counterfactual.py`), applied
  offline to recorded outputs against the dataset answers.

## 4. Success criterion (ALL of, pre-registered; conjunction from STAGE1_PLAN §0)

1. **Paired Δaccuracy** `mean_i[acc(live_i) − acc(off_i)] ≥ +3.0` points.
2. **Per-pair harm bound**: `harmed_i ≤ helped_i` for every pair `i`
   (question-level: harmed = off-correct ∧ live-wrong, helped = converse).
3. **Pooled harm rate** `Σ harmed_i / 700 ≤ 0.02`.
4. **Token neutral-or-better**: mean live `(input_len + output_len)` ≤
   `1.001 ×` mean off.

**Void, not failure:** a live run with a missing/short audit trail
(≠ 100 read records) or any nonzero `embed_cache.misses` voids the campaign —
the analysis reports VOID and stops; no re-run without a new pre-registration.

The +3.0-point bar is FIXED (STAGE1_PLAN §0; rationale: ≈2× the sh_6k off-run
SD 1.52 measured in `t4_s2_trials_summary.json`). The frozen-substrate A/A
measurement below is CONTEXT for interpreting the result, not a knob: if the
new floor is larger than 1.5, the bar does not move and the discussion must
say power was lower than planned.

## 5. A/A noise floor on the frozen substrate (measured before the campaign)

Protocol: `stage0_results/stage1_aa_driver.sh` — 1 discarded warmup + 5
`HNAV_MODE=off` runs on **sh_32k** against the identical frozen substrate;
analysis `stage0_results/stage1_aa_analysis.py` →
`stage0_results/stage1_aa_summary.json`.

**Declared transfer assumption:** the noise floor is measured on sh_32k, not
sh_64k, because sh_64k is confirmatory and every inference spent on it before
the campaign is a spent shot. sh_32k shares the substrate, the agent config,
the prompt shape and the deterministic-serving flags; the assumption is that
run-to-run output variance does not grow materially with the longer prompts.
This is a stated limitation of the design.

Measured (5 runs, sh_32k): per-run `substring_exact_match` =
**{{AA_SEM_PER_RUN}}**, mean **{{AA_MEAN}}**, SD **{{AA_SD}}**; pairwise
output-mismatch mean **{{AA_MISMATCH_MEAN}}**, max **{{AA_MISMATCH_MAX}}**
(10 pairs).

## 6. Calibration summary and the Note-1 accounting

Full artifact: `stage0_results/stage1_operating_point.json`; grid data:
`hnav/_out/stage1_calibration.json` (box). Calibration split sh_6k + sh_32k
ONLY; queries and grading prompt are the campaign's own (templated retrieval
query per `RAGSystem`, Memory-numbered prompt, max_tokens 10).

- **Chosen operating point:** {{OP_SUMMARY — BLOCKED: the objective returned
  no feasible cell with net > 0; see the STATUS banner}}
- **Coverage on the calibration split (measured, grading-independent):**
  the frozen ambiguity screen (`nmargin`/`H_z`, mode `any`) fires on 94/200
  questions; across the grid, 68–115 of 200 questions see an order change
  (filter ON: 68–113). Net help: **none of the 162 cells achieved net > 0**
  under the harm caps (cell-level helped/harmed extraction queued — box).
- **Supervisor Note 1, measured (2026-08-15, real NLI, sh_6k+sh_32k):**
  WITHOUT the subject-identity screen the bidirectional-contradiction
  criterion false-verifies **33–93%** of verified pairs depending on the
  cosine screen — e.g. at cos 0.90/frozen r: 12,896 different-key pairs
  rubber-stamped vs 923 true supersessions (rate 0.933); at cos 0.94 the rate
  is still 0.33–0.39. The dominant class is exactly the audit's measured
  shape (same template, different subject); same-key/same-object restatements
  contributed 0 at every cell. WITH the adapter's parsed-key equality screen
  (`MABAdapter.same_key_pair`) the measured false-verified rate is **0.000 at
  every one of the 162 grid cells** (verification precision 1.00). The screen
  is therefore frozen ON in any future operating point; its harm contribution
  is measurably zero on the calibration split.
- Unparseable facts (parser coverage 99.5%+) are REJECTED by the screen, not
  waved through; the coverage cost is included in the numbers above.

## 7. What is NOT claimed

- Nothing about sh_262k (m3 measured net harm there; out of scope, declared).
- Nothing about the write path (NO_GO, permanent).
- Nothing about other substrates, dtypes or prompt shapes; the bf16 retrieval
  deviation and the sh_32k→sh_64k noise transfer are stated limitations.
- Calibration net-help does not transfer numerically; only the sign and the
  mechanism are hypothesized to transfer. The campaign is the test.

## 8. Execution checklist (in order, single shot)

1. This file + operating point + analysis code committed; suite green on the
   box; supervisor audit of calibration + this document PASSED.
2. `serve_stage1_chat.sh` and `serve_stage1_embed.sh` up; user's :8000/GPU0
   untouched.
3. `nohup bash stage0_results/stage1_campaign_driver.sh ...` (driver enforces
   the preconditions itself).
4. `python stage0_results/stage1_campaign_analysis.py` → committed result +
   report. No second campaign without a new pre-registration.
