# H-Nav Stage 0 — build notes

Local build phase (T2–T8), written on an Intel MacBook Air with no GPU and no
`torch`. Everything here is what a reader needs in order to trust — or distrust —
the code before it runs on the 2× RTX 4090 box.

`pytest hnav/tests/ -q` → **151 passed**, with no torch, no GPU and no network.

---

## 1. What was built

| Task | Files | What it is |
| --- | --- | --- |
| T3 | `hnav/config.py`, `core/{types,embedding,replica,audit}.py` | Core types, the two retriever replicas, the audit log, and the embedder interface |
| T4 | `adapters/{mab_adapter,clbench_adapter}.py`, `labeling/conflict_index.py`, 4 benchmark edits | Shadow-mode instrumentation for both arenas |
| T5 | `core/retrieval_signals.py`, `stage0/m2_retrieval_calibration.py` | Rank/margin/entropy/dH/churn, and the M2 measurement |
| T2 | `stage0/m1b_grouping_ablation.py` | Regex-vs-geometry grouping ablation |
| T6 | `core/{geometry,diff_geometry}.py`, `labeling/{labels,counterfactual}.py`, `stage0/m3_headroom.py` | Geometry signals, the retained labels, counterfactual replay, headroom |
| T7 | `stage0/m4_marginal_diff_test.py` | The H2 nested-model test |
| T8 | `stage0/report.py` | `STAGE0_REPORT.md` and the GO/NO_GO gate |

Not built, deliberately: **`core/write_policy.py` and `core/read_policy.py` do not
exist.** They are live-intervention code and are gated behind the T8 human
decision. `test_no_raw_entropy_in_policy.py::test_stage0_ships_no_policy_modules`
fails if either appears.

---

## 2. Things found while building that change how the remote run behaves

### 2.1 `FACT_RE` alone matches **zero** facts on a real memorize chunk

The brief specifies exploding a chunk with
`FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)`. That regex is exact on the
raw `context` string, but the benchmark does not hand H-Nav the raw context. It
hands it the output of `chunk_text_into_sentences`
(`utils/eval_other_utils.py:173`), which ends in

```python
text_chunks.append(" ".join(current_chunk_sentences))
```

Every newline that fell on a sentence boundary becomes a space, so a
line-anchored regex has nothing to anchor to. Measured on the committed `sh_6k`
context: **`FACT_RE` finds 0 of 455 facts** on the joined form.

`explode_facts` therefore tries `FACT_RE` first — it is the prescribed regex and
is exact on raw context — and falls back to `FACT_RE_INLINE`, which runs each
fact to the next serial. On `sh_6k` the two forms give byte-identical results,
455/455. Covered by `test_chunking_and_facts.py`, including an explicit assertion
that the line-anchored form fails, so the fallback can never be quietly deleted
as redundant.

**Without this, every write-side signal on the primary arena would have been
computed from one fact per 4096-token chunk.**

### 2.2 A single arena-wide score convention

`RetrievalView.scores` is always "higher is better". `NumpyCosineReplica` emits
the native `(bank @ q) * 100.0` unchanged. LangChain FAISS returns **squared L2
distance**, so the retriever hook passes `score_kind="l2sq"` and
`to_similarity` converts with `cos = 1 − d²/2`, exact for the L2-normalized
embeddings both shipped embedders produce and strictly decreasing, so ranks are
untouched. One conversion, one place, one test.

### 2.3 The harm criterion implies a minimum sample size

A consequence of the frozen gate, not a choice made here: with **zero** observed
harm, the 95% Wilson upper bound only falls below 0.03 at n ≈ 124. A component
with the minimum 40 positives cannot clear the harm row however clean it is.
Worth knowing before sizing the campaign. Asserted in
`test_report.py::test_the_harm_criterion_implies_a_minimum_sample_size`.

---

## 3. What is tested locally

Every numeric quantity is checked against a **closed-form or independently
computed answer**, not against "it ran".

- **Replica fidelity (M0).** 1,200 sampled `(store, query)` pairs against the
  native two-liner inlined verbatim from `qwen3_embedding_memory.py:218`:
  100% top-1, 100% top-k, 100% full-ranking identity, **zero** score error.
  Plus a deliberate exact-tie fixture, a non-mutation check, and
  fast-path-equals-rebuild for `simulate_insert`.
- **Retrieval signals.** `ln m` for an orthonormal neighbourhood and 0 for a
  collinear one (`H_vn`); uniform scores giving maximum entropy and full tie
  counts; exact churn, displaced-id and rank-shift counts; raw-softmax
  saturation demonstrated on a `cosine × 100` fixture.
- **Geometry.** QR residual 0 inside a span, 1 outside, 0.8 for a known partial
  projection; ABTT refusing below `min_fit_n`, removing an injected dominant
  direction, and being deterministic; the `tau_t` trajectory hand-computed.
- **Labels.** 455 real write contexts from `sh_6k` and 120 read contexts, each
  label compared against an oracle reimplemented independently in the test, plus
  determinism and the `READ_CLEAR` complement property.
- **Counterfactual.** `normalize_answer` pinned to the benchmark's evaluator;
  the five classes exercised with stub models that reproduce specific failure
  modes (latest-wins, first-fact-anchored); prompt-cache accounting.
- **H2 statistics.** Informative *and* null synthetic regimes: a correct
  implementation must show ΔAUC > 0 with tiny LRT p on one and neither on the
  other. Bootstrap, cross-validation and matrix assembly checked separately.
- **Report.** All three NO_GO verdict types produced from inputs that are
  unambiguously each one; unmeasured criteria proved never to render as
  failures; Wilson interval endpoints.
- **Leakage.** The AST scanner is run against three deliberate violations
  (a `["answers"]` subscript, a `["questions"]` subscript, a `counterfactual`
  import) and must flag all three, and against clean prose it must not flag.
  A scanner that cannot fail is decoration.
- **No torch.** Every module under `hnav/` is imported in a fresh subprocess and
  the test fails if `torch`, `transformers`, `faiss`, `openai`, `langchain` or
  `vllm` lands in `sys.modules`; a second test forbids them as top-level imports
  textually.

### Pipelines actually executed end to end, locally

`--smoke-embedder` swaps in `HashEmbedder` and `--stub-llm` swaps in a
deterministic answer stub, so M1b, M2, M3 (counterfactuals included) and M4 have
all been **run to completion on the real dataset** on this laptop. The numbers
are meaningless and are written to separate `*_SMOKE.json` files so they can
never be mistaken for data — but the plumbing, the JSON shapes and the failure
paths are exercised.

---

## 4. What is necessarily untested until it reaches the GPU

Listed plainly, because these are where a first-try failure would come from.

1. **The real embedder.** `HFEmbedder` has never been constructed — no torch
   here. Its pooling mirrors `Qwen3Embedding4BEmbeddings`
   (`embedding_retriever.py:58`) line for line, and its cache layout is
   byte-compatible with T1's, but *loading a model* is untested. **Run T1 first;
   it is the cheapest thing that exercises this path.**
2. **Live-index M0.** `FaissFlatReplica` is verified against squared-L2
   ordering, not against a live LangChain FAISS index. FAISS's internal tie order
   is not reproducible from outside it. Until that check runs, the report prints
   M0 as NOT RUN and treats `rank_self`, `margin`, `dH_*` and `churn` as
   provisional. Producing `m0_replica_fidelity.json` is the first remote job
   after T1.
3. **Run-level shadow neutrality.** The acceptance criterion is byte-identical
   model output between `HNAV_MODE=off` and `shadow` on `sh_6k`/`sh_32k` at
   `temperature=0`. That needs the benchmark, the LLM server and a GPU. What is
   verified here is the *mechanism*: hooks return the caller's own object by
   identity, no store is mutated, the wrapper makes exactly one inner call, and
   all four benchmark edits keep their native branch intact.
4. **The four benchmark files import torch at module level** (they are benchmark
   code, not H-Nav code), so they are checked by `ast.parse` and marker
   assertions rather than by import. They have never been executed.
5. **`local_answer_fn` against vLLM.** The client is constructed lazily and has
   never talked to an endpoint. `--stub-llm` exercises everything downstream of
   it.
6. **The real chunker.** `nltk`/`tiktoken` are absent here, so `build_chunks`
   fell back to line grouping in every local run. The fallback is recorded as
   `"fallback_chunker": true` in every output file — **if that flag is `true` in
   a GPU run, nltk or punkt is missing and the chunking is not the benchmark's.**
7. **Scale.** The largest local run was `sh_6k` (455 facts, 2 chunks). Nothing
   has been run at 18,332 facts. The two places that could bite: the QR basis
   (mitigated by `_QRBasis`, refreshed every 500 admissions rather than rebuilt
   per candidate) and the `sh_262k` bank matrix at ~188 MB in float32.

---

## 5. Deviations from the brief, with reasons

1. **`explode_facts` has a second regex.** §2.1. The prescribed `FACT_RE` is
   tried first and is exact on raw context; on a real chunk it matches nothing.
2. **`hnav/config.py` and `hnav/core/embedding.py` are additions.** The brief
   requires reading config from `.env` and mocking the embedder behind one
   interface; both needed a home, and neither belonged in an existing file.
   `hnav/deploy/.env.template` was referenced by the brief but did not exist, so
   it was written (and un-gitignored, since `.env.*` was matching it).
3. **`diff_novelty` is defined against the store's prior object vocabulary**,
   not as a transform of `diff_sim`. Defining it as `sqrt(1 − diff_sim²)` would
   have made it a deterministic function of a feature already in the model, and
   M4's likelihood-ratio test would then be testing a nonlinearity rather than
   added information.
4. **M4 reports two extra numbers**: a key-clustered bootstrap CI and a
   group-k-fold cross-validated ΔAUC. The protocol's subset-clustered CI is
   still primary and still decides the verdict, but the calibration split has
   **two** subsets, so that bootstrap resamples from two clusters. Reporting only
   it would be reporting a number that cannot mean what it looks like.
5. **`test_leakage_audit` exempts docstrings.** Prose has to be able to state the
   rule. Comments are invisible to the AST anyway. Every other string constant,
   identifier, attribute, keyword and dict key is scanned.
6. **`READ_DISTRACTOR` / `READ_MISSING` use the offline question→key mapping.**
   The protocol marks them online-computable, but that needs a query-side parser
   Stage 0 does not build. Their M3 rates are therefore an **upper bound** on
   what an online detector could reach, and the code and the report both say so.
7. **The QR basis is refreshed every N admissions**, not rebuilt per candidate.
   A 512-column QR per candidate over 18,332 candidates is not a measurement.
   The basis is a random subsample either way, so this is a change of sampling
   frequency, not of estimator; the interval is in every output file.
8. **`AuditLogger` truncates the persisted ranking at 200 entries.** Signals are
   computed on the full pre-truncation ranking; only the stored copy is
   shortened, or an 18,332-entry list would dominate every log record.

---

## 6. Order to run on the GPU box

```bash
cp hnav/deploy/.env.template .env      # then edit
bash hnav/deploy/setup_remote.sh
python hnav/deploy/check_env.py
pytest hnav/tests/ -q                  # must still be green with torch installed

bash hnav/deploy/run_t1.sh             # T1 — the S3 gate. STOP and report.
python hnav/stage0/m1b_grouping_ablation.py       # T2, reuses T1's cache
python hnav/stage0/m2_retrieval_calibration.py    # T5
python hnav/stage0/m3_headroom.py                 # T6, needs the LLM server
python hnav/stage0/m4_marginal_diff_test.py       # T7, calibration split only
python hnav/stage0/report.py --strict             # T8 — HARD STOP
```

`run_t1.sh` exits 2 if the S3 gate fires. **Stop there and report to a human** —
that is the kill switch, and everything after it assumes it passed.

---

## 7. T9 — Stage-1 Faz A: the two-stage read gate

Built after the T8 verdict (`KAPI_KARARI.md`) and the user's 2026-08-15 decision
(`STAGE1_PLAN.md` §0), as the precondition for the Faz B agent.

1. **`hnav/core/read_gate.py`** — benchmark-agnostic. Stage 1: pair-cosine
   screen (default 0.92 = mean of M1b best-F1 taus on the calibration split)
   → connected components → group-level R screen (min leave-one-out span
   residual < frozen `r_min` 0.1924). Preconditioned on ranking ambiguity via
   the frozen `nmargin`/`H_z` thresholds, consumed only through `for_policy()`.
   Stage 2: bidirectional NLI — a pair is a verified conflict only if the
   contradiction score clears the threshold in BOTH directions. LATEST is
   named via an adapter-supplied `latest_key`; ties/missing keys refuse to
   guess. The gate decides; it mutates nothing. `read_policy.py` (rerank
   execution) is deliberately NOT built here — Faz B.
2. **NLI engine**: `cross-encoder/nli-deberta-v3-large`, measured on the box:
   1.76 GB fp32 weights on GPU1, 1.98 GB steady with batch-16 inference
   (peak allocated 1.88 GB), 3.9 s load, ~235 ms/batch-16 on GPU. Fits next
   to the ~17 GB fp32 embedder with ~5.6 GB slack. The arena-shaped
   supersession pair scores contradiction 0.99996 in both directions.
   `check_env.py` gains a gate: weights cached + 3-pair smoke with known
   labels (contradiction / entailment / neutral, on CPU) + the arena pair
   asserted bidirectionally.
3. **Protocol transition** in `test_no_raw_entropy_in_policy.py`:
   `write_policy.py` forbidden FOREVER (measured NO_GO, not a deferral);
   `read_policy.py` permitted post-T8; the `H_raw` AST scan now also covers
   `read_gate.py`.

Deliberately untested / known limits:

- `CrossEncoderNLI` has no automated GPU test — validated by `check_env`'s
  CPU smoke plus the one-off GPU measurement above. `StubNLI` (deterministic,
  torch-free, TEST USE ONLY like `HashEmbedder`) carries all closed-form tests.
- With the frozen defaults the gate is **precision-first by construction**:
  for a two-member group whose other candidates are unrelated, the LOO
  residual is `sqrt(1 − cos²)`, so `r_min` 0.1924 implies pair cosine ≳ 0.981
  — stricter than M1's median true-conflict sim 0.964. Stated in the module
  docstring; Faz B's coverage-balanced calibration (sh_6k+sh_32k ONLY) is
  where the balance is chosen deliberately.
- The gate has not yet seen real embeddings end-to-end (no adapter wiring
  yet); that wiring plus threshold tuning is Faz B's first task.

## 8. T10 — CrossEp instrumentation repair + M5 write-headroom (measurement only)

Charter: `HNAV_VISION_GAP.md` §4 step 4. Full findings and the [GATE]-feeding
numbers: `CROSSEP_HEADROOM_RAPORU.md` (repo root). No policy code anywhere.

1. **`CLBenchAdapter.on_extract` blind spot killed**: geometry / diff /
   retrieval_effect now flow when modules are injected. Signals are computed
   in ONE space — the adapter re-embeds bank texts with its own embedder
   (memoized; native DashScope vectors are dimensionally incompatible with
   H-Nav candidate vectors, so the old mixed-space path would have crashed
   and been swallowed). Predecessor = nearest admitted neighbour (no
   keys/serials in this arena; no look-ahead). Probe = the candidate's own
   text (`simulate_insert` default). `on_retrieve` got the same single-space
   fix. Wrapper recovers `context_id` from the backend's per-context
   `memory_dir`; the inner extract call receives the caller's kwargs
   untouched. Shadow legality unchanged: PASS-only, no LLM calls, identity
   returns, off = exact no-op.
2. **CrossEp calibration split frozen**: `hnav/labeling/crossep_split.json`
   (+ derivation in `crossep_split.py`, seed 20260815) — cluster-level
   (`context_id`), stratified by category: 48 calibration / 72 held-out
   clusters (347/537 samples). Any CrossEp threshold may be fit on
   calibration clusters ONLY; the suite fails if artifact and derivation
   ever disagree.
3. **`hnav/stage0/crossep_m5_write_headroom.py`**: replays a per-context
   write stream through the wired shadow instrumentation; cluster-first
   stats (never candidate-pooled); streams: qwen3_embedding (reconstructed
   exactly, transcribed 1024-token chunker, `fallback_chunker` recorded),
   mem0_history (reads any run's `history.db`; no run exists yet),
   generic_jsonl. Optional offline NLI reuses the T9 cross-encoder
   (labeling privilege; stub engine only under `--smoke-embedder`).
4. **Measured (box, CPU, real chunker, smoke embedder)** — MD5-dup and
   lexical-Jaccard are embedder-free and REAL; cosine numbers in the smoke
   file are meaningless by construction: 120 contexts, 7,879 write events;
   exact-dup cluster-mean **0.117 calibration / 0.072 held-out** (89/120
   clusters have dups; worst 0.706); Jaccard≥0.9 rate 0.164/0.112. Versus
   the MAB substrate's duplicate_rate 0.000 everywhere: the write-cascade
   question is live on this substrate, pending the real-embedder run
   (≈0.3–0.4 GPU-h, orchestrator-scheduled) + NLI base rates.
5. **MemOS triage**: `.gitignore:90` (`memories/`) swallowed
   `MemOS/src/memos/memories/` at vendoring — never committed, hence
   `_MEMOS_AVAILABLE=False` on every checkout. Cheap fix documented in the
   report; excluded from M5 scope (its stream is LLM-extracted and no run
   artifact exists).

Tests 175 → 196 (`test_crossep_adapter_signals.py`, `test_crossep_split.py`,
`test_crossep_m5_headroom.py`), all green locally and on the box; leakage
audit and no-torch-at-import untouched and green.

## 9. T11 — Stage-1 Faz B: rerank policy, live wiring, calibration, substrate

Built per `STAGE1_PLAN.md` §2 Faz B, on top of the audited Faz A gate, with
the two binding supervisor findings addressed head-on.

1. **`hnav/core/read_policy.py`** — the single permitted policy module.
   `rerank_order` promotes each verified group's LATEST carrier immediately
   above its highest-ranked stale rival; deterministic (plans sorted by first
   stale position, single pop-and-insert each), token-neutral by construction
   (non-permutations raise). `ReadRerankPolicy` packages gate + rerank into a
   `Decision` that is born `shadow=True`; only the adapter arms it, only under
   `HNAV_MODE=live`.
2. **Note-1 mitigation, measured then frozen.** The gate gained an optional
   `pair_filter` (adapter-supplied identity evidence, applied to cosine-passed
   pairs BEFORE NLI). Calibration measured the exposure on sh_6k+sh_32k with
   the real NLI: WITHOUT the screen the bidirectional criterion false-verifies
   at 33–93% of verified pairs depending on the cosine screen (dominant class:
   same-template/different-subject, the audit's measured shape — e.g. 12,896
   cross-key rubber-stamps vs 923 true supersessions at cos 0.90). WITH
   parsed-key equality (`MABAdapter.same_key_pair`, unparseable pairs
   rejected) the measured false-verified rate is 0.000 at every one of 162
   grid cells, verification precision 1.00. The screen is frozen ON.
3. **Live wiring** (`mab_adapter` + the retriever hook): candidate pool =
   facts in the retrieved page capped at `top_m` by query-fact cosine
   (`select_pool`, shared verbatim with the calibration harness);
   `latest_key` = the benchmark serial; `apply_read_decision` returns the
   same chunk set and count, new order only, native fallback counted on any
   irregularity. `get_adapter` assembles the default stack for hook callers
   (signals, read audit, cache-first NON-persisting endpoint embedder so a
   bf16-served fallback can never poison the fp32 cache, CPU-capable NLI).
   Off stays inert; shadow computes and logs but never applies; the seam in
   `embedding_retriever.py` acts only on an armed (non-shadow) decision.
   `require_not_live` lifted from the adapter ONLY — every Stage-0 script
   still refuses live (both halves pinned in tests).
4. **Calibration** (`hnav/stage1/calibrate_read_policy.py`): sh_6k+sh_32k
   only (others refused), campaign-faithful queries (RAGSystem's templated
   retrieval extraction) and grading prompt (Memory-numbered, system message,
   max_tokens 10, against the frozen :8003 substrate), 162-cell grid replayed
   through the REAL gate + rerank with a replay NLI that refuses unknown
   pairs. ABTT A/B logged (AUC same-key separation, raw→whitened: sh_6k
   0.936→0.955, sh_32k 0.988→0.991 — evidence FOR whitening, no decision uses
   it). Objective pre-registered in the module docstring.
5. **Substrate frozen** (`serve_stage1_chat.sh` / `serve_stage1_embed.sh`):
   :8003 Qwen3-4B bf16 weights, 65536 window, fp8 KV, eager, batch 1, no
   prefix cache, util 0.58; :8001 embeddings **bf16** util 0.33 — a declared
   deviation forced by arithmetic (fp32 embed weights + a 65k window do not
   fit one 24 GB card; GPU0 carries the user's :8000, untouched). Gate
   geometry stays fp32 via the cache-first embedder; both arms share the
   bf16 retrieval.
6. **Campaign tooling, not run by the builder**: `stage1_campaign_driver.sh`
   (7 off + 7 live interleaved, refuses to start without the committed
   pre-registration + operating point + a live-stack preflight),
   `stage1_campaign_analysis.py` (the pre-registered criteria, verbatim),
   `stage1_aa_driver.sh`/`stage1_aa_analysis.py` (frozen-substrate noise
   floor, sh_32k), `stage1_shadow_check.sh` (realized-coverage rehearsal).

Known fidelity gaps, declared rather than hidden:

- The benchmark stores MEMORIZE-TEMPLATED chunks (wrapper + per-chunk
  timestamp); the harness ranks raw chunks — the same simplification M2/M3
  made. Fact-level gate geometry is unaffected (pool records are the clean
  admitted facts); the chunk-ranking and grading-prompt deltas are absorbed
  by the shadow coverage check on sh_32k, and the retrieve-time re-parse of
  templated chunks can contaminate only the LAST fact's *text* in the
  chunk-membership map, never the fact records themselves.
- Campaign retrieval runs on bf16 embeddings; nmargin/H_z firing rates under
  bf16 are a declared transfer assumption (same check).
- The A/A noise floor is measured on sh_32k, not the confirmatory sh_64k —
  declared in the pre-registration as a transfer assumption.

### 9b. Faz B calibration outcome (preliminary — box verification queued)

The graded evaluation ran to completion on the box against the frozen :8003
substrate (~470 chat calls; benchmark prompt shape, max_tokens 10) and the
pre-registered objective printed:

    NO feasible operating point with net > 0. REPORT AND STOP — do not force one.

i.e. on sh_6k+sh_32k no grid cell produced positive net help within the harm
caps. Grading-independent numbers already extracted: ambiguity screen (mode
"any") fires 94/200; order changes on 68–115/200 questions per cell; Note-1
false-verified rate 0.000 with the key screen vs 0.33–0.93 without.
Cell-level helped/harmed distributions live in `hnav/_out/
stage1_calibration.json` on the box (unreachable at time of writing).

A plausible mechanism for a genuine null, to be tested against the cell data:
the conflict prompt TELLS the model to resolve by highest serial across the
whole pool, and on the calibration subsets every chunk fits the context —
chunk ORDER may simply not bind this model's answer. If the null stands, the
rerank-only Stage-1 design has no measurable lever on this substrate, and the
campaign must not run — that is a result, the same way the write-path NO_GO
was. Human decision required (STAGE1_PLAN anticipated only the positive
case). No operating point frozen; pre-registration committed as DRAFT/BLOCKED.

### 9c. T11 supervisor-audit fixes (local)

Two corrections from the T11 audit, both applied:

1. **The "every `hnav/stage0/` script refuses live" claim was not literally
   true** when first made: 6 of 9 called the guard;
   `m1_geometry_calibration.py` and `report.py` had none. Practical risk was
   nil (neither touches the adapters or the benchmark hooks), but an invariant
   asserted in prose and enforced nowhere decays, so the guard is now
   **uniform and mechanically enforced**. `report.py` calls
   `cfg.require_not_live()`; `m1_geometry_calibration.py` — deliberately
   stdlib-only, it imports the validated parser by path and never imports the
   `hnav` package — carries an inline `require_not_live(env)` twin with
   identical semantics. `hnav/tests/test_stage0_refuses_live.py` AST-scans
   every `hnav/stage0/*.py` for a live guard reachable from `main()`
   (docstrings, comments and guards parked in uncalled helpers do not count),
   exercises BOTH guard forms for actual raising behaviour, and carries
   negative controls so the scan can fail. `MABAdapter` remains the single
   deliberate exemption, pinned separately in `test_shadow_neutrality.py`.
2. **Test count corrected.** The "236 passed" figure in the T11 commit
   messages was a PRE-MERGE count; the post-merge baseline including Thrust-2's
   tests is 238, and with the 15 new guard tests the suite is **253**.

Unrelated latent bug observed while verifying (NOT fixed — out of T11 scope,
flagged for whoever owns `report.py`): on Windows, `report.py` crashes with
`UnicodeEncodeError` writing `STAGE0_REPORT.md`, because `path.write_text(text)`
takes no `encoding` and the report contains `≥`. Reproduced with all T11
changes stashed, so it predates this work; harmless on the Linux box (UTF-8
default). One-word fix: `write_text(text, encoding="utf-8")`.

## 10. T12 — the 512-token truncation defect, corrected

**What was wrong.** `HFEmbedder.__init__` carried `max_length=512` and
`build_embedder` never overrode it, because it passed four POSITIONAL
arguments and stopped one short of that parameter. The standalone T1 embedder
(`m1_geometry_calibration.py`) hardcoded `max_length=512` in its own
`tok(...)` call. Meanwhile the benchmark chunks context at 4096 *tiktoken*
tokens. Measured on the calibration split with the real chunker: chunks reach
**4,333 tiktoken tokens / 17,675 characters**, so 512 tokens captured roughly
the first **12%** of the largest chunk.

**What that invalidates.** Every offline chunk-level signal, and every
threshold fit from one, was computed from a truncated prefix while the LIVE
path consumed the benchmark's own full-length ranking (the served endpoint
truncates at its `--max-model-len 16384`, which no real chunk reaches). The
two paths were not measuring the same vectors. M0's 1.0000 replica fidelity
does NOT cover this: it reused the benchmark's own vectors rather than
recomputing them through `HFEmbedder`. Fact-level signals are largely
unaffected — a single fact is far below 512 tokens — so the damage is
concentrated in chunk-level geometry and anything derived from it.

**The correction.**
- `DEFAULT_MAX_LENGTH = 8192` in `hnav/core/embedding.py`, configurable via
  `HNAV_EMBED_MAX_LENGTH` / `HNavConfig.embed_max_length`. 8192 covers the
  worst measured chunk with ~1.7x headroom for tiktoken↔Qwen tokenizer
  disagreement, and stays under both ceilings that bound the live path:
  Qwen3-Embedding-4B's 32768 native context and the embed endpoint's 16384
  window. (The model's real limit is asserted from the model card; the
  re-fit runbook re-checks it on the box from `config.json`.)
- `build_embedder` now passes every argument BY KEYWORD, so a future
  omission is visible at the call site instead of silently defaulting.
- `m1_geometry_calibration.py`'s standalone embedder takes `max_length`,
  reads the same env var, and prints it in the run banner.
- **`cache_key` now includes the length** (`model|dtype|L8192`) and the
  argument is REQUIRED. This is the load-bearing half: the cache key is
  `sha256(namespace||text)`, so without it the corrected embedder would have
  read back the ~24k vectors the truncated one had already written on the box,
  and the fix would have looked like a no-op while changing nothing. Old
  entries stay on disk under the old namespace as provenance.

**Regression cover** (`hnav/tests/test_embedding_truncation.py`, 7 tests):
the configured length covers the largest chunk MEASURED from the dataset (not
asserted from memory); the length actually reaches the tokenizer (behavioural,
via a fake tokenizer, so it holds without a GPU); the cache namespace makes a
length change a MISS rather than a wrong hit; and `build_embedder` wires the
length into both the model and the namespace. Verified to FAIL on both silent
reverts (truncation back to 512, namespace dropped): 3 tests fail in each case.

**Consequence — thresholds must be re-fit.** `nmargin`/`H_z`/`r_min` in
`stage0_results/final/m3_headroom.json` were fit on truncated vectors and are
not valid for the corrected embedder. The re-fit is CALIBRATION-SPLIT ONLY and
is specified in `hnav/deploy/REFIT_RUNBOOK.md`; until it lands, no frozen
threshold from the truncated era may be used to justify a live decision.

### 10b. T12 note 5 — the same defect class in the NLI cross-encoder

Found by the T12 audit: `CrossEncoderNLI` defaulted to `max_length=256`, and
**premise and hypothesis share that budget**. Harmless in the primary arena (a
fact is one short sentence, far below it — which is why every T11
primary-arena NLI number stands unchanged) but not in CrossEp:
`crossep_m5_write_headroom.run_nli` truncates each chunk to 1200 chars and
pairs them, so each side was cut to roughly its first ~128 tokens.

**Measured on the real CL-bench contexts with the real chunker** (60 contexts,
505 chunks), per side after the 1200-char cut: **p50 291 / p90 342 / max 674**
tiktoken tokens — **94.7% of sides exceed the old 128-token half-budget**.

**The honest part: this one cannot be fully fixed by configuration.** 512 is
DeBERTa-v3's own `max_position_embeddings`, so it is a ceiling, not a headroom
choice — and **71.5% of CrossEp pairs still exceed it** (pair p50 585, max
1351 tokens). Unlike the embedder (8192 ≪ 32768, fully fixed), the residue
here is STRUCTURAL. It is therefore *reported* rather than assumed away:
`CrossEncoderNLI` counts `n_scored`/`n_truncated` and exposes
`truncation_rate` / `truncation_report()`, and M5's `run_nli` output carries a
`"truncation"` block. If the CrossEp NLI numbers matter, that rate must be
quoted beside them.

Corrections, mirroring the embedder fix:
- `NLI_MAX_LENGTH_DEFAULT = 512`, configurable via `HNAV_NLI_MAX_LENGTH` /
  `cfg.nli_max_length`;
- `build_nli` and M5's construction pass **every argument by keyword** — both
  previously stopped before `max_length` and inherited a default nobody chose;
- `check_position_limit()` refuses a budget above the checkpoint's own
  `max_position_embeddings` (module-level so the guard is unit-testable
  without loading 1.7 GB of weights) — no silent extrapolation.

**Caching, checked explicitly** (the embedder's namespace hazard is why):
`CrossEncoderNLI` and `StubNLI` hold **no cache** — scores are recomputed per
call, so there is no key to poison. But `hnav/stage1/calibrate_read_policy.py`
**persists** NLI scores (`nli_table` in `stage1_prepass_*.json`, keyed by
`sha1(premise)|sha1(hypothesis)` with no engine parameters) and `--evaluate`
replays them, which is the same hazard. The prepass now stamps
`nli_config` (model, max_length, stub) and `--evaluate` REFUSES a table
scored under a different configuration, rather than silently mixing.

Regression cover: `hnav/tests/test_nli_truncation.py`, 9 tests, torch-free
(numpy shims for the tokenizer/tensor surface, so the counter logic is
verifiable on a torchless machine like the rest of the numeric core). Verified
to fail on each silent revert: default back to 256 (2 fail), `build_nli`
keyword dropped (1), M5 call-site keyword dropped (1). Includes an AST check
of the M5 call site and a measured assertion, from the real data, that CrossEp
pairs genuinely exceed both the old and the new budget.

**Blocks:** the CrossEp M5 real-embedder run (`--nli cpu`) must not launch
before this lands.

---

## 11. T13 — fact-level mechanisms, and closing the oracle-to-detector gap

### 11.1 What the probe settled

The T12 oracle probe made the mechanism question decidable: on `sh_6k`'s
conflicted stratum, deleting the stale fact takes accuracy from 4/74 to 66/74;
moving the newest fact to the very END takes it to 20/74; moving it to the
FRONT takes it to 1/74. `sh_32k` replicates all three. The model anchors on the
**late-appearing** value, and it does so per FACT — which is also why T11's
chunk-level upward rerank was harmful: it moved the superseder *away* from the
position that helps, at a granularity (230–260 facts per chunk) that scrambles
hundreds of unrelated ones.

What the probe could not answer is whether any of that survives a detector.
Every probe arm reads the expected answer to decide which fact to cut or move.

### 11.2 The two mechanisms, and the page-contract decision

`hnav/core/read_policy.py` keeps `rerank_order`/`ReadRerankPolicy` untouched —
it is the evidence for the T11 null, not legacy — and adds two actions driven by
the same `GateDecision`:

| action | ids | effect | tokens |
| --- | --- | --- | --- |
| `SUPPRESS` | `suppress_ids` = every stale member of every verified group | dropped from the page | **down** |
| `DEMOTE_LATE` | `demote_ids` = each verified group's LATEST carrier | moved to the end of the page | **neutral** * |

\* A move always preserves the fact multiset exactly, and the character count
too — except when the moved fact was ALONE in its chunk, where it takes no
separator with it but acquires one on arrival (`+len(sep)`, source chunk left
empty in place so the page count never changes). A chunk here carries 230-260
facts, so that corner cannot arise; `page_edit` reports `delta_chars` either way
and the sh_6k run measured exactly 0.00%.

**The decision that mattered:** the benchmark hook returns a page of *chunk*
texts, so a fact-level edit has to be expressed as a rewrite of chunk text. The
policy does **not** do that rewrite. It emits ids
(`payload["drop_ids"]` / `payload["move_last_ids"]`) and the ADAPTER splices,
because locating a fact inside a chunk requires the benchmark's serial numbering
and its sentence-joining chunker — knowledge hard rule 2 keeps out of
`hnav/core/`. The adapter side is `mab_adapter.fact_spans` / `page_edit`.

The splice is byte-exact and the exactness is a contract, not an aspiration:

- surviving facts keep their **original serials**, gaps included — the prompt
  states its rule in terms of serial order, so renumbering would redefine the task;
- surviving text is spliced from the original bytes: no re-rendering, no reflow,
  no whitespace normalisation;
- a fact owns the separator that FOLLOWS it, and the **last** fact of a chunk
  owns the one BEFORE it — without that asymmetry, deleting the last fact of a
  newline-joined context leaves a blank line;
- the chunker's **dangling next-fact serial** (`"…Leeds. 307."`) is outside
  every span: those bytes belong to a fact whose text lives in the next chunk;
- `DEMOTE_LATE` re-appends moved facts in ascending serial order **before** the
  last chunk's trailing whitespace, so the newest ends up last and the trailing
  bytes survive;
- the page keeps its length and block order in every case. Only contents change.

Proven rather than asserted: on the real 455-fact `sh_6k` context,
`page_edit(drop…)` and `page_edit(move…)` are byte-identical to the probe's own
`suppress()`/`move_to_end()` + `render_context()` over 60 random draws
(`test_read_policy_facts.py`). Without that the detector/oracle ratio would be
comparing two different interventions.

Invariants: `HNAV_MODE=off` is a byte-identical no-op; decisions are born
`shadow=True` and only the adapter arms them under live; any irregularity — an
id off the page, overlapping id sets, an unparseable chunk — returns the native
page and increments `n_rerank_fallbacks`; `H_raw` is never consulted.
`HNAV_READ_MECHANISM` selects the live mechanism and defaults to `rerank`, so
adding these changed no existing command's behaviour.

### 11.3 The frozen operating point

`hnav/stage1/detector_gap.py --select` scores the same 162-cell grid
`calibrate_read_policy` declared, on **detection quality only** — no LLM, no
accuracy, no gold answer — and freezes
`stage0_results/stage1_operating_point.json` before any arm is graded. The rule
is pre-registered in `SELECTION_RULE`: require `pair_filter=True` and
zero suppressions that would change what the page says a key's current value is;
maximise pair recall within the pool; break ties toward the more conservative
gate. 81 of the 162 cells are admissible — exactly the `pair_filter=True` half,
so in practice the identity screen is the binding requirement. It is a
requirement rather than a preference because of what the other half looks like:
with the screen off, median pair precision across cells is **0.137**, the median
cell would delete **769** facts that carry their key's current value, and the
median cell cuts the gold-valued fact on **4** conflicted questions (max 15).

```
cos_pair 0.90 · r_min 0.44 (loose) · ambiguity_mode none · nli 0.90 · pair_filter True
```

Calibration split (sh_6k + sh_32k, 200 questions):

| | pooled | sh_6k | sh_32k |
| --- | --- | --- | --- |
| pair precision | **1.0000** (2,673 verified, 0 false) | 1.0000 | 1.0000 |
| pair recall, in pool | 0.9784 | 0.9806 | 0.9759 |
| pair recall, whole page | 0.0269 | 0.0885 | 0.0151 |
| conflicted-question recall | **133/139 = 0.957** | 72/74 = 0.973 | 61/65 = 0.938 |
| facts named stale | 2,673 - all genuinely superseded | 1,416 | 1,257 |
| would change a key's current value | **0** | 0 | 0 |

`ambiguity_mode="none"` disables the frozen Stage-0 `nmargin`/`H_z`
precondition. Declared, argued in the artifact's `ambiguity_note`, and asserted
to be declared by `test_threshold_provenance.py`: those two signals are the only
gate input derived from CHUNK embeddings, which are still truncated at 512 of
~4096 tokens (§10, un-refit because re-embedding needs a GPU both servers hold);
they are the dominant recall bottleneck (conflicted-question recall 0.957 with the screen off,
0.403 at `ambiguity_mode="any"`, 0.144 at `"all"`); and the volume-limiting job they did is now done by the identity
screen plus bidirectional NLI at precision 1.00. FACT vectors are unaffected and
are *proven* to be the prepass's own — every pair cosine recomputed from the
loaded vectors reproduces the prepass's stored value to 8.9e-16, or the run
refuses.

### 11.4 The measurement: same 5 arms, detector-driven

`sh_6k`, 500 real calls, the frozen :8003 substrate, the probe's harness
imported verbatim so the two runs are comparable by construction. The cross-run
native check came back **identical** (29/100 in both), which is what licenses
the ratios.

| arm | overall | unique | conflicted | b/c | net | exact p | tok Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.290 | 25/26 | 4/74 (5.4%) | — | — | — | 0 |
| native_repeat | 0.290 | 25/26 | 4/74 | 0/0 | 0 | 1.0 | 0 |
| **detector_suppress** | **0.900** | 24/26 | **66/74 (89.2%)** | 1/62 | **+61** | 1.4e-17 | **−3.48%** |
| detector_demote_late | 0.320 | 25/26 | 7/74 (9.5%) | 2/5 | +3 | 0.45 | 0.00% |
| detector_anti | 0.250 | 25/26 | 0/74 (0.0%) | 4/0 | −4 | 0.125 | 0.00% |

**Detector-achieved / oracle-achieved:**

| mechanism | oracle arm | net ratio | conflicted-gain ratio |
| --- | --- | --- | --- |
| suppress | oracle_suppress | 61/62 = **0.984** | 62/62 = **1.000** |
| demote_late | oracle_recency | 3/17 = 0.176 | 3/16 = 0.188 |
| anti | anti | −4/−3 (both harmful; direction confirmed) | — |

On the stratum it targets, the detector is **exactly** the oracle: conflicted
b/c is 0/62 for both, 66/74 correct for both. The whole difference between the
two runs is one flip on the *unique* stratum — question 76, where native
answered "Shinzō Abe" and the suppressed arm answered "Sinzō Abe". A dropped
letter, not a wrong fact; the substring-exact evaluator scores it wrong. That is
the entire measured cost of replacing the oracle with the detector.

### 11.4b `sh_32k`: the same run on the other half of the calibration split

500 calls, ~17.2M prompt tokens, same substrate, same frozen operating point.
Cross-run native check identical again (42/100 in both runs).

| arm | overall | unique | conflicted | b/c | net | exact p | tok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.420 | 35/35 | 7/65 (10.8%) | - | - | - | 0 |
| native_repeat | 0.420 | 35/35 | 7/65 | 0/0 | 0 | 1.0 | 0 |
| **detector_suppress** | **0.860** | 35/35 | **51/65 (78.5%)** | 1/45 | **+44** | 1.3e-12 | -0.63% |
| detector_demote_late | 0.510 | 35/35 | 16/65 (24.6%) | 1/10 | +9 | 0.012 | 0.00% |
| detector_anti | 0.480 | 35/35 | 13/65 (20.0%) | 5/11 | +6 | 0.21 | 0.00% |

| mechanism | oracle arm | net ratio | conflicted-gain ratio |
| --- | --- | --- | --- |
| suppress | oracle_suppress | 44/46 = **0.957** | 44/46 = **0.957** |
| demote_late | oracle_recency | 9/26 = 0.346 | 9/26 = 0.346 |
| anti | anti | +6 vs -4 — **sign disagreement, see below** |  |

**The unique stratum is untouched: 35/35 under every arm, 0/0 discordant under
every arm.** The protective condition holds exactly here, and the whole harm of
the run is one conflicted question (index 8): native answered "London", the
suppressed arm "Washington, D.C.". That is the `n_conflicted_gold_cut` case the
frozen artifact predicted — the detector applies the benchmark's own serial rule,
so on the ~3% of questions where the gold value is *not* the newest fact it
deletes the gold. A bounded, countable property of the arena's rule rather than
a detector defect, and it must be declared in the sh_64k pre-registration.

**The complication worth stating plainly.** `detector_anti` HELPED at `sh_32k`
(+6, n.s.) where the oracle's `anti` hurt (-4). Two differences explain it and
neither rescues placement: the oracle arm also moved the most recent STALE fact
to the END (a deliberately adversarial second edit this mirror does not make),
and the detector aggregates ~13 latest carriers at one edge of a 2,310-fact
context instead of moving one fact. At that length both edges look privileged —
`demote_late` (+9) and `anti` (+6) both help — so at 32k the honest description
of the detector's placement mechanism is "collect the newest facts at *an* edge",
not "put the newest fact last". The clean directional result belongs to the
oracle's single-fact arms; the detector's multi-group version does not reproduce
it. Suppression is untouched by any of this.

### 11.5 Where the residual gap lives

Both `sh_6k` misses have **both** key members inside the 50-fact pool, so the
pool cap is not what loses them:

- **q26** (BBC director): `"…is Tony Hall, Baron Hall of Birkenhead."` vs
  `"…is Narendra Modi."` — pair cosine falls below the 0.90 screen because the
  objects are lexically far apart. A **geometry** miss.
- **q60** (John McVie's genre): `"rock music"` vs `"rap rock"`, cosine 0.9816,
  but bidirectional contradiction is 0.002 / 0.193 — the NLI reads them as
  compatible refinements rather than a conflict. A **semantics** miss.

Neither is a policy defect, and the two failure modes want different fixes.

`demote_late`'s shortfall is a property of the mechanism as specified, not of
detection: the detector finds 9-20 verified groups per question (median 14 on
`sh_6k`, and exactly one stale member each), so 14 latest-carriers get appended
and only one of them can actually be last. The oracle arm moved exactly one fact — the queried key's — to the very
end. Same detector, same edit primitive, a fifth of the effect.

---

## 12. T13b — the retrieval-path bridge, and the two things it changed

Registered in advance as `stage0_results/stage1_preregistration_v2.md` §10,
**explicitly not part of any confirmatory claim.** Calibration split only,
500 calls per subset, frozen :8003, page = the benchmark's own top-k chunks in
similarity-rank order, every arm performed through the shipped `page_edit`.

### 12.1 Why it had to exist

All T13 evidence was whole-context — one `Memory 1:` block. That was a
*documented* deviation, justified because retrieval is COMPLETE on the
calibration split (2 and 9 chunks ≤ `top_k` 10). Measured while scoping the
confirmatory run, the justification expires at `sh_64k`: 17 chunks, 10
retrieved, and a whole-context prompt there is **75,886 tokens against a 65,536
window** — infeasible on the frozen substrate, and not the deployed setting
either. So the whole-context→retrieval gap had to be measured, not assumed.

### 12.2 Results

| | `sh_6k` retrieval | (`sh_6k` whole-ctx) | `sh_32k` retrieval | (`sh_32k` whole-ctx) |
| --- | --- | --- | --- | --- |
| native overall | 0.280 | 0.290 | **0.480** | 0.420 |
| native conflicted | 2/74 | 4/74 | **13/65** | 7/65 |
| **suppress** conflicted | **68/74 (91.9%)** | 66/74 | **52/65 (80.0%)** | 51/65 |
| suppress net (b/c) | **+66** (0/66) | +61 (1/62) | **+38** (2/40) | +44 (1/45) |
| suppress exact p | 2.7e-20 | 1.4e-17 | 4.1e-10 | 1.3e-12 |
| demote_late net | +3 (n.s.) | +3 (n.s.) | **+10, p=0.031** | +9, p=0.012 |
| **anti net** | **−1** | −4 | **−6** | **+6** |
| token Δ (suppress) | −3.48% | −3.48% | −0.63% | −0.63% |
| A/A floor | 0/0 | 0/0 | 0/0 | 0/0 |
| page-edit mismatches | 0 | 0 | 0 | 0 |

**The mechanism survives the harness change.** Absolute conflicted accuracy is
essentially identical or better (68 vs 66, 52 vs 51). The *net* falls at
`sh_32k` (+38 vs +44) for a reason that is visible rather than mysterious: the
rank-ordered multi-block page is itself a **better baseline** there — native
conflicted 13/65 against 7/65 — so there is less headroom to recover, not less
mechanism.

### 12.3 What it changed — the `anti` control stops being inconsistent

§11.4b had to report `anti` as sign-inconsistent: −4 on `sh_6k` but **+6** on
`sh_32k` whole-context. In the retrieval harness it is **−1 and −6** — harmful
on both subsets, matching the oracle's whole-context `anti` (−3, −4) and
`sh_6k`'s sign.

So the sign flip was an artifact of the whole-context layout: in one
2,310-fact block, stacking ~13 newest facts at *either* edge helps, because both
edges of a very long single block are privileged. Give the model the page it
actually gets — nine `Memory i:` blocks — and the direction is consistent:
newest-last helps (`demote_late` +10, p=0.031), newest-first hurts (−6). **The
placement hypothesis is better supported in the deployed layout than in the one
it was discovered in**, and pre-registration v2 §8.6 should be read with that.

### 12.4 What it changed — the protective criterion fails here, and that matters

`detector_suppress` harmed exactly 2 questions on `sh_32k` retrieval:

| # | stratum | native → arm | gold fact deleted? | §5b class |
| --- | --- | --- | --- | --- |
| 9 | conflicted | `Rome` → `Watertown` | **yes** (serial 1291) | `information_loss` |
| 14 | **unique** | `Beirut` → *"The provided knowledge pool does not contain any fact about"* | **no** | `information_loss` |

Question 9 is the *predicted* gold-not-latest mode (prereg §6) and is fine.

Question 14 is not, and it is the important one. The gold fact was **still on
the page** — nothing about it was deleted — and the model nonetheless refused to
answer. Under the pre-registered §5b rule (`malformed_generation` requires gold
present AND `SequenceMatcher ≥ 0.8`; ratio here is 0.12) it classifies as
**`information_loss`**, and §5b says a single unique-stratum `information_loss`
**voids the protective claim whatever the net is**.

This is the calibration split and an exploratory arm, so nothing is voided. But
read literally and applied to `sh_64k`, the registered protective criterion
would have failed on evidence like this. Reported as classified, not
reclassified after the fact. It identifies a third harm mode that neither the
whole-context runs nor the oracle probe exposed — **refusal after edit**: the
edit changes the page enough that the model declines, without any needed fact
having been removed. It deserves its own name and its own count before the
confirmatory shot is spent.

### 12.5 A real bug this arm caught

The first retrieval run refused to send anything: the integrity check found that
`page_edit` appended moved facts *after* a chunk's trailing bytes, and the
benchmark's chunker ends `sh_6k` chunk 0 with `"...Leeds. 307."` — fact 307's
serial, whose text opens chunk 1. Appending after it made the inline parser read
the entire appended sentence as **the text of fact 307**, so the moved fact
silently lost its own serial — the supersession key the whole mechanism turns
on. Observed directly: `(5, "Pedro Pierluisi worked in...")` became
`(307, "5. Pedro Pierluisi worked in...")`.

`_content_tail` now treats a trailing dangling serial exactly like trailing
whitespace: nothing may be appended after either. This affected `DEMOTE_LATE` on
any page whose last chunk ends that way; `SUPPRESS` and every committed
whole-context result are untouched, because a single-block context has no
dangling tail.

---

## 13. T13c — the `sh_64k` confirmatory result

Pre-registered at `stage0_results/stage1_preregistration_v2.md` (commit
`b0ed608`, 2026-08-15T19:52:24+03:00, plus Amendments 1–4). Fired
**22:00:02**, finished **23:07:19**, at HEAD `3a2ebb9`. 500 calls, ~19.8M prompt
tokens, frozen `:8003`, retrieval-path harness, page read from the benchmark's
own vectors. Artifact:
`stage0_results/stage1/detector_gap_confirmatory_sh64k.json`.

### 13.0 A note on the suite counts quoted in T13 commit messages

The working tree is shared with a concurrently-committing agent (the T10 /
CrossEp line), so the **global** `pytest hnav/tests/ -q` total moves for reasons
that have nothing to do with T13 — it passed 465, 479 and 485 within one hour
while T13 added a single test. Two corrections follow from that:

* the banner commit (`76f8fb2`) quotes "466 passed / 1 skipped"; the observed
  total at that moment was **479 passed / 2 skipped**. The number was written
  from the expected count rather than the printed one. Corrected here.
* the stable figure for T13's own work is its own files:
  **`test_detector_gap.py` 86, `test_read_policy_facts.py` 42**. Global totals
  in T13 commit messages should be read as "the suite was green when this
  landed", not as an attribution of test count to T13.

### 13.1 Result

| arm | overall | unique | conflicted | b/c | net | exact p | tok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.450 | 28/34 | 17/66 (25.8%) | — | — | — | 0 |
| native_repeat | 0.450 | 28/34 | 17/66 | 0/0 | 0 | 1.0 | 0 |
| **detector_suppress** | **0.640** | 27/34 | **37/66 (56.1%)** | **0/20** | **+20** | **1.907e-06** | **−0.31%** |
| detector_demote_late | 0.480 | 28/34 | 20/66 (30.3%) | 2/5 | +3 | 0.4531 | 0.00% |
| detector_anti | 0.430 | 28/34 | 15/66 (22.7%) | 3/1 | −2 | 0.625 | 0.00% |

`b/c` is the **conflicted-stratum** McNemar; `detector_suppress`'s overall cells
are b=1 / c=20, net +19, p=2.098e-05.

**Primary criterion: MET.** Conflicted net +20 ≥ +10, exact p = 1.9e-06 < 0.01,
token delta −0.31% ≤ 0. Conflicted **b = 0** — not one conflicted question was
harmed.

**Protective claim: VOID, by one question.** Unique stratum b=1 / c=0. Question
77, `refusal_after_edit`: native `"John Milton"` → suppressed arm *"The provided
knowledge pool does not contain any information about"*. Not borderline —
`SequenceMatcher` ratio **0.182** against the 0.8 threshold, and the queried
key's target serial 2558 was **not** in the suppressed set, so the gold
information was demonstrably still on the page. Net −1 satisfies §5a; §5b voids
the protective claim because the class is not `malformed_generation`.

### 13.2 The registered conclusion, applied verbatim

> **Effective but not yet safe.** Fact-level suppression of detector-verified
> stale memory raises conflicted-stratum accuracy from 17/66 to 37/66 (+20 net
> discordant pairs, exact p = 1.9e-06) at a measured cost of one non-conflicted
> question via refusal-after-edit, and **may not be recommended for deployment
> on traffic containing non-conflicted queries** until that mechanism is
> understood and eliminated.

Per Amendment 4 (R2), condition 5 voids the **protective claim only**: the run
stands, the accuracy result stands, and the shot **is** spent.

### 13.3 Void conditions, by number

Stamped in the artifact as a `void_conditions` object rather than left to be
reassembled from scattered counters.

| # | voids | status | observed |
| --- | --- | --- | --- |
| 1 page-edit mismatch | run | **pass** | 0 |
| 2 native in band | run | **pass** | overall 0.450 in [0.30, 0.50]; unique native 28/34 = 0.824 ≥ 0.80 |
| 3 A/A floor zero | run | **pass** | b=0, c=0 |
| 4 no harmful suppression | run | **pass** | 0 harmful of **735** suppressed; all 735 genuinely superseded |
| **5 protected stratum** | **protective claim only** | **FAIL** | q77, `refusal_after_edit` |
| 6/7 page source | run | **pass** | `benchmark` |
| 8 guards + positive control | run | **pass** | containment 0, page_edit_errors 0, control OK |

**Verdict: run NOT void, shot spent, protective claim void.**

**Both condition-2 margins are thin and are stated here before a reader finds
them.** Native overall **0.450** sits at the top edge of the pre-fixed
[0.30, 0.50] band; unique native **28/34 = 0.824** clears the 0.80 floor by a
single question. Both are genuine passes against a band fixed in advance — and
the unique stratum is markedly noisier here than on calibration, where it was
26/26 and 35/35. The protective design presumes that stratum is near-noiseless,
and at `sh_64k` it is not.

### 13.4 The gold-cut prediction — missed, and the reason is the interesting part

Registered prediction (§6): **2** gold-cuts, from exposure 2/66 × conflicted
recall 0.957 = 1.91, with 0/1/2 the only arithmetically possible outcomes.

**Observed: 1 deletion, 0 accuracy losses.** The two exposed questions,
re-derived from the artifact:

| q | key | gold | latest | gold suppressed? | native | suppressed arm |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | author of *Red Storm Rising* | 7 "Tom Clancy" | 1219 "Torquato Tasso" | **no** — key never acted on | wrong ("Torquato Tasso") | wrong |
| 20 | continent of Great Britain | 2374 "Europe" | 2468 "Asia" | **yes** | **correct** ("Europe") | **correct** ("Europe") |

So the prediction missed low by one, and the single real gold deletion cost
nothing: with "Europe" removed and only "Asia" left on the page, the model
**still answered "Europe"**.

That is not reassurance, and must not be written as any. It weakens the implicit
`gold_cut ⇒ harm` assumption the registered prediction rested on, but the reason
it fails here is that the model answered from **world knowledge over the
context** — the H-parametric behaviour the T12 probe was designed to separate
from H-position. The answer was right for a reason the benchmark cannot see.
Read strictly: on this question the evaluator was satisfied despite the page
having been made *less* correct, so "0 accuracy losses from gold cuts" is
evidence about the evaluator, not evidence that gold cuts are safe.

> **Correction of record.** The commit message of `077ca68` and the first
> relayed summary stated these two questions **backwards** — they attributed the
> deletion to q18 and the "already wrong natively" explanation to q20. The table
> above is the re-derived truth. The artifact carries the same correction in its
> `corrections` field. An accompanying attribution to the 50-fact pool cap is
> **withdrawn**: per-question pools were not recorded in that artifact, so the
> claim was not verifiable. `detector_gap` now records `n_pool` and `pool` per
> question so that any future claim of that shape can be checked.

### 13.5 Harm classes, and the arms that were not the claim

| arm | harms | classes |
| --- | --- | --- |
| detector_suppress | 1 | `refusal_after_edit` 1 |
| detector_demote_late | 2 | `information_loss` 2 |
| detector_anti | 3 | `information_loss` 3 |

`gold_cut` = **0** in every arm. `refusal_after_edit` = 1 falls in Amendment 4's
registered **1–3 band**: it voids the protective claim and is reported as an
unresolved harm mechanism warranting a replicate — it is **not** the ≥ 4 scaling
effect.

`detector_demote_late` (+3, p=0.45) remains indistinguishable from noise, as on
both calibration subsets. `detector_anti` is harmful a third time (−2), so
across the deployed layout it is **−1, −6, −2**: once the page is the one the
model actually gets, the direction control never once pointed the wrong way.
That is the resolution of the §11.4b inconsistency, now confirmed on held-out
data.

### 13.6 The effect shrank, as registered

+62 / +66 conflicted net on `sh_6k`, +44 / +38 on `sh_32k`, **+20** here.
Pre-registration §8.2 registered that the `sh_64k` effect would be **smaller**
than `sh_32k`'s and forbade presenting the calibration nets as a forecast; §4's
threshold was set at +10 for exactly this reason. Suppression volume fell in
step: 1,416 and 1,257 facts on the calibration subsets against **735** here.

**No oracle-ceiling ratio exists for `sh_64k` and none may be quoted for it.**
The oracle probe is whole-context, and whole-context does not fit the window at
this scale (§0), so no oracle arm was ever run — the artifact's
`detector_vs_oracle` is correctly empty. The 0.984 / 0.957 figures are
calibration-only and additionally cross-harness.

## 11. T13 — the long-sequence OOM, and why it was not the math path alone

**Symptom** (found by the Thrust-2 agent on the CrossEp M5 real-embedder run):
`embedder.encode([content])` OOM'd a 24 GB card on a SINGLE text, so batch
size was irrelevant by construction — batch 8 → 2 → 1 failed identically.

**Measured root cause — a GQA/fused-kernel interaction, not merely "SDPA chose
the math path".** Qwen3-Embedding-4B has 32 query heads and 8 KV heads.
Transformers' `sdpa_attention` calls `use_gqa_in_sdpa()`, which returns True
when the attention mask is `None` — the single-text case, where an all-ones
mask is optimized away — and then passes SDPA **un-expanded** K/V with
`enable_gqa=True`. PyTorch's fused kernels reject that for dense inputs:

    both fused kernels require query, key and value to have the same num_heads.
    Query.sizes(): [1, 32, 4885, 128], Key sizes(): [1, 8, 4885, 128]

so it falls back to MATH, which materializes the `[1, 32, N, N]` score matrix
(730 MiB per 8-head chunk at N=4885) on top of ~15.1 GiB of fp32 weights.

Two things this explains that the "math path" framing does not:
- **Forcing `EFFICIENT_ATTENTION` does not work** — it aborts with "No
  available kernel", because what is missing is *eligibility*, not priority.
  Measured directly.
- **FLASH is never an option** under the fp32 pin ("Expected query, key and
  value to all be of dtype: {Half, BFloat16}").

**The fix** (`expanded_gqa_attention`, `hnav/core/embedding.py`): make
`use_gqa_in_sdpa` return False for the duration of the forward, so
transformers takes its own `repeat_kv` path and materializes K/V at 32 heads.
The fused kernel then becomes eligible and PyTorch selects it automatically —
no backend forcing, hence no abort risk. Fails open (a transformers rename
degrades to "unpatched" and is *recorded* as `gqa_expansion_applied=False`,
never silently assumed). Applied in both embedders — `hnav/core/embedding.py`
and the standalone `m1_geometry_calibration.py`, which imports it rather than
re-implementing it so the two cannot drift.

Scientifically neutral by construction: `repeat_kv` broadcasts exactly the K/V
that `enable_gqa=True` would have broadcast internally. fp32 pin, model,
`max_length`, and the `L8192` cache namespace are all unchanged.

### Equivalence proof (required before commit; box, RTX 4090, fp32, torch 2.13)

**A. Against the committed fp32 cache** — re-encoded texts already present in
the legacy `Qwen_Qwen3-Embedding-4B|float32` namespace, at that namespace's
`max_length=512`:

| sample | n | token range | max\|1−cos\| | max\|Δcomponent\| |
|---|---|---|---|---|
| facts (short) | 40 | 8–16 | 4.768e-07 | 2.980e-07 |
| chunks (long text) | 4 | 2455–5008 | 2.384e-07 | 1.639e-07 |

**Stated limitation:** those cached vectors were themselves computed at
`max_length=512`, so row 2 is a long *text* truncated to 512 — it does not
exercise a long *sequence* through attention. No cached vector was ever
computed above 512 tokens, so the cache cannot answer that question.

**B. Patched vs UNPATCHED at full sequence length** (`max_length=8192`), which
is the regime the re-fit will run in — this supplies what A cannot:

| tokens | patched | unpatched | max\|1−cos\| | max\|Δcomponent\| |
|---|---|---|---|---|
| 58 | OK 15.12 GiB | OK 15.12 GiB | 0.000e+00 | 9.313e-08 |
| 2455 | OK 16.14 GiB | OK 17.63 GiB | **0.000e+00** | 1.043e-07 |
| 4885–5008 (9 chunks) | OK 17.17–17.21 GiB | **OOM** | — | — |

Where both paths run the cosines are **bit-identical** and components agree to
fp32 epsilon (~1.2e-07). Where they differ, the old path simply does not run:
**9 of the 11 real calibration chunks OOM unpatched.** The patched path also
uses *less* memory where both succeed (16.14 vs 17.63 GiB at 2455 tokens).

Both well under the pre-registered 1e-5 stop threshold and at the ~1e-6 level
expected. Backend selection is deterministic here: forcing
`EFFICIENT_ATTENTION` on top of the patch produced bit-identical vectors
(max|Δ| = 0.0) and an identical peak, i.e. the automatic choice is already the
efficient kernel once eligibility is satisfied.

### Consequence for the queued re-fit

The re-fit (`hnav/deploy/REFIT_RUNBOOK.md`) would have OOM'd on essentially
every MAB chunk at `L8192`: the calibration chunks are 4,885–5,008 Qwen tokens,
squarely in the failing regime. It is now unblocked, on the same card, with the
fp32 pin and the `L8192` namespace intact.

### Durable CrossEp measurement (from the Thrust-2 agent, no GPU needed)

CrossEp texts: p50 1,045 / p90 1,130 / p99 1,411 / max 5,076 tokens.
**0.00% exceed 8192**, so the `L8192` pin never truncates CrossEp — but
**92.55% exceed 512**, so the pre-T12 truncation was cutting nearly every
CrossEp text. The T12 truncation fix therefore mattered *more* on CrossEp than
on MAB, and that belongs in the write-up.

### 11b. T13b — token-budget batching (found by the re-fit, fixed, verified)

The re-fit's m2 stage OOM'd even with §11 in place, and in a different place:
the **MLP**, not attention. `emb.encode(chunks)` handed 9 real benchmark chunks
of ~5,000 tokens to one padded forward (~45k tokens), whose fp32
`[tokens, 9728]` activations (~1.75 GiB each) exhaust what remains after the
15.1 GiB of weights. §11 made attention cheap; it did not make a 45k-token
batch cheap.

Fixed batch *count* is simply the wrong unit when texts differ in length by two
orders of magnitude (a fact is ~15 tokens, a chunk ~5,000), because padding
makes a batch cost `count × longest`. `HFEmbedder._batches` now bounds that
product by `DEFAULT_MAX_BATCH_TOKENS = 8192`
(`HNAV_EMBED_MAX_BATCH_TOKENS` / `cfg.embed_max_batch_tokens`) as well as by
the count cap: short facts still batch freely, long chunks go one at a time
(5,008 ≤ 8192 < 2 × 5,008).

**Vector-neutral, verified rather than argued.** Attention and mean-pooling are
both masked, so grouping cannot change a text's own vector; the cache namespace
is therefore unchanged. Measured against L8192 vectors that `m1` had already
cached under a *different* grouping (its standalone embedder, fixed count 8):

| sample | n | grouping then → now | max\|1−cos\| | max\|Δcomponent\| |
|---|---|---|---|---|
| facts | 64 | 8×8 → 8×8 | 5.960e-07 | 4.470e-07 |

and the 11 calibration chunks (58–5,008 tokens) now encode in 11 single-text
batches, all finite, all unit-norm. fp32 epsilon, two orders under the 1e-5
stop line.

**Note for anyone reading the cache:** two implementations write the same
namespace — `hnav/core/embedding.py` and the standalone
`m1_geometry_calibration.py` — and they group differently. Vectors agree to
~5e-07, so which one computed a given entry first is not recoverable from the
file and does not matter at this tolerance. It is recorded here rather than
discovered later.
