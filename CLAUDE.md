# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Two things layered on top of each other:

1. **EvoMemBench** (upstream) — a benchmark suite evaluating agent memory across a 2×2 taxonomy
   (in-episode/cross-episode × knowledge/execution). Six task suites, each a near-independent
   vendored codebase with its own `requirements.txt`, README and conda environment. Memory backends
   (`mem0`, `A-mem`, `MemOS`, `MemoryOS`, `MemoBrain`, `memagent`) are vendored under
   `EvoMemBench-Memory-Systems/` and installed editable. See `README.md` for the dataset/path table.

2. **`hnav/`** — an H-Nav Stage-0 measurement campaign built *on* the benchmark, followed by the
   conflict-label audit and the gold conflict dataset (current work, branch
   `claude/hnav-presentation-evidence`). This is where nearly all active work happens.
   H-Nav is a governance layer for evolving vector memory: it inspects candidate memory writes and
   retrievals using embedding geometry and retrieval-side signals.

**`HNAV_AGENT_BRIEF.md` is the authoritative spec.** Read it before touching `hnav/`. It is
self-contained and its rules are load-bearing — violating one invalidates the research, not just the
code. `hnav/BUILD_NOTES.md` records what was built and what is deliberately untested;
`hnav/NEXT_STEPS.md` is the operational checklist and **overrides `hnav/prompts/PROMPT_B_remote_run.md`
wherever they disagree**.

## Thesis framing — binding (user decisions 2026-09-17)

**`THESIS_CLAIMS.md` is the binding statement of what the thesis claims.** Read it before
touching any thesis-facing document. **`MANUSCRIPT_DRAFT.md`** is the journal-style draft written
from it (2026-09-17; Related Work left blank by design, title provisional). The decisions it records:

- **The thesis method is H-Nav-GEO** (`hnav/geometry_filter/geo_artifact.py`, arm
  `pipelines/hnav_geo/`): whitened cosine + slot probe on the edit direction, verified by a small
  NLI cross-encoder, then fact-level deletion of superseded versions. No fact parser and no LLM call
  at inference; the suppression plan is byte-identical across answering models.
- **A full trial run was performed** — five answering models × three context sizes × five arms,
  ~19,500 completions, one shot per cell, A/A floor 0 everywhere — and **the achievement to record
  is that geometry + NLI surpasses the native baseline in all 15 model × subset cells** (significant
  at 5 % in 11 of 15) while shortening the prompt. That is the headline; write from the strengths.
- **The thesis comparison is native baseline vs H-Nav-GEO only.** The schema-keyed arms
  (`hnav_raw`, `hnav_idonly`, `hnav_abtt`) stay in the repository as reference measurements but are
  outside the thesis comparison; `THESIS_CLAIMS.md` §10 gives the reason (a template parser has no
  counterpart in a real store) and uses them only as an oracle reference for that design decision.
- No "harm-free" claim is made and harm is not a headline; the unique stratum is reported as the
  do-no-harm check. One quantified limitation line stays (`THESIS_CLAIMS.md` §12).
- Generality is claimed across all five measured models, with the embedder-agnostic-by-construction
  argument and the real-agent-memory hypothesis stated as hypothesis; sh_262k is excluded with its
  reason written.

`LATEST_RESULTS_REPORT.md` §12 and the `hnav_idonly`-centred story in `TEZ_HIKAYESI.md` are
superseded on framing and kept as dated records. Committed artifacts, void records and measured
numbers are never edited to fit the framing; the framing selects and orders them.

## Hard invariants (from the brief §1)

These are not style preferences. Each one exists because breaking it silently produces a number that
looks valid and isn't.

- **No leakage into any online path.** Gold answers, benchmark `questions`/`answers` keys, future
  facts, and evaluator output may appear **only** in the offline tier — `hnav/labeling/`,
  `hnav/stage0/` and `hnav/stage1/` (the last holds `calibrate_read_policy.py` and
  `stale_suppression_probe.py`, both offline oracles). Nothing under `hnav/core/` or
  `hnav/adapters/` may reference them, and nothing online may import the offline tier. Enforced by
  an AST scan in `hnav/tests/test_leakage_audit.py` (`ONLINE_DIRS = hnav/core, hnav/adapters`).
  The questions and the fact contexts live in the *same JSON file*, so this is the easiest mistake
  available.
- **`hnav/core/` imports nothing from a benchmark.** Adapters own all benchmark-specific knowledge
  (chunk parsing, serial numbers, prompt formats).
- **Write-time vs read-time visibility differ.** Use `latest_before(key, serial)` on the write path;
  `latest(key)` is read-only. `hnav/labeling/conflict_index.py` deliberately ships two classes for
  this reason.
- **Shadow mode is byte-identical to off.** Hooked functions return the caller's own object; no store
  mutation, no extra LLM calls.
- **`HNAV_MODE` defaults to `off`.** A stray import must never move a benchmark number.
  `HNAV_MODE=live` is refused during Stage 0 (`config.require_not_live()`).
- **Calibration split is `sh_6k` + `sh_32k`.** Never tune on `sh_64k` / `sh_262k`. `m3_headroom.py`
  refuses to fit thresholds without a calibration subset; `m4_marginal_diff_test.py` refuses a
  non-calibration split outright.
- **`H_raw` (softmax over raw `cosine × 100` scores) is logged but never feeds a decision.** Use the
  z-scored `H_z`. Asserted by `test_no_raw_entropy_in_policy.py`.
- **Never reuse the prior BFCL port's numeric thresholds.** Different scales, different base rates.
- **Stop at every `[GATE]`** in the brief and report to a human. Do not proceed on your own judgment.
- **`hnav/core/write_policy.py` must not exist — ever.** The T8 verdict (`KAPI_KARARI.md`: write
  headroom measured at ~0, NO_GO) made this permanent. `read_policy.py` is PERMITTED post-T8
  (user decision 2026-08-15, `STAGE1_PLAN.md` §0 — read-path rerank only); it and
  `read_gate.py` are still barred from `H_raw` by the AST scan in
  `test_no_raw_entropy_in_policy.py`, which also fails on any other `*policy*` module.
- **Do not add H-Nav to `In-Episode-Execution/` or `Cross-Episode-Execution/`.** Both are BFCL — the
  substrate where the prior attempt returned null.
- **Do not rewrite `hnav/labeling/conflict_analysis.py::parse`.** Validated at 99.5%+ coverage;
  import it.

## Commands

Everything below runs from the repo root (`EvoMemBench/`).

```bash
# Full test suite — no torch, no GPU, no network required. ~550 tests.
pytest hnav/tests/ -q

# One file / one test
pytest hnav/tests/test_geometry.py -q
pytest hnav/tests/test_geometry.py::test_qr_residual_is_zero_inside_the_span_and_one_outside -q

# T0 — reproduce the committed measurements before writing anything (stdlib only)
python3 hnav/labeling/conflict_analysis.py     # expect sh_262k: 11,037 keys / 7,197 conflicted (65.2%)
python3 hnav/labeling/gold_rule.py             # expect 77% of sh_262k questions conflicted, 73/77 gold-is-LATEST
python3 hnav/labeling/marginal_diff.py         # lexical proxy only — no threshold may be derived from it

# Gold conflict dataset — rebuild is deterministic from committed inputs; the builder
# asserts the frozen tier counts (core 2,388 / fork 282 / rejected 12 / discovered 105 /
# negative 51,782) and fails loudly on drift. Never re-baseline those numbers.
python3 hnav/labeling/build_gold_conflict_dataset.py
```

GPU-box setup and the Stage-0 pipeline, in order (see `hnav/BUILD_NOTES.md` §6):

```bash
bash hnav/deploy/setup_remote.sh          # .venv-hnav, torch+cu124, nltk/tiktoken, ~8GB weights
cp hnav/deploy/.env.template .env         # then edit
source .venv-hnav/bin/activate
python hnav/deploy/check_env.py           # GATE — must exit 0
pytest hnav/tests/ -q                     # must still be green with torch installed

bash hnav/deploy/run_t1.sh                # T1/M1 — the S3 kill switch. Exit 2 = gate fired, STOP.
python hnav/stage0/m1b_grouping_ablation.py       # T2, reuses T1's embedding cache
python hnav/stage0/m2_retrieval_calibration.py    # T5
python hnav/stage0/m3_headroom.py                 # T6, needs the LLM server; run under tmux
python hnav/stage0/m4_marginal_diff_test.py       # T7, calibration split only
python hnav/stage0/report.py --strict             # T8 — HARD STOP
```

Fast local iteration without a GPU or an LLM: `--smoke-embedder` swaps in `HashEmbedder`,
`--stub-llm` swaps in a deterministic answer stub. Both write to separate `*_SMOKE.json` files so
smoke numbers can never be mistaken for data. `--subsets sh_6k --max-pairs 50` gives a ~2-minute run.

## Architecture

### `hnav/` layering

The import direction is strict and is the main thing to preserve:

```
hnav/core/       benchmark-agnostic. types, embedding, replica, audit,
                 geometry, diff_geometry, retrieval_signals.
                 Imports nothing from a benchmark and never sees gold answers.
hnav/adapters/   the only place that knows a benchmark exists.
                 mab_adapter (primary arena), clbench_adapter (secondary).
hnav/labeling/   offline. May read questions/answers. conflict_analysis (the
                 validated fact parser), conflict_index, labels, counterfactual,
                 gold_rule, marginal_diff, plus the audit chain: export_conflict_pairs,
                 export_audit_candidates, audit_runner, build_gold_conflict_dataset.
hnav/stage0/     the measurements M1/M1b/M2/M3/M4 + report.py. May read gold.
hnav/config.py   every setting; read from repo-root .env, os.environ wins.
hnav/deploy/     .env.template, check_env.py, setup_remote.sh, run_t1.sh
```

`hnav/tests/test_no_torch_at_import.py` imports every module under `hnav/` in a fresh subprocess and
fails if `torch`, `transformers`, `faiss`, `openai`, `langchain` or `vllm` lands in `sys.modules`.
Keep heavy imports lazy.

### The two arenas and the four benchmark hooks

**Primary arena:** `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/`, dataset
`Conflict_Resolution`, single-hop subsets `factconsolidation_sh_{6k,32k,64k,262k}` — 400 questions.
Each context is a numbered fact list where a later serial supersedes an earlier one with the same
subject+relation; gold is the highest serial. Evaluator is `substring_exact_match` — deterministic,
offline, free. **Secondary arena:** `Cross-Episode-Knowledge/CROSSEP-KNOW/`.

Four benchmark files carry guarded edits, all no-ops when `HNAV_MODE=off`. Grep them for `hnav` /
`HNAV` before changing anything nearby:

| File | Hook |
| --- | --- |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/embedding_retriever.py` | `TextRetriever.retrieve` — emits the full pre-truncation ranking, returns the same `top_k` as before |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/agent.py` | `send_message` — adapter callbacks at entry/exit |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/registry.py` | `build_memory` — wraps the backend in `HNavMemoryWrapper` |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/infer_context_memory.py` | additive `"hnav": {...}` result field |

These four import torch at module level (they are benchmark code) and are therefore checked by
`ast.parse` and marker assertions rather than by import.

### The conflict-label audit chain (`stage0_results/conflict_pairs/`, all committed)

Provenance is a strict pipeline; each stage reads only the previous stage's committed file:

```
export_conflict_pairs.py        conflict_pairs.json — the 2,682 parser-tagged pairs
export_audit_candidates.py      audit_candidates_cos080.jsonl.gz — all 87,102 pairs at
                                cos ≥ 0.80 under the campaign embeddings
                                (Qwen3-Embedding-4B float32 L8192)
audit_runner.py                 audit_results_gpt5mini.jsonl.gz — GPT-5-mini verdicts for
                                54,569 pairs ($20 budget stop; tagged pairs 100% covered)
build_gold_conflict_dataset.py  gold_conflict_dataset.jsonl.gz + summaries — the gold set
```

Rules baked into the gold dataset (user decisions 2026-08-26 — do not relitigate):

- **Dual labels, `gold_update` is THE default.** `gold_update` follows the benchmark convention
  (later serial supersedes the same key); `gold_strict` means values logically cannot coexist and
  always implies `gold_update`. The `update_only_fork` tier (282 pairs) is gold under update
  semantics only and carries `disputed_by_judge` — it is a definitional fork, not a parser error.
- **Triage keys on the judge's recorded alignment flags, never on `reason_code`.** The reason
  taxonomy is noisy (`relation_paraphrase` was used for *value* paraphrases on pairs whose relation
  template is identical by construction). Fork = alignment held but values judged compatible;
  `rejected` (12 pairs) = the judge refuted the slot alignment itself.
- **`discovered_unverified` (105 single-judge positives) stays quarantined** — never gold, never in
  the negative pool, until independently adjudicated. Known judge errors live there (e.g. the
  Galileo two-books pair).
- **The eval set is balanced 1:1 per subset and cosine-matched** (0.01 bins, seed 20260824,
  fallback distance recorded per record). Matching cannot fully remove the cosine signal — verified
  non-conflicts above cos 0.95 barely exist — so the summary reports `cosine_only_auc`
  (0.96 / 0.91 / 0.89 for sh_6k / sh_32k / sh_64k). **Any geometry filter must be scored against
  that baseline**, or inside the ~0.87–0.97 overlap band where cosine is uninformative; a headline
  AUC alone is meaningless on this set.
- **Selection frame is cos ≥ 0.80; the 32,533 unaudited candidates are excluded entirely.** The
  dataset says nothing about conflicts below that similarity, and prevalence claims from the
  negative pool must weight for the 34.5%-audited bulk tail.
- Records carry `split` (sh_6k + sh_32k calibration, sh_64k confirmatory) — fit filter thresholds
  on calibration only, same as everywhere else in H-Nav.
- The repo-root `*.gz` ignore rule swallows new archives here: any new `.gz` needs a `!name.gz`
  line in `stage0_results/conflict_pairs/.gitignore` or it silently never lands in a commit.

`hnav/tests/test_gold_conflict_dataset.py` re-derives tiers, labels, balance, bin matching and the
AUC from the raw audit files — deliberately not from the builder's own expectation table.

### Conventions that bite

- **`RetrievalView.scores` is always "higher is better".** `NumpyCosineReplica` emits the native
  `(bank @ q) * 100.0` unchanged; LangChain FAISS returns squared L2, so the hook passes
  `score_kind="l2sq"` and `to_similarity` converts once via `cos = 1 − d²/2`. One conversion, one
  place.
- **`FACT_RE` alone matches zero facts on a real memorize chunk.** The benchmark hands H-Nav the
  output of `chunk_text_into_sentences`, which joins sentences with spaces, so a line-anchored regex
  has nothing to anchor to. `explode_facts` tries `FACT_RE` first (exact on raw context) then falls
  back to `FACT_RE_INLINE`. `test_chunking_and_facts.py` asserts the line-anchored form *fails*, so
  the fallback cannot be quietly deleted as redundant.
- **`StoreView.with_provisional(cand)` must stay non-mutating.** Returning a new view is what makes
  `dH` and churn computable before a write commits.
- **Report stratified, never pooled across subsets.** Store sizes span 455 → 18,332 facts.
- **New-model pipeline runs (`pipelines/`) use sh_6k + sh_32k + sh_64k — never sh_262k.**
  User decision 2026-08-24, applies identically to every model in the series: sh_262k's contexts
  would exceed smaller answering models' context windows, turning a memory comparison into a
  context-length comparison. Enforced by `ALLOWED_SUBSETS` in `pipelines/_shared/runner.py`;
  do not widen it per model.
- **Cluster CrossEp-Know by `context_id`.** ICC = 0.346, design effect 3.20, effective N ≈ 276 of 884.
- **`mh_*` (multi-hop) subsets are exploratory only** — question→fact mapping is not 1:1.
- **Check `"fallback_chunker": false` in M2 output.** `true` means nltk/punkt is missing and the
  chunking is not the benchmark's.

### Output and cache layout (both gitignored)

`hnav/_out/` holds `m{0,1,1b,2,3,4}_*.json`; `report.py` reads them off disk and writes
`STAGE0_REPORT.md`. `hnav/_cache/emb/` holds `sha256(model|dtype||text)`-keyed `.npy` embeddings —
T1 pays for ~26k of them once, every later task is free. **Do not delete it, and do not copy it
between machines** unless embedder model *and* dtype match exactly. Dtype is pinned once chosen;
drift changes cosines and moves every threshold.

Anything in `hnav/_out/` you want to keep must be committed deliberately.

## Testing philosophy here

Every numeric quantity is checked against a closed-form or independently computed answer, not
against "it ran": `ln m` for an orthonormal neighbourhood, QR residual 0 inside a span and 1 outside,
labels compared to an oracle reimplemented separately in the test, H2 statistics run on both
informative *and* null synthetic regimes. The leakage scanner is itself tested against three
deliberate violations — a scanner that cannot fail is decoration. Match this standard when adding
tests.

## Commit convention

Commit after each task with the task ID in the message (`T5: retrieval signals and M2 retrieval
calibration`); audit-chain work uses an `Audit:` prefix instead. Stage-0 landed on
`claude/evomembench-hnav-analysis-nfwl9z` (PR #1); current work lands on
`claude/hnav-presentation-evidence`.
