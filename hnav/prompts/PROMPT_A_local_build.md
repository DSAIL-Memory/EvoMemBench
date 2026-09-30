# PROMPT A — Local build (run this on the MacBook)

> Paste everything below the line into your coding agent, in the repo root,
> on branch `claude/evomembench-hnav-analysis-nfwl9z`.

---

You are implementing Stage 0 of an H-Nav port into EvoMemBench, for a master's thesis.

**Read `HNAV_AGENT_BRIEF.md` in this repo first. It is the authoritative spec.** This
prompt only tells you what is different about *this* phase.

## Your environment and its hard limits

You are on an **Intel MacBook Air 2017**. There is **no GPU, and `torch` cannot be
installed** — no macOS x86-64 wheels exist for current versions. Do not try. Do not
suggest workarounds. Do not add torch to any local test path.

You **do** have `numpy`, `scipy`, and `scikit-learn`. That is enough to build and test
every piece of logic that matters.

The finished code will be copied to a Linux box with 2× RTX 4090, where it must run
first try. **You cannot test against a GPU or a real model, so everything you write
must be verified another way** — see "How you must test" below.

## What already exists — do not rewrite these

| Path | Status |
| --- | --- |
| `hnav/labeling/conflict_analysis.py` | Validated fact parser, 99.5%+ coverage. `parse(text) -> (relation, subject, object) \| None`. **Import it. Do not write another parser.** |
| `hnav/labeling/gold_rule.py`, `marginal_diff.py` | Committed measurements. Leave alone. |
| `hnav/stage0/m1_geometry_calibration.py` | **T1, the Stage-0 gate. Already written and logic-tested. Do not modify** unless you find an actual bug, and say so loudly if you do. |
| `hnav/deploy/*` | Setup, pre-flight, launch scripts. Already written. |

## What you are building — T2 through T8

Implement exactly the tasks in `HNAV_AGENT_BRIEF.md` §3, **T2 through T8**. T0 and T1
are done.

Build them in this order, committing after each with the task ID in the message:

1. **T3** — `hnav/core/types.py`, `replica.py`, `audit.py` + `hnav/tests/test_replica_fidelity.py`
2. **T4** — `hnav/adapters/{mab_adapter,clbench_adapter}.py`, the four guarded benchmark
   edits, `hnav/tests/{test_shadow_neutrality,test_leakage_audit}.py`
3. **T5** — `hnav/core/retrieval_signals.py`, `hnav/stage0/m2_retrieval_calibration.py`
4. **T2** — `hnav/stage0/m1b_grouping_ablation.py` (needs T1's embedding cache, so it
   runs on the remote — but write it now)
5. **T6** — `hnav/labeling/{labels,conflict_index,counterfactual}.py`, `hnav/stage0/m3_headroom.py`
6. **T7** — `hnav/stage0/m4_marginal_diff_test.py`
7. **T8** — `hnav/stage0/report.py`
8. **Also**: `hnav/core/{geometry,diff_geometry}.py`, needed by T6/T7.

**Do NOT write `hnav/core/write_policy.py` or `read_policy.py`.** Those are live-
intervention code and are gated behind a human decision at T8. Writing them now would
be wrong.

## How you must test, given no GPU

This is the part that decides whether the remote run works first try.

1. **Every module must import cleanly with no torch installed.** Any torch/transformers
   import goes *inside* a function or method, never at module top level. Add a test that
   imports every module in `hnav/` and fails if `torch` appears in `sys.modules`.
2. **Test the numeric core with synthetic vectors.** `RetrieverReplica`, entropy,
   margins, `dH`, churn, QR residual, ABTT whitening — all are pure numpy. Build small
   fixtures with known answers (orthonormal bases, deliberate ties, duplicated rows) and
   assert exact values. Do not settle for "it runs".
3. **Test against the real data files, which ARE present locally.** `Conflict_Resolution.json`
   and `CL-bench_context_ge5.jsonl` are committed. Labeling, conflict indexing, and
   counterfactual set-construction can and must be tested on real data.
4. **Mock the embedder behind one interface.** Define an `EmbedderProtocol` with
   `encode(list[str]) -> np.ndarray`, and provide `HashEmbedder` — deterministic
   pseudo-random unit vectors seeded by text hash — for tests. The real GPU embedder is
   one implementation; tests never touch it.
5. **Mock the LLM endpoint.** Counterfactual code must accept an injected
   `answer_fn(prompt) -> str`. Tests pass a stub. No network in any test.
6. **`pytest hnav/tests/ -q` must pass locally, with no torch, no GPU, no network.**
   That is the acceptance bar for this phase.

## Configuration you must respect

Read config from the repo-root `.env` (template at `hnav/deploy/.env.template`).
Never hardcode. The values that matter:

- `HNAV_MODE` — `off` | `shadow` | `live`, **default `off`**
- `HNAV_EMBED_DEVICE=1` — GPU1. GPU0 hosts the LLM server; never touch it.
- `HNAV_EMBED_MODEL=Qwen/Qwen3-Embedding-4B`, `HNAV_EMBED_DTYPE=float32`
- `HNAV_LLM_BASE_URL=http://localhost:8000/v1`, `HNAV_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507`,
  `HNAV_LLM_TEMPERATURE=0`
- Cache to `HNAV_CACHE_DIR`, write outputs to `HNAV_OUT_DIR`. Both gitignored.

The answer model is a **local Qwen3-4B-Instruct-2507 under vLLM**. There is no external
API and no API key. Use the `openai` client pointed at the local base URL.

## Rules you cannot break

All seven "Hard rules" in `HNAV_AGENT_BRIEF.md` §1 apply. The two that are easiest to
violate by accident:

- **Gold answers live in the same JSON file as the fact contexts.** Any online signal or
  decision path that reads `questions` or `answers` invalidates the research. `test_leakage_audit`
  must catch it by AST scan, and that test must actually be able to fail — write a
  deliberate violation, confirm it trips, then remove it.
- **`latest_before(key, serial)` on the write path, `latest(key)` on the read path.**
  Two separate methods on two separate classes. At write time, facts with a higher
  serial have not been observed yet; using them is look-ahead.

## When you finish

- `pytest hnav/tests/ -q` green, no torch, no network.
- Every module documented with its task ID.
- A short `hnav/BUILD_NOTES.md`: what you built, what is tested locally, **what is
  necessarily untested until it reaches the GPU**, and any deviation from the brief with
  your reasoning.
- Commit and push to `claude/evomembench-hnav-analysis-nfwl9z`.

Then **stop**. Do not attempt to run T1–T8; they need the GPU machine.
