# H-Nav × EvoMemBench — Implementation Brief for a Coding Agent

**Read this file first. It is self-contained.** Background docs exist in this repo and are worth
reading for *why*, but every instruction needed to do the work is here.

- `EVOMEMBENCH_HNAV_REPO_ANALYSIS.md` — what EvoMemBench is, measured properties, hook mapping
- `EVOMEMBENCH_HNAV_IMPLEMENTATION_PLAN.md` — module design and full pseudocode
- `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` — preregistered measurements and the GO/NO_GO gate

**Scope of this brief:** tasks **T0 – T8** only. That is Stage 0: measurement and shadow-mode
instrumentation. It ends at a decision gate. **Do not implement write/read policies or run live
interventions.** Those are unlocked only after the T8 gate is evaluated by a human.

---

## 0. Context you need

**What H-Nav is.** A governance layer for evolving vector memory. It inspects each candidate
memory write and each retrieval using (a) embedding geometry — `sim_max`, QR residual novelty,
adaptive threshold `tau_t`, ABTT whitening — and (b) retrieval-side signals — rank-of-self,
top1−top2 margin, entropy, entropy deltas across a provisional insert, top-k churn. It then either
governs the write (`PASS` / `SUPPRESS` / `REWRITE` / `DEFER`) or repairs the read
(`RERANK` / `EXPAND` / `TRIM` / `ANNOTATE`).

**Why this benchmark.** A prior port to BFCL returned null: the target failure classes were only
~3.5% and ~0.7–0.9% of decisions. In EvoMemBench's `Conflict_Resolution` subset the equivalent
class is **65–77% of questions**. That base-rate difference is the entire reason for this work.

**Primary arena.** `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/`, dataset
`Conflict_Resolution`, single-hop subsets `factconsolidation_sh_{6k,32k,64k,262k}` = 400 questions.
Each context is a numbered fact list where a later fact supersedes an earlier one with the same
subject+relation; the gold answer is the **highest serial number**. Evaluator is
`substring_exact_match` — deterministic, offline, free.

**Secondary arena.** `Cross-Episode-Knowledge/CROSSEP-KNOW/` — 884 samples, 120 contexts, clean
`Memory.retrieve()` / `Memory.extract()` interface.

**Out of scope, do not touch:** `In-Episode-Execution/`, `Cross-Episode-Execution/`. The first two
are BFCL (the substrate that already failed); the others are not replayable.

**Already committed and working** (stdlib only, no setup):
```
hnav/labeling/conflict_analysis.py   # relation-template induction + (relation,subject) grouping
hnav/labeling/gold_rule.py           # verifies gold == latest serial
hnav/labeling/marginal_diff.py       # whole-blob vs diff separation, LEXICAL PROXY only
```
`conflict_analysis.parse(fact_text) -> (relation_key, subject, object) | None`, 99.5%+ coverage.
**Reuse it. Do not rewrite the parser.**

---

## 1. Hard rules — violating any of these invalidates the research

1. **No leakage into any online path.** Gold answers, benchmark `questions`/`answers` keys, future
   facts, and evaluator output must never reach code that computes a signal or a decision. They are
   permitted **only** in offline labeling/analysis modules under `hnav/labeling/` and `hnav/stage0/`.
2. **Write-time vs read-time visibility differ.** At write time only facts with a *smaller* serial
   have been observed — using the whole-store index is look-ahead. At read time the full store is
   legitimately visible. Enforce with two distinct APIs: `latest_before(key, serial)` (write) and
   `latest(key)` (read only).
3. **Shadow mode must be byte-identical to off.** Every hooked function returns exactly what it
   would have returned with H-Nav disabled. No store mutation, no extra LLM calls.
4. **Default off.** `HNAV_MODE` env var ∈ {`off`, `shadow`, `live`}, default `off`. A stray import
   must never change a benchmark number.
5. **Never reuse BFCL's numeric thresholds.** Score scales and base rates differ. Every threshold is
   fit on the calibration split (`sh_6k` + `sh_32k`) and frozen before touching the confirmatory
   split (`sh_64k` + `sh_262k`).
6. **`H_raw` (softmax over raw retrieval scores) is logged but must never feed a decision.** Native
   scores are `cosine × 100`; raw-score entropy is expected to be scale-degenerate. Use z-scored
   entropy `H_z` as primary.
7. **Stop at every `[GATE]`.** Report results and wait for a human. Do not proceed on your own
   judgment.

---

## 2. Environment

Only the primary arena is needed for T0–T7. Do **not** set up all six suites.

```bash
conda create -n hnav python=3.11 && conda activate hnav
pip install numpy scipy scikit-learn torch transformers sentence-transformers
# T3+ only:
pip install -r In-Episode-Knowledge/INEP-KNOW/requirements.txt
```

**Embedder — use a local model, not an API.** This gives determinism (the shipped configs use
`temperature` 0.7–1.0) and removes the DashScope key dependency.
- Preferred: `Qwen/Qwen3-Embedding-4B` — named by the benchmark's own config.
- Cheap fallback: `facebook/contriever` — natively supported at
  `methods/embedding_retriever.py:22`, ~110M params.

Record which embedder you used in every output file. **The embedder chosen in T1 must be the same
one used in all later tasks** — mixing them invalidates the calibration.

Cache embeddings to `hnav/_cache/emb/<sha256(text)>_<model>.npy` (gitignored). The first pass costs
~26k embeddings; every later task is then free.

---

## 3. Tasks

### T0 — Reproduce the committed measurements
**Goal:** confirm the environment and data are intact before writing anything.
```bash
python3 hnav/labeling/conflict_analysis.py
python3 hnav/labeling/gold_rule.py
python3 hnav/labeling/marginal_diff.py
```
**Accept when:** `sh_262k` reports 11,037 keys / 7,197 conflicted (65.2%) / all groups size 2; and
`gold_rule` reports 77% of `sh_262k` questions on a conflicted key with 73/77 gold-is-LATEST.
**If numbers differ:** stop and report. The data file changed.

---

### T1 — M1: geometry calibration with real embeddings  `[GATE]`
**This is the kill switch. Do it before writing any H-Nav module.**

The committed `marginal_diff.py` uses a char-3gram **lexical proxy** because no embedder was
available. It is not a substitute and no threshold may be derived from it. Replace it with real
embeddings.

**File:** `hnav/stage0/m1_geometry_calibration.py`

**Method:**
1. For each subset, use `conflict_analysis.parse` to build `(relation, subject) -> [(serial, text, object)]`.
2. Keep conflicted keys (>1 distinct object). Each has exactly 2 members: `old` = lower serial,
   `new` = higher serial.
3. Build a control set of **random non-conflicting fact pairs**, matched in count.
4. Embed and L2-normalize: `old_full`, `new_full`, `old_object`, `new_object`.
5. Compute per pair:
   - `whole_blob_sim = cos(old_full, new_full)`
   - `diff_sim       = cos(old_object, new_object)`
   - `qr_residual`   = novelty of `new_full` against the matrix of all facts with serial < new's
   - the same for control pairs
6. Optionally repeat with ABTT whitening fitted per subset. **Refuse to whiten when the store has
   < 200 facts** and log the fallback rate.

**Report:** per subset — mean/p10/p50/p90 for `whole_blob_sim`, `diff_sim`, `qr_residual`, whitened
and unwhitened; separation AUC (conflict pairs vs control pairs) on `whole_blob_sim`; the exact-
duplicate rate (expected ~0).

**GATE / STOP condition (`S3`):** if median `whole_blob_sim` for conflict pairs is **< 0.70**
unwhitened, the near-duplicate premise fails. **Stop. Report. Do not write the geometry modules.**
Only the read-side path would remain viable.

---

### T2 — M1b: regex-vs-geometry grouping ablation
Runs off T1's embeddings; no new compute. **This decides whether the eventual result is
attributable to H-Nav at all**, so it is not optional.

The benchmark hands you two shortcuts: facts are templated (so `parse()` groups them with a regex)
and each carries a serial number (so supersession is explicit). A critic will say any gain comes
from the metadata, not the geometry. Measure it directly.

**File:** `hnav/stage0/m1b_grouping_ablation.py`

Treat "which facts are competing versions of the same fact?" as a retrieval problem and compare two
groupers against the `parse()`-derived ground truth:
- **Regex grouper** — `parse()` → exact `(relation, subject)` match. This is the oracle.
- **Geometry grouper** — nearest neighbours of a fact above a similarity threshold, no parsing.

**Report:** precision / recall / F1 of the geometry grouper against the regex grouper, swept over
the threshold; the PR curve; and the F1 at the operating point where coverage matches.

**Interpretation to record in the output, verbatim:**
> Geometry that recovers the regex grouping *without parsing* is what licenses applying H-Nav to
> CrossEp-Know, where no templates and no serial numbers exist. High F1 → the detector is
> validated. Low F1 → any downstream gain is attributable to the metadata, not to geometry, and
> must be reported as such.

---

### T3 — Core types, retriever replica, M0 fidelity  `[GATE]`
**Files:** `hnav/core/types.py`, `hnav/core/replica.py`, `hnav/core/audit.py`,
`hnav/tests/test_replica_fidelity.py`

Types (`types.py`): `MemoryRecord`, `Candidate`, `StoreView`, `RetrievalView`, `Decision` — full
field lists in `EVOMEMBENCH_HNAV_IMPLEMENTATION_PLAN.md` §2.1. `StoreView.with_provisional(cand)`
must be **non-mutating** (returns a new view) — this is what makes `dH`/churn computable before a
write commits.

Replicas:
- `NumpyCosineReplica` — targets `Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/qwen3_embedding_memory.py:218`.
  The native computation is exactly:
  ```python
  scores = (bank_matrix @ query_vec) * 100.0
  order  = np.argsort(scores)[::-1]
  ```
  Bank embeddings are persisted at `<memory_dir>/embeddings.jsonl`, so this replica is bit-exact.
  Note `np.argsort` is not stable — document tie behaviour rather than silently differing.
- `FaissFlatReplica` — targets `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/embedding_retriever.py:185`.
  LangChain FAISS; use `similarity_search_with_score_by_vector`.

Both expose:
```python
def rank(self, store: StoreView, query: str) -> RetrievalView          # FULL pre-truncation ranking
def simulate_insert(self, store, cand, probes) -> dict[str, RetrievalView]   # {"before":…, "after":…}
```

**Accept when (`M0`):** ≥1,000 sampled real `(store, query)` pairs per arena show ≥99.9% exact
top-k identity vs the native retriever. `NumpyCosineReplica` must be 100% modulo documented ties.
**If below threshold:** `rank_self`, `margin`, `dH_self`, `dH_neighbor`, `churn` are all invalid.
Stop, report the maximum achievable fidelity, and state explicitly which signals die.

---

### T4 — Benchmark hooks + shadow neutrality  `[GATE]`
Four edits, all no-ops when `HNAV_MODE=off`.

| File | Location | Change |
| --- | --- | --- |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/embedding_retriever.py` | `185` `TextRetriever.retrieve` | Obtain the **full** ranking + scores via `similarity_search_with_score_by_vector`; emit to the adapter; **return the same `top_k` page-contents as before** |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/agent.py` | `967` `send_message` | Adapter callbacks at entry/exit, gated on `HNAV_MODE` |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/registry.py` | `build_memory` | When enabled, wrap the backend in `HNavMemoryWrapper` (delegates verbatim in shadow) |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/infer_context_memory.py` | `196` result dict | Add an additive `"hnav": {...}` field |

**Files:** `hnav/adapters/mab_adapter.py`, `hnav/adapters/clbench_adapter.py`,
`hnav/tests/test_shadow_neutrality.py`, `hnav/tests/test_leakage_audit.py`

Adapters own all benchmark-specific knowledge (chunk parsing, serial numbers, prompt formats).
`hnav/core/` must import nothing from the benchmark. In `mab_adapter`, explode each 4096-token
memorize chunk into per-fact candidates with `FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)`
and set `Candidate.version` = the fact's serial number.

**Accept when:**
- `HNAV_MODE=off` vs `shadow` on `sh_6k` and `sh_32k` at `temperature=0` produce **byte-identical**
  model outputs and identical per-question `substring_exact_match`. Any difference is a bug — at
  `temperature=0` with a deterministic evaluator there is no legitimate source of variation.
- Token counts (`stats.tokens.inference`, `extract_llm`) unchanged vs baseline.
- `test_leakage_audit` passes: an AST scan finds no reference to `answers`, `gold`, or `rubrics`,
  and no import of `hnav.labeling.counterfactual`, anywhere under `hnav/core/` or `hnav/adapters/`.

---

### T5 — M2: retrieval calibration
**File:** `hnav/stage0/m2_retrieval_calibration.py`, `hnav/core/retrieval_signals.py`

Over all 400 primary-arena questions using the full pre-truncation ranking, compute and report
distributions for: score scale and tie frequency; `top1`, `top2`, `margin`, `nmargin`; `H_raw`
(logged only), `H_z`, `H_vn` (von Neumann entropy over the top-m Gram matrix, m≈50); effective
neighbourhood size; dispersion; and `dH_self`, `dH_neighbor`, `churn@k` across simulated
provisional inserts.

**Explicitly report a verdict on raw-score entropy degeneracy on `cosine×100` scores.** A refutation
would revise the prior BFCL finding and is just as publishable as a confirmation.

Add `hnav/tests/test_no_raw_entropy_in_policy.py` asserting `H_raw` never appears in any
`Decision.reasons`.

---

### T6 — Labeling + counterfactual + M3 headroom
**Files:** `hnav/labeling/labels.py`, `hnav/labeling/conflict_index.py`,
`hnav/labeling/counterfactual.py`, `hnav/stage0/m3_headroom.py`

Implement the retained labels exactly as defined in `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` §3.
`WRITE_DESTRUCTIVE_OVERWRITE` is **dropped** — no write path in EvoMemBench ever overwrites.

Counterfactual replay is cheap on the primary arena: the store is append-only and the retriever is
a pure function of the store, so `S_without` is `S_with` minus one row of the bank matrix — no
re-run of the memorize phase. Grading is free. Cost per counterfactual = one LLM answer call.

Classes: `must_write`, `must_suppress`, `may_suppress`, `inert/superseded`, `uncertain`
(definitions in `EVOMEMBENCH_HNAV_IMPLEMENTATION_PLAN.md` §7.2).

**Report:** write-side and read-side headroom tables per §M3 of the protocol — base rates, would-
intervene rate, could-change-correctness rate, coverage, precision, harm rate.

---

### T7 — M4: the H2 test
**File:** `hnav/stage0/m4_marginal_diff_test.py`

On the **calibration split only**, fit nested logistic models with `y` = "candidate is `must_write`":
```
M_base : y ~ whole_blob_sim + qr_residual
M_diff : y ~ whole_blob_sim + qr_residual + diff_sim + diff_novelty
```
Report the likelihood-ratio test, ΔAUC with a subset-clustered bootstrap CI (10,000 resamples), and
calibration curves.

**H2 passes iff ΔAUC > 0 with 95% CI excluding 0 AND LRT p < 0.01.**

---

### T8 — Stage-0 report  `[GATE — HARD STOP]`
**File:** `hnav/stage0/report.py` → `STAGE0_REPORT.md`

Contents: M0 fidelity table and any invalidated signals; M1 real-embedding distributions
(superseding the lexical proxy); M1b grouping-ablation verdict; M2 distributions and the raw-entropy
verdict; M3 headroom per component; M4/H2 verdict; then a GO/NO_GO decision per component against
the frozen gate in `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` §4.

Tag each NO_GO with **which of three verdicts** it is — they must never be conflated:
1. **benchmark** — the class is too rare or absent → *"EvoMemBench does not generate this class."*
2. **detection** — the class is abundant but the signal does not predict it → genuine evidence against H-Nav.
3. **policy** — signal predicts, intervention does not repair.

**STOP HERE.** Do not implement `write_policy.py` / `read_policy.py`, do not run live arms.

---

## 4. Things a reasonable agent gets wrong here — read before starting

- **Do not rewrite the fact parser.** `hnav/labeling/conflict_analysis.py::parse` is validated at
  99.5%+ coverage. Import it.
- **Do not start with `hnav/core/`.** T1 is a kill switch and needs none of it. Writing geometry
  modules before T1 risks wasting all of it.
- **Do not use the benchmark's `questions` / `answers` in any signal.** They sit in the *same JSON
  file* as the fact contexts. This is the easiest and most fatal mistake available.
- **Do not use `latest(key)` on the write path.** Use `latest_before(key, serial)`.
- **Do not tune anything on `sh_64k` or `sh_262k`.** Calibration split is `sh_6k` + `sh_32k`.
- **Do not pool across subsets.** Store sizes span 455 → 18,332 facts and retrieval difficulty
  scales with them (49% → 98.5% of conflict pairs land in different chunks). Report stratified.
- **Do not treat 884 CrossEp-Know samples as independent.** ICC(context) = 0.346, design effect
  3.20, effective N ≈ 276. Cluster by `context_id`.
- **Do not use `mh_*` subsets for primary analysis.** Multi-hop questions traverse ≥2 facts, so
  question→fact mapping is not 1:1. Exploratory only.
- **Do not add H-Nav to `In-Episode-Execution/` or `Cross-Episode-Execution/`.** Both are BFCL.

---

## 5. Deliverables

```
hnav/core/{types,replica,audit,retrieval_signals,geometry,diff_geometry}.py
hnav/adapters/{mab_adapter,clbench_adapter}.py
hnav/labeling/{labels,conflict_index,counterfactual}.py
hnav/stage0/{m1_geometry_calibration,m1b_grouping_ablation,m2_retrieval_calibration,
             m3_headroom,m4_marginal_diff_test,report}.py
hnav/tests/{test_replica_fidelity,test_shadow_neutrality,test_leakage_audit,
            test_no_raw_entropy_in_policy,test_label_definitions}.py
STAGE0_REPORT.md
```
Plus the four guarded benchmark edits from T4.

Commit after each task with the task ID in the message. Push to
`claude/evomembench-hnav-analysis-nfwl9z` (PR #1). **Report and stop at every `[GATE]`.**
