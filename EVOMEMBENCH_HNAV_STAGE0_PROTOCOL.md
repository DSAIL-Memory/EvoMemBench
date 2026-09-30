# EvoMemBench × H-Nav — Stage-0 Protocol (Preregistration)

**Purpose.** State exactly what is measured **before** any live H-Nav intervention, and the frozen
rules that decide which H-Nav components earn a live experiment.

**Status.** Draft for freezing. Once committed and git-tagged (`stage0-frozen`), §4–§6 must not be
edited. Any later change requires a new tag and must be disclosed in the write-up.

**Scope.**
- **Primary arena:** `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/`, `Conflict_Resolution`,
  single-hop subsets `factconsolidation_sh_{6k,32k,64k,262k}` — 400 questions, 4 fact stores.
- **Secondary arena:** `Cross-Episode-Knowledge/CROSSEP-KNOW/` — 884 samples, 120 contexts.
- **Excluded:** InEp-Exec, CrossEp-Tool (both BFCL — the substrate that already produced the null
  result), CrossEp-Web, CrossEp-Emb.
- **Calibration split:** `sh_6k` + `sh_32k`. **Confirmatory split:** `sh_64k` + `sh_262k`.
  Multi-hop (`mh_*`) subsets are exploratory only (question→fact mapping is not 1:1).

---

## 1. What is already established (no further measurement needed)

From `EVOMEMBENCH_HNAV_REPO_ANALYSIS.md`, computed from source and data in this repository:

| Fact | Value | Where |
| --- | --- | --- |
| Conflicted `(relation, subject)` keys | 54.6 – 65.2% | analysis §4.2 |
| Conflict group size | exactly 2, always | analysis §4.2 |
| Questions turning on a conflicted key (`sh_*`) | 65 – 77% | analysis §4.2 |
| Gold = latest serial number | 95 – 100% | analysis §4.2 |
| Conflict pairs spanning different chunks (262k) | 98.5% | analysis §10 |
| mem0 write path | append-only; dedup = exact MD5 only | analysis §4.1 |
| Native semantic dedup anywhere | **none** | analysis §4.1 |
| CrossEp-Know baseline accuracy | 23.87% | analysis §5.1 |
| CrossEp-Know ICC(context) / design effect | 0.346 / 3.20 | analysis §5.2 |
| Retriever replica feasibility (primary + secondary) | exact | analysis §7 |

These are **not** re-litigated in Stage 0. Stage 0 measures the things that require real embeddings,
real rankings, and real counterfactuals.

---

## 2. Measurements (M0–M4)

All performed in **shadow mode only**. No intervention. No store mutation.

### M0 — Retriever-replica fidelity

- **Measure:** exact rank-list identity between `RetrieverReplica` and the native retriever, over
  ≥1,000 randomly sampled real `(store state, query)` pairs per arena.
- **Report:** top-1 agreement, top-k set agreement, full-ranking Kendall τ, score max-abs-error,
  tie frequency.
- **Threshold:** ≥99.9% exact top-k identity. `NumpyCosineReplica` must be 100% modulo documented
  `np.argsort` tie order (`qwen3_embedding_memory.py:221`).
- **If it fails:** all of `rank_self`, `margin`, `dH_self`, `dH_neighbor`, `churn` are invalid.
  Document maximum achievable fidelity and drop those mechanisms explicitly — do **not** silently
  substitute a different retriever.

### M1 — Geometry calibration (real embeddings)

This replaces the char-3gram **proxy** used in analysis §9. The proxy is not a substitute and no
threshold may be set from it.

- **Embedder:** the native one. Prefer a **local** model (Qwen3-Embedding-4B or contriever via HF)
  for determinism and zero API dependency.
- **Measure**, over all conflict pairs and a matched sample of non-conflict pairs:
  - `sim_max` distribution (whole-blob, whitened and unwhitened)
  - QR residual novelty `r`
  - ABTT whitening stability vs store size (refuse below `min_fit_n`, report the fallback rate)
  - `tau_t` trajectory under the adaptive rule
  - exact-duplicate rate (expected ~0 on the primary arena — all facts are unique strings)
  - **verbatim-value overlap** between old and new facts
- **Report:** per-subset distributions with percentiles; separation between conflict pairs and
  random pairs (AUC).

### M2 — Retrieval calibration

- **Measure**, over all 400 primary-arena questions using the full pre-truncation ranking:
  - score scale, distribution, tie frequency
  - top1−top2 margin and normalized margin
  - `H_raw` (softmax over raw scores) — **logged to demonstrate degeneracy, never used to decide**
  - `H_z` (z-scored), `H_vn` (von Neumann over the top-m Gram matrix)
  - effective neighborhood size, dispersion
  - `dH_self`, `dH_neighbor`, `churn@k` across simulated provisional inserts
- **Explicit check:** confirm or refute raw-score entropy degeneracy on `cosine×100` scores. Report
  the result either way — a refutation is itself informative and would revise the BFCL finding.

### M3 — Headroom (per candidate H-Nav family)

Counterfactual labels per implementation plan §7. Grading on the primary arena is free and
deterministic.

**Write side:** total write decisions; duplicate rate; near-duplicate rate; update rate;
marginal-critical-delta rate; conflict rate; stale-replacement rate; how often geometry *would*
intervene; how often that intervention *could* change downstream correctness.

**Read side:** total reads; ambiguous-read rate; low-margin rate; stale/conflicting retrieval rate;
high-`dH` rate; high-churn rate; relevant-below-k rate; downstream failure rate within each.

**Counterfactual classes:** `must_write`, `must_suppress`, `may_suppress`, `inert/superseded`,
`uncertain` — per implementation plan §7.2.

### M4 — H2: does marginal-diff geometry add information beyond whole-blob geometry?

Nested logistic models on the **calibration split only**:

```
M_base : y ~ whole_blob_sim + qr_residual
M_diff : y ~ whole_blob_sim + qr_residual + diff_sim + diff_novelty
y      = "candidate is must_write"
```

Report LRT, ΔAUC with subset-clustered bootstrap CI, and calibration curves.

**H2 passes iff** ΔAUC > 0 with 95% CI excluding 0 **and** LRT p < 0.01.

This is the BFCL H2 re-run in an environment where `y` is expected at 65–77% rather than 3.5%, so a
pass is directly actionable rather than academic.

---

## 3. Failure labels retained

Only labels with a deterministic operational definition on this data are retained. Dropped labels
are listed with the reason — dropping a label because the benchmark does not generate the class is
a *finding*, not a gap.

### Write side

| Label | Definition | Online? | Retained |
| --- | --- | --- | --- |
| `WRITE_CRITICAL_DELTA` | candidate shares `(relation, subject)` with an admitted fact, differs in object, and whole-blob sim ≥ τ_high while diff sim ≤ τ_low | **yes** | **YES** |
| `WRITE_CONFLICT` | candidate shares `(relation, subject)` with an admitted fact, differs in object | **yes** | **YES** |
| `WRITE_STALE_SUPERSEDE` | a *later* fact exists for the same key | **no** (needs future) | **YES, offline only** |
| `WRITE_REDUNDANT` | `sim_max ≥ τ` and no new verbatim value | **yes** | **YES** (mainly CrossEp-Know) |
| `WRITE_DUPLICATE` | exact string/MD5 match | **yes** | YES, but measured at ~0 on the primary arena |
| `WRITE_UNNECESSARY` | admitted, and no downstream question maps to its key | **no** | YES, offline only |
| `WRITE_MISSED_UPDATE` | a fact present in the source text never became a candidate | **no** | YES, offline only (CrossEp-Know) |
| `WRITE_DESTRUCTIVE_OVERWRITE` | — | — | **DROPPED — the class does not exist.** No write path in EvoMemBench ever overwrites (analysis §4.1). |

### Read side

| Label | Definition | Online? | Retained |
| --- | --- | --- | --- |
| `READ_STALE` | a retrieved fact is superseded by a higher-serial fact in the store | **yes** (read time: whole store is legitimately visible) | **YES — primary target** |
| `READ_RELEVANT_BELOW_K` | the superseding fact exists but ranks below `top_k` | **yes** | **YES — primary target** |
| `READ_CONFLICT` | ≥2 retrieved facts share a key with differing objects | **yes** | **YES** |
| `READ_AMBIGUOUS` | normalized margin below the calibrated percentile | **yes** | YES |
| `READ_HIGH_INTERFERENCE` | `H_z` / `H_vn` above the calibrated percentile | **yes** | YES |
| `READ_DISTRACTOR` | a retrieved fact shares the subject but not the queried relation | **yes** | YES |
| `READ_MISSING` | no retrieved fact matches the queried key | **yes** | YES |
| `READ_CLEAR` | none of the above | **yes** | YES (the do-no-harm stratum) |

---

## 4. GO / NO_GO gate  ⟨FROZEN⟩

The task offers the BFCL precedent (≥25 positives, coverage ≥5%, precision ≥0.90, Wilson LB ≥0.80,
harm ≤0.05, ΔAcc ≈ ≥2pp) and instructs that it not be reused blindly. Below is the adapted gate,
with every deviation justified by a measured EvoMemBench property.

| Criterion | BFCL precedent | **EvoMemBench value** | Justification |
| --- | --- | --- | --- |
| Positive target cases | ≥25 | **≥40** (primary), ≥25 (secondary) | Base rates are 20–100× BFCL's (65–77% vs 3.5%), so positives are abundant and a higher bar costs nothing. Raising it guards against a component that only fires on a handful of cases. |
| Intervention coverage | ≥5% | **≥10%** (primary), ≥5% (secondary) | Same reason. On the primary arena a component covering <10% is not engaging the dominant failure class and cannot plausibly move a 400-question endpoint. |
| Precision | ≥0.90 | **≥0.90** (unchanged) | Retained. |
| Wilson 95% lower bound on precision | ≥0.80 | **≥0.80** (unchanged) | Retained. |
| Harmful-intervention rate, 95% upper bound | ≤0.05 | **≤0.03** (primary), ≤0.05 (secondary) | **Tightened.** In a supersession benchmark a wrong SUPPRESS deletes the unique answer-bearing fact — harm is not recoverable by other memories, unlike BFCL where redundant paths existed. |
| Expected ΔAcc | ≈ ≥2pp | **≥3pp** (primary), **≥2pp on the rubric-level endpoint** (secondary) | Primary N=400 with a paired McNemar design. At baseline ≈0.5 and a discordance rate of ~15%, 3pp (12 questions) is detectable; 2pp would sit inside the noise band of the A0′ replicate arm. Secondary uses rubric-level pass rate because binary all-or-nothing has only 10.7% one-rubric-away mass (analysis §5.1) and would need an implausible effect to move. |
| **Statistical unit** | — | **question**, paired; cluster-robust by **subset** (primary, 4–8 clusters) and by **context_id** (secondary, 120 clusters, ICC 0.346) | Measured; see analysis §5.2. |
| **A0′ noise floor** | — | **new requirement:** ΔAcc must exceed the A0′ replicate spread | Configs ship `temperature` 0.7–1.0. Without this, a within-noise result could be read as a gain. |

A component proceeds to a live arm **only if it meets every row** on the calibration split, with
thresholds then frozen and applied unchanged to the confirmatory split.

### Interpreting a failure — three distinct verdicts

These must never be conflated:

1. **NO_GO (benchmark)** — the target class is too rare or absent.
   → *"EvoMemBench does not meaningfully generate this failure class."*
   (Already the verdict for `WRITE_DESTRUCTIVE_OVERWRITE` and `WRITE_DUPLICATE` on the primary arena.)
2. **NO_GO (detection)** — the class is abundant but the signal does not predict it.
   → *"H-Nav's signal does not generalize to this failure class."* Genuine evidence against H-Nav.
3. **NO_GO (policy)** — signal predicts, intervention does not repair.
   → *"Detection succeeds, the proposed policy fails."* Separate detection from policy.

---

## 5. Pre-registered predictions

Stated now so they can be wrong.

| # | Prediction | Falsified if |
| --- | --- | --- |
| P1 | Real-embedding `sim_max` for conflict pairs will be high (p50 ≥ 0.85 unwhitened) | p50 < 0.7 → S3 fires; geometry premise fails |
| P2 | H2 passes (marginal-diff adds information beyond whole-blob) | LRT p ≥ 0.01 or ΔAUC CI includes 0 |
| P3 | Raw-score entropy is degenerate on `cosine×100`; `H_z` is materially better | `H_raw` matches or beats `H_z` in M3 predictiveness |
| P4 | **Read-side (A3) shows larger headroom than write-side (A1)** on the primary arena | write-side coverage/ΔAcc exceeds read-side |
| P5 | Geometry-only suppression (A1) is **neutral-to-harmful** on Conflict_Resolution, because every fact is unique and the newest is always needed | A1 shows ΔAcc ≥ 3pp with harm ≤ 0.03 |
| P6 | Headroom scales with store size: 6k ≪ 262k | no monotone trend across subsets |

P5 is deliberately a prediction *against* an H-Nav component. Recording it now prevents a null A1
result being reframed later as expected.

---

## 6. Freeze list

Committed and git-tagged before any live arm:

1. `tau_t` policy and all numeric thresholds (`τ_high`, `τ_low`, `r_min`, margin/entropy percentiles)
2. Feature definitions (`hnav/core/*.py` signal computations)
3. Label definitions (§3)
4. GO/NO_GO criteria (§4)
5. Primary endpoint, primary comparison, statistical tests (implementation plan §10)
6. Arm list and negative controls (implementation plan §9)
7. Calibration/confirmatory split assignment

**Nothing is tuned on `sh_64k` or `sh_262k`.**

---

## 7. Stage-0 exit report — required contents

1. M0 fidelity table; explicit list of any H-Nav signals invalidated
2. M1 real-embedding distributions, **superseding the lexical proxy in analysis §9**
3. M2 retrieval distributions + the raw-entropy degeneracy verdict
4. M3 headroom table, per component, with counterfactual class counts
5. M4 / H2 verdict with LRT and ΔAUC CI
6. GO/NO_GO decision per component, each tagged with verdict type 1/2/3 from §4
7. Final arm list, derived from the above — **not** copied from BFCL
8. Compute budget for the live campaign

**No live H-Nav intervention runs before this report exists and its gate is evaluated.**
