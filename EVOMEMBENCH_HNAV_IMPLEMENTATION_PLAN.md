# EvoMemBench × H-Nav — Implementation Plan

Companion to `EVOMEMBENCH_HNAV_REPO_ANALYSIS.md`. Read that first; this document assumes its
findings, in particular:

- **Primary arena:** `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/`, dataset
  `Conflict_Resolution` (factconsolidation), single-hop subsets — 400 questions.
- **Secondary arena:** `Cross-Episode-Knowledge/CROSSEP-KNOW/` — 884 samples, 120 contexts.
- **NO_GO:** InEp-Exec and CrossEp-Tool (both are BFCL), CrossEp-Web, CrossEp-Emb.

Everything here is designed so that **Stage 0 is behavior-neutral and cheap**, and no expensive
run happens before the GO/NO_GO gate in `EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` is evaluated.

---

## 1. Files to add / change

New code lives in one new top-level package. Benchmark forks are limited to **four one-to-three-line
hook edits**, all of which are no-ops when H-Nav is disabled.

### 1.1 New package (all new files)

```text
hnav/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py                 # Candidate, StoreView, MemoryRecord, Decision, RetrievalView
│   ├── geometry.py              # GeometryModule: ABTT whitening, sim_max, QR residual, tau_t
│   ├── diff_geometry.py         # DiffGeometryModule: marginal-diff extraction + scoring
│   ├── retrieval_signals.py     # RetrievalSignals: rank/margin/entropy/dH/churn
│   ├── replica.py               # RetrieverReplica (+ NumpyCosineReplica, FaissReplica, BM25Replica)
│   ├── write_policy.py          # WritePolicy: PASS | SUPPRESS | REWRITE | DEFER
│   ├── read_policy.py           # ReadPolicy: PASS | RERANK | EXPAND | TRIM | ANNOTATE
│   └── audit.py                 # AuditLogger (JSONL, one record per decision)
├── adapters/
│   ├── __init__.py
│   ├── mab_adapter.py           # InEp-Know / MemoryAgentBench  (PRIMARY)
│   └── clbench_adapter.py       # CrossEp-Know / CROSSEP-KNOW   (SECONDARY)
├── labeling/
│   ├── __init__.py
│   ├── fact_templates.py        # induced relation templates (99.5%+ coverage)
│   ├── conflict_index.py        # (relation, subject) -> [(serial, text, object)]
│   ├── labels.py                # WRITE_* / READ_* operational definitions
│   └── counterfactual.py        # S_with / S_without replay + outcome labeling
├── stage0/
│   ├── __init__.py
│   ├── m1_geometry_calibration.py
│   ├── m2_retrieval_calibration.py
│   ├── m3_headroom.py
│   ├── m4_marginal_diff_test.py     # the H2 test
│   └── report.py
└── tests/
    ├── test_replica_fidelity.py
    ├── test_shadow_neutrality.py
    ├── test_leakage_audit.py
    └── test_label_definitions.py
```

### 1.2 Benchmark edits (minimal, guarded, no-op when disabled)

| File | Line | Change | Neutral when off? |
| --- | --- | --- | --- |
| `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/embedding_retriever.py` | `185` `TextRetriever.retrieve` | Use `similarity_search_with_score_by_vector` to obtain the **full** ranking + scores; emit to the adapter; return the same `top_k` page-contents as before | **Yes** — return value byte-identical |
| `…/MemoryAgentBench/agent.py` | `967` `send_message` | Two adapter callbacks at entry/exit, gated on `HNAV_ENABLED` | **Yes** |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/registry.py` | `build_memory` | If `HNAV_ENABLED`, wrap the constructed backend in `HNavMemoryWrapper` | **Yes** — wrapper delegates verbatim in shadow mode |
| `…/CROSSEP-KNOW/infer_context_memory.py` | `196` result dict | Add `"hnav": {...}` to the written record | **Yes** — additive field, evaluator ignores unknown keys (`eval.py` reads only `model_output`/`rubrics`) |

`HNAV_ENABLED` / `HNAV_MODE` (`off` | `shadow` | `live`) are read from the environment, defaulting
to `off`. **Default-off is a hard requirement:** a stray import must never change a benchmark number.

---

## 2. Adapter architecture

```text
            EvoMemBench suite
                   │
    ┌──────────────┴───────────────┐
    │                              │
MABAdapter                   CLBenchAdapter
(agent.send_message,         (Memory.retrieve /
 TextRetriever.retrieve)      Memory.extract)
    │                              │
    └──────────────┬───────────────┘
                   ▼
               HNavCore
    ┌──────┬───────┬────────┬────────┬───────┬────────┐
 Geometry  Diff  Retrieval  Write    Read   Replica  Audit
  Module  Module  Signals   Policy  Policy          Logger
```

The adapters own **all** benchmark-specific knowledge (types, chunking, prompt formats, serial
numbers). `HNavCore` sees only `hnav.core.types`. This is what makes the BFCL H-Nav logic reusable
without importing anything from BFCL.

### 2.1 Core types (`hnav/core/types.py`)

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str                 # MAB: "fact:<serial>"; CLBench: f"{task_id}_chunk{chunk_index}"
    text: str
    vector: np.ndarray | None      # L2-normalized, native embedding space
    version: int                   # MAB: fact serial number; CLBench: monotone write counter
    metadata: dict                 # context_category, sub_category, task_id, created_at, ...

@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    vector: np.ndarray | None
    op: str                        # "ADD" (EvoMemBench never overwrites — see analysis §4.1)
    prior_id: str | None           # resolved predecessor, if any
    prior_text: str | None
    version: int
    metadata: dict

@dataclass
class StoreView:
    records: list[MemoryRecord]
    matrix: np.ndarray | None      # (N, d) L2-normalized bank matrix
    def with_provisional(self, cand) -> "StoreView": ...   # non-mutating, for dH/churn

@dataclass
class RetrievalView:
    query: str
    query_vector: np.ndarray
    ids: list[str]                 # FULL pre-truncation ranking
    scores: np.ndarray             # aligned, pre-truncation
    top_k: int                     # what the benchmark will actually use

@dataclass
class Decision:
    action: str                    # PASS | SUPPRESS | REWRITE | DEFER   (write)
                                   # PASS | RERANK | EXPAND | TRIM | ANNOTATE (read)
    payload: Any = None            # rewritten text / reordered id list
    reasons: dict = field(default_factory=dict)   # every signal that fed the decision
    shadow: bool = True            # if True, caller MUST ignore .action
```

`Decision.shadow` is a defence-in-depth invariant: in shadow mode the core still computes a full
decision (so it is logged and analyzable), but the adapter asserts `shadow is True` and discards it.

---

## 3. Module APIs

### `GeometryModule` (`hnav/core/geometry.py`)

```python
class GeometryModule:
    def __init__(self, whitening: ABTTWhitening | None, tau_policy: TauPolicy): ...
    def fit_whitening(self, store: StoreView) -> None:
        """Fit ABTT on the store's bank matrix. READ-ONLY w.r.t. native retrieval."""
    def compute(self, candidate: Candidate, store: StoreView) -> GeometrySignals:
        """-> sim_max, argmax_id, qr_residual_r, tau_t, is_exact_dup,
              verbatim_new_values, retrievability_floor_ok"""
```

**Note (analysis §8):** fit ABTT per store. On the 6k subsets (455 facts) and on early CrossEp-Know
contexts (<20 chunks) whitening statistics are unstable — the module must refuse to whiten below
`min_fit_n` (proposed: 200) and fall back to raw cosine, logging which path was taken.

### `DiffGeometryModule` (`hnav/core/diff_geometry.py`)

```python
class DiffGeometryModule:
    def compute_if_update(self, old: str | None, new: str) -> DiffSignals | None:
        """-> whole_blob_sim, diff_span_old, diff_span_new, diff_sim,
              diff_novelty, is_critical_delta"""
```

Diff-span extraction is **adapter-supplied** because it is benchmark-specific. For the primary
arena it is exact and cheap: parse both facts with the induced relation templates and take the
object slots (99.5%+ coverage — see `hnav/labeling/fact_templates.py`). Fall back to token-level
diff when parsing fails.

### `RetrievalSignals` (`hnav/core/retrieval_signals.py`)

```python
class RetrievalSignals:
    def compute(self, view: RetrievalView, self_id: str | None = None) -> RetrievalSignalSet:
        """-> rank_self, top1, top2, margin, nmargin,
              H_raw (logged, NOT used for decisions),
              H_z (z-scored, primary),
              H_vn (von Neumann over top-m Gram matrix),
              eff_size, dispersion"""
    def compute_delta(self, before: RetrievalView, after: RetrievalView,
                      probes: list[str]) -> DeltaSignalSet:
        """-> dH_self, dH_neighbor, churn@k, rank_shift"""
```

**Explicitly recorded design constraint (task §5, §7):** `H_raw` is computed and logged for
completeness but is **barred from every decision path**. Native scores are `cosine × 100`
(`qwen3_embedding_memory.py:218`), so a raw softmax is scale-degenerate exactly as BFCL found.
`H_z` is primary; `H_vn` is the secondary geometry-aware variant. A unit test asserts no policy
reads `H_raw`.

### `RetrieverReplica` (`hnav/core/replica.py`)

```python
class RetrieverReplica(Protocol):
    def rank(self, store: StoreView, query: str) -> RetrievalView: ...
    def simulate_insert(self, store: StoreView, cand: Candidate,
                        probes: list[str]) -> dict[str, RetrievalView]: ...

class NumpyCosineReplica:      # CrossEp-Know Qwen3EmbeddingMemory — BIT-EXACT
class FaissFlatReplica:        # InEp-Know TextRetriever — exact given same embedder
class BM25OkapiReplica:        # CrossEp-Know BM25Memory — exact
```

`NumpyCosineReplica.rank` is literally the native computation:

```python
scores = (store.matrix @ qv) * 100.0
order  = np.argsort(scores)[::-1]
```

`simulate_insert` appends one row to `store.matrix` and rescores — this is what makes `dH_self`,
`dH_neighbor` and `churn` computable *before* the write commits, which is the whole point of
building the replica first.

### `WritePolicy` / `ReadPolicy`

```python
class WritePolicy:
    def decide(self, geometry, diff, retrieval_effect, ctx) -> Decision: ...

class ReadPolicy:
    def decide(self, view: RetrievalView, store: StoreView, ctx) -> Decision: ...
```

Composition order for `WritePolicy` (frozen before live runs):

```
1. exact-duplicate fast path      → SUPPRESS
2. verbatim-value guard           → veto SUPPRESS if candidate introduces a new verbatim value
3. marginal-diff critical delta   → veto SUPPRESS, force PASS
4. retrievability floor           → veto SUPPRESS if suppressing makes a probe unretrievable
5. sim_max >= tau_t AND r < r_min → SUPPRESS
6. otherwise                      → PASS
```

Steps 2–4 are **vetoes on suppression**, never triggers for it. In an append-only, supersession-
based benchmark, a wrong SUPPRESS destroys the answer; a wrong PASS only costs context. The policy
is deliberately asymmetric toward PASS. (Analysis §9's "Howard/Harvard" case at whole-blob 0.949 is
precisely why steps 2–3 exist.)

---

## 4. Real EvoMemBench-specific pseudocode

### 4.1 Write path — InEp-Know / MemoryAgentBench (PRIMARY)

Native call site: `agent.py:967 AgentWrapper.send_message(message, memorizing=True, …)`.
The `message` is a 4096-token chunk of numbered facts, not a single fact, so the adapter explodes
it into per-fact candidates.

```python
# hnav/adapters/mab_adapter.py

FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)

def before_memorize(self, message: str, context_id: int, chunk_index: int):
    """Hook at agent.py:967, memorizing=True. Shadow mode: logs only, returns `message`."""
    store = self.store_view()                      # facts admitted so far, this context
    decisions = []

    for serial_str, fact_text in FACT_RE.findall(message):
        serial = int(serial_str)
        parsed = fact_templates.parse(fact_text)   # (relation, subject, object) | None

        cand = Candidate(
            id=f"fact:{serial}", text=fact_text,
            vector=self.embed(fact_text),          # native embedder, cached to disk
            op="ADD",
            prior_id=None, prior_text=None,
            version=serial,                        # ← the benchmark's own supersession key
            metadata={"context_id": context_id, "chunk_index": chunk_index,
                      "relation": parsed[0] if parsed else None,
                      "subject":  parsed[1] if parsed else None,
                      "object":   parsed[2] if parsed else None},
        )

        # Resolve the predecessor ONLINE: the newest already-admitted fact with the
        # same (relation, subject). Uses only facts with serial < current serial.
        prior = self.conflict_index.latest_before(parsed, serial) if parsed else None
        if prior:
            cand = replace(cand, prior_id=prior.id, prior_text=prior.text)

        geometry = self.geometry.compute(cand, store)
        diff     = self.diff.compute_if_update(cand.prior_text, cand.text)

        probes   = self.probe_generator(cand)      # see 4.3 — decision-time info only
        effect   = self.replica.simulate_insert(store, cand, probes)
        rsig     = self.signals.compute_delta(
                       before=effect["before"], after=effect["after"], probes=probes)

        decision = self.write_policy.decide(geometry, diff, rsig, ctx=cand.metadata)
        self.audit.log_write(cand, geometry, diff, rsig, decision,
                             native_action="ADD", store_size=len(store.records))
        decisions.append((cand, decision))

        store = store.with_provisional(cand)       # native behaviour = always admit

    if self.mode == "shadow":
        return message                             # ← byte-identical, guaranteed

    return self.apply_write_decisions(message, decisions)   # live mode only
```

### 4.2 Read path — InEp-Know / MemoryAgentBench

Native call site: `methods/embedding_retriever.py:196`, inside `TextRetriever.retrieve`.

```python
# patched TextRetriever.retrieve
def retrieve(self, query: str, top_k: int = 3) -> List[str]:
    qv = self.embedding_model.embed_query(query)
    scored = self.vectorstore.similarity_search_with_score_by_vector(
        qv, k=len(self._current_documents))        # FULL pre-truncation ranking
    native_page = [d.page_content for d, _ in scored[:top_k]]

    if not HNAV_ENABLED:
        return native_page

    view = RetrievalView(query=query, query_vector=np.asarray(qv),
                         ids=[self.doc_id(d) for d, _ in scored],
                         scores=np.array([s for _, s in scored]),
                         top_k=top_k)
    decision = hnav.after_retrieval(view, store=self.store_view(), query=query)
    hnav.audit.log_read(view, decision, native_top_k=[self.doc_id(d) for d, _ in scored[:top_k]])

    if hnav.mode == "shadow":
        assert decision.shadow
        return native_page                          # ← byte-identical, guaranteed
    return hnav.apply_read_decision(decision, scored, top_k)
```

```python
# hnav/adapters/mab_adapter.py
def after_retrieval(self, view: RetrievalView, store: StoreView, query: str) -> Decision:
    sig = self.signals.compute(view)               # rank/margin/H_z/H_vn/eff_size

    # Stale-supersession detection — the primary read-side target (analysis §10).
    # Uses ONLY the retrieved chunks and the store; never the gold answer.
    retrieved = view.ids[: view.top_k]
    stale_hits = []
    for rec in self.facts_in(retrieved):
        key = (rec.metadata["relation"], rec.metadata["subject"])
        newest = self.conflict_index.latest(key)   # highest serial in the WHOLE store
        if newest is not None and newest.version > rec.version:
            stale_hits.append((rec, newest, newest.id in retrieved))

    missing = [(old, new) for old, new, present in stale_hits if not present]

    return self.read_policy.decide(
        view, store,
        ctx={"signals": sig, "stale_hits": stale_hits, "superseder_missing": missing},
    )
```

`ReadPolicy` in live mode returns `EXPAND` with the missing superseding facts appended (bounded
budget), or `ANNOTATE` to mark stale entries. Both are pure prompt-construction changes upstream of
the LLM; the evaluator is untouched.

### 4.3 Probe generation (leakage-critical)

```python
def probe_generator(self, cand: Candidate) -> list[str]:
    """Deterministic probes from decision-time information ONLY.
    MUST NOT use: benchmark questions, gold answers, future facts, evaluator output."""
    if cand.metadata.get("subject"):
        rel = cand.metadata["relation"]
        return [self.templates.to_probe(rel, cand.metadata["subject"])]  # e.g. "Thomas Kyd was born in the city of"
    return [cand.text[:200]]
```

The benchmark's questions are **available on disk** in the same JSON as the context. It would be
trivially easy — and fatal — to use them as probes. `tests/test_leakage_audit.py` statically asserts
that no module under `hnav/core/` or `hnav/adapters/` imports from `hnav/labeling/counterfactual.py`
or reads the `questions` / `answers` keys. See §11.

### 4.4 Write path — CrossEp-Know (SECONDARY)

```python
# hnav/adapters/clbench_adapter.py
class HNavMemoryWrapper(Memory):
    """Wraps any cl_bench_memory backend. Shadow mode delegates verbatim."""
    def __init__(self, inner: Memory, hnav: HNavCore, mode: str):
        self._inner, self._hnav, self._mode = inner, hnav, mode

    def retrieve(self, query: str) -> tuple:
        text, stats = self._inner.retrieve(query)
        view = self._hnav.replica.rank(self.store_view(), query)   # exact replica
        decision = self._hnav.after_retrieval(view, self.store_view(), query)
        self._hnav.audit.log_read(view, decision, native_text=text)
        if self._mode == "shadow":
            return text, stats                     # ← byte-identical
        return self._hnav.apply_read_decision_text(decision, stats)

    def extract(self, content: str, **kwargs) -> dict:
        cand = self.to_candidate(content, **kwargs)
        geometry = self._hnav.geometry.compute(cand, self.store_view())
        diff     = self._hnav.diff.compute_if_update(self.resolve_previous(cand), cand.text)
        effect   = self._hnav.replica.simulate_insert(self.store_view(), cand,
                                                      self._hnav.probe_generator(cand))
        decision = self._hnav.write_policy.decide(geometry, diff, effect, ctx=kwargs)
        self._hnav.audit.log_write(cand, geometry, diff, effect, decision, native_action="ADD")
        if self._mode == "shadow" or decision.action == "PASS":
            return self._inner.extract(content, **kwargs)
        if decision.action == "SUPPRESS":
            return {"latency_s": 0.0, "llm_usage": {}, "embed_usage": {},
                    "verdict": "hnav_suppressed"}
        if decision.action == "REWRITE":
            return self._inner.extract(decision.payload, **kwargs)
```

Note the `extract` contract makes suppression clean: `infer_context_memory.py:171` uses the return
value only for latency/token accounting (`:186–193`). Returning zeroed stats is well-formed.

---

## 5. Shadow-mode implementation

Shadow mode is defined by three invariants, each enforced by a test:

| Invariant | Enforcement |
| --- | --- |
| **I1 — Return-value identity.** Every hooked function returns exactly what it would have returned with H-Nav off. | `tests/test_shadow_neutrality.py` runs the same seed twice (off vs shadow) with `temperature=0` and asserts byte-identical output JSONL, ignoring the additive `hnav` field. |
| **I2 — No store mutation.** H-Nav never inserts, reorders, or deletes. `StoreView.with_provisional` returns a new object. | `StoreView` records are `frozen=True`; the replica operates on copies. |
| **I3 — No extra model calls.** Shadow adds embedding calls only (cached), never LLM calls. | Token-usage assertion: `stats.tokens.inference` and `extract_llm` unchanged vs baseline. |

**Acceptance test (task §9):**

> Run `factconsolidation_sh_6k` and `sh_32k` with `HNAV_MODE=off` and `HNAV_MODE=shadow`,
> `temperature=0`, same seed. Assert identical `substring_exact_match` per question and identical
> model outputs. Any difference is a bug, not stochastic variation — at `temperature=0` with a
> deterministic evaluator there is no legitimate source of variation.

For CrossEp-Know, exact identity requires `temperature=0` too; the shipped configs use 0.7–1.0
(analysis §13.3). **Set temperature to 0 for all neutrality tests.** If the endpoint cannot honor
`temperature=0`, fall back to a 3-replicate distributional equivalence check and document it.

### 5.1 Embedding cost control

Shadow mode needs an embedding for every fact (18,332 in `sh_262k`) and every query. Cache to disk
keyed by `sha256(text) + model`, so:

- the first Stage-0 pass pays for ~26k embeddings total across all 8 subsets,
- every subsequent replay (all of Stage 0, all counterfactuals) is free.

With a local HF embedder (contriever or Qwen3-Embedding-4B), cost is GPU-minutes and there is no
API dependency at all — **strongly preferred**, and it also fixes the determinism problem.

---

## 6. Required logs

One JSONL record per decision, written by `AuditLogger`. Fields marked **[offline]** are filled in
by a *separate* post-hoc pass and are never visible to the online path.

### 6.1 Write record

```jsonc
{
  "schema": "hnav.write.v1",
  "run_id": "...", "arm": "A0-shadow", "mode": "shadow",
  "suite": "InEp-Know", "subset": "factconsolidation_sh_262k",
  "context_id": 0, "step": 41, "chunk_index": 3,
  "candidate_id": "fact:7334", "operation": "ADD",
  "candidate_text": "...", "candidate_version": 7334,
  "prior_id": "fact:4731", "prior_text": "...", "prior_version": 4731,
  "store_size": 7301,

  "geometry": {"sim_max": 0.0, "argmax_id": "...", "qr_residual_r": 0.0,
               "tau_t": 0.0, "is_exact_dup": false, "whitened": true,
               "verbatim_new_values": ["Harvard University"],
               "retrievability_floor_ok": true},

  "diff": {"whole_blob_sim": 0.0, "diff_span_old": "Howard University",
           "diff_span_new": "Harvard University", "diff_sim": 0.0,
           "diff_novelty": 0.0, "is_critical_delta": true, "parse_ok": true},

  "retrieval_effect": {"probes": ["..."], "rank_self": 3, "top1": 0.0, "top2": 0.0,
                       "margin": 0.0, "nmargin": 0.0,
                       "H_raw": 0.0, "H_z": 0.0, "H_vn": 0.0,
                       "dH_self": 0.0, "dH_neighbor": 0.0,
                       "churn_at_k": 2, "eff_size": 0.0, "dispersion": 0.0},

  "native_action": "ADD",
  "hnav_action": "PASS",
  "hnav_reasons": {"veto": "critical_delta"},

  "outcome": {                                   // [offline]
    "downstream_question_ids": ["sh_262k_q17"],
    "label": "WRITE_CRITICAL_DELTA",
    "counterfactual_class": "must_write",
    "correct_with": true, "correct_without": false
  }
}
```

### 6.2 Read record

```jsonc
{
  "schema": "hnav.read.v1",
  "run_id": "...", "arm": "...", "mode": "shadow",
  "subset": "factconsolidation_sh_262k", "question_id": "sh_262k_q17",
  "query": "...", "store_size": 18332, "top_k": 10,
  "ranking": {"ids": ["..."], "scores": [0.0], "truncated_at": 10},
  "signals": {"rank_self": null, "top1": 0.0, "top2": 0.0, "margin": 0.0, "nmargin": 0.0,
              "H_raw": 0.0, "H_z": 0.0, "H_vn": 0.0, "eff_size": 0.0, "dispersion": 0.0},
  "conflict": {"stale_in_topk": [{"old_id": "fact:4731", "new_id": "fact:7334",
                                  "superseder_retrieved": false, "rank_of_superseder": 34}],
               "n_conflicted_keys_in_topk": 3},
  "native_topk_ids": ["..."],
  "hnav_action": "PASS", "hnav_payload_ids": null,
  "outcome": {"label": "READ_STALE", "correct": false}    // [offline]
}
```

`H_raw` is logged but a test asserts it never appears in `hnav_reasons`.

---

## 7. Counterfactual-labeling design

The primary arena makes true counterfactual replay affordable, which is the single biggest
methodological advantage over BFCL.

### 7.1 What replay costs

The evaluator is `substring_exact_match` — free and deterministic. So the cost of one counterfactual
is exactly **one LLM answer call**. Nothing else.

### 7.2 Write-side counterfactual

For a candidate fact `c` at serial `t`, define:

```
S_with    = store containing c
S_without = store with c withheld
```

Because the store is append-only and the retriever is a pure function of the store, `S_without` is
obtained by dropping one row from the bank matrix — no re-run of the memorize phase is needed.
Downstream questions affected by `c` are exactly those whose gold answer maps to `c`'s
`(relation, subject)` key, computable offline from `fact_templates` + the answer list.

```python
def label_write(c, affected_questions):
    with_ok    = [answer_and_grade(q, retrieve(S_with, q))    for q in affected_questions]
    without_ok = [answer_and_grade(q, retrieve(S_without, q)) for q in affected_questions]
    ...
```

| Class | Definition |
| --- | --- |
| `must_write` | ∃q: `correct_with(q)` ∧ ¬`correct_without(q)` |
| `must_suppress` | ∃q: ¬`correct_with(q)` ∧ `correct_without(q)` |
| `may_suppress` | ∀q: `correct_with(q)` == `correct_without(q)`, and `c` is not the newest for its key |
| `inert/superseded` | `c` is superseded by a later fact and no question maps to it |
| `uncertain` | affected-question set empty, or grading unstable across replicates |

**Expected shape, from the measured structure (analysis §4.2):** the newest fact of each conflicted
key is a `must_write` candidate for any question on that key — that is 65–77% of questions. The
older twin is the `must_suppress` / stale candidate. This is the direct inverse of BFCL's 3.5% /
0.9%.

### 7.3 Read-side counterfactual

Cheaper still and does not even require a store variant:

```
top_k_native   = native ranking truncated at k
top_k_repaired = native ranking with the missing superseding fact injected
```

Grade both. The difference isolates the **retrieval bottleneck** exactly, holding the model and
store fixed. Given 98.5% of conflict pairs are cross-chunk at 262k, this is where the largest
measurable effect should be.

### 7.4 CrossEp-Know approximation (documented limitation)

True counterfactual replay is **not affordable** there: the store is model-generated
(`format_trajectory(messages, response_text)`), so withholding a write changes every subsequent
episode's context, requiring a full re-run of the context, and each grade costs a `gpt-5.1` judge
call over ~24 rubrics. The defensible approximation:

- **Frozen-store counterfactual.** Take a completed baseline run's store; vary only the *retrieved
  set* for a single question; re-answer and re-grade. This isolates read-side effects exactly and
  costs one judge call per counterfactual.
- **Write-side effects are NOT counterfactually labeled** in CrossEp-Know. Report write-side
  results there as *associational only*, and say so explicitly in any write-up.

Additionally: use **rubric-level pass rate** (`requirement_status`, already recorded per sample) as
the sensitive secondary endpoint, since binary all-or-nothing dilutes single-fact repairs
(analysis §5.1).

---

## 8. Stage-0 analysis pipeline

Five measurements, in order. Each is cheap; none requires a live intervention.

| ID | Measurement | Input | Output | Cost |
| --- | --- | --- | --- | --- |
| **M0** | Replica fidelity | one shadow run per arena | rank-identity rate vs native | minutes |
| **M1** | Geometry calibration | real embeddings of all conflict pairs | `sim_max`, QR-`r`, `tau_t` distributions; **replaces the lexical proxy in analysis §9** | GPU-minutes (local embedder) |
| **M2** | Retrieval calibration | full pre-truncation rankings | margin, `H_z`, `H_vn`, `eff_size` distributions; raw-entropy degeneracy check | minutes |
| **M3** | Headroom | M1 + M2 + counterfactual labels | per-class base rates, coverage, precision, expected ΔAcc | 1 LLM call per counterfactual |
| **M4** | **H2 marginal-diff test** | M1 | does diff geometry add predictive information beyond whole-blob? | minutes |

**M4 is the scientifically pivotal one** and is worth stating precisely, since BFCL's H2 passed as a
signal but had no headroom:

> **H2 (EvoMemBench form).** Let `y` = "this candidate is `must_write`" (i.e. suppressing it breaks
> a downstream question). Fit two nested logistic models on a dev split:
> `M_base`: `y ~ whole_blob_sim + qr_residual`
> `M_diff`: `y ~ whole_blob_sim + qr_residual + diff_sim + diff_novelty`
> Report the likelihood-ratio test, ΔAUC with a context-clustered bootstrap CI, and calibration.
> **H2 passes iff ΔAUC > 0 with CI excluding 0 AND the LRT p < 0.01.**
>
> Unlike BFCL, `y` here is *not* rare: it is expected at 65–77%. So a positive H2 is
> immediately actionable rather than academic.

Reproducible measurement scripts from this analysis (already written and run; port into
`hnav/stage0/`):

```
conflict_analysis.py   # template induction, (relation,subject) grouping, conflict census
gold_rule.py           # verifies gold == latest-serial, question-level headroom
marginal_diff.py       # whole-blob vs marginal-diff separation
```

---

## 9. Proposed experiment arms

**Do not run any of these until the Stage-0 gate passes.** Arm selection is *determined by* Stage 0,
not chosen in advance — the list below is the candidate superset.

| Arm | Description | Runs only if |
| --- | --- | --- |
| **A0** | Native EvoMemBench baseline, `temperature=0` | always |
| **A0′** | Replicated baseline (3 seeds, or 3 replicates at `temperature=0.7` if the endpoint forces sampling) | always — establishes the noise floor |
| **A1** | Geometry-only write admission (GeometryGate + exact-dup + `tau_t`) | M3 shows write-side coverage ≥ gate |
| **A2** | A1 + marginal-diff-aware geometry (critical-delta veto) | **M4 (H2) passes** |
| **A3** | Read-only: rank/margin/`dH`/churn + stale-supersession repair | M3 read-side coverage ≥ gate |
| **A4** | Full justified H-Nav — union of components that passed | ≥2 components pass |

**Expected a priori:** A3 is the strongest candidate on the primary arena (98.5% cross-chunk
conflict pairs, direct repair available), A2 second. A1 alone may be *weak or harmful* on
Conflict_Resolution because there is nothing to suppress — all facts are unique and the newest is
always needed. That asymmetry is a genuine prediction and worth pre-registering.

### Negative controls (mandatory for any live arm)

| Control | Purpose |
| --- | --- |
| **C1 — shuffled signal** | Permute `dH`/margin across examples within a subset; re-apply the identical policy at identical coverage. Must collapse to baseline. |
| **C2 — random intervention at matched coverage** | Intervene on a random subset of the same size. Distinguishes "the signal picked the right cases" from "intervening helps generally." |
| **C3 — equal-compute** | Give A0 the same extra context budget A3 spends on expansion (e.g. top-k+Δ). **Essential:** an `EXPAND` policy that adds facts might win purely by adding context. C3 is the control that makes A3 interpretable. |
| **C4 — irrelevant probe** | Replace probes with probes from a different `(relation, subject)`. Must collapse. |

C3 is the one most likely to overturn a naive positive result and must not be skipped.

---

## 10. Statistical analysis

| Item | Choice | Rationale |
| --- | --- | --- |
| Primary endpoint | `substring_exact_match` on `factconsolidation_sh_*` (400 questions) | Deterministic, free, per-question |
| Primary comparison | Arm vs A0, **paired by question** | Same store, same model, same prompt |
| Primary test | **McNemar** on discordant pairs | Correct for paired binary outcomes; sidesteps most of the clustering penalty |
| Clustering | Cluster-robust by **subset** (8) for store-level claims; by **context_id** (120) for CrossEp-Know | Measured ICC 0.346, design effect 3.20, effective N ≈ 276 (analysis §5.2) |
| CIs | Cluster bootstrap, 10,000 resamples | Small cluster count → bootstrap over `t`-approximation |
| Multiplicity | Holm–Bonferroni across arms; strata are **pre-registered** and reported with CIs, not p-values | Prevents strata-fishing |
| Secondary (CrossEp-Know) | Rubric-level pass rate from `requirement_status` | Binary all-or-nothing is too blunt (analysis §5.1) |

**Stratified reporting is the thesis analysis**, not a robustness check. Every live arm reports:

```
overall Δ, CI, positive flips, negative flips, coverage, precision, harm rate, cost, latency
```

stratified by: `critical-delta` / `duplicate` / `conflict` / `stale` writes; and
`clear` / `ambiguous` / `conflicting` / `stale` / `high-interference` reads.

The claim being tested is **"H-Nav preferentially improves the strata it claims to detect"** —
i.e. a significant stratum × arm interaction, not merely a positive main effect.

---

## 11. Research-integrity audit

| Rule | Enforcement |
| --- | --- |
| No gold answers online | `tests/test_leakage_audit.py`: static AST scan asserting no module under `hnav/core/` or `hnav/adapters/` references `answers`, `gold`, `rubrics`, or imports `hnav.labeling.counterfactual`. |
| No future facts online | `conflict_index.latest_before(key, serial)` is the only online index API; `latest(key)` (whole-store) is permitted on the **read** path only, because at read time all facts are legitimately in the store. Enforced by separate class boundaries. |
| No evaluator output online | `AuditLogger.outcome` is written by a distinct offline pass into a separate file, joined by `run_id + candidate_id`. The online record is closed before grading runs. |
| Dev/test split | Calibrate `tau_t`, `r_min`, thresholds on `sh_6k` + `sh_32k`; **freeze**; confirm on `sh_64k` + `sh_262k`. Never tune on the confirmatory subsets. |
| Frozen artifacts | Thresholds, feature definitions, GO/NO_GO criteria, and primary comparisons committed to git with a tag before any live arm. |

The `latest_before` / `latest` distinction is the subtlest leakage risk in the whole design and
deserves a reviewer's attention: at **write** time, only facts with a smaller serial have been
observed, so using the whole-store index would be look-ahead. At **read** time the full store is
legitimately present.

---

## 12. Implementation order

```
1. hnav/labeling/fact_templates.py + conflict_index.py     ← already validated (99.5%+ coverage)
2. hnav/core/types.py, audit.py
3. hnav/core/replica.py (NumpyCosineReplica, FaissFlatReplica) + M0 fidelity test
4. Benchmark hook edits (4 files), default-off  + I1/I2/I3 neutrality tests
5. Embedding cache + local embedder wiring (determinism!)
6. M1, M2 calibration                                       ← replaces the lexical proxy
7. hnav/core/geometry.py, diff_geometry.py, retrieval_signals.py
8. hnav/labeling/counterfactual.py + labels.py  → M3
9. M4 (H2 test)
10. STAGE-0 REPORT → GO/NO_GO gate  ◄── STOP. No live code before this passes.
11. write_policy.py / read_policy.py (only components that passed)
12. Live arms + negative controls
```

Steps 1–10 involve **no intervention and no expensive campaign**. Step 10 is a hard stop.

---

## 13. Tests / acceptance criteria

| Test | Criterion |
| --- | --- |
| `test_replica_fidelity` | ≥99.9% exact rank-list identity vs native on ≥1,000 sampled `(store, query)` pairs; 100% for `NumpyCosineReplica` modulo documented `argsort` tie order |
| `test_shadow_neutrality` | Byte-identical outputs and identical per-question scores, `HNAV_MODE=off` vs `shadow`, `temperature=0` |
| `test_shadow_no_llm_calls` | `stats.tokens.inference` and `extract_llm` unchanged vs baseline |
| `test_leakage_audit` | AST scan passes; zero references to gold/answers/rubrics in online modules |
| `test_label_definitions` | Every label has a deterministic function, a declared online/offline status, and ≥50 hand-checkable examples |
| `test_template_coverage` | ≥99% parse coverage on every Conflict_Resolution subset |
| `test_no_raw_entropy_in_policy` | `H_raw` never appears in any `Decision.reasons` produced by a policy |

---

## 14. Compute requirements

Stage 0 (everything up to the gate):

| Item | Estimate |
| --- | --- |
| Embeddings, all 8 subsets + queries | ~26k texts, one-off, cached. Local Qwen3-Embedding-4B: ~10 GPU-min. |
| Shadow runs, `sh_*` (400 questions) | 400 LLM answer calls per arm |
| Counterfactual labeling, write-side | ~1 call per (candidate × affected question); bound to the ~7k conflicted keys sampled down to a dev set of ~1,000 → ~1,000 calls |
| Counterfactual labeling, read-side | ~400 calls (one repaired-retrieval variant per question) |
| Grading | **free** (deterministic substring match) |
| **Stage-0 total** | **~2k LLM calls + ~10 GPU-min.** No GPU campaign. |

Live arms (post-gate): 400 questions × (4 arms + 4 controls) ≈ 3,200 calls per model, ×2 models for
cross-model validation ≈ 6,400. Still small. **The expensive component is CrossEp-Know's `gpt-5.1`
judge (884 samples × arms), which is the main reason to keep CrossEp-Know secondary and to use
rubric-level scoring rather than more arms.**

---

## 15. STOP / GO decision points

| # | Point | STOP condition |
| --- | --- | --- |
| **S1** | After M0 | Replica fidelity < 99.9% on the primary arena → fix or drop all `dH`/churn/rank signals and document which mechanisms die with them |
| **S2** | After neutrality tests | Any shadow-vs-off difference at `temperature=0` → **hard stop**, fix before proceeding |
| **S3** | After M1 | Real-embedding `sim_max` for conflict pairs is **not** high (say p50 < 0.7) → the "near-duplicate that is actually an update" premise fails; DROP GeometryGate and marginal-diff, keep read-side only |
| **S4** | After M3 | Any component failing the frozen gate (see Stage-0 protocol) → **NO_GO that component**, and report *"EvoMemBench does not generate this class"*, not *"H-Nav failed"* |
| **S5** | After M4 | H2 fails → DROP A2; A1/A3 may still proceed |
| **S6** | Before live | Thresholds/criteria not committed and tagged in git → do not run |
| **S7** | After first live arm | Negative control C1/C2/C4 does **not** collapse, or C3 (equal-compute) explains the gain → the result is not attributable to H-Nav; report as such |

S7 is the one that most often converts an apparent success into an honest null. It is
pre-registered here deliberately.
