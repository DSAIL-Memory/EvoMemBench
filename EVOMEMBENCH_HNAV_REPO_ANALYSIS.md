# EvoMemBench × H-Nav — Repository Analysis

**Status:** Stage-0 analysis, pre-implementation.
**Scope:** determine whether H-Nav can be *meaningfully and scientifically* ported to EvoMemBench.
**Verdict up front:** Yes — but not to the benchmark as a whole. Exactly one subset
(`InEp-Know / Conflict_Resolution`) is a near-ideal H-Nav arena, one (`CrossEp-Know`) is a
structurally clean but statistically weak secondary, and the rest are NO_GO. Details in §11, §14.

---

## 0. Provenance and a required caveat about the blueprint

`HNAV_EVOMEMBENCH_PORTING_BLUEPRINT.md` **was not present** in this environment. I checked:

- the repository working tree and all of `git log --all --diff-filter=A` (no such path ever committed),
- the whole filesystem (`find / -iname '*hnav*'`),
- the session scratchpad.

The task description itself restates the H-Nav specification in substantial detail — the component
list (§5 of the task), the BFCL base rates (`must_write` ≈ 3.5%, `must_suppress` ≈ 0.7–0.9%), the
intervention verbs (`PASS / SUPPRESS / REWRITE / DEFER`), the GO/NO_GO gate, the failure taxonomy
and the arm structure. **I have used that as the working definition of H-Nav.** Everything in this
document about *EvoMemBench* is verified against source; everything about *H-Nav's internals* is
taken from the task text and should be re-checked against the real blueprint before implementation.

Specifically, these H-Nav details are **assumed, not verified**, and are the ones most likely to
need correction:

| Assumed | Why it matters |
| --- | --- |
| ABTT whitening is applied to a *governance* embedding space, separate from the native retrieval space | Determines whether we need a second embedding pass (§8) |
| `tau_t` is an adaptive per-step threshold over `sim_max` | Determines the calibration protocol |
| `dH_self` / `dH_neighbor` are entropy *deltas* across a provisional insert | Determines whether RetrieverReplica must support simulated inserts (it can — §7) |
| The retrievability floor is a *veto* on SUPPRESS, not a separate gate | Changes the write-policy composition order |

None of these change the conclusions below, because the conclusions rest on EvoMemBench's
measured properties, not on H-Nav's exact parameterization.

---

## 1. EvoMemBench architecture

EvoMemBench is not one benchmark. It is a **harness-of-harnesses**: six task suites, each a
vendored third-party benchmark with a thin memory-adapter layer bolted on, plus a shared directory
of vendored memory backends. There is **no common runner, no common data format, and no common
memory API across suites.** This is the single most important architectural fact for the port:
*a "port of H-Nav to EvoMemBench" is not one integration, it is a choice of which suite to integrate with.*

| Suite | Dir | Source benchmark | Samples | Memory API |
| --- | --- | --- | --- | --- |
| InEp-Know | `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/` | MemoryAgentBench | 2,800 | `AgentWrapper.send_message(msg, memorizing=…)` |
| InEp-Exec | `In-Episode-Execution/INEP-EXEC/` | BFCL v4 multi-turn (long-context) | 800 | BFCL internals |
| CrossEp-Know | `Cross-Episode-Knowledge/CROSSEP-KNOW/` | CL-Bench | 884 | `Memory.retrieve()` / `Memory.extract()` |
| CrossEp-Tool | `Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/` | BFCL v4 multi-turn (base) | 800 | `EvolveLab/base_memory.py` |
| CrossEp-Web | `Cross-Episode-Execution/Web-Search/CROSSEP-WEB/` | xbench-DeepSearch, WebWalkerQA | 270 | Flash-Searcher |
| CrossEp-Emb | `Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/` | ALFWorld / AgentGym | 200 | agentenv |

Shared backends live in `EvoMemBench-Memory-Systems/`: `mem0`, `A-mem`, `MemOS`, `MemoryOS`,
`MemoBrain`, `memagent`.

Two suites have clean, single-chokepoint memory interfaces (InEp-Know, CrossEp-Know). The rest
require forking benchmark internals. **Two of the six (InEp-Exec, CrossEp-Tool) are BFCL** — the
substrate where H-Nav has already returned a null result. Porting to those would re-run the
experiment that already failed, on the same data.

---

## 2. End-to-end memory lifecycle

### 2.1 CrossEp-Know (`Cross-Episode-Knowledge/CROSSEP-KNOW/`) — the cleanest lifecycle

Real flow, verified in `infer_context_memory.py`:

```text
CL-bench_context_ge5.jsonl  (884 samples, 120 contexts, 5–12 samples/context)
    │  group_by_context()                              infer_context_memory.py:74
    ▼
per-context serial loop (contexts run in parallel threads)  :90 process_context_group
    │  build_memory(memory_type, …, memory_dir=<run>/<context_id>)   :119
    ▼
for each sample in context, IN FILE ORDER:
    query = get_last_user_message(messages)            trajectory.py
    memory_text, stats = memory.retrieve(query)        :152   ◄── READ HOOK
    augmented = inject_memory_into_messages(msgs, memory_text)  :153
                   └─ appends memory_text to the SYSTEM message  trajectory.py:inject
    response, err, usage = call_openai_api(client, augmented, model)   :156
    trajectory = format_trajectory(messages, response_text)            :170
    extract_stats = memory.extract(content=trajectory,                 :171   ◄── WRITE HOOK
                                   task_id, query,
                                   context_category, sub_category)
    append_jsonl(result, output_path)                                  :208
```

State mutation: `memory.extract()` is the *only* mutation point. Memory is isolated per
`context_id` on disk under `memories/context_memory/<run>/<context_id>/`.

**Critical property:** what gets written to memory is derived from the *model's own output*
(`format_trajectory(messages, response_text)`), plus the ground-truth assistant demonstrations
already present in `messages`. Memory content is therefore partly model-generated and varies
between runs.

### 2.2 InEp-Know (`In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/`)

```text
Accurate_Retrieval.json / Conflict_Resolution.json
    │  ConversationCreator._load_and_process_dataset()   conversation_creator.py:71
    │  get_chunks() → chunk_text_into_sentences(context, chunk_size)   :261
    │      chunk_size = 4096 tokens (dataset_config) or agent_chunk_size
    ▼
MEMORIZE PHASE — for each chunk:
    agent.send_message(chunk, memorizing=True, …)        agent.py:967   ◄── WRITE HOOK
        ├─ Long_context_agent → self.context += chunk               :1000
        ├─ rag                → self.chunks.append(chunk)           (_handle_rag_agent)
        └─ mem0/memos/amem/…  → backend .add()                      (_handle_memory_agent)
    ▼
QUERY PHASE — for each of the 100 questions:
    agent.send_message(query, memorizing=False, …)       agent.py:967   ◄── READ HOOK
        └─ rag → RAGSystem.answer_query()             embedding_retriever.py:252
             retrieval_query = regex-extracted from the templated prompt   :256
             retrieved = retriever.retrieve(retrieval_query, top_k)        :266
             prompt = "Memory 1:\n…" + query                               :274
    ▼
metrics_summarization(...)  → substring_exact_match      utils/eval_other_utils.py:463
```

`send_message(message, memorizing, query_id, context_id)` at `agent.py:967` is a **single
chokepoint that every memory backend routes through** — the best hook in the entire repository.

---

## 3. Retrieval architecture

Three genuinely different retrievers are in play. They differ enormously in how H-Nav-friendly they are.

### 3.1 `Qwen3EmbeddingMemory` (CrossEp-Know) — fully transparent

`cl_bench_memory/qwen3_embedding_memory.py:197`

```python
bank_matrix = np.stack(self.embeddings)      # L2-normalized, in memory
scores = (bank_matrix @ query_vec) * 100.0   # cosine × 100, FULL candidate set
k = min(self.top_k, len(self.memory_bank))
top_indices = np.argsort(scores)[::-1][:k]
```

- Similarity: cosine on L2-normalized vectors, scaled ×100. Range `[-100, 100]`, practically `[0, 100]`.
- **Candidate set = the entire bank.** Scores for every memory are computed before truncation.
- Embeddings are **persisted to disk** (`embeddings.jsonl`, one `{id, text, embedding}` per chunk).
- Store granularity: sentence-boundary chunks of ≤1024 tokens (`chunking.py`), one `extract()` → N chunks.
- Tie-breaking: `np.argsort` default kind is quicksort — **not stable**. Exact ties are resolved
  non-deterministically w.r.t. insertion order. In practice ties are measure-zero for float cosines.

This is the single best retrieval target in the repo: exact replication is a five-line function
over data already on disk.

### 3.2 `Mem0Memory` (CrossEp-Know) / mem0 backend — hybrid, replicable with effort

`EvoMemBench-Memory-Systems/mem0/mem0/memory/main.py:1343 _search_vector_store`

```text
internal_limit = max(limit*4, 60)          # ← OVER-FETCH POOL, larger than top-k
semantic  = vector_store.search(...)        # Qdrant cosine, top internal_limit
keyword   = vector_store.keyword_search(...)# BM25 over lemmatized text
bm25_score = normalize_bm25(raw, midpoint, steepness)   # query-adaptive sigmoid
entity_boosts = _compute_entity_boosts(...)  # [0, 0.5], from a separate entity vector store
scored = score_and_rank(semantic, bm25_scores, entity_boosts, threshold=0.1, top_k=limit)
```

- Pre-truncation scores **are** available: the pool is ≥60 candidates for `top_k=10`.
- But the final ranking is a **three-way fusion** (dense + normalized BM25 + entity boost) with a
  0.1 threshold. Replicating it requires reproducing `score_and_rank`, `normalize_bm25`,
  `get_bm25_params`, the lemmatizer, and the entity store. Feasible, but it is a real port, not a
  five-line function.

### 3.3 `TextRetriever` (InEp-Know RAG) — LangChain FAISS, scores discarded

`methods/embedding_retriever.py:185`

```python
results = self.vectorstore.similarity_search(query, k=initial_k)
return [doc.page_content for doc in results][:top_k]
```

- FAISS index built once per context (`FAISS.from_documents`), in process.
- **`similarity_search` throws the scores away.** This is the only real retrieval-side obstacle in
  the whole analysis — and it is a *one-line* obstacle: LangChain's FAISS vectorstore exposes
  `similarity_search_with_score_by_vector(embedding, k)`. Swapping the call and raising `k` to the
  full corpus size yields the complete pre-truncation ranking with no change to what is returned
  to the benchmark.
- Embedders configurable: DashScope `text-embedding-v4`, `Qwen/Qwen3-Embedding-4B` (API or local
  HF), `facebook/contriever` (local), `nvidia/NV-Embed-v2` (local), OpenAI. The local HF options
  (`ContrieverEmbeddings`, `Qwen3Embedding4BEmbeddings`) are deterministic and API-key-free — the
  right choice for a reproducible H-Nav campaign.
- `retrieve_num`: 10 for embedding/BM25/graph RAG, **100** for mem0
  (`configs/agent_conf/RAG_Agents/deepseek-chat/*.yaml`).

---

## 4. Update / evolution mechanics — what actually makes memory "evolve"

This is the section that decides the port, so it is the most heavily verified.

### 4.1 The mem0 write path is APPEND-ONLY

The vendored mem0 in `EvoMemBench-Memory-Systems/mem0/` uses the **V3 phased batch pipeline**
with `ADDITIVE_EXTRACTION_PROMPT` (`main.py:699–961`). Verified:

- The system prompt states: *"Your sole operation is ADD"* (`configs/prompts.py:468`).
- Every returned event is hardcoded `"event": "ADD"` (`main.py:692`, `:849`, `:961`).
- `_update_memory` and `_delete_memory` exist (`main.py:1657`, `:1722`) but are reachable **only**
  from the public `update()` / `delete()` APIs (`main.py:1501`, `:1524`). Neither CL-bench's
  `Mem0Memory` nor MemoryAgentBench's mem0 handler ever calls them.
- The only dedup is an **exact MD5 hash match** (`main.py:799–803`):

```python
mem_hash = hashlib.md5(text.encode()).hexdigest()
if mem_hash in existing_hashes or mem_hash in seen_hashes:
    continue      # ← the ONLY suppression in the entire write path
```

- Semantic dedup is delegated entirely to the LLM's judgment, via prompt text
  ("If new information … is semantically equivalent to an Existing Memory … skip it"), over the
  top-10 retrieved existing memories. There is no geometric check whatsoever.

**Consequence, and it is the central one for this port:** in EvoMemBench, near-duplicate and
superseding memories are *not suppressed by construction*. They accumulate. The store is designed
to grow monotonically and to contain stale-alongside-current facts. **The failure class H-Nav's
GeometryGate targets is not merely present — the native system has no mechanism that could
prevent it.** That is the opposite of the BFCL situation.

### 4.2 The Conflict_Resolution dataset is *engineered* to contain superseding updates

`In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json` — 8 subsets
(`{sh,mh} × {6k, 32k, 64k, 262k}`), 100 questions each, **800 questions total**. Each context is a
numbered fact list:

```text
0. Thomas Kyd was born in the city of London.
...
306. Thomas Kyd was born in the city of Leeds.      ← supersedes #0
```

The task prompt (`utils/templates.py:36`) states the resolution rule explicitly:

> *"Each fact in the knowledge pool is provided with a serial number at the beginning, and the
> newer fact has larger serial number. You need to solve the conflicts of facts in the knowledge
> pool by finding the newest fact with larger serial number."*

**Measured** (my analysis; templates induced from the corpus, 99.5–99.7% parse coverage, grouping
by `(relation, subject)`):

| Subset | Facts | Distinct (rel,subj) keys | Conflicted keys | % keys | Facts in conflict | Group sizes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sh/mh_6k` | 455 | 293 | 160 | 54.6% | 70.3% | all exactly 2 |
| `sh/mh_32k` | 2,310 | 1,467 | 835 | 56.9% | 72.3% | all exactly 2 |
| `sh/mh_64k` | 4,580 | 2,876 | 1,687 | 58.7% | 73.7% | all exactly 2 |
| `sh/mh_262k` | 18,332 | 11,037 | 7,197 | 65.2% | 78.5% | all exactly 2 |

Every conflict group is **exactly size 2** — one original fact, one superseding update. This is a
clean, controlled `(stale, current)` design, not incidental noise.

**Question-level headroom** (single-hop subsets, where a question maps to one fact):

| Subset | Questions on a conflicted key | Gold = LATEST | Gold = EARLIEST |
| --- | ---: | ---: | ---: |
| `sh_6k` | 74% | 74 / 74 (100%) | 0 |
| `sh_32k` | 65% | 63 / 65 (97%) | 2 |
| `sh_64k` | 66% | 64 / 66 (97%) | 2 |
| `sh_262k` | 77% | 73 / 77 (95%) | 4 |

(The handful of EARLIEST cases are almost certainly artifacts of my subject-substring matcher
colliding on a similar subject, not genuine rule violations — the prompt is unambiguous.)

**This is the headline number: 65–77% of questions turn on correctly resolving a stale-vs-current
fact pair.** In BFCL the analogous target classes were 3.5% and 0.7–0.9%. That is a 20–100×
difference in base rate, and it is the entire reason this port is worth doing.

### 4.3 CrossEp-Know evolution is real but uncontrolled

Memory accumulates across 5–12 samples within a `context_id`. Repeated facts arise naturally
(same domain rules restated across episodes), but there is **no ground-truth annotation of which
memories supersede which**. Conflicts are emergent and unlabeled. Useful as a realism check;
unusable as a primary target for supervised failure labeling.

---

## 5. Evaluator

| Suite | Evaluator | Determinism | Counterfactual replay cost |
| --- | --- | --- | --- |
| InEp-Know / Conflict_Resolution | `substring_exact_match` over gold answer list, `utils/eval_other_utils.py:296` | **Fully deterministic, offline, free** | ~0 |
| InEp-Know / EventQA | `eventqa_recall` | deterministic | ~0 |
| InEp-Know / LongMemEval | LLM judge (`llm_based_eval/longmem_qa_evaluate.py`) | stochastic | 1 judge call |
| CrossEp-Know | LLM-as-judge, **binary all-or-nothing over a ~24-rubric checklist**, `eval.py:80` | stochastic, `gpt-5.1` | 1 judge call/sample |

The Conflict_Resolution evaluator being a **free, deterministic string match** is a decisive
practical advantage: counterfactual re-evaluation (§11 of the task) costs nothing but the LLM
answer call, and the labeling pipeline needs no judge at all.

### 5.1 The CrossEp-Know evaluator is hostile to small interventions

Measured on the shipped baseline
(`outputs/context_nomemory/…_ctx_nomemory_graded.jsonl`, N=884, DeepSeek-v3.2, no memory):

- **Binary accuracy = 23.87%** (211/884). Large nominal headroom.
- **Rubric-level pass rate = 78.2%** (16,299/20,831), mean 23.6 rubrics/sample.
- Among the 673 incorrect samples, failed-rubric counts: 1 → 14.1%, 2 → 14.1%, 3 → 11.6%,
  4 → 9.7%, 5 → 8.9%, ≥9 → 23.2%.
- **Only 10.7% of all samples are one rubric away from scoring 1.** Ceiling if *every*
  one-rubric-away case were fixed: 34.6%.

So CrossEp-Know's 76% failure mass is mostly *deep* failure (median ~4 failed rubrics), not
one-fact-away failure. A memory intervention that repairs a single retrieved fact will usually
**not flip the binary score**. This is precisely the BFCL trap — a real signal on decisions that
cannot change the final answer — and it recurs here in a different guise. It is the main reason
CrossEp-Know is secondary rather than primary.

### 5.2 Statistical dependency unit

CrossEp-Know memory is isolated per `context_id` and accumulates serially within it, so samples
within a context are **not independent**. Measured on the baseline:

```
ICC(context)  = 0.346
mean cluster  = 7.36 samples
design effect = 1 + (n₀−1)·ICC = 3.20
effective N   ≈ 276  (not 884)
```

**Any unpaired analysis must use context-clustered standard errors.** Treating 884 samples as
independent overstates precision by ~1.8× in the standard error. For paired arm comparisons on
identical questions, McNemar on discordant pairs is the correct test and recovers much of the power.

For InEp-Know/Conflict_Resolution the unit is cleaner: each of the 8 subsets is one independent
context (one fact store); the 100 questions within a subset share that store. Cluster on
**subset** (8 clusters) for store-level claims, and treat the 800 questions as paired units for
arm-vs-arm comparisons on the same store.

---

## 6. Exact H-Nav hook mapping (the 12 blueprint requirements)

Resolved against the two viable suites. `A` = InEp-Know (`MemoryAgentBench`),
`B` = CrossEp-Know (`CROSSEP-KNOW`).

| # | H-Nav requirement | EvoMemBench file | class / function | Compatible? | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Incoming memory candidate | A `agent.py:967`<br>B `infer_context_memory.py:171` | `AgentWrapper.send_message(memorizing=True)`<br>`Memory.extract(content=…)` | **YES** | Both are single chokepoints. In A the candidate is a 4096-token chunk; in B a formatted trajectory. Sub-candidates (mem0 extracted facts) require hooking `mem0/memory/main.py:794` instead. |
| 2 | Memory-store representation | B `qwen3_embedding_memory.py:67` | `self.memory_bank[]` + `self.embeddings[]`, mirrored to `memory_bank.jsonl` / `embeddings.jsonl` | **YES** | Fields: `task_id`, `chunk_index`, `chunk_text`, `context_category`, `sub_category`. **No timestamp/version field** — must be added (monotone counter is sufficient and is what the benchmark's own rule uses). mem0 payloads do carry `created_at`/`updated_at`/`hash`. |
| 3 | Write-decision hook | B `Memory.extract` (ABC in `base.py`) | subclass wrapper | **YES** | `extract()` returns only a stats dict; the benchmark ignores its content except for token accounting. Suppressing a write is invisible to the harness. `PASS`/`SUPPRESS`/`REWRITE`/`DEFER` are all implementable without touching the runner. |
| 4 | Embedding interface | B `qwen3_embedding_memory.py:140` `_embed`, `:162` `_embed_batch`<br>A `embedding_retriever.py:51,87` | OpenAI-compatible `embeddings.create` or local HF | **YES** | Reusable directly. A separate ABTT-whitened governance space is trivially constructible: fit whitening on the persisted `embeddings.jsonl` and keep it *read-only* w.r.t. native retrieval. |
| 5 | Pre-truncation retrieval scores | B `qwen3_embedding_memory.py:218`<br>mem0 `main.py:1356`<br>A `embedding_retriever.py:196` | full `scores` vector / `internal_limit≥60` pool / **discarded** | **B: YES, mem0: YES, A: needs 1-line change** | A's `similarity_search` → `similarity_search_with_score_by_vector`. No behavioral change if the returned slice is unchanged. |
| 6 | Query construction | B `trajectory.py:get_last_user_message`<br>A `embedding_retriever.py:256` regex | deterministic | **YES** | Both derive the query from information available at decision time only. No leakage. |
| 7 | Context-injection point | B `trajectory.py:inject_memory_into_messages`<br>A `embedding_retriever.py:274` | appends to system message / prepends `Memory i:` block | **YES** | Trim / rerank / expand / annotate are all pure string ops upstream of the LLM call. Evaluator never sees them. `memory_retrieved` is already logged per sample in B's output records. |
| 8 | Evaluator | A `eval_other_utils.py:296`<br>B `eval.py` | `substring_exact_match` / LLM judge | **A: YES (free, deterministic)**<br>**B: yes, but 1 judge call each** | A permits unlimited counterfactual re-evaluation. B does not, at reasonable cost. |
| 9 | No-action detection | A `send_message(memorizing=…)` flag<br>B `retrieve()` returning `""` | explicit | **YES** | B: `retrieve` returns `("" , stats)` when the bank is empty (`qwen3_embedding_memory.py:199`) — a clean no-read signal. First sample in every context is always a no-read. |
| 10 | Store evolution driver | A `data/Conflict_Resolution.json` (engineered)<br>B serial context loop (emergent) | — | **A: YES, labeled. B: yes, unlabeled** | See §4. A is the only place with ground-truth supersession. |
| 11 | Retriever-replica feasibility | B `qwen3_embedding_memory.py:218` | 5-line numpy | **EXACT** | See §7. |
| 12 | Statistical dependency unit | — | — | **RESOLVED** | A: subset (8 stores) / question (800, paired). B: `context_id` (120 clusters, ICC 0.346, deff 3.20). See §5.2. |

---

## 7. Retriever-replica feasibility

| Retriever | Replica fidelity | Effort | Basis |
| --- | --- | --- | --- |
| `Qwen3EmbeddingMemory` (B) | **Bit-exact** (up to `argsort` tie order) | ~5 lines | Scoring is `(bank_matrix @ q) * 100`, embeddings persisted on disk. Nothing hidden. |
| InEp-Know RAG (A) | **Exact**, given the same embedder | small | LangChain FAISS `IndexFlatL2` over L2-normalized vectors; `similarity_search_with_score_by_vector` returns the same ordering the native call slices. Use a *local* embedder (contriever / Qwen3-4B via HF) for determinism. |
| `BM25Memory` (B) | **Exact** | ~10 lines | `BM25Okapi` over whitespace tokens, rebuilt from `memory_bank.jsonl`. |
| mem0 hybrid | **Approximate** without real work; exact is a genuine port | medium–high | Must reproduce `score_and_rank` + `normalize_bm25` + `get_bm25_params` + lemmatizer + entity store. |
| GraphRAG / MemOS / MemoryOS / A-mem | Not attempted | high | Out of scope for Stage 0. |

**Recommendation:** build `RetrieverReplica` against `Qwen3EmbeddingMemory` and the InEp-Know
FAISS path only. These two cover the primary and secondary arenas exactly. Do **not** attempt a
mem0 replica in Stage 0 — if H-Nav needs mem0, run H-Nav *inside* mem0's own search path rather
than replicating it.

Validation protocol (§8 of the task) is cheap here: for B, replay every `(store state, query)` pair
from a real run using the persisted `embeddings.jsonl`, and assert exact rank-list identity against
the live `retrieve()`. Target: 100% agreement, not "near-exact" — anything less indicates a bug,
because the computation is identical by construction.

---

## 8. Geometry compatibility

| H-Nav geometry component | EvoMemBench status |
| --- | --- |
| Benchmark-compatible embedder | **Available.** DashScope `text-embedding-v4` (1024-d) or local Qwen3-Embedding-4B / contriever. Vectors already L2-normalized and persisted. |
| ABTT whitening (separate governance space) | **Constructible.** Fit on the persisted bank embeddings per context/store. Read-only w.r.t. native retrieval — no risk of perturbing benchmark behavior. Caveat: for the 6k stores the bank is small (≈2 chunks in CrossEp-Know terms); whitening statistics will be unstable on small stores. Fit per-subset on InEp-Know where stores are large (455–18,332 facts). |
| `sim_max` | **Direct.** Already computed as part of the score vector. |
| QR residual novelty `r` | **Direct.** Standard numpy on the bank matrix. Cost is `O(N·d)` per candidate; at N=18k, d=1024 this is ~75 MB and milliseconds. Fine. |
| Adaptive `tau_t` | **Portable as a mechanism**, but **BFCL's numeric calibration must not be reused.** Score scale differs (`cosine×100` vs whatever BFCL used) and the duplicate base rate differs by ~20×. Re-fit on a designated dev split. |
| Exact-duplicate fast path | **Redundant with mem0's MD5 check, additive elsewhere.** In `Qwen3EmbeddingMemory` and `BM25Memory` there is *no* dedup at all, so an exact-duplicate path has real work to do there. Measured: 0 exact-duplicate fact lines in Conflict_Resolution (all 455–18,332 facts are unique strings) — so exact-dup is **inert on the primary arena** and only matters for CrossEp-Know trajectory chunks. |
| Verbatim-value guard | **Highly relevant.** See §9 — the differing span is a bare entity name ("Leeds", "Harvard University"). A verbatim-value guard is exactly the mechanism that must fire to prevent a superseding fact being suppressed as a near-duplicate. |
| Retrievability floor | **Relevant and testable.** With top-k=10 over 67 chunks (262k), a suppressed-or-demoted memory that later becomes the answer is a measurable harm. The floor is the right guard. |

---

## 9. Marginal-diff compatibility — the strongest result in this analysis

This is where EvoMemBench most clearly differs from BFCL. Measured on all 7,197 real conflict
pairs in `factconsolidation_sh_262k`, using a **char-3gram cosine as a lexical proxy** for
whole-blob similarity (see caveat below):

| Quantity | mean | p10 | p50 | p90 |
| --- | ---: | ---: | ---: | ---: |
| **Whole-blob** similarity (full fact vs full fact) | **0.760** | 0.643 | 0.771 | 0.860 |
| **Marginal-diff** similarity (differing object span only) | **0.059** | 0.000 | 0.000 | 0.250 |

- Object-span token Jaccard: mean 0.025; **89.4% of pairs share zero tokens in the changed span.**
- **Critical-delta cases** (whole-blob ≥ 0.80 **and** diff ≤ 0.30): **2,376 = 33.0% of conflict pairs.**

Worked examples, taken verbatim from the data:

```
whole=0.949  diff=0.775
  old #4731 : The univeristy where Isaiah Washington was educated is Howard University.
  new #7334 : The univeristy where Isaiah Washington was educated is Harvard University.

whole=0.944  diff=0.000
  old #4469 : The headquarters of Italia Conti Academy of Theatre Arts is located in the city of London.
  new #12238: The headquarters of Italia Conti Academy of Theatre Arts is located in the city of Paris.
```

The first is the canonical adversarial case: **one letter** separates the two facts, whole-blob
similarity 0.949 — any similarity-threshold dedup would suppress the update — and the suppressed
token *is the gold answer*. The second is the canonical semantic case: identical framing, disjoint
objects.

**Caveat, stated plainly:** char-3gram cosine is a *lexical proxy*, not the embedding cosine H-Nav
actually uses. I could not compute real embeddings in this environment (no API credentials, no
local model). The proxy establishes that these pairs are near-identical *in surface form* and that
the changed span is disjoint — conditions under which any competent sentence embedder will place
them close. **Measuring the real `sim_max` distribution under `text-embedding-v4` / Qwen3-4B is
Stage-0 deliverable M1 and is a prerequisite for setting any threshold.** Do not port BFCL's
numbers.

**Why this matters for the thesis:** BFCL's H2 (marginal-diff geometry adds information beyond
whole-blob geometry) reportedly *passed as a signal* but had no downstream headroom. Here the
signal has 33% prevalence among conflict pairs, and 65–77% of questions ask about exactly these
pairs. If marginal-diff geometry is ever going to be outcome-critical, this is the dataset.

---

## 10. Retrieval-signal compatibility

Using the native retriever properties from §3:

| Signal | Feasible? | Notes |
| --- | --- | --- |
| `rank_self` | **Yes** | Full score vector available (B) or via `_with_score` (A). |
| top1−top2 margin | **Yes** | Direct. |
| normalized margin | **Yes** | Needed: raw cosine×100 differences are small and scale-dependent. |
| Raw-score softmax entropy | **Feasible but expected to be degenerate** | Scores are `cosine×100 ∈ [0,100]`; a softmax over values that differ by ~1–5 units is near-uniform, and over ×100-scaled values is near-one-hot. Exactly the scale-degeneracy BFCL found. **Do not use raw-score entropy as the decision mechanism.** |
| z-scored entropy | **Yes — preferred** | Standardize scores within the candidate set first. |
| von Neumann neighborhood entropy | **Yes** | Requires the Gram matrix of the top-m neighborhood. At m≈50 this is a 50×50 eigendecomposition per query — negligible. |
| `dH_self` / `dH_neighbor` across a provisional insert | **Yes** | The replica supports simulated inserts trivially: append a row to `bank_matrix` and rescore. This is the strongest argument for building the replica first. |
| Retrieval churn in top-k | **Yes** | Compare top-k sets before/after provisional insert. |
| Neighborhood effective size / dispersion | **Yes** | Standard. |

**Retrieval-interference prevalence, measured.** For chunked RAG on Conflict_Resolution
(chunk_size=4096 tokens ⇒ ~274–289 facts/chunk), how often are the stale and current versions of a
fact in *different* chunks — i.e. how often is getting the right answer a retrieval-navigation
problem rather than a reading problem?

| Subset | Facts | Est. chunks | Conflict pairs | Median index gap | Same chunk | **Different chunk** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sh_6k` | 455 | 2 | 160 | 147 | 50.6% | 49.4% |
| `sh_32k` | 2,310 | 8 | 835 | 640 | 13.3% | **86.7%** |
| `sh_64k` | 4,580 | 16 | 1,687 | 1,266 | 6.1% | **93.9%** |
| `sh_262k` | 18,332 | 67 | 7,197 | 5,280 | 1.5% | **98.5%** |

At 262k, **98.5% of conflict pairs span different chunks, with top-k=10 out of ~67 chunks.** If the
retriever surfaces the stale chunk and not the superseding one, the answer is wrong with certainty
and the LLM has no way to recover — the serial-number rule it was given is unusable when the newer
fact is absent from context. This is `READ_STALE` / `READ_RELEVANT_BELOW_K` with near-total
prevalence, and it is **directly actionable**: a read-side policy that detects a conflicted
`(relation, subject)` in the retrieved set and pulls in the highest-serial variant is a concrete,
cheap, testable intervention.

This is the second strong H-Nav opportunity, and unlike the write side it does not depend on the
store being mutable.

---

## 11. Failure-class availability

Measured or firmly inferred prevalences. **A** = InEp-Know/Conflict_Resolution, **B** = CrossEp-Know.

### Write side

| Label | A | B | Basis |
| --- | --- | --- | --- |
| `WRITE_REDUNDANT` | low | **high** | A: all facts are unique strings (0 exact dups). B: trajectory chunks across 5–12 same-context episodes restate domain rules; native store has **zero** semantic dedup. |
| `WRITE_CRITICAL_DELTA` | **very high** — 33% of conflict pairs are whole≥0.80 ∧ diff≤0.30 | unknown/unlabeled | §9 |
| `WRITE_CONFLICT` | **very high** — 54.6–65.2% of keys | emergent, unlabeled | §4.2 |
| `WRITE_STALE_SUPERSEDE` | **very high** — every conflict pair is exactly (stale, current) | unlabeled | §4.2 |
| `WRITE_DESTRUCTIVE_OVERWRITE` | **absent** | **absent** | No write path in EvoMemBench ever overwrites. Append-only by construction (§4.1). **DROP this label.** |
| `WRITE_DUPLICATE` (exact) | **absent** (0/18,332) | low | mem0's MD5 check already handles it where it occurs. |
| `WRITE_UNNECESSARY` | n/a | moderate | Store growth is unbounded; no eviction anywhere. |
| `WRITE_MISSED_UPDATE` | n/a | moderate | LLM extraction may drop facts; only offline-detectable. |

### Read side

| Label | A | B | Basis |
| --- | --- | --- | --- |
| `READ_CLEAR` | 23–35% (questions on unique facts) | majority | §4.2 |
| `READ_CONFLICT` | **65–77% of questions** | emergent | §4.2 |
| `READ_STALE` | **86.7–98.5% of conflict pairs are cross-chunk** | unlabeled | §10 |
| `READ_RELEVANT_BELOW_K` | **high** — top-10 of ~67 chunks | measurable | §10 |
| `READ_AMBIGUOUS` (low margin) | measurable, needs real embeddings | measurable | Stage-0 M1 |
| `READ_DISTRACTOR` | **high by construction** — the stale twin is a maximally strong distractor | moderate | §9 |
| `READ_HIGH_INTERFERENCE` | **high** — stores of 455–18,332 near-templated facts | low (small stores) | §10 |
| `READ_MISSING` | low | high (first sample in each context always has empty memory) | `qwen3_embedding_memory.py:199` |

**Bottom line:** EvoMemBench does not merely contain H-Nav's target failure classes — one subset is
*constructed out of them*, at base rates 20–100× BFCL's, with ground-truth supersession labels and
a free deterministic evaluator.

---

## 12. Benchmark vs model bottleneck

The task (§20) rightly warns against claiming a general "LLM geometry bottleneck." The measured
structure here lets us separate the layers cleanly, which is itself a reason to prefer this arena:

| Layer | Can it be isolated? | How |
| --- | --- | --- |
| **Benchmark/task structure** | Yes | The conflict structure is static and fully enumerable offline, independent of any model. Already done (§4.2). |
| **Memory/write system** | Yes | Compare `Qwen3EmbeddingMemory` (no dedup) vs mem0 (MD5 + LLM-judgment dedup) on identical data. |
| **Retriever** | **Yes, cleanly** | The long-context agent sees *all* facts with no retrieval at all. The RAG agent sees top-k chunks. **Long-context accuracy − RAG accuracy on conflicted questions isolates the retrieval bottleneck exactly.** This contrast is already supported by the repo's own configs (`Long_Context_Agents/` vs `RAG_Agents/`) and requires no new code. |
| **LLM behavior** | Yes | Within the long-context arm, failures on conflicted questions are pure reasoning failures (all facts present, rule stated in the prompt). |

That ladder — *facts present & rule stated* → *facts retrieved* → *facts written* — is unusually
clean, and it means a positive H-Nav result here can be attributed to a specific layer rather than
asserted generically. **Any claim must be conditioned on (benchmark subset, memory backend, model).**
Cross-model validation (≥2 model families) is required before any layer-general claim; do not run it yet.

---

## 13. Unresolved issues

1. **The blueprint is missing (§0).** All H-Nav-internal specifics are reconstructed from the task
   description. Re-verify before writing `HNavCore`.
2. **No real embeddings were computed.** §9's separation uses a lexical proxy. The real `sim_max`,
   margin, and entropy distributions under `text-embedding-v4` / Qwen3-Embedding-4B are unmeasured.
   **This is Stage-0 deliverable M1 and gates every threshold.**
3. **Non-determinism.** `temperature` is 0.7–1.0 across `configs/agent_conf/RAG_Agents/*`
   (`Embedding_rag_*` = 0.7, `Simple_rag_bm25`/`graph_rag`/`mem0` = 1.0). Seeds exist in the data
   configs (`seed: 42`) but do not control API sampling. **Set `temperature=0` for all H-Nav arms,
   or budget ≥3 replicates for the A0′ variance arm.** This is not optional: the CrossEp-Know
   binary metric is high-variance, and an uncontrolled 0.7-temperature baseline cannot support a
   2pp effect claim.
4. **CrossEp-Know's evaluator dilutes interventions** (§5.1): median 4 failed rubrics among
   incorrect samples; only 10.7% are one-rubric-away. Consider pre-registering the **rubric-level
   pass rate** as a secondary endpoint there — it is strictly more sensitive and is already
   recorded per-sample in `requirement_status`.
5. **`np.argsort` tie instability** (`qwen3_embedding_memory.py:221`). Measure-zero in practice for
   float cosines, but the replica must document it rather than silently differ.
6. **Multi-hop (`mh_*`) subsets are harder to label.** A multi-hop question traverses ≥2 facts, so
   "the question targets conflicted key K" is not a clean 1:1 map. My question-level headroom
   numbers are measured on `sh_*` only. **Restrict primary analysis to `sh_*` (400 questions);
   treat `mh_*` as exploratory.**
7. **Store-size imbalance.** The 6k subsets have 455 facts and ~2 chunks — retrieval is nearly
   trivial there (49% of conflict pairs co-located, top-10 covers everything). Real retrieval
   headroom lives in 32k/64k/262k. Report stratified by store size; do not pool.
8. **mem0's `retrieve_num=100`** vs 10 elsewhere makes cross-backend comparison unfair unless
   controlled. Fix top-k across arms or report it as a covariate.
9. **CrossEp-Know memory content is model-generated** (`format_trajectory(messages, response_text)`),
   so the store differs between arms even before H-Nav acts. Write-side interventions therefore
   perturb the store *and* the baseline drifts. Pair on question, not on store state.

---

## 14. Suite-level recommendation

| Suite | Recommendation | Reason |
| --- | --- | --- |
| **InEp-Know / Conflict_Resolution** | **PRIMARY** | 800 questions; 55–65% conflicted keys; 65–77% of questions turn on stale-vs-current; ground-truth supersession by construction; free deterministic evaluator; exact retriever replica; clean benchmark/retriever/model attribution ladder. |
| **CrossEp-Know** | **SECONDARY** | Cleanest API in the repo (`Memory` ABC); genuinely evolving append-only store; exact replica for `qwen3_embedding_4b`. But: unlabeled conflicts, all-or-nothing 24-rubric evaluator, ICC 0.346 (effective N≈276), LLM-judge cost per counterfactual. Good for *realism* and for write-side redundancy; weak for confirmatory statistics. |
| InEp-Know / Accurate_Retrieval | OPTIONAL | Useful as a **negative control arena**: retrieval stress without engineered conflicts. Good for measuring H-Nav's harm rate where it should do nothing. |
| **InEp-Exec**, **CrossEp-Tool** | **NO_GO** | Both are BFCL. Re-running H-Nav on the substrate that already produced the null result adds no information and consumes the budget. |
| CrossEp-Web | NO_GO | Live web/Serper dependency; not replayable; no stable store. |
| CrossEp-Emb | NO_GO | ALFWorld; no retrieval store of the required form. |

---

## 15. Summary of measured facts

Everything below was computed from the repository in this session; scripts are reproduced in the
implementation plan (§8 of that document).

```
Conflict_Resolution:  8 subsets × 100 questions = 800
  template parse coverage                      99.5 – 99.7 %
  conflicted (relation,subject) keys           54.6 – 65.2 %
  facts living inside a conflict pair          70.3 – 78.5 %
  conflict group size                          exactly 2, always
  questions targeting a conflicted key (sh_*)  65 – 77 %
  gold = latest serial number                  95 – 100 %
  whole-blob sim (lexical proxy), mean         0.760  (p90 0.860)
  marginal-diff sim, mean                      0.059  (89.4 % zero-overlap)
  critical-delta (whole≥.80 ∧ diff≤.30)        33.0 % of conflict pairs
  conflict pairs spanning different chunks     49 % (6k) → 98.5 % (262k)

CrossEp-Know baseline (deepseek-v3.2, no memory, N=884):
  binary accuracy                              23.87 %
  rubric-level pass rate                       78.24 %  (23.6 rubrics/sample)
  one-rubric-away from correct                 10.7 % of all samples
  ICC(context) / design effect / effective N   0.346 / 3.20 / ≈276

mem0 write path                                APPEND-ONLY; dedup = exact MD5 only
```

For comparison, the BFCL base rates given in the task: `must_write` ≈ 3.5%, `must_suppress` ≈ 0.7–0.9%.
