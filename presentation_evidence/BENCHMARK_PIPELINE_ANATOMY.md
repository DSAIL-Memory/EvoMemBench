# Anatomy of the Benchmark Pipeline — and Where H-Nav Plugs In

*A walkthrough of how one INEP-KNOW / Conflict_Resolution benchmark run
actually works: what data goes in, what code runs, what the model sees, how its
output is scored, where every file lands — and, in detail, where and how H-Nav
attaches to this pipeline. Written for a reader who has not seen the codebase.*

**Provenance.** Written 2026-08-18, revised 2026-08-19, entirely offline.
Every example below is a real record read from a file committed in this
repository (the reading scripts are in `presentation_evidence/_scripts/`).
Nothing is paraphrased from memory; each stage cites the source file and line
numbers so you can open the exact spot. The two scoring functions were
transcribed verbatim and re-executed locally on the real records to confirm
that the recorded verdicts reproduce.

**How to read the 🔎 boxes.** Each stage ends with a *data journey* box showing
what one real question — sh_6k, question index 1, `factconsolidation_sh_6k_no1`
— looked like as it passed through that stage. The boxes are generated offline
by `presentation_evidence/_scripts/make_data_journey.py` and carry provenance
labels:

- **COMMITTED** — quoted verbatim from a file in this repository.
- **RECONSTRUCTED** — rebuilt offline from committed inputs plus the
  repository's own code; any deviation is stated.
- **NOT IN REPOSITORY** — existed only on the GPU box.

---

## 0. The cast, in one table

| Component | What it is | Where |
|---|---|---|
| Dataset | `Conflict_Resolution.json` (~3.1 MB, JSON list) | `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/` |
| Entry point | `main.py` — the evaluation loop | `.../MemoryAgentBench/main.py` |
| Data prep | `ConversationCreator` — templating + chunking | `conversation_creator.py` |
| Agent | `AgentWrapper` — routes memorize/query calls | `agent.py` |
| Retriever | `TextRetriever` + `RAGSystem` (FAISS + LLM call) | `methods/embedding_retriever.py` |
| Prompts | system message + memorize/query templates | `utils/templates.py:2, 36-44` |
| Scoring | normalization + substring match + F1/ROUGE | `utils/eval_other_utils.py` |
| Answering LLM | Qwen3-4B-Instruct-2507, served by vLLM (OpenAI-compatible) | `:8000`/`:8003` on the GPU box |
| Embedder | Qwen3-Embedding-4B, served by vLLM | `:8001` on the GPU box |
| Agent config used in our runs | `Embedding_rag_local-qwen-qwen3_embedding_4b.yaml` — `temperature: 0`, `retrieve_num: 10` | `configs/agent_conf/RAG_Agents/local-qwen/` |
| Dataset config | e.g. `Factconsolidation_sh_6k.yaml` — `chunk_size: 4096`, `generation_max_length: 10` | `configs/data_conf/Conflict_Resolution/` |
| H-Nav hooks | 2 guarded edits in this arena (of 4 repo-wide), no-ops at `HNAV_MODE=off` | `agent.py:988-1020`, `embedding_retriever.py:217-258` |

Launch command (per subset, per arm):

```bash
python main.py \
  --agent_config  configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml
```

---

## 1. Stage 1 — the input data

`Conflict_Resolution.json` is a list of context items. Each item has:

- **`context`** — one long string: a preamble (`Here is a list of facts:`)
  followed by numbered facts, one per line. A later serial number with the
  same *subject + relation* **supersedes** an earlier one. That is the entire
  test.
- **`questions`** / **`answers`** — parallel lists; the gold answer is always
  the value of the **highest-serial** fact of the queried key.
- **`metadata.qa_pair_ids`** — e.g. `factconsolidation_sh_6k_no0`; the YAML's
  `sub_dataset` field selects the item.

Real record (entry index 4, the sh_6k item — 455 facts, 100 questions):

```
context (excerpt):
  ...
  91. Nobuhiro Watsuki is famous for Rurouni Kenshin.        <- the stale fact
  ...
  259. Nobuhiro Watsuki is famous for The Fairly OddParents. <- supersedes it
  ...
questions[1]: "What is Nobuhiro Watsuki famous for?"
answers[1]:   ["The Fairly OddParents"]
```

Scale of the subsets we used: sh_6k = 455 facts, sh_32k = 2,310, sh_64k =
4,580, sh_262k = 18,333 — 100 questions each (fact counts per
`stage0_results/final/m1b_grouping_ablation.json` → `n_facts`). The facts are
deliberately **counterfactual** (elsewhere in this item: "The chief executive
officer of Microsoft is Steve Jobs."), so the model cannot fall back on world
knowledge — it must read the page.

⚠ The questions, answers, and contexts live in the **same file**. Any online
component that opens this file could trivially leak gold answers — this is why
H-Nav's leakage rule (brief §1) exists and is enforced by an AST scan
(`hnav/tests/test_leakage_audit.py`).

> ### 🔎 Data journey — Stage 1: dataset entry  [COMMITTED]
>
> `In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json`,
> the entry with `metadata.qa_pair_ids[0] = "factconsolidation_sh_6k_no0"`:
>
> ```
> context, line for serial 91 : Nobuhiro Watsuki is famous for Rurouni Kenshin.
> context, line for serial 259: Nobuhiro Watsuki is famous for The Fairly OddParents.
> questions[1]                : What is Nobuhiro Watsuki famous for?
> answers[1]                  : ['The Fairly OddParents']
> ```

---

## 2. Stage 2 — data preparation (`ConversationCreator`)

`main.py:197` constructs `ConversationCreator`, which does two things **before
any model is involved**:

**(a) Query templating** (`conversation_creator.py:184-190`). Every raw
question is embedded into an instruction template (`utils/templates.py:41`,
the `rag_agent` entry for `factconsolidation`). From this point on, "query" in
the code means the full 791-character instruction block, not the bare
question. Two things about this template matter:

- It *already tells the model the resolution rule*: "the newer fact has larger
  serial number … finding the newest fact". The model is not asked to discover
  the rule; it is asked to apply a stated one.
- It *pre-empts world knowledge*: answer "**only** from the knowledge pool …
  rather than the real facts in real world" — and the one-shot example's
  answer is deliberately false (Donald Trump as president of Russia), to
  demonstrate that the pool outranks reality.

**(b) Chunking** (`conversation_creator.py:261-275` →
`utils/eval_other_utils.py:173-222`). The context string is split into
sentences with NLTK `punkt`, and the sentences are packed greedily into chunks
of at most 4096 tiktoken tokens. Two consequences worth knowing:

- Sentences are re-joined **with spaces**, so the "one fact per line"
  structure of the raw context is destroyed inside a chunk. (This is why
  H-Nav's line-anchored fact regex matches nothing on a real chunk and a
  fallback inline regex exists — `hnav/adapters/mab_adapter.py`, pinned by
  `hnav/tests/test_chunking_and_facts.py`.)
- Chunk count per subset (committed in
  `stage0_results/final/m2_retrieval_calibration.json`): sh_6k → 2, sh_32k →
  9, sh_64k → 17, sh_262k → 67. Compare these with `retrieve_num: 10` in
  Stage 4: on sh_6k everything always fits on the retrieved page; on sh_64k
  and sh_262k it cannot.

> ### 🔎 Data journey — Stage 2a: the templated query  [COMMITTED]
>
> The full templated query is stored per question in every committed run
> file — `stage0_results/t4_s2_evidence/sh_6k_off_results.json` →
> `data[1].query` (791 chars, quoted in full):
>
> ```
> Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. 
>  You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. 
> 
> For example:
> 
>  [Knowledge Pool] 
> 
>  Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? 
> Answer: Donald Trump 
> 
>  Now Answer the Question: Based on the provided Knowledge Pool, What is Nobuhiro Watsuki famous for? 
> Answer:
> ```
>
> ### 🔎 Data journey — Stage 2b: chunking  [statistics COMMITTED; chunk texts NOT IN REPOSITORY]
>
> The chunk texts are transient — built in memory on each run and never saved.
> What the repository does hold is the per-subset chunking record, in
> `stage0_results/final/m2_retrieval_calibration.json`, including the
> `fallback_chunker` flag proving the benchmark's own NLTK chunker ran (not a
> substitute):
>
> ```
> subset     n_chunks  fallback_chunker  top_k
> sh_6k             2             False     10
> sh_32k            9             False     10
> sh_64k           17             False     10
> sh_262k          67             False     10
> ```
>
> (The confirmatory artifact independently records `n_chunks_total: 17`,
> `n_chunks_on_page: 10` for sh_64k. Chunk boundaries cannot be regenerated on
> this machine — nltk punkt data and tiktoken are not installed locally — but
> for sh_6k both chunks are always retrieved, so no boundary matters
> downstream.)

---

## 3. Stage 3 — memorization (how data enters the "memory")

Per context item, `initialization.py:145-172` creates the agent and feeds it
each chunk once:

```python
for chunk in context_chunks:                       # initialization.py:359
    agent.send_message(chunk, memorizing=True, ...)
```

For the RAG agent (`agent.py:1463-1478`), memorizing a chunk means:

1. Wrap it in the **memorize template** (`utils/templates.py:38`), which
   inserts a real wall-clock timestamp:

   ```
   Dialogue between User and Assistant 2026-08-14 21:03:11 \n<User> The
   following context is the facts I have learned:
   Here is a list of facts: 0. Thomas Kyd was born in the city of London.
   1. The chairperson of Fatah is Mahmoud Abbas. 2. ... [~4096 tokens]
    <Assistant> I have learned the facts and I will answer the question you ask.
   ```

2. Append the wrapped chunk to a plain Python list, `self.chunks`. **That
   list is the entire memory.** No embedding, no LLM call, no summarization
   happens at memorize time for this agent — `token_stats` in the run file
   confirms it: `total_memorization_input_tokens: 0`.

The agent state is saved to disk after memorization (`initialization.py:170`,
`agent.save_agent()`), so repeated runs over the same context reload instead
of re-memorizing.

> ### 🔎 Data journey — Stage 3: the memorize-wrapped chunk  [template COMMITTED; wrapped text NOT IN REPOSITORY]
>
> Each chunk is wrapped in the memorize template (`utils/templates.py:38`)
> before storage. The wrapped strings are not saved anywhere — and since they
> contain a wall-clock timestamp, they differ on every run. The template,
> verbatim:
>
> ```
> Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.
> ```
>
> Evidence that memorization makes zero LLM calls: the committed run file's
> `token_stats.total_memorization_input_tokens` = 0.

---

## 4. Stage 4 — retrieval (per question)

`main.py:119` sends each templated query with `memorizing=False` →
`_handle_rag_agent` → `_process_rag_query` (`agent.py:1483`) →
`_handle_embedding_rag` (`agent.py:1699`).

**(a) The vector store, built once per context** (`agent.py:1717-1725`). The
agent name contains `rag_qwen3_embedding_4b`, so `embedding_model_name =
"Qwen/Qwen3-Embedding-4B"` (`agent.py:1710-1711`). Although an in-process
transformers class exists in the same file, this name is routed to LangChain
`OpenAIEmbeddings` pointed at the `OPENAI_BASE_URL` read from the
**directory-local `.env`** (`embedding_retriever.py:176-183`) — i.e. the
`:8001` embedding server. Each memorize-wrapped chunk is embedded once and the
vectors go into a **FAISS flat index** (exact search; `build_vectorstore`,
`:196-205`).

**(b) Retrieval-query extraction** (`embedding_retriever.py:314-322`). The
query that gets *embedded* is not the full templated prompt — a regex takes
everything after `"Now Answer the Question:"`:

```
Retrieval query: "Based on the provided Knowledge Pool, What is Nobuhiro
Watsuki famous for? \nAnswer:"
```

**(c) Search** (`TextRetriever.retrieve`, `:207-258`). FAISS similarity
search returns the **top `retrieve_num = 10` chunks**, in similarity-rank
order (scores are squared L2 over normalized embeddings). The returned set of
chunks is called the **page** throughout this document — it is the only
context the answering model will ever see. **← H-Nav hook 2 lives here**
(§9).

Consequence: on sh_6k both chunks always fit on the page — retrieval is
complete and the page equals the whole context. On sh_64k only 10 of the 17
chunks fit — retrieval is **incomplete**, which bounds both what the model can
answer and what any intervention can achieve. The confirmatory artifact
records this explicitly: `n_chunks_total: 17`, `n_chunks_on_page: 10`,
`retrieval_complete: false` (also a stated limitation in
`HNAV_FINAL_REPORT.md` §11).

> ### 🔎 Data journey — Stage 4a: the retrieval query  [RECONSTRUCTED, exact]
>
> The benchmark extracts the embedding query from the templated query with
> the regex at `embedding_retriever.py:314`. Re-applying that regex to the
> committed query string (pure `re`, deterministic — no model involved):
>
> ```
> Based on the provided Knowledge Pool, What is Nobuhiro Watsuki famous for? 
> Answer:
> ```
>
> ### 🔎 Data journey — Stage 4b: the retrieved page  [NOT IN REPOSITORY; sh_6k content equivalent COMMITTED]
>
> The benchmark writes each retrieved page to `outputs/rag_retrieved/...` on
> the GPU box and then drops it from the result record (`agent.py:1518-1526`)
> — those files were never committed. For **sh_6k** this loses nothing: the
> store is 2 chunks and `retrieve_num = 10`, so the page always contains the
> whole context. The probe harness's whole-context page IS byte-exactly
> reproducible offline — see
> `presentation_evidence/data/item01_page_excerpt.txt` (455 facts, verified
> identical to the dataset's `context` field). For sh_64k the page exists only
> on the box (`detector_inputs.prepass` in the confirmatory artifact).
>
> Retrieval evidence that IS committed: `input_len` per question (page + query
> token count) — for this question, `data[1].input_len` = 6711.

---

## 5. Stage 5 — what is fed into the model

`RAGSystem.answer_query` (`embedding_retriever.py:332-350`) assembles exactly
two chat messages.

**The system prompt — yes, there is one** (`utils/templates.py:2`, fetched at
`agent.py:1744`):

> `You are a helpful assistant that can read the context and memorize it for future retrieval.`

**The user message** = the retrieved chunks glued into `Memory i:` blocks,
then the templated query appended (`embedding_retriever.py:332-333`):

```
Memory 1:
Dialogue between User and Assistant 2026-08-14 ... <User> The following context
is the facts I have learned: Here is a list of facts: 0. Thomas Kyd was born in
the city of London. ... 91. Nobuhiro Watsuki is famous for Rurouni Kenshin. ...
 <Assistant> I have learned the facts and I will answer the question you ask.
Memory 2:
Dialogue between User and Assistant ... 259. Nobuhiro Watsuki is famous for The
Fairly OddParents. ...
Pretend you are a knowledge management system. Each fact in the knowledge pool
is provided with a serial number at the beginning, and the newer fact has
larger serial number. ...
 Now Answer the Question: Based on the provided Knowledge Pool, What is
Nobuhiro Watsuki famous for?
Answer:
```

Both the stale fact (#91) and the superseding fact (#259) are on the page
together — retrieval did not fail; the conflict *arrived intact*. This is
evidence item 1 of the presentation pack.

**The API call** (`embedding_retriever.py:345-350`): one plain OpenAI-client
`chat.completions.create` with `model = Qwen/Qwen3-4B-Instruct-2507`,
`temperature = 0` (our config; the benchmark's shipped deepseek configs use
0.7 — pinning 0 is what makes A/A repeats and byte-identity checks
meaningful), and `max_tokens = generation_max_length = 10`. **Ten tokens.**
The model cannot explain, hedge, or reason out loud; it must emit a short
phrase.

Measured size, from the committed run record: `input_len: 6711` tokens for
the sh_6k Nobuhiro question (retrieved page + templated query).

> ### 🔎 Data journey — Stage 5: the prompt sent to the model  [hash COMMITTED; probe text RECONSTRUCTED]
>
> The benchmark run keeps no prompt text, only sizes. The Stage-1 probe run
> of this same question keeps a SHA-1 prefix of every arm's exact prompt
> (`stale_suppression_probe_sh6k.json` → `per_question[1].arms.*.prompt_sha`):
>
> ```
> native           prompt_sha=b97826cc85c85bc2  n_facts=455
> native_repeat    prompt_sha=b97826cc85c85bc2  n_facts=455
> oracle_suppress  prompt_sha=1e25e93d53184ed5  n_facts=454
> oracle_recency   prompt_sha=52212d7a755c9d1c  n_facts=455
> anti             prompt_sha=6217dc0571cb1a54  n_facts=455
> ```
>
> The probe's native prompt is reproducible byte-exactly offline
> (`render_context` + `build_prompt`; head/tail quoted in
> item01_page_excerpt.txt).

---

## 6. Stage 6 — how the model processes it

One forward pass. No tools, no retries, no chain-of-thought (the 10-token
budget forbids it). The task decomposes into: (a) find the queried key's facts
among hundreds on the page, (b) apply the stated serial-recency rule, (c) emit
the value only.

What actually happens, per the committed evidence: step (a) succeeds and step
(b) fails. Of 575 errors across the 8 committed sh_6k runs, **572 name a
superseded value of the correct key** (`stage0_results/question_strata.json`)
— the model finds the right subject but picks an old value. The position
experiments show the answer follows *where the fact sits in the text* far more
than the serial-number rule (`fig05a`/`fig05b` in the evidence pack).

The real failure, from `stage0_results/t4_s2_evidence/sh_6k_off_results.json`,
`data[1]`:

```json
"output":       "Rurouni Kenshin and The Fairly",
"answer":       ["The Fairly OddParents"],
"input_len":    6711,
"output_len":   10,
"substring_exact_match": false
```

Note the shape of this failure: the model saw **both** values, started listing
them, and the 10-token cap cut it off mid-phrase. In the oracle probe's rerun
of the same question (same prompt contract, one Memory block) the native
answer is a clean `"Rurouni Kenshin"` — the stale value alone.

> ### 🔎 Data journey — Stage 6: the model output  [COMMITTED, twice]
>
> Benchmark run (`sh_6k_off_results.json` → `data[1]`):
>
> ```
> output       : 'Rurouni Kenshin and The Fairly'
> parsed_output: 'Rurouni Kenshin and The Fairly'
> output_len   : 10 tokens (cap: generation_max_length = 10)
> ```
>
> Probe run, all five arms (`stale_suppression_probe_sh6k.json` →
> `per_question[1]`):
>
> ```
> native           -> 'Rurouni Kenshin'            correct=False
> native_repeat    -> 'Rurouni Kenshin'            correct=False
> oracle_suppress  -> 'The Fairly OddParents'      correct=True
> oracle_recency   -> 'Rurouni Kenshin'            correct=False
> anti             -> 'Rurouni Kenshin'            correct=False
> ```

---

## 7. Stage 7 — parsing and scoring

Back in `main.py:122` → `metrics_summarization`
(`utils/eval_other_utils.py:463`) → `post_process` (`:337`).
`factconsolidation_*` matches no special case, so `default_post_process`
(`:435`) runs:

1. **Parsing.** `parse_output` (`:140`) strips an `Answer:` prefix / takes
   the first line. Every metric is computed on **both** the raw output and
   the parsed output, and the **max** is kept — formatting is never punished.
2. **Normalization** (`normalize_answer`, `:28-44`): lowercase → remove all
   punctuation → remove the articles a/an/the → collapse whitespace.
3. **The metric that counts** — `substring_exact_match` (`:101-112`):
   `normalize(gold) in normalize(prediction)`, max over the gold list
   (`:115`). Deterministic, offline, no LLM judge.

Re-executed on the real record above (functions transcribed verbatim and run
— `presentation_evidence/_scripts/`):

```
pred: 'Rurouni Kenshin and The Fairly'   -> norm: 'rurouni kenshin and fairly'
gold: 'The Fairly OddParents'            -> norm: 'fairly oddparents'
'fairly oddparents' in 'rurouni kenshin and fairly'  ->  False   ✗
(recorded verdict in the artifact: false — reproduces exactly)
```

A correct one for contrast, same file, `query_id 3`:

```
pred: 'Shahnameh'  gold: ['Shahnameh']  ->  True   ✓
```

Also computed and stored per question (but not headline): `exact_match`,
token-level `f1` (0.333 for the Nobuhiro row), `rougeL_f1` / `rougeLsum_f1`
(0.5), and their recalls. The headline number everywhere in this project is
`substring_exact_match` — a question is right iff the normalized gold string
appears inside the normalized 10-token output.

Two sharp edges to admit if asked: an answer that lists *both* values in full
would score correct (mostly neutralized by the 10-token cap, though the cap
can also *create* failures, like the truncation above); and the evaluator
checks the string, not the provenance — a right string produced for a wrong
reason still scores.

> ### 🔎 Data journey — Stage 7: scoring  [verdict COMMITTED; recomputation reproduces it]
>
> `substring_exact_match` = normalize(gold) in normalize(prediction), where
> normalize = lowercase, strip punctuation, drop a/an/the, collapse spaces
> (`utils/eval_other_utils.py:28-44, 101-112`). Re-executed on the committed
> row:
>
> ```
> norm(output) = 'rurouni kenshin and fairly'
> norm(gold)   = 'fairly oddparents'
> substring    -> False        recorded verdict: False  (reproduces)
> other stored metrics: exact_match=False, f1=0.333, rougeL_f1=0.500
> ```

---

## 8. Stage 8 — where the outputs land

Per query (frequency = 1, `main.py:156`), the whole state is rewritten to one
JSON in the agent config's `output_dir`:

```
outputs/<output_dir>/..._<sub_dataset>_.json      (on the GPU box, gitignored)
├── agent_config      (the exact YAML used, embedded)
├── dataset_config    (ditto)
├── data[]            one record per question:
│     output, parsed_output, answer, query (full templated prompt),
│     query_id, qa_pair_id, input_len, output_len,
│     exact_match, f1, substring_exact_match, rouge*,
│     memory_construction_time, query_time_len
├── metrics           per-metric lists over all questions
├── averaged_metrics  means ×100  (e.g. substring_exact_match: 29.0 = 29%)
└── token_stats       total input/output tokens, avg inference time
```

The retrieved page for each query is additionally written to
`outputs/rag_retrieved/<agent>/k_10/<sub_dataset>/chunksize_4096/query_<q>_context_<c>.json`
(`agent.py:1518-1526`) and then **dropped from the record** — which is why the
raw pages are not among the committed artifacts.

**What is committed in this repo** (the box's `outputs/` is not):

| Committed file | What it holds |
|---|---|
| `stage0_results/t4_s2_evidence/sh_6k_{off,offA,offB,detA..D,shadow}_results.json` | 8 full benchmark run files (the format above), sh_6k |
| `stage0_results/question_strata.json` | those 8 runs re-graded per stratum + error classes |
| `stage0_results/t4_s2_trials_summary.json` | off-vs-shadow noise-floor analysis (TOST, permutation) |
| `stage0_results/final/m0..m4_*.json` | Stage-0 instrument measurements |
| `stage0_results/stage1/stale_suppression_probe_sh{6,32}k.json` | oracle probe runs (5 arms × 100 questions each) |
| `stage0_results/stage1/stage1_calibration.json` | 162-cell detector calibration |
| `stage0_results/stage1_operating_point.json` | the frozen detector operating point |
| `stage0_results/stage1/detector_gap_sh{6,32}k.json` (+`_retrieval_` variants) | detector-vs-oracle comparisons |
| `stage0_results/stage1/detector_gap_confirmatory_sh64k.json` | **the held-out confirmatory run** |

H-Nav's own working outputs (`hnav/_out/`, `hnav/_cache/emb/`) are
gitignored; anything kept was copied into `stage0_results/` deliberately.

### Are the intermediate stage outputs in the repository?

Per stage, what can be shown from committed files versus what existed only on
the GPU box. A single question (sh_6k index 1) is traced hop-by-hop, with the
actual data quoted at every stage, in
**`presentation_evidence/data/data_journey_sh6k_q1.md`** — that file is the
screenshot-ready exhibit for this table.

| Stage output | In repo? | Where / what instead |
|---|---|---|
| Dataset entry (facts, question, gold) | **COMMITTED** | `data/Conflict_Resolution.json` |
| Templated query (full prompt block per question) | **COMMITTED** | every run file, `data[].query` — e.g. `t4_s2_evidence/sh_6k_off_results.json` |
| Chunk texts | not stored (transient, in-memory) | per-subset chunk **statistics** are committed: `m2_retrieval_calibration.json` → `n_chunks` = 2 / 9 / 17 / 67 for sh_6k/32k/64k/262k, `fallback_chunker: false` on all four |
| Memorize-wrapped chunks | not stored (contain run timestamps) | template committed at `utils/templates.py:38`; `token_stats.total_memorization_input_tokens = 0` proves no LLM touched them |
| Retrieval query | derivable, exact | the extraction is a pure regex (`embedding_retriever.py:314`) over the committed `query` string — re-applied offline it reproduces deterministically |
| Retrieved page | **NOT IN REPOSITORY** (written to `outputs/rag_retrieved/` on the box, then dropped from the record) | sh_6k: page ≡ whole context (2 chunks, k=10) — byte-exact reproduction committed as `presentation_evidence/data/item01_page_excerpt.txt`; sh_64k: box-only prepass; committed proxies: `input_len` per question, `n_chunks_on_page: 10` |
| Exact prompt sent to the LLM | benchmark run: sizes only | probe runs commit a **SHA-1 per arm per question** (`per_question[].arms.*.prompt_sha`), and the probe's native prompt is byte-exactly reproducible offline |
| Model output | **COMMITTED, twice** | benchmark rows (`data[].output`, `parsed_output`) and probe arms (`per_question[].arms.*.output`) |
| Scoring verdict | **COMMITTED** | `data[].substring_exact_match` (+ f1, ROUGE); the scoring functions re-executed offline reproduce every checked verdict |
| Aggregates | **COMMITTED** | `averaged_metrics` / `token_stats` in each run file; stratified re-grade in `question_strata.json` |

> ### 🔎 Data journey — Stage 8: aggregation  [COMMITTED]
>
> `averaged_metrics` in the same run file (means ×100 over 100 questions):
>
> ```
> substring_exact_match  29.00
> exact_match            29.00
> f1                     30.73
> ```
>
> Stratified re-grade of this run: `stage0_results/question_strata.json` →
> the `runs[]` entry where `run == "sh_6k_off"` (unique 26/26, conflicted
> 3/74).

---

## 9. Stage 9 — where H-Nav comes into play

Two different things carry the name H-Nav in this repository, and keeping
them apart is the single most important thing for reading the results:

1. **The hooks** — two small, guarded edits inside the benchmark's own code.
   They give H-Nav *eyes* on the running pipeline (and, in one mode that was
   never used for any headline number, *hands*). This is the **online path**.
2. **The probe harness** — a separate offline program under `hnav/stage1/`
   that rebuilds the benchmark's exact prompt and scoring, so controlled
   experiments can be run one edit at a time. **Every headline number in this
   project comes from here**, not from live benchmark runs. This is the
   **offline path**.

H-Nav never replaces any part of the pipeline. It watches it from two fixed
points, and it experiments on a faithful copy of it.

### The whole picture in one diagram

```
ONLINE PATH — the benchmark itself, as it runs on the GPU box

  Conflict_Resolution.json
          │
          ▼
  [2] template the question + chunk the context
          │
          ▼
  [3] memorize chunks ────────────────►  HOOK 1  agent.py send_message
          │                              sees every chunk and every query pass
          ▼                              by, and the result on the way out;
  [4] FAISS search → top-10 "page" ───►  HOOK 2  TextRetriever.retrieve
          │                              receives the FULL similarity ranking;
          ▼                              this is the only place a page could
  [5] prompt = Memory blocks + query     ever be edited
          │
          ▼                              modes:  off    = hooks do nothing
  [6] LLM  (temperature 0, 10 tokens)            shadow = watch + log only
          │                                      live   = may edit the page
          ▼                                               (refused for every
  [7] substring scoring                                    headline number)
          │
          ▼
  [8] results JSON


OFFLINE PATH — the Stage-1 probe harness (where H-Nav's numbers come from)

  same question ──► rebuild stages 5–7 exactly: same system message, same
                    template, same "Memory i:" page shape, temperature 0,
                    max_tokens 10, same substring grader
                        │
                        ├── arm: native           page untouched (baseline)
                        ├── arm: native_repeat    same page again (noise floor)
                        ├── arm: *_suppress       stale facts deleted from page
                        ├── arm: *_recency        newest fact moved to the end
                        └── arm: anti             adversarial placement
                        each arm = one LLM call, scored by the same rule;
                        the ONLY difference between arms is one page edit
```

### 9a. The two hooks inside the benchmark (online path)

Both are no-ops unless the environment variable `HNAV_MODE` is set; the
default is `off`. A stray import can never move a benchmark number.

**Hook 1 — `send_message` entry/exit** (`agent.py:988-1020`).
`send_message` is the one door everything passes through: every chunk on its
way into memory, every question on its way to an answer. The hook announces
each passage to H-Nav's adapter on the way in and on the way out — so H-Nav
can see *what* is being stored and *what* is being asked — but the dispatch
and the return value are untouched. This hook only ever observes.

**Hook 2 — `TextRetriever.retrieve`** (`embedding_retriever.py:217-258`).
When the benchmark asks FAISS for the 10 most similar chunks, the hook asks
for the ranking of **all** chunks instead (the index is exact, so the first
10 are the same 10 either way), hands the full ranking, the scores, and the
query vector to H-Nav's adapter, and then returns exactly the same top-10
page the benchmark would have produced anyway. This is the one place in the
whole pipeline where a page-editing decision *could* be applied
(`apply_read_decision`, `embedding_retriever.py:253-254`) — which is why the
modes matter:

- **`off`** — the hooks return immediately; the run is the vanilla benchmark.
- **`shadow`** — H-Nav computes and logs its signals, but the returned page
  and the answer are byte-identical to `off`. This is verified two ways:
  mechanism tests (the hooked functions return the caller's own objects; no
  store mutation), and at run level — across 10 `off` runs and 5 `shadow`
  runs, the off↔shadow disagreement rate is 2.42%, *below* the off↔off
  noise floor of 3.04% (`t4_s2_trials_summary.json`), i.e. shadow is
  statistically indistinguishable from off (TOST-equivalent).
- **`live`** — the only mode where the page may actually be edited: rerank
  chunks, splice named stale facts out of chunk text (*suppress*), or move
  named facts to the end (*demote*) — always preserving the chunk count and
  block order. `live` was refused throughout Stage 0
  (`config.require_not_live()`); **no headline number in this project comes
  from a live benchmark run.**

The design rule behind all of this: `hnav/core/` is benchmark-agnostic and
never sees gold answers; everything benchmark-specific lives in
`hnav/adapters/mab_adapter.py`; everything that reads `questions`/`answers`
lives offline under `hnav/labeling/`, `hnav/stage0/`, `hnav/stage1/`. An AST
scan (`hnav/tests/test_leakage_audit.py`) fails the test suite if any online
module ever imports the offline tier.

### 9b. The Stage-1 probe harness (offline path — where the headline numbers come from)

Why a separate harness at all? Because the question we need answered is
causal: *does deleting a stale fact from the page fix the model's answer?* To
answer it you must ask the model the same question twice with **exactly one
thing changed** — and the benchmark has no switch for that. The probe harness
is that switch.

**Fidelity.** The harness bypasses `main.py` but re-implements stages 5–7 of
the pipeline exactly, in `hnav/stage1/calibrate_read_policy.py:86-103`: the
same system message, the same `rag_agent` query template (copied from
`utils/templates.py` rather than imported, to preserve the layering rule),
the same `Memory i:` page shape, `temperature=0`, `max_tokens=10`, and the
same grader (`substring_exact_match`). The probe scripts then import this
prompt builder, which is why every probe artifact records the contract in its
`harness` block:

```json
"prompt_shape":  "RAGSystem: 'Memory 1:\n<whole context>\n' + templated query",
"prompt_source": "hnav.stage1.calibrate_read_policy (imported verbatim)",
"grader":        "hnav.labeling.counterfactual.substring_exact_match"
```

One declared deviation (recorded in the artifact's
`harness.deviation_from_campaign`): the oracle probes put the whole context
in **one** Memory block, in context order, because the placement arms need a
defined "end of context". The confirmatory run instead uses the
**benchmark's own retrieved top-10 page** (`page_source: "benchmark"` in the
artifact) — the real production page shape.

**The five arms.** Per question, the harness builds five versions of the
page, sends each to the model once, and grades each with the same rule. The
arm definitions, verbatim from the committed artifact:

| Arm | The one change made to the page | What it isolates |
|---|---|---|
| `native` | "untouched context" | the baseline |
| `native_repeat` | "same prompt, independent second call (A/A floor)" | the noise floor at temperature 0 |
| `oracle_suppress` | "delete every non-expected-value fact of the queried key; serials NOT renumbered" | does *removing the stale facts* fix the answer? |
| `oracle_recency` | "move the highest-serial fact of the queried key to the END" | does *text position* of the newest fact matter? |
| `anti` | "highest-serial fact to the FRONT, most recent stale fact LAST" | adversarial placement — can position alone break a question? |

The `oracle_*` arms are allowed to use the answer key (they run in the
offline tier); they measure the **ceiling** — how much accuracy is available
if stale facts could be identified perfectly. The detector arms (below)
replace the gold plan with H-Nav's own plan and measure how much of that
ceiling H-Nav actually reaches without ever seeing an answer.

#### Worked example — one question, before and after

Question: *"What is Nobuhiro Watsuki famous for?"* (sh_6k, index 1). Every
value below is from `stale_suppression_probe_sh6k.json`, `per_question[1]`.

**Before (native arm).** The page is the full 455-fact context, containing
both lines:

```
91.  Nobuhiro Watsuki is famous for Rurouni Kenshin.          <- stale
259. Nobuhiro Watsuki is famous for The Fairly OddParents.    <- newest = gold
```

The model answers `'Rurouni Kenshin'` — the stale value. Wrong. The
independent repeat call (`native_repeat`) returns the identical string, so
the failure is deterministic, not sampling noise.

**The edit (oracle_suppress arm).** Exactly one sentence — fact 91 — is
deleted from the page. 455 facts become 454; the serial numbers are *not*
renumbered; nothing else changes (the per-arm prompt SHA-1s in the Stage 5
box document that the prompts differ only here).

**After.** The model answers `'The Fairly OddParents'`. Right.

**The controls on the same question.** Moving the newest fact #259 to the end
of the page (`oracle_recency`) does *not* fix it — still
`'Rurouni Kenshin'`. The adversarial placement (`anti`) doesn't change it
either. So for this question the cause of the failure is specifically the
*presence* of the stale sentence, and deleting it is what flips the answer.
That is the causal pattern the whole Stage-1 campaign generalizes: one page
edit in, one answer flip out, with an A/A repeat guarding against noise.

#### How the detector finds stale facts — without the answer key

The oracle needed gold labels to know fact 91 was stale. The detector must
reach the same conclusion using only what is on the page. It builds its
suppression plan by requiring a series of independent checks to all pass:

1. **Find suspiciously similar pairs.** Every fact on the page is embedded
   (Qwen3-Embedding-4B — the same embedder the benchmark uses). Two facts
   whose cosine similarity is ≥ 0.90 become a candidate pair: "these two
   sentences say nearly the same thing."
2. **Confirm they are about the same subject.** A parsed subject-identity
   screen (`pair_filter`) keeps a pair only if both facts name the same
   subject and relation — e.g. both about what Nobuhiro Watsuki is famous
   for. This discards facts that merely *sound* alike.
3. **Confirm they actually disagree.** A natural-language-inference model
   must judge the pair contradictory, checked in both directions, with
   contradiction probability ≥ 0.90. "X is famous for Rurouni Kenshin" vs
   "X is famous for The Fairly OddParents" contradict; a paraphrase pair does
   not.
4. **Keep the newest, suppress the rest.** Within each verified group of
   conflicting facts, the fact with the highest serial number is kept and
   every other member goes on the suppression list — the same recency rule
   the prompt states. (A geometry screen, the QR span residual, runs at its
   loose setting `r_min = 0.44`; the Stage-0 ambiguity screen — `nmargin` /
   `H_z` — is *disabled* at this operating point, a declared decision
   recorded in the artifact together with its reason: those signals were
   computed from chunk embeddings truncated at 512 tokens and, with the
   screen on, question-level recall collapses from 0.97 to 0.16.)

All thresholds were **frozen before any answer was graded**, in
`stage0_results/stage1_operating_point.json`, fitted on detection quality
only — no LLM, no accuracy, no gold answers — and only on the calibration
split (sh_6k + sh_32k; sh_64k and sh_262k were explicitly refused at fit
time). On the 200 calibration questions the frozen plan proposed 2,673
suppressions, and all 2,673 were genuinely superseded facts — precision 1.0,
with 97.8% of the ground-truth conflict pairs in the pool recovered.

#### The confirmatory run — held-out data, real retrieved page

`detector_gap_confirmatory_sh64k.json` is the same five-arm design run once
on **sh_64k**, a subset never touched during calibration, using the
benchmark's own retrieved top-10 page. A real suppression plan from it
(question index 0, key "The New York Times | was written in the language
of"):

```json
"plan": {"suppress_serials": [820, 1071, 1327, 1414, 1654, 1711, 2336,
                              2436, 2716, 2800, 3148], "n_pairs_verified": 11},
"arms": {"native":            {"output": "English", "n_facts": 2806},
         "detector_suppress": {"output": "English", "n_facts": 2795}}
```

Eleven detector-verified superseded facts were spliced out of the page
(2,806 → 2,795 facts, prompt 665 characters shorter); on this particular
question the answer did not change — it is one of the 29 conflicted questions
suppression did not fix.

Across the whole run:

- **Conflicted-stratum accuracy 17/66 → 37/66.** McNemar pairing: 0 questions
  harmed, 20 fixed, exact p = 1.9×10⁻⁶. Prompt tokens −0.31%.
- **Suppression precision 735/735** — every fact the detector deleted across
  the run was independently re-checked offline and every one was genuinely
  superseded.
- **But the pre-registered protective criterion was voided** — by exactly one
  question outside the conflicted stratum (index 77): the native run answered
  `"John Milton"`; after the edit the model answered `"The provided knowledge
  pool does not contain any information about"` — a refusal induced by the
  edit, even though no gold fact was cut. Under the pre-registration, one
  such harm voids the safety claim. The registered verdict is therefore:
  **effective, but not yet safe** — stated in the same breath as the
  improvement, per `HNAV_FINAL_REPORT.md` §10.

### 9c. The offline analysis tier (never touches the model)

`hnav/labeling/conflict_analysis.py::parse` (lines 53-68) — a deterministic
regex parser over the templated facts (99.5%+ coverage) — builds the
key/serial index used for stratification (`question_strata.json`), the truth
pairs for the geometry ablation, and the independent 735/735 recount. It
reads the dataset file directly and therefore lives strictly in the offline
tier.

> ### 🔎 Data journey — Stage 9: what H-Nav did to this question  [COMMITTED]
>
> Oracle plan and outcome (`stale_suppression_probe_sh6k.json` →
> `per_question[1]`):
>
> ```json
> {
>   "gold_serials": [259],
>   "stale_serials": [91],
>   "latest": 259,
>   "stale_anchor": 91,
>   "gold_is_latest": true
> }
> ```
>
> Deleting stale fact #91 flips the answer: native `'Rurouni Kenshin'`
> (wrong) → oracle_suppress `'The Fairly OddParents'` (right), with the A/A
> repeat identical to native (noise floor 0).

---

## Appendix — quick reference of every load-bearing location

| What | Where |
|---|---|
| Dataset | `MemoryAgentBench/data/Conflict_Resolution.json` |
| Eval loop | `main.py:190-226`; per-query save `main.py:156` |
| Query templating | `conversation_creator.py:184-190` |
| Chunking (4096 tok, sentence-packed) | `utils/eval_other_utils.py:173-222` |
| Memorize template | `utils/templates.py:38`; applied `agent.py:1467-1471` |
| System prompt | `utils/templates.py:2` |
| Query template (recency rule) | `utils/templates.py:41` |
| Embedder routing (server, not in-process) | `embedding_retriever.py:176-183` |
| Retrieval-query regex | `embedding_retriever.py:314-322` |
| FAISS search + **H-Nav hook 2** | `embedding_retriever.py:207-258` |
| Prompt assembly (`Memory i:` + query) | `embedding_retriever.py:332-334` |
| LLM call (`max_tokens=10`) | `embedding_retriever.py:345-350` |
| `send_message` **H-Nav hook 1** | `agent.py:988-1020` |
| Output parsing | `utils/eval_other_utils.py:140-166` |
| Normalization + substring match | `utils/eval_other_utils.py:28-44, 101-112` |
| Per-run result JSON schema | `main.py:73-112` |
| Committed run artifacts | `stage0_results/t4_s2_evidence/`, `stage0_results/stage1/`, `stage0_results/final/` |
| Probe harness (prompt contract copy) | `hnav/stage1/calibrate_read_policy.py:86-103` |
| Frozen detector operating point | `stage0_results/stage1_operating_point.json` |
| Confirmatory run | `hnav/stage1/detector_gap.py` → `stage0_results/stage1/detector_gap_confirmatory_sh64k.json` |
| Fact parser (offline) | `hnav/labeling/conflict_analysis.py:53-68` |
| Per-stage data journey exhibit | `presentation_evidence/data/data_journey_sh6k_q1.md` |
