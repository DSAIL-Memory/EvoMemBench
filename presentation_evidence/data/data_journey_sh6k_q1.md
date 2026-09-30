# Data journey — sh_6k, question index 1 (`factconsolidation_sh_6k_no1`)

One question traced through every stage of the benchmark pipeline. Each hop
quotes the actual data and names the committed file it was read from, or says
plainly that the intermediate is not stored. Built offline by
`presentation_evidence/_scripts/make_data_journey.py`.

Provenance labels: **COMMITTED** = quoted verbatim from a file in this repo;
**RECONSTRUCTED** = rebuilt offline from committed inputs + the repo's own code,
with the deviation stated; **NOT IN REPOSITORY** = existed only on the GPU box.

---

## Stage 1 — dataset entry  [COMMITTED]

`In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json`,
entry with `metadata.qa_pair_ids[0] = "factconsolidation_sh_6k_no0"`:

```
context, line for serial 91 : Nobuhiro Watsuki is famous for Rurouni Kenshin.
context, line for serial 259: Nobuhiro Watsuki is famous for The Fairly OddParents.
questions[1]                : What is Nobuhiro Watsuki famous for?
answers[1]                  : ['The Fairly OddParents']
```

## Stage 2a — templated query  [COMMITTED]

The full templated query is stored per question in every committed run file —
`stage0_results/t4_s2_evidence/sh_6k_off_results.json` -> `data[1].query`
(791 chars, quoted in full):

```
Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. 
 You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. 

For example:

 [Knowledge Pool] 

 Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? 
Answer: Donald Trump 

 Now Answer the Question: Based on the provided Knowledge Pool, What is Nobuhiro Watsuki famous for? 
Answer:
```

## Stage 2b — chunking  [statistics COMMITTED; chunk texts NOT IN REPOSITORY]

The chunk texts are transient (built in memory each run). What the repo does
hold is the per-subset chunking record, in
`stage0_results/final/m2_retrieval_calibration.json` — including the
`fallback_chunker` flag proving the benchmark's own nltk chunker was used:

```
subset     n_chunks  fallback_chunker  top_k
sh_6k             2             False     10
sh_32k            9             False     10
sh_64k           17             False     10
sh_262k          67             False     10
```

(The confirmatory artifact independently records `n_chunks_total: 17`,
`n_chunks_on_page: 10` for sh_64k. Chunk boundaries cannot be regenerated on
this machine — nltk punkt data and tiktoken are not installed locally — but
for sh_6k both chunks are always retrieved, so no boundary matters downstream.)

## Stage 3 — memorize-wrapped chunk  [template COMMITTED; wrapped text NOT IN REPOSITORY]

Each chunk is wrapped in the memorize template (`utils/templates.py:38`)
before storage. The wrapped strings are not saved anywhere (and contain a
wall-clock timestamp, so they differ per run). Template, verbatim:

```
Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.
```

Evidence that memorization makes zero LLM calls: the committed run file's
`token_stats.total_memorization_input_tokens` = 0.

## Stage 4a — retrieval query  [RECONSTRUCTED, exact]

The benchmark extracts the embedding query from the templated query with the
regex at `embedding_retriever.py:314`. Re-applying that regex to the committed
query string (pure `re`, deterministic — no model involved):

```
Based on the provided Knowledge Pool, What is Nobuhiro Watsuki famous for? 
Answer:
```

## Stage 4b — retrieved page  [NOT IN REPOSITORY; sh_6k content equivalent COMMITTED]

The benchmark writes each retrieved page to `outputs/rag_retrieved/...` on the
GPU box and then drops it from the result record (`agent.py:1518-1526`) — those
files were never committed. For **sh_6k** this loses nothing: the store is 2
chunks and `retrieve_num = 10`, so the page always contains the whole context.
The probe harness's whole-context page IS byte-exactly reproducible offline —
see `presentation_evidence/data/item01_page_excerpt.txt` (455 facts, verified
identical to the dataset's `context` field). For sh_64k the page is only on
the box (`detector_inputs.prepass` in the confirmatory artifact).

Retrieval evidence that IS committed: `input_len` per question (page + query
token count) — for this question, `data[1].input_len` = 6711.

## Stage 5 — the prompt sent to the model  [hash COMMITTED; probe text RECONSTRUCTED]

The benchmark run keeps no prompt text, only sizes. The Stage-1 probe run of
this same question keeps a SHA-1 prefix of every arm's exact prompt
(`stale_suppression_probe_sh6k.json` -> `per_question[1].arms.*.prompt_sha`):

```
native           prompt_sha=b97826cc85c85bc2  n_facts=455
native_repeat    prompt_sha=b97826cc85c85bc2  n_facts=455
oracle_suppress  prompt_sha=1e25e93d53184ed5  n_facts=454
oracle_recency   prompt_sha=52212d7a755c9d1c  n_facts=455
anti             prompt_sha=6217dc0571cb1a54  n_facts=455
```

The probe's native prompt is reproducible byte-exactly offline
(`render_context` + `build_prompt`; head/tail quoted in item01_page_excerpt.txt).

## Stage 6 — model output  [COMMITTED, twice]

Benchmark run (`sh_6k_off_results.json` -> `data[1]`):

```
output       : 'Rurouni Kenshin and The Fairly'
parsed_output: 'Rurouni Kenshin and The Fairly'
output_len   : 10 tokens (cap: generation_max_length = 10)
```

Probe run, all five arms (`stale_suppression_probe_sh6k.json` -> `per_question[1]`):

```
native           -> 'Rurouni Kenshin'            correct=False
native_repeat    -> 'Rurouni Kenshin'            correct=False
oracle_suppress  -> 'The Fairly OddParents'      correct=True
oracle_recency   -> 'Rurouni Kenshin'            correct=False
anti             -> 'Rurouni Kenshin'            correct=False
```

## Stage 7 — scoring  [COMMITTED verdict; recomputation reproduces it]

`substring_exact_match` = normalize(gold) in normalize(prediction), where
normalize = lowercase, strip punctuation, drop a/an/the, collapse spaces
(`utils/eval_other_utils.py:28-44, 101-112`). Re-executed on the committed row:

```
norm(output) = 'rurouni kenshin and fairly'
norm(gold)   = 'fairly oddparents'
substring    -> False        recorded verdict: False  (reproduces)
other stored metrics: exact_match=False, f1=0.333, rougeL_f1=0.500
```

## Stage 8 — aggregation  [COMMITTED]

`averaged_metrics` in the same run file (means x100 over 100 questions):

```
substring_exact_match  29.00
exact_match            29.00
f1                     30.73
```

Stratified re-grade of this run: `stage0_results/question_strata.json` ->
`runs[]` where `run == "sh_6k_off"` (unique 26/26, conflicted 3/74).

## Stage 9 — what H-Nav did to this question  [COMMITTED]

Oracle plan and outcome (`stale_suppression_probe_sh6k.json` -> `per_question[1]`):

```
{
  "gold_serials": [
    259
  ],
  "stale_serials": [
    91
  ],
  "latest": 259,
  "stale_anchor": 91,
  "gold_is_latest": true
}
```

Deleting stale fact #91 flips the answer: native 'Rurouni Kenshin'
(wrong) -> oracle_suppress 'The Fairly OddParents' (right),
with the A/A repeat identical to native (noise floor 0).
