# INEP-KNOW: In-Episode Knowledge Evaluation

`INEP-KNOW` is the in-episode knowledge track of EvoMemBench.

The evaluation code lives in `MemoryAgentBench/`. Unless noted otherwise, run commands from that directory.

## Repository Layout

```text
INEP-KNOW/
├── MemoryAgentBench/
│   ├── agent.py                         # Agent wrapper and backend routing
│   ├── conversation_creator.py          # Context/query construction
│   ├── initialization.py                # Output and cache path helpers
│   ├── main.py                          # Evaluation entry point
│   ├── configs/
│   │   ├── agent_conf/                  # Agent, RAG, and memory backend configs
│   │   └── data_conf/                   # Dataset configs
│   ├── bash_files/
│   │   ├── configs/                     # Script config lists
│   │   └── sh/                          # Experiment and smoke-test scripts
│   ├── data/                            # Optional local JSON data
│   ├── agents/                          # Agent and memory caches
│   └── outputs/                         # Evaluation outputs
├── requirements.txt
└── README.md
```

## Installation

Create a fresh Python environment and install the dependencies:

```bash
conda create -n inep-know python=3.12
conda activate inep-know

pip install -r requirements.txt
pip install "numpy<2"
```

You can also install from inside `MemoryAgentBench/`:

```bash
cd MemoryAgentBench
pip install -r requirements.txt
pip install "numpy<2"
```

## Data

By default, the benchmark loads data from the Hugging Face dataset:

```text
ai-hyz/MemoryAgentBench
```

Supported splits include:

- `Accurate_Retrieval`
- `Conflict_Resolution`

Local JSON files can also be placed under:

```text
MemoryAgentBench/data/*.json
```

Dataset YAML files live in:

```text
MemoryAgentBench/configs/data_conf/**/*.yaml
```

Each dataset config usually defines:

| Field | Typical value | Description |
| --- | --- | --- |
| `dataset` | `Accurate_Retrieval`, `Conflict_Resolution` | Top-level benchmark split. |
| `sub_dataset` | Config-specific | Subset name used for sample filtering and output filenames. |
| `chunk_size` | `4096` | Context chunk size. Memory agents can override this with `agent_chunk_size`. |

## Environment Variables

Create `MemoryAgentBench/.env` and add the credentials required by the models or memory systems you plan to run. Do not commit real keys.

```bash
DEEPSEEK_API_KEY=<your-ark-or-deepseek-api-key>
BATCH_MODEL=<your-ark-batch-endpoint>

DASHSCOPE_API_KEY=<your-dashscope-api-key>
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Common variables:

| Variable | Required when | Description |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Running DeepSeek or Ark-compatible models | Main model and batch inference credential. |
| `BATCH_MODEL` | Running Ark batch inference | Batch inference endpoint name. |
| `DASHSCOPE_API_KEY` | Running embedding-based RAG or memory backends | DashScope credential for embeddings or compatible services. |
| `DASHSCOPE_BASE_URL` | Using DashScope-compatible endpoints | OpenAI-compatible DashScope base URL. |

Long-context baselines may only need the main model credential. Memory and RAG systems such as `mem0`, `amem`, `memos`, and `memoryos` usually also require embedding-service credentials.

## Quick Start

Run a small smoke test before launching full experiments:

```bash
cd MemoryAgentBench

python main.py \
  --agent_config configs/agent_conf/RAG_Agents/deepseek-chat/Simple_rag_deepseek-chat-bm25.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml \
  --max_test_queries_ablation 2 \
  --force
```

Results are written under the `output_dir` configured by the agent YAML, grouped by dataset:

```text
outputs/<agent-output-dir>/<dataset>/*_results.json
```

For example:

```text
outputs/deepseek-chat-amem/Conflict_Resolution/factconsolidation_sh_262k_..._results.json
```

## Running Evaluations

The unified entry point is:

```bash
python main.py --agent_config <agent-yaml> --dataset_config <dataset-yaml>
```

Useful command-line options:

| Option | Description |
| --- | --- |
| `--agent_config` | YAML config for a long-context, RAG, or memory agent. |
| `--dataset_config` | YAML config for a benchmark task/subset. |
| `--max_test_queries_ablation N` | Limit the run to at most `N` queries. Useful for smoke tests. |
| `--chunk_size_ablation N` | Override dataset `chunk_size`; also overrides `agent_chunk_size` when present. |
| `--force` | Re-run even if a matching result file already exists. |

`main.py` loads YAML configs first, then applies command-line overrides. If an agent config defines `generation_max_length`, it takes priority over the dataset value.

## Agent Configs

Agent YAML files live in:

```text
MemoryAgentBench/configs/agent_conf/**/*.yaml
```

Common fields:

| Field | Typical values | Description |
| --- | --- | --- |
| `agent_name` | `Long_context_agent_*`, `Simple_rag_*`, `Structure_rag_*`, `Embedding_rag_*` | Selects initialization and `send_message()` routing in `AgentWrapper`. |
| `model` | `deepseek-chat` | Main answer model or agent model. |
| `temperature` | `0.7`, `1.0` | Sampling temperature. |
| `input_length_limit` | `95000` to `10000000` | Input budget for the answer model. |
| `output_dir` | `./outputs/<agent-name>` | Root directory for result JSON files. |
| `retrieve_num` | `10` | Top-k retrieval count for RAG and memory agents. |
| `agent_chunk_size` | `4096` | Chunk size used when writing context into a memory backend. |

## Experiment Scripts

Reusable scripts are stored in:

```text
MemoryAgentBench/bash_files/sh/
```

Run a script from `MemoryAgentBench/`:

```bash
bash bash_files/sh/run_smoke_baselines.sh
```

## Adding a New Memory Backend

The easiest implementation path is to follow an existing backend such as `mem0`, `amem`, `memos`, or `memoryos`.

### 1. Add an Agent Config

Create a YAML file under:

```text
MemoryAgentBench/configs/agent_conf/RAG_Agents/<new_memory>/
```

Example:

```yaml
agent_name: Structure_rag_<new_memory>
model: deepseek-v3-2-251201
temperature: 1.0
input_length_limit: 10000000
buffer_length: 1000
output_dir: ./outputs/deepseek-chat-<new_memory>

retrieve_num: 10
agent_chunk_size: 4096
```

`agent_name` should include a backend keyword that `AgentWrapper` can recognize. If your backend does not need chunk-by-chunk memory writes, omit `agent_chunk_size` and the dataset `chunk_size` will be used.

### 2. Register Initialization

In `MemoryAgentBench/agent.py`, add a branch to `AgentWrapper._initialize_agent_by_type()`:

```python
elif self._is_agent_type("<new_memory>"):
    self._initialize_<new_memory>_agent(agent_config, dataset_config)
```

Then implement the initializer:

```python
def _initialize_<new_memory>_agent(self, agent_config, dataset_config):
    self.retrieve_num = agent_config["retrieve_num"]
    self.agent_start_time = time.time()
    self.client = self._create_oai_client()
    # Initialize memory storage, embeddings, vector stores, clients, and counters.
```

The initializer is the right place to read YAML fields, initialize model clients, set up memory state, and create token counters such as:

```python
self._<new_memory>_memorization_input_tokens = 0
self._<new_memory>_memorization_output_tokens = 0
```

### 3. Implement Memorization and Query Handling

Add a handler in `MemoryAgentBench/agent.py`:

```python
def _handle_<new_memory>_agent(self, message, memorizing, query_id, context_id):
    if memorizing:
        # `message` is one context chunk.
        # Write it into your memory backend or update your index here.
        return "Memorized"

    memory_construction_time = time.time() - self.agent_start_time
    # 1. Extract or build a retrieval query.
    # 2. Retrieve relevant memory.
    # 3. Build the prompt with retrieved memory.
    # 4. Call self._chat_complete(...).
    # 5. Return the standard response dict.
```

Query responses should use the standard output helper:

```python
output = self._create_standard_response(
    response_text,
    input_tokens,
    output_tokens,
    memory_construction_time,
    query_time_len,
)
```

If memory construction uses an LLM, include memorization token counts:

```python
output["memorization_input_len"] = self._<new_memory>_memorization_input_tokens
output["memorization_output_len"] = self._<new_memory>_memorization_output_tokens
```

`main.py` aggregates these fields into the final result JSON.

### 4. Route `send_message()`

Add the backend to `AgentWrapper.send_message()`:

```python
elif self._is_agent_type("<new_memory>"):
    return self._handle_<new_memory>_agent(message, memorizing, query_id, context_id)
```

The evaluation lifecycle is:

1. `initialize_and_memorize_agent()` creates the agent.
2. Each context chunk is passed to `send_message(..., memorizing=True, context_id=...)`.
3. `save_agent()` persists agent state.
4. Each query is passed to `send_message(..., memorizing=False, query_id=..., context_id=...)`.
5. The result JSON is updated after each query.

### 5. Support Cache Save/Load

If your backend persists state, add branches to `AgentWrapper.save_agent()` and `AgentWrapper.load_agent()`.

Also update `generate_agent_save_folder()` in `MemoryAgentBench/initialization.py` so the backend receives a context-specific cache directory:

```text
agents/<agent_name>_<sub_dataset>_chunk<agent_chunk_size>_model<model>/exp_<context_id>
```

Backends that rebuild memory from scratch can keep save/load minimal, but they should not interrupt the evaluation flow.

### 6. Reuse Prompt Templates

Prompt templates are selected in `MemoryAgentBench/utils/templates.py`:

```python
get_template(self.sub_dataset, "system", self.agent_name)
get_template(self.sub_dataset, "memorize", self.agent_name)
get_template(self.sub_dataset, "query", self.agent_name)
```

Reuse existing templates when possible. If the new `agent_name` does not match an existing branch, add the required logic in `templates.py`.

`ConversationCreator` expects samples to include:

- `context`
- `questions`
- `answers`

Optional metadata fields include:

- `question_dates`
- `question_types`
- `question_ids`
- `previous_events`
- `qa_pair_ids`
- `source`

### 7. Add Run Scripts

Full experiment scripts use the `run_exp_<method>.sh` naming pattern:

```text
MemoryAgentBench/bash_files/sh/run_exp_<new_memory>.sh
```

For a minimal smoke test:

```bash
cd MemoryAgentBench

python main.py \
  --agent_config configs/agent_conf/RAG_Agents/<new_memory>/<new_memory>.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml \
  --max_test_queries_ablation 2 \
  --force
```

After the run, check that a non-empty result file exists:

```text
outputs/<agent-output-dir>/Conflict_Resolution/factconsolidation_sh_262k*_results.json
```

If you add a smoke orchestrator that runs multiple methods or a reduced subset, use the `run_smoke_<scope>.sh` naming pattern.

## Troubleshooting

- Missing credentials: verify `MemoryAgentBench/.env` and the environment variables required by your selected model or backend.
- No result file: check that `output_dir` in the agent YAML points to the directory you are inspecting.
- Existing result is reused: add `--force` to overwrite prior results.
- Backend is not selected: confirm that `agent_name` contains the keyword used in `AgentWrapper._is_agent_type(...)`.
- Unexpected chunking: check `--chunk_size_ablation`, dataset `chunk_size`, and agent `agent_chunk_size`.

## Citation

If you use EvoMemBench or `INEP-KNOW` in your work, please cite the project according to the citation information provided in the main repository.
