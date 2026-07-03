# CrossEp-Web

CrossEp-Web is the web-search cross-episode execution benchmark in
EvoMemBench. It evaluates whether an agent memory system can learn reusable
search strategies, procedures, and task experience from one set of web-search
episodes, then improve performance on a later set of episodes.

This task suite combines:

- **XBench DeepSearch**: web-search questions that require multi-step evidence
  gathering and answer synthesis.
- **WebWalkerQA**: website-grounded question answering tasks that require
  browsing, reading, and reasoning over web pages.

The main entry point is `Flash-Searcher-main/run_pipeline.py`, which runs the
two benchmarks in sequence with a shared memory provider.

## Repository Layout

```text
CROSSEP-WEB/
|-- README.md
`-- Flash-Searcher-main/
    |-- run_pipeline.py
    |-- run_flash_searcher_xbench.py
    |-- run_flash_searcher_webwalkerqa.py
    |-- run_full_pipeline_*.sh
    |-- EvolveLab/
    |   |-- base_memory.py
    |   |-- config.py
    |   |-- memory_types.py
    |   `-- providers/
    |-- FlashOAgents/
    |-- data/
    |   |-- xbench/
    |   `-- webwalkerqa/
    |-- pipeline_output/
    `-- storage/
```

Important paths:

| Path | Description |
| --- | --- |
| `Flash-Searcher-main/` | Working directory for all run commands. |
| `Flash-Searcher-main/run_pipeline.py` | Sequential cross-benchmark pipeline. |
| `Flash-Searcher-main/data/xbench/DeepSearch.csv` | Default XBench input file. |
| `Flash-Searcher-main/data/webwalkerqa/webwalkerqa_subset_170.jsonl` | Default WebWalkerQA subset. |
| `Flash-Searcher-main/pipeline_output/` | Default output directory. |
| `Flash-Searcher-main/storage/` | Default persistent memory storage. |
| `Flash-Searcher-main/.env` | Local API, model, search, and crawling configuration. |

## Installation

Create a dedicated environment for this task:

```bash
conda create -n CROSSEP-WEB python=3.10
conda activate CROSSEP-WEB

cd Flash-Searcher-main
pip install -r requirements.txt
playwright install
```

`playwright install` downloads the browser binaries used by the web-search and
browsing tools.

## Configuration

Create a local `.env` file under `Flash-Searcher-main/` before running the
benchmark.

```bash
# === Core model configuration ===
DEFAULT_MODEL="deepseek-v3-2-251201"

# === Volcengine Ark / OpenAI-compatible inference ===
OPENAI_API_KEY="your_ark_api_key"
OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_API_KEY="your_ark_api_key"
BATCH_MODEL="your_ark_batch_endpoint"

# === Search and web access ===
SERPER_API_KEY="your_serper_api_key"
WEB_ACCESS_PROVIDER="crawl4ai"

# === Embedding / memory backend endpoints ===
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY="your_dashscope_api_key"
```

Common variables:

| Variable | Required | Description |
| --- | --- | --- |
| `DEFAULT_MODEL` | Yes | Main reasoning model used by the search agent. |
| `OPENAI_API_KEY` | Yes | API key for the OpenAI-compatible model endpoint. |
| `OPENAI_BASE_URL` | Yes | Base URL for the OpenAI-compatible model endpoint. |
| `DEEPSEEK_API_KEY` | Yes | API key used by Ark model and batch inference code. It is often the same as `OPENAI_API_KEY`. |
| `BATCH_MODEL` | Yes | Volcengine Ark batch inference endpoint name. |
| `SERPER_API_KEY` | Yes | Serper API key used for web search. |
| `WEB_ACCESS_PROVIDER` | Yes | Web crawling backend. `crawl4ai` is the recommended open-source option; `jina` is also supported. |
| `DASHSCOPE_API_KEY` | Backend-dependent | Required by memory providers that use DashScope embeddings. |
| `DASHSCOPE_BASE_URL` | Backend-dependent | OpenAI-compatible DashScope endpoint. |

Do not commit real API keys. Keep secrets in your local `.env` file.

## Data

All benchmark data is expected under `Flash-Searcher-main/data/`.

The repository includes the default files used by the pipeline:

- `data/xbench/DeepSearch.csv`
- `data/webwalkerqa/webwalkerqa_subset_170.jsonl`

If you use the full WebWalkerQA release, place it at:

```text
Flash-Searcher-main/data/webwalkerqa/webwalkerqa_main.jsonl
```

## Recommended Starting Point

For community users and new memory-provider integrations, the recommended
entry point is the mem0 full-pipeline script:

```bash
cd Flash-Searcher-main
bash run_full_pipeline_mem0.sh --xbench-n 2 --ww-n 2 --concurrency 2
```

This script is the most complete runnable template in this directory. 

## Running a Direct Smoke Test

Start with a small run to verify that the model endpoint, search API, browser
tooling, and memory provider are all configured correctly:

```bash
cd Flash-Searcher-main

python run_pipeline.py \
  --memory_provider lightweight_memory \
  --xbench_sample_num 2 \
  --webwalkerqa_sample_num 2 \
  --concurrency 2 \
  --output_dir ./pipeline_output/smoke_lightweight_memory
```

To run without an external memory provider:

```bash
python run_pipeline.py \
  --xbench_sample_num 2 \
  --webwalkerqa_sample_num 2 \
  --output_dir ./pipeline_output/smoke_no_memory
```

## Sequential Evaluation Protocol

`run_pipeline.py` evaluates cross-episode memory by splitting the run into two
phases:

1. **Phase 1 writes memory.** Tasks are processed sequentially with
   `concurrency=1`. After each task, the pipeline calls
   `take_in_memory(...)` on the memory provider.
2. **Phase 2 reads memory.** Tasks may run concurrently according to
   `--concurrency`. Memory writing is disabled, and the agent only calls
   `provide_memory(...)`.

The benchmark order is controlled by `--order`:

| Order | Meaning |
| --- | --- |
| `xbench_webwalkerqa` | Learn from XBench, then evaluate on WebWalkerQA. |
| `webwalkerqa_xbench` | Learn from WebWalkerQA, then evaluate on XBench. |

Example:

```bash
python run_pipeline.py \
  --memory_provider mem0 \
  --order xbench_webwalkerqa \
  --xbench_infile data/xbench/DeepSearch.csv \
  --webwalkerqa_infile data/webwalkerqa/webwalkerqa_subset_170.jsonl \
  --concurrency 4 \
  --output_dir ./pipeline_output/mem0_xbench_to_webwalkerqa
```

Useful run options:

| Option | Description |
| --- | --- |
| `--memory_provider` | Memory provider name registered in `EvolveLab/memory_types.py`. Omit for the no-memory baseline. |
| `--memory_storage_root` | Overrides provider storage paths so separate runs do not share memory state. |
| `--memory_dir` | Provider-specific memory directory override for providers that read `memory_dir`. |
| `--order` | Sequential transfer order. |
| `--concurrency` | Phase 2 parallelism. Phase 1 always runs sequentially. |
| `--xbench_sample_num` | Limit XBench to the first N tasks. |
| `--webwalkerqa_sample_num` | Limit WebWalkerQA to the first N tasks. |
| `--skip_xbench` | Skip XBench. |
| `--skip_webwalkerqa` | Skip WebWalkerQA. |

## Outputs

Each run creates a timestamped directory under `--output_dir`, usually named:

```text
pipeline_<memory_provider>_<timestamp>/
```

The main files are:

| Output | Description |
| --- | --- |
| `pipeline_report.txt` | Summary metrics for XBench, WebWalkerQA, and the combined run. |
| `xbench_results.jsonl` | Per-task XBench results. |
| `webwalkerqa_results.jsonl` | Per-task WebWalkerQA results. |
| `xbench_runs/` | Detailed XBench run artifacts. |
| `webwalkerqa_runs/` | Detailed WebWalkerQA run artifacts. |

## Supported Memory Providers

Registered providers are listed in `EvolveLab/memory_types.py`. The current
set includes retrieval, long-term memory, procedural memory, and lightweight
baselines such as:

- `bm25`
- `qwen3_embedding_4b`
- `graphrag`
- `mem0`
- `a_mem`
- `memos`
- `memory_os`
- `agent_kb`
- `agent_workflow_memory`
- `skillweaver`
- `ace`
- `reasoning_bank`
- `lightweight_memory`

Provider-specific configuration lives in `EvolveLab/config.py`.

## Adding a New Memory Provider

The unified memory interface is under `Flash-Searcher-main/EvolveLab/`.

### 1. Register the provider type

Add a unique entry to `MemoryType` in `EvolveLab/memory_types.py`:

```python
class MemoryType(Enum):
    YOUR_MEMORY = "your_memory"
```

The enum value is the string passed to `--memory_provider`.

### 2. Register the provider implementation

Add the class and module mapping in `PROVIDER_MAPPING`:

```python
PROVIDER_MAPPING = {
    MemoryType.YOUR_MEMORY: ("YourMemoryProvider", "your_memory_provider"),
}
```

This expects:

```text
EvolveLab/providers/your_memory_provider.py
```

with a class named `YourMemoryProvider`.

### 3. Add default configuration

Add provider defaults in `EvolveLab/config.py`:

```python
MemoryType.YOUR_MEMORY: {
    "memory_dir": "./storage/your_memory",
    "top_k": 3,
    "return_on_in": False,
    "model": None,
}
```

At minimum, define storage, retrieval count, and whether memories should be
returned during the in-episode `MemoryStatus.IN` stage. Providers that need LLM
or embedding endpoints can follow the patterns used by `mem0`, `a_mem`,
`memos`, and `reasoning_bank`.

### 4. Map storage overrides if needed

If your provider does not read `memory_dir` directly, add a mapping in
`compute_storage_overrides()`:

```python
if p == MemoryType.YOUR_MEMORY:
    return {
        "db_path": os.path.join(root, "your_memory.json"),
        "index_dir": os.path.join(root, "index"),
    }
```

This keeps storage isolated when running different benchmark orders or repeated
experiments.

### 5. Implement `BaseMemoryProvider`

Create `EvolveLab/providers/your_memory_provider.py`:

```python
from typing import Any, Dict, Optional

from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import (
    MemoryRequest,
    MemoryResponse,
    MemoryItem,
    MemoryItemType,
    MemoryType,
    TrajectoryData,
)


class YourMemoryProvider(BaseMemoryProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(memory_type=MemoryType.YOUR_MEMORY, config=config or {})
        self.memory_dir = self.config.get("memory_dir", "./storage/your_memory")
        self.top_k = self.config.get("top_k", 3)
        self.model = self.config.get("model")

    def initialize(self) -> bool:
        return True

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        memories = [
            MemoryItem(
                id="example",
                content="A reusable search tactic or task-solving insight for the agent.",
                metadata={"source": "your_memory"},
                score=1.0,
                type=MemoryItemType.TEXT,
            )
        ]
        return MemoryResponse(
            memories=memories[: self.top_k],
            memory_type=self.memory_type,
            total_count=min(len(memories), self.top_k),
        )

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        return True, "memory ingested"
```

Useful references:

- `EvolveLab/providers/base_provider_template.py`
- `EvolveLab/providers/mem0_provider.py`
- `EvolveLab/providers/reasoning_bank_provider.py`

## Memory Interface Semantics

The search agent calls the provider through two methods:

### `provide_memory(request)`

Returns memory items that will be injected into the agent context.

Relevant request fields:

| Field | Description |
| --- | --- |
| `request.query` | Current task question. |
| `request.context` | Current agent execution context. |
| `request.status` | `MemoryStatus.BEGIN` or `MemoryStatus.IN`. |
| `request.additional_params["step_number"]` | Current agent step number. |

### `take_in_memory(trajectory_data)`

Writes new memory after an episode.

Relevant trajectory fields:

| Field | Description |
| --- | --- |
| `trajectory_data.query` | Task question. |
| `trajectory_data.trajectory` | Agent message and tool-use trajectory. |
| `trajectory_data.result` | Final answer or result. |
| `trajectory_data.metadata` | Metadata such as `task_id`, `status`, `is_correct`, and `full_query`. |

## Full-Run Scripts

Convenience scripts are available under `Flash-Searcher-main/`:

```text
run_full_pipeline_*.sh
```

Use `run_full_pipeline_mem0.sh` as the primary reference when creating a new
script. It includes the full command-building pattern, order handling, storage
isolation, logging, and post-run analysis flow. Other provider scripts are
useful for provider-specific defaults, but the mem0 script is the clearest
template for the complete CrossEp-Web run lifecycle.
