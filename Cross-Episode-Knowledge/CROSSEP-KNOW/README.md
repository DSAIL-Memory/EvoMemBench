# CROSSEP-KNOW

CROSSEP-KNOW is an evaluation pipeline for studying cross-episode memory systems on context-grouped CL-bench data. It is designed for settings where an agent processes multiple episodes that belong to the same context and should benefit from memory accumulated earlier in that context.

The pipeline keeps memory isolated by `context_id`:

- Samples from the same context are processed sequentially, so memory can accumulate over time.
- Different contexts can run in parallel.
- Each context gets an independent memory directory, preventing cross-context leakage.

The standard workflow has three stages:

1. `infer_context_memory.py` runs context-grouped inference with either a memory backend or a no-memory baseline.
2. `eval.py` grades model outputs with an LLM-as-judge binary score.
3. `analyze_comparison_new.py` optionally compares a memory run against a no-memory baseline for quality and cost.

## Repository Layout

```text
CL-bench_context_ge5.jsonl          # Default evaluation data
run_context_memory.sh               # Main three-stage pipeline entry point
infer_context_memory.py             # Context-grouped memory inference
eval.py                             # LLM-as-judge evaluation
analyze_comparison_new.py           # Baseline vs. memory comparison
cl_bench_memory/                    # Memory backend implementations
scripts/                            # Backend-specific run and smoke-test scripts
tier_table_kit/                     # Easy/Medium/Hard score table utilities
requirements.txt                    # Python dependencies
```

## Installation

From the project root:

```bash
conda create -n CROSSEP-KNOW python=3.12
conda activate CROSSEP-KNOW
pip install -r requirements.txt
```

Some external memory systems are expected to be installed from a sibling checkout, for example:

```bash
pip install -e ../../EvoMemBench-Memory-Systems/mem0       # provides mem0
pip install -e ../../EvoMemBench-Memory-Systems/A-mem      # provides agentic_memory
pip install -e ../../EvoMemBench-Memory-Systems/MemOS      # provides memos
pip install -e ../../EvoMemBench-Memory-Systems/MemoryOS   # provides memoryos
```

You can verify the optional backends with:

```bash
python -c "from mem0 import Memory; \
           from agentic_memory.memory_system import AgenticMemorySystem; \
           from memos.configs.memory import MemoryConfigFactory; \
           from memoryos import Memoryos; print('OK')"
```

## Required Data File

Before running any CROSSEP-KNOW pipeline or smoke-test script, place the default CL-bench data file at:

```text
Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl
```

## Configuration

Create a `.env` file in this directory or the project root. Do not commit real API keys.

```bash
# DashScope embeddings, used by the default embedding provider
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=<your_dashscope_api_key>

# DeepSeek / Volcengine Ark inference
DEEPSEEK_API_KEY=<your_deepseek_or_ark_api_key>
DEEPSEEK_BASE_URL=<your_deepseek_compatible_base_url>
BATCH_MODEL=<your_ark_batch_endpoint>

# OpenAI-compatible judge endpoint
OPENAI_API_KEY=<your_openai_compatible_api_key>
OPENAI_BASE_URL=<your_openai_compatible_base_url>
```

Credential selection rules:

- Inference models whose names start with `deepseek` use `DEEPSEEK_API_KEY`.
- If `BATCH_MODEL` is set, DeepSeek inference uses the Volcengine Ark batch endpoint and replaces the requested model with `BATCH_MODEL`.
- If `BATCH_MODEL` is not set, DeepSeek inference uses `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` through an OpenAI-compatible client.
- The default judge model is `gpt-5.1`, so evaluation uses `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- The default embedding provider is `dashscope`, which uses `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`.

## Quick Start

The recommended entry point is:

```bash
bash run_context_memory.sh [options]
```

This script runs inference, evaluation, and, unless skipped, comparison analysis.

Run a no-memory baseline:

```bash
bash run_context_memory.sh \
    --no-memory \
    --workers 50 \
    --skip-compare
```

Run a small memory smoke test:

```bash
bash run_context_memory.sh \
    --memory-type reasoning_bank \
    --max-samples 2 \
    --workers 1 \
    --eval-workers 1 \
    --top-k 1 \
    --skip-compare
```

Run a full memory experiment:

```bash
bash run_context_memory.sh \
    --memory-type mem0 \
    --top-k 10 \
    --workers 5
```

Compare against a specific no-memory baseline:

```bash
bash run_context_memory.sh \
    --memory-type reasoning_bank \
    --top-k 10 \
    --baseline outputs/context_nomemory/CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl
```

Backend-specific scripts under `scripts/` provide ready-to-edit examples, such as `scripts/mem0_compare.sh`.

## Data and Outputs

The default input file is:

```text
CL-bench_context_ge5.jsonl
```

This file is required locally before running the benchmark.

Evaluation and comparison artifacts:

```text
<raw_inference_stem>_graded.jsonl
<raw_inference_stem>_comparison_report.md
```

## Supported Memory Backends

The memory factory currently supports:

```text
mem0
bm25
qwen3_embedding_4b
graphrag
reasoning_bank
ace
agent_kb
agent_workflow
lightweight
skillweaver
a_mem
memos
memoryos
```

Some backends require optional external packages. If a backend is unavailable, the factory raises an installation hint for that backend.

## Main Options

`run_context_memory.sh` is the preferred interface for routine experiments.

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `CL-bench_context_ge5` | Dataset basename without `.jsonl` |
| `--infer-model` | `deepseek-v3-2-251201` | Model under evaluation |
| `--judge-model` | `gpt-5.1` | LLM-as-judge model |
| `--workers` | `50` | Number of parallel contexts during inference |
| `--eval-workers` | `50` | Number of parallel evaluation workers |
| `--top-k` | `10` | Memory retrieval top-k |
| `--memory-type` | `mem0` | Memory backend |
| `--max-samples` | empty | Limit total samples for debugging |
| `--baseline` | `outputs/context_nomemory/CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl` | Baseline graded file for comparison |
| `--skip-compare` | `false` | Skip comparison analysis |
| `--no-memory` | `false` | Disable retrieve, inject, and extract |

## Easy / Medium / Hard Scores

`eval.py` produces per-sample binary scores. To aggregate scores by Easy, Medium, and Hard difficulty tiers, run:

```bash
python3 tier_table_kit/generate_difficulty_table.py <results_folder>
```

For example:

```bash
python3 tier_table_kit/generate_difficulty_table.py outputs/context_mem0
```

This writes:

```text
outputs/context_mem0/difficulty_score_table.tsv
```

The TSV can be pasted into a spreadsheet. It includes the four CL-bench categories and Overall scores split into `Easy`, `Medium`, and `Hard`. The tier mapping is defined in `tier_table_kit/difficulty_tiers.csv`; see `tier_table_kit/README.md` for details.

## Adding a New Memory Backend

All memory systems are called through the `Memory` interface in `cl_bench_memory/base.py`.

### 1. Implement the Backend

Create a file such as:

```text
cl_bench_memory/new_backend_memory.py
```

Implement a `Memory` subclass. The constructor should accept the standard arguments passed by the pipeline, even if the backend ignores some of them.

```python
import os
import time

from .base import Memory


class NewBackendMemory(Memory):
    def __init__(
        self,
        client=None,
        model=None,
        memory_dir=None,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=10,
        extract_model=None,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.extract_model = extract_model or model
        self.memory_dir = memory_dir
        self.embed_client = embed_client or client
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.memory_bank = []

        os.makedirs(memory_dir, exist_ok=True)

    def retrieve(self, query: str) -> tuple:
        t0 = time.time()
        memory_text = ""
        return memory_text, {
            "latency_s": time.time() - t0,
            "embed_usage": {},
        }

    def extract(self, content: str, **kwargs) -> dict:
        t0 = time.time()
        return {
            "latency_s": time.time() - t0,
            "llm_usage": {},
            "embed_usage": {},
        }
```

### 2. Register the Backend

Update `cl_bench_memory/registry.py`:

```python
from .new_backend_memory import NewBackendMemory


def build_memory(memory_type: str, **kwargs):
    if memory_type == "new_memory":
        return NewBackendMemory(**kwargs)
```

### 3. Export the Backend

Update `cl_bench_memory/__init__.py`:

```python
from .new_backend_memory import NewBackendMemory

__all__ = [
    ...
    "NewBackendMemory",
]
```

For optional-dependency backends, use the same guarded-export style as the existing optional backends.

### 4. Add a Run Script

Add a backend-specific script:

```text
scripts/new_memory_compare.sh
```

Use `scripts/mem0_compare.sh` as a template.

### 5. Run a Smoke Test

```bash
bash run_context_memory.sh \
    --memory-type new_memory \
    --max-samples 2 \
    --workers 1 \
    --eval-workers 1 \
    --top-k 1 \
    --skip-compare
```
