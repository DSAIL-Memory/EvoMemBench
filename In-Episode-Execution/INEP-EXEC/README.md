# BFCL In-Episode Execution

This directory contains the in-episode execution benchmark used by EvoMemBench for evaluating how language models and memory systems behave within a single multi-turn task episode. It extends the Berkeley Function Calling Leaderboard (BFCL) evaluation flow with long-context settings, per-turn memory retrieval/update hooks, and scripts for running several memory backends under fixed context budgets.

Run commands below from this directory unless noted otherwise:

```text
In-Episode-Execution/INEP-EXEC
```

## What This Benchmark Evaluates

In-episode execution focuses on function-calling tasks where a model must keep track of user goals, tool results, and prior turns while completing one episode. The benchmark can run:

- no-memory baselines;
- retrieval-style memory systems that inject relevant memories into the latest user message;
- compression-style memory systems that maintain a compact working context;
- long-context variants with context budgets such as 8k, 16k, 32k, 64k, and 128k.

The benchmark is split into four task subsets:

- `gorilla_fs`
- `vehicle_control`
- `trading_bot`
- `travel_api`

## Installation

Create a dedicated Python environment and install the package in editable mode:

```bash
conda create -n INEP-EXEC python=3.11 -y
conda activate INEP-EXEC

pip install -e .
pip install volcengine-python-sdk[ark]
```

Bundled evaluation scripts activate the `INEP-EXEC` conda environment by default.

## Optional Memory System Dependencies

The no-memory baseline does not require memory-system packages. To evaluate external memory backends, install the corresponding packages from the repository-level memory systems directory:

```bash
pip install -e ../../EvoMemBench-Memory-Systems/mem0
pip install -e ../../EvoMemBench-Memory-Systems/A-mem
pip install -e ../../EvoMemBench-Memory-Systems/MemOS
pip install -e ../../EvoMemBench-Memory-Systems/MemoryOS
```

You can verify the main imports with:

```bash
python -c "from mem0 import Memory; \
           from agentic_memory.memory_system import AgenticMemorySystem; \
           from memos.configs.memory import MemoryConfigFactory; \
           from memoryos import Memoryos; print('OK')"
```

## Data Layout

Benchmark data is stored under:

```text
bfcl_eval/data
```

Key files:

| Path | Purpose |
| --- | --- |
| `bfcl_eval/data/BFCL_v4_multi_turn_ours.json` | Main in-episode test set. |
| `bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_ours.json` | Reference answers. |
| `bfcl_eval/data/multi_turn_func_doc/*.json` | Function documentation for each tool/API category. |
| `bfcl_eval/scripts/in_episode/ids/*.json` | Sample-id lists for the four task subsets. |

## Configuration

Create a local `.env` file in this directory. A template is available at:

```text
bfcl_eval/.env.example
```

Common variables:

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | API key for the Ark/DeepSeek batch endpoint used by the main model. |
| `BATCH_MODEL` | Batch inference model or endpoint name. |
| `DASHSCOPE_API_KEY` | Required by several embedding-backed memory systems. |
| `DASHSCOPE_BASE_URL` | DashScope OpenAI-compatible base URL. |
| `BFCL_MEMORY_MODEL` | Optional memory-side LLM name. Defaults to `BATCH_MODEL` when unset. |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint settings for supported models or embeddings. |
| `MEMOBRAIN_VLLM_URL`, `MEMOBRAIN_MODEL` | Optional MemoBrain service endpoint and model name. |

## Quick Start

Run a small no-memory smoke test:

```bash
bash scripts/run_eval_inepisode_fc_8k_lc.sh \
  --limit-per-subset 1 \
  --workers 2 \
  --run-name smoke_no_memory_8k
```

Run the 8k mem0 in-episode script:

```bash
bash scripts/run_eval_mem0_inepisode_fc_8k_lc.sh \
  --limit-per-subset 2 \
  --workers 2 \
  --run-name smoke_mem0_8k
```

Results are written under `in_episode_results/`. Multi-subset scripts also create merged outputs such as:

```text
combined_summary.txt
combined_summary.csv
combined_per_sample.csv
```

## Running Evaluations

Reusable scripts live in:

```text
scripts
```

Script names follow this pattern:

```text
scripts/run_eval_<memory_type>_inepisode_fc_<budget>_lc.sh
```

Examples:

```bash
bash scripts/run_eval_bm25_inepisode_fc_8k_lc.sh
bash scripts/run_eval_mem0_inepisode_fc_32k_lc.sh
bash scripts/run_eval_reasoning_bank_inepisode_fc_64k_lc.sh
bash scripts/run_eval_inepisode_fc_128k_lc.sh
```

Useful flags supported by the subset-parallel scripts include:

| Flag | Description |
| --- | --- |
| `--limit-per-subset N` | Run only the first `N` samples from each subset. |
| `--workers N` | Number of workers per subset. |
| `--run-name NAME` | Human-readable run name used in output paths. |
| `--context-budget N` | Token budget per LLM call. |
| `--keep-per-sample-dir` | Keep per-sample memory stores after the run. |
| `--keep-going` | Merge completed subset summaries even if a subset fails. |

You can also call the Python entry point directly:

```bash
python -m bfcl_eval.scripts.in_episode.run_batch_eval \
  --num-workers 2 \
  --output-dir in_episode_results \
  --run-name direct_smoke \
  --memory-type none \
  --context-budget 8000 \
  --mode FC \
  --limit 4
```

Supported `--memory-type` values include:

```text
none, mem0, memagent, memobrain, memos, memoryos, amem,
agent_kb, agent_workflow, skillweaver, lightweight,
bm25, qwen3_embedding, graphrag, ace, reasoning_bank
```

## Memory Backend Lifecycle

Retrieval-style memory backends implement `bfcl_eval.memory.base.Memory`. During an evaluation:

1. `begin_sample()` resets memory usage metrics at the start of each sample.
2. `utilize(user_text)` runs before each model call. If it returns non-empty text, the text is appended to the latest user message.
3. `update(turn_delta)` runs after each turn and receives the user, assistant, and tool-message delta for that turn.
4. `drain_usage()` runs at sample end. Its output is stored in per-sample JSON fields such as `memory_usage` and `memory_totals`.

The summary logic recognizes these memory-cost fields:

```text
latency_s, input_tokens, output_tokens, embedding_tokens,
n_llm_calls, n_embed_calls, n_utilize, n_update
```

Compression-style backends such as `memagent` and `memobrain` do not follow the regular per-turn retrieval injection path. They are handled in `bfcl_eval/model_handler/base_handler.py` through the compression branch that maintains `_compressed_messages`.

## Adding a Memory Backend

Adding a new retrieval-style memory backend usually requires four changes.

### 1. Implement the backend

Create a backend under `bfcl_eval/memory/` and implement at least `utilize()`, `update()`, and `load_from_disk()`:

```python
from pathlib import Path

from bfcl_eval.memory.base import Memory


class NewMemory(Memory):
    def __init__(
        self,
        storage_dir: Path | str,
        top_k: int = 3,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ):
        super().__init__(memory_type="new_memory", clear=clear, readonly=readonly)
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k

        if clear:
            # Reset this backend's persistent files or indexes here.
            pass

    def utilize(self, user_text: str) -> str:
        # Return memory text to append to the current user message.
        return ""

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        # Store the current turn delta.

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "NewMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
```

### 2. Register the backend

Add the backend to `bfcl_eval/memory/factory.py`:

```python
if memory_type == "new_memory":
    from bfcl_eval.memory.new_memory_backend import NewMemory
    return NewMemory(**kwargs)
```

Also update the supported-values error message.

### 3. Wire the CLI arguments

Update `bfcl_eval/scripts/in_episode/run_batch_eval.py`:

- add the new value to `--memory-type choices`;
- add the required setup branch for API keys, model names, embeddings, or local services;
- pass shared constructor arguments through `memory_build_kwargs`.

For local-only retrieval backends, the default constructor inputs are usually enough:

```text
storage_dir, clear, readonly
```

For embedding-backed or LLM-backed systems, follow the existing `mem0`, `reasoning_bank`, `skillweaver`, or `ace` branches as templates.

### 4. Add a run script

Add a script under `scripts/` using the naming convention:

```text
scripts/run_eval_<memory_type>_inepisode_fc_<budget>_lc.sh
```

The existing `mem0` scripts are useful templates for per-sample isolated stores, subset-parallel execution, and summary merging.

## Development Notes

- Keep per-sample memory isolated when evaluating in-episode behavior only.
- Use `--memory-scope sample` when backend state should start empty for every sample.
- Use `--resume` for direct Python runs that should skip completed sample result files.
- Use `--dry-run` to inspect selected sample ids without calling the model API.
- For compression-style systems, update both the backend/factory path and the compression logic in `bfcl_eval/model_handler/base_handler.py`.
