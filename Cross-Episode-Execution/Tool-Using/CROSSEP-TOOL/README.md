# CROSSEP-TOOL

CROSSEP-TOOL is the tool-use track of EvoMemBench's cross-episode execution benchmark. This package builds on the Berkeley Function Calling Leaderboard (BFCL) multi-turn evaluation stack and adds:

- cross-episode memory setup and reuse;
- in-environment and cross-environment evaluation scripts;
- adapters for multiple memory backends;

## Quick Start

Create a dedicated environment from the repository root:

```bash
conda create -n CROSSEP-TOOL python=3.11 -y
conda activate CROSSEP-TOOL

# Install this package in editable mode.
pip install -e .

# Required by the batch evaluator.
pip install volcengine-python-sdk[ark]

# Runtime dependencies used by the cross-episode runner.
pip install python-dotenv overrides tenacity filelock "tree_sitter==0.21.3" \
    "tree-sitter-java==0.21.0" "tree-sitter-javascript==0.21.4" "httpx[socks]" \
    networkx
```

Configure API credentials by copying `bfcl_eval/.env.example` to `bfcl_eval/.env` and filling in the keys required by the model and memory backend you plan to evaluate.

At minimum, batch inference requires:

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | API key for the Ark/DeepSeek batch endpoint. |
| `BATCH_MODEL` | Batch inference endpoint or model name. |

Some memory backends also require:

| Variable | Purpose |
| --- | --- |
| `DASHSCOPE_API_KEY` | Embeddings and LLM calls for several memory systems. |
| `DASHSCOPE_BASE_URL` | DashScope OpenAI-compatible endpoint. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Optional OpenAI-compatible embedding or model endpoint. |

## Optional Memory-System Dependencies

Skip this section if you only run `no_memory` or backends that do not depend on the external memory-system packages.

From `Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL`, install the optional local packages:

```bash
# mem0
pip install -e ../../../EvoMemBench-Memory-Systems/mem0

# AMem, imported as agentic_memory; uses ChromaDB.
pip install -e ../../../EvoMemBench-Memory-Systems/A-mem
pip install chromadb

# MemOS, imported as memos; package name is MemoryOS.
pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
pip install qdrant-client

# MemoryOS, imported as memoryos; package name is memoryos-pypi.
pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS
```

You can confirm that the intended `mem0` source tree is active with:

```bash
conda run -n CROSSEP-TOOL python -c "import mem0, os; print(os.path.dirname(mem0.__file__))"
```

The expected path should end with:

```text
EvoMemBench-Memory-Systems/mem0/mem0
```

## Running Evaluations

The `*_inenv_allcrossenv_fc.sh` scripts run CROSSEP-TOOL in two phases:

1. **Phase 1: in-environment episodes.** The runner evaluates four source subsets and builds a separate memory bank for each source environment.
2. **Phase 2: cross-environment transfer.** The runner evaluates all 12 ordered source-target environment pairs using read-only copies of the Phase 1 memory banks.

Run the fastest sanity check first:

```bash
bash scripts/run_smoke_tests.sh --only no_memory --limit 1
```

Then run a full backend evaluation. For example, the `mem0` script defaults to `top_k=3`:

```bash
bash scripts/run_eval_mem0_inenv_allcrossenv_fc.sh
```

Other backend scripts are available under `scripts/`, including:

- `run_eval_no_memory_inenv_fc.sh`
- `run_eval_bm25_inenv_allcrossenv_fc.sh`
- `run_eval_qwen3_embedding_inenv_allcrossenv_fc.sh`
- `run_eval_reasoning_bank_inenv_allcrossenv_fc.sh`
- `run_eval_ace_inenv_allcrossenv_fc.sh`
- `run_eval_graphrag_inenv_allcrossenv_fc.sh`
- `run_eval_amem_inenv_crossenv_full_fc.sh`
- `run_eval_memos_inenv_crossenv_full_fc.sh`
- `run_eval_memoryos_inenv_crossenv_full_fc.sh`

Results are written to:

```text
cross_episode_results/runs/<run-name>_<memory-type>_<timestamp>/
```

Each run directory contains Phase 1 outputs, Phase 2 pair outputs, copied memory banks, and merged summary files.

## Repository Map

All paths below are relative to `Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL`.

| Path | Description |
| --- | --- |
| `bfcl_eval/data/BFCL_v4_multi_turn_ours.json` | Main multi-turn prompt and test entries. |
| `bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_ours.json` | Ground-truth answers used for success and progress scoring. |
| `bfcl_eval/data/multi_turn_func_doc/*.json` | Environment-specific API/function documentation loaded from `involved_classes`. |
| `bfcl_eval/scripts/cross_episode/run_batch_generate.py` | Generates model outputs and writes `result.jsonl`; most memory-backend runtime wiring lives here. |
| `bfcl_eval/scripts/cross_episode/run_batch_evaluate.py` | Reads `result.jsonl` and writes per-sample JSON plus summaries. |
| `bfcl_eval/memory/base.py` | Abstract `Memory` interface implemented by all memory backends. |
| `bfcl_eval/memory/factory.py` | Maps `--memory-type` values to concrete backend classes. |
| `scripts/run_eval_*_fc.sh` | End-to-end launch scripts for supported backends. |
| `scripts/merge_summaries.py` | Merges Phase 1 and Phase 2 result summaries. |

## Result Layout

A completed run has the following structure:

```text
cross_episode_results/runs/<run-name>_<memory-type>_<timestamp>/
|-- phase1/
|   `-- <source-subset>/
|       |-- memory/
|       |-- result.jsonl
|       `-- ...
|-- phase2/
|   `-- <source>__to__<target>/
|       |-- result.jsonl
|       `-- ...
|-- _banks/
|   `-- <source>__to__<target>/
|-- combined_summary.txt
|-- combined_summary.csv
`-- combined_per_sample.csv
```

`phase1/<subset>/memory/` stores the memory bank learned from a source subset. Phase 2 reads from copied banks under `_banks/` so that parallel source-target evaluations do not share a writable storage directory.

## Adding a Memory Backend

To integrate a new memory system, implement the backend, register it, add a launch script, and run smoke tests.

### 1. Implement the backend

Create `bfcl_eval/memory/<new_backend>.py` and inherit from `bfcl_eval.memory.base.Memory`:

```python
from pathlib import Path

from bfcl_eval.memory.base import Memory


class NewBackendMemory(Memory):
    def __init__(
        self,
        *,
        storage_dir: Path | str,
        ark_api_key: str = "",
        ark_model: str = "",
        top_k: int = 3,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ) -> None:
        ...

    def utilize(self, system_prompt: str) -> str:
        ...

    def update(self, trajectory: list[dict]) -> None:
        ...

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "NewBackendMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
```

Useful references:

| Backend style | Reference implementation |
| --- | --- |
| JSON/JSONL bank | `bfcl_eval/memory/bm25_backend.py` |
| Local vector store | `bfcl_eval/memory/mem0_backend.py` |
| EvolveLab provider adapter | `bfcl_eval/memory/lightweight_memory.py` |

### 2. Register the memory type

Add the new type in `bfcl_eval/memory/factory.py`:

```python
if memory_type == "<new_type>":
    from bfcl_eval.memory.<new_backend> import NewBackendMemory

    return NewBackendMemory(**kwargs)
```

Also update the supported-value message for unknown backends.

Then update `bfcl_eval/scripts/cross_episode/run_batch_generate.py`:

- add `<new_type>` to the `--memory-type` choices;
- validate required environment variables;
- set backend-specific defaults;
- add reproducibility-critical fields to `memory_config_log`, such as `memory_top_k`, embedding model, storage directory, and `readonly`.

Common runner defaults:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--memory-type` | `none` | Disable memory. |
| `--memory-readonly` | `False` | Phase 1 writes memory; Phase 2 scripts enable read-only mode. |
| `--memory-top-k` | `3` | Number of retrieved memories for backends that use top-k retrieval. |
| `--mode` | `prompting` | Python entry default; repository scripts should pass `--mode FC`. |

### 3. Add a launch script

Create `scripts/run_eval_<new_type>_inenv_allcrossenv_fc.sh`.

Start from the closest existing script:

- embedding-based memory with LLM extraction: `scripts/run_eval_mem0_inenv_allcrossenv_fc.sh`
- pure retrieval/RAG: `scripts/run_eval_qwen3_embedding_inenv_allcrossenv_fc.sh`
- no top-k requirement: `scripts/run_eval_ace_inenv_allcrossenv_fc.sh`

Use `TOP_K=3` as the default unless the backend has a clear reason to ignore it.

### 4. Validate

First run a dry-run argument check:

```bash
python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
    --memory-type <new_type> \
    --limit 1 \
    --dry-run
```

Then run a minimal smoke test:

```bash
bash scripts/run_eval_<new_type>_inenv_allcrossenv_fc.sh \
    --limit-per-subset 1 \
    --run-name smoke_<new_type>
```

Check that:

- `run_config.json` records `memory.memory_type` as `<new_type>`;
- `phase1/<subset>/memory/` contains the backend's persistent files;
- `phase2/<source>__to__<target>/result.jsonl` exists;
- `combined_summary.csv` and `combined_per_sample.csv` are generated.
