# CROSSEP-EMB: Cross-Episode Embodied AI Evaluation

`CROSSEP-EMB` is the embodied-AI component of EvoMemBench's cross-episode execution benchmark. It evaluates whether an agent memory system can accumulate reusable experience in ALFWorld tasks and transfer that experience across related task categories.

This directory contains:

- an ALFWorld HTTP environment server wrapper;
- evaluation scripts for memory-augmented ALFWorld agents;
- all-pairs in-environment and cross-environment experiment launchers;
- adapters for retrieval, long-term memory, and procedural memory baselines.

## Evaluation Workflow

The benchmark is organized around two phases:

1. **In-environment evaluation**: each ALFWorld task category builds and uses its own memory store.
2. **Cross-environment transfer**: memory learned from a source task category is reused in read-only mode on a different target category.

The all-pairs scripts run the six ALFWorld task categories as sources and targets, producing 6 in-environment runs and 30 source-to-target transfer runs.

## Repository Layout

```text
CROSSEP-EMB/
|-- agentenv-alfworld/          # ALFWorld HTTP environment service
|-- agentenv/                   # Agent environment client package
|-- scripts/
|   |-- eval_alfworld_with_memory.py
|   |-- generate_alfworld_report.py
|   |-- memory/                 # Memory backend adapters
|   |-- smoke_test_*.sh
|   `-- eval_alfworld_all_*_batch_all_pairs.sh
`-- README.md
```

## Setup

We recommend using two Conda environments: one for the ALFWorld server and one for benchmark execution.

### 1. Start the ALFWorld Server

```bash
conda create --name CROSSEP-EMB-server python=3.9
conda activate CROSSEP-EMB-server

cd agentenv-alfworld
bash ./setup.sh
alfworld --host 0.0.0.0 --port 36005
```

You can check that the server is running with:

```bash
curl http://localhost:36005/
# Expected response: This is environment AlfWorld.
```

If setup fails, see `agentenv-alfworld/README.md`, especially the known-issues section.

### 2. Create the Evaluation Environment

In a second terminal:

```bash
conda create --name CROSSEP-EMB python=3.10
conda activate CROSSEP-EMB

pip install tiktoken
pip install 'volcengine-python-sdk[ark]'
```

The Volcengine Ark SDK is required by the default batch-mode agent and by several memory adapters that call an LLM during memory extraction.

### 3. Optional Memory Backend Dependencies

Some memory baselines rely on vendored packages under `EvoMemBench-Memory-Systems/`. Install only the systems you plan to evaluate. The paths below are relative to this `CROSSEP-EMB/` directory.

```bash
# mem0: ChromaDB vector store + SQLite history
pip install -e ../../../EvoMemBench-Memory-Systems/mem0
pip install chromadb

# A-mem: Zettelkasten-style agentic memory
pip install -e ../../../EvoMemBench-Memory-Systems/A-mem

# MemoryOS: hierarchical short-, mid-, and long-term memory
pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi

# MemOS: general text memory with Qdrant support
pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
```

Other adapters in `scripts/memory/` may only need standard Python dependencies plus the API keys described below.

## Environment Variables

`scripts/eval_alfworld_with_memory.py` loads environment variables from `CROSSEP-EMB/.env`. Create this file before running evaluations.

```bash
# Default agent LLM path
DEEPSEEK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
# (Recommended) if you choose batch inference mode, please fill in this variable
BATCH_MODEL=...
# if you choose online inference mode, please fill in this variable
DEEPSEEK_API_KEY=...

# OpenAI-compatible endpoint used when --model is set
OPENAI_API_KEY=...
OPENAI_BASE_URL=...

# DashScope text-embedding-v4 endpoint used by embedding-based adapters
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Keep `.env` local and do not commit API keys.

## Running Evaluations

The main Python entry point is:

```bash
python scripts/eval_alfworld_with_memory.py
```

For most users, it is easier to start from the provided shell scripts.

Run a smoke test for one backend:

```bash
bash scripts/smoke_test_eval_alfworld_mem0_all_pairs.sh
```

Run a full all-pairs experiment:

```bash
bash scripts/eval_alfworld_all_mem0_batch_all_pairs.sh
```

The repository includes all-pairs scripts for the following memory types:

| Memory type | Script family |
| --- | --- |
| No memory | `*_no_memory_*.sh` |
| BM25 | `*_bm25_*.sh` |
| Qwen3 embedding | `*_qwen3_embedding_*.sh` |
| GraphRAG | `*_graphrag_*.sh` |
| mem0 | `*_mem0_*.sh` |
| A-mem | `*_amem_*.sh` |
| MemoryOS | `*_memoryos_*.sh` |
| MemOS | `*_memos_*.sh` |
| AgentKB | `*_agent_kb_*.sh` |
| AWM / agent workflow memory | `*_awm_*.sh` |
| Lightweight memory | `*_lightweight_*.sh` |
| SkillWeaver | `*_skillweaver_*.sh` |
| ACE | `*_ace_*.sh` |
| ReasoningBank | `*_reasoning_bank_*.sh` |

Use the `mem0` scripts as a reference when adding a new backend or adapting paths, API parameters, top-k values, and output directories.

## Outputs

All-pairs scripts create an output directory similar to:

```text
output/<run_name>/
  configs/        # Auto-generated memory configs for each task category
  memory/         # Persistent memory stores
  logs/           # Per-job stdout logs
  in_env/         # Six task-specific in-environment runs
  cross_env/      # Source-to-target transfer runs
  final_report.txt
```

`final_report.txt` summarizes the in-environment and cross-environment results. Raw episode-level outputs remain under the corresponding `in_env/` and `cross_env/` subdirectories.

## Adding a New Memory Backend

To evaluate a new memory system, add a task-specific adapter, register it in the memory factory, and create smoke/full run scripts.

### 1. Implement an Adapter

Create `scripts/memory/<your_memory>_adapter.py` and subclass `BaseMemory`:

```python
from .base import BaseMemory, MemoryCallStats


class YourMemory(BaseMemory):
    def __init__(
        self,
        read_only: bool = False,
        top_k: int = 3,
        store_path: str = "./storage/your_memory.json",
        **kwargs,
    ):
        super().__init__(memory_type="your_memory", read_only=read_only)
        self._top_k = top_k
        self._store_path = store_path

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]
        # Retrieve memories with query and append them to conversation[0]["content"].
        return conversation, MemoryCallStats()

    def update(
        self,
        conversation: list[dict],
        data_idx: int | None = None,
        reward: float | None = None,
    ) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        # Store the trajectory or extracted memories.
        # If the backend should learn only from successful episodes,
        # skip updates when reward is not None and reward < 1.
        return MemoryCallStats()

    def load_from_disk(self, **kwargs) -> None:
        # Optional reload hook used by --memory_load_args.
        return None
```

Adapter contract:

- `inject()` is called once after environment reset and before the first agent LLM call.
- `update()` is called once after each episode with the completed trajectory.
- Persistent data should be written under paths supplied in the runtime config, preferably inside the script-created `$MEMORY_DIR`.

### 2. Register the Adapter

Edit `scripts/memory/base.py`:

```python
if memory_type == "your_memory":
    from .your_memory_adapter import YourMemory
    return YourMemory(read_only=read_only, **kwargs)
```

Also update the unknown-type error message so `your_memory` appears in the supported list.

### 3. Define Runtime Configuration

Pass backend options through `--memory_config` as JSON. Avoid hard-coding experiment output paths or API keys inside the adapter.

A minimal config usually looks like:

```json
{
  "agent_id": "pick_and_place_simple",
  "top_k": 3,
  "store_path": "/abs/path/to/output/memory/your_memory_pick_and_place_simple.json",
  "ark_batch_config": {
    "api_key": "${DEEPSEEK_API_KEY}",
    "model": "${BATCH_MODEL}"
  },
  "embed": {
    "model": "text-embedding-v4",
    "api_key": "${DASHSCOPE_API_KEY}",
    "base_url": "${DASHSCOPE_BASE_URL}"
  }
}
```

Suggested defaults:

- For pure retrieval-augmented memory, start with `top_k=10`.
- When storing trajectory chunks, start with `chunk_size=1024`.
- For LLM-extracted or refined memories, start with `top_k=3`.

### 4. Add Run Scripts

Copy an existing smoke-test script and full all-pairs script, then update:

- `memory_type`;
- generated config fields;
- output directory name;
- backend-specific dependency checks;
- any top-k, chunking, embedding, or LLM parameters.

Run the smoke test before launching the full all-pairs evaluation.
