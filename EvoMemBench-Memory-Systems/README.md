# EvoMemBench Memory Systems

Localized forks of memory libraries used as dependencies of the EvoMemBench benchmarks.

## Contents

| Subdirectory | Import name | Description |
|---|---|---|
| `mem0/` | `mem0` | Production memory layer for LLM agents (dist name: mem0ai) |
| `A-mem/` | `agentic_memory` | Zettelkasten-style agentic memory with semantic evolution |
| `MemOS/` | `memos` | Memory Operating System with Qdrant vector backend |
| `MemoryOS/` | `memoryos` | Three-tier hierarchical memory (short / mid / long-term) |
| `MemoBrain/` | `memobrain` | Reasoning graph-based working memory with memorize/recall workflow (requires local vLLM server) |
| `memagent/` | _(server only)_ | `BytedTsinghua-SIA/RL-MemoryAgent-14B` served via vLLM (no Python package; server-side deployment only) |

## Installation

Each subdirectory is an independent pip-installable package. Install them one by one in editable mode from the repo root:

```bash
pip install -e ./mem0
pip install -e ./A-mem
pip install -e ./MemOS
pip install -e ./MemoryOS
pip install -e ./MemoBrain
```

### MemoBrain: additional setup

MemoBrain requires a locally running `TommyChien/MemoBrain-14B` instance served via vLLM. Start the server before using the package:

```bash
# Single GPU, default port 8001
bash MemoBrain/run_server.sh --gpus 0

# Two GPUs with tensor parallelism
bash MemoBrain/run_server.sh --gpus 0,1 --tp 2
```

See `MemoBrain/README.md` for the full list of server options.

### memagent: additional setup

memagent requires a locally running `BytedTsinghua-SIA/RL-MemoryAgent-14B` instance served via vLLM. Start the server before running evaluations:

```bash
# Two GPUs with tensor parallelism (default), port 8000
bash memagent/run_server.sh --gpus 1,2 --tp 2

# Single GPU (≥ 40 GB)
bash memagent/run_server.sh --gpus 0 --tp 1
```

See `memagent/README.md` for the full list of server options.

## Verification

After installation, the following imports should all succeed in a Python shell:

```python
from mem0 import Memory
from agentic_memory.memory_system import AgenticMemorySystem
from memos.configs.memory import MemoryConfigFactory
from memoryos import Memoryos
from memobrain import MemoBrain
```
