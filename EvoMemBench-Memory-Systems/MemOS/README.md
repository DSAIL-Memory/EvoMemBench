# MemOS

Localized fork of MemOS (Memory Operating System for LLMs — general text memory with Qdrant vector backend), used as a dependency of the EvoMemBench benchmarks.

## Installation

Clone this repository, then install it in editable mode from the repo root:

```bash
cd MemOS
pip install -e .
```

After installation, `from memos.configs.memory import MemoryConfigFactory` should work in your Python environment.

