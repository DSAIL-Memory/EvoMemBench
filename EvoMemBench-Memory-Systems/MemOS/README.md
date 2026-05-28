# MemOS

Localized fork of MemOS (Memory Operating System for LLMs — general text memory with Qdrant vector backend), used as a dependency of the EvoMemBench benchmarks.

## Installation

Clone this repository, then install it in editable mode from the repo root:

```bash
git clone git@github.com:IcohM/MemOS.git
cd MemOS
pip install -e .
```

After installation, `from memos.configs.memory import MemoryConfigFactory` should work in your Python environment.

> **Note**: the PyPI package name is `MemoryOS` (historical upstream naming). Our sibling repo `IcohM/MemoryOS` uses `memoryos-pypi` as its PyPI name to avoid collision, so the two can coexist.
