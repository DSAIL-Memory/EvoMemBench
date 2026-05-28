# MemoryOS

Localized fork of MemoryOS (three-tier hierarchical memory: short-term / mid-term / long-term), used as a dependency of the EvoMemBench benchmarks.

## Installation

Clone this repository, then install it in editable mode from the repo root:

```bash
git clone git@github.com:IcohM/MemoryOS.git
cd MemoryOS
pip install -e .
```

After installation, `from memoryos import Memoryos` should work in your Python environment.

> **Note**: the PyPI package name is `memoryos-pypi` (to avoid collision with MemOS, which historically registers the name `MemoryOS` on PyPI). The import name `memoryos` is preserved.
