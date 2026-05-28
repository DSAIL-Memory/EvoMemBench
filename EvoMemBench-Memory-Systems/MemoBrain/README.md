# MemoBrain

Localized fork of MemoBrain (reasoning graph-based working memory for LLM agents — memorize + recall/compress workflow), used as a dependency of the EvoMemBench benchmarks.

This fork runs entirely on locally deployed models via vLLM, with no dependency on external APIs.

## Setup

### 1. Start the vLLM server

MemoBrain requires a locally running `TommyChien/MemoBrain-14B` instance. Use the provided script:

```bash
bash run_server.sh [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--port` | `8001` | Port to serve on |
| `--gpus` | `0` | `CUDA_VISIBLE_DEVICES` |
| `--tp` | `1` | Tensor parallel size |
| `--model` | `TommyChien/MemoBrain-14B` | Model name or local path |
| `--download-dir` | `$HF_HOME` or `~/.cache/huggingface` | HuggingFace cache dir |
| `--max-model-len` | `32768` | Max context length (tokens) |
| `--gpu-util` | `0.95` | GPU memory utilization |
| `--conda-env` | _(none)_ | Conda env to activate before serving |

Examples:

```bash
# Single GPU, default port 8001
bash run_server.sh --gpus 0

# Two GPUs, tensor parallel 2
bash run_server.sh --gpus 5,6 --tp 2

# Custom port and download dir, activate a conda env
bash run_server.sh --port 8002 --gpus 4 --download-dir ./llms --conda-env vllm_env
```

### 2. Install the package

```bash
pip install -e ./MemoBrain
```

After installation, `from memobrain import MemoBrain` works in any Python environment.

## Usage

Instantiate `MemoBrain` by pointing it at the local server:

```python
from memobrain import MemoBrain

memory = MemoBrain(
    api_key="empty",
    base_url="http://localhost:8001/v1",
    model_name="TommyChien/MemoBrain-14B",
)
```

Core API:

| Method | Description |
|---|---|
| `memory.init_memory(task)` | Initialize graph with the task description |
| `await memory.memorize(messages)` | Ingest new conversation turns into the graph |
| `await memory.recall()` | Flush/fold the graph and return compressed messages |
| `memory.save_memory(path)` | Persist graph to JSON |
| `memory.load_memory(path)` | Restore graph from JSON |
