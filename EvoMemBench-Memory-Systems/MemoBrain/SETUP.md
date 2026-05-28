# MemoBrain Environment Setup

This document covers how to create the Python environment and install the dependencies needed to run `bash run_server.sh`.

## Hardware Requirements

| Requirement | Minimum |
|---|---|
| GPU | NVIDIA, compute capability ≥ 8.0 (Ampere or newer) |
| VRAM (single GPU) | ≥ 40 GB (e.g. A100 40 GB, A6000 48 GB, H100) |
| VRAM (two GPUs) | 2 × 24 GB with `--gpus 0,1 --tp 2` |
| Disk (model cache) | ~30 GB free under `$HF_HOME` (default `~/.cache/huggingface`) |
| CUDA driver | ≥ 525 (compatible with CUDA 12.1 PyTorch wheels) |

**Why Ampere?** `run_server.sh` hard-codes `--dtype bfloat16`. Native bfloat16 compute requires compute capability ≥ 8.0. Older cards (V100, T4) will error at startup.

**VRAM budget:** `TommyChien/MemoBrain-14B` in bf16 ≈ 28 GB of weights. The script's defaults (`--gpu-util 0.95`, `--max-model-len 32768`) leave little headroom, so a single 40 GB card is the comfortable single-GPU minimum. A single 24 GB card will OOM; use two with tensor parallelism instead.

## Step 1 — Create the conda environment

```bash
conda create -n memobrain python=3.10 -y
conda activate memobrain
```

Python 3.8–3.11 all work (per `pyproject.toml`), but 3.10 matches vLLM's best-tested configuration.

## Step 2 — Install server dependencies

```bash
pip install --upgrade pip
pip install vllm
```

`pip install vllm` pulls in a CUDA-enabled PyTorch, `transformers`, and `huggingface_hub` automatically. Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
```

## Step 3 — Install the MemoBrain package

From the repo root:

```bash
pip install -e ./MemoBrain
```

This installs the client-side deps (`openai`, `pydantic`) declared in `pyproject.toml` and makes `from memobrain import MemoBrain` available.

## Step 4 — (Optional) Pre-download the model

If the serve host has limited internet access at runtime, download the model weights once in advance:

```bash
huggingface-cli download TommyChien/MemoBrain-14B
```

`huggingface-cli` is installed with `huggingface_hub` (a vLLM dependency). The weights land in `$HF_HOME`. When you pass `--download-dir` to `run_server.sh`, it exports that path as `$HF_HOME`, so the pre-downloaded snapshot is found automatically.

## Step 5 — Launch the server

```bash
# Single GPU (≥ 40 GB VRAM), activate conda env inside the script
bash MemoBrain/run_server.sh --conda-env memobrain --gpus 0

# Two GPUs with tensor parallelism (2 × 24 GB)
bash MemoBrain/run_server.sh --conda-env memobrain --gpus 0,1 --tp 2

# Custom port and download directory
bash MemoBrain/run_server.sh --conda-env memobrain --gpus 0 --port 8002 --download-dir /data/llms
```

`--conda-env memobrain` tells the script to activate the environment before calling Python — safe to use even from a non-conda shell.

## Verification

When the server is ready it logs:

```
Uvicorn running on http://0.0.0.0:8001
```

From another terminal, confirm it responds:

```bash
curl http://localhost:8001/v1/models
```

Expected: JSON listing `TommyChien/MemoBrain-14B`.

End-to-end Python smoke test:

```python
from memobrain import MemoBrain

mem = MemoBrain(
    api_key="empty",
    base_url="http://localhost:8001/v1",
    model_name="TommyChien/MemoBrain-14B",
)
```

No exception means both the package install and the server connection are working.

---

See [`README.md`](README.md) for the full `run_server.sh` flag reference and the Python API (`memorize`, `recall`, `save_memory`, `load_memory`).
