# RL-MemoryAgent Setup Walkthrough

## Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA 12.x driver (vLLM 0.8.5 does not support CUDA 13.x)
- Two GPUs with ≥ 24 GB VRAM each (or one GPU with ≥ 40 GB)

## Step 1 — Create the conda environment

```bash
# From EvoMemBench-Memory-Systems repo root
bash memagent/setup_env.sh
```

This creates the `vllm_memagent` conda env (Python 3.10) and installs:
- `torch==2.6.0` (cu124 wheel)
- `vllm==0.8.5`
- `transformers==4.51.3`
- `huggingface_hub`

The versions are pinned; do not upgrade vLLM without verifying CUDA compatibility.

## Step 2 — Download model weights

The model weights (~28 GB) should already be present in `./llms/` if the repo was set up correctly. Check with:

```bash
ls llms/RL-MemoryAgent-14B/*.safetensors
# expect 7 shards: model-00001-of-00007.safetensors ... model-00007-of-00007.safetensors
```

If not present, either re-run setup with prefetch:

```bash
bash memagent/setup_env.sh --prefetch
```

Or run the manual download script (uses hf-mirror.com, parallel wget):

```bash
bash memagent/download_model.sh
```

## Step 3 — Start the server

```bash
# From repo root (scripts resolve paths relative to themselves)
bash memagent/run_server.sh
# defaults: port 8000, GPUs 1,2, tp 2
```

Startup takes ~60–90 seconds (model load + CUDA graph capture). You will see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 4 — Smoke test

```bash
curl http://localhost:8000/v1/models
```

Expected response:

```json
{"object":"list","data":[{"id":"BytedTsinghua-SIA/RL-MemoryAgent-14B",...}]}
```

## Machine-specific notes (current setup)

- GPUs 1,2 are assigned to RL-MemoryAgent (port 8000); GPUs 3,4 are used by MemoBrain (port 8001).
- `vllm_memagent` env (vLLM 0.8.5) is shared between this server and MemoBrain's server on this machine — both work fine with it.
- Model is also available as a pre-resolved HF snapshot at `./llms/hub/models--BytedTsinghua-SIA--RL-MemoryAgent-14B/snapshots/<hash>/`; `run_server.sh` auto-detects and uses this path to avoid `huggingface_hub` version issues.
