# memagent — RL-MemoryAgent-14B vLLM Server

Local vLLM deployment of `BytedTsinghua-SIA/RL-MemoryAgent-14B`, a recurrent memory agent trained with reinforcement learning. No Python package is installed — this directory only contains the server scripts.

## Quick start

```bash
# From EvoMemBench-Memory-Systems repo root

# First-time setup (creates vllm_memagent conda env, installs pinned vLLM 0.8.5)
bash memagent/setup_env.sh

# Start the server (GPUs 1,2, port 8000)
bash memagent/run_server.sh

# Verify
curl http://localhost:8000/v1/models
```

## `run_server.sh` options

| Flag | Default | Description |
|---|---|---|
| `--port` | `8000` | Port to serve on |
| `--gpus` | `1,2` | `CUDA_VISIBLE_DEVICES` value |
| `--tp` | `2` | Tensor parallel size |
| `--model` | `BytedTsinghua-SIA/RL-MemoryAgent-14B` | Model name or local path |
| `--download-dir` | `$HF_HOME` or `./llms` | HuggingFace cache dir |
| `--max-model-len` | `32768` | Max context length (tokens) |
| `--gpu-util` | `0.90` | GPU memory utilization |
| `--max-num-seqs` | `32` | Max concurrent sequences |
| `--conda-env` | `vllm_memagent` | Conda env to activate |

## `setup_env.sh` options

| Flag | Default | Description |
|---|---|---|
| `--conda-env` | `vllm_memagent` | Conda env name to create |
| `--python` | `3.10` | Python version |
| `--download-dir` | `./llms` | HuggingFace cache dir |
| `--prefetch` | _(off)_ | Pre-download model weights |

## Client-side config (MemoryAgentBench)

Evaluations that call this server use the following agent config:

```yaml
# configs/agent_conf/RAG_Agents/memagent/Recurrent_memory_memagent-14b.yaml
agent_name: Recurrent_memory_memagent
model: deepseek-chat
temperature: 1.0
agent_chunk_size: 5000
memagent_model: BytedTsinghua-SIA/RL-MemoryAgent-14B
memagent_url: http://localhost:8000/v1
memagent_api_key: EMPTY
recurrent_max_new: 1024
```

The server must be running at `memagent_url` before launching an evaluation run.
