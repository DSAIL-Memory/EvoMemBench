# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MemoBrain is a localized fork of a reasoning graph-based working memory system for LLM agents. It runs entirely on locally deployed models via vLLM (no external API dependency) and is used as a dependency in the EvoMemBench benchmark suite.

## Setup & Running

### 0. First-time setup (one-off)

```bash
# From repo root (EvoMemBench-Memory-Systems/)
bash MemoBrain/setup_env.sh --prefetch
```

This creates the `memobrain` conda env, installs vLLM + MemoBrain, and downloads the model (~28 GB) into `./llms/` via hf-mirror.com. Steps are idempotent — safe to re-run if interrupted.

> **Note:** The `memobrain` conda env (vLLM 0.20.2) requires CUDA 13.x and will NOT work on this machine (driver supports ≤ CUDA 12.9). Use the `vllm_memagent` env instead for running the server (see below). The `vllm_memagent` env is set up via `../memagent/setup_env.sh` (vLLM 0.8.5, PyTorch 2.6+cu124).

### 1. Start the vLLM model server

**On this machine**, always use the `vllm_memagent` conda env (vLLM 0.8.5, PyTorch 2.6+cu124) and GPUs 3,4 (GPUs 1,2 are occupied by RL-MemoryAgent-14B on port 8000 — see `../memagent/`):

```bash
# From repo root (EvoMemBench-Memory-Systems/)
bash MemoBrain/run_server.sh --conda-env vllm_memagent --gpus 3,4
```

The server starts on port 8001. Startup takes ~60s (model load ~5s, torch.compile from cache ~11s, graph capture ~26s). Verify with:

```bash
curl http://localhost:8001/v1/models
```

Key options: `--port` (default: 8001), `--gpus` (default: 2,3), `--tp` (default: 2), `--model` (default: `TommyChien/MemoBrain-14B`), `--conda-env`.

**Known constraints on this machine:**
- GPUs 3,4 are the free ones; GPUs 1,2 run RL-MemoryAgent-14B on port 8000 (see `../memagent/run_server.sh`)
- `GPU_UTIL` is set to 0.85 (not 0.95) to avoid OOM during sampler warmup
- Model is cached at `./llms/hub/models--TommyChien--MemoBrain-14B/snapshots/<hash>/`; `run_server.sh` resolves the path directly to bypass `huggingface_hub` version issues in `vllm_memagent`

### 2. Install the package

```bash
pip install -e .
```

After installation, `from memobrain import MemoBrain` is available.

## Core Architecture

The package is under `src/` and exports a single public class `MemoBrain`.

**`src/memobrain.py` — `MemoBrain` class (main API)**
- Uses `AsyncOpenAI` pointed at the local vLLM server
- `init_memory(task)`: seeds the `ReasoningGraph` with a task node
- `await memorize(messages)`: groups messages into user/assistant pairs, calls the LLM to generate a `MemoryPatch` (JSON), and applies it to the graph
- `await recall()`: calls the LLM to produce flush/fold operations on the graph, then runs `_organize()` which replaces, summarizes, or drops message turns and returns a compressed message list
- `save_memory(path)` / `load_memory(path)`: JSON serialization of the graph

**`src/problem_tree.py` — `ReasoningGraph`**
- Directed graph of `GraphNode` objects. Node kinds: `"task"`, `"subtask"`, `"evidence"`, `"summary"`
- `apply_patch(patch_json, turn_ids)`: processes `add_node` and `add_edge` operations from a `MemoryPatch`
- `flush_node(id)`: marks a node inactive (its turns will be compressed in `_organize`)
- `fold_nodes(ids, notes, rationale)`: merges multiple nodes into a single `summary` node
- `pretty_print()`: renders the graph as a string fed to the LLM during both `memorize` and `recall`

**`src/schema.py` — Pydantic models**
- `MemoryPatch` contains lists of `AddNodeOp` and `AddEdgeOp` — this is the JSON structure the LLM must return during `memorize`
- `FlushAndFoldOp` / the recall response schema expects `{"flush_ops": [...], "fold_ops": [...]}` — both are parsed with `json.loads` directly from the LLM output

**`src/prompts.py` — System prompts**
- `MEMORY_SYS_PROMPT`: instructs the LLM to return a `MemoryPatch` JSON for each conversation pair
- `FLUSH_AND_FOLD_PROMPT`: instructs the LLM to return flush/fold ops for graph compression

## Key Behaviors to Know

- Both `memorize` and `recall` parse LLM responses as raw JSON (with markdown code-fence stripping). If the LLM returns malformed JSON, the operation raises an exception. Retry logic runs up to 3 times with 3-second sleeps.
- `_organize()` always protects the first 3 messages and last 4 messages from replacement/removal.
- The graph's `pretty_print()` output is passed directly as the user message to the LLM — changes to node representation affect both `memorize` and `recall` prompts.
- `memorize` increments `start_idx` by 2 per pair but uses a fixed initial offset — if a pair fails to apply, the index still advances, which can cause turn ID misalignment.

## Dependencies

- `openai>=1.0.0` — used as the async client for the local vLLM server
- `pydantic>=2.0.0` — schema validation

No test suite exists. The `examples/` directory contains integration-style scripts (`run_task.py`, `react_with_memory.py`) that serve as functional verification.
