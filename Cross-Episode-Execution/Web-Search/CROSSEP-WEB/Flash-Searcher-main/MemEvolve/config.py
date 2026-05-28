#!/usr/bin/env python
# coding=utf-8

"""
Central configuration for MemEvolve.

Includes:
- Dataset mappings for quick selection by name.
- Common magic numbers used across phases.
"""

from pathlib import Path

# Dataset name -> task log directory
# Dataset name -> source file for evaluation runners
DEFAULT_DATASETS = {
    "gaia": "data/gaia/validation/metadata.jsonl",
    "webwalkerqa": "data/webwalkerqa/webwalkerqa_subset_170.jsonl",
    "xbench": "data/xbench/DeepSearch.csv",
    "taskcraft": "data/taskcraft/sampled_dataset.jsonl",
    "coin_flip": "data/coin_flip/coin_flip_test.jsonl",
}

# Runner entry scripts per dataset
DEFAULT_RUNNERS = {
    "gaia": "run_flash_searcher_mm_gaia.py",
    "webwalkerqa": "run_flash_searcher_webwalkerqa.py",
    "xbench": "run_flash_searcher_mm_xbench.py",
    "taskcraft": "run_flash_searcher_mm_taskcraft.py",
    "coin_flip": "run_coin_flip.py",
}

# Validation / smoke test limits
SMOKE_TASK_LIMIT = 5  # number of tasks to sample per dataset during validation
IMPORT_TIMEOUT_SEC = 10  # provider import timeout

# Trajectory tools
TRAJECTORY_VIEWER_MAX_TASKS = 3
TRAJECTORY_SUMMARY_TEMPERATURE = 0.3
MEMORY_DB_MAX_LINES = 200
MEMORY_DB_SAMPLE_LINES = 100
MEMORY_DB_JSON_LIKE_THRESHOLD = 10

# Analyzer / Generator defaults
ANALYSIS_MAX_STEPS = 20
CREATIVITY_INDEX = 0.5
# Map creativity_index (0-1) to temperature = BASE + SPAN * idx
CREATIVITY_TEMP_BASE = 0.3
CREATIVITY_TEMP_SPAN = 0.9

# Evolution workflow defaults
EVOLVE_TASK_BATCH_X = 20         # each round: run x tasks per provider
EVOLVE_TOP_T = 2                 # select top t systems after first evaluation
EVOLVE_EXTRA_SAMPLE_Y = 5        # reuse y tasks from first batch for finalists
EVOLVE_GENERATED_M = 3           # number of systems to generate per round (m)
EVOLVE_RANDOM_SEED = 42