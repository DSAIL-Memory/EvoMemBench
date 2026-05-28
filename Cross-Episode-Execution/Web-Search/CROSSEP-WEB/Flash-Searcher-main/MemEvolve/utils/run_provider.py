#!/usr/bin/env python
# coding=utf-8

"""
run_provider for AutoEvolver.

Invokes dataset-specific runner scripts and ensures logs are in the expected format.
All runner scripts use eval_utils.create_run_directory which creates nested directories
like: base_dir/dataset_runs/provider_timestamp/*.json

This function copies those logs to output_dir/*.json for MemEvolve analysis compatibility.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import List

from ..config import DEFAULT_RUNNERS


def run_provider(dataset_name: str, provider_name: str, task_indices: List[int], dataset_file: Path, output_dir: Path) -> Path:
    """
    Invoke dataset-specific runner scripts to evaluate a provider.

    Args:
        dataset_name: Dataset name (gaia / webwalkerqa / xbench / coin_flip)
        provider_name: Memory provider name
        task_indices: List of integer indices to run (0-based)
        dataset_file: Path to source dataset file (jsonl/csv)
        output_dir: Directory where logs should be placed (*.json files directly)

    Returns:
        output_dir path (containing *.json task logs)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_arg = ",".join(str(i + 1) for i in task_indices)
    runner = DEFAULT_RUNNERS.get(dataset_name)
    if runner is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    outfile = output_dir / "results.jsonl"
    runner_path = Path(__file__).resolve().parent.parent.parent / runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")
    
    cmd = [
        "python",
        str(runner_path),
        "--infile",
        str(dataset_file),
        "--outfile",
        str(outfile),
        "--task_indices",
        task_arg,
        "--memory_provider",
        provider_name,
        "--concurrency",
        "1",
        "--direct_output_dir",
        str(output_dir),
    ]
    
    print(f"\n[Runner] Executing: {' '.join(cmd[:2])} ...")
    print(f"[Runner] Working directory: {Path.cwd()}")
    print(f"[Runner] Output will be displayed below:\n")
    print("-" * 60)
    
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    
    process.wait()
    
    print("-" * 60)
    
    if process.returncode != 0:
        print(f"\n[Runner] Error occurred (exit code: {process.returncode})")
        raise RuntimeError(f"Runner failed ({dataset_name}) with exit code {process.returncode}")
    
    print(f"\n[Runner] Execution completed successfully")
    
    json_files = list(output_dir.glob("*.json"))
    if json_files:
        print(f"[Runner] Found {len(json_files)} task logs in output_dir")
        return output_dir
    
    print(f"[Runner] No JSON files in output_dir, attempting fallback copy...")
    outfile_parent = outfile.parent
    runs_subdir = outfile_parent / f"{dataset_name}_runs"
    
    if runs_subdir.exists():
        run_dirs = sorted([d for d in runs_subdir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime)
        if run_dirs:
            actual_logs_dir = run_dirs[-1]
            print(f"[Runner] Found logs in: {actual_logs_dir}")
            print(f"[Runner] Copying to: {output_dir}")
            
            for json_file in actual_logs_dir.glob("*.json"):
                target_file = output_dir / json_file.name
                shutil.copy2(json_file, target_file)
                print(f"[Runner] Copied: {json_file.name}")
            
            json_count = len(list(output_dir.glob("*.json")))
            print(f"[Runner] Total JSON files in output_dir: {json_count}")
    
    return output_dir
