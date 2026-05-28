#!/usr/bin/env python3
"""
Analyze per-question step counts from a pipeline run log file.

Phase 1 (sequential, concurrency=1): steps are parsed directly from log.
Phase 2 (parallel): steps are read from the JSONL result file (trajectory).

Usage:
    python analyze_steps.py <log_file> [log_file2 ...]

Examples:
    python analyze_steps.py pipeline_output/run_xbench_webwalkerqa_20260426_192850.log
    python analyze_steps.py pipeline_output/run_*.log
"""

import re
import sys
import os
import json
import argparse

STEP_PAT = re.compile(r'━+\s+Step\s+(\d+)\s+━+')
TASK_DONE_PAT = re.compile(r'INFO - Task done \[(\d+)/(\d+)\]:')
PHASE_START_PAT = re.compile(
    r'INFO - === Phase (\d+): (\S+) \((\d+) tasks, concurrency=(\d+)\) ==='
)
PHASE_DONE_PAT = re.compile(r'INFO - === Phase \d+ done')
OUTDIR_PAT = re.compile(r'INFO - Pipeline output directory: (.+)')


def parse_log(log_path: str) -> dict:
    """Parse a pipeline log file and return phase-level step stats."""
    phases = []          # list of {dataset, concurrency, total, steps_from_log}
    current_phase = None
    current_steps = 0    # steps accumulated since last Task done (Phase 1 only)
    outdir = None

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # Output directory
            m = OUTDIR_PAT.search(line)
            if m:
                outdir = m.group(1).strip()
                continue

            # Phase start
            m = PHASE_START_PAT.search(line)
            if m:
                current_phase = {
                    "phase_num": int(m.group(1)),
                    "dataset": m.group(2),
                    "total": int(m.group(3)),
                    "concurrency": int(m.group(4)),
                    "steps_per_task": [],   # populated during sequential phase
                }
                current_steps = 0
                phases.append(current_phase)
                continue

            # Phase done
            if PHASE_DONE_PAT.search(line):
                current_phase = None
                continue

            if current_phase is None:
                continue

            # Step counter (only meaningful in sequential phase)
            if current_phase["concurrency"] == 1 and STEP_PAT.search(line):
                current_steps += 1
                continue

            # Task done
            m = TASK_DONE_PAT.search(line)
            if m:
                if current_phase["concurrency"] == 1:
                    current_phase["steps_per_task"].append(current_steps)
                    current_steps = 0

    return {"outdir": outdir, "phases": phases}


def steps_from_jsonl(jsonl_path: str) -> list[int]:
    """Read per-task action step counts from a JSONL result file."""
    counts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            traj = d.get("agent_trajectory", [])
            counts.append(sum(1 for t in traj if t.get("name") == "action"))
    return counts


def find_jsonl(outdir: str, dataset: str) -> str | None:
    """Locate the JSONL result file for the given dataset inside outdir."""
    dataset_lower = dataset.lower()
    candidates = {
        "xbench": "xbench_results.jsonl",
        "webwalkerqa": "webwalkerqa_results.jsonl",
    }
    filename = candidates.get(dataset_lower)
    if filename is None:
        return None
    path = os.path.join(outdir, filename)
    return path if os.path.exists(path) else None


def print_stats(label: str, steps: list[int]) -> None:
    if not steps:
        print(f"  {label}: no data")
        return
    avg = sum(steps) / len(steps)
    print(f"  {label}:")
    print(f"    Avg Steps : {avg:.2f}")
    print(f"    Min       : {min(steps)}")
    print(f"    Max       : {max(steps)}")
    print(f"    n         : {len(steps)}")


def analyze(log_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    result = parse_log(log_path)
    outdir = result["outdir"]
    phases = result["phases"]

    if outdir and not os.path.isabs(outdir):
        # outdir is relative to the CWD when the pipeline ran (typically the
        # Flash-Searcher-main directory, one level above the log file).
        log_abs = os.path.abspath(log_path)
        log_dir = os.path.dirname(log_abs)
        parent_dir = os.path.dirname(log_dir)
        resolved = os.path.normpath(os.path.join(parent_dir, outdir))
        if os.path.isdir(resolved):
            outdir = resolved
        else:
            # Fallback: try relative to log's own directory
            outdir = os.path.normpath(os.path.join(log_dir, outdir))

    if not phases:
        print("  No phase information found in log.")
        return

    for ph in phases:
        dataset = ph["dataset"]
        concurrency = ph["concurrency"]
        total = ph["total"]
        label = f"Phase {ph['phase_num']} — {dataset} ({total} tasks, concurrency={concurrency})"
        print(f"\n{label}")

        steps = None

        # Sequential phase: use log-parsed steps
        if concurrency == 1 and ph["steps_per_task"]:
            steps = ph["steps_per_task"]
            source = "log (sequential)"

        # Try JSONL result file (more reliable; works for parallel phases too)
        if outdir:
            jsonl = find_jsonl(outdir, dataset)
            if jsonl:
                try:
                    steps = steps_from_jsonl(jsonl)
                    source = f"JSONL ({os.path.basename(jsonl)})"
                except Exception as e:
                    source = f"JSONL (error: {e})"

        if steps:
            print(f"  Source: {source}")
            print_stats(dataset, steps)
        elif concurrency > 1:
            print(f"  Parallel phase — JSONL result file not found at: {outdir}")
            print(f"  (Results may still be in progress or stored elsewhere)")
        else:
            print(f"  No step data available.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="Pipeline run log file(s)")
    args = parser.parse_args()

    for log_path in args.logs:
        if not os.path.exists(log_path):
            print(f"File not found: {log_path}", file=sys.stderr)
            continue
        analyze(log_path)


if __name__ == "__main__":
    main()
