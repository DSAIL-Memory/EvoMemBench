#!/usr/bin/env python
# coding=utf-8
"""
Analyze benchmark results from JSONL output files or run directories.

Computes accuracy, average tokens, and average latency.
Supports both XBench (score=0/1) and WebWalkerQA (judgement='correct'/'incorrect').

Usage examples
--------------
# Single JSONL file
python analyze_results.py results.jsonl

# Multiple files for side-by-side comparison
python analyze_results.py xbench_results.jsonl webwalkerqa_results.jsonl

# From a run directory (reads all *.json files inside)
python analyze_results.py --run_dir ./xbench_output/xbench_runs/20250101_120000/

# Mix of files and directories
python analyze_results.py xbench.jsonl --run_dir webwalkerqa_runs/

# Save report to a file
python analyze_results.py results.jsonl --output my_report.txt
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results_from_jsonl(path: str) -> List[dict]:
    """Load all result records from a JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno} in {path}: {e}",
                      file=sys.stderr)
    return results


def load_results_from_dir(run_dir: str) -> List[dict]:
    """Load all result records from *.json files inside a directory."""
    pattern = os.path.join(run_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  Warning: no *.json files found in {run_dir}", file=sys.stderr)
        return []
    results = []
    for fp in files:
        # Skip report.txt and other non-result files that end in .json but are
        # likely reports (they won't parse as a single dict anyway).
        if os.path.basename(fp) in ("report.txt",):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                results.append(obj)
            elif isinstance(obj, list):
                results.extend(obj)
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Correctness detection
# ---------------------------------------------------------------------------

def is_correct(result: dict) -> bool:
    """
    Return True if this result record represents a correct answer.

    Handles both benchmark formats:
    - XBench:      ``score`` field (int 0 or 1)
    - WebWalkerQA: ``judgement`` field (str 'correct' / 'incorrect')
    """
    if "score" in result:
        return result["score"] == 1
    j = result.get("judgement", "")
    return isinstance(j, str) and j.strip().lower() == "correct"


def _detect_dataset(results: List[dict]) -> str:
    """Guess the dataset name from result record keys."""
    if not results:
        return "Unknown"
    sample = results[0]
    if "score" in sample and "task_id" in sample:
        return "XBench"
    if "judgement" in sample and "root_url" in sample:
        return "WebWalkerQA"
    return "Unknown"


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(results: List[dict], dataset_name: str = "") -> dict:
    """
    Compute evaluation statistics from a list of result records.

    Returns a dict with keys:
        dataset, total, successful, errors,
        correct, incorrect, accuracy,
        total_tokens, total_prompt_tokens, total_completion_tokens,
        avg_tokens, avg_prompt_tokens, avg_completion_tokens,
        total_latency, avg_latency,
        by_difficulty (dict, only if difficulty info present)
    """
    if not dataset_name:
        dataset_name = _detect_dataset(results)

    total = len(results)
    if total == 0:
        return {"dataset": dataset_name, "total": 0}

    successful = sum(1 for r in results if r.get("status") == "success")
    errors = total - successful

    correct = sum(1 for r in results if is_correct(r))
    incorrect = successful - correct  # only count graded successes as incorrect

    total_tokens = 0
    total_prompt = 0
    total_completion = 0
    total_latency = 0.0

    for r in results:
        m = r.get("metrics", {})
        total_tokens += m.get("total_tokens", 0)
        total_prompt += m.get("prompt_tokens", 0)
        total_completion += m.get("completion_tokens", 0)
        total_latency += m.get("elapsed_time", 0.0)

    # By-difficulty breakdown
    by_difficulty: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "correct": 0,
                                                           "tokens": 0, "latency": 0.0})
    for r in results:
        diff = r.get("difficulty") or r.get("level") or r.get("info", {}).get("difficulty_level")
        if diff:
            by_difficulty[diff]["total"] += 1
            by_difficulty[diff]["correct"] += int(is_correct(r))
            by_difficulty[diff]["tokens"] += r.get("metrics", {}).get("total_tokens", 0)
            by_difficulty[diff]["latency"] += r.get("metrics", {}).get("elapsed_time", 0.0)

    return {
        "dataset": dataset_name,
        "total": total,
        "successful": successful,
        "errors": errors,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "avg_tokens": total_tokens / total,
        "avg_prompt_tokens": total_prompt / total,
        "avg_completion_tokens": total_completion / total,
        "total_latency": total_latency,
        "avg_latency": total_latency / total,
        "by_difficulty": dict(by_difficulty) if by_difficulty else None,
    }


# ---------------------------------------------------------------------------
# Formatting / output
# ---------------------------------------------------------------------------

def _fmt_accuracy(stats: dict) -> str:
    c, t = stats["correct"], stats["total"]
    return f"{c}/{t} ({stats['accuracy'] * 100:.2f}%)"


def _build_report(stats_list: List[dict]) -> str:
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("Benchmark Results Analysis")
    lines.append(sep)
    lines.append("")

    for stats in stats_list:
        name = stats.get("dataset", "Unknown")
        total = stats.get("total", 0)
        if total == 0:
            lines.append(f"[{name}] No results found.\n")
            continue

        lines.append(f"Dataset     : {name}")
        lines.append(f"Total Tasks : {total}  "
                     f"(Successful: {stats['successful']}, Errors: {stats['errors']})")
        lines.append(f"Accuracy    : {_fmt_accuracy(stats)}")
        lines.append("")
        lines.append("Token Usage (per task averages)")
        lines.append(f"  Avg Total Tokens      : {stats['avg_tokens']:.1f}")
        lines.append(f"  Avg Prompt Tokens     : {stats['avg_prompt_tokens']:.1f}")
        lines.append(f"  Avg Completion Tokens : {stats['avg_completion_tokens']:.1f}")
        lines.append(f"  Total Tokens (sum)    : {stats['total_tokens']:,}")
        lines.append("")
        lines.append("Latency")
        lines.append(f"  Avg Latency/Task : {stats['avg_latency']:.2f}s")
        lines.append(f"  Total Latency    : {stats['total_latency']:.1f}s "
                     f"({stats['total_latency'] / 60:.1f}m)")
        lines.append("")

        by_diff = stats.get("by_difficulty")
        if by_diff:
            lines.append("By Difficulty")
            for level in sorted(by_diff.keys()):
                d = by_diff[level]
                n = d["total"]
                acc = d["correct"] / n * 100 if n else 0
                avg_tok = d["tokens"] / n if n else 0
                avg_lat = d["latency"] / n if n else 0
                lines.append(f"  {level:<8} : {d['correct']}/{n} ({acc:.1f}%)  "
                              f"avg_tokens={avg_tok:.0f}  avg_latency={avg_lat:.1f}s")
            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    # Comparison table if more than one dataset
    if len(stats_list) > 1:
        lines.append("Comparison Table")
        lines.append("-" * 80)
        header = f"{'Dataset':<22} {'Accuracy':>14} {'Avg Tokens':>12} {'Avg Latency':>13}"
        lines.append(header)
        lines.append("-" * 80)
        for s in stats_list:
            if s.get("total", 0) == 0:
                continue
            row = (f"{s['dataset']:<22} "
                   f"{_fmt_accuracy(s):>14} "
                   f"{s['avg_tokens']:>12.1f} "
                   f"{s['avg_latency']:>12.2f}s")
            lines.append(row)
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def print_stats_table(stats_list: List[dict]) -> None:
    print(_build_report(stats_list))


def save_report(stats_list: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_build_report(stats_list))
    print(f"Report saved to: {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results: accuracy, avg tokens, avg latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*",
        help="One or more JSONL result files to analyze",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Run directory containing *.json per-task result files",
    )
    parser.add_argument(
        "--run_dirs", nargs="+", default=None,
        help="Multiple run directories to analyze side-by-side",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save report to this file path",
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Override dataset name (used when auto-detection is wrong)",
    )
    args = parser.parse_args()

    # Collect all sources
    sources: List[dict] = []  # list of {path, kind, name}

    for fp in (args.files or []):
        if not os.path.exists(fp):
            print(f"Error: file not found: {fp}", file=sys.stderr)
            sys.exit(1)
        sources.append({"path": fp, "kind": "jsonl", "name": args.dataset_name or os.path.basename(fp)})

    if args.run_dir:
        if not os.path.isdir(args.run_dir):
            print(f"Error: directory not found: {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        sources.append({"path": args.run_dir, "kind": "dir",
                        "name": args.dataset_name or os.path.basename(args.run_dir.rstrip("/"))})

    for rd in (args.run_dirs or []):
        if not os.path.isdir(rd):
            print(f"Error: directory not found: {rd}", file=sys.stderr)
            sys.exit(1)
        sources.append({"path": rd, "kind": "dir",
                        "name": args.dataset_name or os.path.basename(rd.rstrip("/"))})

    if not sources:
        parser.print_help()
        sys.exit(0)

    stats_list = []
    for src in sources:
        print(f"Loading: {src['path']} ...", file=sys.stderr)
        if src["kind"] == "jsonl":
            results = load_results_from_jsonl(src["path"])
        else:
            results = load_results_from_dir(src["path"])

        print(f"  → {len(results)} records", file=sys.stderr)
        stats = compute_stats(results, dataset_name=src["name"])
        stats_list.append(stats)

    print_stats_table(stats_list)

    if args.output:
        save_report(stats_list, args.output)


if __name__ == "__main__":
    main()
