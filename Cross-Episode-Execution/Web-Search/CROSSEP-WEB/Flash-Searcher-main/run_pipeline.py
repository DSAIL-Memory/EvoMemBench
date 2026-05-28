#!/usr/bin/env python
# coding=utf-8
"""
Cross-Benchmark Sequential Pipeline with Shared Memory

Runs XBench and/or WebWalkerQA sequentially using a single shared memory
provider so that knowledge accumulated on one benchmark is available to the
next.

Example usages
--------------
# Phase 1 (XBench only) to pre-warm memory, then Phase 2 (WebWalkerQA)
python run_pipeline.py \
    --memory_provider lightweight_memory \
    --xbench_infile data/xbench/DeepSearch.csv \
    --xbench_sample_num 50 \
    --webwalkerqa_infile data/webwalkerqa/webwalkerqa_subset_170.jsonl \
    --webwalkerqa_sample_num 50 \
    --output_dir pipeline_output/

# Skip one benchmark (only accumulate on XBench, no WebWalkerQA evaluation)
python run_pipeline.py \
    --memory_provider lightweight_memory \
    --xbench_infile data/xbench/DeepSearch.csv \
    --skip_webwalkerqa \
    --output_dir pipeline_output/warmup_only/

# Run without memory (baseline pipeline)
python run_pipeline.py \
    --xbench_infile data/xbench/DeepSearch.csv \
    --webwalkerqa_infile data/webwalkerqa/webwalkerqa_subset_170.jsonl \
    --output_dir pipeline_output/baseline/
"""

import copy
import os
import json
import base64
import csv
import argparse
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv

from FlashOAgents import OpenAIServerModel, ArkBatchServerModel, build_model
from eval_utils import generate_unified_report, create_run_directory
from utils import read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xor_decrypt(data, key):
    key_bytes = key.encode("utf-8")
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def _load_xbench(infile, sample_num=None, task_indices=None):
    data = []
    with open(infile, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for q in reader:
            key = q["canary"]
            q["prompt"] = _xor_decrypt(base64.b64decode(q["prompt"]), key).decode("utf-8")
            q["answer"] = _xor_decrypt(base64.b64decode(q["answer"]), key).decode("utf-8")
            data.append(q)

    if task_indices:
        from run_flash_searcher_xbench import parse_task_indices
        selected = parse_task_indices(task_indices)
        data = [data[i - 1] for i in sorted(selected) if 0 < i <= len(data)]
        logger.info(f"XBench: selected {len(data)} tasks from indices {task_indices}")
    elif sample_num is not None:
        data = data[:sample_num]
        logger.info(f"XBench: limited to first {sample_num} tasks")

    logger.info(f"XBench: loaded {len(data)} tasks")
    return data


def _load_webwalkerqa(infile, sample_num=None, task_indices=None,
                      difficulty=None, lang=None, domain=None, question_type=None):
    if infile.lower().endswith(".json"):
        with open(infile, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = []
        for idx, it in enumerate(raw):
            if isinstance(it, dict):
                it = dict(it)
                it["_global_index"] = idx + 1
            data.append(it)
    else:
        raw = read_jsonl(infile)
        data = []
        for idx, it in enumerate(raw):
            if isinstance(it, dict):
                it = dict(it)
                it["_global_index"] = idx + 1
            data.append(it)

    # Filters
    if difficulty:
        data = [it for it in data if isinstance(it, dict)
                and it.get("info", {}).get("difficulty_level", "").lower() == difficulty.lower()]
        logger.info(f"WebWalkerQA: after difficulty={difficulty} filter: {len(data)} tasks")
    if lang:
        data = [it for it in data if isinstance(it, dict)
                and it.get("info", {}).get("lang", "").lower() == lang.lower()]
        logger.info(f"WebWalkerQA: after lang={lang} filter: {len(data)} tasks")
    if domain:
        data = [it for it in data if isinstance(it, dict)
                and it.get("info", {}).get("domain", "").lower() == domain.lower()]
        logger.info(f"WebWalkerQA: after domain={domain} filter: {len(data)} tasks")
    if question_type:
        data = [it for it in data if isinstance(it, dict)
                and it.get("info", {}).get("type", "").lower() == question_type.lower()]
        logger.info(f"WebWalkerQA: after type={question_type} filter: {len(data)} tasks")

    if task_indices:
        from run_flash_searcher_webwalkerqa import parse_task_indices
        selected = parse_task_indices(task_indices)
        data = [data[i - 1] for i in sorted(selected) if 0 < i <= len(data)]
        logger.info(f"WebWalkerQA: selected {len(data)} tasks from indices {task_indices}")
    elif sample_num is not None:
        data = data[:sample_num]
        logger.info(f"WebWalkerQA: limited to first {sample_num} tasks")

    logger.info(f"WebWalkerQA: loaded {len(data)} tasks")
    return data


def _generate_pipeline_report(xbench_results, ww_results, output_dir):
    """Write a combined summary report for the pipeline run."""
    report_path = os.path.join(output_dir, "pipeline_report.txt")
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("Pipeline Evaluation Report")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _section(name, results):
        if not results:
            lines.append(f"[{name}] No results.\n")
            return
        total = len(results)

        def is_correct(r):
            if "score" in r:
                return r["score"] == 1
            j = r.get("judgement", "")
            return isinstance(j, str) and j.strip().lower() == "correct"

        correct = sum(1 for r in results if is_correct(r))
        total_tokens = sum(r.get("metrics", {}).get("total_tokens", 0) for r in results)
        total_time = sum(r.get("metrics", {}).get("elapsed_time", 0.0) for r in results)

        lines.append(f"--- {name} ---")
        lines.append(f"  Tasks     : {total}")
        lines.append(f"  Accuracy  : {correct}/{total} = {correct / total * 100:.2f}%")
        lines.append(f"  Avg Tokens: {total_tokens / total:.1f}")
        lines.append(f"  Avg Latency: {total_time / total:.2f}s")
        lines.append("")

    _section("XBench", xbench_results)
    _section("WebWalkerQA", ww_results)

    all_results = (xbench_results or []) + (ww_results or [])
    if all_results:
        def is_correct(r):
            if "score" in r:
                return r["score"] == 1
            j = r.get("judgement", "")
            return isinstance(j, str) and j.strip().lower() == "correct"

        total = len(all_results)
        correct = sum(1 for r in all_results if is_correct(r))
        total_tokens = sum(r.get("metrics", {}).get("total_tokens", 0) for r in all_results)
        total_time = sum(r.get("metrics", {}).get("elapsed_time", 0.0) for r in all_results)
        lines.append("--- Combined ---")
        lines.append(f"  Tasks      : {total}")
        lines.append(f"  Accuracy   : {correct}/{total} = {correct / total * 100:.2f}%")
        lines.append(f"  Avg Tokens : {total_tokens / total:.1f}")
        lines.append(f"  Avg Latency: {total_time / total:.2f}s")
        lines.append("")

    lines.append("=" * 80)
    text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + text)
    logger.info(f"Pipeline report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_xbench_phase(phase_num, args, model_config, custom_role_conversions,
                       output_dir, memory_tag, timestamp,
                       shared_memory_provider, memory_lock):
    """Execute the XBench phase and return results."""
    if args.skip_xbench:
        logger.info("Skipping XBench (--skip_xbench)")
        return []

    if not args.xbench_infile:
        logger.error("--xbench_infile is required unless --skip_xbench is set")
        return []

    from run_flash_searcher_xbench import run_xbench_tasks

    judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_api_base = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if args.memory_provider:
        judge_model = build_model(
            args.memory_provider,
            model_id=args.judge_model,
            custom_role_conversions=custom_role_conversions,
            api_key=judge_api_key,
            api_base=judge_api_base,
            max_completion_tokens=4096,
        )
    else:
        judge_model = ArkBatchServerModel(
            model_id=args.judge_model,
            custom_role_conversions=custom_role_conversions,
            max_completion_tokens=4096,
        )

    xbench_data = _load_xbench(
        args.xbench_infile,
        sample_num=args.xbench_sample_num,
        task_indices=args.xbench_task_indices,
    )

    xbench_run_dir = os.path.join(output_dir, "xbench_runs", f"{memory_tag}{timestamp}")
    os.makedirs(xbench_run_dir, exist_ok=True)

    logger.info(f"=== Phase {phase_num}: XBench ({len(xbench_data)} tasks, concurrency={args.concurrency}) ===")
    results = run_xbench_tasks(
        xbench_data, model_config, judge_model, args, xbench_run_dir,
        shared_memory_provider=shared_memory_provider,
        memory_lock=memory_lock,
    )

    generate_unified_report(
        results,
        os.path.join(xbench_run_dir, "report.txt"),
        dataset_name="XBench",
        has_levels=False,
    )

    xbench_outfile = os.path.join(output_dir, "xbench_results.jsonl")
    write_jsonl(xbench_outfile, results)
    logger.info(f"XBench results saved to {xbench_outfile}")
    logger.info(f"=== Phase {phase_num} done. Memory entries: {_memory_count(shared_memory_provider)} ===")
    return results


def _run_webwalkerqa_phase(phase_num, args, model_config,
                            output_dir, memory_tag, timestamp,
                            shared_memory_provider, memory_lock):
    """Execute the WebWalkerQA phase and return results."""
    if args.skip_webwalkerqa:
        logger.info("Skipping WebWalkerQA (--skip_webwalkerqa)")
        return []

    if not args.webwalkerqa_infile:
        logger.error("--webwalkerqa_infile is required unless --skip_webwalkerqa is set")
        return []

    from run_flash_searcher_webwalkerqa import run_webwalkerqa_tasks

    ww_data = _load_webwalkerqa(
        args.webwalkerqa_infile,
        sample_num=args.webwalkerqa_sample_num,
        task_indices=args.webwalkerqa_task_indices,
        difficulty=args.webwalkerqa_difficulty,
        lang=args.webwalkerqa_lang,
        domain=args.webwalkerqa_domain,
        question_type=args.webwalkerqa_question_type,
    )

    ww_run_dir = os.path.join(output_dir, "webwalkerqa_runs", f"{memory_tag}{timestamp}")
    os.makedirs(ww_run_dir, exist_ok=True)

    logger.info(f"=== Phase {phase_num}: WebWalkerQA ({len(ww_data)} tasks, concurrency={args.concurrency}) ===")
    results = run_webwalkerqa_tasks(
        ww_data, model_config, args, ww_run_dir,
        shared_memory_provider=shared_memory_provider,
        memory_lock=memory_lock,
    )

    generate_unified_report(
        results,
        os.path.join(ww_run_dir, "report.txt"),
        dataset_name="WebWalkerQA",
        has_levels=True,
        level_key="difficulty",
    )

    ww_outfile = os.path.join(output_dir, "webwalkerqa_results.jsonl")
    write_jsonl(ww_outfile, results)
    logger.info(f"WebWalkerQA results saved to {ww_outfile}")
    logger.info(f"=== Phase {phase_num} done. Memory entries: {_memory_count(shared_memory_provider)} ===")
    return results


def main(args):
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

    model_config = {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_tag = (args.memory_provider + "_") if args.memory_provider else "nomemory_"
    output_dir = os.path.join(args.output_dir, f"pipeline_{memory_tag}{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Pipeline output directory: {output_dir}")
    logger.info(f"Execution order: {args.order}")

    # ------------------------------------------------------------------
    # Create ONE shared memory provider (if requested)
    # ------------------------------------------------------------------
    shared_memory_provider = None
    memory_lock = None

    if args.memory_provider:
        from run_flash_searcher_xbench import load_memory_provider
        memory_model = build_model(args.memory_provider, **model_config)
        overrides = {}
        if getattr(args, "memory_dir", None):
            overrides["memory_dir"] = args.memory_dir
        if getattr(args, "memory_storage_root", None):
            from EvolveLab.memory_types import MemoryType as _MemoryType
            from EvolveLab.config import compute_storage_overrides as _compute_storage_overrides
            overrides.update(_compute_storage_overrides(
                _MemoryType(args.memory_provider), args.memory_storage_root
            ))
        shared_memory_provider = load_memory_provider(args.memory_provider, memory_model,
                                                      config_overrides=overrides or None)
        if shared_memory_provider:
            memory_lock = threading.Lock()
            logger.info(f"Shared memory provider ready: {args.memory_provider}")
        else:
            logger.warning("Failed to create shared memory provider; running without memory")

    # ------------------------------------------------------------------
    # Phase-specific args: Phase 1 always sequential (memory must
    # accumulate in order); Phase 2 uses the configured concurrency
    # (memory is already pre-loaded, parallel evaluation is safe).
    # ------------------------------------------------------------------
    phase1_args = copy.copy(args)
    phase1_args.concurrency = 1          # sequential — each task's memories help the next

    phase2_args = copy.copy(args)
    phase2_args.enable_memory_evolution = False  # Phase 2 reads memory only, never writes back

    logger.info(
        f"Concurrency plan: Phase 1 (memory accumulation) = 1 (sequential); "
        f"Phase 2 (evaluation) = {phase2_args.concurrency}"
    )

    # ------------------------------------------------------------------
    # Run phases in the requested order
    # ------------------------------------------------------------------
    xbench_results = []
    ww_results = []

    phase_kwargs = dict(
        model_config=model_config,
        output_dir=output_dir,
        memory_tag=memory_tag,
        timestamp=timestamp,
        shared_memory_provider=shared_memory_provider,
        memory_lock=memory_lock,
    )

    if args.order == "webwalkerqa_xbench":
        # Phase 1: WebWalkerQA (sequential, accumulate memory)
        # Phase 2: XBench (parallel, use accumulated memory)
        ww_results = _run_webwalkerqa_phase(
            phase_num=1, args=phase1_args, **phase_kwargs)
        xbench_results = _run_xbench_phase(
            phase_num=2, args=phase2_args,
            custom_role_conversions=custom_role_conversions, **phase_kwargs)
    else:
        # Phase 1: XBench (sequential, accumulate memory)
        # Phase 2: WebWalkerQA (parallel, use accumulated memory)
        xbench_results = _run_xbench_phase(
            phase_num=1, args=phase1_args,
            custom_role_conversions=custom_role_conversions, **phase_kwargs)
        ww_results = _run_webwalkerqa_phase(
            phase_num=2, args=phase2_args, **phase_kwargs)

    # ------------------------------------------------------------------
    # Combined pipeline report
    # ------------------------------------------------------------------
    _generate_pipeline_report(xbench_results, ww_results, output_dir)


def _memory_count(provider):
    """Best-effort count of entries in the memory provider."""
    if provider is None:
        return 0
    for attr in ("memory_bank", "longterm_db", "database"):
        obj = getattr(provider, attr, None)
        if obj is not None:
            if isinstance(obj, dict):
                return sum(len(v) if isinstance(v, list) else 1 for v in obj.values())
            if isinstance(obj, list):
                return len(obj)
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-benchmark pipeline runner with shared memory"
    )

    # ---- shared / common ----
    parser.add_argument("--memory_provider", type=str, default=None,
                        help="Memory provider type (e.g. 'lightweight_memory', 'agent_kb')")
    parser.add_argument("--memory_dir", type=str, default=None,
                        help="Override provider's memory_dir (e.g. for BM25 parallel isolated runs)")
    parser.add_argument("--memory_storage_root", type=str, default=None,
                        help="Override storage root; mapped to provider-specific config keys "
                             "via compute_storage_overrides() — used for --both-orders isolation "
                             "for providers that don't read memory_dir (agent_kb, AWM, etc.)")
    parser.add_argument("--order", type=str, default="xbench_webwalkerqa",
                        choices=["xbench_webwalkerqa", "webwalkerqa_xbench"],
                        help="Phase execution order: 'xbench_webwalkerqa' (default) accumulates "
                             "memory on XBench then evaluates on WebWalkerQA; "
                             "'webwalkerqa_xbench' reverses this.")
    parser.add_argument("--output_dir", type=str, default="./pipeline_output",
                        help="Base output directory")
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary_interval", type=int, default=8)
    parser.add_argument("--prompts_type", type=str, default="default")
    parser.add_argument("--enable_memory_evolution", action="store_true", default=True)
    parser.add_argument("--disable_memory_evolution", dest="enable_memory_evolution",
                        action="store_false")
    parser.add_argument("--judge_model", type=str,
                        default=os.getenv("DEFAULT_JUDGE_MODEL", "gemini-2.5-flash"))

    # ---- XBench ----
    parser.add_argument("--skip_xbench", action="store_true", default=False,
                        help="Skip XBench phase")
    parser.add_argument("--xbench_infile", type=str,
                        default="./data/xbench/DeepSearch.csv")
    parser.add_argument("--xbench_outfile", type=str, default=None,
                        help="Optional explicit outfile path for XBench JSONL")
    parser.add_argument("--xbench_sample_num", type=int, default=None)
    parser.add_argument("--xbench_task_indices", type=str, default=None)

    # ---- WebWalkerQA ----
    parser.add_argument("--skip_webwalkerqa", action="store_true", default=False,
                        help="Skip WebWalkerQA phase")
    parser.add_argument("--webwalkerqa_infile", type=str,
                        default="./data/webwalkerqa/webwalkerqa_subset_170.jsonl")
    parser.add_argument("--webwalkerqa_sample_num", type=int, default=None)
    parser.add_argument("--webwalkerqa_task_indices", type=str, default=None)
    parser.add_argument("--webwalkerqa_difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--webwalkerqa_lang", type=str, default=None,
                        choices=["en", "zh"])
    parser.add_argument("--webwalkerqa_domain", type=str, default=None)
    parser.add_argument("--webwalkerqa_question_type", type=str, default=None,
                        choices=["single_source", "multi_source"])

    args = parser.parse_args()

    # pipeline uses args.outfile in run_xbench_tasks safe_write; point to pipeline dir
    args.outfile = args.xbench_outfile  # may be None (safe_write guards with getattr)
    args.skip_summary = False
    args.direct_output_dir = None

    main(args)
