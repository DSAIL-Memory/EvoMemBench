"""Inference Script with Context-Grouped Memory - Using Standard OpenAI API

Extends infer_with_memory.py with context-grouped serial inference:
- Samples from the same context are processed SERIALLY with memory accumulation
- Different contexts are processed IN PARALLEL
- Each context has its own isolated memory bank (no cross-context contamination)

Usage:
    # Basic usage
    python infer_context_memory.py --model deepseek-v3-2-251201 --input CL-bench_context_ge5.jsonl

    # Quick test (max-samples limits total items, not contexts)
    python infer_context_memory.py --model deepseek-v3-2-251201 --max-samples 15

    # Custom parallelism and memory settings
    python infer_context_memory.py --model deepseek-v3-2-251201 --workers 10 --top-k 5

    # Custom API endpoint
    python infer_context_memory.py --model deepseek-v3-2-251201 --base-url https://api.deepseek.com/v1 --api-key KEY
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from cl_bench_memory import (
    build_memory,
    call_openai_api,
    format_trajectory,
    get_last_user_message,
    inject_memory_into_messages,
)
from cl_bench_memory.io import append_jsonl
from cl_bench_memory.logging_utils import log


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_api_credentials(model_name, cli_api_key=None, cli_base_url=None):
    if model_name.lower().startswith("deepseek"):
        default_key = os.getenv("DEEPSEEK_API_KEY")
        default_url = os.getenv("DEEPSEEK_BASE_URL")
        provider = "DeepSeek"
    else:
        default_key = os.getenv("OPENAI_API_KEY")
        default_url = os.getenv("OPENAI_BASE_URL")
        provider = "OpenAI"
    api_key = cli_api_key or default_key
    base_url = cli_base_url or default_url
    return api_key, base_url, provider


def _sanitize_for_filename(label: str) -> str:
    return label.replace("/", "_").replace(":", "_")


def group_by_context(data):
    """Group samples by context_id, preserving file order within each group.

    Returns:
        dict: {context_id: [(item, task_id), ...]} in file order
    """
    groups = {}
    for item in data:
        context_id = item["metadata"]["context_id"]
        task_id = item["metadata"]["task_id"]
        if context_id not in groups:
            groups[context_id] = []
        groups[context_id].append((item, task_id))
    return groups


def process_context_group(
    context_id,
    items,
    client,
    embed_client,
    args,
    run_memory_dir,
    output_path,
    output_lock,
):
    """Process all samples for one context serially with an isolated memory bank.

    Args:
        context_id: The context identifier
        items: List of (item, task_id) pairs in serial order
        client: OpenAI client for inference
        embed_client: OpenAI client for embeddings
        args: Parsed argparse namespace
        run_memory_dir: Base memory directory for this run (context subdir appended)
        output_path: Path to output JSONL file
        output_lock: threading.Lock for thread-safe writes

    Returns:
        dict: {context_id, success, failed, memory_entries_final}
    """
    no_memory = getattr(args, "no_memory", False)

    if not no_memory:
        context_memory_dir = os.path.join(run_memory_dir, context_id)
        memory = build_memory(
            args.memory_type,
            client=client,
            model=args.model,
            memory_dir=context_memory_dir,
            embed_provider=args.embed_provider,
            embed_model=args.embed_model,
            embed_client=embed_client,
            top_k=args.top_k,
            extract_model=args.extract_model,
        )
    else:
        memory = None

    success_count = 0
    fail_count = 0

    for item, task_id in items:
        messages = item.get("messages")
        if not messages:
            log(f"   [ctx={context_id[:8]}] Sample {task_id[:8]} skipped: no messages")
            fail_count += 1
            continue

        metadata = item.get("metadata", {})
        sample_t0 = time.time()

        if no_memory:
            memory_text = ""
            retrieve_stats = {"latency_s": 0.0, "embed_usage": {}}
            augmented_messages = messages
        else:
            query = get_last_user_message(messages)
            memory_text, retrieve_stats = memory.retrieve(query)
            augmented_messages = inject_memory_into_messages(messages, memory_text)

        infer_t0 = time.time()
        response_text, error, infer_usage = call_openai_api(
            client, augmented_messages, args.model
        )
        infer_latency = time.time() - infer_t0

        if error:
            log(f"   [ctx={context_id[:8]}] Sample {task_id[:8]} failed: {error}")
            fail_count += 1
            continue

        if no_memory:
            extract_stats = {"latency_s": 0.0, "llm_usage": {}, "embed_usage": {}}
        else:
            query = get_last_user_message(messages)
            trajectory = format_trajectory(messages, response_text)
            extract_stats = memory.extract(
                content=trajectory,
                task_id=task_id,
                query=query,
                context_category=metadata.get("context_category", ""),
                sub_category=metadata.get("sub_category", ""),
            )

        total_latency = time.time() - sample_t0

        sample_stats = {
            "latency": {
                "retrieve_s": round(retrieve_stats["latency_s"], 4),
                "inference_s": round(infer_latency, 4),
                "extract_s": round(extract_stats["latency_s"], 4),
                "total_s": round(total_latency, 4),
            },
            "tokens": {
                "inference": infer_usage,
                "extract_llm": extract_stats["llm_usage"],
                "retrieve_embed": retrieve_stats["embed_usage"],
                "extract_embed": extract_stats["embed_usage"],
            },
        }

        result = {
            "idx": task_id,
            "messages": messages,
            "model_output": response_text,
            "rubrics": item.get("rubrics", []),
            "metadata": metadata,
            "memory_type": "none" if no_memory else args.memory_type,
            "memory_retrieved": memory_text if memory_text else "",
            "stats": sample_stats,
        }

        with output_lock:
            append_jsonl(result, output_path)

        success_count += 1

    return {
        "context_id": context_id,
        "success": success_count,
        "failed": fail_count,
        "memory_entries_final": len(memory.memory_bank) if memory is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Context-Grouped Memory Inference - CL-bench"
    )
    parser.add_argument("--model", type=str, default="deepseek-v3-2-251201", help="Model name")
    parser.add_argument(
        "--input",
        type=str,
        default="CL-bench_context_ge5.jsonl",
        help="Input file path",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max parallel contexts (one thread per context)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max total samples to process"
    )
    parser.add_argument(
        "--memory-type",
        type=str,
        default="reasoning_bank",
        help="Which memory mechanism to use (e.g. reasoning_bank)",
    )
    parser.add_argument(
        "--memory-dir-prefix",
        type=str,
        default="memories/context_memory/",
        help="Base directory for per-context memory banks",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="dashscope",
        choices=["dashscope", "openai", "gemini", "qwen", "qwen3"],
        help="Embedding backend (default dashscope; qwen3: .env QWEN3_EMBED_API_KEY + QWEN3_EMBED_BASE_URL)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-v4",
        help="Embedding model name (dashscope/openai providers)",
    )
    parser.add_argument(
        "--embed-api-key",
        type=str,
        default=None,
        help="Embedding API key (overrides env: DASHSCOPE_API_KEY or OPENAI per provider)",
    )
    parser.add_argument(
        "--embed-base-url",
        type=str,
        default=None,
        help="Embedding API base URL",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of memory entries to retrieve"
    )
    parser.add_argument(
        "--extract-model",
        type=str,
        default=None,
        help="Model for memory extraction (defaults to --model)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable memory: skip retrieve, inject, and extract steps (plain inference)",
    )

    args = parser.parse_args()
    load_dotenv(override=True)

    # Auto-generate output path and memory run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    model_safe = _sanitize_for_filename(args.model)
    mt_safe = _sanitize_for_filename(args.memory_type)

    if args.no_memory:
        if args.max_samples:
            run_suffix = f"ctx_nomemory_max{args.max_samples}"
        else:
            run_suffix = f"ctx_nomemory_{timestamp}"
    elif args.max_samples:
        run_suffix = f"ctx_memory_topk{args.top_k}_max{args.max_samples}"
    else:
        run_suffix = f"ctx_memory_topk{args.top_k}_{timestamp}"

    if args.output is None:
        if args.no_memory:
            args.output = f"outputs/context_nomemory/{input_basename}_{model_safe}_{run_suffix}.jsonl"
        else:
            args.output = f"outputs/context_{mt_safe}/{input_basename}_{model_safe}_{run_suffix}.jsonl"

    run_memory_dir = os.path.join(
        args.memory_dir_prefix,
        f"{input_basename}_{model_safe}_{mt_safe}_topk{args.top_k}_{timestamp if not args.max_samples else 'max' + str(args.max_samples)}",
    )

    log(f"Input file: {args.input}")
    log(f"Output file: {args.output}")
    log(f"Model: {args.model}")
    log(f"Memory type: {'none (disabled)' if args.no_memory else args.memory_type}")
    log(f"Memory dir prefix: {run_memory_dir}")
    log(f"Top-K: {args.top_k if not args.no_memory else 'N/A'}")
    log(f"Workers: {args.workers}")
    log(f"Embed provider: {'N/A (no-memory)' if args.no_memory else args.embed_provider}")
    if not args.no_memory and args.embed_provider in ("openai", "dashscope"):
        log(f"Embed model: {args.embed_model}")

    api_key, base_url, provider = get_api_credentials(
        args.model, args.api_key, args.base_url
    )
    if not api_key:
        log(
            f"Error: No API key found for {provider}. "
            "Set env var or use --api-key"
        )
        return

    _timeout = httpx.Timeout(1200.0, connect=30.0)
    client_kwargs = {"api_key": api_key, "timeout": _timeout, "max_retries": 20}
    if base_url:
        client_kwargs["base_url"] = base_url
        log(f"Using {provider} API: {base_url}")

    client = OpenAI(**client_kwargs)

    # Use Volcengine Ark batch API when BATCH_MODEL is configured
    batch_model = os.getenv("BATCH_MODEL")
    if batch_model and args.model.lower().startswith("deepseek"):
        try:
            from volcenginesdkarkruntime import Ark
            client = Ark(api_key=api_key)
            args.model = batch_model
            log(f"Using Volcengine Ark batch API: endpoint = {batch_model}")
        except ImportError:
            log("Warning: volcenginesdkarkruntime not installed; falling back to OpenAI client")

    embed_client = None
    if args.no_memory:
        log("Memory disabled: skipping embed client setup")
    elif args.embed_provider == "dashscope":
        ds_key = args.embed_api_key or os.getenv("DASHSCOPE_API_KEY")
        ds_url = args.embed_base_url or os.getenv("DASHSCOPE_BASE_URL")
        if not ds_key:
            log(
                "Error: DashScope embedding requires DASHSCOPE_API_KEY "
                "in .env or --embed-api-key"
            )
            return
        if not ds_url:
            log(
                "Error: DashScope embedding requires DASHSCOPE_BASE_URL "
                "in .env or --embed-base-url"
            )
            return
        embed_client = OpenAI(api_key=ds_key, base_url=ds_url, timeout=_timeout, max_retries=20)
        log("Using DashScope API for embeddings")
    elif args.embed_provider == "qwen3":
        q3_key = args.embed_api_key or os.getenv("QWEN3_EMBED_API_KEY")
        q3_url = args.embed_base_url or os.getenv("QWEN3_EMBED_BASE_URL")
        if not q3_key:
            log("Error: qwen3 embedding requires QWEN3_EMBED_API_KEY in .env or --embed-api-key")
            return
        if not q3_url:
            log("Error: qwen3 embedding requires QWEN3_EMBED_BASE_URL in .env or --embed-base-url")
            return
        embed_client = OpenAI(api_key=q3_key, base_url=q3_url, timeout=_timeout, max_retries=20)
        log(f"Using Qwen3-Embedding API: {q3_url}")
    elif args.embed_provider == "openai":
        if args.embed_api_key or args.embed_base_url:
            embed_kwargs = {
                "api_key": args.embed_api_key or os.getenv("OPENAI_API_KEY") or api_key,
                "timeout": _timeout,
                "max_retries": 20,
            }
            if args.embed_base_url:
                embed_kwargs["base_url"] = args.embed_base_url
            embed_client = OpenAI(**embed_kwargs)
            log("Using separate embedding API")
        elif base_url:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                embed_client = OpenAI(api_key=openai_key, base_url=base_url, timeout=_timeout, max_retries=20)
                log(f"Using {provider} API for embeddings")
            else:
                log(
                    f"Warning: {provider} API may not support embeddings. "
                    "Set OPENAI_API_KEY or use --embed-api-key."
                )
                return

    log("Loading data...")
    data = load_jsonl(args.input)
    log(f"   Total {len(data)} samples")

    if args.max_samples:
        data = data[: args.max_samples]
        log(f"   Limited to {args.max_samples} samples")

    context_groups = group_by_context(data)
    log(f"   {len(context_groups)} unique contexts")

    total_samples = sum(len(items) for items in context_groups.values())
    log(f"Starting inference: {len(context_groups)} contexts, {total_samples} samples...")

    output_lock = threading.Lock()
    num_workers = min(args.workers, len(context_groups))

    agg = {
        "success": 0,
        "failed": 0,
        "latency_retrieve": 0.0,
        "latency_inference": 0.0,
        "latency_extract": 0.0,
        "latency_total": 0.0,
        "tokens_inference": 0,
        "tokens_extract_llm": 0,
        "tokens_embed": 0,
        "memory_entries": 0,
    }
    agg_lock = threading.Lock()

    def _sum_tokens(usage_dict):
        return usage_dict.get("total_tokens", 0)

    def _collect_stats_from_output():
        """Re-read output to aggregate stats after all contexts complete."""
        if not os.path.exists(args.output):
            return
        written = load_jsonl(args.output)
        for rec in written:
            s = rec.get("stats", {})
            lat = s.get("latency", {})
            tok = s.get("tokens", {})
            with agg_lock:
                agg["success"] += 1
                agg["latency_retrieve"] += lat.get("retrieve_s", 0)
                agg["latency_inference"] += lat.get("inference_s", 0)
                agg["latency_extract"] += lat.get("extract_s", 0)
                agg["latency_total"] += lat.get("total_s", 0)
                agg["tokens_inference"] += _sum_tokens(tok.get("inference", {}))
                agg["tokens_extract_llm"] += _sum_tokens(tok.get("extract_llm", {}))
                agg["tokens_embed"] += (
                    _sum_tokens(tok.get("retrieve_embed", {}))
                    + _sum_tokens(tok.get("extract_embed", {}))
                )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_context_group,
                context_id,
                items,
                client,
                embed_client,
                args,
                run_memory_dir,
                args.output,
                output_lock,
            ): context_id
            for context_id, items in context_groups.items()
        }

        with tqdm(total=len(context_groups), desc="Contexts") as pbar:
            for future in as_completed(futures):
                result = future.result()
                with agg_lock:
                    agg["failed"] += result["failed"]
                    agg["memory_entries"] += result["memory_entries_final"]
                pbar.update(1)
                pbar.set_postfix(
                    ctx=result["context_id"][:8],
                    ok=result["success"],
                    fail=result["failed"],
                )

    # Collect stats from newly written records
    _collect_stats_from_output()

    log("=" * 50)
    log("Context-memory inference completed!")
    log(f"   Success: {agg['success']}")
    log(f"   Failed: {agg['failed']}")
    log(f"   Total memory entries across contexts: {agg['memory_entries']}")
    log(f"   Output: {args.output}")

    n = agg["success"]
    if n > 0:
        log("-- Token usage (total, new samples only) --")
        log(f"   Inference:   {agg['tokens_inference']:,}")
        log(f"   Extract LLM: {agg['tokens_extract_llm']:,}")
        log(f"   Embedding:   {agg['tokens_embed']:,}")
        # log(f"   Grand total: {agg['tokens_inference'] + agg['tokens_extract_llm'] + agg['tokens_embed']:,}")
        log(f"   Grand total: {agg['tokens_inference'] + agg['tokens_extract_llm']:,}")
        log("-- Avg latency per sample --")
        log(f"   Retrieve:  {agg['latency_retrieve']/n:.2f}s")
        log(f"   Inference: {agg['latency_inference']/n:.2f}s")
        log(f"   Extract:   {agg['latency_extract']/n:.2f}s")
        log(f"   Total:     {agg['latency_total']/n:.2f}s")


if __name__ == "__main__":
    main()
