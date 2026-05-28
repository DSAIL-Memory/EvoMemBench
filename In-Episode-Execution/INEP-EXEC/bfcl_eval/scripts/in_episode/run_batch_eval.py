"""In-episode combined generate+evaluate driver for the multi_turn_ours category.

Drives the canonical BaseHandler.inference_multi_turn_prompting flow using
the Volcengine Ark batch endpoint, then immediately evaluates each sample
inline using compute_success_rate / compute_progress_score.

Deprecated in favour of the two-phase pipeline:
  run_batch_generate (inference only, writes result.jsonl)
  run_batch_evaluate (reads result.jsonl, writes per-sample JSONs + summary)

Kept for backwards compatibility with older shell scripts.

Usage:
    python -m bfcl_eval.scripts.in_episode.run_batch_eval [options]

Reads DEEPSEEK_API_KEY and BATCH_MODEL from the environment (or PROJECT_ROOT/.env).
Optional memory: set --memory-type mem0 and provide DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
)
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.factory import build_memory, build_memory_for_sample
from bfcl_eval.model_handler.api_inference.ark_batch import ArkBatchFCHandler, ArkBatchPromptingHandler
from bfcl_eval.scripts.in_episode.aggregator import (
    SampleResult,
    build_summary,
    load_id_to_category,
    sample_result_from_record,
)
from bfcl_eval.scripts.in_episode.func_doc_loader import load_function_docs
from bfcl_eval.scripts.in_episode.metrics import (
    compute_progress_score,
    compute_success_rate,
)
from bfcl_eval.utils import load_file, make_json_serializable

DATASET_PATH = PROMPT_PATH / "BFCL_v4_multi_turn_ours.json"
GROUND_TRUTH_PATH = POSSIBLE_ANSWER_PATH / "BFCL_v4_multi_turn_ours.json"
IDS_DIR = Path(__file__).resolve().parent / "ids"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated list of sample ids to run.")
    p.add_argument("--output-dir", type=Path, default=Path("in_episode_results"))
    p.add_argument("--run-name", type=str, default="auto")
    p.add_argument("--max-step", type=int, default=MAXIMUM_STEP_LIMIT)
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--resume", action="store_true",
                   help="Skip samples whose results/<id>.json already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call the API; print the planned ids and exit.")
    # Memory flags
    p.add_argument(
        "--memory-type",
        choices=[
            "none", "mem0", "memagent", "memobrain", "memos", "memoryos", "amem",
            "agent_kb", "agent_workflow", "skillweaver", "lightweight",
            "bm25", "qwen3_embedding", "graphrag", "ace", "reasoning_bank",
        ],
        default="none",
        help="Memory backend to use (default: none).",
    )
    p.add_argument("--memory-clear", action=argparse.BooleanOptionalAction, default=True,
                   help="Clear the memory store before this run (default: True).")
    p.add_argument("--memory-readonly", action="store_true", default=False,
                   help="Only read from memory (utilize), never write (update).")
    p.add_argument("--memory-load-from", type=Path, default=None,
                   help="Load an existing memory store from this directory (implies --no-memory-clear).")
    p.add_argument("--memory-top-k", type=int, default=3,
                   help="Number of memories to retrieve per sample (default: 3).")
    p.add_argument("--memory-scope", choices=["process", "sample"], default="process",
                   help="'process': one shared store across all samples (default); "
                        "'sample': each sample gets a fresh isolated store.")
    p.add_argument("--memory-keep-per-sample-dir", action="store_true", default=False,
                   help="Keep per-sample Qdrant dirs after inference (default: delete them).")
    p.add_argument("--context-budget", type=int, default=128000,
                   help="Token budget for LLM requests; oldest history is dropped if exceeded "
                        "(default: 128000 = 128k). Use 65536 or 32768 to reduce.")
    p.add_argument("--mode", choices=["prompting", "FC"], default="prompting",
                   help="LLM call mode: 'prompting' (default) or 'FC' (native function calling).")
    p.add_argument("--long-context", action="store_true", default=False,
                   help="Inject long-context noise into backend tool returns (same data as "
                        "multi_turn_long_context). Default: off.")
    return p.parse_args()


def setup_logging(error_log_path: Path) -> logging.Logger:
    logger = logging.getLogger("in_episode_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(error_log_path)
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def derive_run_name(model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{safe_model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def filter_entries(
    entries: list[dict],
    explicit_ids: list[str] | None,
    limit: int | None,
    skip_ids: set[str],
) -> list[dict]:
    out = entries
    if explicit_ids is not None:
        wanted = set(explicit_ids)
        out = [e for e in out if e["id"] in wanted]
    if skip_ids:
        out = [e for e in out if e["id"] not in skip_ids]
    if limit is not None:
        out = out[:limit]
    return out


def process_one_sample(
    handler: ArkBatchPromptingHandler,
    entry: dict,
    ground_truth: list[list[str]],
    category: str,
    results_dir: Path,
    memory: Optional[Memory] = None,
    context_budget: int = 128000,
    long_context: bool = False,
) -> SampleResult:
    # Deep-copy so that _pre_query_processing_prompting mutations do not
    # corrupt the shared entry dict used by other workers.
    test_entry = copy.deepcopy(entry)
    test_entry["function"] = load_function_docs(
        test_entry["involved_classes"], test_entry.get("excluded_function")
    )

    error: Optional[str] = None
    try:
        all_responses, metadata = handler.inference(
            test_entry,
            include_input_log=True,
            exclude_state_log=False,
            external_memory=memory,
            context_budget=context_budget,
            long_context=long_context,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        all_responses, metadata = [], {}
        logging.getLogger("in_episode_eval").warning(
            "inference failed for %s: %r\n%s",
            entry["id"],
            exc,
            traceback.format_exc(),
        )

    decoded_responses: list[list[list[str]]] = metadata.get("decoded_responses", [])
    model_slug = handler.model_name_underline_replaced

    # Pad decoded_responses to match ground_truth length so that
    # compute_success_rate / multi_turn_checker don't index out of range
    # when inference exited early (force_quit or mid-turn exception).
    while len(decoded_responses) < len(ground_truth):
        decoded_responses.append([])

    if error is not None:
        success_rate = 0
        checker_result = {"valid": False, "error_type": "inference_failed"}
        progress = 0.0
        k_passed = 0
    else:
        success_rate, checker_result = compute_success_rate(
            entry, decoded_responses, ground_truth, model_slug
        )
        progress, k_passed = compute_progress_score(
            entry, decoded_responses, ground_truth, model_slug
        )

    # ------------------------------------------------------------------ #
    # Token / latency accounting                                           #
    # ------------------------------------------------------------------ #
    inference_totals = metadata.get("sample_totals") or {
        "step_count": 0,
        "latency_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    raw_mem = metadata.get("memory_usage")
    if raw_mem is not None:
        memory_totals = dict(raw_mem)
        memory_totals.setdefault(
            "total_tokens",
            raw_mem["input_tokens"] + raw_mem["output_tokens"],
        )
    else:
        memory_totals = {
            "latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "embedding_tokens": 0,
            "n_llm_calls": 0,
            "n_embed_calls": 0,
            "n_utilize": 0,
            "n_update": 0,
        }

    total_totals = {
        "latency_s": round(inference_totals["latency_s"] + memory_totals["latency_s"], 4),
        "input_tokens": inference_totals["input_tokens"] + memory_totals["input_tokens"],
        "output_tokens": inference_totals["output_tokens"] + memory_totals["output_tokens"],
        "total_tokens": inference_totals["total_tokens"] + memory_totals["total_tokens"],
    }

    # ------------------------------------------------------------------ #
    # Persist per-sample JSON                                              #
    # ------------------------------------------------------------------ #
    record = {
        "id": entry["id"],
        "category": category,
        "involved_classes": entry["involved_classes"],
        "num_turns": len(entry["question"]),
        "decoded_responses": decoded_responses,
        "ground_truth": ground_truth,
        "per_turn_totals": metadata.get("per_turn_totals", []),
        "inference_totals": inference_totals,
        "memory_totals": memory_totals,
        "total_totals": total_totals,
        "metrics": {
            "success_rate": success_rate,
            "progress_score": progress,
            "k_passed": k_passed,
        },
        "checker_result": make_json_serializable(checker_result),
        "error": error,
        "force_quit": metadata.get("force_quit", False),
        "inference_log": make_json_serializable(metadata.get("inference_log", [])),
        "full_message_history": make_json_serializable(
            metadata.get("full_message_history", [])
        ),
    }
    out_path = results_dir / f"{entry['id']}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return SampleResult(
        id=entry["id"],
        category=category,
        success=success_rate,
        progress=progress,
        step_count=inference_totals["step_count"],
        error=error,
        latency_inference=inference_totals["latency_s"],
        input_tokens_inference=inference_totals["input_tokens"],
        output_tokens_inference=inference_totals["output_tokens"],
        total_tokens_inference=inference_totals["total_tokens"],
        latency_memory=memory_totals["latency_s"],
        input_tokens_memory=memory_totals["input_tokens"],
        output_tokens_memory=memory_totals["output_tokens"],
        total_tokens_memory=memory_totals["total_tokens"],
        embedding_tokens=memory_totals["embedding_tokens"],
        latency_total=total_totals["latency_s"],
        input_tokens_total=total_totals["input_tokens"],
        output_tokens_total=total_totals["output_tokens"],
        total_tokens_total=total_totals["total_tokens"],
    )


def _sample_result_from_record(rec: dict, id_to_category: dict[str, str]) -> SampleResult:
    """Thin wrapper retained for backwards compatibility; delegates to aggregator."""
    return sample_result_from_record(rec, id_to_category)


def main() -> int:
    args = parse_args()

    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)

    backend = os.environ.get("BATCH_BACKEND", "ark").lower()
    if backend == "gpt5mini":
        api_key = os.environ.get("GPT5MINI_API_KEY")
        required_vars = [("GPT5MINI_API_KEY", api_key)]
    elif backend == "gemini3flash":
        api_key = os.environ.get("GEMINI3FLASH_API_KEY")
        required_vars = [("GEMINI3FLASH_API_KEY", api_key)]
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        required_vars = [("DEEPSEEK_API_KEY", api_key)]
    model = os.environ.get("BATCH_MODEL")
    required_vars.append(("BATCH_MODEL", model))
    missing = [name for name, val in required_vars if not val]
    if missing and not args.dry_run:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        return 2
    model = model or "<unset>"

    run_name = args.run_name if args.run_name != "auto" else derive_run_name(model)
    output_dir = args.output_dir / run_name
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "errors.log")

    # ------------------------------------------------------------------ #
    # Memory setup                                                         #
    # ------------------------------------------------------------------ #
    if args.memory_scope == "sample" and (args.memory_load_from or args.memory_readonly):
        print(
            "--memory-scope sample is incompatible with --memory-load-from and --memory-readonly",
            file=sys.stderr,
        )
        return 2

    memory: Optional[Memory] = None
    memory_build_kwargs: dict = {}   # reused per-sample when memory_scope == "sample"
    memory_config_log: dict = {
        "memory_type": args.memory_type,
        "memory_scope": args.memory_scope,
        "context_budget": args.context_budget,
    }

    # Memory types that use Qdrant + DashScope embeddings.
    _EMBEDDING_MEMORY_TYPES = {"mem0", "memos", "memoryos", "amem", "agent_kb", "agent_workflow"}
    # Memory types that use the Ark LLM for synthesis only (no DashScope embeddings).
    _LLM_MEMORY_TYPES = {"skillweaver", "lightweight"}

    if args.memory_type != "none" and not args.dry_run:
        mem_model = os.environ.get("BFCL_MEMORY_MODEL") or model

        if args.memory_type in _EMBEDDING_MEMORY_TYPES:
            dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            dashscope_base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            mem_missing = [
                name for name, val in [
                    ("DASHSCOPE_API_KEY", dashscope_api_key),
                    ("DASHSCOPE_BASE_URL", dashscope_base_url),
                ] if not val
            ]
            if mem_missing:
                print(
                    f"Memory requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            # Kwargs shared by both process-scope and per-sample builds.
            memory_build_kwargs = {
                "ark_api_key": api_key,
                "ark_model": mem_model,
                "dashscope_api_key": dashscope_api_key,
                "dashscope_base_url": dashscope_base_url,
                "top_k": args.memory_top_k,
            }
        elif args.memory_type in _LLM_MEMORY_TYPES:
            # EvolveLab providers using Ark LLM only (no DashScope embeddings needed).
            memory_build_kwargs = {
                "ark_api_key": api_key,
                "ark_model": mem_model,
                "top_k": args.memory_top_k,
            }
        elif args.memory_type == "ace":
            # ACE playbook: Ark LLM only, no DashScope, no top_k.
            memory_build_kwargs = {
                "ark_api_key": api_key,
                "ark_model": mem_model,
            }
        elif args.memory_type == "reasoning_bank":
            # ReasoningBank: DashScope embeddings + Ark LLM, no top_k.
            dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            dashscope_base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            mem_missing = [
                name for name, val in [
                    ("DASHSCOPE_API_KEY", dashscope_api_key),
                    ("DASHSCOPE_BASE_URL", dashscope_base_url),
                ] if not val
            ]
            if mem_missing:
                print(
                    f"Memory requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            memory_build_kwargs = {
                "ark_api_key": api_key,
                "ark_model": mem_model,
                "dashscope_api_key": dashscope_api_key,
                "dashscope_base_url": dashscope_base_url,
            }
        elif args.memory_type == "memobrain":
            memory_build_kwargs = {
                "vllm_base_url": os.environ.get("MEMOBRAIN_VLLM_URL", "http://localhost:8001/v1"),
                "vllm_model": os.environ.get("MEMOBRAIN_MODEL", "TommyChien/MemoBrain-14B"),
            }
        else:
            # Compression-only types (memagent): no embedding dependencies.
            memory_build_kwargs = {}

        if args.memory_scope == "process":
            memory_dir: Path = (
                args.memory_load_from if args.memory_load_from else (output_dir / "memory")
            )
            memory_dir.mkdir(parents=True, exist_ok=True)
            do_clear = args.memory_clear and not args.memory_load_from

            logger.info(
                f"Memory: type={args.memory_type}, scope=process, dir={memory_dir}, "
                f"clear={do_clear}, readonly={args.memory_readonly}, top_k={args.memory_top_k}"
            )
            memory = build_memory(
                args.memory_type,
                storage_dir=memory_dir,
                clear=do_clear,
                readonly=args.memory_readonly,
                **memory_build_kwargs,
            )
            memory_config_log.update({
                "memory_dir": str(memory_dir),
                "memory_model": mem_model,
                "memory_clear": do_clear,
                "memory_readonly": args.memory_readonly,
                "memory_top_k": args.memory_top_k,
            })
        else:  # sample scope
            logger.info(
                f"Memory: type={args.memory_type}, scope=sample, "
                f"base_dir={output_dir}, top_k={args.memory_top_k}, "
                f"keep_dir={args.memory_keep_per_sample_dir}"
            )
            memory_config_log.update({
                "memory_model": mem_model,
                "memory_top_k": args.memory_top_k,
                "keep_per_sample_dir": args.memory_keep_per_sample_dir,
            })

    entries = load_file(DATASET_PATH, sort_by_id=False, use_lock=False)
    ground_truths = {
        gt["id"]: gt["ground_truth"]
        for gt in load_file(GROUND_TRUTH_PATH, sort_by_id=False, use_lock=False)
    }
    id_to_category = load_id_to_category(IDS_DIR)

    explicit_ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    skip_ids: set[str] = set()
    if args.resume:
        for path in results_dir.glob("*.json"):
            try:
                with open(path) as f:
                    rec = json.load(f)
                if "metrics" in rec and rec.get("error") is None:
                    skip_ids.add(rec["id"])
            except Exception:
                continue

    selected = filter_entries(entries, explicit_ids, args.limit, skip_ids)

    config = {
        "model": model,
        "mode": args.mode,
        "run_name": run_name,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "ids": explicit_ids,
        "max_step": args.max_step,
        "request_timeout": args.request_timeout,
        "resume": args.resume,
        "skipped_via_resume": sorted(skip_ids),
        "selected_count": len(selected),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "context_budget": args.context_budget,
        "memory": memory_config_log,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(
        f"Run {run_name}: {len(selected)} samples "
        f"({len(skip_ids)} resumed-skip), workers={args.num_workers}, output={output_dir}"
    )

    if args.dry_run:
        for e in selected:
            print(e["id"])
        return 0

    # One shared handler — ArkBatchClient uses per-thread TLS internally.
    if backend in ("gpt5mini", "gemini3flash"):
        if args.memory_type != "none":
            print(f"BATCH_BACKEND={backend} only supports --memory-type none", file=sys.stderr)
            return 2
        if backend == "gemini3flash":
            from bfcl_eval.model_handler.api_inference.gemini3flash import (
                Gemini3FlashFCHandler as _FCHandler,
                Gemini3FlashPromptingHandler as _PromptingHandler,
            )
        else:
            from bfcl_eval.model_handler.api_inference.gpt5mini import (
                Gpt5MiniFCHandler as _FCHandler,
                Gpt5MiniPromptingHandler as _PromptingHandler,
            )
        handler_cls = _FCHandler if args.mode == "FC" else _PromptingHandler
        handler = handler_cls(
            model_name=model,
            temperature=1,
            registry_name=model,
            is_fc_model=(args.mode == "FC"),
        )
    elif args.mode == "FC":
        handler = ArkBatchFCHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=True,
            ark_api_key=api_key,
            request_timeout=args.request_timeout,
        )
    else:
        handler = ArkBatchPromptingHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=False,
            ark_api_key=api_key,
            request_timeout=args.request_timeout,
        )

    sample_results: list[SampleResult] = []
    write_lock = threading.Lock()

    def _task(entry):
        gt = ground_truths.get(entry["id"])
        if gt is None:
            raise KeyError(f"No ground truth for {entry['id']}")
        category = id_to_category.get(entry["id"], "uncategorized")

        per_sample_mem = memory
        per_sample_dir = None
        if args.memory_scope == "sample" and args.memory_type != "none":
            per_sample_mem, per_sample_dir = build_memory_for_sample(
                args.memory_type, output_dir, entry["id"], **memory_build_kwargs
            )

        try:
            result = process_one_sample(
                handler, entry, gt, category, results_dir,
                memory=per_sample_mem,
                context_budget=args.context_budget,
                long_context=args.long_context,
            )
        finally:
            if per_sample_dir is not None:
                per_sample_mem = None  # release Qdrant client reference before rmtree
                if not args.memory_keep_per_sample_dir:
                    shutil.rmtree(per_sample_dir, ignore_errors=True)

        return result

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_task, e): e["id"] for e in selected}
        with tqdm(total=len(futures), desc="samples") as pbar:
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.warning(f"sample {sid} failed: {exc!r}\n{traceback.format_exc()}")
                    res = SampleResult(
                        id=sid,
                        category=id_to_category.get(sid, "uncategorized"),
                        success=0, progress=0.0,
                        step_count=0, error=f"{type(exc).__name__}: {exc}",
                    )
                with write_lock:
                    sample_results.append(res)
                pbar.update(1)

    # If resuming, fold previously completed results back in for the summary.
    if args.resume:
        existing_ids = {s.id for s in sample_results}
        for path in results_dir.glob("*.json"):
            try:
                with open(path) as f:
                    rec = json.load(f)
            except Exception:
                continue
            if rec["id"] in existing_ids:
                continue
            sample_results.append(_sample_result_from_record(rec, id_to_category))

    sample_results.sort(key=lambda s: s.id)
    build_summary(sample_results, output_dir, IDS_DIR, model, run_name)

    n_pass = sum(s.success for s in sample_results)
    n_fail = sum(1 for s in sample_results if s.error)
    logger.info(
        f"Done. {n_pass}/{len(sample_results)} passed, {n_fail} errored. "
        f"Summary at {output_dir / 'summary.txt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
