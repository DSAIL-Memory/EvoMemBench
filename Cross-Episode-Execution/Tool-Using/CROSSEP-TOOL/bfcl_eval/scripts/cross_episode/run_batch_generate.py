"""Cross-episode inference driver — generate phase only.

Runs BaseHandler.inference_multi_turn_prompting for every selected sample and
writes each result as a JSON line to ``<output_dir>/<run_name>/result.jsonl``.
The JSONL uses the same field names as the original BFCL ``*_result.json`` files
(id, result, input_token_count, output_token_count, latency, inference_log) plus
fork-specific telemetry fields (category, per_turn_totals, sample_totals,
memory_usage, force_quit, full_message_history, error).

Pass the generated ``result.jsonl`` to ``run_batch_evaluate`` to compute
success_rate, progress_score and per-sample JSON artefacts.

Usage:
    python -m bfcl_eval.scripts.cross_episode.run_batch_generate [options]

Reads DEEPSEEK_API_KEY and BATCH_MODEL from the environment (or PROJECT_ROOT/.env).
Optional memory: set --memory-type mem0 (requires DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL)
or --memory-type ace (no extra env vars needed; ACE has no top-k retrieval).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
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
from bfcl_eval.memory.factory import build_memory
from bfcl_eval.model_handler.api_inference.ark_batch import ArkBatchFCHandler, ArkBatchPromptingHandler
from bfcl_eval.scripts.cross_episode.aggregator import load_id_to_category
from bfcl_eval.scripts.cross_episode.func_doc_loader import load_function_docs
from bfcl_eval.utils import load_file, make_json_serializable

DATASET_PATH = PROMPT_PATH / "BFCL_v4_multi_turn_ours.json"
IDS_DIR = Path(__file__).resolve().parent / "ids"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated list of sample ids to run.")
    p.add_argument("--output-dir", type=Path, default=Path("cross_episode_results"))
    p.add_argument("--run-name", type=str, default="auto")
    p.add_argument("--max-step", type=int, default=MAXIMUM_STEP_LIMIT)
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--resume", action="store_true",
                   help="Skip samples already present in result.jsonl.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call the API; print the planned ids and exit.")
    # Memory flags
    p.add_argument("--memory-type",
                   choices=["none", "mem0", "ace", "reasoning_bank",
                            "bm25", "qwen3_embedding", "graphrag",
                            "agent_kb", "agent_workflow", "skillweaver", "lightweight",
                            "amem", "memos", "memoryos"],
                   default="none",
                   help="Memory backend to use (default: none).")
    p.add_argument("--memory-clear", action=argparse.BooleanOptionalAction, default=True,
                   help="Clear the memory store before this run (default: True).")
    p.add_argument("--memory-readonly", action="store_true", default=False,
                   help="Only read from memory (utilize), never write (update).")
    p.add_argument("--memory-load-from", type=Path, default=None,
                   help="Load an existing memory store from this directory (implies --no-memory-clear).")
    p.add_argument("--memory-top-k", type=int, default=3,
                   help="Number of memories to retrieve per sample (default: 3).")
    p.add_argument("--mode", choices=["prompting", "FC"], default="prompting",
                   help="LLM call mode: 'prompting' (default) or 'FC' (native function calling).")
    return p.parse_args()


def setup_logging(error_log_path: Path) -> logging.Logger:
    logger = logging.getLogger("cross_episode_generate")
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
    category: str,
    jsonl_path: Path,
    jsonl_lock: threading.Lock,
    logger: logging.Logger,
    memory: Optional[Memory] = None,
) -> Optional[str]:
    """Run inference for one sample and append a JSON line to result.jsonl.

    Returns the error string if inference failed, else None.
    """
    test_entry = copy.deepcopy(entry)
    test_entry["function"] = load_function_docs(
        test_entry["involved_classes"], test_entry.get("excluded_function")
    )

    error: Optional[str] = None
    all_responses: list = []
    metadata: dict = {}
    try:
        all_responses, metadata = handler.inference(
            test_entry,
            include_input_log=True,
            exclude_state_log=False,
            external_memory=memory,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "inference failed for %s: %r\n%s",
            entry["id"],
            exc,
            traceback.format_exc(),
        )

    line = make_json_serializable({
        # Original BFCL-compatible fields
        "id": entry["id"],
        "result": all_responses,
        "input_token_count": metadata.get("input_token_count", []),
        "output_token_count": metadata.get("output_token_count", []),
        "latency": metadata.get("latency", []),
        "inference_log": metadata.get("inference_log", []),
        # Fork-specific telemetry consumed by run_batch_evaluate
        "category": category,
        "involved_classes": entry["involved_classes"],
        "num_turns": len(entry["question"]),
        "per_turn_totals": metadata.get("per_turn_totals", []),
        "sample_totals": metadata.get("sample_totals", {}),
        "memory_usage": metadata.get("memory_usage"),
        "force_quit": metadata.get("force_quit", False),
        "full_message_history": metadata.get("full_message_history", []),
        "error": error,
    })

    with jsonl_lock:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    return error


def main() -> int:
    args = parse_args()

    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("BATCH_MODEL")
    missing = [name for name, val in [("DEEPSEEK_API_KEY", api_key), ("BATCH_MODEL", model)] if not val]
    if missing and not args.dry_run:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        return 2
    model = model or "<unset>"

    run_name = args.run_name if args.run_name != "auto" else derive_run_name(model)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "errors.log")

    # ------------------------------------------------------------------ #
    # Memory setup                                                         #
    # ------------------------------------------------------------------ #
    memory: Optional[Memory] = None
    memory_config_log: dict = {"memory_type": args.memory_type}

    if args.memory_type != "none" and not args.dry_run:
        mem_model = os.environ.get("BFCL_MEMORY_MODEL") or model
        memory_dir: Path = (
            args.memory_load_from if args.memory_load_from else (output_dir / "memory")
        )
        memory_dir.mkdir(parents=True, exist_ok=True)
        do_clear = args.memory_clear and not args.memory_load_from

        common_kwargs: dict = dict(
            storage_dir=memory_dir,
            ark_api_key=api_key,
            ark_model=mem_model,
            clear=do_clear,
            readonly=args.memory_readonly,
        )

        if args.memory_type == "mem0":
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
                    f"Memory (mem0) requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            common_kwargs.update(
                dashscope_api_key=dashscope_api_key,
                dashscope_base_url=dashscope_base_url,
                top_k=args.memory_top_k,
            )
            memory_config_log["memory_top_k"] = args.memory_top_k
        elif args.memory_type == "ace":
            # ACE needs no extra env vars and has no top-k retrieval.
            pass
        elif args.memory_type == "reasoning_bank":
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
                    f"Memory (reasoning_bank) requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            common_kwargs.update(
                dashscope_api_key=dashscope_api_key,
                dashscope_base_url=dashscope_base_url,
            )
        elif args.memory_type == "bm25":
            common_kwargs.update(top_k=args.memory_top_k)
            memory_config_log["memory_top_k"] = args.memory_top_k
        elif args.memory_type == "qwen3_embedding":
            # Prefer OPENAI_* credentials; fall back to DASHSCOPE_*.
            embed_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
            embed_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            embed_model = os.environ.get("BFCL_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
            mem_missing = [
                name for name, val in [("embedding API key", embed_key), ("embedding base URL", embed_url)]
                if not val
            ]
            if mem_missing:
                print(
                    f"Memory (qwen3_embedding) requires embedding credentials. "
                    f"Set OPENAI_API_KEY + OPENAI_BASE_URL (or DASHSCOPE_API_KEY + DASHSCOPE_BASE_URL).",
                    file=sys.stderr,
                )
                return 2
            common_kwargs.update(
                dashscope_api_key=embed_key,
                dashscope_base_url=embed_url,
                embed_model=embed_model,
                top_k=args.memory_top_k,
            )
            memory_config_log["memory_top_k"] = args.memory_top_k
            memory_config_log["embed_model"] = embed_model
        elif args.memory_type == "graphrag":
            dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            dashscope_base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            embed_model = os.environ.get("BFCL_EMBED_MODEL", "text-embedding-v4")
            mem_missing = [
                name for name, val in [
                    ("DASHSCOPE_API_KEY", dashscope_api_key),
                    ("DASHSCOPE_BASE_URL", dashscope_base_url),
                ] if not val
            ]
            if mem_missing:
                print(
                    f"Memory (graphrag) requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            common_kwargs.update(
                dashscope_api_key=dashscope_api_key,
                dashscope_base_url=dashscope_base_url,
                embed_model=embed_model,
                top_k=args.memory_top_k,
            )
            memory_config_log["memory_top_k"] = args.memory_top_k
            memory_config_log["embed_model"] = embed_model
        elif args.memory_type in ("agent_kb", "agent_workflow", "lightweight"):
            common_kwargs.update(top_k=args.memory_top_k)
            memory_config_log["memory_top_k"] = args.memory_top_k
        elif args.memory_type == "skillweaver":
            # SkillWeaver has no top_k — injects all keyword-matched skills.
            pass
        elif args.memory_type in ("amem", "memos", "memoryos"):
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
                    f"Memory ({args.memory_type}) requires env vars: {', '.join(mem_missing)}",
                    file=sys.stderr,
                )
                return 2
            common_kwargs.update(
                dashscope_api_key=dashscope_api_key,
                dashscope_base_url=dashscope_base_url,
                top_k=args.memory_top_k,
            )
            memory_config_log["memory_top_k"] = args.memory_top_k

        logger.info(
            "Memory: type=%s, dir=%s, clear=%s, readonly=%s%s",
            args.memory_type, memory_dir, do_clear, args.memory_readonly,
            f", top_k={args.memory_top_k}" if args.memory_type in
                ("mem0", "bm25", "qwen3_embedding", "graphrag",
                 "agent_kb", "agent_workflow", "lightweight",
                 "amem", "memos", "memoryos") else "",
        )
        memory = build_memory(args.memory_type, **common_kwargs)
        memory_config_log.update({
            "memory_dir": str(memory_dir),
            "memory_model": mem_model,
            "memory_clear": do_clear,
            "memory_readonly": args.memory_readonly,
        })

    entries = load_file(DATASET_PATH, sort_by_id=False, use_lock=False)
    id_to_category = load_id_to_category(IDS_DIR)

    jsonl_path = output_dir / "result.jsonl"
    explicit_ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    skip_ids: set[str] = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    if rec.get("id") and rec.get("error") is None:
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
        "memory": memory_config_log,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(
        f"Generate {run_name}: {len(selected)} samples "
        f"({len(skip_ids)} resumed-skip), workers={args.num_workers}, output={output_dir}"
    )

    if args.dry_run:
        for e in selected:
            print(e["id"])
        return 0

    # One shared handler — ArkBatchClient uses per-thread TLS internally.
    if args.mode == "FC":
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

    jsonl_lock = threading.Lock()

    def _task(entry):
        category = id_to_category.get(entry["id"], "uncategorized")
        return process_one_sample(
            handler, entry, category, jsonl_path, jsonl_lock, logger, memory=memory
        )

    n_error = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_task, e): e["id"] for e in selected}
        with tqdm(total=len(futures), desc="generate") as pbar:
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    err = fut.result()
                except Exception as exc:
                    logger.warning(f"sample {sid} worker failed: {exc!r}\n{traceback.format_exc()}")
                    err = f"{type(exc).__name__}: {exc}"
                if err is not None:
                    n_error += 1
                pbar.update(1)

    logger.info(
        f"Done. {len(selected) - n_error}/{len(selected)} succeeded, "
        f"{n_error} errored. Results at {jsonl_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
