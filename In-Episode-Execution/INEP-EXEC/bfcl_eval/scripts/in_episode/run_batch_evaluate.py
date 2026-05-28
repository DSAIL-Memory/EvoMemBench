"""In-episode evaluation driver — evaluate phase only.

Reads ``result.jsonl`` produced by ``run_batch_generate``, then for each
sample:

1. Calls ``_evaluate_single_multi_turn_entry`` (eval_runner.py:166-260),
   which mirrors the original BFCL two-phase logic:
   - post-hoc ``decode_execute`` (no reliance on the inline decoded_responses
     field that run_batch_eval.py carried);
   - force_terminated short-circuit when len(decoded) != len(ground_truth);
   - ``multi_turn_checker`` call for full success/failure verdict.

2. Additionally computes ``progress_score`` (longest passing prefix), which
   the original BFCL eval_runner does not produce.

Writes one ``<id>.json`` per sample plus ``summary.{txt,csv}`` and
``per_sample.csv`` into the same directory as the input JSONL (or to
``--output-dir`` if supplied).

Usage:
    python -m bfcl_eval.scripts.in_episode.run_batch_evaluate \\
        --input path/to/result.jsonl [--num-workers N] [--ids id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl_eval.model_handler.api_inference.ark_batch import ArkBatchFCHandler, ArkBatchPromptingHandler
from bfcl_eval.scripts.in_episode.aggregator import (
    SampleResult,
    build_summary,
    load_id_to_category,
    sample_result_from_record,
)
from bfcl_eval.scripts.in_episode.metrics import compute_progress_score
from bfcl_eval.utils import load_file, make_json_serializable

DATASET_PATH = PROMPT_PATH / "BFCL_v4_multi_turn_ours.json"
GROUND_TRUTH_PATH = POSSIBLE_ANSWER_PATH / "BFCL_v4_multi_turn_ours.json"
IDS_DIR = Path(__file__).resolve().parent / "ids"
TEST_CATEGORY = "multi_turn_ours"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Path to result.jsonl produced by run_batch_generate, or its parent directory.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for per-sample JSONs and summary (default: parent of --input).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated list of sample ids to evaluate.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def setup_logging(error_log_path: Path) -> logging.Logger:
    logger = logging.getLogger("in_episode_evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(error_log_path, mode="a")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def _post_hoc_decode(
    handler: ArkBatchPromptingHandler,
    raw_result: list[list[str]],
) -> list[list[list[str]]]:
    """Decode raw per-turn/per-step text into executable function-call strings.

    Mirrors eval_runner.py:217-236 — empty / unparseable steps are silently
    dropped (not stored as empty entries), matching original BFCL behaviour.
    """
    decoded: list[list[list[str]]] = []
    for turn_steps in raw_result:
        decoded_turn: list[list[str]] = []
        for step_text in turn_steps:
            try:
                r: list[str] = handler.decode_execute(step_text, has_tool_call_tag=False)
                if not is_empty_execute_response(r):
                    decoded_turn.append(r)
            except Exception:
                continue
        decoded.append(decoded_turn)
    return decoded


def _evaluate_entry(
    handler: ArkBatchPromptingHandler,
    sid: str,
    raw_result: list[list[str]],
    ground_truth: list[list[str]],
    prompt_entry: dict,
    model_slug: str,
) -> tuple[int, dict, list[list[list[str]]]]:
    """Evaluate one sample using original BFCL two-phase logic.

    Mirrors eval_runner.py:_evaluate_single_multi_turn_entry (lines 166-260):
      - type check → inference_error
      - length mismatch → force_terminated (no checker call)
      - post-hoc decode_execute loop
      - multi_turn_checker call

    Returns (success_rate 0|1, checker_result dict, decoded list).
    The decoded list is always populated (may be shorter than ground_truth for
    force_terminated) so that compute_progress_score can credit prefix turns.
    """
    # Remove function doc if present (cosmetic, not needed by checker)
    prompt_entry.pop("function", None)

    if not isinstance(raw_result, list):
        return 0, {
            "valid": False,
            "error": {
                "error_message": ["Model did not output a list of responses."],
                "error_type": "multi_turn:inference_error",
            },
        }, []

    decoded = _post_hoc_decode(handler, raw_result)

    # Original force_terminated guard: short-circuit before checker when turns
    # produced by inference don't match the number of ground-truth turns.
    if len(decoded) != len(ground_truth):
        return 0, {
            "valid": False,
            "error": {
                "error_message": [
                    f"Model was force-terminated during inference phase. "
                    f"The length of the model result turns ({len(decoded)}) does not match "
                    f"the length of the ground truth turns ({len(ground_truth)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
        }, decoded

    checker_result = multi_turn_checker(
        decoded, ground_truth, prompt_entry, TEST_CATEGORY, model_slug
    )
    success = int(bool(checker_result.get("valid", False)))
    return success, checker_result, decoded


def process_one_sample(
    handler: ArkBatchPromptingHandler,
    record: dict,
    prompt_entries: dict[str, dict],
    ground_truths: dict[str, list[list[str]]],
    id_to_category: dict[str, str],
    output_dir: Path,
    model_slug: str,
    logger: logging.Logger,
) -> SampleResult:
    sid = record["id"]
    raw_result: list[list[str]] = record.get("result") or []
    error: Optional[str] = record.get("error")
    force_quit: bool = record.get("force_quit", False)
    category = record.get("category") or id_to_category.get(sid, "uncategorized")

    # Deep-copy so _evaluate_single_multi_turn_entry's mutation (deletes
    # "function" if present) does not affect the shared prompt_entries index.
    prompt_entry = copy.deepcopy(prompt_entries.get(sid, {"id": sid}))
    ground_truth: list[list[str]] = ground_truths.get(sid, [])

    # ------------------------------------------------------------------ #
    # Token / latency accounting from generate phase                       #
    # ------------------------------------------------------------------ #
    inference_totals = record.get("sample_totals") or {
        "step_count": 0,
        "latency_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    raw_mem = record.get("memory_usage")
    if raw_mem is not None:
        memory_totals = dict(raw_mem)
        memory_totals.setdefault(
            "total_tokens",
            raw_mem.get("input_tokens", 0) + raw_mem.get("output_tokens", 0),
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
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    if error is not None:
        success_rate = 0
        checker_result = {"valid": False, "error_type": "inference_failed"}
        decoded: list[list[list[str]]] = []
        progress, k_passed = 0.0, 0
    else:
        # _evaluate_entry mirrors original eval_runner.py:_evaluate_single_multi_turn_entry
        # (lines 166-260): type check → force_terminated guard → post-hoc decode →
        # multi_turn_checker.  It also returns the decoded list so that
        # compute_progress_score can credit completed prefix turns.
        success_rate, checker_result, decoded = _evaluate_entry(
            handler, sid, raw_result, ground_truth, prompt_entry, model_slug,
        )
        progress, k_passed = compute_progress_score(
            prompt_entry, decoded, ground_truth, model_slug
        )

    # ------------------------------------------------------------------ #
    # Persist per-sample JSON (same schema as run_batch_eval.py)          #
    # ------------------------------------------------------------------ #
    rec = {
        "id": sid,
        "category": category,
        "involved_classes": record.get("involved_classes", []),
        "num_turns": record.get("num_turns", len(ground_truth)),
        "decoded_responses": decoded,
        "ground_truth": ground_truth,
        "per_turn_totals": record.get("per_turn_totals", []),
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
        "force_quit": force_quit,
        "inference_log": make_json_serializable(record.get("inference_log", [])),
        "full_message_history": make_json_serializable(record.get("full_message_history", [])),
    }
    out_path = output_dir / f"{sid}.json"
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)

    return sample_result_from_record(rec, id_to_category)


def main() -> int:
    args = parse_args()

    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)

    # Resolve input JSONL path
    input_path = args.input
    if input_path.is_dir():
        input_path = input_path / "result.jsonl"
    if not input_path.exists():
        print(f"[ERROR] result.jsonl not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir if args.output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "errors.log")

    # Derive model name, run_name, and mode from run_config.json if present
    model = os.environ.get("BATCH_MODEL", "unknown")
    run_name = output_dir.name
    mode = "prompting"
    config_path = input_path.parent / "run_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                run_config = json.load(f)
            model = run_config.get("model", model)
            run_name = run_config.get("run_name", run_name)
            mode = run_config.get("mode", "prompting")
        except Exception:
            pass

    model_slug = model.replace("/", "_").replace(":", "_")

    # Load records from JSONL
    records: list[dict] = []
    with open(input_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except Exception:
                logger.warning(f"Skipping malformed JSONL line: {raw[:80]!r}")

    # Filter by explicit ids / limit
    explicit_ids: Optional[set[str]] = None
    if args.ids:
        explicit_ids = {s.strip() for s in args.ids.split(",")}
    if explicit_ids is not None:
        records = [r for r in records if r.get("id") in explicit_ids]
    if args.limit is not None:
        records = records[:args.limit]

    # Load dataset and ground truth (index by id for O(1) lookup)
    all_entries = load_file(DATASET_PATH, sort_by_id=False, use_lock=False)
    prompt_entries: dict[str, dict] = {e["id"]: e for e in all_entries}
    ground_truths: dict[str, list[list[str]]] = {
        gt["id"]: gt["ground_truth"]
        for gt in load_file(GROUND_TRUTH_PATH, sort_by_id=False, use_lock=False)
    }
    id_to_category = load_id_to_category(IDS_DIR)

    # Instantiate handler for decode_execute (no API calls made during evaluate).
    # Mode is read from run_config.json so the post-hoc decoder matches the generate phase.
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if mode == "FC":
        handler = ArkBatchFCHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=True,
            ark_api_key=api_key or "placeholder",
            request_timeout=1200,
        )
    else:
        handler = ArkBatchPromptingHandler(
            model_name=model,
            temperature=0,
            registry_name=model,
            is_fc_model=False,
            ark_api_key=api_key or "placeholder",
            request_timeout=1200,
        )

    logger.info(
        f"Evaluate {run_name}: {len(records)} samples, workers={args.num_workers}, output={output_dir}"
    )

    sample_results: list[SampleResult] = []

    def _task(rec: dict) -> SampleResult:
        return process_one_sample(
            handler, rec, prompt_entries, ground_truths,
            id_to_category, output_dir, model_slug, logger,
        )

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_task, rec): rec.get("id", "?") for rec in records}
        with tqdm(total=len(futures), desc="evaluate") as pbar:
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.warning(f"sample {sid} evaluate failed: {exc!r}\n{traceback.format_exc()}")
                    res = SampleResult(
                        id=sid,
                        category=id_to_category.get(sid, "uncategorized"),
                        success=0, progress=0.0,
                        step_count=0, error=f"{type(exc).__name__}: {exc}",
                    )
                sample_results.append(res)
                pbar.update(1)

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
