"""Build summary.txt and summary.csv from per-sample results."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CATEGORY_FILES = [
    ("gorilla_fs", "ids_gorilla_fs.json"),
    ("vehicle_control", "ids_vehicle_control.json"),
    ("trading_bot", "ids_trading_bot.json"),
    ("travel_api", "ids_travel_api.json"),
]
TEST_CATEGORY_KEY = "multi_turn_ours"


@dataclass
class SampleResult:
    id: str
    category: str
    success: int
    progress: float
    step_count: int
    error: Optional[str]
    # Inference (main LLM calls only)
    latency_inference: float = 0.0
    input_tokens_inference: int = 0
    output_tokens_inference: int = 0
    total_tokens_inference: int = 0
    # Memory (LLM + embedding calls inside mem0)
    latency_memory: float = 0.0
    input_tokens_memory: int = 0
    output_tokens_memory: int = 0
    total_tokens_memory: int = 0
    embedding_tokens: int = 0
    # Total (inference + memory)
    latency_total: float = 0.0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_tokens_total: int = 0


def load_id_to_category(ids_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for category, filename in CATEGORY_FILES:
        with open(ids_dir / filename) as f:
            data = json.load(f)
        for sample_id in data[TEST_CATEGORY_KEY]:
            mapping[sample_id] = category
    return mapping


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(vals: list[int], p: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _dist_row(label: str, toks: list[int]) -> str:
    if not toks:
        return f"{label:<20}  (no data)\n"
    n = len(toks)
    mean = sum(toks) / n
    return (
        f"{label:<20}  n={n:4d}  min={min(toks):>7,}  mean={mean:>8,.0f}"
        f"  p50={_percentile(toks,50):>7,}  p75={_percentile(toks,75):>7,}"
        f"  p90={_percentile(toks,90):>7,}  p95={_percentile(toks,95):>7,}"
        f"  p99={_percentile(toks,99):>7,}  max={max(toks):>7,}\n"
    )


def _aggregate_block(samples: list[SampleResult]) -> dict:
    completed = [s for s in samples if s.error is None]
    return {
        "n": len(samples),
        "n_failed": len(samples) - len(completed),
        "success_rate": _safe_mean([s.success for s in samples]),
        "progress_score": _safe_mean([s.progress for s in samples]),
        "avg_step_count": _safe_mean([s.step_count for s in completed]),
        # Inference
        "avg_latency_s_inference": _safe_mean([s.latency_inference for s in completed]),
        "avg_input_tokens_inference": _safe_mean([s.input_tokens_inference for s in completed]),
        "avg_output_tokens_inference": _safe_mean([s.output_tokens_inference for s in completed]),
        "avg_total_tokens_inference": _safe_mean([s.total_tokens_inference for s in completed]),
        # Memory
        "avg_latency_s_memory": _safe_mean([s.latency_memory for s in completed]),
        "avg_input_tokens_memory": _safe_mean([s.input_tokens_memory for s in completed]),
        "avg_output_tokens_memory": _safe_mean([s.output_tokens_memory for s in completed]),
        "avg_total_tokens_memory": _safe_mean([s.total_tokens_memory for s in completed]),
        "avg_embedding_tokens": _safe_mean([s.embedding_tokens for s in completed]),
        # Total
        "avg_latency_s_total": _safe_mean([s.latency_total for s in completed]),
        "avg_input_tokens_total": _safe_mean([s.input_tokens_total for s in completed]),
        "avg_output_tokens_total": _safe_mean([s.output_tokens_total for s in completed]),
        "avg_total_tokens_total": _safe_mean([s.total_tokens_total for s in completed]),
    }


def build_summary(
    samples: list[SampleResult],
    output_dir: Path,
    ids_dir: Path,
    model_name: str,
    run_name: str,
) -> None:
    overall = _aggregate_block(samples)

    by_cat: dict[str, list[SampleResult]] = {cat: [] for cat, _ in CATEGORY_FILES}
    uncategorized: list[SampleResult] = []
    for s in samples:
        if s.category in by_cat:
            by_cat[s.category].append(s)
        else:
            uncategorized.append(s)
    cat_aggregates = {cat: _aggregate_block(items) for cat, items in by_cat.items()}

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        bar = "=" * 90
        sep = "-" * 90
        f.write(f"{bar}\n")
        f.write("BFCL multi_turn_ours batch evaluation\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"Run name:     {run_name}\n")
        f.write(
            f"Total samples: {overall['n']}    "
            f"Failed (api/runtime): {overall['n_failed']}    "
            f"Uncategorized: {len(uncategorized)}\n"
        )
        f.write(f"{bar}\n\n")

        f.write(f"OVERALL ({overall['n']} samples)\n")
        f.write(f"{sep}\n")
        f.write(f"success_rate                    : {overall['success_rate']:.4f}\n")
        f.write(f"progress_score                  : {overall['progress_score']:.4f}\n")
        f.write(f"avg_step_count                  : {overall['avg_step_count']:.2f}\n")
        f.write("\n  -- Inference --\n")
        f.write(f"  avg_latency_s                 : {overall['avg_latency_s_inference']:.2f}\n")
        f.write(f"  avg_total_tokens              : {overall['avg_total_tokens_inference']:.1f}\n")
        f.write(f"  avg_input_tokens              : {overall['avg_input_tokens_inference']:.1f}\n")
        f.write(f"  avg_output_tokens             : {overall['avg_output_tokens_inference']:.1f}\n")
        f.write("\n  -- Memory --\n")
        f.write(f"  avg_latency_s                 : {overall['avg_latency_s_memory']:.2f}\n")
        f.write(f"  avg_total_tokens              : {overall['avg_total_tokens_memory']:.1f}\n")
        f.write(f"  avg_input_tokens              : {overall['avg_input_tokens_memory']:.1f}\n")
        f.write(f"  avg_output_tokens             : {overall['avg_output_tokens_memory']:.1f}\n")
        f.write(f"  avg_embedding_tokens          : {overall['avg_embedding_tokens']:.1f}\n")
        f.write("\n  -- Total (inference + memory) --\n")
        f.write(f"  avg_latency_s                 : {overall['avg_latency_s_total']:.2f}\n")
        f.write(f"  avg_total_tokens              : {overall['avg_total_tokens_total']:.1f}\n")
        f.write(f"  avg_input_tokens              : {overall['avg_input_tokens_total']:.1f}\n")
        f.write(f"  avg_output_tokens             : {overall['avg_output_tokens_total']:.1f}\n\n")

        f.write("PER-CATEGORY BREAKDOWN\n")
        f.write(f"{sep}\n")
        metric_rows = [
            ("n",                           "{:>{w}d}",   "n"),
            ("n_failed",                    "{:>{w}d}",   "n_failed"),
            ("success_rate",                "{:>{w}.4f}", "success_rate"),
            ("progress_score",              "{:>{w}.4f}", "progress_score"),
            ("avg_step_count",              "{:>{w}.2f}", "avg_step_count"),
            ("avg_latency_s [infer]",       "{:>{w}.2f}", "avg_latency_s_inference"),
            ("avg_latency_s [mem]",         "{:>{w}.2f}", "avg_latency_s_memory"),
            ("avg_latency_s [total]",       "{:>{w}.2f}", "avg_latency_s_total"),
            ("avg_total_tokens [infer]",    "{:>{w}.1f}", "avg_total_tokens_inference"),
            ("avg_total_tokens [mem]",      "{:>{w}.1f}", "avg_total_tokens_memory"),
            ("avg_total_tokens [total]",    "{:>{w}.1f}", "avg_total_tokens_total"),
            ("avg_embedding_tokens",        "{:>{w}.1f}", "avg_embedding_tokens"),
        ]
        label_w = 28
        col_w = {cat: max(len(cat), 12) for cat, _ in CATEGORY_FILES}
        f.write(
            f"{'metric':<{label_w}}"
            + "  ".join(f"{cat:>{col_w[cat]}}" for cat, _ in CATEGORY_FILES)
            + "\n"
        )
        for label, fmt, key in metric_rows:
            cells = [fmt.format(cat_aggregates[cat][key], w=col_w[cat]) for cat, _ in CATEGORY_FILES]
            f.write(f"{label:<{label_w}}" + "  ".join(cells) + "\n")
        if uncategorized:
            f.write(f"\n(Note: {len(uncategorized)} samples did not match any ids_*.json category.)\n")

        results_dir = output_dir / "results"
        step_toks_all: list[int] = []
        step_toks_by_cat: dict[str, list[int]] = {cat: [] for cat, _ in CATEGORY_FILES}
        cat_of_id = {s.id: s.category for s in samples}
        if results_dir.is_dir():
            for p in sorted(results_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue
                cat = cat_of_id.get(d.get("id", p.stem))
                for turn in d.get("per_turn_totals", []):
                    for tok in turn.get("input_tokens_per_step", []):
                        step_toks_all.append(tok)
                        if cat in step_toks_by_cat:
                            step_toks_by_cat[cat].append(tok)
        if step_toks_all:
            f.write("\nPER-STEP INPUT TOKEN DISTRIBUTION\n")
            f.write(f"{sep}\n")
            f.write(_dist_row("overall", step_toks_all))
            for cat, _ in CATEGORY_FILES:
                f.write(_dist_row(cat, step_toks_by_cat[cat]))
        f.write(f"{bar}\n")

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "scope", "n", "n_failed",
        "success_rate", "progress_score", "avg_step_count",
        "avg_latency_s_inference", "avg_input_tokens_inference",
        "avg_output_tokens_inference", "avg_total_tokens_inference",
        "avg_latency_s_memory", "avg_input_tokens_memory",
        "avg_output_tokens_memory", "avg_total_tokens_memory",
        "avg_embedding_tokens",
        "avg_latency_s_total", "avg_input_tokens_total",
        "avg_output_tokens_total", "avg_total_tokens_total",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"scope": "overall", **overall})
        for cat, _ in CATEGORY_FILES:
            writer.writerow({"scope": cat, **cat_aggregates[cat]})

    per_sample_csv = output_dir / "per_sample.csv"
    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "category", "success", "progress", "step_count", "error",
                "latency_s_inference", "input_tokens_inference",
                "output_tokens_inference", "total_tokens_inference",
                "latency_s_memory", "input_tokens_memory",
                "output_tokens_memory", "total_tokens_memory",
                "embedding_tokens",
                "latency_s_total", "input_tokens_total",
                "output_tokens_total", "total_tokens_total",
            ],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s.id,
                "category": s.category,
                "success": s.success,
                "progress": round(s.progress, 4),
                "step_count": s.step_count,
                "error": s.error or "",
                "latency_s_inference": round(s.latency_inference, 4),
                "input_tokens_inference": s.input_tokens_inference,
                "output_tokens_inference": s.output_tokens_inference,
                "total_tokens_inference": s.total_tokens_inference,
                "latency_s_memory": round(s.latency_memory, 4),
                "input_tokens_memory": s.input_tokens_memory,
                "output_tokens_memory": s.output_tokens_memory,
                "total_tokens_memory": s.total_tokens_memory,
                "embedding_tokens": s.embedding_tokens,
                "latency_s_total": round(s.latency_total, 4),
                "input_tokens_total": s.input_tokens_total,
                "output_tokens_total": s.output_tokens_total,
                "total_tokens_total": s.total_tokens_total,
            })


def sample_result_from_record(rec: dict, id_to_category: dict[str, str]) -> SampleResult:
    """Re-hydrate a SampleResult from a persisted per-sample JSON (for resume or evaluate phase)."""
    metrics_d = rec.get("metrics", {})
    inf = rec.get("inference_totals") or rec.get("totals", {})
    mem = rec.get("memory_totals", {})
    tot = rec.get("total_totals") or {}
    if not tot:
        tot = {
            "latency_s": inf.get("latency_s", 0.0),
            "input_tokens": inf.get("input_tokens", 0),
            "output_tokens": inf.get("output_tokens", 0),
            "total_tokens": inf.get("total_tokens", 0),
        }
    return SampleResult(
        id=rec["id"],
        category=rec.get("category") or id_to_category.get(rec["id"], "uncategorized"),
        success=int(metrics_d.get("success_rate", 0)),
        progress=float(metrics_d.get("progress_score", 0.0)),
        step_count=int(inf.get("step_count", 0)),
        error=rec.get("error"),
        latency_inference=float(inf.get("latency_s", 0.0)),
        input_tokens_inference=int(inf.get("input_tokens", 0)),
        output_tokens_inference=int(inf.get("output_tokens", 0)),
        total_tokens_inference=int(inf.get("total_tokens", 0)),
        latency_memory=float(mem.get("latency_s", 0.0)),
        input_tokens_memory=int(mem.get("input_tokens", 0)),
        output_tokens_memory=int(mem.get("output_tokens", 0)),
        total_tokens_memory=int(mem.get("total_tokens", 0)),
        embedding_tokens=int(mem.get("embedding_tokens", 0)),
        latency_total=float(tot.get("latency_s", 0.0)),
        input_tokens_total=int(tot.get("input_tokens", 0)),
        output_tokens_total=int(tot.get("output_tokens", 0)),
        total_tokens_total=int(tot.get("total_tokens", 0)),
    )
