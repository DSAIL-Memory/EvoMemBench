"""Merge per-sample CSVs from all Phase-1 and Phase-2 sub-runs into two combined files.

Writes into BASE_DIR:
  combined_per_sample.csv  — all individual sample rows, tagged with phase+setting
  combined_summary.csv     — aggregate metrics per sub-run + phase-level + overall
  combined_summary.txt     — human-readable version of the combined summary
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PER_SAMPLE_FIELDS = [
    "id", "category", "success", "progress", "step_count", "error",
    "latency_s_inference", "input_tokens_inference",
    "output_tokens_inference", "total_tokens_inference",
    "latency_s_memory", "input_tokens_memory",
    "output_tokens_memory", "total_tokens_memory", "embedding_tokens",
    "latency_s_total", "input_tokens_total",
    "output_tokens_total", "total_tokens_total",
]

SUMMARY_FIELDS = [
    "phase", "setting",
    "n", "n_failed", "success_rate", "progress_score", "avg_step_count",
    "avg_latency_s_inference", "avg_input_tokens_inference",
    "avg_output_tokens_inference", "avg_total_tokens_inference",
    "avg_latency_s_memory", "avg_input_tokens_memory",
    "avg_output_tokens_memory", "avg_total_tokens_memory",
    "avg_embedding_tokens",
    "avg_latency_s_total", "avg_input_tokens_total",
    "avg_output_tokens_total", "avg_total_tokens_total",
]

# Canonical column orders (must match the shell script's SUBSETS / P2_A/B_SUBSETS).
P1_SUBSETS = ["gorilla_fs", "vehicle_control", "trading_bot", "travel_api"]
P2_SETTINGS_DIR = [
    "trading_bot__to__travel_api",
    "travel_api__to__trading_bot",
    "gorilla_fs__to__vehicle_control",
    "vehicle_control__to__gorilla_fs",
]
P2_SETTINGS_LABEL = [s.replace("__to__", "→") for s in P2_SETTINGS_DIR]

# Metric rows for the per-group tables: (label, fmt_str, agg_key)
# Total first, then Inference, then Memory — per user preference.
TABLE_METRIC_ROWS = [
    ("n",                           "{:>{w}d}",   "n"),
    ("n_failed",                    "{:>{w}d}",   "n_failed"),
    ("success_rate",                "{:>{w}.4f}", "success_rate"),
    ("progress_score",              "{:>{w}.4f}", "progress_score"),
    ("avg_step_count",              "{:>{w}.2f}", "avg_step_count"),
    ("avg_latency_s [total]",       "{:>{w}.2f}", "avg_latency_s_total"),
    ("avg_total_tokens [total]",    "{:>{w}.1f}", "avg_total_tokens_total"),
    ("avg_latency_s [infer]",       "{:>{w}.2f}", "avg_latency_s_inference"),
    ("avg_total_tokens [infer]",    "{:>{w}.1f}", "avg_total_tokens_inference"),
    ("avg_latency_s [mem]",         "{:>{w}.2f}", "avg_latency_s_memory"),
    ("avg_total_tokens [mem]",      "{:>{w}.1f}", "avg_total_tokens_memory"),
    ("avg_embedding_tokens",        "{:>{w}.1f}", "avg_embedding_tokens"),
]


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


def _collect_all_step_tokens(base: Path) -> dict[str, list[int]]:
    """Collect input_tokens_per_step from all results/*.json under phase1/phase2."""
    result: dict[str, list[int]] = {"overall": []}
    for subset in P1_SUBSETS:
        result[subset] = []
    for phase_dir in [base / "phase1", base / "phase2"]:
        if not phase_dir.is_dir():
            continue
        for sub_dir in sorted(phase_dir.iterdir()):
            subset = sub_dir.name
            results_dir = sub_dir / "results"
            if not results_dir.is_dir():
                continue
            for p in sorted(results_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue
                for turn in d.get("per_turn_totals", []):
                    for tok in turn.get("input_tokens_per_step", []):
                        result["overall"].append(tok)
                        if subset in result:
                            result[subset].append(tok)
    return result


def _write_step_token_dist_section(f, step_toks: dict[str, list[int]]) -> None:
    sep = "-" * 90
    f.write("PER-STEP INPUT TOKEN DISTRIBUTION\n")
    f.write(f"{sep}\n")
    f.write(_dist_row("overall", step_toks["overall"]))
    for subset in P1_SUBSETS:
        if step_toks.get(subset):
            f.write(_dist_row(subset, step_toks[subset]))
    f.write("\n")


def _aggregate(rows: list[dict]) -> dict:
    completed = [r for r in rows if not r.get("error")]
    return {
        "n": len(rows),
        "n_failed": len(rows) - len(completed),
        "success_rate": round(_safe_mean([int(r["success"]) for r in rows]), 6),
        "progress_score": round(_safe_mean([float(r["progress"]) for r in rows]), 6),
        "avg_step_count": round(_safe_mean([int(r["step_count"]) for r in completed]), 4),
        "avg_latency_s_inference": round(_safe_mean([float(r["latency_s_inference"]) for r in completed]), 4),
        "avg_input_tokens_inference": round(_safe_mean([int(r["input_tokens_inference"]) for r in completed]), 2),
        "avg_output_tokens_inference": round(_safe_mean([int(r["output_tokens_inference"]) for r in completed]), 2),
        "avg_total_tokens_inference": round(_safe_mean([int(r["total_tokens_inference"]) for r in completed]), 2),
        "avg_latency_s_memory": round(_safe_mean([float(r["latency_s_memory"]) for r in completed]), 4),
        "avg_input_tokens_memory": round(_safe_mean([int(r["input_tokens_memory"]) for r in completed]), 2),
        "avg_output_tokens_memory": round(_safe_mean([int(r["output_tokens_memory"]) for r in completed]), 2),
        "avg_total_tokens_memory": round(_safe_mean([int(r["total_tokens_memory"]) for r in completed]), 2),
        "avg_embedding_tokens": round(_safe_mean([int(r["embedding_tokens"]) for r in completed]), 2),
        "avg_latency_s_total": round(_safe_mean([float(r["latency_s_total"]) for r in completed]), 4),
        "avg_input_tokens_total": round(_safe_mean([int(r["input_tokens_total"]) for r in completed]), 2),
        "avg_output_tokens_total": round(_safe_mean([int(r["output_tokens_total"]) for r in completed]), 2),
        "avg_total_tokens_total": round(_safe_mean([int(r["total_tokens_total"]) for r in completed]), 2),
    }


def _write_overall_block(f, agg: dict) -> None:
    sep = "-" * 90
    f.write(f"OVERALL ({agg['n']} samples)\n")
    f.write(f"{sep}\n")
    f.write(f"success_rate                    : {agg['success_rate']:.4f}\n")
    f.write(f"progress_score                  : {agg['progress_score']:.4f}\n")
    f.write(f"avg_step_count                  : {agg['avg_step_count']:.2f}\n")
    f.write("\n  -- Total (inference + memory) --\n")
    f.write(f"  avg_latency_s                 : {agg['avg_latency_s_total']:.2f}\n")
    f.write(f"  avg_total_tokens              : {agg['avg_total_tokens_total']:.1f}\n")
    f.write(f"  avg_input_tokens              : {agg['avg_input_tokens_total']:.1f}\n")
    f.write(f"  avg_output_tokens             : {agg['avg_output_tokens_total']:.1f}\n")
    f.write("\n  -- Inference --\n")
    f.write(f"  avg_latency_s                 : {agg['avg_latency_s_inference']:.2f}\n")
    f.write(f"  avg_total_tokens              : {agg['avg_total_tokens_inference']:.1f}\n")
    f.write(f"  avg_input_tokens              : {agg['avg_input_tokens_inference']:.1f}\n")
    f.write(f"  avg_output_tokens             : {agg['avg_output_tokens_inference']:.1f}\n")
    f.write("\n  -- Memory --\n")
    f.write(f"  avg_latency_s                 : {agg['avg_latency_s_memory']:.2f}\n")
    f.write(f"  avg_total_tokens              : {agg['avg_total_tokens_memory']:.1f}\n")
    f.write(f"  avg_input_tokens              : {agg['avg_input_tokens_memory']:.1f}\n")
    f.write(f"  avg_output_tokens             : {agg['avg_output_tokens_memory']:.1f}\n")
    f.write(f"  avg_embedding_tokens          : {agg['avg_embedding_tokens']:.1f}\n\n")


def _write_phase_table(
    f,
    title: str,
    col_keys: list[str],   # directory-style keys into group_rows
    col_labels: list[str], # human-readable column headers
    group_rows: dict[tuple[str, str], list[dict]],
    phase_label: str,
) -> None:
    sep = "-" * 90
    f.write(f"{title}\n")
    f.write(f"{sep}\n")

    col_w = {k: max(len(lbl), 12) for k, lbl in zip(col_keys, col_labels)}
    label_w = 28

    # Header row
    f.write(f"{'metric':<{label_w}}")
    f.write("  ".join(f"{lbl:>{col_w[k]}}" for k, lbl in zip(col_keys, col_labels)))
    f.write("\n")

    # Pre-compute aggregates; use None sentinel for missing groups
    aggs: dict[str, dict | None] = {}
    for k in col_keys:
        rows = group_rows.get((phase_label, k))
        aggs[k] = _aggregate(rows) if rows is not None else None

    # Data rows
    for label, fmt, key in TABLE_METRIC_ROWS:
        cells = []
        for k in col_keys:
            if aggs[k] is None:
                cells.append(f"{'—':>{col_w[k]}}")
            else:
                val = aggs[k][key]
                cells.append(fmt.format(val, w=col_w[k]))
        f.write(f"{label:<{label_w}}" + "  ".join(cells) + "\n")
    f.write("\n")


def _write_combined_txt(base: Path, group_rows: dict[tuple[str, str], list[dict]]) -> Path:
    all_rows = [r for rows in group_rows.values() for r in rows]
    overall = _aggregate(all_rows) if all_rows else _aggregate([])

    bar = "=" * 90
    out_path = base / "combined_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"{bar}\n")
        f.write("BFCL multi_turn_ours evaluation (mem0 | in-env + cross-env)\n")
        f.write(f"Run dir:        {base.name}\n")
        f.write(
            f"Total samples:  {overall['n']}    "
            f"Failed (api/runtime): {overall['n_failed']}\n"
        )
        f.write(f"{bar}\n\n")

        _write_overall_block(f, overall)

        p1_keys = [k for k in P1_SUBSETS if ("1", k) in group_rows]
        if p1_keys:
            _write_phase_table(
                f,
                "PHASE 1 — in-env, per subset",
                p1_keys,
                p1_keys,
                group_rows,
                "1",
            )

        p2_present = [
            (dk, lbl)
            for dk, lbl in zip(P2_SETTINGS_DIR, P2_SETTINGS_LABEL)
            if ("2", dk) in group_rows
        ]
        if p2_present:
            p2_keys, p2_labels = zip(*p2_present)
            _write_phase_table(
                f,
                "PHASE 2 — cross-env, per setting (a → b)",
                list(p2_keys),
                list(p2_labels),
                group_rows,
                "2",
            )

        step_toks = _collect_all_step_tokens(base)
        if step_toks["overall"]:
            _write_step_token_dist_section(f, step_toks)

        f.write(f"{bar}\n")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_dir", type=Path)
    args = ap.parse_args()
    base: Path = args.base_dir

    all_rows: list[dict] = []
    group_rows: dict[tuple[str, str], list[dict]] = {}  # (phase, setting) → rows

    for phase_label, phase_dir in [("1", base / "phase1"), ("2", base / "phase2")]:
        if not phase_dir.is_dir():
            continue
        for sub_dir in sorted(phase_dir.iterdir()):
            csv_path = sub_dir / "per_sample.csv"
            if not csv_path.is_file():
                continue
            setting = sub_dir.name
            key = (phase_label, setting)
            group_rows[key] = []
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    tagged = {"phase": phase_label, "setting": setting, **row}
                    all_rows.append(tagged)
                    group_rows[key].append(row)

    # combined_per_sample.csv
    per_sample_out = base / "combined_per_sample.csv"
    with open(per_sample_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "setting"] + PER_SAMPLE_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # combined_summary.csv
    summary_rows: list[dict] = []
    p1_all: list[dict] = []
    p2_all: list[dict] = []

    for (phase_label, setting), rows in group_rows.items():
        agg = _aggregate(rows)
        summary_rows.append({"phase": phase_label, "setting": setting, **agg})
        if phase_label == "1":
            p1_all.extend(rows)
        else:
            p2_all.extend(rows)

    if p1_all:
        agg = _aggregate(p1_all)
        summary_rows.append({"phase": "1", "setting": "aggregate", **agg})
    if p2_all:
        agg = _aggregate(p2_all)
        summary_rows.append({"phase": "2", "setting": "aggregate", **agg})
    overall_rows = [r for rows in group_rows.values() for r in rows]
    if overall_rows:
        agg = _aggregate(overall_rows)
        summary_rows.append({"phase": "overall", "setting": "aggregate", **agg})

    summary_out = base / "combined_summary.csv"
    with open(summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    # combined_summary.txt
    summary_txt_out = _write_combined_txt(base, group_rows)

    print(f"  combined_per_sample.csv : {per_sample_out}  ({len(all_rows)} rows)")
    print(f"  combined_summary.csv    : {summary_out}  ({len(summary_rows)} rows)")
    print(f"  combined_summary.txt    : {summary_txt_out}")


if __name__ == "__main__":
    main()
