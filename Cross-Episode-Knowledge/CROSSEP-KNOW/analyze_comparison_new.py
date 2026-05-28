"""Comparison Analysis Script - Context-Memory vs. Independent Baseline
(Extended version: compares efficiency against baseline inference first)

Compares two graded JSONL files on:
  - Effectiveness: solving rates (overall, per context_category, per context_id)
  - Efficiency: baseline vs memory token/latency comparison (if --baseline-infer provided),
                then memory overhead breakdown (if --infer-memory provided)

Usage:
    python analyze_comparison_new.py \\
        --baseline outputs/CL-bench_context_ge5_deepseek-chat_..._graded.jsonl \\
        --memory   outputs/context_reasoning_bank/CL-bench_context_ge5_..._graded.jsonl \\
        --baseline-infer outputs/context_nomemory/CL-bench_context_ge5_..._ctx_nomemory_....jsonl \\
        --infer-memory outputs/context_reasoning_bank/CL-bench_context_ge5_..._ctx_memory_topk10_....jsonl \\
        --output   comparison_report.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_graded(path):
    """Load a graded JSONL file.

    Returns:
        dict: {task_id -> {"score": int, "context_id": str,
                           "context_category": str, "sub_category": str}}
    """
    records = _load_jsonl(path)
    result = {}
    for rec in records:
        # task_id: prefer metadata.task_id, fallback to idx
        task_id = rec.get("metadata", {}).get("task_id") or rec.get("idx")
        if task_id is None:
            continue
        score = rec.get("score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        meta = rec.get("metadata", {})
        result[task_id] = {
            "score": score,
            "context_id": meta.get("context_id", ""),
            "context_category": meta.get("context_category", "Unknown"),
            "sub_category": meta.get("sub_category", "Unknown"),
        }
    return result


def _sum_tokens(usage_dict):
    if not isinstance(usage_dict, dict):
        return 0
    return usage_dict.get("total_tokens", 0)


def load_infer_stats(path):
    """Load inference JSONL with stats field.

    Returns:
        dict: {task_id -> {"latency": {...}, "tokens": {...}}}
              or empty dict if path is None
    """
    if path is None:
        return {}
    records = _load_jsonl(path)
    result = {}
    for rec in records:
        task_id = rec.get("idx") or rec.get("metadata", {}).get("task_id")
        if task_id is None:
            continue
        stats = rec.get("stats")
        if stats:
            result[task_id] = stats
    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_overall(baseline, memory):
    """Compute overall solving rates on the intersection of task_ids.

    Returns:
        dict with keys: n_common, n_only_baseline, n_only_memory,
                        baseline_rate, memory_rate, delta
    """
    b_ids = set(baseline.keys())
    m_ids = set(memory.keys())
    common = b_ids & m_ids

    n = len(common)
    b_score = sum(baseline[t]["score"] for t in common)
    m_score = sum(memory[t]["score"] for t in common)

    b_rate = b_score / n if n > 0 else 0.0
    m_rate = m_score / n if n > 0 else 0.0

    return {
        "n_common": n,
        "n_only_baseline": len(b_ids - m_ids),
        "n_only_memory": len(m_ids - b_ids),
        "baseline_score": b_score,
        "memory_score": m_score,
        "baseline_rate": b_rate,
        "memory_rate": m_rate,
        "delta": m_rate - b_rate,
    }


def compute_per_category(baseline, memory, infer_stats=None, baseline_infer_stats=None):
    """Compute solving rates grouped by context_category.

    Returns:
        list of dicts sorted by category name, each with:
        {category, n, baseline_rate, memory_rate, delta,
         total_baseline_tokens (or None), total_baseline_latency_s (or None),
         total_memory_tokens (or None), total_memory_latency_s (or None)}
    """
    common = set(baseline.keys()) & set(memory.keys())
    cat_data = defaultdict(lambda: {
        "b": 0, "m": 0, "n": 0,
        "bl_tokens": 0, "bl_latency": 0.0, "bl_stats_count": 0,
        "mem_tokens": 0, "mem_latency": 0.0, "mem_stats_count": 0,
    })

    for t in common:
        cat = baseline[t]["context_category"]
        cat_data[cat]["n"] += 1
        cat_data[cat]["b"] += baseline[t]["score"]
        cat_data[cat]["m"] += memory[t]["score"]

        if baseline_infer_stats and t in baseline_infer_stats:
            bl_tok = baseline_infer_stats[t].get("tokens", {})
            bl_lat = baseline_infer_stats[t].get("latency", {})
            cat_data[cat]["bl_tokens"] += _sum_tokens(bl_tok.get("inference", {}))
            cat_data[cat]["bl_latency"] += bl_lat.get("inference_s", 0.0)
            cat_data[cat]["bl_stats_count"] += 1

        if infer_stats and t in infer_stats:
            mem_tok = infer_stats[t].get("tokens", {})
            mem_lat = infer_stats[t].get("latency", {})
            cat_data[cat]["mem_tokens"] += (
                _sum_tokens(mem_tok.get("inference", {}))
                + _sum_tokens(mem_tok.get("extract_llm", {}))
            )
            cat_data[cat]["mem_latency"] += (
                mem_lat.get("inference_s", 0.0)
                + mem_lat.get("retrieve_s", 0.0)
                + mem_lat.get("extract_s", 0.0)
            )
            cat_data[cat]["mem_stats_count"] += 1

    rows = []
    for cat in sorted(cat_data.keys()):
        d = cat_data[cat]
        n = d["n"]
        b_rate = d["b"] / n if n > 0 else 0.0
        m_rate = d["m"] / n if n > 0 else 0.0
        bl_n = d["bl_stats_count"]
        mem_n = d["mem_stats_count"]
        rows.append({
            "category": cat,
            "n": n,
            "baseline_rate": b_rate,
            "memory_rate": m_rate,
            "delta": m_rate - b_rate,
            "avg_baseline_tokens": d["bl_tokens"] / bl_n if bl_n > 0 else None,
            "avg_baseline_latency_s": d["bl_latency"] / bl_n if bl_n > 0 else None,
            "avg_memory_tokens": d["mem_tokens"] / mem_n if mem_n > 0 else None,
            "avg_memory_latency_s": d["mem_latency"] / mem_n if mem_n > 0 else None,
        })
    return rows


def compute_per_context(baseline, memory, infer_stats):
    """Compute solving rates per context_id.

    Args:
        infer_stats: dict from load_infer_stats (may be empty)

    Returns:
        list of dicts sorted by delta descending, each with:
        {context_id, context_category, sub_category, n_tasks,
         baseline_rate, memory_rate, delta,
         avg_extra_tokens (or None), avg_extra_latency_s (or None)}
    """
    common = set(baseline.keys()) & set(memory.keys())

    # Group by context_id
    ctx_data = defaultdict(lambda: {
        "b": 0, "m": 0, "n": 0,
        "category": "", "sub_category": "",
        "extra_tokens": 0, "extra_latency": 0.0, "stats_count": 0,
    })

    for t in common:
        ctx_id = baseline[t]["context_id"]
        d = ctx_data[ctx_id]
        d["n"] += 1
        d["b"] += baseline[t]["score"]
        d["m"] += memory[t]["score"]
        d["category"] = baseline[t]["context_category"]
        d["sub_category"] = baseline[t]["sub_category"]

        # Efficiency stats from memory inference file
        if t in infer_stats:
            tok = infer_stats[t].get("tokens", {})
            lat = infer_stats[t].get("latency", {})
            # Extra cost = memory overhead (not inference itself)
            extra_tok = _sum_tokens(tok.get("extract_llm", {}))
            extra_lat = (
                lat.get("retrieve_s", 0.0) + lat.get("extract_s", 0.0)
            )
            d["extra_tokens"] += extra_tok
            d["extra_latency"] += extra_lat
            d["stats_count"] += 1

    rows = []
    for ctx_id, d in ctx_data.items():
        n = d["n"]
        b_rate = d["b"] / n if n > 0 else 0.0
        m_rate = d["m"] / n if n > 0 else 0.0
        has_stats = d["stats_count"] > 0
        rows.append({
            "context_id": ctx_id,
            "context_category": d["category"],
            "sub_category": d["sub_category"],
            "n_tasks": n,
            "baseline_n1": d["b"],
            "memory_n1": d["m"],
            "baseline_rate": b_rate,
            "memory_rate": m_rate,
            "delta": m_rate - b_rate,
            "avg_extra_tokens": d["extra_tokens"] / d["stats_count"] if has_stats else None,
            "avg_extra_latency_s": d["extra_latency"] / d["stats_count"] if has_stats else None,
        })

    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def compute_efficiency(baseline, memory, infer_stats, baseline_infer_stats=None):
    """Compute aggregate efficiency metrics.

    Args:
        baseline: graded dict for baseline setting
        memory: graded dict for memory setting
        infer_stats: raw inference stats dict for memory run (from load_infer_stats)
        baseline_infer_stats: raw inference stats dict for baseline (no-memory) run.
            When provided, a side-by-side baseline vs memory comparison is computed
            and included under the "baseline_comparison" key.

    Returns:
        dict with token/latency overhead and cost-effectiveness metrics,
        plus optional "baseline_comparison" sub-dict when baseline_infer_stats
        is provided.
    """
    if not infer_stats:
        return None

    common = set(baseline.keys()) & set(memory.keys())
    common_with_stats = common & set(infer_stats.keys())

    if not common_with_stats:
        return None

    total_extra_tokens = 0
    total_extra_latency = 0.0
    total_inference_tokens = 0
    total_inference_latency = 0.0
    total_embed_tokens = 0

    for t in common_with_stats:
        tok = infer_stats[t].get("tokens", {})
        lat = infer_stats[t].get("latency", {})
        total_extra_tokens += _sum_tokens(tok.get("extract_llm", {}))
        total_extra_latency += lat.get("retrieve_s", 0.0) + lat.get("extract_s", 0.0)
        total_inference_tokens += _sum_tokens(tok.get("inference", {}))
        total_inference_latency += lat.get("inference_s", 0.0)
        total_embed_tokens += (
            _sum_tokens(tok.get("retrieve_embed", {}))
            + _sum_tokens(tok.get("extract_embed", {}))
        )

    n = len(common_with_stats)
    b_rate = sum(baseline[t]["score"] for t in common_with_stats) / n
    m_rate = sum(memory[t]["score"] for t in common_with_stats) / n
    delta = m_rate - b_rate

    avg_extra_tokens = total_extra_tokens / n
    avg_extra_latency = total_extra_latency / n

    if delta > 0:
        tokens_per_pct_point = total_extra_tokens / (delta * 100)
    else:
        tokens_per_pct_point = None

    result = {
        "n_with_stats": n,
        "total_extra_tokens": total_extra_tokens,
        "avg_extra_tokens_per_sample": avg_extra_tokens,
        "total_embed_tokens": total_embed_tokens,
        "avg_embed_tokens_per_sample": total_embed_tokens / n,
        "total_inference_tokens": total_inference_tokens,
        "avg_inference_tokens_per_sample": total_inference_tokens / n,
        "avg_inference_latency_per_sample": total_inference_latency / n,
        "extra_token_overhead_pct": (total_extra_tokens / total_inference_tokens * 100)
        if total_inference_tokens > 0 else None,
        "total_extra_latency_s": total_extra_latency,
        "avg_extra_latency_s_per_sample": avg_extra_latency,
        "score_improvement_delta": delta,
        "tokens_per_pct_point_gain": tokens_per_pct_point,
        "baseline_comparison": None,
    }

    # Compute side-by-side efficiency comparison vs baseline inference
    if baseline_infer_stats:
        common_with_both = common_with_stats & set(baseline_infer_stats.keys())
        if common_with_both:
            bl_total_tokens = 0
            bl_total_latency = 0.0
            mem_total_tokens = 0
            mem_extra_tokens = 0
            mem_total_latency = 0.0
            mem_extra_latency = 0.0

            for t in common_with_both:
                # Baseline (no-memory) stats
                bl_tok = baseline_infer_stats[t].get("tokens", {})
                bl_lat = baseline_infer_stats[t].get("latency", {})
                bl_total_tokens += _sum_tokens(bl_tok.get("inference", {}))
                bl_total_latency += bl_lat.get("inference_s", 0.0)

                # Memory stats
                mem_tok = infer_stats[t].get("tokens", {})
                mem_lat = infer_stats[t].get("latency", {})
                mem_inf_tok = _sum_tokens(mem_tok.get("inference", {}))
                mem_ext_tok = _sum_tokens(mem_tok.get("extract_llm", {}))
                mem_total_tokens += mem_inf_tok + mem_ext_tok
                mem_extra_tokens += mem_ext_tok
                mem_inf_lat = mem_lat.get("inference_s", 0.0)
                mem_ext_lat = mem_lat.get("retrieve_s", 0.0) + mem_lat.get("extract_s", 0.0)
                mem_total_latency += mem_inf_lat + mem_ext_lat
                mem_extra_latency += mem_ext_lat

            nb = len(common_with_both)
            avg_bl_tokens = bl_total_tokens / nb
            avg_bl_latency = bl_total_latency / nb
            avg_mem_total_tokens = mem_total_tokens / nb
            avg_mem_total_latency = mem_total_latency / nb

            result["baseline_comparison"] = {
                "n": nb,
                "avg_baseline_tokens_per_sample": avg_bl_tokens,
                "avg_baseline_latency_per_sample": avg_bl_latency,
                "avg_memory_total_tokens_per_sample": avg_mem_total_tokens,
                "avg_memory_extra_tokens_per_sample": mem_extra_tokens / nb,
                "avg_memory_total_latency_per_sample": avg_mem_total_latency,
                "avg_memory_extra_latency_per_sample": mem_extra_latency / nb,
                "token_overhead_vs_baseline_pct": (
                    (avg_mem_total_tokens - avg_bl_tokens) / avg_bl_tokens * 100
                ) if avg_bl_tokens > 0 else None,
                "latency_overhead_vs_baseline_pct": (
                    (avg_mem_total_latency - avg_bl_latency) / avg_bl_latency * 100
                ) if avg_bl_latency > 0 else None,
            }

    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt_rate(r):
    return f"{r*100:.1f}%"


def _fmt_delta(d):
    sign = "+" if d > 0 else ""
    return f"{sign}{d*100:.1f}%"


def _fmt_pct(v):
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _render_table(headers, rows):
    """Return aligned markdown table lines."""
    all_rows = [list(headers)] + [list(r) for r in rows]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    def fmt_row(row):
        return "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return [fmt_row(headers), sep] + [fmt_row(r) for r in rows]


def render_markdown(overall, per_category, per_context, efficiency, args):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# CL-bench Comparison Report: Context-Memory vs. Independent Baseline")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Baseline:** `{args.baseline}`")
    lines.append(f"**Memory setting:** `{args.memory}`")
    if args.infer_memory:
        lines.append(f"**Inference stats (memory):** `{args.infer_memory}`")
    if args.baseline_infer:
        lines.append(f"**Inference stats (baseline):** `{args.baseline_infer}`")
    lines.append("")

    # Overall
    lines.append("## Overall Effectiveness")
    lines.append("")
    lines.append(f"Common samples: **{overall['n_common']}**")
    if overall["n_only_baseline"] > 0:
        lines.append(f"  (only in baseline: {overall['n_only_baseline']}, only in memory: {overall['n_only_memory']})")
    lines.append("")
    lines.extend(_render_table(
        ["Setting", "Score=1", "Solving Rate"],
        [
            ["Baseline (independent)", str(overall["baseline_score"]),  _fmt_rate(overall["baseline_rate"])],
            ["Context-memory",         str(overall["memory_score"]),    _fmt_rate(overall["memory_rate"])],
            ["Delta",                  f"{overall['memory_score'] - overall['baseline_score']:+d}", _fmt_delta(overall["delta"])],
        ],
    ))
    lines.append("")

    # Per category
    lines.append("## Effectiveness & Efficiency by Context Category")
    lines.append("")
    has_bl_stat  = per_category and per_category[0]["avg_baseline_tokens"] is not None
    has_mem_stat = per_category and per_category[0]["avg_memory_tokens"]   is not None

    cat_headers = ["Category", "N", "Baseline", "Memory", "Delta"]
    if has_bl_stat:
        cat_headers += ["No-Memory Avg Tokens/Sample", "No-Memory Avg Latency/Sample"]
    if has_mem_stat:
        cat_headers += ["Current Method Avg Tokens/Sample", "Current Method Avg Latency/Sample"]

    cat_rows = []
    for row in per_category:
        r = [row["category"], str(row["n"]),
             _fmt_rate(row["baseline_rate"]), _fmt_rate(row["memory_rate"]), _fmt_delta(row["delta"])]
        if has_bl_stat:
            r += [
                f"{row['avg_baseline_tokens']:,.0f}" if row["avg_baseline_tokens"] is not None else "N/A",
                f"{row['avg_baseline_latency_s']:.2f}s" if row["avg_baseline_latency_s"] is not None else "N/A",
            ]
        if has_mem_stat:
            r += [
                f"{row['avg_memory_tokens']:,.0f}" if row["avg_memory_tokens"] is not None else "N/A",
                f"{row['avg_memory_latency_s']:.2f}s" if row["avg_memory_latency_s"] is not None else "N/A",
            ]
        cat_rows.append(r)
    lines.extend(_render_table(cat_headers, cat_rows))
    lines.append("")

    # Efficiency
    if efficiency:
        lines.append("## Efficiency Summary")
        lines.append("")

        bc = efficiency.get("baseline_comparison")
        if bc:
            lines.append("### Baseline vs Memory Efficiency Comparison")
            lines.append("")
            lines.append(f"*(Based on {bc['n']} samples present in both baseline and memory inference files)*")
            lines.append("")
            lines.extend(_render_table(
                ["Metric", "Baseline (no-memory)", "Context-memory", "Overhead"],
                [
                    ["Avg tokens/sample",
                     f"{bc['avg_baseline_tokens_per_sample']:,.0f}",
                     f"{bc['avg_memory_total_tokens_per_sample']:,.0f}",
                     _fmt_pct(bc["token_overhead_vs_baseline_pct"])],
                    ["Avg latency/sample",
                     f"{bc['avg_baseline_latency_per_sample']:.2f}s",
                     f"{bc['avg_memory_total_latency_per_sample']:.2f}s",
                     _fmt_pct(bc["latency_overhead_vs_baseline_pct"])],
                    ["Avg extra tokens (memory overhead)", "—",
                     f"{bc['avg_memory_extra_tokens_per_sample']:,.0f}", "—"],
                    ["Avg extra latency (retrieve+extract)", "—",
                     f"{bc['avg_memory_extra_latency_per_sample']:.2f}s", "—"],
                ],
            ))
            lines.append("")

        lines.append("### Memory Overhead Breakdown")
        lines.append("")
        lines.append(f"*(Based on {efficiency['n_with_stats']} samples with memory inference stats)*")
        lines.append("")
        overhead_rows = [
            ["Avg inference tokens/sample",                f"{efficiency['avg_inference_tokens_per_sample']:,.0f}"],
            ["Avg extra tokens/sample (memory overhead)",  f"{efficiency['avg_extra_tokens_per_sample']:,.0f}"],
        ]
        if efficiency["extra_token_overhead_pct"] is not None:
            overhead_rows.append(["Token overhead % (vs memory inference tokens)", f"{efficiency['extra_token_overhead_pct']:.1f}%"])
        overhead_rows += [
            ["Avg embedding tokens/sample (retrieve+extract)", f"{efficiency['avg_embed_tokens_per_sample']:,.0f}"],
            ["Avg extra latency/sample (retrieve+extract)",    f"{efficiency['avg_extra_latency_s_per_sample']:.2f}s"],
            ["Score improvement (delta)",                      _fmt_delta(efficiency["score_improvement_delta"])],
            ["Extra tokens per +1 pct point gain",
             f"{efficiency['tokens_per_pct_point_gain']:,.0f}" if efficiency["tokens_per_pct_point_gain"] is not None else "N/A (no improvement)"],
        ]
        lines.extend(_render_table(["Metric", "Value"], overhead_rows))
        lines.append("")

    # Per-context summary
    n_improve = sum(1 for r in per_context if r["delta"] > 0)
    n_same    = sum(1 for r in per_context if r["delta"] == 0)
    n_regress = sum(1 for r in per_context if r["delta"] < 0)

    lines.append("## Per-Context Analysis")
    lines.append("")
    lines.append(
        f"Contexts: {len(per_context)} total | "
        f"{n_improve} improving | {n_same} unchanged | {n_regress} regressing"
    )
    lines.append("")

    has_efficiency = per_context and per_context[0]["avg_extra_tokens"] is not None
    ctx_headers = ["Context ID", "Category", "Sub-category", "N", "Baseline", "Memory", "Delta"]
    if has_efficiency:
        ctx_headers += ["Avg Extra Tokens", "Avg Extra Latency"]

    ctx_rows = []
    for row in per_context:
        r = [row["context_id"][:8], row["context_category"], row["sub_category"],
             str(row["n_tasks"]), _fmt_rate(row["baseline_rate"]),
             _fmt_rate(row["memory_rate"]), _fmt_delta(row["delta"])]
        if has_efficiency:
            r += [
                f"{row['avg_extra_tokens']:,.0f}" if row["avg_extra_tokens"] is not None else "N/A",
                f"{row['avg_extra_latency_s']:.2f}s" if row["avg_extra_latency_s"] is not None else "N/A",
            ]
        ctx_rows.append(r)
    lines.extend(_render_table(ctx_headers, ctx_rows))
    lines.append("")

    # Top improving contexts
    top_improve = [r for r in per_context if r["delta"] > 0][:10]
    if top_improve:
        lines.append("## Top Improving Contexts (by delta)")
        lines.append("")
        top_headers = ["Context ID", "Category", "Sub-category", "N", "Baseline", "Memory", "Delta"]
        lines.extend(_render_table(top_headers, [
            [r["context_id"][:8], r["context_category"], r["sub_category"],
             str(r["n_tasks"]), _fmt_rate(r["baseline_rate"]), _fmt_rate(r["memory_rate"]), _fmt_delta(r["delta"])]
            for r in top_improve
        ]))
        lines.append("")

    # Regressing contexts
    regressing = [r for r in per_context if r["delta"] < 0]
    if regressing:
        lines.append("## Regressing Contexts (delta < 0)")
        lines.append("")
        reg_headers = ["Context ID", "Category", "Sub-category", "N", "Baseline", "Memory", "Delta"]
        lines.extend(_render_table(reg_headers, [
            [r["context_id"][:8], r["context_category"], r["sub_category"],
             str(r["n_tasks"]), _fmt_rate(r["baseline_rate"]), _fmt_rate(r["memory_rate"]), _fmt_delta(r["delta"])]
            for r in sorted(regressing, key=lambda r: r["delta"])
        ]))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare context-memory vs. baseline inference results"
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline graded JSONL (independent inference)"
    )
    parser.add_argument(
        "--memory", required=True, help="Context-memory graded JSONL"
    )
    parser.add_argument(
        "--baseline-infer",
        default=None,
        help="Baseline (no-memory) raw inference JSONL (for efficiency comparison vs baseline)",
    )
    parser.add_argument(
        "--infer-memory",
        default=None,
        help="Context-memory raw inference JSONL (for efficiency stats)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write report to this file (also printed to stdout)",
    )

    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    baseline = load_graded(args.baseline)
    print(f"  {len(baseline)} graded samples")

    print(f"Loading memory setting: {args.memory}")
    memory = load_graded(args.memory)
    print(f"  {len(memory)} graded samples")

    baseline_infer_stats = {}
    if args.baseline_infer:
        print(f"Loading baseline inference stats: {args.baseline_infer}")
        baseline_infer_stats = load_infer_stats(args.baseline_infer)
        print(f"  {len(baseline_infer_stats)} samples with stats")

    infer_stats = {}
    if args.infer_memory:
        print(f"Loading memory inference stats: {args.infer_memory}")
        infer_stats = load_infer_stats(args.infer_memory)
        print(f"  {len(infer_stats)} samples with stats")

    print("Computing metrics...")
    overall = compute_overall(baseline, memory)
    per_category = compute_per_category(baseline, memory, infer_stats, baseline_infer_stats)
    per_context = compute_per_context(baseline, memory, infer_stats)
    efficiency = compute_efficiency(baseline, memory, infer_stats, baseline_infer_stats)

    report = render_markdown(overall, per_category, per_context, efficiency, args)

    # Always print to stdout
    print("\n" + "=" * 70)
    print(report)

    # Write to file
    output_path = args.output
    if output_path is None:
        memory_stem = os.path.splitext(args.memory)[0]
        output_path = f"{memory_stem}_comparison_report.md"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to: {output_path}")

    # Short terminal summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Samples compared: {overall['n_common']}")
    print(f"Baseline solving rate:       {_fmt_rate(overall['baseline_rate'])}")
    print(f"Context-memory solving rate: {_fmt_rate(overall['memory_rate'])}")
    print(f"Delta: {_fmt_delta(overall['delta'])}")
    if efficiency:
        bc = efficiency.get("baseline_comparison")
        if bc:
            print(f"-- Efficiency vs baseline ({bc['n']} samples) --")
            print(f"Avg tokens baseline:  {bc['avg_baseline_tokens_per_sample']:,.0f}")
            print(f"Avg tokens memory:    {bc['avg_memory_total_tokens_per_sample']:,.0f}  ({_fmt_pct(bc['token_overhead_vs_baseline_pct'])})")
            print(f"Avg latency baseline: {bc['avg_baseline_latency_per_sample']:.2f}s")
            print(f"Avg latency memory:   {bc['avg_memory_total_latency_per_sample']:.2f}s  ({_fmt_pct(bc['latency_overhead_vs_baseline_pct'])})")
        else:
            print(f"Avg extra tokens/sample:  {efficiency['avg_extra_tokens_per_sample']:,.0f}")
            print(f"Avg extra latency/sample: {efficiency['avg_extra_latency_s_per_sample']:.2f}s")


if __name__ == "__main__":
    main()
