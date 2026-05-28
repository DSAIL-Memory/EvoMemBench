#!/usr/bin/env python3
"""
Extract one sample per context_category and render in readable format.
Shows: task metadata, injected memory, question, model output, rubrics.
Also prints statistics on memory item counts across all samples.

Writes to: <input_stem>_memory_samples_readable.txt (same directory as input path).
"""

import json
import os
import sys
import textwrap
from collections import defaultdict

INPUT_FILE = "outputs/context_reasoning_bank/CL-bench_context_ge5_deepseek-chat_ctx_memory_topk1_20260404_175852.jsonl"
MAX_CONTEXT_CHARS = 2000  # truncate long context for readability


def wrap(text, width=100, indent="  "):
    lines = text.split("\n")
    result = []
    for line in lines:
        if len(line) <= width:
            result.append(indent + line)
        else:
            wrapped = textwrap.wrap(line, width=width - len(indent))
            result.extend(indent + w for w in wrapped)
    return "\n".join(result)


def separator(char="=", width=100):
    return char * width


def count_memory_items(memory_retrieved: str) -> int:
    """Count the number of '# Memory Item' blocks in the retrieved memory string."""
    if not memory_retrieved or not memory_retrieved.strip():
        return 0
    return memory_retrieved.count("# Memory Item")


def compute_stats(counts: list) -> dict:
    if not counts:
        return {}
    n = len(counts)
    total = sum(counts)
    mean = total / n
    sorted_c = sorted(counts)
    mid = n // 2
    median = sorted_c[mid] if n % 2 == 1 else (sorted_c[mid - 1] + sorted_c[mid]) / 2
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "min": sorted_c[0],
        "max": sorted_c[-1],
        "total": total,
    }


def print_memory_stats(all_data: list, *, out=sys.stdout):
    """Print memory item count statistics across all samples."""
    # Per-category counts
    cat_counts = defaultdict(list)
    for data in all_data:
        cat = data["metadata"]["context_category"]
        n = count_memory_items(data.get("memory_retrieved", ""))
        cat_counts[cat].append(n)

    all_counts = [n for counts in cat_counts.values() for n in counts]
    overall = compute_stats(all_counts)

    # Distribution of counts (0, 1, 2, ...)
    dist = defaultdict(int)
    for n in all_counts:
        dist[n] += 1

    print(separator("="), file=out)
    print("MEMORY ITEM COUNT STATISTICS (all samples)", file=out)
    print(separator("="), file=out)
    print(f"  Total samples : {overall['n']}", file=out)
    print(f"  Mean          : {overall['mean']:.2f}", file=out)
    print(f"  Median        : {overall['median']:.1f}", file=out)
    print(f"  Min / Max     : {overall['min']} / {overall['max']}", file=out)
    non_zero = [n for n in all_counts if n > 0]
    if non_zero:
        nz = compute_stats(non_zero)
        print(
            f"  Mean (non-zero): {nz['mean']:.2f}  ({len(non_zero)} samples with memory)",
            file=out,
        )
    zero_count = dist[0]
    print(
        f"  Zero-memory   : {zero_count} samples  ({100*zero_count/overall['n']:.1f}%)",
        file=out,
    )

    print(f"\n  Distribution (# memory items → # samples):", file=out)
    for k in sorted(dist):
        bar = "#" * min(dist[k], 60)
        print(f"    {k:2d}  {dist[k]:5d}  {bar}", file=out)

    ordered_cats = [
        "Domain Knowledge Reasoning",
        "Rule System Application",
        "Procedural Task Execution",
        "Empirical Discovery & Simulation",
    ]
    for cat in cat_counts:
        if cat not in ordered_cats:
            ordered_cats.append(cat)

    print(f"\n  Per-category breakdown:", file=out)
    for cat in ordered_cats:
        if cat not in cat_counts:
            continue
        s = compute_stats(cat_counts[cat])
        nz_cat = [n for n in cat_counts[cat] if n > 0]
        print(
            f"    {cat}", file=out
        )
        print(
            f"      n={s['n']}  mean={s['mean']:.2f}  median={s['median']:.1f}"
            f"  min={s['min']}  max={s['max']}"
            f"  non-zero={len(nz_cat)} ({100*len(nz_cat)/s['n']:.1f}%)",
            file=out,
        )
    print(file=out)


def print_sample(data, category_idx, total, *, out=sys.stdout):
    meta = data["metadata"]
    messages = data["messages"]
    memory_retrieved = data.get("memory_retrieved", "")
    model_output = data.get("model_output", "")
    rubrics = data.get("rubrics", [])

    # The user message contains context + question; split at the question (last turn)
    user_content = messages[1]["content"] if len(messages) > 1 else ""
    system_content = messages[0]["content"] if messages else ""

    # Heuristic: the question is likely the last paragraph / short sentence
    # Try to split context from question by finding the last newline block
    lines = user_content.rstrip().split("\n")
    # Walk back to find the question (non-empty lines at the end)
    question_lines = []
    context_lines = lines[:]
    while context_lines and context_lines[-1].strip():
        question_lines.insert(0, context_lines.pop())
    # Remove trailing blank lines from context
    while context_lines and not context_lines[-1].strip():
        context_lines.pop()

    question = "\n".join(question_lines)
    context_text = "\n".join(context_lines)

    print(separator("="), file=out)
    print(
        f"SAMPLE {category_idx}/{total}: {meta['context_category']} / {meta['sub_category']}",
        file=out,
    )
    print(f"task_id:    {meta['task_id']}", file=out)
    print(f"context_id: {meta['context_id']}", file=out)
    print(separator("-"), file=out)

    # System prompt
    print("\n[SYSTEM PROMPT]", file=out)
    print(wrap(system_content), file=out)

    # Context (truncated)
    print(
        f"\n[CONTEXT]  ({len(context_text)} chars total, showing first {MAX_CONTEXT_CHARS})",
        file=out,
    )
    display_ctx = context_text[:MAX_CONTEXT_CHARS]
    if len(context_text) > MAX_CONTEXT_CHARS:
        display_ctx += f"\n... [truncated {len(context_text) - MAX_CONTEXT_CHARS} chars] ..."
    print(wrap(display_ctx), file=out)

    # Question
    print("\n[QUESTION / TASK]", file=out)
    print(wrap(question), file=out)

    # Memory injected
    n_mem = count_memory_items(memory_retrieved)
    if memory_retrieved and memory_retrieved.strip():
        print(f"\n[MEMORY INJECTED INTO PROMPT]  ({n_mem} items)", file=out)
        print(wrap(memory_retrieved.strip()), file=out)
    else:
        print(
            "\n[MEMORY INJECTED INTO PROMPT]  <empty — no memory retrieved>",
            file=out,
        )

    # Model output
    print("\n[MODEL OUTPUT]", file=out)
    if model_output:
        print(wrap(str(model_output)), file=out)
    else:
        print("  <empty>", file=out)

    # Rubrics
    print(f"\n[RUBRICS]  ({len(rubrics)} items)", file=out)
    for i, r in enumerate(rubrics, 1):
        print(f"  {i:2d}. {r}", file=out)

    print(file=out)


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_path = os.path.splitext(input_file)[0] + "_memory_samples_readable.txt"

    all_data = []
    categories = {}
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
            cat = data["metadata"]["context_category"]
            mr = data.get("memory_retrieved", "")
            if cat not in categories:
                categories[cat] = data
            elif not categories[cat].get("memory_retrieved", "").strip() and mr.strip():
                # Upgrade to a sample that actually has memory
                categories[cat] = data

    ordered_cats = [
        "Domain Knowledge Reasoning",
        "Rule System Application",
        "Procedural Task Execution",
        "Empirical Discovery & Simulation",
    ]
    # Add any unexpected categories at the end
    for cat in categories:
        if cat not in ordered_cats:
            ordered_cats.append(cat)

    samples = [(cat, categories[cat]) for cat in ordered_cats if cat in categories]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        print(f"Extracted {len(samples)} samples (one per context_category)", file=out)
        print(f"Source: {input_file}\n", file=out)

        print_memory_stats(all_data, out=out)

        for idx, (cat, data) in enumerate(samples, 1):
            print_sample(data, idx, len(samples), out=out)

    print(f"Wrote: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
