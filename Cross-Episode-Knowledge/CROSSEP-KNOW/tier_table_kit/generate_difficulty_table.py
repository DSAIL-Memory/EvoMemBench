"""
Generate a difficulty-tier score TSV for any folder of *_graded.jsonl files.

Usage:
    python3 generate_difficulty_table.py <folder> [--tiers PATH] [--output PATH]

Examples:
    python3 generate_difficulty_table.py path/to/context_new_method
    python3 generate_difficulty_table.py path/to/context_new_method --output results.tsv
    python3 generate_difficulty_table.py path/to/context_new_method \\
        --tiers path/to/custom_tiers.csv

Requirements for each *_graded.jsonl file:
    Each line must be a JSON object with:
    - metadata.context_id  (str)
    - metadata.context_category  (str, one of the 4 CL-bench categories)
    - score  (int/float, binary 0 or 1)
"""

import argparse
import csv
import glob
import json
import os
import pathlib
import re
from collections import defaultdict

CATEGORIES = [
    "Domain Knowledge Reasoning",
    "Empirical Discovery & Simulation",
    "Procedural Task Execution",
    "Rule System Application",
]

TIERS_ORDER = ["Easy", "Medium", "Hard"]

# Default: difficulty_tiers.csv sitting next to this script
DEFAULT_TIERS_CSV = str(pathlib.Path(__file__).parent / "difficulty_tiers.csv")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tiers(path):
    """Return dict: context_id -> (category, difficulty)."""
    tiers = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tiers[r["context_id"]] = (r["context_category"], r["difficulty"])
    return tiers


def per_context_means(jsonl_path):
    """Return dict: context_id -> mean score across that context's samples."""
    by_ctx = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d.get("metadata", {}).get("context_id")
            score = d.get("score")
            if cid is None or score is None:
                continue
            by_ctx[cid].append(float(score))
    return {cid: sum(scores) / len(scores) for cid, scores in by_ctx.items()}


def bucket_means(ctx_means, tiers):
    """Return dict: (category, tier) -> mean of per-context scores in bucket."""
    buckets = defaultdict(list)
    for cid, mean_score in ctx_means.items():
        if cid not in tiers:
            continue
        cat, tier = tiers[cid]
        buckets[(cat, tier)].append(mean_score)
    return {key: sum(v) / len(v) for key, v in buckets.items()}


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label(folder, fname, multiple):
    """Build a short row label from folder name + filename."""
    method = os.path.basename(folder.rstrip("/"))
    if method.startswith("context_"):
        method = method[len("context_"):]

    stem = fname.replace("_graded.jsonl", "")
    m = re.search(r"_topk(\d+)_", stem)
    variant = f"topk{m.group(1)}" if m else ("nomemory" if "_ctx_nomemory_" in stem else "")
    label = f"{method}({variant})" if variant else method

    if multiple:
        ts = re.search(r"(\d{8}_\d{6})", stem)
        if ts:
            label += f"_{ts.group(1)}"
    return label


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------

def write_tsv(rows_data, out_path):
    """
    rows_data: list of (label, {(cat, tier): float}) tuples.
    Writes TSV with two-row merged header + data rows.
    Values are ×100, 1 decimal place (i.e. percentage points).
    Last 3 columns are Overall Easy/Medium/Hard = mean across 4 categories.
    """
    tsv_rows = []

    # Header row 1: category names (each spans 3 cols via trailing empty cells)
    header1 = [""]
    for cat in CATEGORIES:
        header1 += [cat, "", ""]
    header1 += ["Overall", "", ""]
    tsv_rows.append(header1)

    # Header row 2: Method + tier names repeated
    header2 = ["Method"]
    for _ in CATEGORIES:
        header2 += TIERS_ORDER
    header2 += TIERS_ORDER
    tsv_rows.append(header2)

    # Data rows
    for label, cell_means in rows_data:
        row = [label]
        cat_vals = {}
        for cat in CATEGORIES:
            for tier in TIERS_ORDER:
                v = cell_means.get((cat, tier))
                cat_vals[(cat, tier)] = v
                row.append(f"{v*100:.1f}" if v is not None else "")
        # Overall columns
        for tier in TIERS_ORDER:
            vals = [cat_vals.get((cat, tier)) for cat in CATEGORIES]
            vals = [v for v in vals if v is not None]
            overall = sum(vals) / len(vals) if vals else None
            row.append(f"{overall*100:.1f}" if overall is not None else "")
        tsv_rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(tsv_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate a difficulty-tier score TSV from a folder of graded JSONL files."
    )
    ap.add_argument("folder", help="Folder containing *_graded.jsonl files")
    ap.add_argument(
        "--tiers",
        default=DEFAULT_TIERS_CSV,
        help=f"Path to difficulty_tiers.csv (default: {DEFAULT_TIERS_CSV})",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output TSV path (default: <folder>/difficulty_score_table.tsv)",
    )
    args = ap.parse_args()

    folder = args.folder
    out_path = args.output or os.path.join(folder, "difficulty_score_table.tsv")

    tiers = load_tiers(args.tiers)
    print(f"Loaded {len(tiers)} tier assignments from {args.tiers}")

    graded_files = sorted(glob.glob(os.path.join(folder, "*_graded.jsonl")))
    if not graded_files:
        raise SystemExit(f"No *_graded.jsonl files found in: {folder}")

    print(f"Found {len(graded_files)} graded file(s)")
    multiple = len(graded_files) > 1

    rows_data = []
    for path in graded_files:
        fname = os.path.basename(path)
        label = parse_label(folder, fname, multiple)
        ctx_means = per_context_means(path)
        cell_means = bucket_means(ctx_means, tiers)
        rows_data.append((label, cell_means))
        print(f"  {fname}")
        print(f"    -> label: {label},  contexts loaded: {len(ctx_means)}")

    write_tsv(rows_data, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
