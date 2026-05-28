"""
Generate a difficulty-tier score TSV for any folder of *_graded.jsonl files.

Usage:
    python3 generate_folder_difficulty_table.py <folder> [--tiers PATH] [--output PATH]

Reads every `*_graded.jsonl` file in the folder, groups samples by `context_id`
(taking the per-context mean score), then averages those per-context means
across each (category × difficulty-tier) bucket defined in `difficulty_tiers.csv`.

Output: a TSV with the same two-row header layout as
`outputs/useful/difficulty_score_table.tsv`.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

CATEGORIES = [
    "Domain Knowledge Reasoning",
    "Empirical Discovery & Simulation",
    "Procedural Task Execution",
    "Rule System Application",
]

TIERS_ORDER = ["Easy", "Medium", "Hard"]

DEFAULT_TIERS_CSV = "outputs/useful/difficulty_tiers.csv"


def load_tiers(path):
    """Return dict: context_id -> (category, difficulty)."""
    tiers = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tiers[r["context_id"]] = (r["context_category"], r["difficulty"])
    return tiers


def parse_label(folder, fname):
    """Build a row label from folder + filename."""
    method = os.path.basename(folder.rstrip("/"))
    if method.startswith("context_"):
        method = method[len("context_"):]

    stem = fname.replace("_graded.jsonl", "")
    m = re.search(r"_topk(\d+)_", stem)
    variant = f"topk{m.group(1)}" if m else ("nomemory" if "_ctx_nomemory_" in stem else "")
    ts = re.search(r"(\d{8}_\d{6})", stem)
    timestamp = ts.group(1) if ts else ""
    return method, variant, timestamp


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
    """Return dict: (category, tier) -> mean per-context score within the bucket."""
    buckets = defaultdict(list)
    for cid, mean_score in ctx_means.items():
        if cid not in tiers:
            continue
        cat, tier = tiers[cid]
        buckets[(cat, tier)].append(mean_score)
    return {key: sum(v) / len(v) for key, v in buckets.items()}


def make_row_label(method, variant, timestamp, multiple):
    parts = [method]
    if variant:
        parts.append(f"({variant})")
    label = "".join(parts)
    if multiple and timestamp:
        label += f"_{timestamp}"
    return label


def write_tsv(rows_data, out_path):
    """rows_data: list of (label, {(cat, tier): val}) tuples."""
    rows = []

    header1 = [""]
    for cat in CATEGORIES:
        header1 += [cat, "", ""]
    header1 += ["Overall", "", ""]
    rows.append(header1)

    header2 = ["Method"]
    for _ in CATEGORIES:
        header2 += TIERS_ORDER
    header2 += TIERS_ORDER
    rows.append(header2)

    for label, cell_means in rows_data:
        row = [label]
        cat_vals = {}
        for cat in CATEGORIES:
            for tier in TIERS_ORDER:
                v = cell_means.get((cat, tier))
                cat_vals[(cat, tier)] = v
                row.append(f"{v*100:.1f}" if v is not None else "")
        for tier in TIERS_ORDER:
            vals = [cat_vals[(cat, tier)] for cat in CATEGORIES
                    if cat_vals.get((cat, tier)) is not None]
            overall = sum(vals) / len(vals) if vals else None
            row.append(f"{overall*100:.1f}" if overall is not None else "")
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing *_graded.jsonl files")
    ap.add_argument("--tiers", default=DEFAULT_TIERS_CSV,
                    help=f"Tier lookup CSV (default: {DEFAULT_TIERS_CSV})")
    ap.add_argument("--output", default=None,
                    help="Output TSV path (default: <folder>/difficulty_score_table.tsv)")
    args = ap.parse_args()

    folder = args.folder
    out_path = args.output or os.path.join(folder, "difficulty_score_table.tsv")

    tiers = load_tiers(args.tiers)
    print(f"Loaded {len(tiers)} tier assignments from {args.tiers}")

    graded_files = sorted(glob.glob(os.path.join(folder, "*_graded.jsonl")))
    if not graded_files:
        raise SystemExit(f"No *_graded.jsonl files found in {folder}")

    print(f"Found {len(graded_files)} graded file(s) in {folder}")

    multiple = len(graded_files) > 1
    rows_data = []
    for path in graded_files:
        fname = os.path.basename(path)
        method, variant, ts = parse_label(folder, fname)
        label = make_row_label(method, variant, ts, multiple)

        ctx_means = per_context_means(path)
        cell_means = bucket_means(ctx_means, tiers)
        rows_data.append((label, cell_means))
        print(f"  {fname} -> {label}  ({len(ctx_means)} contexts)")

    write_tsv(rows_data, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
