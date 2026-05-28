import json
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

USEFUL_DIR = Path("outputs/useful")
CATEGORIES = [
    "Domain Knowledge Reasoning",
    "Empirical Discovery & Simulation",
    "Procedural Task Execution",
    "Rule System Application",
]

# Sort key for deterministic row order
SETTING_ORDER = [
    ("deepseek-chat", "nomemory"),
    ("deepseek-v3", "nomemory"),
    ("deepseek-v3", "ace"),
    ("deepseek-v3", "bm25"),
    ("deepseek-v3", "graphrag"),
    ("deepseek-v3", "mem0"),
    ("deepseek-v3", "qwen3_embedding_4b"),
    ("deepseek-v3", "reasoning_bank"),
]


def setting_sort_key(row):
    model, mem = row["model"], row["memory_type"]
    try:
        return SETTING_ORDER.index((model, mem))
    except ValueError:
        return len(SETTING_ORDER)


def parse_setting(path: Path):
    dir_name = path.parent.name  # e.g. "context_ace"
    memory_type = re.sub(r"^context_", "", dir_name)  # e.g. "ace"

    stem = path.stem  # filename without .jsonl
    # model: prefer deepseek-chat, else deepseek-v3
    if "deepseek-chat" in stem:
        model = "deepseek-chat"
    elif "deepseek-v3" in stem:
        model = "deepseek-v3"
    else:
        model = stem.split("_")[0]

    topk_match = re.search(r"topk(\d+)", stem)
    topk = f"topk{topk_match.group(1)}" if topk_match else None

    if topk:
        label = f"{model} / {memory_type} ({topk})"
    else:
        label = f"{model} / {memory_type}"

    return {"label": label, "model": model, "memory_type": memory_type}


def compute_metrics(records, category=None):
    total = passed_samples = yes = no = 0
    for r in records:
        if category and r["metadata"].get("context_category") != category:
            continue
        total += 1
        passed_samples += 1 if r.get("score") == 1 else 0
        for v in r.get("requirement_status", []):
            if v == "yes":
                yes += 1
            elif v == "no":
                no += 1
    sample_rate = passed_samples / total if total else float("nan")
    rubric_rate = yes / (yes + no) if (yes + no) else float("nan")
    return sample_rate, rubric_rate


def fmt(v):
    return f"{v * 100:.2f}%" if v == v else "N/A"  # nan check


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_rows():
    files = sorted(glob(str(USEFUL_DIR / "**" / "*_graded.jsonl"), recursive=True))
    rows = []
    for fpath in files:
        path = Path(fpath)
        setting = parse_setting(path)
        records = load_records(path)
        row = {**setting, "path": str(path)}
        s, r = compute_metrics(records)
        row["Overall"] = (s, r)
        for cat in CATEGORIES:
            s, r = compute_metrics(records, cat)
            row[cat] = (s, r)
        rows.append(row)
    rows.sort(key=setting_sort_key)
    return rows


def write_tsv(rows, out_path):
    cols = ["Overall"] + CATEGORIES
    # Header row 1: category labels spanning 2 sub-columns each
    h1 = ["Setting"] + [c for c in cols for _ in (0, 1)]
    # Header row 2: sub-column labels
    h2 = [""] + ["Sample%", "Rubric%"] * len(cols)
    lines = ["\t".join(h1), "\t".join(h2)]
    for row in rows:
        cells = [row["label"]]
        for col in cols:
            s, r = row[col]
            cells += [fmt(s), fmt(r)]
        lines.append("\t".join(cells))
    out_path.write_text("\n".join(lines) + "\n")


def write_txt(rows, out_path):
    cols = ["Overall"] + CATEGORIES

    # Shorten category names for TXT header
    SHORT = {
        "Overall": "Overall",
        "Domain Knowledge Reasoning": "Domain KR",
        "Empirical Discovery & Simulation": "Empirical D&S",
        "Procedural Task Execution": "Procedural TE",
        "Rule System Application": "Rule SA",
    }

    # Collect all cell values first to compute column widths
    sub_labels = ["Sample%", "Rubric%"]
    # col_pairs: list of (header_short, [(s,r), ...]) per col group
    col_pairs = []
    for col in cols:
        values = [(fmt(row[col][0]), fmt(row[col][1])) for row in rows]
        header = SHORT[col]
        # width = max of header, sub-labels, and values
        w0 = max(len(sub_labels[0]), max(len(v[0]) for v in values))
        w1 = max(len(sub_labels[1]), max(len(v[1]) for v in values))
        header_w = max(len(header), w0 + 3 + w1)  # 3 = " | "
        col_pairs.append((col, SHORT[col], values, w0, w1, header_w))

    setting_w = max(len("Setting"), max(len(r["label"]) for r in rows))

    def pad(s, w):
        return s.ljust(w)

    sep = "-" * (setting_w + 3 + sum(p[5] + 3 for p in col_pairs))

    lines = []
    # Header row 1
    h1 = "| " + pad("Setting", setting_w) + " |"
    for _, short, _, w0, w1, hw in col_pairs:
        cell = pad(short, hw)
        h1 += " " + cell + " |"
    lines.append(h1)

    # Header row 2
    h2 = "| " + pad("", setting_w) + " |"
    for _, _, _, w0, w1, _ in col_pairs:
        h2 += " " + pad("Sample%", w0) + " | " + pad("Rubric%", w1) + " |"
    lines.append(h2)

    lines.append(sep)

    for i, row in enumerate(rows):
        line = "| " + pad(row["label"], setting_w) + " |"
        for j, (col, _, values, w0, w1, _) in enumerate(col_pairs):
            s_str, r_str = values[i]
            line += " " + pad(s_str, w0) + " | " + pad(r_str, w1) + " |"
        lines.append(line)

    out_path.write_text("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    out_tsv = USEFUL_DIR / "stats_summary.tsv"
    out_txt = USEFUL_DIR / "stats_summary.txt"
    write_tsv(rows, out_tsv)
    write_txt(rows, out_txt)
    print(f"Written: {out_tsv}")
    print(f"Written: {out_txt}")
    print(f"\nRows: {len(rows)}")
    for r in rows:
        s, rb = r["Overall"]
        print(f"  {r['label']:50s}  Sample={fmt(s)}  Rubric={fmt(rb)}")


if __name__ == "__main__":
    main()
