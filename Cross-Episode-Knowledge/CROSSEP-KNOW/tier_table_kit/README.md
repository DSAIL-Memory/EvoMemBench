# Difficulty-Tier Score Table Kit

This utility generates an Excel-ready TSV summary from EvoMemBench / CROSSEP-KNOW
graded result files. It groups results by task category and difficulty tier
(`Easy`, `Medium`, and `Hard`) so different methods can be compared at a more
interpretable level than raw per-sample scores.

```bash
python3 generate_difficulty_table.py <results_folder>
```

The script uses only the Python 3 standard library.

## What It Does

Given a folder containing one or more `*_graded.jsonl` files, the script:

1. Reads each graded JSONL file.
2. Computes the mean score for each `context_id`.
3. Looks up the precomputed difficulty tier for each context.
4. Aggregates scores by `category x tier`.
5. Writes a tab-separated table that can be pasted directly into Excel,
   Google Sheets, or other spreadsheet tools.

By default, the output is written to:

```text
<results_folder>/difficulty_score_table.tsv
```

## Files

```text
tier_table_kit/
├── generate_difficulty_table.py   # Main CLI script
├── difficulty_tiers.csv           # Precomputed tier assignments for 120 contexts
└── README.md                      # Usage guide
```

## Difficulty Tiers

`difficulty_tiers.csv` assigns each context to a difficulty tier using the
sample pass rate of the `deepseek-v3-2-251201` no-memory baseline. Tiering is
performed separately within each category, so `Easy`, `Medium`, and `Hard`
reflect relative difficulty among contexts of the same type.

| Tier | Interpretation | Assignment Rule |
| --- | --- | --- |
| Hard | The baseline rarely solves these contexts | Lowest third within the category |
| Medium | The baseline solves these contexts partially | Middle third within the category |
| Easy | The baseline usually solves these contexts well | Highest third within the category |

The bundled tier table covers 120 contexts:

| Category | Hard | Medium | Easy | Total |
| --- | ---: | ---: | ---: | ---: |
| Domain Knowledge Reasoning | 15 | 15 | 15 | 45 |
| Empirical Discovery & Simulation | 1 | 1 | 2 | 4 |
| Procedural Task Execution | 13 | 13 | 15 | 41 |
| Rule System Application | 10 | 10 | 10 | 30 |

## Input Format

Each input folder should contain one or more files matching:

```text
*_graded.jsonl
```

Each line in a graded JSONL file should be a JSON object with at least the
following fields:

```json
{
  "metadata": {
    "context_id": "<uuid>",
    "context_category": "<category name>"
  },
  "score": 0
}
```

Field requirements:

| Field | Description |
| --- | --- |
| `metadata.context_id` | Context identifier. It must appear in `difficulty_tiers.csv`. |
| `metadata.context_category` | Category name for the context. |
| `score` | Binary score, usually `0` or `1`. |

Contexts that are not present in the tier table are skipped.

## Usage

### Basic Usage

```bash
python3 generate_difficulty_table.py path/to/context_new_method
```

### Custom Output Path

```bash
python3 generate_difficulty_table.py path/to/context_new_method \
  --output my_table.tsv
```

### Custom Tier File

```bash
python3 generate_difficulty_table.py path/to/context_new_method \
  --tiers path/to/custom_difficulty_tiers.csv
```

## CLI Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `folder` | Yes | None | Folder containing `*_graded.jsonl` files. |
| `--tiers` | No | `difficulty_tiers.csv` next to the script | CSV file containing context difficulty assignments. |
| `--output` | No | `<folder>/difficulty_score_table.tsv` | Path for the generated TSV table. |

## Output Format

The generated TSV has two header rows:

```text
        Domain Knowledge Reasoning          ...     Overall
Method  Easy    Medium  Hard                ...     Easy    Medium  Hard
```

Each data row corresponds to one graded result file. Values are percentages
with one decimal place.

For each `category x tier` cell, the value is:

```text
mean sample pass rate over all contexts in that category and tier
```

The `Overall` columns are simple averages across the available category-level
values for the same difficulty tier.

When a folder contains multiple `*_graded.jsonl` files, each file becomes a
separate row. The script builds compact row labels from the folder and file
names, including variants such as `topk10` or `nomemory` when they can be
inferred.

## Example

```bash
python3 generate_difficulty_table.py results/context_mem0_topk10
```

Example console output:

```text
Loaded 120 tier assignments from difficulty_tiers.csv
Found 1 graded file(s)
  example_topk10_graded.jsonl
    -> label: mem0(topk10),  contexts loaded: 120

Saved: results/context_mem0_topk10/difficulty_score_table.tsv
```

You can open the resulting TSV directly in a spreadsheet editor or paste it
into an existing benchmark comparison table.
