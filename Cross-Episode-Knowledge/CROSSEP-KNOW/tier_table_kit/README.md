# 一句话总结用法

```bash
python3 generate_difficulty_table.py <results_folder>
```

# CL-bench 难度分桶评分表生成工具

给定任意一个包含 `*_graded.jsonl` 文件的结果文件夹，自动按预定义的难度分桶（Easy / Medium / Hard）汇总各 category 的平均分，生成可直接粘贴进 Excel 的 TSV 表格。

---

## 目录结构

```
tier_table_kit/
├── generate_difficulty_table.py   # 主脚本
├── difficulty_tiers.csv           # 预计算的分桶表（120 个 context）
└── README.md                      # 本文档
```

---

## 依赖

仅使用 Python 3 标准库，**无需额外安装任何包**。

---

## 分桶逻辑说明

`difficulty_tiers.csv` 基于 **deepseek-v3-2-251201 no-memory baseline** 的 sample pass rate，在每个 category 内部独立三等分：

| Tier | 含义 | 分配方式 |
|------|------|----------|
| Hard | 模型原本几乎答不出来 | 每类分数最低的 1/3 contexts |
| Medium | 模型能部分作答 | 每类分数中间的 1/3 contexts |
| Easy | 模型原本就能答好 | 每类分数最高的 1/3 contexts |

各 category 的 context 数量：

| Category | Hard | Medium | Easy | 合计 |
|----------|------|--------|------|------|
| Domain Knowledge Reasoning | 15 | 15 | 15 | 45 |
| Empirical Discovery & Simulation | 1 | 1 | 2 | 4 |
| Procedural Task Execution | 13 | 13 | 15 | 41 |
| Rule System Application | 10 | 10 | 10 | 30 |

---

## 对输入文件的要求

结果文件夹中的每个 `*_graded.jsonl`，每行需包含以下字段：

```json
{
  "metadata": {
    "context_id": "<uuid>",
    "context_category": "<category name>"
  },
  "score": 0
}
```

- `score`：二元评分，0 或 1。
- `context_id` 必须在 `difficulty_tiers.csv` 中有对应记录（即属于已评测的 120 个 contexts）。

---

## 使用方法

### 基本用法

```bash
python3 generate_difficulty_table.py <results_folder>
```

输出文件默认保存在 `<results_folder>/difficulty_score_table.tsv`。

### 示例

```bash
# 单个方法的结果文件夹
python3 generate_difficulty_table.py path/to/context_new_method

# 指定输出路径
python3 generate_difficulty_table.py path/to/context_new_method --output my_table.tsv
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `folder` | 包含 `*_graded.jsonl` 的结果文件夹（必填） | — |
| `--tiers` | `difficulty_tiers.csv` 路径 | 脚本同目录下的 `difficulty_tiers.csv` |
| `--output` | 输出 TSV 路径 | `<folder>/difficulty_score_table.tsv` |

---

## 输出格式

生成的 TSV 包含两行表头，可直接粘贴到 Excel 中（Tab 分隔，自动分列）：

```
(空)    Domain Knowledge Reasoning          ...  Overall
Method  Easy  Medium  Hard  Easy  Medium  Hard  ...  Easy  Medium  Hard
方法名   xx.x  xx.x   xx.x  ...                      xx.x   xx.x   xx.x
```

- 数值为各 (category × tier) 桶内所有 context 的**平均 sample pass rate**，以百分比表示（保留 1 位小数）。
- **Overall** 列 = 四个 category 同 tier 值的简单平均数。
- 一个文件夹中有多个 `*_graded.jsonl` 时，每个文件占一行，行标签自动附加时间戳以区分。
- 行标签自动从文件夹名（`context_` 前缀会被剥离）和文件名（提取 `topkN` 或 `nomemory`）拼合生成，例如 `mem0(topk10)`。
