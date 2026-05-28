# CROSSEP-KNOW

`CROSSEP-KNOW` 用于在 context-grouped CL-bench 数据上评测跨 episode memory 系统。主流程按 `context_id` 隔离 memory：同一 context 内样本串行执行并持续累积 memory，不同 context 并行执行，避免跨 context 污染。

主流程包含三步：

1. `infer_context_memory.py`：带 memory 或 no-memory 的上下文分组推理。
2. `eval.py`：使用 LLM-as-judge 对模型输出打 0/1 分。
3. `analyze_comparison_new.py`：可选，对比 no-memory baseline 和 memory run 的效果与开销。

## 环境安装

建议从项目根目录执行：

```bash
conda create -n CROSSEP-KNOW python=3.12
conda activate CROSSEP-KNOW
pip install -r requirements.txt
pip install volcengine-python-sdk
```

### 可选外部 Memory Backend

部分外部 memory 系统以 editable 方式从 `../../EvoMemBench-Memory-Systems/` 安装：

```bash
pip install -e ../../EvoMemBench-Memory-Systems/mem0       # provides `mem0` (dist name: mem0ai)
pip install -e ../../EvoMemBench-Memory-Systems/A-mem      # provides `agentic_memory`
pip install -e ../../EvoMemBench-Memory-Systems/MemOS      # provides `memos`
pip install -e ../../EvoMemBench-Memory-Systems/MemoryOS   # provides `memoryos` (dist name: memoryos-pypi)
```

验证：

```bash
python -c "from mem0 import Memory; \
           from agentic_memory.memory_system import AgenticMemorySystem; \
           from memos.configs.memory import MemoryConfigFactory; \
           from memoryos import Memoryos; print('OK')"
```

## 数据路径与产物路径

关键文件和目录：

```text
CL-bench_context_ge5.jsonl          # 默认评测数据
run_context_memory.sh               # 三阶段主流程入口
infer_context_memory.py             # context-grouped memory inference
eval.py                             # LLM-as-judge 评分
analyze_comparison_new.py           # baseline vs memory 对比分析
cl_bench_memory/                    # memory backend 实现
scripts/                            # 已有 compare/smoke 脚本
tier_table_kit/                     # easy/medium/hard 难度分桶评分表生成工具
requirements.txt                    # Python 依赖
```

默认数据：

- 数据文件：`CL-bench_context_ge5.jsonl`
- `run_context_memory.sh` 默认输入名：`CL-bench_context_ge5`
- `run_context_memory.sh --input NAME` 会读取 `${NAME}.jsonl`

运行产物：

```text
outputs/context_<memory_type>/      # memory run 的原始推理和评分结果
outputs/context_nomemory/           # --no-memory run 的原始推理和评分结果
outputs/_smoke*/                    # smoke test 日志和报告
memories/context_memory/            # 每次运行、每个 context 的 memory 持久化目录
```

memory run 的默认原始推理输出：

```text
outputs/context_<memory_type>/<input>_<model>_ctx_memory_topk<k>_<timestamp-or-maxN>.jsonl
```

no-memory run 的默认原始推理输出：

```text
outputs/context_nomemory/<input>_<model>_ctx_nomemory_<timestamp-or-maxN>.jsonl
```

评分输出：

```text
<raw_inference_stem>_graded.jsonl
```

对比报告：

```text
<raw_inference_stem>_comparison_report.md
```

difficulty tier 评分表：

```text
outputs/context_<memory_type>/difficulty_score_table.tsv
```

memory 持久化目录按 run 和 context 隔离：

```text
memories/context_memory/<input>_<model>_<memory_type>_topk<k>_<timestamp-or-maxN>/<context_id>/
```

## 环境变量

环境变量应写在 `.env` 中。请只提交变量名和占位符，不要提交真实 API key。

```bash
# === DashScope embedding：默认 embedding provider ===
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=<your_dashscope_api_key>

# === Volcengine Ark Batch API：deepseek 推理模型默认使用 ===
DEEPSEEK_API_KEY=<your_ark_api_key>
BATCH_MODEL=<your_ark_batch_endpoint>

# === OpenAI-compatible API：默认 judge model=gpt-5.1 使用 ===
OPENAI_API_KEY=<your_openai_compatible_api_key>
OPENAI_BASE_URL=<your_openai_compatible_base_url>

```

规则：

- 推理模型 `--model` 以 `deepseek` 开头时，读取 `DEEPSEEK_API_KEY`；如果同时配置了 `BATCH_MODEL`，`infer_context_memory.py` 会优先使用 Volcengine Ark batch endpoint，并将实际模型替换为 `BATCH_MODEL`。
- 如果没有配置 `BATCH_MODEL`，deepseek 推理会使用 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL` 走 OpenAI-compatible 直连。
- 评分模型 `--judge-model` 默认是 `gpt-5.1`，不以 `deepseek` 开头，因此读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。
- 默认 embedding provider 是 `dashscope`，读取 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。

## 运行流程

推荐统一使用主脚本：

```bash
bash run_context_memory.sh [options]
```

主脚本会依次执行：

```bash
python infer_context_memory.py ...
python eval.py ...
python analyze_comparison_new.py ...   # 如果没有设置 --skip-compare
```

no-memory baseline：

```bash
bash run_context_memory.sh \
    --no-memory \
    --workers 50 \
    --skip-compare
```

小样本调试：

```bash
bash run_context_memory.sh \
    --memory-type reasoning_bank \
    --max-samples 2 \
    --workers 1 \
    --eval-workers 1 \
    --top-k 1 \
    --skip-compare
```

正式 memory run 示例：

```bash
bash run_context_memory.sh \
    --memory-type mem0 \
    --top-k 10 \
    --workers 5
```

指定 baseline 做对比：

```bash
bash run_context_memory.sh \
    --memory-type reasoning_bank \
    --top-k 10 \
    --baseline outputs/context_nomemory/CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl
```

已有脚本可直接作为各 backend 的参考运行入口，推荐参考`scripts/mem0_compare.sh`。

## Easy / Medium / Hard 分桶评分

`eval.py` 生成的 `*_graded.jsonl` 只包含逐样本 0/1 分数。如果需要分别输出 Easy、Medium、Hard 三个难度档位的分数，需要在评分完成后使用 `tier_table_kit`。

基本用法：

```bash
python3 tier_table_kit/generate_difficulty_table.py <results_folder>
```

例如，对 mem0 的评分结果生成难度分桶表：

```bash
python3 tier_table_kit/generate_difficulty_table.py outputs/context_mem0
```

脚本会读取目标目录下的所有 `*_graded.jsonl` 文件，并输出：

```text
outputs/context_mem0/difficulty_score_table.tsv
```

该 TSV 可直接粘贴到 Excel 中，列包括四个 CL-bench category 以及 Overall 下的 `Easy`、`Medium`、`Hard` 分数。分桶依据来自 `tier_table_kit/difficulty_tiers.csv`，更多细节见 `tier_table_kit/README.md`。

## 参数与默认值

### `run_context_memory.sh`

日常实验优先使用这个入口。注意：脚本顶部变量才是当前真实默认值，`--help` 中的部分注释可能滞后。

| 参数 | 当前默认值 | 含义 |
| --- | --- | --- |
| `--input` | `CL-bench_context_ge5` | 数据集 basename，不带 `.jsonl` |
| `--infer-model` | `deepseek-v3-2-251201` | 被测推理模型 |
| `--judge-model` | `gpt-5.1` | 评分模型 |
| `--workers` | `50` | 推理阶段并行 context 数 |
| `--eval-workers` | `50` | 评分阶段并行 worker 数 |
| `--top-k` | `10` | memory retrieval top-k |
| `--memory-type` | `mem0` | memory backend 类型 |
| `--max-samples` | 空 | 限制总样本数，调试用 |
| `--baseline` | `outputs/context_nomemory/CL-bench_context_ge5_deepseek-v3-2-251201_ctx_nomemory_graded.jsonl` | 默认对比 baseline |
| `--skip-compare` | `false` | 跳过对比分析 |
| `--no-memory` | `false` | 禁用 retrieve、inject、extract |


## 新 Memory 系统适配指南

所有 backend 都通过 `cl_bench_memory/base.py` 中的 `Memory` 抽象接口被主流程调用。

### 1. 新增 Backend 类

新增文件。建议文件名沿用已有 backend 风格：

```text
cl_bench_memory/new_backend_memory.py
```

实现 `Memory` 子类。构造函数必须兼容主流程传入的标准参数，不使用的参数也要接收或通过 `**kwargs` 忽略：

```python
import os
import time

from .base import Memory


class NewBackendMemory(Memory):
    def __init__(
        self,
        client=None,
        model=None,
        memory_dir=None,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=10,
        extract_model=None,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.extract_model = extract_model or model
        self.memory_dir = memory_dir
        self.embed_client = embed_client or client
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.memory_bank = []

        os.makedirs(memory_dir, exist_ok=True)

    def retrieve(self, query: str) -> tuple:
        t0 = time.time()
        memory_text = ""
        return memory_text, {
            "latency_s": time.time() - t0,
            "embed_usage": {},
        }

    def extract(self, content: str, **kwargs) -> dict:
        t0 = time.time()
        return {
            "latency_s": time.time() - t0,
            "llm_usage": {},
            "embed_usage": {},
        }
```

接口要求：

- `retrieve(query)` 返回 `(memory_text, stats)`。
- `extract(content, **kwargs)` 负责写入/更新 memory，并返回 `stats`。
- `retrieve` 的 `stats` 至少包含 `latency_s` 和 `embed_usage`。
- `extract` 的 `stats` 至少包含 `latency_s`、`llm_usage` 和 `embed_usage`。
- `memory_text` 会由 `inject_memory_into_messages` 注入到 system message。
- `self.memory_bank` 应反映当前已存储条目，因为主流程用 `len(memory.memory_bank)` 汇总最终 memory 数量。
- 持久化文件只能写入传入的 `memory_dir`。

`infer_context_memory.py` 调用 `extract` 时会传入：

```text
task_id
query
context_category
sub_category
```

`content` 是由 `format_trajectory(messages, response_text)` 生成的完整 trajectory。

### 2. 保持 Context 隔离

主流程按 `metadata.context_id` 分组：

- 同一 context 内样本串行执行，memory 持续累积。
- 不同 context 并行执行。
- 每个 context 都会实例化独立 backend，并传入独立目录：

```text
<run_memory_dir>/<context_id>/
```

backend 不应使用跨 context 的全局可变状态，也不应从其他 context 目录读取 memory。

### 3. 注册 Backend

修改 `cl_bench_memory/registry.py`：

```python
from .new_backend_memory import NewBackendMemory


def build_memory(memory_type: str, **kwargs):
    if memory_type == "new_memory":
        return NewBackendMemory(**kwargs)
```

同时把 `new_memory` 加入未知类型报错中的 supported list。

如果 backend 有可选外部依赖，参考 `a_mem`、`memos`、`memoryos` 的写法：

```python
try:
    from .new_backend_memory import NewBackendMemory
    _NEW_MEMORY_AVAILABLE = True
except ImportError:
    _NEW_MEMORY_AVAILABLE = False
```

当用户请求该 backend 但依赖缺失时，抛出包含安装命令的清晰错误。

### 4. 导出 Backend

修改 `cl_bench_memory/__init__.py`：

```python
from .new_backend_memory import NewBackendMemory

__all__ = [
    ...
    "NewBackendMemory",
]
```

如果是可选依赖 backend，使用和 `AMemMemory`、`MemOSMemory`、`MemoryOSMemory` 一样的 guarded import。

### 5. 构建运行脚本

新增正式运行脚本：

```text
scripts/new_memory_compare.sh
```

模板：可参考`mem0_compare.sh`

### 6. 少样本测试可运行

```bash
bash run_context_memory.sh \
    --memory-type new_memory \
    --max-samples 2 \
    --workers 1 \
    --eval-workers 1 \
    --top-k 1 \
    --skip-compare
```
