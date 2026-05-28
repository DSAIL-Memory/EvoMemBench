# CROSSEP-WEB 环境与运行说明

## 1. 环境配置

```bash
conda create -n CROSSEP-WEB python=3.10
conda activate CROSSEP-WEB
cd Flash-Searcher-main
pip install -r requirements.txt
playwright install   # 下载浏览器二进制文件
```

## 2. 数据路径

所有数据文件放在 `Flash-Searcher-main/data/` 目录下。

## 3. 环境变量设置

运行前请在 `Flash-Searcher-main/` 目录下创建 `.env` 文件。该文件用于配置主模型、Ark batch endpoint、网页搜索/爬取工具，以及部分 memory backend 需要的 embedding 或 OpenAI-compatible endpoint。

```bash
# === Core Configuration (Required) ===
DEFAULT_MODEL="deepseek-v3-2-251201"

# === Volcengine Ark Batch API ===
OPENAI_API_KEY="your_ark_api_key"
OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_API_KEY="your_ark_api_key"
BATCH_MODEL="your_ark_batch_endpoint"

# === Search & Crawl Tools ===
SERPER_API_KEY="your_serper_api_key"

# 网页爬取后端：crawl4ai，免费开源，不需要额外 API key
WEB_ACCESS_PROVIDER="crawl4ai"

# === Embedding / Memory backend endpoints ===
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY="your_dashscope_api_key"
```

常用变量说明：

| 变量名 | 是否必填 | 说明 |
| --- | --- | --- |
| `DEFAULT_MODEL` | 必填 | 主推理模型名称。当前配置使用 `deepseek-v3-2-251201`；如果需要 reasoning，可按实际 endpoint 切换为对应模型。 |
| `OPENAI_API_KEY` | 必填 | 主模型调用使用的 Ark/OpenAI-compatible API key。 |
| `DEEPSEEK_API_KEY` | 必填 | ArkModel 与 batch 相关逻辑使用的 Ark API key，通常可与 `OPENAI_API_KEY` 相同。 |
| `BATCH_MODEL` | 必填 | Volcengine Ark batch inference endpoint 名称。 |
| `SERPER_API_KEY` | 必填 | 网页搜索 API key，用于 Serper 搜索服务。 |
| `WEB_ACCESS_PROVIDER` | 必填 | 网页爬取后端。推荐使用 `crawl4ai`，无需额外 API key；也可设置为 `jina`。 |

注意：不要把真实 `.env` 中的 API key 提交到仓库；README 中建议只保留占位符。

## 4. 数据路径

本任务的代码入口在 `Flash-Searcher-main/` 下，推荐所有运行命令都先进入该目录。

常用路径如下：

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 运行目录 | `Flash-Searcher-main/` | `run_pipeline.py`、评测脚本和 memory 适配代码所在目录。 |
| XBench 数据 | `Flash-Searcher-main/data/xbench/DeepSearch.csv` | 默认 `--xbench_infile`。 |
| WebWalkerQA 原始数据 | `Flash-Searcher-main/data/webwalkerqa/webwalkerqa_main.jsonl` | 从 HuggingFace 下载后重命名得到。 |
| WebWalkerQA 评测子集 | `Flash-Searcher-main/data/webwalkerqa/webwalkerqa_subset_170.jsonl` | 默认 `--webwalkerqa_infile`。 |
| 运行输出 | `Flash-Searcher-main/pipeline_output/` | 默认 `--output_dir`，每次运行会生成带时间戳的子目录。 |
| memory 存储 | `Flash-Searcher-main/storage/` | 各 provider 的默认持久化目录。 |
| 环境变量文件 | `Flash-Searcher-main/.env` | API、模型、搜索和网页访问配置。 |

`run_pipeline.py` 会在 `--output_dir` 下创建形如 `pipeline_<memory_provider>_<timestamp>/` 的目录，主要输出包括：

- `pipeline_report.txt`：XBench、WebWalkerQA 和 combined 结果汇总。
- `xbench_results.jsonl`：XBench 每题结果。
- `webwalkerqa_results.jsonl`：WebWalkerQA 每题结果。
- `xbench_runs/`、`webwalkerqa_runs/`：单 benchmark 的详细报告目录。

## 5. 新 memory 系统适配指南

统一 memory 接口在 `Flash-Searcher-main/EvolveLab/` 下。测试一个新的 memory 系统时，应接入这个接口，再通过 `run_pipeline.py --memory_provider <provider_name>` 运行跨数据集顺序评测。

### 5.1 需要修改或新增的代码

1. 在 `EvolveLab/memory_types.py` 中注册 memory 类型。

   ```python
   class MemoryType(Enum):
       YOUR_MEMORY = "your_memory"
   ```

   `MemoryType` 的 value 就是命令行中的 `--memory_provider` 参数值，必须唯一，不能和已有类型重复。

2. 在 `EvolveLab/memory_types.py` 的 `PROVIDER_MAPPING` 中注册 provider 类和模块名。

   ```python
   PROVIDER_MAPPING = {
       MemoryType.YOUR_MEMORY: ("YourMemoryProvider", "your_memory_provider"),
   }
   ```

   上例要求 provider 文件为 `EvolveLab/providers/your_memory_provider.py`，类名为 `YourMemoryProvider`。

3. 在 `EvolveLab/config.py` 的 `DEFAULT_CONFIG["providers"]` 中添加默认参数。

   ```python
   MemoryType.YOUR_MEMORY: {
       "memory_dir": "./storage/your_memory",
       "top_k": 3,
       "return_on_in": False,
       "model": None,
   },
   ```

   建议至少提供存储目录、检索数量和是否在 `MemoryStatus.IN` 阶段返回记忆等参数。需要 LLM 或 embedding 的 provider 可以参考 `mem0`、`a_mem`、`memos`、`reasoning_bank` 的配置结构。

4. 如果 provider 不直接读取 `memory_dir`，需要在 `EvolveLab/config.py` 的 `compute_storage_overrides()` 中添加映射。

   ```python
   if p == MemoryType.YOUR_MEMORY:
       return {
           "db_path": os.path.join(root, "your_memory.json"),
           "index_dir": os.path.join(root, "index"),
       }
   ```

   这个函数用于 `--memory_storage_root`，尤其在 `--both-orders` 同时运行两种顺序时隔离存储，避免两个进程写同一份 memory。

5. 在 `EvolveLab/providers/` 下新增 provider 文件，实现 `BaseMemoryProvider`。

   最小骨架如下：

   ```python
   from typing import Any, Dict, Optional

   from EvolveLab.base_memory import BaseMemoryProvider
   from EvolveLab.memory_types import (
       MemoryRequest,
       MemoryResponse,
       MemoryItem,
       MemoryItemType,
       MemoryType,
       TrajectoryData,
   )


   class YourMemoryProvider(BaseMemoryProvider):
       def __init__(self, config: Optional[Dict[str, Any]] = None):
           super().__init__(memory_type=MemoryType.YOUR_MEMORY, config=config or {})
           self.memory_dir = self.config.get("memory_dir", "./storage/your_memory")
           self.top_k = self.config.get("top_k", 3)
           self.model = self.config.get("model")

       def initialize(self) -> bool:
           # 加载已有 memory、初始化索引或创建存储目录
           return True

       def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
           memories = [
               MemoryItem(
                   id="example",
                   content="给 agent 的可执行建议或已沉淀知识。",
                   metadata={"source": "your_memory"},
                   score=1.0,
                   type=MemoryItemType.TEXT,
               )
           ]
           return MemoryResponse(
               memories=memories[: self.top_k],
               memory_type=self.memory_type,
               total_count=min(len(memories), self.top_k),
           )

       def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
           # 从 trajectory_data.query / trajectory / result / metadata 中抽取并持久化 memory
           return True, "memory ingested"
   ```

可参考已有实现：

- `EvolveLab/providers/base_provider_template.py`：最小接口模板。
- `EvolveLab/providers/mem0_provider.py`：mem0 的接口。

### 5.2 接口语义

`SearchAgent` 会在运行过程中调用 provider：

- `provide_memory(request)`：读取 memory，返回给 agent 的提示内容。
  - `request.query`：当前问题。
  - `request.context`：当前 agent 执行上下文。
  - `request.status`：`MemoryStatus.BEGIN` 或 `MemoryStatus.IN`。
  - `request.additional_params["step_number"]`：当前步骤数。
- `take_in_memory(trajectory_data)`：写入 memory。
  - `trajectory_data.query`：当前任务问题。
  - `trajectory_data.trajectory`：agent 消息轨迹。
  - `trajectory_data.result`：最终回答或结果。
  - `trajectory_data.metadata`：包含 `task_id`、`status`、`is_correct`、`full_query` 等字段。

跨数据集 pipeline 的默认逻辑是：

- Phase 1 顺序执行，`concurrency` 强制为 `1`，每题结束后调用 `take_in_memory()` 写入 memory。
- Phase 2 按 `--concurrency` 并发执行，自动关闭写入，只调用 `provide_memory()` 读取 Phase 1 积累的 memory。
- `--order xbench_webwalkerqa` 表示 XBench 写入 memory，再用 WebWalkerQA 评测。
- `--order webwalkerqa_xbench` 表示 WebWalkerQA 写入 memory，再用 XBench 评测。

## 6. 脚本构建指南

已有脚本位于 `Flash-Searcher-main/run_full_pipeline_*.sh`，推荐参考 mem0。

## 7. 测试新的 memory 系统（最小 smoke test）

先用少量任务确认 provider 能加载、能检索、能写入：

```bash
cd Flash-Searcher-main

python run_pipeline.py \
  --memory_provider your_memory \
  --xbench_sample_num 2 \
  --webwalkerqa_sample_num 2 \
  --concurrency 2 \
  --output_dir ./pipeline_output/smoke_your_memory
```
