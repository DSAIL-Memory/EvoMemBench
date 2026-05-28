# Agent Environments - AlfWorld（CROSSEP-EMB）

本目录是 CROSSEP-EMB 的 ALFWorld 跨 Episode 记忆评测工程。整体流程如下：

1. 启动 ALFWorld HTTP 环境服务。
2. 使用 `scripts/eval_alfworld_all_*_batch_all_pairs.sh` 运行 in-env 与 cross-env 的 all-pairs 实验。

## 环境配置

### 1. 配置并启动 ALFWorld 服务端环境

先创建用于部署 ALFWorld 服务端的 Conda 环境，并启动环境服务。如果安装过程中遇到问题，请参考 `agentenv-alfworld/README.md` 中的 `Known issues & fixes`。

```sh
conda create --name CROSSEP-EMB-server python=3.9
conda activate CROSSEP-EMB-server
cd agentenv-alfworld
bash ./setup.sh
alfworld --host 0.0.0.0 --port 36005
```

可通过下面的命令检查服务是否正常启动：

```sh
curl http://localhost:36005/
# 如果返回 "This is environment AlfWorld."，说明服务已正常运行
```

### 2. 配置评测运行环境

再创建用于运行评测脚本的 Conda 环境：

```sh
conda create --name CROSSEP-EMB python=3.10
conda activate CROSSEP-EMB
pip install tiktoken
pip install 'volcengine-python-sdk[ark]'   # 用于 Ark batch-mode agent 以及 memory LLM 调用
```

### 记忆系统依赖（可选）

如果需要评测以下 memory baseline，请将对应的 vendored memory system 以可编辑模式安装。下面的路径均相对于当前 `CROSSEP-EMB/` 目录。

```sh
# mem0（ChromaDB vector store + SQLite）
pip install -e ../../../EvoMemBench-Memory-Systems/mem0
pip install chromadb

# A-mem（Zettelkasten-style agentic memory）
pip install -e ../../../EvoMemBench-Memory-Systems/A-mem

# MemoryOS（三层层级式记忆）
pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS/memoryos-pypi

# MemOS（基于 Qdrant 的通用文本记忆）
pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
```

## 环境变量

`scripts/eval_alfworld_with_memory.py` 会从仓库根目录加载 `.env`。在运行评测前，请先创建 `CROSSEP-EMB/.env`：

```sh
# 未设置 --model 时，Agent LLM 使用的默认路径
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# --api_mode batch 以及许多 memory adapter 使用的 Ark batch API 模型
BATCH_MODEL=...

# 设置 --model 时使用的 OpenAI-compatible 路径
OPENAI_API_KEY=...
OPENAI_BASE_URL=...

# DashScope/text-embedding-v4 配置使用的 embedding endpoint
DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 评测入口

可参考 `scripts/eval_alfworld_all_mem0_batch_all_pairs.sh` 了解完整运行方式；实际代码入口为 `scripts/eval_alfworld_with_memory.py`。

## 接入新的记忆系统

如果需要评测新的 memory system，需要先实现 adapter，再注册 adapter，最后补充 smoke/full 运行脚本。

### 1. 添加 memory adapter

新建 `scripts/memory/<your_memory>_adapter.py`，并继承 `BaseMemory`：

```python
from .base import BaseMemory, MemoryCallStats


class YourMemory(BaseMemory):
    def __init__(self, read_only: bool = False, top_k: int = 3, store_path: str = "./storage/your_memory.json", **kwargs):
        super().__init__(memory_type="your_memory", read_only=read_only)
        self._top_k = top_k
        self._store_path = store_path

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()
        query = conversation[2]["content"]
        # 根据 query 检索记忆，并将检索结果追加到 conversation[0]["content"]。
        return conversation, MemoryCallStats()

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()
        # 存储轨迹或提取后的记忆。若系统只学习成功轨迹，可使用 reward >= 1 作为判断条件。
        return MemoryCallStats()

    def load_from_disk(self, **kwargs) -> None:
        # 可选：供 --memory_load_args 使用的重新加载钩子。
        return None
```

Adapter 需要满足以下约定：

- `inject()` 会在环境 reset 之后、第一次调用 agent LLM 之前执行一次。
- `conversation[2]["content"]` 是 ALFWorld 的第一条 observation，默认应将其作为检索 query。
- 将检索到的内容注入 `conversation[0]["content"]`，通常建议放在清晰的标题下，例如 `--- Retrieved Memories ---`。
- `inject()` 需要返回修改后的 conversation，以及一个 `MemoryCallStats` 实例。
- `update()` 会在 episode 结束后，以完整轨迹为输入执行一次。
- 当 `read_only=True` 时，`update()` 必须为空操作，因为 cross-env 脚本依赖这一行为。
- 如果 memory system 只应从成功轨迹中学习，请在 `reward is not None and reward < 1` 时跳过更新。
- 所有持久化数据都应存放到配置中指定的路径下，最好放在脚本创建的 `$MEMORY_DIR` 内。

### 2. 注册 adapter

编辑 `scripts/memory/base.py`：

```python
if memory_type == "your_memory":
    from .your_memory_adapter import YourMemory
    return YourMemory(read_only=read_only, **kwargs)
```

同时更新未知类型的报错信息，确保新的 `memory_type` 出现在支持列表中。

### 3. 定义配置参数

通过 `--memory_config` 传入 JSON，暴露运行时可调整的参数。不要在 adapter 中硬编码实验输出路径或 API key。

推荐的最小配置结构如下：

```json
{
  "agent_id": "pick_and_place_simple",
  "top_k": 3,
  "store_path": "/abs/path/to/output/memory/your_memory_pick_and_place_simple.json",
  "ark_batch_config": {
    "api_key": "${DEEPSEEK_API_KEY}",
    "model": "${BATCH_MODEL}"
  },
  "embed": {
    "model": "text-embedding-v4",
    "api_key": "${DASHSCOPE_API_KEY}",
    "base_url": "${DASHSCOPE_BASE_URL}"
  }
}
```

建议在 adapter 构造函数中设置保守默认值：

- 对于纯 RAG 方法，建议设置 `top_k=10`；存储 trajectory chunk 时，建议设置 `chunk_size=1024`。
- 对于涉及 LLM 抽取与精炼的 memory，建议设置 `top_k=3`。

## 运行已有脚本

端到端 smoke test 单个 memory backend：

```sh
bash scripts/smoke_test_eval_alfworld_mem0_all_pairs.sh
```

运行完整 all-pairs 实验：

```sh
bash scripts/eval_alfworld_all_mem0_batch_all_pairs.sh
```

all-pairs memory 脚本会生成如下目录结构：

```text
output/<run_name>/
  configs/        # 每个任务自动生成的 memory config
  memory/         # 持久化的 memory store
  logs/           # 每个 job 的 stdout 日志
  in_env/         # 六个 task-specific in-env 运行结果
  cross_env/      # source-to-target transfer 运行结果
  final_report.txt
```
