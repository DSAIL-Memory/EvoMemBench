# BFCL In-Episode Execution 评测指南

本文档介绍 `INEP-EXEC` 任务的环境配置、数据路径、环境变量，以及新增 memory 系统时需要改动的位置。除特殊说明外，以下命令都建议在当前目录执行：

```text
EvoMemBench-simplified/In-Episode-Execution/INEP-EXEC
```

## 环境配置

```bash
# 1. 创建并激活专用 Conda 环境
conda create -n INEP-EXEC python=3.11 -y
conda activate INEP-EXEC

# 2. 在当前项目根目录以可编辑模式安装 BFCL
pip install -e .

# 3. 安装 Volcengine Ark SDK，批量评测入口需要使用
pip install volcengine-python-sdk[ark]
```

## Memory 系统依赖安装（可选）

如果只运行 no-memory baseline，可以跳过本节。若要评测下列 memory backend，需要从 `../../EvoMemBench-Memory-Systems/` 安装对应依赖：

```bash
pip install -e ../../EvoMemBench-Memory-Systems/mem0       # 提供 `mem0`，发行包名为 mem0ai
pip install -e ../../EvoMemBench-Memory-Systems/A-mem      # 提供 `agentic_memory`
pip install -e ../../EvoMemBench-Memory-Systems/MemOS      # 提供 `memos`
pip install -e ../../EvoMemBench-Memory-Systems/MemoryOS   # 提供 `memoryos`，发行包名为 memoryos-pypi
```

安装完成后，可以用下面的命令检查主要 import 是否可用：

```bash
python -c "from mem0 import Memory; \
           from agentic_memory.memory_system import AgenticMemorySystem; \
           from memos.configs.memory import MemoryConfigFactory; \
           from memoryos import Memoryos; print('OK')"
```

## 数据路径

In-episode execution 数据位于：

```text
bfcl_eval/data
```

核心文件如下：

| 路径 | 说明 |
| --- | --- |
| `bfcl_eval/data/BFCL_v4_multi_turn_ours.json` | 主测试集。 |
| `bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_ours.json` | 标准答案。 |
| `bfcl_eval/data/multi_turn_func_doc/*.json` | 各类工具/API 的 function doc。 |
| `bfcl_eval/scripts/in_episode/ids/*.json` | 4 个子集的样本 id 列表；脚本会按 `gorilla_fs`、`vehicle_control`、`trading_bot`、`travel_api` 分组并行运行。 |

## 环境变量

常用环境变量需要写入 `.env` 文件。

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | Ark/DeepSeek batch endpoint key，主模型调用需要。 |
| `BATCH_MODEL` | 批量推理接入点名称。 |
| `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` | mem0、reasoning_bank、amem、memos、memoryos 以及部分 embedding backend 需要。 |
| `OPENAI_API_KEY`、`OPENAI_BASE_URL` | 可用于 OpenAI-compatible embedding 或替代模型 endpoint。 |

## 运行脚本

现有脚本位于：

```text
scripts
```

新增或修改运行脚本时，建议优先参考：

- `scripts/run_eval_mem0_inepisode_fc_8k_lc.sh`
- `scripts/run_eval_mem0_inepisode_fc_16k_lc.sh`
- `scripts/run_eval_mem0_inepisode_fc_32k_lc.sh`
- `scripts/run_eval_mem0_inepisode_fc_64k_lc.sh`
- `scripts/run_eval_mem0_inepisode_fc_128k_lc.sh`

这些脚本可作为 per-sample、Qdrant/DashScope 类 backend 的模板。

新脚本建议按以下格式命名：

```text
scripts/run_eval_<memory_type>_inepisode_fc_<budget>_lc.sh
```

其中 `<memory_type>` 对应 `--memory-type`，`<budget>` 对应上下文预算，例如 `8k`、`16k`、`32k`、`64k` 或 `128k`。

## 新 Memory 系统适配指南

新增一个 memory 系统通常需要适配 4 个位置：

1. 在 `bfcl_eval/memory/` 下实现 backend。
2. 在 `bfcl_eval/memory/factory.py` 中注册 `memory_type`。
3. 在 `bfcl_eval/scripts/in_episode/run_batch_eval.py` 中补充参数 choices 和构造参数分支。
4. 在 `scripts/` 下新增运行脚本，复用现有 subset 并行和 summary merge 逻辑。

### 1. 实现 Memory backend

常规 memory backend 需要继承 `bfcl_eval.memory.base.Memory`，并至少实现 `utilize()`、`update()` 和 `load_from_disk()`。

```python
from pathlib import Path

from bfcl_eval.memory.base import Memory


class NewMemory(Memory):
    def __init__(
        self,
        storage_dir: Path | str,
        top_k: int = 3,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ):
        super().__init__(memory_type="new_memory", clear=clear, readonly=readonly)
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        # clear=True 时，删除或重置本 backend 的持久化文件。
        # readonly=True 时，update() 必须 no-op。

    def utilize(self, user_text: str) -> str:
        # 根据当前 user_text 检索 memory，并返回要追加到 user message 的文本。
        # 如果没有可用 memory，返回空字符串。
        return ""

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        # trajectory 是当前 turn 的消息增量，形如：
        # [{"role": ..., "content": ...}, ...]

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "NewMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
```

常规 backend 的生命周期如下：

- sample 开始：`begin_sample()` 重置 memory 计费指标。
- 每个 turn 调模型前：调用 `utilize(user_text)`；如果返回非空文本，会追加到最后一条 user message。
- 每个 turn 结束：调用 `update(turn_delta)`，写入当前 turn 的 user、assistant、tool 轨迹。
- sample 结束：调用 `drain_usage()`；结果会写入 per-sample JSON 的 `memory_usage` / `memory_totals`。

如果 backend 需要统计 LLM、embedding 或延迟开销，请使用 `bfcl_eval.memory.metrics` 中的计数工具。汇总逻辑会读取以下字段：

```text
latency_s, input_tokens, output_tokens, embedding_tokens,
n_llm_calls, n_embed_calls, n_utilize, n_update
```

### 2. 注册 factory

在 `bfcl_eval/memory/factory.py` 的 `build_memory()` 中加入新分支：

```python
if memory_type == "new_memory":
    from bfcl_eval.memory.new_memory_backend import NewMemory
    return NewMemory(**kwargs)
```

同时更新未知类型报错中的 supported values，确保参数错误时能看到完整的可用列表。

### 3. 适配 `run_batch_eval.py` 参数

在 `bfcl_eval/scripts/in_episode/run_batch_eval.py` 中完成两处改动：

1. 将新类型加入 `--memory-type` 的 `choices`。
2. 在 memory setup 中按依赖类型配置 `memory_build_kwargs`。

常见分类如下：

```python
# 需要 DashScope embedding + Ark LLM
_EMBEDDING_MEMORY_TYPES = {
    "mem0", "memos", "memoryos", "amem",
    "agent_kb", "agent_workflow",
    "new_memory",
}

# 只需要 Ark LLM，不需要 embedding
_LLM_MEMORY_TYPES = {"skillweaver", "lightweight", "new_memory"}
```

如果新 backend 只使用本地检索，不需要额外 API key，可以走默认分支，让入口只传入：

```text
storage_dir, clear, readonly
```

如果新 backend 需要指定检索数量，请确保入口传入：

```python
memory_build_kwargs = {
    "top_k": args.memory_top_k,
}
```

如果新 backend 依赖独立服务，例如 MemoBrain，请新增专用分支，并从环境变量读取 URL/model 等配置。不要复用含义无关的变量。

### 4. 注意 compression 类 backend

`memagent` 和 `memobrain` 属于压缩/工作记忆路径，不走常规的每 turn `utilize()` / `update()` 注入逻辑。它们在 `bfcl_eval/model_handler/base_handler.py` 中通过 compression 分支维护 `_compressed_messages`。

如果新系统是“检索后注入 user message”的普通 memory，只需要按上面的 `Memory` 接口适配即可。如果新系统需要替换上下文压缩逻辑，除了新增 backend 和 factory，还需要在 `base_handler.py` 的 compression memory 类型集合及压缩分支中做专门适配。
