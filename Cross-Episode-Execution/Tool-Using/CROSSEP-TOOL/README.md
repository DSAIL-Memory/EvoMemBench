# 环境配置

```bash
# 1. 创建并激活专用 Conda 环境
conda create -n CROSSEP-TOOL python=3.11 -y
conda activate CROSSEP-TOOL

# 2. 在仓库根目录以可编辑模式安装 BFCL
pip install -e .

# 3. 安装 Volcengine Ark SDK，batch evaluator 需要使用
pip install volcengine-python-sdk[ark]

# 4. 安装其余运行时依赖
pip install python-dotenv overrides tenacity filelock "tree_sitter==0.21.3" \
    "tree-sitter-java==0.21.0" "tree-sitter-javascript==0.21.4" "httpx[socks]" \
    networkx
```

# Memory 系统依赖安装（可选）

如果不评测这些 memory backend，可以跳过本节。

```bash
pip install -e ../../../EvoMemBench-Memory-Systems/mem0

# AMem（import name: agentic_memory）：使用 ChromaDB 进行向量存储
pip install -e ../../../EvoMemBench-Memory-Systems/A-mem
pip install chromadb

# MemOS（import name: memos）：使用 Qdrant 的本地文件模式
#   注意：发行包名称是 "MemoryOS" v2.0.9，而不是 "memos"。
#   这是预期行为；pip list 会显示 "MemoryOS"，但代码中应使用 `import memos`。
pip install -e ../../../EvoMemBench-Memory-Systems/MemOS
pip install qdrant-client

# MemoryOS（import name: memoryos）：同样使用 Qdrant 本地文件模式
#   发行包名称为 "memoryos-pypi" v1.0.0，不会与 MemOS 冲突。
pip install -e ../../../EvoMemBench-Memory-Systems/MemoryOS
```

安装完成后，可以检查当前生效的 mem0 源码路径：

```bash
conda run -n CROSSEP-TOOL python -c "import mem0, os; print(os.path.dirname(mem0.__file__))"
# 预期输出：.../EvoMemBench-Memory-Systems/mem0/mem0
```

## 运行评测（全部 12 个 cross-env pair）

`*_inenv_allcrossenv_fc.sh` 脚本会在一次调用中完成两阶段评测：

1. Phase 1：运行 4 个 in-env subset。
2. Phase 2：运行全部 12 个有序的 cross-environment pair。

可以先运行 no-memory 冒烟测试，快速确认环境是否配置正确：

```bash
# Smoke test：单个 backend，1 个 sample，适合做最快速的 sanity check
bash scripts/run_smoke_tests.sh --only no_memory --limit 1
```

完整运行脚本建议参考 `scripts/run_eval_mem0_inenv_allcrossenv_fc.sh`，其中默认 `top_k` 为 3。

运行结果会写入：

```text
cross_episode_results/runs/<run-name>_<memory>_<timestamp>/
```

该目录下会包含 `phase1/`（4 个 subset）、`phase2/`（12 个 pair 目录）以及合并后的 CSV 汇总文件。

## 数据路径

以下路径均以当前目录为基准。

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 主测试数据 | `bfcl_eval/data/BFCL_v4_multi_turn_ours.json` | `multi_turn_ours` 的 prompt/test entries，`run_batch_generate.py` 和 `run_batch_eval.py` 默认读取。 |
| 标准答案 | `bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_ours.json` | `run_batch_evaluate.py` 计算 success/progress 时读取。 |
| 函数文档 | `bfcl_eval/data/multi_turn_func_doc/*.json` | 各环境 API/function doc；运行时按 `involved_classes` 加载。 |
| 运行脚本 | `scripts/run_eval_*_fc.sh` | 各 backend 的完整 in-env + cross-env 启动脚本。 |
| Python 入口 | `bfcl_eval/scripts/cross_episode/run_batch_generate.py` | 生成模型输出并写入 `result.jsonl`；新 memory backend 的参数接入主要在这里。 |
| Python 入口 | `bfcl_eval/scripts/cross_episode/run_batch_evaluate.py` | 读取 `result.jsonl` 并生成 per-sample JSON 与 summary。 |
| memory 接口 | `bfcl_eval/memory/base.py` | 所有 memory backend 需要实现的 `Memory` 抽象接口。 |
| memory 注册 | `bfcl_eval/memory/factory.py` | `--memory-type` 到具体 backend class 的映射。 |
| 结果目录 | `cross_episode_results/runs/<run-name>_<memory-type>_<timestamp>/` | 每次脚本运行都会生成一个独立目录。 |
| Phase 1 memory | `cross_episode_results/runs/.../phase1/<subset>/memory/` | 每个 in-env subset 独立积累的 memory bank。 |
| Phase 2 bank copy | `cross_episode_results/runs/.../_banks/<source>__to__<target>/` | Phase 2 使用的只读 memory 副本；用于避免多个 pair 并发时共享同一存储目录。 |
| 汇总文件 | `combined_summary.txt`、`combined_summary.csv`、`combined_per_sample.csv` | 脚本最后由 `scripts/merge_summaries.py` 汇总生成。 |

## 适配新的 memory 系统

接入新的 memory system 时，通常需要完成四件事：实现 backend class、注册 memory type、补充运行脚本，并完成 smoke 验证。

### 1. 实现 backend class

新增 `bfcl_eval/memory/<new_backend>.py`，并继承 `bfcl_eval.memory.base.Memory`。至少需要实现以下方法：

```python
class NewBackendMemory(Memory):
    def utilize(self, system_prompt: str) -> str:
        ...

    def update(self, trajectory: list[dict]) -> None:
        ...

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "NewBackendMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
```

方法约定如下：

- `utilize()` 会在每个 sample 开始时调用，用于检索历史 episode，并返回一段可追加到 system prompt 的文本。
- 如果没有可用 memory，`utilize()` 应返回空字符串。
- `update()` 会在 sample 结束后收到完整的 `full_message_history`，用于把本轮轨迹写入 memory。
- 当 `readonly=True` 时，`update()` 必须直接 no-op，因为 Phase 2 会以只读方式加载 Phase 1 的 memory bank。

构造函数建议兼容以下通用参数，便于复用现有 runner 和脚本：

```python
def __init__(
    self,
    *,
    storage_dir: Path | str,
    ark_api_key: str = "",
    ark_model: str = "",
    top_k: int = 3,
    clear: bool = False,
    readonly: bool = False,
    **kwargs,
) -> None:
    ...
```

持久化文件必须写入 `storage_dir` 下。可参考以下已有实现：

- JSON/JSONL bank：参考 `bfcl_eval/memory/bm25_backend.py`
- Qdrant/local vector store：参考 `bfcl_eval/memory/mem0_backend.py`
- EvolveLab provider adapter：参考 `bfcl_eval/memory/lightweight_memory.py`

### 2. 注册 memory type

在 `bfcl_eval/memory/factory.py` 中加入映射：

```python
if memory_type == "<new_type>":
    from bfcl_eval.memory.<new_backend> import NewBackendMemory
    return NewBackendMemory(**kwargs)
```

同时更新 unknown backend 报错中的 supported values，保证参数错误时提示完整。

还需要在 `bfcl_eval/scripts/cross_episode/run_batch_generate.py` 中更新以下位置：

- 在 `parse_args()` 的 `--memory-type choices` 中加入 `<new_type>`。
- 在 memory setup 分支中，为 `<new_type>` 检查必要环境变量、设置默认参数，并把参数写入 `common_kwargs`。
- 在 `memory_config_log` 中记录会影响实验复现的参数，例如 `memory_top_k`、`embed_model`、`storage_dir`、是否 `readonly`。

当前 runner 的通用默认值如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--memory-type` | `none` | 不启用 memory。 |
| `--memory-readonly` | `False` | Phase 1 写入；Phase 2 脚本会显式打开 readonly。 |
| `--memory-top-k` | `3` | 检索数量默认值；不需要 top-k 的 backend 可忽略。 |
| `--mode` | `prompting` | Python 入口默认使用 prompting；仓库脚本必须默认传 `--mode FC`。 |

常用环境变量如下，需要写入`.env`中。

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | Ark/DeepSeek batch endpoint key，主模型调用需要。 |
| `BATCH_MODEL` | 批量推理接入点名称。 |
| `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` | mem0、reasoning_bank、amem、memos、memoryos 以及部分 embedding backend 需要。 |
| `OPENAI_API_KEY`、`OPENAI_BASE_URL` | 可用于 OpenAI-compatible embedding 或替代模型 endpoint。 |

### 3. 构建运行脚本

在 `scripts/` 下新增 `run_eval_<new_type>_inenv_allcrossenv_fc.sh`。建议优先参考最接近的现有脚本：

- 多数 memory 系统（LLM 提取 memory + 基于 embedding 的 top-k 检索）：参考 `scripts/run_eval_mem0_inenv_allcrossenv_fc.sh`
- 纯 RAG：参考 `scripts/run_eval_qwen3_embedding_inenv_allcrossenv_fc.sh`
- 不需要 top-k 的系统：参考 `scripts/run_eval_ace_inenv_allcrossenv_fc.sh`

脚本中的 `TOP_K` 默认设置为 3。

### 4. 验证流程

先做不调用 API 的参数检查：

```bash
python -m bfcl_eval.scripts.cross_episode.run_batch_generate \
    --memory-type <new_type> \
    --limit 1 \
    --dry-run
```

再运行最小 smoke：

```bash
bash scripts/run_eval_<new_type>_inenv_allcrossenv_fc.sh \
    --limit-per-subset 1 \
    --run-name smoke_<new_type>
```

最后检查以下输出是否正常：

- `cross_episode_results/runs/.../run_config.json` 中的 `memory.memory_type` 为 `<new_type>`。
- `phase1/<subset>/memory/` 下存在该 backend 的持久化文件。
- `phase2/<source>__to__<target>/result.jsonl` 存在，并包含样本输出。
- `result.jsonl` 中存在 `memory_usage`；per-sample JSON 中存在展开后的 `memory_totals`，且能反映 `n_utilize`/`n_update`。
- `combined_summary.csv` 和 `combined_per_sample.csv` 成功生成。
