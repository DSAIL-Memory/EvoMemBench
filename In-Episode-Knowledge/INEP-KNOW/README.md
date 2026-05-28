# MemoryAgentBench In-Episode Knowledge 评测指南

本文档介绍 `INEP-KNOW` 任务的环境配置、数据路径、运行方式、默认参数，以及新增 memory 系统时需要改动的位置。除特殊说明外，以下命令都建议在 `MemoryAgentBench/` 目录下执行。

## 1. 环境配置

```bash
conda create -n INEP-KNOW python=3.12
conda activate INEP-KNOW
pip install -r requirements.txt
pip install "numpy<2"
```

## 2. 数据路径

评测代码位于 `MemoryAgentBench/`。进入该目录后再运行评测命令：

```bash
cd MemoryAgentBench
```

### 2.1 数据来源

- 默认数据源：Hugging Face 数据集 `ai-hyz/MemoryAgentBench`。
- 默认支持的 split：
  - `Accurate_Retrieval`
  - `Conflict_Resolution`

### 2.2 本地路径

- 数据文件：`MemoryAgentBench/data/*.json`
- 数据配置文件：`MemoryAgentBench/configs/data_conf/**/*.yaml`
- agent 配置文件：`MemoryAgentBench/configs/agent_conf/**/*.yaml`
- 运行输出：`MemoryAgentBench/outputs/<agent-output-dir>/<dataset>/*_results.json`
- agent / memory 缓存：`MemoryAgentBench/agents/`
- RAG 检索中间结果：`MemoryAgentBench/outputs/rag_retrieved/`
- 参考脚本：`MemoryAgentBench/bash_files/sh/*.sh`

## 3. 基本运行方式

统一入口是 `MemoryAgentBench/main.py`：

常用参数：

- `--agent_config`：memory / RAG / long-context agent 的 YAML 配置。
- `--dataset_config`：任务和子数据集 YAML 配置。
- `--max_test_queries_ablation N`：最多测试 N 个 query，适合 smoke test。
- `--force`：即使已有结果文件也强制重新运行。

结果会保存在 agent YAML 的 `output_dir` 下，并按 dataset 大类分目录。例如：

```text
outputs/deepseek-chat-amem/Conflict_Resolution/factconsolidation_sh_262k_..._results.json
```

## 4. 默认参数配置

默认参数主要来自 `MemoryAgentBench/configs/`，分为两类：

- `configs/data_conf/**/*.yaml`：定义数据集、上下文长度、chunk、生成长度、样本数和模板行为。
- `configs/agent_conf/**/*.yaml`：定义 agent 类型、模型、输入预算、检索参数、输出目录和 memory 系统专属参数。

### 4.1 Dataset 配置默认值

所有 dataset YAML 基本都包含以下字段：

| 参数 | 默认/常见取值 | 说明 |
| --- | --- | --- |
| `dataset` | `Accurate_Retrieval` / `Conflict_Resolution` | 数据集 split。 |
| `sub_dataset` | 由具体 YAML 指定 | 用于过滤 Hugging Face 样本的 `metadata.source`，也会进入输出文件名。 |
| `chunk_size` | `4096` | 默认 context 切分粒度；如果 agent YAML 设置了 `agent_chunk_size`，memory agent 会优先使用 `agent_chunk_size`。 |

### 4.2 Agent 配置默认值

所有 agent YAML 都包含以下通用字段：

| 参数 | 默认/常见取值 | 说明 |
| --- | --- | --- |
| `agent_name` | 例如 `Long_context_agent_*`、`Simple_rag_bm25`、`Structure_rag_mem0`、`Embedding_rag_memos` | 决定 `AgentWrapper` 的初始化和 `send_message()` 路由分支。 |
| `model` | `deepseek-chat`、`deepseek-v3-2-251201`、`gpt-4o-mini`、`gpt-5-mini`、`gemini-*`、`claude-*` 等 | 最终回答模型或 agent 主模型。 |
| `temperature` | DeepSeek 实验多为 `1.0`，OpenAI/Gemini/Claude 系列多为 `0.7` | 传给回答模型或构建模型的采样温度。 |
| `input_length_limit` | `95000` 到 `10000000` | 输入预算。long-context 模型通常贴近模型窗口。 |
| `output_dir` | `./outputs/<agent-name>` | 结果 JSON 输出根目录。 |

RAG 和 memory agent 常见额外字段：

| 参数 | 默认/常见取值 | 说明 |
| --- | --- | --- |
| `retrieve_num` | `10` | query 阶段检索 top-k。 |
| `agent_chunk_size` | `4096` | memory agent 写入 context 时的 chunk 大小。 |

新增 memory 系统时，通常只需要沿用通用字段，并按系统能力增加专属字段。

### 4.3 命令行覆盖规则

`main.py` 会先读取 YAML，再应用命令行覆盖：

- `--chunk_size_ablation N`：当 `N > 0` 时覆盖 dataset `chunk_size`；如果 agent YAML 带 `agent_chunk_size`，也会覆盖 `agent_chunk_size`。
- `--max_test_queries_ablation N`：当 `N > 0` 时限制本次最多评测 N 个 query，并写入 `dataset_config["max_test_queries"]`。
- `--force`：忽略已有结果进度，强制从头运行并覆盖同名结果文件。
- `generation_max_length`：如果 agent YAML 单独设置该字段，则优先使用 agent 配置；否则使用 dataset YAML。
- 输出文件名由 `dataset_config` 和 `agent_config` 共同生成，默认包含 `sub_dataset`、`tag`、`context_max_length`、`generation_max_length`、`shots`、`max_test_samples`，并按 agent 类型追加 `k`、`chunk`、`mode` 等信息。

## 5. 新 Memory 系统适配指南

适配一个新的 memory 系统需要同时完成代码适配、参数适配和脚本构建。现有实现可参考 mem0。

### 5.1 新增 agent YAML

在 `MemoryAgentBench/configs/agent_conf/RAG_Agents/<new_memory>/` 下新增 YAML，例如：

```yaml
agent_name: Structure_rag_<new_memory>
model: deepseek-v3-2-251201
temperature: 1.0
input_length_limit: 10000000
buffer_length: 1000
output_dir: ./outputs/deepseek-chat-<new_memory>

retrieve_num: 10
agent_chunk_size: 4096
```

常用字段说明：

- `agent_name`：必须包含可被 `AgentWrapper` 识别的 memory 系统关键字，例如 `amem`、`memos`、`memoryos`、`memobrain`。
- `model`：最终回答阶段使用的模型；DeepSeek 路径会走 Ark batch / OpenAI-compatible client。
- `output_dir`：结果输出目录，smoke test 脚本也需要使用同一路径检查 JSON。
- `retrieve_num`：query 阶段检索 memory 的 top-k。
- `agent_chunk_size`：memory 系统逐 chunk 写入时的 chunk 大小。只要设置了该字段，`ConversationCreator` 就会用它切分 context。
- 其他字段按系统需要增加，例如 `embed_model`、`vector_dimension`、`short_term_capacity`、`token_budget`、`*_url`、`*_api_key`。

如果新系统不需要逐 chunk 写入，可以不设置 `agent_chunk_size`，代码会使用 dataset YAML 里的 `chunk_size`。

### 5.2 注册初始化分支

在 `MemoryAgentBench/agent.py` 的 `AgentWrapper._initialize_agent_by_type()` 中添加新系统分支：

```python
elif self._is_agent_type("<new_memory>"):
    self._initialize_<new_memory>_agent(agent_config, dataset_config)
```

然后实现初始化方法：

```python
def _initialize_<new_memory>_agent(self, agent_config, dataset_config):
    self.retrieve_num = agent_config["retrieve_num"]
    self.agent_start_time = time.time()
    self.client = self._create_oai_client()
    # 初始化新 memory 系统、embedding、vector store、外部服务 client 等。
```

初始化阶段建议完成：

- 读取 YAML 参数和环境变量。
- 建立 memory 系统实例或延迟初始化所需状态。
- 初始化回答阶段 LLM client。
- 初始化 token 统计字段，例如 `self._<new_memory>_memorization_input_tokens` 和 `self._<new_memory>_memorization_output_tokens`。
- 如果按 context 隔离 memory，记录 `current_context_id`，在 context 变化时创建新的 memory 实例。

### 5.3 实现 memory 写入与查询接口

新增 handler：

```python
def _handle_<new_memory>_agent(self, message, memorizing, query_id, context_id):
    if memorizing:
        # message 是一个 context chunk。
        # 在这里写入 memory / 更新 recurrent memory / 构建索引。
        return "Memorized"

    memory_construction_time = time.time() - self.agent_start_time
    # 1. 从 message 中抽取检索 query，必要时可复用 self._extract_retrieval_query(message)
    # 2. 从新 memory 系统检索相关内容
    # 3. 拼接 system prompt + retrieved memory
    # 4. 调用 self._chat_complete(...) 生成答案
    # 5. 返回统一结果 dict
```

query 阶段必须返回统一格式：

```python
output = self._create_standard_response(
    response_text,
    input_tokens,
    output_tokens,
    memory_construction_time,
    query_time_len,
)
```

如果 memory 构建阶段调用了 LLM，需要额外写入：

```python
output["memorization_input_len"] = self._<new_memory>_memorization_input_tokens
output["memorization_output_len"] = self._<new_memory>_memorization_output_tokens
```

这两个字段会被 `main.py` 汇总到最终 JSON 的 `token_stats` 中。

### 5.4 注册 send_message 路由

在 `AgentWrapper.send_message()` 中添加新系统分支：

```python
elif self._is_agent_type("<new_memory>"):
    return self._handle_<new_memory>_agent(message, memorizing, query_id, context_id)
```

评测生命周期固定为：

1. `initialize_and_memorize_agent()` 创建 agent。
2. 对每个 context chunk 调用 `send_message(chunk, memorizing=True, context_id=...)`。
3. `save_agent()` 保存 agent 状态。
4. 对每个 query 调用 `send_message(query, memorizing=False, query_id=..., context_id=...)`。
5. 每个 query 后保存一次结果 JSON。

### 5.5 适配缓存保存与加载

如果新 memory 系统需要持久化状态，在 `AgentWrapper.save_agent()` 和 `AgentWrapper.load_agent()` 中补充分支。

示例：

```python
if self._is_agent_type("<new_memory>"):
    os.makedirs(self.agent_save_to_folder, exist_ok=True)
    # 保存 memory 状态到 self.agent_save_to_folder
```

同时在 `MemoryAgentBench/initialization.py` 的 `generate_agent_save_folder()` 中，把新系统关键字加入 memory agent 类型列表。这样每个 context 会使用独立缓存目录：

```text
agents/<agent_name>_<sub_dataset>_chunk<agent_chunk_size>_model<model>/exp_<context_id>
```

如果新系统是纯 memory、每次运行都重建，也应保证 `save_agent()` 不会破坏主流程。

### 5.6 模板与数据格式适配

query 和 memorization 的文本模板来自 `MemoryAgentBench/utils/templates.py`：

```python
get_template(self.sub_dataset, "system", self.agent_name)
get_template(self.sub_dataset, "memorize", self.agent_name)
get_template(self.sub_dataset, "query", self.agent_name)
```

新增 memory 系统时优先复用现有模板。如果 `agent_name` 导致模板分支无法匹配，需要在 `templates.py` 中补齐对应逻辑。

`ConversationCreator` 期望每条样本包含：

- `context`
- `questions`
- `answers`
- 可选元数据：`question_dates`、`question_types`、`question_ids`、`previous_events`、`qa_pair_ids`、`source`

### 5.7 环境变量

运行前需要在 `MemoryAgentBench/.env` 中配置模型、embedding 和批量推理相关环境变量。当前项目使用的 `.env` 主要包含以下三类服务：

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 必需 | 火山方舟/DeepSeek API key。DeepSeek 主模型调用和 Ark batch 推理会读取该变量。 |
| `BATCH_MODEL` | 必需 | 火山方舟 batch inference 接入点名称，例如 `ep-...`。批量推理脚本需要使用。 |
| `DASHSCOPE_API_KEY` | 使用 memory/RAG backend 时必需 | DashScope API key，主要用于 embedding 或依赖 DashScope-compatible endpoint 的 memory 系统。 |
| `DASHSCOPE_BASE_URL` | 使用 DashScope 时必需 | DashScope OpenAI-compatible endpoint，默认可使用 `https://dashscope.aliyuncs.com/compatible-mode/v1`。 |

推荐 `.env` 模板如下。请只填写自己的 key，不要把真实密钥提交到仓库：

```bash
DEEPSEEK_API_KEY=<your-ark-or-deepseek-api-key>
BATCH_MODEL=<your-ark-batch-endpoint>

DASHSCOPE_API_KEY=<your-dashscope-api-key>
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果只运行不依赖 embedding 的 long-context baseline，通常只需要配置主模型相关变量。若运行 mem0、amem、memos、memoryos 等 memory/RAG 系统，通常还需要同时配置 DashScope 相关变量。

## 6. 脚本构建指南

现有脚本位于：

```text
MemoryAgentBench/bash_files/sh/
```

### 6.1 单系统全量实验脚本

参考 `run_amem.sh`、`run_memos.sh`、`run_memoryos.sh`、`run_memobrain.sh`。新系统脚本建议命名为：

```text
bash_files/sh/run_<new_memory>.sh
```

模板可参考 `bash_files/sh/run_exp_mem0.sh`。

### 6.2 Smoke test 脚本

```bash
cd MemoryAgentBench

python main.py \
  --agent_config configs/agent_conf/RAG_Agents/<new_memory>/<new_memory>.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml \
  --max_test_queries_ablation 2 \
  --force
```

运行后检查 YAML 中 `output_dir` 对应目录下是否生成非空 JSON：

```text
outputs/<agent-output-dir>/Conflict_Resolution/factconsolidation_sh_262k*_results.json
```
