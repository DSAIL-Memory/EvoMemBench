# EvoMemBench-v1

EvoMemBench 包含 in-episode / cross-episode、knowledge / execution 等不同评测任务。正式运行前请优先阅读对应子目录下的 `README.md`。

## 1. 目录导航

| 目录 | 说明 |
| --- | --- |
| `In-Episode-Knowledge/INEP-KNOW/` | In-episode knowledge 评测，入口为 `MemoryAgentBench/`。 |
| `In-Episode-Execution/INEP-EXEC/` | In-episode execution / BFCL 多轮工具调用评测。 |
| `Cross-Episode-Knowledge/CROSSEP-KNOW/` | Cross-episode knowledge 评测，基于 context-grouped CL-bench 数据。 |
| `Cross-Episode-Execution/` | Cross-episode execution 相关任务，包含 embodied AI、tool using、web search 等场景。 |
| `EvoMemBench-Memory-Systems/` | 外部 memory 系统的本地化依赖，例如 mem0、A-mem、MemOS、MemoryOS、MemoBrain 和 memagent。 |

## 2. 环境配置

不同任务依赖不同，建议为每个评测任务单独创建 Conda 环境。具体 Python 版本和依赖安装方式见各任务目录下的 `README.md`。

常见流程如下：

```bash
cd <task-dir>
conda create -n <env-name> python=<version>
conda activate <env-name>
pip install -r requirements.txt
```

如果任务需要使用外部 memory backend，通常还需要从 `EvoMemBench-Memory-Systems/` 中以 editable 模式安装对应依赖。示例：

```bash
pip install -e ../../EvoMemBench-Memory-Systems/mem0
pip install -e ../../EvoMemBench-Memory-Systems/A-mem
pip install -e ../../EvoMemBench-Memory-Systems/MemOS
pip install -e ../../EvoMemBench-Memory-Systems/MemoryOS
```

## 3. Memory 系统适配

适配新的 memory 系统时，可以先判断它是否能作为通用 backend 复用：

- 如果是通用 memory 方法，不需要针对具体任务场景定制 prompt，建议将核心实现放在 `EvoMemBench-Memory-Systems/` 中。不同评测任务只需要各自实现 adapter 或注册入口。
- 如果 memory 行为强依赖任务场景，例如需要为不同数据集重写 prompt、输入格式或压缩逻辑，建议在对应任务目录中实现任务专属代码。

已有系统可作为参考：

| 系统 | 参考目录 |
| --- | --- |
| mem0 | `EvoMemBench-Memory-Systems/mem0/` |
| A-mem | `EvoMemBench-Memory-Systems/A-mem/` |
| MemOS | `EvoMemBench-Memory-Systems/MemOS/` |
| MemoryOS | `EvoMemBench-Memory-Systems/MemoryOS/` |
| MemoBrain | `EvoMemBench-Memory-Systems/MemoBrain/` |
| memagent | `EvoMemBench-Memory-Systems/memagent/` |

## 4. API 与环境变量

当前默认推理链路主要使用 Volcengine Ark / DeepSeek 批量推理 API；默认 embedding 链路主要使用 DashScope `text-embedding-v4`。

### 4.1 Volcengine Ark / DeepSeek

- API Key 申请入口：[Volcengine Ark API Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey?apikey=%7B%7D)
- 批量推理接入点申请入口：[Volcengine Ark Batch Inference](https://console.volcengine.com/ark/region:ark+cn-beijing/batchInference)
- 批量推理接入点建议选择 DeepSeek-V3.2 模型。

常用环境变量：

```bash
DEEPSEEK_API_KEY=<your-ark-or-deepseek-api-key>
BATCH_MODEL=<your-ark-batch-endpoint>
```

### 4.2 DashScope Embedding

- API Key 申请入口：[DashScope API Key](https://bailian.console.aliyun.com/cn-beijing?tab=model#/api-key)
- 默认 embedding model：`text-embedding-v4`

常用环境变量：

```bash
DASHSCOPE_API_KEY=<your-dashscope-api-key>
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

请将真实 key 写入各任务目录要求的 `.env` 文件中，不要提交真实密钥。

## 5. 脚本构建与默认参数

每个任务目录都提供了可参考的运行脚本。新增 memory 系统或新实验配置时，建议优先复制并修改现有 mem0 脚本。

- 输出目录命名规则。
- smoke test 或小样本调试参数。

建议阅读顺序：

1. 先读当前任务目录下的 `README.md`。
2. 找到该任务中已有的 mem0 运行脚本。
3. 复制脚本并替换 memory 类型、配置文件路径、输出目录和必要的 API 参数。
4. 先用小样本 smoke test 验证，再启动全量实验。

## 6. 实验结果记录指南
大部分数据集所使用的指标可以直接从输出文件中直接读取（除了INEP-KNOW指标较多，待确认具体是哪一个）