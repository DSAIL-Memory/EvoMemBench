# `inference_multi_turn_FC` — `external_memory=mem0` 深度分析

> 源文件：`bfcl_eval/model_handler/base_handler.py`（`inference_multi_turn_FC`，第 344–777 行）  
> 记忆后端：`bfcl_eval/memory/mem0_backend.py`（`Mem0Memory`）  
> 指标模块：`bfcl_eval/memory/metrics.py`

---

## 0. 关键前置标志

```python
use_compression = (
    external_memory is not None
    and getattr(external_memory, "memory_type", "") in _COMPRESSION_MEMORY_TYPES
)
# _COMPRESSION_MEMORY_TYPES = frozenset({"memagent", "memobrain"})
# mem0 不在其中 → use_compression = False
```

因此 mem0 路径：
- **不创建** `_compressed_messages` 轨道
- **执行** `utilize()` 检索注入（guard：`not use_compression`）
- **执行** `update()` 增量写入（guard：`not use_compression`）
- 上下文裁剪完全由子类的 `_truncate_messages_to_budget()` 承担

---

## 1. 整体数据流（跨所有轮次）

```
test_entry
  ├─ _pre_query_processing_FC()       → inference_data 初始化（清空 message 列表等）
  ├─ _compile_tools()                 → inference_data["tools"] 填充
  └─ inference_data["context_budget"] = context_budget

external_memory.begin_sample()        → 重置本线程 MemoryUsageLog（latency/tokens/计数）

for turn_idx, current_turn_message:
  │
  ├─ turn_start_idx = len(inference_data["message"])   # 记录本轮开始位置
  │
  ├─ [消息追加]
  │   turn 0: add_first_turn_message_FC()              → message 初始化
  │   turn N: _add_next_turn_user_message_FC()         → 追加 1 条 user 消息
  │
  ├─ [内存 utilize]（第 491–500 行，guard: not use_compression → 执行）
  │   user_text = _extract_user_text_from_messages(inference_data["message"])
  │   snippet   = external_memory.utilize(user_text)   # Qdrant 向量检索
  │   if snippet:
  │       _append_memory_to_last_user_message(         # 原地修改最后一条 user 消息
  │           inference_data["message"], snippet
  │       )
  │
  └─ while True (step loop):
       ├─ _query_FC(inference_data)                    # 发送含 snippet 的消息列表
       ├─ _parse_query_response_FC(api_response)
       ├─ _prev_msg_len = len(inference_data["message"])
       ├─ _add_assistant_message_FC()                  → 追加 1 条 assistant 消息
       │   (use_compression=False → 不维护 _compressed_messages)
       ├─ decode_execute()                             → decoded_model_responses
       ├─ execute_multi_turn_func_call()               → execution_results
       ├─ _prev_msg_len = len(inference_data["message"])
       ├─ _add_execution_results_FC()                  → 追加 tool 结果消息（≥1 条）
       │   (use_compression=False → 不维护 _compressed_messages)
       ├─ count += 1
       └─ if count > MAXIMUM_STEP_LIMIT: force_quit → break

  [内存 update]（第 661–669 行，guard: not use_compression → 执行）
  turn_delta = inference_data["message"][turn_start_idx:]  # 本轮新增所有消息
  external_memory.update(turn_delta)                       # 写入 Qdrant

  if force_quit: break

external_memory.drain_usage()         → MemoryUsageLog（计入 metadata["memory_usage"]）
```

---

## 2. 各阶段细节

### 2.1 `begin_sample()`（第 430 行）

```python
# bfcl_eval/memory/metrics.py
def begin_sample() -> None:
    _TLS.usage = MemoryUsageLog()   # 线程局部重置
```

每个样本在推理前重置 `MemoryUsageLog`（`latency_s / input_tokens / output_tokens / embedding_tokens / n_llm_calls / n_embed_calls / n_utilize / n_update`），保证多线程并行时指标隔离。

---

### 2.2 `utilize()`（每轮前，第 491–500 行）

#### 2.2.1 调用链

```
_extract_user_text_from_messages(inference_data["message"])
  ↓ 从后往前找最后一条 role="user" 的消息，提取文本
  ↓ 支持 str 内容和 list[{"type":"text","text":...}] 多模态内容

Mem0Memory.utilize(user_text)
  ↓ _truncate_query_to_embed_limit(user_text)
  │   # text-embedding-v4 文档限制 8192 tokens
  │   # 超限时保留尾部（tail-preserving），保护任务相关的末尾描述
  ↓ self._mem.search(query=..., filters={"user_id": ...}, top_k=self._top_k)
  │   # 调用 Qdrant 向量检索，返回最相关的 top_k 条记忆
  ↓ 格式化为 "\n\nRelevant memories from prior episodes:\n- mem1\n- mem2\n..."
  ↓ metrics.n_utilize += 1
```

#### 2.2.2 snippet 注入

```python
_append_memory_to_last_user_message(inference_data["message"], snippet)
```

- **原地**修改 `inference_data["message"]` 中最后一条 user 消息
- str 内容：直接 `content += snippet`
- list 内容：找最后一个 `type="text"` part，追加 snippet；若无，则新增一个 text part

**注意**：snippet 被并入 user 消息文本，后续 `_query_FC` 无法区分哪部分是原始用户问题，哪部分是注入记忆。这在 `_truncate_messages_to_budget` 截断时也会一起被裁剪。

---

### 2.3 step loop

```
while True:
    _query_FC()         ← 发送 inference_data["message"]（含 snippet 的 user 消息）
    _add_assistant_message_FC()   → inference_data["message"] + 1 条
    _add_execution_results_FC()   → inference_data["message"] + N 条（每个 tool call 一条）
    count += 1 / force_quit
```

`inference_data["message"]` 是**唯一**的消息列表，随每步单调增长，不做任何压缩。

---

### 2.4 `update()`（每轮后，第 661–669 行）

#### 2.4.1 调用链

```python
turn_delta = inference_data["message"][turn_start_idx:]
# turn_delta 包含：
#   - 本轮 user 消息（含 snippet 注入内容）
#   - 本轮所有步的 assistant 消息
#   - 本轮所有步的 tool 结果消息

external_memory.update(turn_delta)
```

```python
# Mem0Memory.update()
with self._lock:                       # RLock，支持多线程并行调用
    self._mem.add(
        messages=trajectory,           # 传入原始消息列表
        user_id=self._user_id,
        infer=True,                    # 触发 LLM 进行事实提取（fact extraction）
    )
# metrics.n_update += 1
```

- `infer=True`：mem0 会调用 ArkBatchLLM 对 `trajectory` 进行事实提取，将提炼出的结构化事实存入 Qdrant
- `readonly=True`（Phase 2 cross-env）时直接 `return`，不写入
- 写入对象是提炼后的**事实**，而非原始消息文本

#### 2.4.2 `turn_delta` 内容构成

| 消息 | 来源 |
|---|---|
| `user` | 本轮用户问题 + 注入的 snippet（已合并） |
| `assistant` × steps | 本轮各步的 LLM 输出（tool call） |
| `tool` × steps | 本轮各步的执行结果 |

---

### 2.5 `drain_usage()`（所有轮次后，第 717–719 行）

```python
external_memory_usage = external_memory.drain_usage()
# → 返回 MemoryUsageLog，重置线程局部 usage
metadata["memory_usage"] = {
    "latency_s": ..., "input_tokens": ..., "output_tokens": ...,
    "embedding_tokens": ..., "n_llm_calls": ..., "n_embed_calls": ...,
    "n_utilize": ..., "n_update": ...,
}
```

---

## 3. 上下文管理（截断逻辑）

### 3.1 截断发生的位置

截断发生在**子类的 `_query_FC` 内部**，`inference_multi_turn_FC` 不直接调用截断。子类调用：

```python
messages = BaseHandler._truncate_messages_to_budget(
    inference_data["message"],   # 原始完整列表（含 snippet）
    inference_data["tools"],
    inference_data["context_budget"],
)
```

`_truncate_messages_to_budget` **不修改原始列表**，返回一个（可能更短的）副本用于发送。因此 `inference_data["message"]` 始终保留完整历史，用于后续的 `update(turn_delta)`。

### 3.2 预算计算

```python
_RESPONSE_MARGIN = 512
tool_tokens = _count_tokens(tools) if tools else 0
effective = budget - _RESPONSE_MARGIN - tool_tokens
```

- `_RESPONSE_MARGIN = 512`：为模型响应预留的 token 余量
- `tool_tokens`：function schema 的 token 开销，随 `_compile_tools()` 结果变化
- snippet 本身占用的 token 数计入 `effective`（因为已经合并到 user 消息里）

### 3.3 截断策略（FC mode，`n_pin=0`）

```
消息列表分段：
  head        = []   (FC mode, n_pin=0, 无系统消息)
  pre_turn    = messages[0 : last_user_idx]        # 历史轮次消息
  user_msg    = [messages[last_user_idx]]          # 当前轮 user 消息（含 snippet）
  intra_turn  = messages[last_user_idx+1 : last_assistant_after_user]  # 当前轮前序步
  current_step= messages[last_assistant_after_user :]                  # 当前步（必须保留）
```

**丢弃顺序（oldest-first）**：

1. **跨轮历史**（`pre_turn`）：逐条从头部丢弃（最旧先丢）
   - 丢弃后插入占位：`{"role": "user", "content": "[CONTEXT TRUNCATED: N message(s) removed from conversation history]"}`
2. **当前轮前序步**（`intra_turn`）：按完整步（assistant + 其 tool results）从头部丢弃，保持 `tool_call_id` 配对完整性
   - 丢弃后插入占位：`{"role": "user", "content": "[CONTEXT TRUNCATED: N intra-turn message(s) from earlier steps removed]"}`
3. **当前步 tool result 内容裁剪**：若以上两步后仍超预算，对当前步的 tool 消息内容从头部截断（保留尾部，每次裁剪 20%）
   - 裁剪后加前缀：`"[CONTEXT TRUNCATED: tool result trimmed from X to Y chars, leading content omitted]\n"`

**step 0 时**（`last_user_idx` 后无 assistant 消息）：
```
head + middle (cross-turn history) + tail (last user)
```
只丢弃 `middle`，无 intra-turn 可丢。

**相邻 user 消息合并**：截断后若出现连续 user 消息（FC mode, n_pin=0），自动合并为单条（用 `\n\n` 分隔）。

### 3.4 snippet 与截断的交互

| 场景 | 结果 |
|---|---|
| pre_turn 被全部丢弃，当前轮 user_msg 被保留 | snippet 随 user_msg 保留，不受影响 |
| pre_turn 部分丢弃，插入截断占位符 | 占位符 role=user，与后续 user_msg 合并（`\n\n` 拼接） |
| snippet 过长导致当前 user_msg 超过 effective | `_query_FC` 不截断 user_msg 内容（user_msg 是"保留锚"），但若 tool result 超预算则裁剪 tool result |

**潜在问题**：snippet 的 token 成本直接消耗 effective 预算，可能导致更多历史轮次被丢弃。截断后的占位符与 snippet 会被合并到同一 user 消息中，可读性降低。

---

## 4. 跨轮记忆时序

```
Turn 0:
  utilize(turn0_user_text)       → Qdrant 为空 → snippet = ""（无注入）
  ← 执行 step loop →
  update(turn0_delta)            → LLM 事实提取 → 写入 Qdrant
  ↓ Qdrant 现在有 Turn 0 的事实

Turn 1:
  utilize(turn1_user_text)       → 检索 Turn 0 存入的事实 → snippet1
  _append_memory_to_last_user_message(snippet1)
  ← 执行 step loop（LLM 看到 snippet1）→
  update(turn1_delta)            → 写入 Turn 1 的事实（含 snippet1 文本）
  ↓ Qdrant 现在有 Turn 0 + Turn 1 的事实

Turn N:
  utilize(turnN_user_text)       → 检索所有历史事实 → snippetN（最多 top_k 条）
  ...
```

**特点**：
- 第一轮 `utilize()` 必然返回空（Qdrant 空）
- `update()` 写入的是当前轮完整 delta，包含 snippet 注入后的 user 消息；因此后续轮次提炼时，mem0 LLM 会看到注入的记忆内容（可能产生噪声/重复）
- Phase 2 cross-env：`readonly=True`，`update()` 直接返回，只执行 `utilize()` 检索

---

## 5. 函数调用关系图

```
inference_multi_turn_FC()
│
├─ external_memory.begin_sample()               [metrics.py: begin_sample()]
│
├─ for each turn:
│   ├─ add_first_turn_message_FC() / _add_next_turn_user_message_FC()
│   │
│   ├─ _extract_user_text_from_messages()       [base_handler.py:35]
│   ├─ external_memory.utilize()                [mem0_backend.py:159]
│   │   ├─ _truncate_query_to_embed_limit()     [mem0_backend.py:38]
│   │   ├─ self._mem.search()                   [mem0 library]
│   │   └─ metrics.n_utilize += 1
│   ├─ _append_memory_to_last_user_message()    [base_handler.py:53]
│   │
│   ├─ while True:
│   │   ├─ _query_FC()                          [子类实现]
│   │   │   └─ _truncate_messages_to_budget()   [base_handler.py:136]（子类内调用）
│   │   ├─ _parse_query_response_FC()
│   │   ├─ _add_assistant_message_FC()
│   │   ├─ execute_multi_turn_func_call()
│   │   └─ _add_execution_results_FC()
│   │
│   └─ external_memory.update(turn_delta)       [mem0_backend.py:185]
│       ├─ self._lock (RLock)
│       ├─ self._mem.add(..., infer=True)        [mem0 library: LLM 事实提取]
│       └─ metrics.n_update += 1
│
└─ external_memory.drain_usage()                [metrics.py: drain()]
```

---

## 6. 关键设计决策与隐患

| 维度 | 当前设计 | 潜在问题 |
|---|---|---|
| snippet 注入位置 | 原地追加到 `inference_data["message"]` 的最后一条 user 消息 | `inference_data["message"]` 被污染，`update()` 写入的 turn_delta 包含注入内容 |
| 截断时 snippet 处理 | snippet 并入 user_msg，随 user_msg 一起保留（不被截断） | snippet 过大时会挤占历史轮次的 token 预算，加剧截断 |
| update 写入内容 | 原始消息 dict（含 snippet 文本），mem0 运行 `infer=True` 提取事实 | 注入的 snippet 可能被重复提炼写入 Qdrant，形成记忆膨胀 |
| update 写入时机 | 每轮 step loop 结束后立即写入 | 若当前轮推理失败/force_quit，也会写入部分错误轨迹 |
| 截断后的 update | `inference_data["message"]` 保留完整历史（截断只在 `_query_FC` 内生效） | `update(turn_delta)` 写入完整 delta（不截断），防止记忆丢失 |
| `readonly=True` | Phase 2 cross-env 不写入，但仍计 `n_utilize` 指标 | 符合预期，Phase 2 是纯读模式 |
| 线程安全 | `_lock = threading.RLock()`，update 加锁 | Phase 1 workers=1 串行，Phase 2 并行但 readonly，不存在写并发 |

---

## 7. `_truncate_messages_to_budget` 完整逻辑（参考实现，第 136–275 行）

```python
effective = context_budget - 512 - tool_tokens

# FC mode: n_pin = 0 (无系统消息)
# last_user_idx: 最后一条 user 消息的下标

# step k≥1（last_user_idx 后存在 assistant 消息）:
pre_turn     = messages[0 : last_user_idx]
user_msg     = [messages[last_user_idx]]           # 含 snippet，不可丢弃
intra_turn   = messages[last_user_idx+1 : last_asst_after_user]
current_step = messages[last_asst_after_user :]    # 不可丢弃

# 丢弃顺序:
1. pre_turn 逐条从头丢，超限则插入占位符
2. intra_turn 按完整步（asst+tool）从头丢，插入占位符
3. current_step 中 tool 消息内容从头截断（每次 20%），加前缀提示

# step 0（last_user_idx 后无 assistant 消息）:
middle = messages[0 : last_user_idx]
tail   = messages[last_user_idx :]   # 含 snippet
middle 逐条从头丢，插入占位符

# 相邻 user 消息合并（FC mode join point 可能产生）
```

---

## 8. 总结

`external_memory=mem0` 时，`inference_multi_turn_FC` 采用**检索增强（RAG）+ 增量写入**的长期记忆策略：

- **读**：每轮前用当前 user 消息查询 Qdrant，将相关事实拼接进 user 消息
- **写**：每轮后将当前轮完整消息增量（含 snippet 注入后的 user 消息）写入 Qdrant，由 mem0 LLM 提炼事实
- **上下文裁剪**：完全依赖子类 `_query_FC` 内调用的 `_truncate_messages_to_budget`，`inference_multi_turn_FC` 本身不裁剪；裁剪只影响发送给 LLM 的副本，不影响用于 `update()` 的原始 `inference_data["message"]`
