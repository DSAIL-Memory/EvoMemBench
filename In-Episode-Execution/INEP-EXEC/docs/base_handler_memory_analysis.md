# `inference_multi_turn_FC` 三种 `external_memory` 的数据流与上下文管理分析

> 源文件：`bfcl_eval/model_handler/base_handler.py`
> 分析范围：`inference_multi_turn_FC`（第 344–766 行）

---

## 0. 关键前置变量

```python
_COMPRESSION_MEMORY_TYPES = frozenset({"memagent", "memobrain"})   # 第 95 行

use_compression = (
    external_memory is not None
    and getattr(external_memory, "memory_type", "") in _COMPRESSION_MEMORY_TYPES
)   # 第 433–435 行
```

| `external_memory` | `use_compression` |
|---|---|
| `None` | `False` |
| mem0 | `False` |
| memobrain | `True` |

---

## 1. `external_memory = None`

### 整体数据流

```
test_entry
  └─ _pre_query_processing_FC()          → inference_data 初始化
  └─ _compile_tools()                    → inference_data["tools"] 填充
  └─ inference_data["context_budget"] = context_budget

for turn_idx, current_turn_message:
  ├─ turn 0: add_first_turn_message_FC() → inference_data["message"] 初始化
  ├─ turn N: _add_next_turn_user_message_FC() → 追加用户消息
  │
  │  [内存 utilize]  ← 跳过
  │
  └─ while True (step loop):
       _query_FC()                        → 子类使用 inference_data["message"]
       _parse_query_response_FC()
       _add_assistant_message_FC()        → 追加 assistant 消息
       decode_execute()
       execute_multi_turn_func_call()
       _add_execution_results_FC()        → 追加 tool 结果消息
       count += 1 / force_quit 检查

  [内存 update]  ← 跳过

metadata 汇总（无 memory_usage 字段）
```

### 上下文管理

- `inference_data["message"]` 是**唯一**的消息列表，随每轮每步单调增长。
- 上下文裁剪完全依赖子类在 `_query_FC` 内部调用 `_truncate_messages_to_budget()`。
- `_truncate_messages_to_budget` 裁剪顺序（oldest-first）：
  1. 跨轮历史（last user message 之前的消息）
  2. 当前轮的 intra-turn 步骤（assistant + tool-result 对）
  3. 当前步 tool result 内容前缀截断
- 裁剪时插入 `[CONTEXT TRUNCATED: N message(s) removed]` 占位消息。
- **没有外部记忆，没有 snippet 注入，没有压缩摘要。**

---

## 2. `external_memory = mem0`

### 整体数据流

```
test_entry → inference_data 初始化（同 None）

external_memory.begin_sample()             ← 重置本样本的 usage 统计（第 430 行）

for turn_idx, current_turn_message:
  ├─ turn 0: add_first_turn_message_FC()
  ├─ turn N: _add_next_turn_user_message_FC()
  │
  │  [内存 utilize]  ← 执行（第 491–500 行）
  │    user_text = _extract_user_text_from_messages(inference_data["message"])
  │    snippet   = external_memory.utilize(user_text)   # 向 mem0/Qdrant 检索
  │    if snippet:
  │        _append_memory_to_last_user_message(
  │            inference_data["message"], snippet        # 原地追加到最后一条 user 消息
  │        )
  │
  └─ while True (step loop):
       _query_FC()                      → 发送含 snippet 的 inference_data["message"]
       _parse_query_response_FC()
       _add_assistant_message_FC()
       decode_execute()
       execute_multi_turn_func_call()
       _add_execution_results_FC()

  [内存 update]  ← 执行（第 650–658 行）
    turn_delta = inference_data["message"][turn_start_idx:]  # 本轮新增消息
    external_memory.update(turn_delta)                       # 写入 mem0/Qdrant

external_memory.drain_usage()             ← 收集 latency/tokens/调用次数
metadata["memory_usage"] = { ... }        ← 写入输出 metadata
```

### 上下文管理

- `inference_data["message"]` 仍是主消息列表，随轮步增长。
- **每轮开始时**（step loop 前）：mem0 检索片段 snippet 被追加进最后一条 user 消息的文本尾部（原地修改）。
- 因此 `_query_FC` 收到的消息里，user 消息已包含记忆片段。
- 上下文裁剪仍由子类 `_truncate_messages_to_budget` 承担，snippet 被视为普通消息内容。
- **每轮结束后**：本轮消息增量（`turn_start_idx` 之后）同步写入 mem0，供后续轮次检索。
- `_compressed_messages` **不创建**（`use_compression=False`）。

### 记忆读写时序

```
Turn 0:  utilize(query0) → snippet0 注入 → 执行 → update(turn0_delta)
Turn 1:  utilize(query1) → snippet1 注入 → 执行 → update(turn1_delta)
Turn N:  utilize(queryN) → snippetN 注入 → 执行 → update(turnN_delta)
```

---

## 3. `external_memory = memobrain`

### 整体数据流

```
test_entry → inference_data 初始化（同 None）

external_memory.begin_sample()             ← 同 mem0

use_compression = True                     ← memobrain 走压缩路径

for turn_idx, current_turn_message:
  ├─ turn 0:
  │   add_first_turn_message_FC()
  │   inference_data["_compressed_messages"] = list(inference_data["message"])
  │   # 初始化压缩轨道，与 "message" 初始内容相同
  │
  ├─ turn N (N ≥ 1):
  │   _add_next_turn_user_message_FC()     # 追加到 inference_data["message"]
  │   if current_turn_message:             # 同步维护压缩轨道
  │     _tools_tokens = _count_tokens(tools)
  │     effective = context_budget - 512 - _tool_tokens
  │     if _count_tokens(_compressed) + _count_tokens([new_user_msg]) > effective:
  │         # 超预算 → 压缩
  │         summary = BaseHandler._compress_history(
  │             _compressed,
  │             target_tokens = effective - count(new_user_msg) - 100,
  │             memory_type   = "memobrain",
  │             current_user_prompt = new_user_msg["content"],
  │         )
  │         # MemoBrainMemory.compress() 生成 LLM 摘要
  │         combined = summary + "\n\n" + new_user_msg["content"]
  │         inference_data["_compressed_messages"] = [
  │             {"role": "user", "content": combined}
  │         ]
  │     else:
  │         # 未超预算 → 直接追加
  │         inference_data["_compressed_messages"].append(new_user_msg)
  │
  │  [内存 utilize]  ← 跳过（use_compression=True，第 491 行 guard）
  │
  └─ while True (step loop):
       _query_FC()
       # 子类实现应使用 inference_data["_compressed_messages"] 作为实际发送内容
       _parse_query_response_FC()
       # 追加 assistant 消息到两个轨道
       _prev_msg_len = len(inference_data["message"])
       _add_assistant_message_FC()          → inference_data["message"] 增长
       inference_data["_compressed_messages"].extend(
           inference_data["message"][_prev_msg_len:]   # 同步新 assistant 消息
       )
       decode_execute()
       execute_multi_turn_func_call()
       # 追加 tool 结果到两个轨道
       _prev_msg_len = len(inference_data["message"])
       _add_execution_results_FC()          → inference_data["message"] 增长
       inference_data["_compressed_messages"].extend(
           inference_data["message"][_prev_msg_len:]   # 同步新 tool 消息
       )

  [内存 update]  ← 调用但为空操作（第 650–658 行）
    turn_delta = inference_data["message"][turn_start_idx:]
    external_memory.update(turn_delta)     # MemoBrainMemory.update() = pass，无实际写入

external_memory.drain_usage()
metadata["memory_usage"] = { ... }
```

### 双轨道消息设计

| 轨道 | 键 | 用途 |
|---|---|---|
| 完整历史 | `inference_data["message"]` | 状态日志、memory update、元数据输出 |
| 压缩轨道 | `inference_data["_compressed_messages"]` | 实际发送给 LLM 的上下文（由子类使用） |

- **两个轨道同步追加** assistant 消息和 tool 结果（step loop 内）。
- **只有压缩轨道**在新轮开始时做预算检查和压缩。
- `inference_data["message"]` 始终保留完整未裁剪历史，不参与压缩。

### 上下文管理（压缩路径）

压缩在**新轮用户消息加入压缩轨道时**触发（而不是在 query 前）：

```
effective_budget = context_budget - 512(response_margin) - tool_tokens

if tokens(_compressed_messages) + tokens(new_user_msg) > effective_budget:
    summary = MemoBrainMemory.compress(
        messages           = _compressed_messages,
        target_tokens      = effective_budget - tokens(new_user_msg) - 100,
        current_user_prompt = new_user_msg_content,   # 查询导向摘要
    )
    _compressed_messages = [user: summary + "\n\n" + new_user_msg_content]
```

- 压缩后 `_compressed_messages` 被**重置为单条 user 消息**（摘要 + 新问题合并）。
- `_truncate_messages_to_budget` 仍可在子类 `_query_FC` 内被调用，作为兜底裁剪。

### 记忆读写时序

```
Turn 0:  (utilize 跳过) → 执行 → update() 调用但 pass（空操作）
Turn 1:  压缩检查 → 若超预算则 compress() → 执行 → update() 空操作
Turn N:  压缩检查 → ...  → update() 空操作
```

- **没有 snippet 注入**（`utilize()` 返回 `""`，guard 也会拦截）。
- **没有持久化写入**（`update()` = `pass`，`TODO` 未实现）。
- **真正的"记忆更新"是 `_compressed_messages` 的维护**：每步把新 assistant/tool 消息追加进去，新轮超预算时调 `compress()` 截取近端消息后缀生成摘要（当前实现为朴素后缀截取，非 LLM 摘要）。

---

## 4. 三种模式对比总结

| 维度 | `None` | `mem0` | `memobrain` |
|---|---|---|---|
| `use_compression` | `False` | `False` | `True` |
| `begin_sample()` | 否 | 是 | 是 |
| `_compressed_messages` 轨道 | 不存在 | 不存在 | 存在（双轨道） |
| 上下文裁剪方式 | `_truncate_messages_to_budget`（子类） | 同 None + snippet 占用空间 | `_compress_history`（新轮触发）+ 子类兜底 |
| 每轮前 utilize | 否 | 是（检索片段，注入 user 消息） | 否（guard 拦截） |
| snippet 注入位置 | — | 最后一条 user 消息尾部 | — |
| 每轮后 update | 否 | 是（增量写入 mem0） | 调用但空操作（`pass`） |
| query 使用的消息列表 | `inference_data["message"]` | `inference_data["message"]`（含 snippet） | `inference_data["_compressed_messages"]`（子类） |
| `metadata["memory_usage"]` | 无 | 有 | 有 |
| 历史丢失方式 | 静默截断（truncate） | 静默截断 | LLM 生成摘要替换（compress） |

---

## 5. 辅助函数说明

| 函数 | 位置 | 作用 |
|---|---|---|
| `_count_tokens(obj)` | 第 23 行 | tiktoken cl100k_base 估算 token 数，fallback `len/4` |
| `_extract_user_text_from_messages(msgs)` | 第 35 行 | 从消息列表反向查找最后一条 user 消息的文本 |
| `_append_memory_to_last_user_message(msgs, snippet)` | 第 53 行 | 原地追加 snippet 到最后一条 user 消息（str 或 list content） |
| `_truncate_messages_to_budget(messages, tools, budget)` | 第 136 行 | 裁剪消息以适应预算，保留 system/最后 user/当前 step |
| `_compress_history(messages, target_tokens, memory_type, prompt)` | 第 277 行 | 按 memory_type 分发到 MemoBrainMemory.compress 或 fallback 截断拼接 |
