#!/usr/bin/env python
"""
使用 memory 模块评估 ALFWorld。
在 eval_alfworld_correct.py 基础上增加 memory inject / update 两个阶段：
- inject: 每轮 episode 开始前，基于初始观察检索记忆并注入 system prompt
- update: episode 完成后，将完整轨迹存入记忆库

用法：
  python scripts/eval_alfworld_with_memory.py \
      --port 36005 --start_idx 2420 --num_samples 200 \
      --max_rounds 20 --parallel 20 \
      --memory_type mem0

  # 只读模式（不更新记忆库）：
      --readonly_memory

  # 禁用记忆（等同于原始 eval_alfworld_correct.py）：
      --no_memory

  # 使用 JSON 配置文件定制 memory 后端：
      --memory_config path/to/mem0_config.json
"""
import os
import re
import sys
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Add scripts/ to path so `memory` package is importable.
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def load_env():
    env_path = os.path.join(BASE_DIR, '.env')
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key] = val


load_env()

from agentenv.envs.alfworld import AlfWorldEnvClient
from agentenv.controller import APIAgent
from memory import MemoryCallStats, create_memory


class InstrumentedAPIAgent(APIAgent):
    """APIAgent with usage tracking and exponential-backoff retries."""

    def generate(self, conversation, max_retries=8, retry_delay=5.0):
        messages = [{"role": c["role"], "content": c["content"]} for c in conversation]
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                return content, reasoning, usage
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[APIAgent retry {attempt+1}/{max_retries}] {e}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)


class BatchAPIAgent:
    """使用火山引擎 ARK Batch API 的 LLM Agent。

    Token 计数说明：
    - prompt_tokens 原样返回（不扣除 cached），与 standard 模式数值一致
    - cached_tokens 单独返回，用于计算缓存命中率
    """

    def __init__(self, api_key, model):
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError(
                "Batch 模式需要 volcengine SDK，请运行：\n"
                "  pip install 'volcengine-python-sdk[ark]'"
            )
        self.client = Ark(api_key=api_key)
        self.model = model

    def generate(self, conversation, max_retries=8, retry_delay=5.0):
        messages = [{"role": c["role"], "content": c["content"]} for c in conversation]
        for attempt in range(max_retries):
            try:
                response = self.client.batch.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                usage = {}
                if response.usage:
                    cached = 0
                    if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                        "cached_tokens": cached,
                    }
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                return content, reasoning, usage
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[BatchAPIAgent retry {attempt+1}/{max_retries}] {e}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)


class ContextCachingAPIAgent:
    """使用火山引擎 ARK 上下文缓存 API 的 LLM Agent。

    每局 episode 开始时将对话前缀（系统提示词 + Agent 问候）缓存到服务端，
    后续每轮只发送前缀之后的消息，节省系统提示词的重复传输 token。

    Token 计数说明：
    - prompt_tokens 返回值 = 实际发送 token + cached_tokens（语义总量，与标准模式对齐）
    - cached_tokens 单独返回，作为缓存命中效率指标
    """

    def __init__(self, api_key, base_url, model, context_ttl=3600):
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError(
                "上下文缓存模式需要 volcengine SDK，请运行：\n"
                "  pip install 'volcengine-python-sdk[ark]'"
            )
        self._ark_cls = Ark
        self.client = Ark(api_key=api_key, base_url=base_url)
        self.model = model
        self.context_ttl = context_ttl

    def create_episode_context(self, prefix_messages, max_retries=3, retry_delay=5.0):
        """将前缀消息创建为 common_prefix 上下文缓存，返回 context_id。"""
        messages = [{"role": m["role"], "content": m["content"]} for m in prefix_messages]
        for attempt in range(max_retries):
            try:
                cache = self.client.context.create(
                    model=self.model,
                    messages=messages,
                    mode="common_prefix",
                    ttl=self.context_ttl,
                )
                return cache.id
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[ContextCache create retry {attempt+1}/{max_retries}] {e}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)

    def generate(self, conversation, context_id=None, cached_prefix_len=0,
                 max_retries=8, retry_delay=5.0):
        """生成回复。

        context_id 不为 None 时，只发送 conversation[cached_prefix_len:] 并附带 context_id。
        context_id 为 None 时退化为标准完整发送模式。
        返回 (content, reasoning_content, usage_dict)，usage_dict 含 cached_tokens。
        """
        if context_id is not None:
            messages = [{"role": c["role"], "content": c["content"]}
                        for c in conversation[cached_prefix_len:]]
        else:
            messages = [{"role": c["role"], "content": c["content"]} for c in conversation]

        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                )
                if context_id is not None:
                    kwargs["context_id"] = context_id

                response = self.client.chat.completions.create(**kwargs)

                usage = {}
                if response.usage:
                    raw_pt = response.usage.prompt_tokens or 0
                    raw_ct = response.usage.completion_tokens or 0
                    cached = 0
                    if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    usage = {
                        # 语义总 prompt token = 实际发送 + 缓存命中（与标准模式数值一致）
                        "prompt_tokens": raw_pt + cached,
                        "completion_tokens": raw_ct,
                        "cached_tokens": cached,
                    }

                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                return content, reasoning, usage

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[ContextCachingAgent retry {attempt+1}/{max_retries}] {e}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)


def eval_one(data_idx, env_args, agent, max_rounds, memory):
    """Evaluate a single sample. Thread-safe: each call creates its own EnvClient."""
    sample_t0 = time.time()   # 样本总耗时从此刻开始计时（含 env + memory + LLM）
    client = AlfWorldEnvClient(**env_args)
    client.reset(data_idx)
    state = client.observe()

    conversation = [
        {"role": "user",      "content": client.conversation_start[0]["value"], "reasoning_content": None},
        {"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None},
        {"role": "user",      "content": state,                                  "reasoning_content": None},
    ]

    # Memory inject: retrieve relevant past experience and inject into system prompt.
    inject_stats = MemoryCallStats()
    if memory is not None:
        conversation, inject_stats = memory.inject(conversation)

    # Context caching: cache conversation[:2] (system prompt + greeting) per episode.
    context_id = None
    cached_prefix_len = 0
    if isinstance(agent, ContextCachingAPIAgent):
        try:
            context_id = agent.create_episode_context(conversation[:2])
            cached_prefix_len = 2
        except Exception as e:
            print(f"[ContextCache] idx={data_idx} 创建缓存失败，降级为标准模式: {e}")
            context_id = None
            cached_prefix_len = 0

    total_pt = 0
    total_ct = 0
    total_cached = 0
    total_llm_latency = 0.0   # 纯 LLM API 调用耗时（不含 env 交互和 memory）
    reward = 0.0
    rounds = 0

    for _ in range(max_rounds):
        t0 = time.time()
        if context_id is not None:
            text, reasoning, usage = agent.generate(
                conversation, context_id=context_id, cached_prefix_len=cached_prefix_len
            )
        else:
            text, reasoning, usage = agent.generate(conversation)
        total_llm_latency += time.time() - t0
        total_pt += usage.get("prompt_tokens", 0)
        total_ct += usage.get("completion_tokens", 0)
        total_cached += usage.get("cached_tokens", 0)

        conversation.append({"role": "assistant", "content": text, "reasoning_content": reasoning})

        step_output = client.step(text)
        conversation.append({"role": "user", "content": client.observe(), "reasoning_content": None})

        reward = step_output.reward
        rounds += 1
        if step_output.done:
            break

    # Memory update: store trajectory in memory bank.
    update_stats = MemoryCallStats()
    if memory is not None:
        update_stats = memory.update(conversation, data_idx, reward=reward)

    sample_latency = time.time() - sample_t0   # 样本总墙钟耗时

    return data_idx, conversation, reward, total_pt, total_ct, total_cached, sample_latency, total_llm_latency, rounds, inject_stats, update_stats


def write_summary_txt(path, task, results, total_latency_wall, memory_type):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(results)
    done = [r for r in results if 'error' not in r]
    total_score   = sum(r['reward'] for r in done)
    total_success = sum(r['success'] for r in done)
    agent_pt      = sum(r.get('agent_prompt_tokens',    r.get('prompt_tokens', 0)) for r in done)
    agent_ct      = sum(r.get('agent_completion_tokens', r.get('completion_tokens', 0)) for r in done)
    total_lat     = sum(r.get('latency', 0.0) for r in done)
    total_llm_lat = sum(r.get('llm_latency', 0.0) for r in done)
    total_rounds  = sum(r.get('num_rounds', 0) for r in done)
    avg_rounds    = total_rounds / len(done) if done else 0
    total_cached  = sum(r.get('cached_tokens', 0) for r in done)

    # Memory stats aggregates
    inj_it  = sum(r.get('inject_input_tokens', 0) for r in done)
    inj_ot  = sum(r.get('inject_output_tokens', 0) for r in done)
    inj_et  = sum(r.get('inject_embedding_tokens', 0) for r in done)
    inj_ct  = sum(r.get('inject_cached_tokens', 0) for r in done)
    inj_lat = sum(r.get('inject_latency', 0.0) for r in done)
    upd_it  = sum(r.get('update_input_tokens', 0) for r in done)
    upd_ot  = sum(r.get('update_output_tokens', 0) for r in done)
    upd_et  = sum(r.get('update_embedding_tokens', 0) for r in done)
    upd_ct  = sum(r.get('update_cached_tokens', 0) for r in done)
    upd_lat = sum(r.get('update_latency', 0.0) for r in done)
    mem_cached = inj_ct + upd_ct

    mem_it   = inj_it + upd_it
    mem_ot   = inj_ot + upd_ot
    total_pt = agent_pt + mem_it
    total_ct = agent_ct + mem_ot
    total_tt = total_pt + total_ct
    nd = len(done) or 1

    lines = [
        "=" * 60,
        f"Evaluation Summary: {task}",
        f"Memory Type: {memory_type}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "[Performance]",
        f"  Num Samples:      {n}",
        f"  Completed:        {len(done)}",
        f"  Avg Score:        {total_score/n:.4f}" if n else "  Avg Score: N/A",
        f"  Success Rate:     {total_success/n:.4f} ({total_success}/{n})" if n else "  Success Rate: N/A",
        "",
        "[Agent Token Usage]",
        f"  Agent Prompt Tokens:      {agent_pt:,}",
        f"  Agent Completion Tokens:  {agent_ct:,}",
        "[Memory Token Usage]",
        f"  Memory Input Tokens:      {mem_it:,}",
        f"  Memory Output Tokens:     {mem_ot:,}",
        "[Total Token Usage (Agent + Memory)]",
        f"  Total Prompt Tokens:      {total_pt:,}",
        f"  Total Completion Tokens:  {total_ct:,}",
        f"  Total Tokens:             {total_tt:,}",
        f"  Avg Tokens/Sample:        {total_tt//nd:,}",
        f"  Total Cached Tokens (agent):  {total_cached:,}",
        f"  Total Cached Tokens (memory): {mem_cached:,}",
        f"  Total Sample Latency:     {total_lat:.1f}s",
        f"  Avg Sample Latency:       {total_lat/nd:.2f}s",
        f"  Total LLM-Only Latency:   {total_llm_lat:.1f}s",
        f"  Avg LLM-Only Latency:     {total_llm_lat/nd:.2f}s",
        f"  Avg Rounds/Sample:        {avg_rounds:.2f}",
        f"  Wall-clock Time:          {total_latency_wall:.1f}s",
        "",
        "[Memory — Inject]",
        f"  Total Input Tokens:       {inj_it:,}",
        f"  Total Output Tokens:      {inj_ot:,}",
        f"  Total Embedding Tokens:   {inj_et:,}",
        f"  Total Cached Tokens:      {inj_ct:,}",
        f"  Total Latency:            {inj_lat:.1f}s",
        f"  Avg Latency/Sample:       {inj_lat/nd:.2f}s",
        "",
        "[Memory — Update]",
        f"  Total Input Tokens:       {upd_it:,}",
        f"  Total Output Tokens:      {upd_ot:,}",
        f"  Total Embedding Tokens:   {upd_et:,}",
        f"  Total Cached Tokens:      {upd_ct:,}",
        f"  Total Latency:            {upd_lat:.1f}s",
        f"  Avg Latency/Sample:       {upd_lat/nd:.2f}s",
        "",
        "[Per-Sample Results]",
    ]
    for r in sorted(results, key=lambda x: x['data_idx']):
        if 'error' in r:
            lines.append(f"  data_idx={r['data_idx']:<5}  ERROR: {r['error']}")
        else:
            lines.append(
                f"  data_idx={r['data_idx']:<5}  reward={r['reward']:.3f}  "
                f"success={r['success']}  tokens={r.get('total_tokens', 0):>6}  "
                f"latency={r.get('latency', 0):.2f}s  rounds={r.get('num_rounds', '?')}  "
                f"inj_emb={r.get('inject_embedding_tokens', 0)}  upd_in={r.get('update_input_tokens', 0)}"
            )
    lines += ["", "=" * 60, ""]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--start_idx', type=int, default=2420)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--max_rounds', type=int, default=20)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--summary_txt', type=str, default='')
    # Memory args
    parser.add_argument('--memory_type', type=str, default='mem0',
                        help="Memory backend type (default: mem0)")
    parser.add_argument('--memory_config', type=str, default='',
                        help="Path to JSON file with memory backend kwargs. "
                             "Supports ${VAR} placeholders expanded from environment variables.")
    parser.add_argument('--readonly_memory', action='store_true',
                        help="Retrieve memories but do not update the memory store")
    parser.add_argument('--no_memory', action='store_true',
                        help="Disable memory entirely")
    parser.add_argument('--memory_load_args', type=str, default='',
                        help="JSON string of kwargs passed to memory.load_from_disk()")
    parser.add_argument('--indices', type=str, default='',
                        help="JSON array of data indices to evaluate, e.g. '[2423,2426,2429]'. "
                             "When provided, overrides --start_idx and --num_samples.")
    parser.add_argument('--api_mode', type=str, default='standard',
                        choices=['standard', 'context_cache', 'batch'],
                        help="LLM API 调用模式: standard（标准）、context_cache（上下文缓存）"
                             "或 batch（批量 API，需要 volcengine-python-sdk[ark]）")
    parser.add_argument('--model', type=str, default='',
                        help="覆盖推理模型名称。设置后 standard/context_cache 模式改用 "
                             "OPENAI_API_KEY/OPENAI_BASE_URL；不设则沿用 deepseek 默认行为。")
    args = parser.parse_args()

    env_server_base = f"http://localhost:{args.port}"
    output_dir = args.output_dir or os.path.join(BASE_DIR, 'output', 'alfworld_with_memory')
    os.makedirs(output_dir, exist_ok=True)

    # Resolve the list of indices to evaluate.
    if args.indices:
        all_indices = json.loads(args.indices)
    else:
        all_indices = list(range(args.start_idx, args.start_idx + args.num_samples))

    # data_len must be > the maximum index so the env client accepts it.
    data_len = max(all_indices) + 1 if all_indices else args.start_idx + args.num_samples

    if args.model:
        api_key  = os.environ.get('OPENAI_API_KEY', '')
        base_url = os.environ.get('OPENAI_BASE_URL', '')
        model    = args.model
    else:
        api_key  = os.environ.get('DEEPSEEK_API_KEY', '')
        base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
        model    = 'deepseek-v3-2-251201'

    env_args = {
        'env_server_base': env_server_base,
        'data_len': data_len,
        'timeout': 300,
    }

    if args.api_mode == 'context_cache':
        agent = ContextCachingAPIAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    elif args.api_mode == 'batch':
        batch_model = os.environ.get('BATCH_MODEL', '')
        agent = BatchAPIAgent(
            api_key=api_key,
            model=batch_model,
        )
    else:
        agent = InstrumentedAPIAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    # Build memory backend.
    memory = None
    memory_label = "none"
    if not args.no_memory:
        memory_kwargs = {}
        if args.memory_config:
            with open(args.memory_config) as f:
                raw = f.read()
            # Expand ${VAR} placeholders from environment variables.
            raw = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ''), raw)
            memory_kwargs = json.loads(raw)
        memory = create_memory(
            args.memory_type,
            read_only=args.readonly_memory,
            **memory_kwargs,
        )
        if args.memory_load_args:
            load_kwargs = json.loads(args.memory_load_args)
            memory.load_from_disk(**load_kwargs)
        memory_label = args.memory_type + (" (read-only)" if args.readonly_memory else "")

    idx_range_str = (
        f"explicit list ({len(all_indices)} indices)"
        if args.indices
        else f"idx {args.start_idx} ~ {args.start_idx + args.num_samples - 1}"
    )
    print(f"\n{'='*60}")
    print(f"Task:        alfworld (with memory)")
    print(f"API mode:    {args.api_mode}")
    print(f"Memory:      {memory_label}")
    print(f"Server:      {env_server_base}")
    print(f"Samples:     {idx_range_str}")
    print(f"Max rounds:  {args.max_rounds}")
    print(f"Parallel:    {args.parallel}")
    print(f"Output:      {output_dir}")
    print(f"{'='*60}\n")

    # Skip already cached samples.
    pending = []
    cached_results = []
    for data_idx in all_indices:
        out_file = os.path.join(output_dir, f"alfworld_{data_idx}.json")
        if os.path.exists(out_file):
            try:
                with open(out_file) as f:
                    saved = json.load(f)
                _c_apt = saved.get('agent_prompt_tokens',    saved.get('prompt_tokens', 0))
                _c_act = saved.get('agent_completion_tokens', saved.get('completion_tokens', 0))
                _c_mit = saved.get('memory_input_tokens',
                             saved.get('inject_input_tokens', 0) + saved.get('update_input_tokens', 0))
                _c_mot = saved.get('memory_output_tokens',
                             saved.get('inject_output_tokens', 0) + saved.get('update_output_tokens', 0))
                cached_results.append({
                    'data_idx':                  data_idx,
                    'reward':                    saved.get('reward', 0),
                    'success':                   saved.get('success', 0),
                    'agent_prompt_tokens':        _c_apt,
                    'agent_completion_tokens':    _c_act,
                    'memory_input_tokens':        _c_mit,
                    'memory_output_tokens':       _c_mot,
                    'total_prompt_tokens':        _c_apt + _c_mit,
                    'total_completion_tokens':    _c_act + _c_mot,
                    'total_tokens':               _c_apt + _c_act + _c_mit + _c_mot,
                    'cached_tokens':             saved.get('cached_tokens', 0),
                    'latency':                   saved.get('latency', 0.0),
                    'llm_latency':               saved.get('llm_latency', saved.get('latency', 0.0)),
                    'num_rounds':                saved.get('num_rounds', 0),
                    'inject_input_tokens':       saved.get('inject_input_tokens', 0),
                    'inject_output_tokens':      saved.get('inject_output_tokens', 0),
                    'inject_embedding_tokens':   saved.get('inject_embedding_tokens', 0),
                    'inject_cached_tokens':      saved.get('inject_cached_tokens', 0),
                    'inject_latency':            saved.get('inject_latency', 0.0),
                    'update_input_tokens':       saved.get('update_input_tokens', 0),
                    'update_output_tokens':      saved.get('update_output_tokens', 0),
                    'update_embedding_tokens':   saved.get('update_embedding_tokens', 0),
                    'update_cached_tokens':      saved.get('update_cached_tokens', 0),
                    'update_latency':            saved.get('update_latency', 0.0),
                })
                print(f"[cached] data_idx={data_idx}  reward={saved.get('reward', 0)}")
                continue
            except Exception:
                pass
        pending.append(data_idx)

    results = list(cached_results)
    wall_start = time.time()

    if pending:
        print(f"Running {len(pending)} samples with parallelism={args.parallel}...\n")
        success_count = 0
        pbar = tqdm(total=len(pending), unit="sample", dynamic_ncols=True)

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(eval_one, idx, env_args, agent, args.max_rounds, memory): idx
                for idx in pending
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    (data_idx, conversation, reward,
                     pt, ct, cached, latency, llm_latency, rounds,
                     inject_stats, update_stats) = future.result()

                    success = 1 if reward >= 1 else 0
                    if success:
                        success_count += 1

                    _r_mit = inject_stats.input_tokens + update_stats.input_tokens
                    _r_mot = inject_stats.output_tokens + update_stats.output_tokens
                    record = {
                        'data_idx':                  data_idx,
                        'reward':                    reward,
                        'success':                   success,
                        'agent_prompt_tokens':        pt,
                        'agent_completion_tokens':    ct,
                        'memory_input_tokens':        _r_mit,
                        'memory_output_tokens':       _r_mot,
                        'total_prompt_tokens':        pt + _r_mit,
                        'total_completion_tokens':    ct + _r_mot,
                        'total_tokens':               pt + ct + _r_mit + _r_mot,
                        'cached_tokens':             cached,
                        'latency':                   latency,
                        'llm_latency':               llm_latency,
                        'num_rounds':                rounds,
                        'inject_input_tokens':       inject_stats.input_tokens,
                        'inject_output_tokens':      inject_stats.output_tokens,
                        'inject_embedding_tokens':   inject_stats.embedding_tokens,
                        'inject_cached_tokens':      inject_stats.cached_tokens,
                        'inject_latency':            inject_stats.latency,
                        'update_input_tokens':       update_stats.input_tokens,
                        'update_output_tokens':      update_stats.output_tokens,
                        'update_embedding_tokens':   update_stats.embedding_tokens,
                        'update_cached_tokens':      update_stats.cached_tokens,
                        'update_latency':            update_stats.latency,
                    }
                    results.append(record)

                    out_file = os.path.join(output_dir, f"alfworld_{data_idx}.json")
                    with open(out_file, 'w') as f:
                        json.dump({
                            'item_id':                 f'alfworld_{data_idx}',
                            'reward':                  reward,
                            'success':                 success,
                            'agent_prompt_tokens':     pt,
                            'agent_completion_tokens': ct,
                            'memory_input_tokens':     _r_mit,
                            'memory_output_tokens':    _r_mot,
                            'total_prompt_tokens':     pt + _r_mit,
                            'total_completion_tokens': ct + _r_mot,
                            'total_tokens':            pt + ct + _r_mit + _r_mot,
                            'cached_tokens':           cached,
                            'latency':                 latency,
                            'llm_latency':             llm_latency,
                            'num_rounds':              rounds,
                            'inject_input_tokens':     inject_stats.input_tokens,
                            'inject_output_tokens':    inject_stats.output_tokens,
                            'inject_embedding_tokens': inject_stats.embedding_tokens,
                            'inject_cached_tokens':    inject_stats.cached_tokens,
                            'inject_latency':          inject_stats.latency,
                            'update_input_tokens':     update_stats.input_tokens,
                            'update_output_tokens':    update_stats.output_tokens,
                            'update_embedding_tokens': update_stats.embedding_tokens,
                            'update_cached_tokens':    update_stats.cached_tokens,
                            'update_latency':          update_stats.latency,
                            'conversations':           conversation,
                        }, f, ensure_ascii=False, indent=2)

                    completed_so_far = len(results) - len(cached_results)
                    total_done = len([r for r in results if 'error' not in r])
                    cur_success_rate = sum(r['success'] for r in results if 'error' not in r) / max(total_done, 1)
                    pbar.set_postfix({
                        'succ': f"{success_count}/{completed_so_far}",
                        'rate': f"{cur_success_rate:.1%}",
                        'lat':  f"{latency:.0f}s",
                        'inj_emb': inject_stats.embedding_tokens,
                    })
                    pbar.update(1)
                    tqdm.write(
                        f"  idx={data_idx}  reward={reward:.3f}  success={success}"
                        f"  tokens={pt+ct}  lat={latency:.1f}s  llm={llm_latency:.1f}s  rounds={rounds}"
                        f"  inj_emb={inject_stats.embedding_tokens}  upd_in={update_stats.input_tokens}"
                    )

                except Exception as e:
                    results.append({'data_idx': idx, 'reward': 0, 'success': 0, 'error': str(e)})
                    pbar.update(1)
                    tqdm.write(f"  idx={idx}  ERROR: {e}")

        pbar.close()

    wall_elapsed = time.time() - wall_start

    n = len(results)
    done = [r for r in results if 'error' not in r]
    total_success = sum(r['success'] for r in done)
    total_score   = sum(r['reward'] for r in done)
    agent_pt      = sum(r.get('agent_prompt_tokens',    r.get('prompt_tokens', 0)) for r in done)
    agent_ct      = sum(r.get('agent_completion_tokens', r.get('completion_tokens', 0)) for r in done)
    total_cached  = sum(r.get('cached_tokens', 0) for r in done)
    total_lat     = sum(r.get('latency', 0.0) for r in done)
    total_llm_lat = sum(r.get('llm_latency', 0.0) for r in done)
    nd = len(done) or 1

    inj_it  = sum(r.get('inject_input_tokens', 0) for r in done)
    inj_ot  = sum(r.get('inject_output_tokens', 0) for r in done)
    inj_et  = sum(r.get('inject_embedding_tokens', 0) for r in done)
    inj_ct  = sum(r.get('inject_cached_tokens', 0) for r in done)
    inj_lat = sum(r.get('inject_latency', 0.0) for r in done)
    upd_it  = sum(r.get('update_input_tokens', 0) for r in done)
    upd_ot  = sum(r.get('update_output_tokens', 0) for r in done)
    upd_et  = sum(r.get('update_embedding_tokens', 0) for r in done)
    upd_ct  = sum(r.get('update_cached_tokens', 0) for r in done)
    upd_lat = sum(r.get('update_latency', 0.0) for r in done)
    mem_cached = inj_ct + upd_ct

    mem_it   = inj_it + upd_it
    mem_ot   = inj_ot + upd_ot
    total_pt = agent_pt + mem_it
    total_ct = agent_ct + mem_ot
    total_tt = total_pt + total_ct

    print(f"\n{'='*60}")
    print(f"Results for alfworld+memory ({n} samples, {len(done)} completed)")
    print(f"  Avg Score:              {total_score/n:.4f}")
    print(f"  Success Rate:           {total_success/n:.4f} ({total_success}/{n})")
    print(f"  --- Agent tokens ---")
    print(f"  Agent Prompt Tokens:    {agent_pt:,}")
    print(f"  Agent Completion Tokens:{agent_ct:,}")
    print(f"  --- Memory tokens ---")
    print(f"  Memory Input Tokens:    {mem_it:,}")
    print(f"  Memory Output Tokens:   {mem_ot:,}")
    print(f"  --- Total (agent + memory) ---")
    print(f"  Total Prompt Tokens:    {total_pt:,}")
    print(f"  Total Completion Tokens:{total_ct:,}")
    print(f"  Total Tokens:           {total_tt:,}")
    print(f"  Avg Tokens/Sample:      {total_tt//nd:,}")
    print(f"  Total Cached Tokens:    {total_cached:,}  (agent)  /  {mem_cached:,}  (memory)")
    print(f"  Total Sample Latency:   {total_lat:.1f}s  (含 env + memory + LLM)")
    print(f"  Avg Sample Latency:     {total_lat/nd:.2f}s")
    print(f"  Total LLM-Only Latency: {total_llm_lat:.1f}s")
    print(f"  Avg LLM-Only Latency:   {total_llm_lat/nd:.2f}s")
    print(f"  Avg Rounds/Sample:      {sum(r.get('num_rounds',0) for r in done)/nd:.2f}")
    print(f"  Wall-clock Time:        {wall_elapsed:.1f}s")
    print(f"  --- Memory inject ---")
    print(f"  Input Tokens:           {inj_it:,}  Output: {inj_ot:,}  Emb: {inj_et:,}  Cached: {inj_ct:,}")
    print(f"  Total Latency:          {inj_lat:.1f}s  Avg: {inj_lat/nd:.2f}s")
    print(f"  --- Memory update ---")
    print(f"  Input Tokens:           {upd_it:,}  Output: {upd_ot:,}  Emb: {upd_et:,}  Cached: {upd_ct:,}")
    print(f"  Total Latency:          {upd_lat:.1f}s  Avg: {upd_lat/nd:.2f}s")
    print(f"{'='*60}")

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'task': 'alfworld',
            'memory_type': memory_label,
            'module': 'agentenv.envs.alfworld (with memory)',
            'num_samples': n,
            'avg_score': total_score / n if n else 0,
            'success_rate': total_success / n if n else 0,
            'agent_prompt_tokens': agent_pt,
            'agent_completion_tokens': agent_ct,
            'memory_input_tokens': mem_it,
            'memory_output_tokens': mem_ot,
            'total_prompt_tokens': total_pt,
            'total_completion_tokens': total_ct,
            'total_tokens': total_tt,
            'total_cached_tokens': total_cached,
            'total_agent_cached_tokens': total_cached,
            'total_memory_cached_tokens': mem_cached,
            'avg_tokens_per_sample': total_tt // nd,
            'total_sample_latency': total_lat,
            'avg_sample_latency_per_sample': total_lat / nd,
            'total_llm_latency': total_llm_lat,
            'avg_llm_latency_per_sample': total_llm_lat / nd,
            'avg_rounds_per_sample': sum(r.get('num_rounds', 0) for r in done) / nd,
            'wall_clock_seconds': wall_elapsed,
            'memory_inject': {
                'total_input_tokens': inj_it,
                'total_output_tokens': inj_ot,
                'total_embedding_tokens': inj_et,
                'total_cached_tokens': inj_ct,
                'total_latency': inj_lat,
                'avg_latency_per_sample': inj_lat / nd,
            },
            'memory_update': {
                'total_input_tokens': upd_it,
                'total_output_tokens': upd_ot,
                'total_embedding_tokens': upd_et,
                'total_cached_tokens': upd_ct,
                'total_latency': upd_lat,
                'avg_latency_per_sample': upd_lat / nd,
            },
            'results': sorted(results, key=lambda x: x['data_idx']),
        }, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON: {summary_file}")

    summary_txt = args.summary_txt or os.path.join(
        output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    txt_path = write_summary_txt(summary_txt, 'alfworld', results, wall_elapsed, memory_label)
    print(f"Summary TXT:  {txt_path}")


if __name__ == '__main__':
    main()
