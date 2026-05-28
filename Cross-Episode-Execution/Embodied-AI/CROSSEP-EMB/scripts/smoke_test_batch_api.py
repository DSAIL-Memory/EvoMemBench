#!/usr/bin/env python3
"""
冒烟脚本：验证 BATCH_MODEL / DEEPSEEK_API_KEY 从 .env 正确读取，
并向火山引擎 ARK Batch API 发送一条测试请求，打印完整响应和 token 消耗。

用法（在项目根目录下运行）：
    python scripts/smoke_test_batch_api.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# Step 1: 读取 .env
# ──────────────────────────────────────────────────────────────
env_path = os.path.join(BASE_DIR, ".env")
print(f"[1] 读取 .env: {env_path}")
if not os.path.exists(env_path):
    print("    ERROR: .env 文件不存在！")
    sys.exit(1)

with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key, val)

# ──────────────────────────────────────────────────────────────
# Step 2: 检查关键变量
# ──────────────────────────────────────────────────────────────
print("\n[2] 环境变量检查")

BATCH_MODEL   = os.environ.get("BATCH_MODEL", "")
DEEPSEEK_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
ARK_KEY       = os.environ.get("ARK_API_KEY", "")       # 备用 key 名称

def mask(s):
    """只显示前8位和后4位，中间用 *** 遮盖。"""
    if len(s) <= 12:
        return s
    return s[:8] + "***" + s[-4:]

ok = True

if BATCH_MODEL:
    print(f"    BATCH_MODEL      = {BATCH_MODEL}")
    if not BATCH_MODEL.startswith("ep-bi-"):
        print("    WARNING: BATCH_MODEL 不以 'ep-bi-' 开头，可能不是批量推理端点！")
else:
    print("    ERROR: BATCH_MODEL 未设置！")
    ok = False

# BatchAPIAgent 读的是 DEEPSEEK_API_KEY
api_key = DEEPSEEK_KEY or ARK_KEY
key_name = "DEEPSEEK_API_KEY" if DEEPSEEK_KEY else ("ARK_API_KEY" if ARK_KEY else "")

if api_key:
    print(f"    {key_name:<20} = {mask(api_key)}")
    if not (api_key.startswith("ark-") or len(api_key) > 20):
        print("    WARNING: API key 格式看起来不像火山引擎 ARK key（通常以 'ark-' 开头）")
else:
    print("    ERROR: DEEPSEEK_API_KEY 和 ARK_API_KEY 均未设置！")
    ok = False

if not ok:
    print("\n    变量缺失，终止测试。请检查 .env 文件。")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 3: 导入 SDK
# ──────────────────────────────────────────────────────────────
print("\n[3] 导入 volcenginesdkarkruntime")
try:
    from volcenginesdkarkruntime import Ark
    print("    OK: SDK 已安装")
except ImportError:
    print("    ERROR: 未安装，请运行：pip install 'volcengine-python-sdk[ark]'")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 4: 发送测试请求
# ──────────────────────────────────────────────────────────────
print(f"\n[4] 向 Batch API 发送测试请求（model={BATCH_MODEL}）...")

client = Ark(api_key=api_key)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Reply with exactly: BATCH_OK"},
]

t0 = time.time()
try:
    response = client.batch.chat.completions.create(
        model=BATCH_MODEL,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
    )
    elapsed = time.time() - t0
except Exception as e:
    elapsed = time.time() - t0
    print(f"    ERROR: 请求失败（{elapsed:.2f}s）: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 5: 解析并打印结果
# ──────────────────────────────────────────────────────────────
print(f"    请求耗时: {elapsed:.2f}s")

content = response.choices[0].message.content if response.choices else "(empty)"
print(f"    模型回复: {content!r}")

if response.usage:
    u = response.usage
    pt = u.prompt_tokens or 0
    ct = u.completion_tokens or 0
    cached = 0
    if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
    print(f"    Token 消耗:")
    print(f"      prompt_tokens     = {pt}")
    print(f"      completion_tokens = {ct}")
    print(f"      cached_tokens     = {cached}  (已计入 prompt_tokens)")
    print(f"      total             = {pt + ct}")
else:
    print("    WARNING: response.usage 为空，无法获取 token 消耗！")

# finish_reason
fr = response.choices[0].finish_reason if response.choices else "unknown"
print(f"    finish_reason: {fr}")

print("\n[5] 冒烟测试通过 ✓")
print("    如果火山引擎控制台看不到消耗，请在「批量推理 > 使用记录」中查看，")
print("    Batch API 的计费与普通 API 调用记录在不同入口。")
