#!/usr/bin/env python3
"""
Smoke-test script: verifies that BATCH_MODEL / DEEPSEEK_API_KEY are read correctly from .env,
sends one test request to the Volcano Engine ARK Batch API, and prints the full response and token usage.

Usage (run from the project root):
    python scripts/smoke_test_batch_api.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# Step 1: read .env
# ──────────────────────────────────────────────────────────────
env_path = os.path.join(BASE_DIR, ".env")
print(f"[1] Read .env: {env_path}")
if not os.path.exists(env_path):
    print("    ERROR: .env file does not exist!")
    sys.exit(1)

with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key, val)

# ──────────────────────────────────────────────────────────────
# Step 2: check key variables
# ──────────────────────────────────────────────────────────────
print("\n[2] Environment variable check")

BATCH_MODEL   = os.environ.get("BATCH_MODEL", "")
DEEPSEEK_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
ARK_KEY       = os.environ.get("ARK_API_KEY", "")       # fallback key name

def mask(s):
    """Show only the first 8 and last 4 characters, masking the middle with ***."""
    if len(s) <= 12:
        return s
    return s[:8] + "***" + s[-4:]

ok = True

if BATCH_MODEL:
    print(f"    BATCH_MODEL      = {BATCH_MODEL}")
    if not BATCH_MODEL.startswith("ep-bi-"):
        print("    WARNING: BATCH_MODEL does not start with 'ep-bi-'; it may not be a batch inference endpoint!")
else:
    print("    ERROR: BATCH_MODEL is not set!")
    ok = False

# BatchAPIAgent reads DEEPSEEK_API_KEY
api_key = DEEPSEEK_KEY or ARK_KEY
key_name = "DEEPSEEK_API_KEY" if DEEPSEEK_KEY else ("ARK_API_KEY" if ARK_KEY else "")

if api_key:
    print(f"    {key_name:<20} = {mask(api_key)}")
    if not (api_key.startswith("ark-") or len(api_key) > 20):
        print("    WARNING: API key format does not look like a Volcano Engine ARK key (usually starts with 'ark-')")
else:
    print("    ERROR: neither DEEPSEEK_API_KEY nor ARK_API_KEY is set!")
    ok = False

if not ok:
    print("\n    Missing variables; stopping the test. Please check the .env file.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 3: import SDK
# ──────────────────────────────────────────────────────────────
print("\n[3] Import volcenginesdkarkruntime")
try:
    from volcenginesdkarkruntime import Ark
    print("    OK: SDK is installed")
except ImportError:
    print("    ERROR: not installed. Please run: pip install 'volcengine-python-sdk[ark]'")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 4: send test request
# ──────────────────────────────────────────────────────────────
print(f"\n[4] Send test request to Batch API (model={BATCH_MODEL})...")

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
    print(f"    ERROR: request failed ({elapsed:.2f}s): {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Step 5: parse and print the result
# ──────────────────────────────────────────────────────────────
print(f"    request latency: {elapsed:.2f}s")

content = response.choices[0].message.content if response.choices else "(empty)"
print(f"    model reply: {content!r}")

if response.usage:
    u = response.usage
    pt = u.prompt_tokens or 0
    ct = u.completion_tokens or 0
    cached = 0
    if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
    print(f"    Token usage:")
    print(f"      prompt_tokens     = {pt}")
    print(f"      completion_tokens = {ct}")
    print(f"      cached_tokens     = {cached}  (included in prompt_tokens)")
    print(f"      total             = {pt + ct}")
else:
    print("    WARNING: response.usage is empty; token usage cannot be retrieved!")

# finish_reason
fr = response.choices[0].finish_reason if response.choices else "unknown"
print(f"    finish_reason: {fr}")

print("\n[5] Smoke test passed ✓")
print('    If usage is not visible in the Volcano Engine console, check "Batch Inference > Usage Records",')
print("    Batch API billing and standard API call records are shown in different sections.")
