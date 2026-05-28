"""OpenAI-compatible chat API with token usage (used by memory extraction and optional reuse)."""

import time

from .logging_utils import log

try:
    from volcenginesdkarkruntime import Ark as _ArkClient
except ImportError:
    _ArkClient = None


def call_openai_api(client, messages, model, temperature=None, max_retries=3, retry_delay=3, response_format=None):
    """Call OpenAI-compatible API with retries.

    Automatically uses Volcengine Ark batch chat API when client is an Ark instance.

    Returns:
        (response_text, error, usage) where usage is a dict with
        prompt_tokens, completion_tokens, total_tokens (empty dict on error).
    """
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format is not None:
                kwargs["response_format"] = response_format
            if _ArkClient is not None and isinstance(client, _ArkClient):
                response = client.batch.chat.completions.create(**kwargs)
            else:
                response = client.chat.completions.create(**kwargs)
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            return response.choices[0].message.content, None, usage
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(f"   ⚠️ Call failed (attempt {attempt + 1}): {error_msg[:100]}")
                time.sleep(retry_delay)
            else:
                log(f"   ❌ Final failure: {error_msg[:200]}")
                return None, error_msg, {}
    return None, "Unknown error", {}
