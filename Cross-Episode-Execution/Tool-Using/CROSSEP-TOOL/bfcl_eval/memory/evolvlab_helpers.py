"""Common helpers for EvolveLab-based Memory adapters.

Provides:
  - path injection to make `EvolveLab` importable as a top-level package
  - ArkLLMCallable: wraps ArkBatchClient into the callable model interface
    expected by EvolveLab providers  (model(messages) -> obj with .content)
  - trajectory conversion utilities
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Make EvolveLab importable (adds the `ours/` directory to sys.path once)
# ──────────────────────────────────────────────────────────────────────────────
_OURS_ROOT = Path(__file__).resolve().parents[2]  # …/ours
if str(_OURS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OURS_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Stub for storage.tools.tool_wrapper (used only by SkillWeaverProvider).
# The ToolWrapper is not needed in the bfcl_eval context — we only care about
# textual skill descriptions, not wrapped callables.
# ──────────────────────────────────────────────────────────────────────────────
def _inject_tool_wrapper_stub() -> None:
    if "storage.tools.tool_wrapper" in sys.modules:
        return

    class _StubToolWrapper:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def wrap_function(self, func: Any, name: str) -> None:
            return None

        def clear_cache(self) -> None:
            pass

    storage_mod = sys.modules.get("storage") or types.ModuleType("storage")
    tools_mod = sys.modules.get("storage.tools") or types.ModuleType("storage.tools")
    tw_mod = types.ModuleType("storage.tools.tool_wrapper")
    tw_mod.ToolWrapper = _StubToolWrapper  # type: ignore[attr-defined]

    storage_mod.tools = tools_mod  # type: ignore[attr-defined]
    tools_mod.tool_wrapper = tw_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("storage", storage_mod)
    sys.modules.setdefault("storage.tools", tools_mod)
    sys.modules["storage.tools.tool_wrapper"] = tw_mod


_inject_tool_wrapper_stub()


# ──────────────────────────────────────────────────────────────────────────────
# ArkLLMCallable
# ──────────────────────────────────────────────────────────────────────────────
class _LLMResponse:
    """Minimal response object — EvolveLab providers access `.content`."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content


class ArkLLMCallable:
    """Thin callable wrapper around ArkBatchClient.

    EvolveLab providers call ``model(messages)`` and read ``response.content``.
    This class satisfies that contract while routing calls through the shared
    Ark batch client (same as the main inference path) and recording metrics.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        from bfcl_eval.memory.ark_client import ArkBatchClient

        self._client = ArkBatchClient(api_key=api_key)
        self._model_name = model_name

    def __call__(self, messages: list[dict]) -> _LLMResponse:
        from bfcl_eval.memory import metrics

        resp, dt = self._client.chat_completions(
            model=self._model_name, messages=messages
        )
        content: str = resp.choices[0].message.content or ""
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        return _LLMResponse(content)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_query_from_trajectory(trajectory: list[dict]) -> str:
    """Return the first user-turn text as the task query."""
    for msg in trajectory:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            return str(content)
    return ""


def format_trajectory_as_steps(trajectory: list[dict]) -> list[dict]:
    """Convert full_message_history to the step list expected by TrajectoryData."""
    steps: list[dict] = []
    for msg in trajectory:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(str(part))
                elif part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "tool_use":
                    parts.append(
                        f"[tool_call: {part.get('name', '')}({part.get('input', {})})]"
                    )
                elif part.get("type") == "tool_result":
                    parts.append(f"[tool_result: {part.get('content', '')}]")
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        steps.append({"type": role, "content": str(content)})
    return steps


def extract_result_from_trajectory(trajectory: list[dict]) -> str:
    """Return the last assistant-turn text as the final result."""
    for msg in reversed(trajectory):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            return str(content)
    return ""
