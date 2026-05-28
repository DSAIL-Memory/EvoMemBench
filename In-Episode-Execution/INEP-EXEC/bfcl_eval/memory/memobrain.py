"""MemoBrain memory backend — LLM-driven reasoning-graph compression.

Wraps the MemoBrain package (EvoMemBench-Memory-Systems/MemoBrain) as a
compression-type Memory backend.  Unlike long-term memory backends (amem,
mem0, …) this class does NOT retrieve stored facts across samples.  Instead
it maintains a per-sample reasoning graph: new messages are memorized into the
graph after each tool-call step, and recall() compresses the graph via LLM
flush/fold when the context budget is exceeded.

LLM calls are routed through the pretrained MemoBrain-14B model served locally
by vLLM on an OpenAI-compatible endpoint (default: http://localhost:8001/v1).
Start the server with run_memobrain_server.sh before running evaluations.
Override the endpoint with MEMOBRAIN_VLLM_URL / MEMOBRAIN_MODEL env vars.
"""
from __future__ import annotations

import asyncio
import json

from bfcl_eval.memory.base import Memory


# ─────────────────────────────────────────────────────────────────────────────
# vLLM OpenAI-compatible async shim (primary path)
# ─────────────────────────────────────────────────────────────────────────────

def _make_vllm_completion(base_url: str, model: str):
    """Return an async _create_completion that calls a local vLLM server."""
    import time
    from bfcl_eval.memory import metrics

    async def _create_completion(messages, stream=False):
        import openai
        client = openai.AsyncOpenAI(api_key="placeholder", base_url=base_url)
        t0 = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192,
            top_p=0.8,
        )
        dt = time.time() - t0
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(response.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "completion_tokens", 0) or 0),
        )
        return response

    return _create_completion


# ─────────────────────────────────────────────────────────────────────────────
# Message format helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_simplified(messages: list[dict]) -> list[dict]:
    """Convert BFCL FC-mode messages to user/assistant pairs for MemoBrain.

    MemoBrain's _group_pairs only processes user/assistant roles.
    - assistant: preserve content; if empty, stringify tool_calls
    - tool:      becomes a user message (tool result)
    - user/system: pass through as-is
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content") or ""
            if not content:
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    content = json.dumps(tool_calls, ensure_ascii=False)
            if content:
                result.append({"role": "assistant", "content": content})
        elif role == "tool":
            content = msg.get("content", "")
            if content:
                result.append({"role": "user", "content": content})
        elif role in ("user", "system"):
            result.append(msg)
    return result


def _format_recalled(messages: list[dict]) -> str:
    """Format MemoBrain's recall() output as a text summary string.

    Matches the [CONVERSATION HISTORY SUMMARY] format used by memagent so the
    rest of the inference loop handles both compression types identically.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            parts.append(f"[{role}]: {content}")
    return "[CONVERSATION HISTORY SUMMARY]\n" + "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MemoBrainMemory
# ─────────────────────────────────────────────────────────────────────────────

class MemoBrainMemory(Memory):
    """Working-memory backend backed by MemoBrain's LLM reasoning graph.

    Memorize/recall operations are handled by the pretrained MemoBrain-14B
    model served locally via vLLM (OpenAI-compatible API on port 8001).
    No DashScope or vector store needed.

    Lifecycle within a single sample:
      1. init_brain(task_text)         — called at turn 0
      2. memorize(new_msgs)            — called after every tool-call step
      3. recall() -> str               — called when context exceeds budget
         Returns a [CONVERSATION HISTORY SUMMARY] string that replaces
         _compressed_messages (same format as memagent).
    """

    def __init__(
        self,
        *,
        vllm_base_url: str = "http://localhost:8001/v1",
        vllm_model: str = "TommyChien/MemoBrain-14B",
        storage_dir=None,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ):
        super().__init__(clear=clear, readonly=readonly, **kwargs)

        try:
            from memobrain import MemoBrain
        except ImportError as exc:
            raise ImportError(
                "memobrain is required for --memory-type memobrain. "
                "Install it from EvoMemBench-Memory-Systems/MemoBrain/."
            ) from exc

        brain = MemoBrain(api_key="placeholder", base_url="placeholder", model_name=vllm_model)
        brain._create_completion = _make_vllm_completion(vllm_base_url, vllm_model)
        self._brain = brain
        self._vllm_base_url = vllm_base_url
        self._vllm_model = vllm_model

    # ── Memory ABC ────────────────────────────────────────────────────────────

    def utilize(self, user_text: str) -> str:
        return ""

    def update(self, trajectory: list[dict]) -> None:
        pass

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "MemoBrainMemory":
        return cls(**backend_kwargs)

    # ── MemoBrain-specific methods (called from base_handler) ─────────────────

    def init_brain(self, task_text: str) -> None:
        """Initialize (or re-initialize) the reasoning graph for a new sample."""
        from memobrain import MemoBrain
        brain = MemoBrain(api_key="placeholder", base_url="placeholder", model_name=self._vllm_model)
        brain._create_completion = _make_vllm_completion(self._vllm_base_url, self._vllm_model)
        self._brain = brain
        self._brain.init_memory(task_text)

    def memorize(self, messages: list[dict]) -> None:
        """Feed new BFCL messages into the reasoning graph (sync wrapper)."""
        simplified = _to_simplified(messages)
        if not simplified:
            return
        try:
            asyncio.run(self._brain.memorize(simplified))
        except Exception as exc:
            print(f"[MemoBrain] memorize failed: {type(exc).__name__}: {exc}")

    def recall(self) -> str:
        """Flush+fold the reasoning graph and return a summary string (sync wrapper)."""
        recalled = asyncio.run(self._brain.recall())
        return _format_recalled(recalled)
