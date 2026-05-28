"""MemAgent / MemoBrain memory backend stub.

Context management for these types is handled entirely by the inference loop
via LLM-based history compression — retrieve (utilize) is intentionally a no-op.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from bfcl_eval.memory.base import Memory

if TYPE_CHECKING:
    import openai

_MEMORIZE_TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain "
    "the answer to the problem, and a previous memory. Please read the provided "
    "section carefully and update the memory with the new information that helps "
    "to answer the problem. Be sure to retain all relevant details from the "
    "previous memory while adding any new, useful information.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)

_GENERIC_PROMPT = (
    "Extract and remember all important information, facts, events, "
    "and details from this section."
)

# Module-level lazy OpenAI client (thread-safe).
_client: "openai.OpenAI | None" = None


def _get_client() -> "openai.OpenAI":
    global _client
    if _client is None:
        import openai as _openai
        import httpx

        timeout = httpx.Timeout(
            float(os.environ.get("MEMAGENT_TIMEOUT", "1200")), connect=30.0
        )
        # trust_env=False avoids SOCKS/HTTP proxy variables that may not have
        # their required packages installed, since the server is always local.
        http_client = httpx.Client(timeout=timeout, trust_env=False)
        _client = _openai.OpenAI(
            base_url=os.environ.get("MEMAGENT_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("MEMAGENT_API_KEY", "EMPTY"),
            http_client=http_client,
        )
    return _client


def _serialize_messages(messages: list[dict]) -> str:
    """Flatten a BFCL message list to plain text."""
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens.

    Splits on newlines to keep message boundaries intact where possible.
    """
    from bfcl_eval.model_handler.base_handler import _count_tokens

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = _count_tokens(line)
        if current and current_tokens + line_tokens > chunk_size:
            chunks.append("\n".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append("\n".join(current))
    return chunks


class MemAgentMemory(Memory):
    """Stub for memagent/memobrain memory types.

    The inference loop compresses conversation history into the context window
    directly; this class handles the memory lifecycle hooks without retrieval.
    """

    def utilize(self, user_text: str) -> str:
        return ""

    def update(self, trajectory: list[dict]) -> None:
        pass

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "MemAgentMemory":
        return cls(**backend_kwargs)

    @staticmethod
    def compress(
        previous_memory: str,
        new_messages: list[dict],
        current_user_prompt: str,
        target_tokens: int,
    ) -> str:
        """Recurrently update previous_memory with new_messages.

        Only new_messages are chunked and fed into the MemAgent model; the
        prior summary is passed directly as the initial <memory> value so that
        repeated compressions accumulate rather than restart from scratch.
        Falls back to the suffix-keeping heuristic if the server is unreachable.
        """
        from bfcl_eval.model_handler.base_handler import _count_tokens

        problem = current_user_prompt.strip() or _GENERIC_PROMPT
        chunk_size = int(os.environ.get("MEMAGENT_CHUNK_SIZE", "5000"))
        model = os.environ.get("MEMAGENT_MODEL", "BytedTsinghua-SIA/RL-MemoryAgent-14B")
        max_new = int(os.environ.get("MEMAGENT_MAX_NEW", "1024"))

        serialized = _serialize_messages(new_messages)
        if not serialized.strip():
            return previous_memory

        chunks = _split_into_chunks(serialized, chunk_size)
        memory = previous_memory.strip() or "No previous memory"
        try:
            client = _get_client()
            for chunk in chunks:
                formatted = _MEMORIZE_TEMPLATE.format(
                    prompt=problem,
                    memory=memory,
                    chunk=chunk,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted}],
                    temperature=0.1,
                    top_p=0.95,
                    max_tokens=max_new,
                )
                memory = response.choices[0].message.content or memory
        except Exception as exc:
            print(f"[MemAgentMemory] server call failed, falling back to suffix heuristic: {exc}", file=sys.stderr)
            return _suffix_fallback(previous_memory, new_messages, target_tokens)

        # Safety guard: hard-truncate if the memory exceeds the token budget.
        if _count_tokens(memory) > target_tokens:
            words = memory.split()
            while words and _count_tokens(" ".join(words)) > target_tokens:
                words = words[: int(len(words) * 0.9) or 1]
            memory = " ".join(words)

        return memory


def _suffix_fallback(previous_memory: str, new_messages: list[dict], target_tokens: int) -> str:
    """Keep as much of new_messages as fits, prefixed with previous_memory."""
    from bfcl_eval.model_handler.base_handler import _count_tokens

    prefix = ("[PREVIOUS MEMORY]\n" + previous_memory + "\n\n") if previous_memory.strip() else ""
    prefix_tokens = _count_tokens(prefix)
    remaining = max(0, target_tokens - prefix_tokens)

    kept: list[dict] = []
    for msg in reversed(new_messages):
        candidate = [msg] + kept
        if _count_tokens(candidate) <= remaining:
            kept = candidate
        else:
            break
    parts = []
    for msg in kept:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content:
            parts.append(f"[{role}]: {str(content)}")
    return prefix + "[CONVERSATION HISTORY SUMMARY]\n" + "\n".join(parts)
