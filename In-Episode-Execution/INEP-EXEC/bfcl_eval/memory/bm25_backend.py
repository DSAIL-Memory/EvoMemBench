"""Bm25Memory — BM25-based implementation of the Memory interface.

No LLM or embedding API calls are made; retrieval is deterministic and free.
Trajectories are chunked (sentence-boundary aligned, tiktoken budgeted) and
stored as plain JSONL under storage_dir/memory_bank.jsonl.

Port of AgentGym's scripts/memory/bm25_adapter.py adapted to the BFCL
Memory interface (utilize / update / load_from_disk).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory

_UTILIZE_PREFIX = "\n\nRelevant memories from prior episodes:\n"

# ── Tokenizer (tiktoken, whitespace fallback) ────────────────────────────────

try:
    import tiktoken as _tiktoken
    _ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _token_count(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


# ── BM25-specific helpers ────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Whitespace split only — preserves case-sensitive function/param identifiers."""
    return text.split()


def _chunk_text(text: str, chunk_size: int = 1024) -> list[str]:
    """Split text into chunks of at most chunk_size tokens, on sentence boundaries."""
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
    except Exception:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if current and current_tokens + sent_tokens > chunk_size:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _content_str(msg: dict) -> str:
    """Extract a plain string from a message dict regardless of content shape."""
    content = msg.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    return str(content)


def _strip_memory_block(text: str) -> str:
    """Remove an injected _UTILIZE_PREFIX block so memories aren't re-stored."""
    marker = _UTILIZE_PREFIX.strip()
    idx = text.find(marker)
    if idx == -1:
        idx = text.find("\n\nRelevant memories from prior episodes:")
    if idx != -1:
        text = text[:idx]
    return text.rstrip()


def _format_trajectory(trajectory: list[dict]) -> str:
    """Flatten a BFCL full_message_history into indexable text."""
    task_line = ""
    steps: list[str] = []

    for msg in trajectory:
        role = msg.get("role", "unknown")
        raw = _content_str(msg)

        # Strip injected memory block from system messages to prevent snowball.
        if role == "system":
            raw = _strip_memory_block(raw)

        # Derive task description from first user turn.
        if not task_line and role == "user" and raw:
            task_line = raw.splitlines()[0]

        if raw:
            steps.append(f"Step {len(steps) + 1} ({role}): {raw}")

        # Also capture structured tool_calls from assistant turns.
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    call_str = f"{fn.get('name', '')}({fn.get('arguments', '')})"
                    steps.append(f"Step {len(steps) + 1} (tool_call): {call_str}")

    header = f"Task: {task_line}\n\n[Execution Steps]" if task_line else "[Execution Steps]"
    return header + "\n" + "\n".join(steps)


# ── Main class ───────────────────────────────────────────────────────────────

class Bm25Memory(Memory):
    """Cross-episode memory backed by BM25 retrieval over stored trajectory chunks.

    Storage layout under storage_dir/:
        memory_bank.jsonl  — append-only JSONL; each row is one chunk.

    No LLM or embedding API calls. Thread-safe via RLock.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        top_k: int = 3,
        chunk_size: int = 1024,
        clear: bool = False,
        readonly: bool = False,
        **unused,  # absorb ark_api_key, dashscope_*, etc. passed by the driver
    ):
        super().__init__(clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._bank_path = self._storage_dir / "memory_bank.jsonl"
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._lock = threading.RLock()
        self._memory_bank: list[dict] = []
        self._bm25 = None
        self._dirty = False
        self._update_counter = 0

        if clear:
            self._bank_path.write_text("")

        self._load_bank()
        self._ensure_nltk()

    def _ensure_nltk(self) -> None:
        try:
            import nltk
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    def _load_bank(self) -> None:
        if not self._bank_path.exists():
            return
        rows: list[dict] = []
        try:
            with open(self._bank_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "chunk_text" not in row:
                        # Stale schema — reset.
                        rows = []
                        break
                    rows.append(row)
        except Exception:
            rows = []
        self._memory_bank = rows
        if rows:
            self._dirty = True  # trigger index rebuild on first utilize

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(row["chunk_text"]) for row in self._memory_bank]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._dirty = False

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        t0 = time.monotonic()
        with self._lock:
            if not self._memory_bank:
                return ""
            if self._dirty or self._bm25 is None:
                self._build_index()
            if self._bm25 is None:
                return ""

        query_tokens = _tokenize(system_prompt)
        if not query_tokens:
            return ""

        with self._lock:
            scores = self._bm25.get_scores(query_tokens)
            k = min(self._top_k, len(self._memory_bank))
            top_indices = scores.argsort()[::-1][:k]
            hits = [f"- {self._memory_bank[i]['chunk_text']}" for i in top_indices]

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1
            u.latency_s += time.monotonic() - t0

        if not hits:
            return ""
        return _UTILIZE_PREFIX + "\n".join(hits)

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return

        t0 = time.monotonic()
        text = _format_trajectory(trajectory)
        chunks = _chunk_text(text, self._chunk_size)

        with self._lock:
            task_id = str(self._update_counter)
            self._update_counter += 1
            with open(self._bank_path, "a") as f:
                for i, chunk in enumerate(chunks):
                    row = {
                        "task_id": task_id,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "context_category": "",
                        "sub_category": "",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._memory_bank.append(row)
            self._dirty = True

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1
            u.latency_s += time.monotonic() - t0

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "Bm25Memory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
