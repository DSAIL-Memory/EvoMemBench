"""BM25 sparse-retrieval memory backend for ALFWorld evaluation.

No LLM, no embeddings. Each completed trajectory is chunked into ≤chunk_size-token
sentence-boundary segments via NLTK + tiktoken and stored in a JSONL bank file.
Retrieval uses rank_bm25.BM25Okapi with whitespace tokenization (English-only;
ALFWorld is entirely English so jieba is unnecessary).

Mirrors the BM25MemoryProvider in Flash-Searcher-main/EvolveLab/providers/bm25_provider.py
but adapted to the BaseMemory / inject+update API used in this fork.
"""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy

from .base import BaseMemory, MemoryCallStats

# ── module-level helpers (stateless) ──────────────────────────────────────────

_MEMORY_SKIP_MARKER = "--- Retrieved Memories ---"


def _tokenize(text: str) -> list[str]:
    return text.split()


_encoder_cache: object | None = None


def _get_encoder():
    global _encoder_cache
    if _encoder_cache is None:
        try:
            import tiktoken
            try:
                _encoder_cache = tiktoken.encoding_for_model("gpt-4o-mini")
            except KeyError:
                _encoder_cache = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder_cache = None
    return _encoder_cache


def _token_count(text: str) -> int:
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text, allowed_special={"<|endoftext|>"}))
    except Exception:
        return max(1, len(text) // 4)


def _chunk_text(text: str, chunk_size: int = 1024) -> list[str]:
    """Split text into ≤chunk_size-token chunks at sentence boundaries.

    Falls back to [text] if NLTK is unavailable or produces no output.
    """
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
    except Exception:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        n = _token_count(sent)
        if current_tokens + n > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = n
        else:
            current.append(sent)
            current_tokens += n

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _format_trajectory(conversation: list[dict]) -> str:
    """Flatten a conversation list into a plain-text trajectory block.

    Skips the injected memory-guidance block to prevent snowball growth.
    """
    if len(conversation) < 3:
        return ""

    task = conversation[2]["content"].split("\n")[0].strip()
    parts = [f"Task: {task}", "", "[Execution Steps]"]
    step_num = 0
    for msg in conversation:
        content = msg.get("content") or ""
        if _MEMORY_SKIP_MARKER in content:
            # Strip the injected memory block from the system prompt before storing
            idx = content.find("\n\n" + _MEMORY_SKIP_MARKER)
            if idx != -1:
                content = content[:idx]
        if not content.strip():
            continue
        step_num += 1
        role = msg.get("role", "unknown")
        parts.append(f"Step {step_num} ({role}): {content}")
    return "\n".join(parts)


# ── adapter class ──────────────────────────────────────────────────────────────


class Bm25Memory(BaseMemory):
    """BM25 sparse-retrieval memory backend.

    Stores trajectories as chunked JSONL; retrieves top-k matching chunks by
    BM25Okapi score; injects them into the system prompt under
    '--- Retrieved Memories ---'.
    """

    def __init__(
        self,
        read_only: bool = False,
        agent_id: str = "alfworld",
        memory_dir: str | None = None,
        top_k: int = 10,
        chunk_size: int = 1024,
        return_on_in: bool = False,
    ):
        super().__init__(memory_type="bm25", read_only=read_only)
        if memory_dir is None:
            raise ValueError("Bm25Memory requires 'memory_dir' to be set.")
        self._agent_id = agent_id
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._return_on_in = return_on_in
        self._memory_dir = memory_dir
        self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self._memory_bank: list[dict] = []
        self._bm25 = None
        self._dirty = False
        self._lock = threading.RLock()

        os.makedirs(memory_dir, exist_ok=True)
        self._load_bank()
        self._ensure_nltk()

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_nltk():
        try:
            import nltk
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    def _load_bank(self) -> None:
        if not os.path.exists(self._bank_path):
            return
        entries: list[dict] = []
        try:
            with open(self._bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"[BM25:{self._agent_id}] failed to load bank: {e}")
            return

        if entries and "chunk_text" not in entries[0]:
            print(f"[BM25:{self._agent_id}] stale schema in {self._bank_path}; discarding.")
            os.remove(self._bank_path)
            return

        self._memory_bank = entries
        if entries:
            self._dirty = True
        print(f"[BM25:{self._agent_id}] loaded {len(entries)} chunks from disk.")

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi
        if not self._memory_bank:
            self._bm25 = None
            self._dirty = False
            return
        tokenized = [_tokenize(e.get("chunk_text", "")) for e in self._memory_bank]
        if not any(tokenized):
            self._bm25 = None
            self._dirty = False
            return
        self._bm25 = BM25Okapi(tokenized)
        self._dirty = False

    # ── public interface ───────────────────────────────────────────────────────

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        if len(conversation) < 3:
            return conversation, MemoryCallStats()

        query = conversation[2]["content"]
        t0 = time.time()

        with self._lock:
            if not self._memory_bank:
                return conversation, MemoryCallStats(latency=time.time() - t0)

            if self._dirty or self._bm25 is None:
                self._build_index()

            if self._bm25 is None:
                return conversation, MemoryCallStats(latency=time.time() - t0)

            tokenized_query = _tokenize(query)
            if not tokenized_query:
                return conversation, MemoryCallStats(latency=time.time() - t0)

            scores = self._bm25.get_scores(tokenized_query)
            k = min(self._top_k, len(self._memory_bank))
            top_indices = scores.argsort()[::-1][:k]

            hits = []
            for rank, idx in enumerate(top_indices, start=1):
                chunk_text = self._memory_bank[int(idx)].get("chunk_text", "")
                if chunk_text:
                    hits.append(f"- {chunk_text}")

        elapsed = time.time() - t0
        stats = MemoryCallStats(
            latency=elapsed,
            input_tokens=len(tokenized_query),
        )

        if not hits:
            return conversation, stats

        memory_lines = "\n".join(hits)
        injection = f"\n\n--- Retrieved Memories ---\n{memory_lines}\n---"
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = new_conv[0]["content"] + injection
        return new_conv, stats

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        t0 = time.time()
        text = _format_trajectory(conversation)
        if not text.strip():
            return MemoryCallStats(latency=time.time() - t0)

        chunks = _chunk_text(text, self._chunk_size)
        task_id = str(data_idx) if data_idx is not None else ""
        total_tokens = sum(len(_tokenize(c)) for c in chunks)

        with self._lock:
            try:
                with open(self._bank_path, "a", encoding="utf-8") as f:
                    for i, chunk_text in enumerate(chunks):
                        entry = {
                            "task_id": task_id,
                            "chunk_index": i,
                            "chunk_text": chunk_text,
                            "context_category": "",
                            "sub_category": "",
                        }
                        self._memory_bank.append(entry)
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._dirty = True
                print(f"[BM25:{self._agent_id}] idx={task_id} added {len(chunks)} chunk(s) "
                      f"(total={len(self._memory_bank)})")
            except Exception as e:
                print(f"[BM25:{self._agent_id}] update error: {e}")

        return MemoryCallStats(
            latency=time.time() - t0,
            input_tokens=total_tokens,
        )

    def load_from_disk(self, memory_dir: str | None = None, **kwargs) -> None:
        with self._lock:
            if memory_dir is not None:
                self._memory_dir = memory_dir
                self._bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
                os.makedirs(memory_dir, exist_ok=True)
            self._memory_bank = []
            self._bm25 = None
            self._dirty = False
            self._load_bank()


if __name__ == "__main__":
    import shutil

    TEST_DIR = "/tmp/bm25_debug_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("=== BM25 Adapter Smoke Test ===")
    print(f"memory_dir: {TEST_DIR}")

    m = Bm25Memory(memory_dir=TEST_DIR, top_k=3)

    # 模拟一条 ALFWorld 对话（系统提示 + 首次回复 + 第一轮环境观察 + 动作）
    conv = [
        {"role": "user",      "content": "You are an agent in an ALFWorld household environment. Complete the task by issuing actions."},
        {"role": "assistant", "content": "I will complete the task step by step."},
        {"role": "user",      "content": "You are in the middle of a room. Looking quickly around you, you see a fridge 1, a microwave 1, a countertop 1.\nYour task is to: put a cool potato in the microwave."},
        {"role": "assistant", "content": "think: I need to find a potato first.\ngo to fridge 1"},
        {"role": "user",      "content": "You open the fridge. Inside you see a potato 1."},
        {"role": "assistant", "content": "take potato 1 from fridge 1"},
        {"role": "user",      "content": "You pick up the potato 1."},
        {"role": "assistant", "content": "go to microwave 1"},
        {"role": "user",      "content": "You arrive at microwave 1."},
        {"role": "assistant", "content": "put potato 1 in/on microwave 1"},
        {"role": "user",      "content": "Task completed! Reward: 1"},
    ]

    # 1. 首次 inject：无记忆，应返回原 conversation
    new_conv, s = m.inject(conv)
    assert new_conv[0]["content"] == conv[0]["content"], "FAIL: first inject should not modify conv"
    print(f"[1] inject (empty bank): latency={s.latency*1000:.1f}ms  ✓")

    # 2. update：写入 JSONL
    s2 = m.update(conv, data_idx=2420)
    print(f"[2] update: input_tokens={s2.input_tokens}  latency={s2.latency*1000:.1f}ms  chunks={len(m._memory_bank)}  ✓")

    # 3. 再次 inject：应能召回
    new_conv2, s3 = m.inject(conv)
    assert "--- Retrieved Memories ---" in new_conv2[0]["content"], "FAIL: no memories injected"
    print(f"[3] inject (with memory): input_tokens={s3.input_tokens}  latency={s3.latency*1000:.1f}ms  ✓")
    print(f"    system prompt tail: ...{new_conv2[0]['content'][-120:]!r}")

    # 4. read_only：update 应为空操作，bank 不变
    m_ro = Bm25Memory(memory_dir=TEST_DIR, top_k=3, read_only=True)
    lines_before = len(open(m_ro._bank_path).readlines())
    s4 = m_ro.update(conv, data_idx=9999)
    lines_after = len(open(m_ro._bank_path).readlines())
    assert lines_before == lines_after, "FAIL: readonly should not write"
    assert s4.input_tokens == 0, "FAIL: readonly stats should be zero"
    print(f"[4] readonly update: bank lines unchanged ({lines_before})  ✓")

    # 5. 持久化重载
    m2 = Bm25Memory(memory_dir=TEST_DIR, top_k=3)
    assert len(m2._memory_bank) == len(m._memory_bank), "FAIL: reload count mismatch"
    print(f"[5] reload from disk: {len(m2._memory_bank)} chunk(s)  ✓")

    print("\nAll smoke tests passed.")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
