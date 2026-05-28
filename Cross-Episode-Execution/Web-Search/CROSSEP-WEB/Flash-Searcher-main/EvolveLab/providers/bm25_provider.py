"""BM25 sparse retrieval memory provider for EvoMemBench.

Ported from CL-Bench's cl_bench_memory/bm25_memory.py.
Algorithm: rank_bm25.BM25Okapi over chunked trajectory corpus.

Differences from CL-Bench:
- Tokenizer: language-detected hybrid (jieba for CJK, whitespace for others)
  instead of pure whitespace — WebWalkerQA contains Chinese tasks.
- Chunking: faithful port of CL-Bench's NLTK+tiktoken sentence chunker.
- No LLM, no embeddings.
"""

import json
import os
import re
import sys
import threading
from typing import List

import nltk
import tiktoken
from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest, MemoryResponse, MemoryItem, MemoryItemType,
    MemoryStatus, MemoryType, TrajectoryData,
)

# Regex for CJK codepoints (CJK Unified, Hiragana/Katakana, Hangul)
_CJK_RE = re.compile(r'[一-鿿぀-ヿ가-힯]')


def _tokenize(text: str) -> list:
    """Hybrid tokenizer: jieba for CJK text, whitespace for others."""
    if _CJK_RE.search(text):
        import jieba
        return list(jieba.lcut(text))
    return text.split()


def _chunk_text(text: str, chunk_size: int = 1024) -> list:
    """Split text into ≤chunk_size-token chunks at sentence boundaries.

    Ported verbatim from CL-Bench cl_bench_memory/chunking.py.
    Uses tiktoken (gpt-4o-mini encoding) for token counting and NLTK punkt
    for sentence segmentation.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        token_count = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current_token_count + token_count > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_token_count = token_count
        else:
            current_sentences.append(sentence)
            current_token_count += token_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks if chunks else [text]


def _format_trajectory(td: TrajectoryData) -> str:
    """Flatten TrajectoryData into a single text block.

    Skips injected memory-guidance messages to prevent snowball growth.
    """
    parts = [f"Task: {td.query}", "", "[Execution Steps]"]
    step_num = 0
    for step in (td.trajectory or []):
        content = step.get("content", "")
        if isinstance(content, list):
            text = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        else:
            text = str(content)
        if "————Memory System Guidance————" in text:
            continue
        step_num += 1
        step_type = step.get("type", step.get("role", "step"))
        parts.append(f"Step {step_num} ({step_type}): {text}")
    parts.append("")
    if td.result is not None:
        parts.append(f"Final Answer: {td.result}")
    if td.metadata:
        parts.append(f"Correct: {td.metadata.get('is_correct')}")
    return "\n".join(parts)


class BM25MemoryProvider(BaseMemoryProvider):
    """BM25 sparse retrieval over chunked agent trajectories.

    Each take_in_memory() call chunks the trajectory into ≤1024-token
    sentence-boundary segments and stores each as a row in memory_bank.jsonl.
    provide_memory() lazily rebuilds BM25Okapi from the JSONL corpus and
    returns top-k matching chunks.
    """

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.BM25, config or {})
        self._top_k        = self.config.get("top_k", 10)
        self._return_on_in = self.config.get("return_on_in", False)
        self._memory_dir   = self.config.get("memory_dir", "./storage/bm25")
        self._chunk_size   = self.config.get("chunk_size", 1024)
        self._bank_path    = os.path.join(self._memory_dir, "memory_bank.jsonl")
        self._memory_bank: list = []
        self._bm25: BM25Okapi | None = None
        self._dirty = False
        self._lock = threading.RLock()

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            os.makedirs(self._memory_dir, exist_ok=True)
            # Pre-download punkt_tab so first take_in_memory doesn't block
            nltk.download("punkt_tab", quiet=True)

            if os.path.exists(self._bank_path):
                raw_entries = []
                with open(self._bank_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_entries.append(json.loads(line))

                # Stale-schema guard (ported from CL-Bench bm25_memory.py:76-82)
                if raw_entries and "chunk_text" not in raw_entries[0]:
                    print(f"[BM25] stale schema at {self._bank_path}; discarding "
                          f"{len(raw_entries)} entries")
                    os.remove(self._bank_path)
                else:
                    self._memory_bank = raw_entries
                    if self._memory_bank:
                        self._dirty = True  # rebuild index on first retrieve

            print(f"[BM25] initialized (top_k={self._top_k}, "
                  f"return_on_in={self._return_on_in}, "
                  f"loaded={len(self._memory_bank)} chunks)")
            return True
        except Exception as e:
            print(f"[BM25] initialize failed: {e}")
            return False

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(
                    memories=[], memory_type=self.memory_type,
                    total_count=len(self._memory_bank),
                )

            if not self._memory_bank:
                return MemoryResponse(memories=[], memory_type=self.memory_type,
                                      total_count=0)

            if self._dirty or self._bm25 is None:
                self._build_index()

            if self._bm25 is None:
                return MemoryResponse(memories=[], memory_type=self.memory_type,
                                      total_count=len(self._memory_bank))

            tokenized_query = _tokenize(request.query or "")
            if not tokenized_query:
                return MemoryResponse(memories=[], memory_type=self.memory_type,
                                      total_count=len(self._memory_bank))

            scores = self._bm25.get_scores(tokenized_query)
            k = min(self._top_k, len(self._memory_bank))
            top_indices = scores.argsort()[::-1][:k]

            items: List[MemoryItem] = []
            for rank, idx in enumerate(top_indices, start=1):
                entry = self._memory_bank[int(idx)]
                chunk_text = entry.get("chunk_text", "")
                if not chunk_text:
                    continue
                items.append(MemoryItem(
                    id=str(int(idx)),
                    content=f"Memory {rank}:\n{chunk_text}",
                    metadata={
                        "task_id": entry.get("task_id", ""),
                        "chunk_index": entry.get("chunk_index", 0),
                    },
                    score=float(scores[int(idx)]),
                    type=MemoryItemType.TEXT,
                ))

            return MemoryResponse(
                memories=items, memory_type=self.memory_type,
                total_count=len(self._memory_bank),
            )

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            try:
                content = _format_trajectory(trajectory_data)
                full_text = f"{trajectory_data.query}\n\n{content}"
                chunks = _chunk_text(full_text, self._chunk_size)
                task_id = (trajectory_data.metadata or {}).get("task_id", "")

                with open(self._bank_path, "a", encoding="utf-8") as f:
                    for chunk_index, chunk_txt in enumerate(chunks):
                        entry = {
                            "task_id": task_id,
                            "chunk_index": chunk_index,
                            "chunk_text": chunk_txt,
                            "context_category": "",
                            "sub_category": "",
                        }
                        self._memory_bank.append(entry)
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                self._dirty = True
                return True, (f"added {len(chunks)} chunks "
                              f"(total={len(self._memory_bank)})")
            except Exception as e:
                return False, f"take_in_memory error: {e}"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        if not self._memory_bank:
            self._bm25 = None
            self._dirty = False
            return
        tokenized_corpus = [
            _tokenize(entry.get("chunk_text", "")) for entry in self._memory_bank
        ]
        if not any(tokenized_corpus):
            self._bm25 = None
            self._dirty = False
            return
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._dirty = False


if __name__ == "__main__":
    """Standalone smoke test — run via 'BM25 · Provider Smoke' in VS Code."""
    import shutil

    TEST_DIR = "/tmp/bm25_vscode_smoke"
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    provider = BM25MemoryProvider({"memory_dir": TEST_DIR, "top_k": 3})
    assert provider.initialize(), "initialize() failed"

    # English trajectory
    td_en = TrajectoryData(
        query="What is the capital of France?",
        trajectory=[
            {"role": "assistant", "content": "I will search for this."},
            {"role": "tool", "content": "Paris is the capital of France."},
        ],
        result="Paris",
        metadata={"task_id": "en_001", "is_correct": True},
    )
    ok, msg = provider.take_in_memory(td_en)
    print(f"[EN] take_in_memory: {ok} — {msg}")

    # Chinese trajectory
    td_zh = TrajectoryData(
        query="法国的首都是什么?",
        trajectory=[
            {"role": "assistant", "content": "我来搜索一下。"},
            {"role": "tool", "content": "巴黎是法国的首都。"},
        ],
        result="巴黎",
        metadata={"task_id": "zh_001", "is_correct": True},
    )
    ok, msg = provider.take_in_memory(td_zh)
    print(f"[ZH] take_in_memory: {ok} — {msg}")

    # Retrieval — English
    req_en = MemoryRequest(query="capital France", context="", status=MemoryStatus.BEGIN)
    resp_en = provider.provide_memory(req_en)
    print(f"[EN] retrieve: {len(resp_en.memories)} hits")
    for m in resp_en.memories:
        print(f"  score={m.score:.4f}  {m.content[:60]!r}")

    # Retrieval — Chinese
    req_zh = MemoryRequest(query="法国首都", context="", status=MemoryStatus.BEGIN)
    resp_zh = provider.provide_memory(req_zh)
    print(f"[ZH] retrieve: {len(resp_zh.memories)} hits")
    for m in resp_zh.memories:
        print(f"  score={m.score:.4f}  {m.content[:60]!r}")

    # MemoryStatus.IN short-circuit
    req_in = MemoryRequest(query="capital", context="", status=MemoryStatus.IN)
    resp_in = provider.provide_memory(req_in)
    assert len(resp_in.memories) == 0, "IN status should return empty"
    print("[OK] MemoryStatus.IN short-circuit works")

    # Persistence reload
    provider2 = BM25MemoryProvider({"memory_dir": TEST_DIR, "top_k": 3})
    provider2.initialize()
    assert len(provider2._memory_bank) == len(provider._memory_bank), "reload mismatch"
    print(f"[OK] Persistence reload: {len(provider2._memory_bank)} chunks loaded")

    print("\nAll smoke tests passed.")
