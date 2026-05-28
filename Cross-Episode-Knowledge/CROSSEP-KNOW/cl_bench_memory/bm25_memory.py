"""BM25 sparse retrieval memory backend (LLM-free, embedding-free baseline).

Follows MAB's chunk-level approach: each trajectory is split into sentence-boundary
chunks of <= CHUNK_SIZE tokens, each chunk stored and indexed separately. Mirrors
Qwen3EmbeddingMemory so sparse vs dense comparisons are at the same granularity.
"""

import json
import os
import time

from rank_bm25 import BM25Okapi

from .base import Memory
from .chunking import chunk_text
from .io import append_jsonl
from .logging_utils import log

BM25_MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are prior task trajectory chunks retrieved by BM25 keyword match "
    "from my memory of past interactions in this context. They may contain "
    "useful examples, facts, or reasoning patterns. Consider each item and "
    "decide whether it is relevant before using it.\n\n"
)


class BM25Memory(Memory):
    """BM25 sparse retrieval over chunked trajectories.

    Each call to extract() splits the trajectory into sentence-boundary chunks
    (<= CHUNK_SIZE tokens) and stores each chunk as a separate bank entry.
    retrieve() scores all chunks against the query via BM25Okapi and returns
    the top-k matching chunks for system-message injection.

    Storage layout (per context_id directory):
        memory_bank.jsonl  — one JSON record per chunk
    """

    CHUNK_SIZE = 1024  # tokens, same as Qwen3EmbeddingMemory

    def __init__(
        self,
        client=None,
        model=None,
        memory_dir=None,
        embed_provider=None,
        embed_model=None,
        embed_client=None,
        top_k=10,
        extract_model=None,
    ):
        # LLM / embedding kwargs accepted for interface parity; all ignored.
        self.top_k = top_k
        self.memory_dir = memory_dir

        self.memory_bank: list = []
        self._bm25: BM25Okapi | None = None
        self._dirty: bool = False

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")

        self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not os.path.exists(self.bank_path):
            return
        raw_entries = []
        with open(self.bank_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_entries.append(json.loads(line))
        if raw_entries and "chunk_text" not in raw_entries[0]:
            log(
                f"   ⚠️ Stale BM25 memory at {self.bank_path} "
                f"(old schema, no 'chunk_text'); discarding {len(raw_entries)} entries"
            )
            os.remove(self.bank_path)
            return
        self.memory_bank = raw_entries
        if self.memory_bank:
            log(f"   Loaded {len(self.memory_bank)} BM25 chunks from disk")
            self._dirty = True  # rebuild index on first retrieve

    # ── BM25 helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list:
        """Whitespace tokenize — matches MemoryAgentBench BM25 convention."""
        return text.split()

    @staticmethod
    def _build_document(entry: dict) -> str:
        return entry.get("chunk_text", "")

    @staticmethod
    def _format_item(entry: dict) -> str:
        return entry.get("chunk_text", "")

    def _build_index(self) -> None:
        if not self.memory_bank:
            self._bm25 = None
            self._dirty = False
            return
        tokenized_corpus = [
            self._tokenize(self._build_document(e)) for e in self.memory_bank
        ]
        if not any(tokenized_corpus):
            self._bm25 = None
            self._dirty = False
            return
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._dirty = False

    # ── Memory ABC ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        if not self.memory_bank:
            return "", {"latency_s": 0.0, "embed_usage": {}}

        t0 = time.time()

        if self._dirty or self._bm25 is None:
            self._build_index()

        if self._bm25 is None:
            return "", {"latency_s": time.time() - t0, "embed_usage": {}}

        tokenized_query = self._tokenize(query or "")
        if not tokenized_query:
            return "", {"latency_s": time.time() - t0, "embed_usage": {}}

        scores = self._bm25.get_scores(tokenized_query)
        k = min(self.top_k, len(self.memory_bank))
        top_indices = scores.argsort()[::-1][:k]

        items = []
        for rank, idx in enumerate(top_indices, start=1):
            text = self._format_item(self.memory_bank[int(idx)])
            if text:
                items.append(f"Memory {rank}:\n{text}")

        stats = {"latency_s": time.time() - t0, "embed_usage": {}}
        if not items:
            return "", stats
        return BM25_MEMORY_INJECTION_PREFIX + "\n\n".join(items), stats

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        t0 = time.time()

        full_text = f"{query}\n\n{content}" if query else (content or "")
        chunks = chunk_text(full_text, chunk_size=self.CHUNK_SIZE)

        for chunk_index, chunk_txt in enumerate(chunks):
            entry = {
                "task_id": task_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_txt,
                "context_category": context_category or "",
                "sub_category": sub_category or "",
            }
            self.memory_bank.append(entry)
            append_jsonl(entry, self.bank_path)

        self._dirty = True

        return {
            "latency_s": time.time() - t0,
            "llm_usage": {},
            "embed_usage": {},
            "verdict": f"added {len(chunks)} chunks",
        }
