"""A-mem (agentic_memory) provider for EvoMemBench-XbenchWebWalkQA.

Thin wrapper around A-mem's public API:
  - add_note() / search_agentic() / analyze_content()

Design principles:
  - Do NOT modify A-mem or the benchmark; only use A-mem's constructor + public methods
  - Embeddings go through DashScope text-embedding-v4 via a ChromaDB EmbeddingFunction
    (passed through A-mem's public `embedding_function=` constructor parameter)
  - Persistence is a side-car JSON file (A-mem has no save/load API); on initialize
    we re-populate A-mem's in-memory store by replaying add_note() calls
  - `evo_threshold=99999` disables A-mem's consolidate_memories() (known to race
    under concurrency)
"""

import json
import os
import sys
import threading
from typing import List

from chromadb import EmbeddingFunction, Documents, Embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest, MemoryResponse, MemoryItem, MemoryItemType,
    MemoryStatus, MemoryType, TrajectoryData,
)


# ─── DashScope embedding (passed via A-mem's public embedding_function arg) ──

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("r50k_base")  # offline, no download
except Exception:
    _TIKTOKEN_ENC = None

_EMBED_MAX_TOKENS = 8192


def _truncate_to_embed_limit(text: str) -> str:
    """Truncate to DashScope's 8192-token limit. r50k_base overestimates CJK ~2x, safe."""
    if _TIKTOKEN_ENC is None:
        return text[:24000]
    tokens = _TIKTOKEN_ENC.encode(text)
    if len(tokens) <= _EMBED_MAX_TOKENS:
        return text
    return _TIKTOKEN_ENC.decode(tokens[:_EMBED_MAX_TOKENS])


class _DashScopeEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB embedding function calling DashScope via the OpenAI-compatible API.

    Avoids SentenceTransformer (and its tqdm multi-thread crashes) entirely.
    ChromaDB 1.5.5 requires inheriting EmbeddingFunction[Documents] and exposing
    a static name() method.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def name() -> str:
        return "dashscope_text_embedding_v4"

    def __call__(self, input: Documents) -> Embeddings:
        truncated = [_truncate_to_embed_limit(t) for t in input]
        response = self.client.embeddings.create(model=self.model, input=truncated)
        return [item.embedding for item in response.data]


# ─── LLM controller that routes A-mem's LLM calls through benchmark task_model ──

class _TaskModelLLMController:
    """Route A-mem's internal LLM calls through the benchmark's task_model so
    that tokens flow into OpenAIServerModel.total_input/output_tokens and get
    picked up by TokenCounter.from_model(task_model).

    Mimics A-mem's agentic_memory.llm_controller.OpenAIController.get_completion
    signature so A-mem's existing call sites (analyze_content, process_memory)
    work without modification.
    """

    def __init__(self, task_model):
        self._task_model = task_model

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        import re
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt},
        ]
        # response_format intentionally dropped: OpenAIServerModel.__call__
        # doesn't accept it, and DeepSeek is unreliable with json_object anyway.
        # The system prompt is enough to steer the model to JSON output.
        chat_msg = self._task_model(messages, temperature=temperature, max_tokens=32768)
        content = chat_msg.content or ""
        # Strip markdown code fences (```json ... ```) that the model may wrap around JSON.
        # A-mem's OpenAIController does this via _strip_markdown_json; we must mirror it.
        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content).strip()
        return content


# ─── Provider ────────────────────────────────────────────────────────────────


class AMemMemoryProvider(BaseMemoryProvider):
    """A-mem memory provider: Zettelkasten-style agentic memory with semantic evolution."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.A_MEM, config or {})
        self._top_k = self.config.get("top_k", 3)
        self._return_on_in = self.config.get("return_on_in", False)
        self._db_path = self.config.get("db_path", "./storage/a_mem/a_mem_memory.json")
        self._evo_threshold = self.config.get("evo_threshold", 99999)
        self._lock = threading.RLock()
        self._a_mem = None
        self._ef = None

    # ── initialize ─────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as e:
            print(f"[A-mem] import failed: {e}")
            return False

        llm_cfg = self.config.get("llm", {}) or {}
        embed_cfg = self.config.get("embed", {}) or {}

        try:
            self._ef = _DashScopeEmbeddingFunction(
                api_key=embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY"),
                base_url=embed_cfg.get("base_url") or os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                model=embed_cfg.get("model", "text-embedding-v4"),
            )

            self._a_mem = AgenticMemorySystem(
                model_name="all-MiniLM-L6-v2",  # placeholder; overridden by embedding_function
                llm_backend="openai",
                llm_model=llm_cfg.get("model") or os.getenv("ANALYSIS_MODEL", "deepseek-v3-2-251201"),
                evo_threshold=self._evo_threshold,
                api_key=llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY"),
                base_url=llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL"),
                embedding_function=self._ef,
            )

            # Route A-mem's internal LLM calls through the benchmark's task_model
            # so its tokens are automatically counted by OpenAIServerModel.
            task_model = self.config.get("model")
            if task_model is not None:
                self._a_mem.llm_controller.llm = _TaskModelLLMController(task_model)
                print("[A-mem] LLM controller routed through benchmark task_model (tokens tracked)")
            else:
                print("[A-mem] WARNING: no task_model in config; A-mem LLM tokens will not be counted")

            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            loaded = 0
            if os.path.exists(self._db_path):
                with open(self._db_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for note in saved:
                    try:
                        self._a_mem.add_note(
                            note["content"],
                            keywords=note.get("keywords", []),
                            context=note.get("context", "General"),
                            tags=note.get("tags", []),
                        )
                        loaded += 1
                    except Exception as e:
                        print(f"[A-mem] skipped one persisted note on reload: {e}")

            print(f"[A-mem] initialized (top_k={self._top_k}, "
                  f"return_on_in={self._return_on_in}, loaded={loaded} notes from disk)")
            return True
        except Exception as e:
            print(f"[A-mem] initialize failed: {e}")
            return False

    # ── provide_memory ─────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            if self._a_mem is None:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            if request.status == MemoryStatus.IN and not self._return_on_in:
                return MemoryResponse(
                    memories=[], memory_type=self.memory_type,
                    total_count=len(self._a_mem.memories),
                )

            try:
                results = self._a_mem.search_agentic(request.query, k=self._top_k)
            except Exception as e:
                print(f"[A-mem] search_agentic failed: {e}")
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            items: List[MemoryItem] = []
            for i, r in enumerate(results):
                items.append(MemoryItem(
                    id=str(r.get("id", i)),
                    content=r.get("content", ""),
                    metadata={
                        "keywords": r.get("keywords", []),
                        "tags": r.get("tags", []),
                        "context": r.get("context", ""),
                    },
                    score=r.get("score"),
                    type=MemoryItemType.TEXT,
                ))

            return MemoryResponse(
                memories=items, memory_type=self.memory_type,
                total_count=len(self._a_mem.memories),
            )

    # ── take_in_memory ─────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._a_mem is None:
                return False, "A-mem not initialized"

            try:
                content = _format_trajectory(trajectory_data)
                analysis = self._a_mem.analyze_content(content)
                keywords = analysis.get("keywords", [])
                context = analysis.get("context", "General")
                tags = analysis.get("tags", [])

                self._a_mem.add_note(
                    content,
                    keywords=keywords,
                    context=context,
                    tags=tags,
                )

                self._append_json({
                    "content": content,
                    "keywords": keywords,
                    "context": context,
                    "tags": tags,
                })

                return True, f"note added (total={len(self._a_mem.memories)})"
            except Exception as e:
                return False, f"take_in_memory error: {e}"

    # ── helpers ────────────────────────────────────────────────────────────

    def _append_json(self, entry: dict) -> None:
        data = []
        if os.path.exists(self._db_path):
            with open(self._db_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        data.append(entry)
        with open(self._db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _format_trajectory(td: TrajectoryData) -> str:
    """Flatten TrajectoryData into a single text block for A-mem's add_note."""
    parts = [f"Task: {td.query}", "", "[Execution Steps]"]
    step_num = 0
    for step in (td.trajectory or []):
        # Skip injected memory_guidance messages to prevent snowball growth.
        # These are user messages marked with "————Memory System Guidance————".
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
