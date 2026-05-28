"""AmemMemory — A-mem (AgenticMemorySystem) backed Memory interface.

LLM calls inside A-mem are routed through the Ark batch endpoint via a thin
subclass (_MeteredOpenAIController) that replaces the default OpenAIController
on the constructed AgenticMemorySystem instance.

Embeddings use DashScope text-embedding-v4 via a ChromaDB-compatible callable
(_DashScopeEmbeddingFunction) passed at construction time; token usage is
recorded in metrics._TLS.

Persistence uses a custom _PersistentDashScopeRetriever (subclass of
ChromaRetriever) that swaps EphemeralClient for PersistentClient, preserving
memory across samples (Phase-1) and across Phase-2 load-from-disk.

evo_threshold is set to 10_000 so consolidate_memories() never fires during
a 50-sample run.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import chromadb

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.mem0_backend import _truncate_query_to_embed_limit

_UTILIZE_PREFIX = "\n\nRelevant memories from prior episodes:\n"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper: strip markdown fences (mirrors agentic_memory.llm_controller)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_json(text: str) -> str:
    import re
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metered LLM controller (records LLM token usage in metrics._TLS)
# ─────────────────────────────────────────────────────────────────────────────

class _MeteredOpenAIController:
    """Drop-in replacement for agentic_memory.llm_controller.OpenAIController.

    Records LLM token usage via metrics.record_llm_call on every call.
    """

    def __init__(self, model: str, api_key: str):
        import os
        from volcenginesdkarkruntime import Ark
        self.model = os.getenv("BATCH_MODEL", model)
        self._ark = Ark(api_key=api_key)

    def get_completion(self, prompt: str, response_format=None, temperature: float = 0.7) -> str:
        t0 = time.time()
        response = self._ark.batch.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        dt = time.time() - t0
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(response.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "completion_tokens", 0) or 0),
        )
        return _strip_markdown_json(response.choices[0].message.content)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DashScope embedding function (ChromaDB-compatible callable)
# ─────────────────────────────────────────────────────────────────────────────

class _DashScopeEmbeddingFunction:
    """ChromaDB embedding function backed by DashScope text-embedding-v4."""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4"):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    def name(self) -> str:
        return f"dashscope_{self._model}"

    def embed_query(self, input):
        return self(input)

    def __call__(self, input):  # input: List[str] → List[List[float]]
        from openai import OpenAI
        t0 = time.time()
        resp = OpenAI(api_key=self._api_key, base_url=self._base_url).embeddings.create(
            input=list(input), model=self._model
        )
        elapsed = time.time() - t0
        metrics.record_embed_call(
            total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
            latency_s=elapsed,
        )
        return [d.embedding for d in resp.data]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Persistent ChromaDB retriever with DashScope embedding
# ─────────────────────────────────────────────────────────────────────────────

class _PersistentDashScopeRetriever:
    """ChromaDB PersistentClient retriever with DashScope embedding function.

    Mirrors the interface of agentic_memory.retrievers.ChromaRetriever
    (add_document / delete_document / search / _convert_metadata_types).
    Does NOT call ChromaRetriever.__init__() to avoid EphemeralClient creation.
    """

    def __init__(
        self,
        directory: Path,
        collection_name: str,
        embedding_function: _DashScopeEmbeddingFunction,
        extend: bool = False,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function
        self.client = chromadb.PersistentClient(path=str(directory))
        existing = [c.name for c in self.client.list_collections()]
        if collection_name in existing:
            if extend:
                self.collection = self.client.get_collection(
                    name=collection_name, embedding_function=embedding_function
                )
            else:
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name, embedding_function=embedding_function
                )
        else:
            self.collection = self.client.create_collection(
                name=collection_name, embedding_function=embedding_function
            )

    def add_document(self, document: str, metadata: dict, doc_id: str) -> None:
        import json
        processed = {}
        for k, v in metadata.items():
            if isinstance(v, (list, dict)):
                processed[k] = json.dumps(v)
            else:
                processed[k] = str(v)
        self.collection.add(documents=[document], metadatas=[processed], ids=[doc_id])

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5) -> dict:
        import ast
        results = self.collection.query(query_texts=[query], n_results=k)
        if results and results.get("metadatas"):
            for row in results["metadatas"]:
                for meta in (row or []):
                    for key, val in list(meta.items()):
                        if isinstance(val, str):
                            try:
                                meta[key] = ast.literal_eval(val)
                            except Exception:
                                pass
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Trajectory → text helper
# ─────────────────────────────────────────────────────────────────────────────

def _strip_memory_snippet(text: str) -> str:
    """Remove injected memory snippet appended to user messages."""
    idx = text.find(_UTILIZE_PREFIX)
    return text[:idx].rstrip() if idx != -1 else text


def _traj_to_text(trajectory: list[dict]) -> str:
    """Convert a multi-turn trajectory to a single text for add_note()."""
    parts = []
    for m in trajectory:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "system":
            continue
        elif role == "user" and content:
            parts.append(f"User: {_strip_memory_snippet(content)}")
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name", "")
                args = (tc.get("function") or {}).get("arguments", "")
                parts.append(f"[call] {fn}({args})")
            if content:
                parts.append(f"Assistant: {content}")
        elif role == "tool" and content:
            parts.append(f"[result] {content[:500]}")
    return "\n".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6. AmemMemory — the Memory ABC implementation
# ─────────────────────────────────────────────────────────────────────────────

class AmemMemory(Memory):
    """Cross-episode memory backed by A-mem (AgenticMemorySystem).

    Storage layout under storage_dir/:
        chromadb/   — ChromaDB PersistentClient data

    LLM calls (process_memory / analyze_content) are routed through Ark batch
    via _MeteredOpenAIController. Embeddings use DashScope text-embedding-v4
    via _DashScopeEmbeddingFunction with usage recorded in metrics._TLS.

    evo_threshold=10_000 prevents consolidate_memories() from firing during
    a standard 50-sample evaluation run.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        embed_model: str = "text-embedding-v4",
        collection_name: str = "bfcl_amem",
        top_k: int = 5,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as exc:
            raise ImportError(
                "agentic_memory is required for --memory-type amem. "
                "Install it from EvoMemBench-Memory-Systems/A-mem/."
            ) from exc

        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = storage_dir / "chromadb"

        if clear:
            shutil.rmtree(chroma_dir, ignore_errors=True)

        ef = _DashScopeEmbeddingFunction(
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
            model=embed_model,
        )

        self._amem = AgenticMemorySystem(
            llm_backend="openai",
            llm_model=ark_model,
            api_key=ark_api_key,
            evo_threshold=10_000,
            embedding_function=ef,
        )

        # Replace ephemeral LLM controller with metered Ark version
        self._amem.llm_controller.llm = _MeteredOpenAIController(
            model=ark_model, api_key=ark_api_key
        )

        # Replace ephemeral retriever with persistent DashScope-backed one
        extend = not clear
        self._amem.retriever = _PersistentDashScopeRetriever(
            directory=chroma_dir,
            collection_name=collection_name,
            embedding_function=ef,
            extend=extend,
        )
        # Keep refs so consolidate_memories() (if ever triggered) uses our ef
        self._amem._embedding_function = ef
        self._amem._collection_name = collection_name

        self._top_k = top_k
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        query = _truncate_query_to_embed_limit(system_prompt)
        u_obj = getattr(metrics._TLS, "usage", None)
        if u_obj is not None:
            u_obj.n_utilize += 1

        try:
            results = self._amem.retriever.search(query, k=self._top_k)
        except Exception:
            return ""

        if not results or not results.get("ids") or not results["ids"][0]:
            return ""

        metas = (results.get("metadatas") or [[]])[0]
        items: list[str] = []
        for i in range(len(results["ids"][0])):
            meta = metas[i] if i < len(metas) else {}
            content = meta.get("content") or ""
            if content:
                items.append(content)

        if not items:
            return ""
        lines = "\n".join(f"- {c}" for c in items)
        return _UTILIZE_PREFIX + lines

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        text = _traj_to_text(trajectory)
        if not text.strip():
            return
        with self._lock:
            try:
                self._amem.add_note(content=text)
            except Exception:
                pass
        u_obj = getattr(metrics._TLS, "usage", None)
        if u_obj is not None:
            u_obj.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "AmemMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
