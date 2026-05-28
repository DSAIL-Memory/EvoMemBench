"""AgentKBMemory — BFCL adapter for the AgentKB (Agentic Knowledge Base) provider.

Process-scoped: the knowledge base accumulates across samples.
Each sample's trajectory is buffered per-thread and ingested at drain_usage() time.

Embeddings use DashScope text-embedding-v4 via the OpenAI-compatible endpoint
(same as mem0). TF-IDF is kept for the lexical half of hybrid search.

Based on EvolveLab/providers/agent_kb_provider.py.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

from bfcl_eval.memory import metrics
from bfcl_eval.memory.base import Memory
from bfcl_eval.memory.metrics import MemoryUsageLog

_UTILIZE_PREFIX = "\n\nAgent-KB Guidance (from similar past tasks):\n"
_EMBED_MODEL = "text-embedding-v4"
_EMBED_DIMS = 1024
_EMBED_BUDGET_CHARS = 8000 * 4  # ~8k token char-based fallback


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(t in name for t in ("ratelimit", "timeout", "apiconnection", "apierror")):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in ("rate limit", "timeout", "timed out", "connection", "503", "overloaded"))


_embed_retry = retry(
    wait=wait_fixed(2) + wait_random(0, 1),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


class _DashscopeEmbedClient:
    """Standalone DashScope text-embedding-v4 client (OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str, base_url: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @_embed_retry
    def _create(self, texts: List[str]) -> tuple:
        t0 = time.time()
        resp = self._client.embeddings.create(
            model=_EMBED_MODEL,
            input=texts,
            encoding_format="float",
            dimensions=_EMBED_DIMS,
        )
        return resp, time.time() - t0

    def embed(self, text: str) -> np.ndarray:
        text = text.replace("\n", " ")
        if len(text) > _EMBED_BUDGET_CHARS:
            text = text[-_EMBED_BUDGET_CHARS:]
        resp, dt = self._create([text])
        metrics.record_embed_call(
            total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
            latency_s=dt,
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return (N, dims) float32 array."""
        MAX_BATCH = 100
        cleaned = [t.replace("\n", " ")[-_EMBED_BUDGET_CHARS:] for t in texts]
        all_embs: List[List[float]] = []
        for i in range(0, len(cleaned), MAX_BATCH):
            chunk = cleaned[i : i + MAX_BATCH]
            resp, dt = self._create(chunk)
            metrics.record_embed_call(
                total_tokens=int(getattr(resp.usage, "total_tokens", 0) or 0),
                latency_s=dt,
            )
            all_embs.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
        return np.array(all_embs, dtype=np.float32)


class _ArkLLMCallable:
    """Minimal callable wrapping ArkBatchClient for LLM synthesis calls."""

    def __init__(self, api_key: str, model: str):
        from bfcl_eval.memory.ark_client import ArkBatchClient

        self._client = ArkBatchClient(api_key)
        self._model = model

    def __call__(self, messages: list) -> object:
        normalized = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            normalized.append({"role": msg.get("role", "user"), "content": content})
        resp, dt = self._client.chat_completions(self._model, normalized)
        metrics.record_llm_call(
            latency_s=dt,
            input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
        )
        content = getattr(resp.choices[0].message, "content", "") or ""
        return type("_Resp", (), {"content": content})()


def _cosine_sim_matrix(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Return (N,) cosine similarity between query and each doc row."""
    nq = np.linalg.norm(query) or 1e-10
    nd = np.linalg.norm(docs, axis=1)
    nd[nd == 0] = 1e-10
    return np.dot(docs, query) / (nd * nq)


class AgentKBMemory(Memory):
    """Process-scoped knowledge base that stores and retrieves workflow guidance.

    Hybrid search: TF-IDF (lexical) + DashScope text-embedding-v4 (semantic).

    Storage layout under storage_dir/:
        agent_kb.json   — workflow database with cached embedding vectors
    """

    def __init__(
        self,
        storage_dir: "Path | str",
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        top_k: int = 3,
        clear: bool = False,
        readonly: bool = False,
        **kwargs,
    ):
        super().__init__(memory_type="agent_kb", clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._storage_dir / "agent_kb.json"
        self._top_k = top_k
        self._lock = threading.RLock()
        self._tls = threading.local()

        self._llm = _ArkLLMCallable(ark_api_key, ark_model)
        self._embedder = _DashscopeEmbedClient(dashscope_api_key, dashscope_base_url)

        if clear and self._db_path.exists():
            self._db_path.unlink()

        self._db: List[dict] = self._load_db()
        self._index: dict = self._build_index()

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #

    def _load_db(self) -> List[dict]:
        if self._db_path.exists():
            try:
                with open(self._db_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_db(self) -> None:
        with open(self._db_path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, indent=2, ensure_ascii=False)

    def _build_index(self) -> dict:
        """Build TF-IDF and semantic embedding indices over current DB."""
        if not self._db:
            return {"tfidf": None, "vectorizer": None, "embeddings": None}

        from sklearn.feature_extraction.text import TfidfVectorizer

        queries = [item.get("query", "") for item in self._db]

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = None
        try:
            tfidf = vectorizer.fit_transform(queries)
        except Exception:
            vectorizer = None

        # Use cached embeddings when available (avoid re-embedding on every reload).
        cached = [item.get("_embedding") for item in self._db]
        if all(e is not None for e in cached):
            embeddings = np.array(cached, dtype=np.float32)
        else:
            embeddings = self._embedder.embed_batch(queries)
            for i, item in enumerate(self._db):
                item["_embedding"] = embeddings[i].tolist()
            self._save_db()

        return {"tfidf": tfidf, "vectorizer": vectorizer, "embeddings": embeddings}

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def _hybrid_search(self, query: str) -> List[dict]:
        if not self._db:
            return []

        from sklearn.metrics.pairwise import cosine_similarity as sk_cos

        idx = self._index
        scores: Dict[int, float] = defaultdict(float)

        if idx["tfidf"] is not None and idx["vectorizer"] is not None:
            try:
                qvec = idx["vectorizer"].transform([query])
                tfidf_s = sk_cos(qvec, idx["tfidf"]).flatten()
                for i, s in enumerate(tfidf_s):
                    scores[i] += 0.5 * float(s)
            except Exception:
                pass

        if idx["embeddings"] is not None:
            qemb = self._embedder.embed(query)
            sem_s = _cosine_sim_matrix(qemb, idx["embeddings"])
            for i, s in enumerate(sem_s):
                scores[i] += 0.5 * float(s)

        top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[: self._top_k]
        return [self._db[i] for i in top_ids if scores[i] > 0]

    # ------------------------------------------------------------------ #
    # LLM synthesis                                                        #
    # ------------------------------------------------------------------ #

    def _synthesize_guidance(self, results: List[dict], query: str) -> str:
        if not results:
            return ""
        try:
            content = "\n\n".join(
                f"Similar task: {r.get('query', '')}\nGuidance: {r.get('agent_planning', '')}"
                for r in results
                if r.get("agent_planning")
            )
            if not content:
                return ""
            prompt = (
                f"Based on similar past tasks, provide 2-3 concise, actionable suggestions "
                f"for:\nTask: {query}\n\nSimilar experiences:\n{content}\n\n"
                f"Provide numbered suggestions only (no explanations):"
            )
            resp = self._llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            guidance = getattr(resp, "content", "").strip()
            return guidance if guidance else ""
        except Exception:
            plans = [r.get("agent_planning", "") for r in results if r.get("agent_planning")]
            return "; ".join(plans[:2])

    # ------------------------------------------------------------------ #
    # Trajectory ingestion                                                 #
    # ------------------------------------------------------------------ #

    def _summarize_trajectory(self, query: str, trajectory: list) -> Optional[dict]:
        try:
            traj_text = "\n".join(
                f"[{m.get('role', '')}]: {str(m.get('content', ''))[:400]}"
                for m in trajectory
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            )[:3000]

            prompt = (
                f"Analyze this task execution and extract reusable workflow knowledge.\n\n"
                f"Task: {query}\n\nExecution:\n{traj_text}\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"agent_planning": "2-3 sentence strategic plan for similar tasks", '
                f'"agent_experience": "2-3 key lessons discovered"}}'
            )
            resp = self._llm([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            text = getattr(resp, "content", "").strip()

            import re

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*?\}", text, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def begin_sample(self) -> None:
        super().begin_sample()
        self._tls.trajectory = []
        self._tls.query = ""

    def utilize(self, user_text: str) -> str:
        try:
            with self._lock:
                results = self._hybrid_search(user_text)

            if not results:
                u = getattr(metrics._TLS, "usage", None)
                if u is not None:
                    u.n_utilize += 1
                return ""

            guidance = self._synthesize_guidance(results, user_text)

            u = getattr(metrics._TLS, "usage", None)
            if u is not None:
                u.n_utilize += 1

            return (_UTILIZE_PREFIX + guidance) if guidance else ""
        except Exception:
            return ""

    def update(self, trajectory: list[dict]) -> None:
        if self.readonly:
            return
        traj = getattr(self._tls, "trajectory", [])
        traj.extend(trajectory)
        self._tls.trajectory = traj

        if not getattr(self._tls, "query", ""):
            for msg in trajectory:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    self._tls.query = str(content)[:500]
                    break

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    def drain_usage(self) -> MemoryUsageLog:
        trajectory = getattr(self._tls, "trajectory", [])
        query = getattr(self._tls, "query", "")

        if trajectory and query and not self.readonly:
            try:
                summary = self._summarize_trajectory(query, trajectory)
                if summary and (summary.get("agent_planning") or summary.get("agent_experience")):
                    entry = {
                        "query": query,
                        "agent_planning": summary.get("agent_planning", ""),
                        "search_agent_planning": "",
                        "agent_experience": summary.get("agent_experience", ""),
                        "search_agent_experience": "",
                    }
                    with self._lock:
                        self._db.append(entry)
                        # Embed the new entry's query and cache it
                        entry["_embedding"] = self._embedder.embed(query).tolist()
                        self._save_db()
                        self._index = self._build_index()
            except Exception:
                pass

        self._tls.trajectory = []
        self._tls.query = ""
        return metrics.drain()

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "AgentKBMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)
