"""ReasoningBankMemory — ReasoningBank backend for the BFCL Memory interface.

Implements the no-ground-truth adaptation mode (SWE-Bench path):
  1. utilize: embed current system_prompt (instruction-wrapped) → cosine top-1 against
     stored entry embeddings → inject that entry's memory_items into the system prompt.
  2. update: LLM-judge (success/fail) → induce memory_items from trajectory
     (SUCCESSFUL or FAILED variant) → cache unwrapped system_prompt embedding → append
     to bank.jsonl.

Top-k is fixed at 1 (per the original swebench.py:182 call site).
All LLM calls go through ArkBatchClient (Volcengine batch endpoint).
Embeddings use the DashScope OpenAI-compatible endpoint directly (no mem0 dep).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed, wait_random

from bfcl_eval.memory import metrics
from bfcl_eval.memory.ark_client import ArkBatchClient
from bfcl_eval.memory.base import Memory

logger = logging.getLogger(__name__)

_BANK_FILENAME = "bank.jsonl"
_UTILIZE_PREFIX = (
    "\n\nBelow are some memory items that I accumulated from past interaction "
    "from the environment that may be helpful to solve the task. You can use "
    "it when you feel it's relevant. In each step, please first explicitly "
    "discuss if you want to use each memory item or not, and then take action.\n\n"
)
_TASK_INSTRUCTION = (
    "Given the prior tool-use task descriptions, your task is to analyze a "
    "current task's intent and select relevant prior tasks that could help resolve it."
)
_MAX_PARSE_RETRIES = 3
_MAX_TOOL_ARG_CHARS = 2000

_PROMPTS_DIR = Path(__file__).parent / "reasoning_bank_prompts"


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(t in name for t in ("ratelimit", "timeout", "apiconnection", "apierror")):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in ("rate limit", "timeout", "timed out", "connection", "503", "overloaded"))


_EMBED_RETRY = retry(
    wait=wait_fixed(2) + wait_random(0, 1),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


@lru_cache(maxsize=3)
def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _extract_task_text(trajectory: list[dict]) -> str:
    """First user message content; fallback to system; fallback to placeholder."""
    for msg in trajectory:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts)
            if content:
                return str(content)
    for msg in trajectory:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                return str(content)
    return "[task description not available]"


def _extract_clean_system_prompt(trajectory: list[dict]) -> str:
    """First system message, with any injected memory prefix stripped."""
    for msg in trajectory:
        if msg.get("role") == "system":
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts)
            content = str(content)
            # Strip anything injected by utilize()
            if _UTILIZE_PREFIX in content:
                content = content.split(_UTILIZE_PREFIX)[0]
            return content
    return ""


def _serialize_trajectory(trajectory: list[dict]) -> str:
    """Human-readable flat representation of the conversation (tool_call args truncated)."""
    lines: list[str] = []
    for i, msg in enumerate(trajectory):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(parts)
        content = str(content)

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_parts: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "")
                if isinstance(args_raw, dict):
                    args_raw = json.dumps(args_raw)
                if len(args_raw) > _MAX_TOOL_ARG_CHARS:
                    args_raw = args_raw[:_MAX_TOOL_ARG_CHARS] + "…[truncated]"
                tc_parts.append(f"{name}({args_raw})")
            lines.append(f"[{i}] {role}: {content} TOOL_CALLS: {' | '.join(tc_parts)}")
        else:
            lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


def _trajectory_id(trajectory_text: str) -> str:
    return hashlib.sha256(trajectory_text.encode()).hexdigest()[:16]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class ReasoningBankMemory(Memory):
    """Cross-episode memory backend using ReasoningBank (no-GT mode).

    Storage layout under storage_dir/:
        bank.jsonl — one JSON line per past episode:
            {task_id, query, memory_items: list[str], status: "success"|"fail",
             embedding: list[float]}

    Retrieval: embed current system_prompt with instruction-aware wrap → cosine top-1
    against stored embeddings → inject that entry's memory_items.
    Top-k is fixed at 1.
    LLM calls go through ArkBatchClient; embeddings use DashScope OpenAI-compat endpoint.
    """

    def __init__(
        self,
        *,
        storage_dir: Path | str,
        ark_api_key: str,
        ark_model: str,
        dashscope_api_key: str,
        dashscope_base_url: str,
        embed_model: str = "text-embedding-v4",
        embed_dims: int = 1024,
        clear: bool = False,
        readonly: bool = False,
    ):
        super().__init__(clear=clear, readonly=readonly)

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._bank_path = self._storage_dir / _BANK_FILENAME

        self._model = ark_model
        self._embed_model = embed_model
        self._embed_dims = embed_dims
        self._ark_client = ArkBatchClient(api_key=ark_api_key)
        self._embed_client = OpenAI(api_key=dashscope_api_key, base_url=dashscope_base_url)
        self._lock = threading.RLock()

        if clear:
            self._bank: list[dict] = []
            self._write_bank()
        elif self._bank_path.exists():
            self._bank = self._load_bank()
            self._validate_embed_dims()
        else:
            self._bank = []

    # ------------------------------------------------------------------ #
    # Memory interface                                                     #
    # ------------------------------------------------------------------ #

    def utilize(self, system_prompt: str) -> str:
        """Embed system_prompt, retrieve top-1 memory entry, return injected snippet.

        Returns '' when the bank is empty.
        """
        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_utilize += 1

        if not self._bank:
            return ""

        wrapped = f"Instruct: {_TASK_INSTRUCTION}\nQuery: {system_prompt}"
        q_vec = _l2_normalize(np.array(self._embed_one(wrapped), dtype=np.float32))

        bank_vecs = np.array([e["embedding"] for e in self._bank], dtype=np.float32)
        norms = np.linalg.norm(bank_vecs, axis=1, keepdims=True)
        bank_vecs_norm = np.where(norms > 0, bank_vecs / norms, bank_vecs)

        scores = bank_vecs_norm @ q_vec
        best_idx = int(np.argmax(scores))
        entry = self._bank[best_idx]

        memory_text = "\n\n".join(item for item in entry["memory_items"] if item.strip())
        if not memory_text:
            return ""
        return _UTILIZE_PREFIX + memory_text

    def update(self, trajectory: list[dict]) -> None:
        """Run judge → induce → embed → write bank entry from this trajectory."""
        if self.readonly:
            return

        with self._lock:
            try:
                self._run_induction(trajectory)
            except Exception:
                logger.exception(
                    "ReasoningBankMemory.update failed for trajectory of length %d; bank unchanged",
                    len(trajectory),
                )

        u = getattr(metrics._TLS, "usage", None)
        if u is not None:
            u.n_update += 1

    @classmethod
    def load_from_disk(cls, **backend_kwargs) -> "ReasoningBankMemory":
        backend_kwargs.setdefault("clear", False)
        return cls(**backend_kwargs)

    # ------------------------------------------------------------------ #
    # Induction pipeline                                                   #
    # ------------------------------------------------------------------ #

    def _run_induction(self, trajectory: list[dict]) -> None:
        task_text = _extract_task_text(trajectory)
        clean_sys = _extract_clean_system_prompt(trajectory)
        trajectory_text = _serialize_trajectory(trajectory)
        task_id = _trajectory_id(trajectory_text)

        status_bool = self._judge_call(task_text, trajectory_text)
        memory_items = self._induce_call(task_text, trajectory_text, status_bool)

        if not memory_items:
            logger.debug("ReasoningBank: induction produced no memory items for task %s; skipping write", task_id)
            return

        query_text = clean_sys or task_text
        embedding = self._embed_one(query_text)

        entry = {
            "task_id": task_id,
            "query": task_text,
            "memory_items": memory_items,
            "status": "success" if status_bool else "fail",
            "embedding": embedding,
        }
        self._bank.append(entry)
        self._write_bank()

    def _judge_call(self, task_text: str, trajectory_text: str) -> bool:
        template = _load_prompt("bfcl_judge_prompt.txt")
        filled = template.replace("{task}", task_text).replace("{trajectory}", trajectory_text)
        messages = [{"role": "user", "content": filled}]

        for attempt in range(_MAX_PARSE_RETRIES):
            try:
                resp, dt = self._ark_client.chat_completions(self._model, messages)
                metrics.record_llm_call(
                    latency_s=dt,
                    input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
                content = (resp.choices[0].message.content or "").lower()
                return "success" in content
            except Exception:
                logger.exception("ReasoningBank judge: LLM call failed on attempt %d", attempt + 1)
        return False

    def _induce_call(self, task_text: str, trajectory_text: str, success: bool) -> list[str]:
        filename = "bfcl_successful_induce_prompt.txt" if success else "bfcl_failed_induce_prompt.txt"
        template = _load_prompt(filename)
        filled = template.replace("{task}", task_text).replace("{trajectory}", trajectory_text)
        messages = [{"role": "user", "content": filled}]

        for attempt in range(_MAX_PARSE_RETRIES):
            try:
                resp, dt = self._ark_client.chat_completions(self._model, messages)
                metrics.record_llm_call(
                    latency_s=dt,
                    input_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
                raw = (resp.choices[0].message.content or "").strip()
                items = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]
                if items:
                    return items
                logger.debug(
                    "ReasoningBank induce: attempt %d/%d produced empty items",
                    attempt + 1, _MAX_PARSE_RETRIES,
                )
            except Exception:
                logger.exception("ReasoningBank induce: LLM call failed on attempt %d", attempt + 1)
        return []

    # ------------------------------------------------------------------ #
    # Embedding                                                            #
    # ------------------------------------------------------------------ #

    @_EMBED_RETRY
    def _embed_one(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        t0 = time.time()
        resp = self._embed_client.embeddings.create(
            model=self._embed_model,
            input=[text],
            dimensions=self._embed_dims,
            encoding_format="float",
        )
        dt = time.time() - t0
        total_tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
        metrics.record_embed_call(total_tokens, latency_s=dt)
        return resp.data[0].embedding

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _load_bank(self) -> list[dict]:
        bank: list[dict] = []
        with open(self._bank_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    bank.append(json.loads(raw))
                except Exception:
                    logger.warning("ReasoningBank: skipping malformed bank line: %r", raw[:120])
        return bank

    def _write_bank(self) -> None:
        """Atomically rewrite bank.jsonl from self._bank."""
        tmp = self._bank_path.parent / (self._bank_path.name + ".tmp")
        with open(tmp, "w") as f:
            for entry in self._bank:
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp, self._bank_path)

    def _validate_embed_dims(self) -> None:
        for entry in self._bank:
            emb = entry.get("embedding")
            if emb and len(emb) != self._embed_dims:
                logger.warning(
                    "ReasoningBank: loaded bank entry has embedding dim %d, expected %d; "
                    "retrieval scores may be incorrect",
                    len(emb), self._embed_dims,
                )
                break
