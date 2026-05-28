"""ReasoningBank memory provider for EvoMemBench-XbenchWebWalkQA.

Algorithm (ported from reasoning-bank/third_party/src/minisweagent/):
- take_in_memory: distils each completed trajectory into generalizable reasoning items
  via SUCCESSFUL_SI or FAILED_SI prompt, then appends to a JSONL bank + embedding cache.
- provide_memory: embeds the current task query, retrieves the top-k nearest prior
  trajectories by cosine similarity, and returns their distilled items as a text block.

Key deviations from the original SWE-Bench integration:
- DashScope text-embedding-v4 (OpenAI-compatible) replaces Gemini/Vertex embeddings.
- Success/failure is read from TrajectoryData.metadata["is_correct"] (benchmark ground-
  truth, already computed before take_in_memory is called) instead of LLM-as-judge.
  Set use_llm_judge=True in config to restore the original judge as a fallback.
- Asymmetric instruction-prefix embedding is not used (DashScope has no task_type knob).
- LLM induction uses volcenginesdkarkruntime.Ark + batch.chat.completions (same as graphrag_provider).
"""

import json
import os
import threading
import traceback
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)

# ---------------------------------------------------------------------------
# Prompt constants — verbatim from instruction.py (Apache-2.0, Google LLC)
# ---------------------------------------------------------------------------

SUCCESSFUL_SI = """
You are an expert in coding, specifically fixing a given issue in a code repository. You will be given an issue to be fixed, the corresponding trajectory that represents **how an agent successfully resolved the issue**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""

FAILED_SI = """
You are an expert in coding, specifically fixing a given issue in a code repository. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the issue but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""

# Judge prompt — verbatim from swebench.py:249, 256
_JUDGE_SYSTEM = "You are a helpful assistant that judges whether the agent successfully completed the task."
_JUDGE_PROMPT_TMPL = (
    "Task: {task}\n\nTrajectory:\n{trajectory}\n\n"
    "Did the agent successfully complete the task? Answer with 'success' or 'fail' only."
)

# Injection preamble — verbatim from agents/default.py:75-77
_MEMORY_PREAMBLE = (
    "Below are some memory items that I accumulated from past interaction from the "
    "environment that may be helpful to solve the task. You can use it when you feel "
    "it's relevant. In each step, please first explicitly discuss if you want to use "
    "each memory item or not, and then take action.\n\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_content(content: Any) -> str:
    """Coerce OpenAI message content to str (handles list-of-parts for multi-modal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ReasoningBankProvider(BaseMemoryProvider):
    """ReasoningBank: learn from successful/failed trajectories via LLM distillation."""

    def __init__(self, config: dict = None):
        super().__init__(MemoryType.REASONING_BANK, config or {})
        self._bank_path = self.config.get("bank_path", "./storage/reasoning_bank/bank.jsonl")
        self._embeddings_path = self.config.get("embeddings_path", "./storage/reasoning_bank/embeddings.jsonl")
        self._top_k = self.config.get("top_k", 1)
        self._max_items = self.config.get("max_items_per_trajectory", 3)
        self._return_on_in = self.config.get("return_on_in", False)
        self._use_llm_judge = self.config.get("use_llm_judge", False)
        self._lock = threading.RLock()

        self._llm_client = None
        self._llm_model: str = ""
        self._embed_client = None
        self._embed_model: str = ""

        self._bank: List[dict] = []
        self._cache_ids: List[str] = []
        self._cache_matrix: Optional[np.ndarray] = None  # (N, D) L2-normalized

    # ── initialize ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        try:
            from openai import OpenAI
        except ImportError as e:
            print(f"[reasoning_bank] openai package not found: {e}")
            return False

        llm_cfg = self.config.get("llm", {}) or {}
        embed_cfg = self.config.get("embed", {}) or {}

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._bank_path)), exist_ok=True)
            for path in (self._bank_path, self._embeddings_path):
                if not os.path.exists(path):
                    open(path, "w").close()

            # LLM client — Ark batch inference (same pattern as graphrag_provider)
            self._llm_model = llm_cfg.get("model") or os.getenv("BATCH_MODEL", "")
            if not self._llm_model:
                print("[reasoning_bank] WARNING: BATCH_MODEL not set — memory induction will fail")
            try:
                from volcenginesdkarkruntime import Ark
                self._llm_client = Ark(
                    api_key=llm_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
                )
            except ImportError:
                print("[reasoning_bank] WARNING: volcenginesdkarkruntime not installed — "
                      "memory induction disabled; install 'volcengine-python-sdk[ark]'")
                self._llm_client = None

            # Embedding client (DashScope text-embedding-v4)
            self._embed_model = embed_cfg.get("model", "text-embedding-v4")
            self._embed_client = OpenAI(
                api_key=embed_cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY", ""),
                base_url=embed_cfg.get("base_url") or os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )

            # Load persisted state
            self._bank = self._load_bank()
            self._cache_ids, self._cache_matrix = self._load_embeddings()

            print(
                f"[reasoning_bank] initialized (top_k={self._top_k}, "
                f"return_on_in={self._return_on_in}, use_llm_judge={self._use_llm_judge}, "
                f"bank_size={len(self._bank)}, cached_embeddings={len(self._cache_ids)})"
            )
            return True

        except Exception as e:
            print(f"[reasoning_bank] initialize failed: {e}")
            traceback.print_exc()
            return False

    # ── provide_memory ────────────────────────────────────────────────────────

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._lock:
            empty = MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0)

            if self._embed_client is None:
                return empty
            if request.status == MemoryStatus.IN and not self._return_on_in:
                return empty
            if not self._bank or self._cache_matrix is None or len(self._cache_ids) == 0:
                return empty

            try:
                q_vec = self._embed(request.query)  # (1, D)
                scores = (q_vec @ self._cache_matrix.T).squeeze(0)  # (N,)
                k = min(self._top_k, len(self._cache_ids))
                top_indices = np.argsort(scores)[::-1][:k]

                items: List[MemoryItem] = []
                for idx in top_indices:
                    tid = self._cache_ids[idx]
                    score = float(scores[idx])
                    bank_entry = next(
                        (b for b in self._bank if b.get("task_id") == tid), None
                    )
                    if bank_entry is None:
                        continue

                    mem_parts = bank_entry.get("memory_items", [])
                    if not mem_parts:
                        continue

                    content = _MEMORY_PREAMBLE + "\n\n".join(mem_parts)
                    items.append(
                        MemoryItem(
                            id=tid,
                            content=content,
                            metadata={"task_id": tid, "status": bank_entry.get("status", "")},
                            score=score,
                            type=MemoryItemType.TEXT,
                        )
                    )

                return MemoryResponse(memories=items, memory_type=self.memory_type, total_count=len(items))

            except Exception as e:
                print(f"[reasoning_bank] provide_memory failed: {e}")
                return empty

    # ── take_in_memory ────────────────────────────────────────────────────────

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        with self._lock:
            if self._llm_client is None or self._embed_client is None:
                return False, "reasoning_bank not initialized"

            try:
                meta = trajectory_data.metadata or {}
                task_id = meta.get("task_id") or uuid.uuid4().hex
                query = trajectory_data.query or ""

                # Build trajectory text (exclude system messages, coerce multi-modal)
                traj_parts = []
                for msg in (trajectory_data.trajectory or []):
                    if msg.get("role") == "system":
                        continue
                    text = _coerce_content(msg.get("content", ""))
                    if text.strip():
                        traj_parts.append(text)

                if not traj_parts:
                    text = query
                    if trajectory_data.result:
                        text += f"\nResult: {trajectory_data.result}"
                    if not text.strip():
                        return False, "empty trajectory"
                    traj_parts = [text]

                trajectory_text = "\n".join(traj_parts)

                # Determine success/failure
                is_correct = meta.get("is_correct")
                if isinstance(is_correct, bool):
                    status_bool = is_correct
                elif meta.get("status") == "success":
                    status_bool = True
                elif self._use_llm_judge:
                    status_bool = self._llm_judge(query, trajectory_text)
                else:
                    status_bool = False

                status_str = "success" if status_bool else "fail"
                si = SUCCESSFUL_SI if status_bool else FAILED_SI

                # Distil trajectory into reasoning items (verbatim algorithm from swebench.py)
                user_msg = f"**Query:** {query}\n\n**Trajectory:**\n{trajectory_text}"
                response = self._llm_client.batch.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": si.strip()},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=1.0,
                    max_tokens=4096,
                )
                raw = response.choices[0].message.content or ""
                memory_items = [p.strip() for p in raw.split("\n\n") if p.strip()]
                memory_items = memory_items[: self._max_items]

                if not memory_items:
                    return False, "no memory items extracted from trajectory"

                # Persist to bank JSONL
                bank_record = {
                    "task_id": task_id,
                    "query": query,
                    "memory_items": memory_items,
                    "status": status_str,
                }
                with open(self._bank_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(bank_record, ensure_ascii=False) + "\n")
                self._bank.append(bank_record)

                # Compute and persist embedding
                q_vec = self._embed(query)  # (1, D) normalized
                emb_record = {
                    "id": task_id,
                    "text": query,
                    "embedding": q_vec.squeeze(0).tolist(),
                }
                with open(self._embeddings_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(emb_record, ensure_ascii=False) + "\n")

                # Update in-memory cache
                self._cache_ids.append(task_id)
                if self._cache_matrix is None:
                    self._cache_matrix = q_vec
                else:
                    self._cache_matrix = np.vstack([self._cache_matrix, q_vec])

                return True, f"stored {len(memory_items)} memory items, status={status_str}"

            except Exception as e:
                return False, f"take_in_memory error: {e}"

    # ── private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed text and return L2-normalized (1, D) float32 array."""
        resp = self._embed_client.embeddings.create(model=self._embed_model, input=[text])
        vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return _l2_normalize(vec)

    def _llm_judge(self, task: str, trajectory: str) -> bool:
        """LLM-as-judge fallback — verbatim logic from swebench.py:248-263."""
        prompt = _JUDGE_PROMPT_TMPL.format(task=task, trajectory=trajectory)
        response = self._llm_client.batch.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        text = (response.choices[0].message.content or "").strip().lower()
        return "success" in text

    def _load_bank(self) -> List[dict]:
        bank = []
        try:
            with open(self._bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bank.append(json.loads(line))
        except FileNotFoundError:
            pass
        return bank

    def _load_embeddings(self) -> Tuple[List[str], Optional[np.ndarray]]:
        ids: List[str] = []
        vecs: List[np.ndarray] = []
        try:
            with open(self._embeddings_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ids.append(str(rec["id"]))
                    vecs.append(np.array(rec["embedding"], dtype=np.float32))
        except FileNotFoundError:
            pass

        if not vecs:
            return ids, None

        matrix = np.stack(vecs, axis=0)  # (N, D)
        matrix = _l2_normalize(matrix)   # normalize on load (mirrors memory_management.py:114)
        return ids, matrix
