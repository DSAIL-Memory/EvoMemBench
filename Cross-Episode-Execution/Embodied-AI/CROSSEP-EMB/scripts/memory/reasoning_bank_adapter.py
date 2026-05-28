from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy

import numpy as np

from .base import BaseMemory, MemoryCallStats

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class ReasoningBankMemory(BaseMemory):
    """Memory backend using ReasoningBank: JSONL memory store with embedding-based retrieval.

    Inject: embed current task query → cosine similarity against stored queries → top-k items.
    Update: LLM induction (success/fail prompt) → parse memory items → append to JSONL bank.
    Embeddings: DashScope text-embedding-v4 via OpenAI-compatible API.
    LLM: volcengine ARK batch API.
    """

    _DEFAULT_SUCCESSFUL_PROMPT_PATH = os.path.join(
        _SCRIPTS_DIR, "prompts", "alfworld_reasoning_bank_successful_prompt.txt"
    )
    _DEFAULT_FAILED_PROMPT_PATH = os.path.join(
        _SCRIPTS_DIR, "prompts", "alfworld_reasoning_bank_failed_prompt.txt"
    )
    _DEFAULT_JUDGE_PROMPT_PATH = os.path.join(
        _SCRIPTS_DIR, "prompts", "reasoning_bank_judge_prompt.txt"
    )

    def __init__(
        self,
        read_only: bool = False,
        bank_path: str = "reasoning_bank.jsonl",
        embeddings_path: str = "embeddings.jsonl",
        topk: int = 1,
        ark_batch_config: dict | None = None,
        embedder_config: dict | None = None,
        successful_prompt_path: str | None = None,
        failed_prompt_path: str | None = None,
        judge_prompt_path: str | None = None,
    ):
        super().__init__(memory_type="reasoning_bank", read_only=read_only)

        self._bank_path = bank_path
        self._embeddings_path = embeddings_path
        self._topk = topk

        os.makedirs(os.path.dirname(os.path.abspath(bank_path)) or ".", exist_ok=True)

        # ARK batch API client
        self._ark_client = None
        self._batch_model = None
        if ark_batch_config:
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError as e:
                raise ImportError(
                    "ark_batch_config requires volcengine SDK. Run:\n"
                    "  pip install 'volcengine-python-sdk[ark]'"
                ) from e
            self._ark_client = Ark(api_key=ark_batch_config["api_key"])
            self._batch_model = ark_batch_config["model"]

        # DashScope embedder config
        self._embedder_api_key = None
        self._embedder_base_url = None
        self._embedder_model = "text-embedding-v4"
        if embedder_config:
            self._embedder_api_key = embedder_config.get("api_key")
            self._embedder_base_url = embedder_config.get("base_url")
            self._embedder_model = embedder_config.get("model", "text-embedding-v4")

        # Load prompt templates
        sp = successful_prompt_path or self._DEFAULT_SUCCESSFUL_PROMPT_PATH
        fp = failed_prompt_path or self._DEFAULT_FAILED_PROMPT_PATH
        jp = judge_prompt_path or self._DEFAULT_JUDGE_PROMPT_PATH
        with open(sp) as f:
            self._successful_prompt = f.read()
        with open(fp) as f:
            self._failed_prompt = f.read()
        with open(jp) as f:
            self._judge_prompt = f.read()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str, stats: MemoryCallStats) -> list[float] | None:
        """Call DashScope text-embedding-v4 and return the embedding vector."""
        if not self._embedder_api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self._embedder_api_key,
                base_url=self._embedder_base_url,
            )
            response = client.embeddings.create(model=self._embedder_model, input=text)
            stats.embedding_tokens += _count_tokens(text)
            return response.data[0].embedding
        except Exception as e:
            print(f"[ReasoningBank] Embedding API error: {e}")
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def _load_bank(self) -> list[dict]:
        if not os.path.exists(self._bank_path):
            return []
        records = []
        with open(self._bank_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _load_embeddings(self) -> list[dict]:
        if not os.path.exists(self._embeddings_path):
            return []
        records = []
        with open(self._embeddings_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _append_to_jsonl(self, path: str, record: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_task_instruction(self, conversation: list[dict]) -> str:
        """Extract task instruction from the first env observation (conversation[2])."""
        if len(conversation) >= 3:
            content = conversation[2].get("content", "")
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith("your task is to"):
                    return line
            if content.strip():
                return content.strip()[:500]
        # fallback: first non-empty user message
        for msg in conversation:
            if msg.get("role") == "user" and msg.get("content", "").strip():
                return msg["content"].strip()[:500]
        return ""

    def _judge_success(self, conversation: list[dict], stats: MemoryCallStats) -> bool:
        """Use LLM-as-a-judge to determine whether the trajectory succeeded.

        Keeps ground-truth env signals out of the memory module, per EvoMemBench convention.
        Token usage is accumulated into the provided stats (counted as part of update cost).
        """
        task = self._extract_task_instruction(conversation)
        trajectory_text = self._format_trajectory(conversation)
        user_content = f"Task: {task}\n\nTrajectory:\n{trajectory_text}"
        raw_output = self._call_llm(self._judge_prompt, user_content, stats)
        for line in raw_output.splitlines():
            if line.startswith("Status:"):
                verdict = line[len("Status:"):].strip().strip('"').lower()
                return verdict == "success"
        print(f"[ReasoningBank judge] Failed to parse Status from output: {raw_output[:200]!r}")
        return False

    def _format_trajectory(self, conversation: list[dict]) -> str:
        lines = []
        for i, msg in enumerate(conversation):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{i}] {role}: {content}")
        return "\n\n".join(lines)

    def _parse_memory_items(self, text: str) -> list[str]:
        """Split LLM Markdown output into individual memory item strings."""
        if not text:
            return []
        # Split on "# Memory Item N" headers
        parts = re.split(r"(?m)^#\s+Memory Item\s+\d+", text)
        items = []
        for part in parts:
            part = part.strip()
            if part and ("## Title" in part or "## Content" in part):
                items.append("# Memory Item " + str(len(items) + 1) + "\n" + part)
        # If no structured headers found, return whole text as one item
        if not items and text.strip():
            items = [text.strip()]
        return items

    def _call_llm(self, system_prompt: str, user_content: str, stats: MemoryCallStats) -> str:
        if self._ark_client is None:
            raise RuntimeError("ark_batch_config not provided; cannot call LLM")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_retries, retry_delay = 8, 5.0
        for attempt in range(max_retries):
            try:
                response = self._ark_client.batch.chat.completions.create(
                    model=self._batch_model,
                    messages=messages,
                )
                if response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    if (hasattr(response.usage, "prompt_tokens_details")
                            and response.usage.prompt_tokens_details):
                        stats.cached_tokens += getattr(
                            response.usage.prompt_tokens_details, "cached_tokens", 0
                        ) or 0
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[ReasoningBank LLM retry {attempt + 1}/{max_retries}] {e}. "
                      f"Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
        return ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inject(self, conversation: list[dict]) -> tuple[list[dict], MemoryCallStats]:
        stats = MemoryCallStats()
        t0 = time.time()

        query = self._extract_task_instruction(conversation)
        if not query:
            stats.latency = time.time() - t0
            return conversation, stats

        bank = self._load_bank()
        if not bank:
            stats.latency = time.time() - t0
            return conversation, stats

        emb_cache = self._load_embeddings()
        if not emb_cache:
            stats.latency = time.time() - t0
            return conversation, stats

        # Get query embedding
        q_emb = self._get_embedding(query, stats)
        if q_emb is None:
            stats.latency = time.time() - t0
            return conversation, stats

        # Build {task_id -> embedding} map from cache
        id2emb = {str(rec["task_id"]): rec["embedding"] for rec in emb_cache if "embedding" in rec}

        # Score each bank record
        scored = []
        for rec in bank:
            tid = str(rec.get("task_id", ""))
            if tid in id2emb:
                sim = self._cosine_similarity(q_emb, id2emb[tid])
                scored.append((sim, rec))

        if not scored:
            stats.latency = time.time() - t0
            return conversation, stats

        scored.sort(key=lambda x: x[0], reverse=True)
        top_records = [rec for _, rec in scored[: self._topk]]

        # Format memory items
        mem_parts = []
        for rec in top_records:
            items = rec.get("memory_items", [])
            if isinstance(items, list):
                mem_parts.extend(items)
            elif isinstance(items, str):
                mem_parts.append(items)

        if not mem_parts:
            stats.latency = time.time() - t0
            return conversation, stats

        injection = "\n\n--- Retrieved Memories ---\n" + "\n\n".join(mem_parts) + "\n---"
        new_conv = deepcopy(conversation)
        new_conv[0] = dict(new_conv[0])
        new_conv[0]["content"] = new_conv[0]["content"] + injection

        stats.latency = time.time() - t0
        return new_conv, stats

    def update(self, conversation: list[dict], data_idx: int | None = None, reward: float | None = None) -> MemoryCallStats:
        if self.read_only:
            return MemoryCallStats()

        stats = MemoryCallStats()
        t0 = time.time()

        query = self._extract_task_instruction(conversation)
        success = self._judge_success(conversation, stats)
        status = "success" if success else "fail"

        # Choose system prompt
        system_prompt = self._successful_prompt if success else self._failed_prompt

        # Format trajectory as user content
        trajectory_text = self._format_trajectory(conversation)
        user_content = f"**Task:** {query}\n\n**Trajectory:**\n{trajectory_text}"

        # Call LLM to induce memory items
        raw_output = self._call_llm(system_prompt, user_content, stats)
        memory_items = self._parse_memory_items(raw_output)

        if not memory_items:
            print(f"[ReasoningBank] No memory items extracted for data_idx={data_idx}")
        else:
            print(f"[ReasoningBank] Extracted {len(memory_items)} memory item(s) "
                  f"(status={status}, data_idx={data_idx})")

        # Get embedding for the query
        q_emb = self._get_embedding(query, stats)

        # Append to bank JSONL
        bank_record = {
            "task_id": data_idx,
            "query": query,
            "status": status,
            "memory_items": memory_items,
        }
        self._append_to_jsonl(self._bank_path, bank_record)

        # Append to embeddings JSONL
        if q_emb is not None:
            emb_record = {
                "task_id": data_idx,
                "text": query,
                "embedding": q_emb,
            }
            self._append_to_jsonl(self._embeddings_path, emb_record)

        stats.latency = time.time() - t0
        return stats

    def load_from_disk(self, **kwargs) -> None:
        # JSONL files are read fresh on each inject/update call.
        pass
