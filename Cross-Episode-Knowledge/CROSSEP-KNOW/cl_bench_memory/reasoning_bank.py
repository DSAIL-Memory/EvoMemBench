"""ReasoningBank: binary-judged, extract-on-success/failure, query-embedding retrieval.

Port of the SWE-Bench path from reasoning-bank's original implementation:
  1. Binary LLM judge (success|fail, temp 0.0) to decide which extract prompt to use.
  2. LLM extraction using SUCCESSFUL_EXTRACT_PROMPT or FAILED_EXTRACT_PROMPT.
  3. Query embedding stored alongside memory items; retrieved by cosine top-k.
  4. Append-only JSONL persistence; embeddings reconciled on load.
"""

import json
import os
import re
import time

import numpy as np

from .api_client import call_openai_api
from .base import Memory
from .io import append_jsonl
from .logging_utils import log

# ── Prompts ───────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant that judges whether an AI assistant correctly "
    "answered a question based on a knowledge document provided in context."
)

JUDGE_USER_TEMPLATE = """\
Question:
{query}

Trajectory:
{trajectory}

The assistant was given a knowledge document in the conversation context and asked to answer \
the question above based solely on that document. Did the assistant correctly and completely \
answer the question based on the information in the provided context document? \
Answer with 'success' or 'fail' only."""

SUCCESSFUL_EXTRACT_PROMPT = """You are an expert in context learning and knowledge reasoning. You will be given a conversation where an AI model likely gave a CORRECT and COMPLETE response to questions based on new knowledge provided in context.

## Guidelines
Extract generalizable reasoning STRATEGIES that led to success. Focus on:
- How the model located and extracted the relevant information within a long context
- What reading or comprehension approach proved effective for this task type
- How the model identified which rules, data points, or procedures applied to the specific question
- What patterns of attention or inference helped produce a thorough, accurate response

## Important notes
  - You must first analyze the reasoning approach used, then summarize the strategies.
  - You can extract *at most 3* memory items from the conversation.
  - You must not repeat similar or overlapping items.
  - Do NOT copy specific facts or details from the context — focus on generalizable meta-level strategies.
  - Focus on HOW to reason with new knowledge, not WHAT the specific knowledge is.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the reasoning strategy or effective pattern>
```"""

FAILED_EXTRACT_PROMPT = """You are an expert in context learning and knowledge reasoning. You will be given a conversation where an AI model likely gave an INCORRECT or INCOMPLETE response to questions based on new knowledge provided in context.

## Guidelines
Extract generalizable LESSONS about what to avoid and how to improve. Focus on:
- What type of information was likely overlooked, misread, or misinterpreted in the context
- What reasoning blind spots or attention gaps led to the failure
- What risks or pitfalls are common for this type of context-learning task
- What strategy changes or extra attention points would help avoid similar failures

## Important notes
  - You must first analyze what likely went wrong, then summarize the lessons.
  - You can extract *at most 3* memory items from the conversation.
  - You must not repeat similar or overlapping items.
  - Do NOT assume you know the correct answer — focus on identifying risk factors and attention gaps.
  - Lessons should be actionable and generalizable to similar future tasks.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the lesson, risk factor, or attention gap>
```"""

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some memory items that I accumulated from past interactions "
    "that may be helpful to solve the task. You can use it when you feel it's "
    "relevant. Please first explicitly discuss if you want to use each memory "
    "item or not, and then give your response.\n\n"
)

RETRIEVAL_INSTRUCTION = (
    "Given the prior context learning experiences, your task is to analyze "
    "a current query's intent and select relevant prior experiences that "
    "could help answer it."
)

_MAX_TEXT_CHARS = 8000
_EMBED_BATCH_SIZE = 25


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def _truncate_head_tail(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-10) if norm > 0 else vec


def _sum_usage(usages: list) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["total_tokens"] += u.get("total_tokens", 0)
    return total


# ── Main class ────────────────────────────────────────────────────────────────

class ReasoningBankMemory(Memory):
    """Reasoning Bank: binary-judged LLM extraction with query-embedding retrieval.

    Faithfully ports the SWE-Bench no-GT path of the original reasoning-bank:
    one binary judge call (success|fail) → one extract call with SUCCESSFUL or FAILED
    prompt → query embedded for future retrieval. Append-only JSONL persistence.
    """

    def __init__(
        self,
        client,
        model,
        memory_dir,
        embed_provider="dashscope",
        embed_model="text-embedding-v4",
        embed_client=None,
        top_k=10,
        extract_model=None,
        **kwargs,
    ):
        if embed_provider != "dashscope":
            raise ValueError(
                f"ReasoningBankMemory only supports embed_provider='dashscope', "
                f"got {embed_provider!r}"
            )

        self.client = client
        self.embed_client = embed_client or client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_model = embed_model
        self.top_k = top_k
        self.memory_dir = memory_dir

        self.memory_bank: list = []
        self.embeddings: list = []
        self._llm_usage: list = []
        self._embed_usage: list = []

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "memory_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "embeddings.jsonl")

        self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if os.path.exists(self.bank_path):
            with open(self.bank_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.memory_bank.append(json.loads(line))

        if os.path.exists(self.embed_path):
            with open(self.embed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        vec = np.array(obj["embedding"], dtype=np.float32)
                        self.embeddings.append(_l2_normalize(vec))

        if self.memory_bank:
            log(f"   Loaded {len(self.memory_bank)} ReasoningBank entries from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"   ⚠️ Bank({len(self.memory_bank)}) vs embeddings"
                f"({len(self.embeddings)}) mismatch, re-embedding..."
            )
            queries = [e.get("query", "") for e in self.memory_bank]
            self._embed_usage.clear()
            raw = self._embed_batch(queries)
            self.embeddings = [_l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

    def _rewrite_embeddings(self) -> None:
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                entry = self.memory_bank[i]
                record = {
                    "id": entry.get("task_id", str(i)),
                    "text": entry.get("query", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Embedding ────────────────────────────────────────────────────────────

    def _embed(self, text: str, max_retries: int = 3, retry_delay: float = 3.0) -> np.ndarray:
        text = _truncate_head_tail(text, _MAX_TEXT_CHARS)
        for attempt in range(max_retries):
            try:
                response = self.embed_client.embeddings.create(
                    model=self.embed_model,
                    input=text,
                )
                if response.usage:
                    self._embed_usage.append({
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    })
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt < max_retries - 1:
                    log(f"   ⚠️ Embed failed (attempt {attempt + 1}): {str(e)[:100]}")
                    time.sleep(retry_delay)
                else:
                    raise

    def _embed_batch(self, texts: list, max_retries: int = 3, retry_delay: float = 3.0) -> list:
        if not texts:
            return []
        results = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = [_truncate_head_tail(t, _MAX_TEXT_CHARS) for t in texts[i: i + _EMBED_BATCH_SIZE]]
            for attempt in range(max_retries):
                try:
                    response = self.embed_client.embeddings.create(
                        model=self.embed_model,
                        input=batch,
                    )
                    if response.usage:
                        self._embed_usage.append({
                            "prompt_tokens": response.usage.prompt_tokens or 0,
                            "completion_tokens": 0,
                            "total_tokens": response.usage.total_tokens or 0,
                        })
                    ordered = sorted(response.data, key=lambda x: x.index)
                    results.extend(np.array(item.embedding, dtype=np.float32) for item in ordered)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        log(f"   ⚠️ Embed batch failed (attempt {attempt + 1}): {str(e)[:100]}")
                        time.sleep(retry_delay)
                    else:
                        raise
        return results

    # ── Internal logic ───────────────────────────────────────────────────────

    def _judge_response(self, query: str, trajectory: str) -> str:
        """Binary LLM judge (temp 0.0). Returns 'success' or 'fail'."""
        user_msg = JUDGE_USER_TEMPLATE.format(
            query=query or "(no query)",
            trajectory=_truncate_head_tail(trajectory, _MAX_TEXT_CHARS),
        )
        text, error, usage = call_openai_api(
            self.client,
            [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            self.extract_model,
            temperature=0.0,
        )
        if usage:
            self._llm_usage.append(usage)
        if error or not text:
            log(f"   ⚠️ Judge call failed: {error}")
            return "fail"
        m = re.search(r"\b(success|fail)\b", text.lower())
        if m:
            return m.group(1)
        log(f"   ⚠️ Judge output not parseable: {text[:80]!r}; defaulting to 'fail'")
        return "fail"

    @staticmethod
    def _parse_memory_items(text: str) -> list:
        parts = re.split(r"(?=# Memory Item\s*\d+)", text)
        return [p.strip() for p in parts if p.strip() and p.strip().startswith("# Memory Item")]

    # ── Memory ABC ───────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        self._embed_usage.clear()
        t0 = time.time()

        if not self.memory_bank:
            return "", {"latency_s": time.time() - t0, "embed_usage": _sum_usage(self._embed_usage)}

        instruct_query = get_detailed_instruct(RETRIEVAL_INSTRUCTION, query)
        qvec = _l2_normalize(self._embed(instruct_query))

        bank_matrix = np.stack(self.embeddings)
        scores = bank_matrix @ qvec * 100.0
        k = min(self.top_k, len(self.memory_bank))
        top_idx = np.argsort(-scores)[:k]

        items = []
        for idx in top_idx:
            items.extend(self.memory_bank[int(idx)].get("memory_items", []))

        text = MEMORY_INJECTION_PREFIX + "\n\n".join(items) if items else ""
        return text, {"latency_s": time.time() - t0, "embed_usage": _sum_usage(self._embed_usage)}

    def extract(self, content: str, **kwargs) -> dict:
        task_id = kwargs.get("task_id", "")
        query = kwargs.get("query", "")
        context_category = kwargs.get("context_category", "")
        sub_category = kwargs.get("sub_category", "")

        self._llm_usage.clear()
        self._embed_usage.clear()
        t0 = time.time()

        verdict = self._judge_response(query, content)
        log(f"   Judge verdict: {verdict}")

        prompt = SUCCESSFUL_EXTRACT_PROMPT if verdict == "success" else FAILED_EXTRACT_PROMPT
        text, error, usage = call_openai_api(
            self.client,
            [{"role": "system", "content": prompt},
             {"role": "user", "content": _truncate_head_tail(content, _MAX_TEXT_CHARS)}],
            self.extract_model,
            temperature=0.7,
        )
        if usage:
            self._llm_usage.append(usage)

        if error or not text:
            log(f"   ⚠️ Extract call failed: {error}")
            return {
                "latency_s": time.time() - t0,
                "llm_usage": _sum_usage(self._llm_usage),
                "embed_usage": _sum_usage(self._embed_usage),
                "verdict": verdict,
            }

        items = self._parse_memory_items(text)
        if not items:
            log("   ⚠️ No memory items parsed from extract output")
            return {
                "latency_s": time.time() - t0,
                "llm_usage": _sum_usage(self._llm_usage),
                "embed_usage": _sum_usage(self._embed_usage),
                "verdict": verdict,
            }

        instruct_query = get_detailed_instruct(RETRIEVAL_INSTRUCTION, query)
        qvec = _l2_normalize(self._embed(instruct_query))

        entry = {
            "task_id": task_id,
            "query": query,
            "memory_items": items,
            "status": verdict,
            "context_category": context_category,
            "sub_category": sub_category,
        }
        self.memory_bank.append(entry)
        self.embeddings.append(qvec)
        append_jsonl(entry, self.bank_path)
        append_jsonl({"id": task_id, "text": query, "embedding": qvec.tolist()}, self.embed_path)

        return {
            "latency_s": time.time() - t0,
            "llm_usage": _sum_usage(self._llm_usage),
            "embed_usage": _sum_usage(self._embed_usage),
            "verdict": verdict,
        }
