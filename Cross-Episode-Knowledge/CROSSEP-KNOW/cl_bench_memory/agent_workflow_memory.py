"""Agent Workflow Memory: workflow induction + embedding retrieval.

Migrated from MemEvolve/Flash-Searcher AgentWorkflowMemoryProvider.
Core idea: extract reusable, abstracted workflows from successful/failed
trajectories, store them with embeddings, and retrieve via cosine similarity.
"""

import json
import os
import time

import numpy as np

from .api_client import call_openai_api
from .base import Memory
from .io import append_jsonl
from .logging_utils import log

# ─── Workflow Induction Prompts ──────────────────────────────────────────────

SUCCESSFUL_WORKFLOW_PROMPT = """You are an expert workflow analyst for context-learning tasks. You will be given a conversation where an AI model likely gave a CORRECT and COMPLETE response.

## Goal
Extract a generic, reusable workflow from this successful execution trajectory.

## Guidelines
1. **Abstraction**: Convert specific inputs (filenames, URLs, numbers, entity names) into descriptive variable placeholders.
2. **Invariance**: Keep the logical steps and reasoning structure invariant.
3. **Focus on process**: Capture HOW the model navigated the context, located information, and constructed its answer.
4. Extract *at most 3* workflow items.
5. Do NOT copy specific facts from the context — focus on the transferable reasoning process.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Workflow Item i
## Title <the title of the workflow item>
## Description <one sentence summary>
## Steps <numbered list of abstracted steps that led to a correct answer>
```"""

FAILED_WORKFLOW_PROMPT = """You are an expert workflow analyst for context-learning tasks. You will be given a conversation where an AI model likely gave an INCORRECT or INCOMPLETE response.

## Goal
Extract workflow lessons: what went wrong and what process should be followed instead.

## Guidelines
1. **Abstraction**: Convert specific inputs into descriptive variable placeholders.
2. **Diagnose**: Identify which step in the reasoning process failed or was skipped.
3. **Prescribe**: Describe the corrected workflow that would avoid the failure.
4. Extract *at most 3* workflow items.
5. Do NOT assume you know the correct answer — focus on process improvements.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Workflow Item i
## Title <the title of the workflow item>
## Description <one sentence summary of the lesson>
## Steps <numbered list of the corrected workflow steps>
```"""

MEMORY_INJECTION_PREFIX = (
    "\n\nBelow are some workflow memory items accumulated from past interactions "
    "that may help solve this task. Consider using relevant workflows to guide "
    "your reasoning approach.\n\n"
)

RETRIEVAL_INSTRUCTION = (
    "Given the prior workflow experiences, your task is to analyze "
    "a current query's intent and select relevant workflow patterns "
    "that could guide answering it."
)


def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


import re


def _parse_workflow_items(text: str) -> list:
    """Parse markdown workflow items from LLM output."""
    items = []
    parts = re.split(r"(?=# Workflow Item\s*\d+)", text)
    for part in parts:
        part = part.strip()
        if part and part.startswith("# Workflow Item"):
            items.append(part)
    return items


class AgentWorkflowMemory(Memory):
    """Workflow memory: stores LLM-induced abstracted workflows; retrieves via embedding similarity."""

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
    ):
        self.client = client
        self.embed_client = embed_client or client
        self.model = model
        self.extract_model = extract_model or model
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.top_k = top_k
        self.memory_dir = memory_dir

        self.memory_bank = []
        self.embeddings = []
        self._last_embed_usage = {}

        os.makedirs(memory_dir, exist_ok=True)
        self.bank_path = os.path.join(memory_dir, "workflow_bank.jsonl")
        self.embed_path = os.path.join(memory_dir, "workflow_embeddings.jsonl")

        self._init_embed_provider()
        self._load_from_disk()

    def _load_from_disk(self):
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
                        self.embeddings.append(self._l2_normalize(vec))

        if self.memory_bank:
            log(f"   Loaded {len(self.memory_bank)} workflow entries from disk")

        if len(self.memory_bank) != len(self.embeddings):
            log(
                f"   Bank({len(self.memory_bank)}) vs embeddings"
                f"({len(self.embeddings)}) mismatch, re-embedding..."
            )
            queries = [entry.get("query", "") for entry in self.memory_bank]
            raw = self._embed_batch(queries)
            self.embeddings = [self._l2_normalize(v) for v in raw]
            self._rewrite_embeddings()

    def _init_embed_provider(self):
        if self.embed_provider == "gemini":
            from vertexai.language_models import TextEmbeddingModel

            self._gemini_embed_model = TextEmbeddingModel.from_pretrained(
                "gemini-embedding-001"
            )
        elif self.embed_provider == "qwen":
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", padding_side="left"
            )
            self._qwen_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B")
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._qwen_model = self._qwen_model.to(self._device)
        elif self.embed_provider in ("openai", "dashscope"):
            pass
        else:
            raise ValueError(f"Unknown embed_provider: {self.embed_provider}")

    # ─── Embedding helpers ───────────────────────────────────────────────────

    @staticmethod
    def _truncate_head_tail(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10) if norm > 0 else vec

    def _embed(self, text: str) -> np.ndarray:
        text = self._truncate_head_tail(text, 8000)
        if self.embed_provider == "gemini":
            return self._embed_with_gemini(text)
        if self.embed_provider == "qwen":
            return self._embed_with_qwen(text)
        return self._embed_with_openai(text)

    def _embed_with_gemini(self, text: str) -> np.ndarray:
        from vertexai.language_models import TextEmbeddingInput

        text_input = TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")
        resp = self._gemini_embed_model.get_embeddings(
            [text_input], output_dimensionality=3072
        )
        return np.array(resp[0].values, dtype=np.float32)

    def _embed_with_qwen(self, text: str) -> np.ndarray:
        import torch

        batch = self._qwen_tokenizer(
            [text],
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._qwen_model(**batch)
            last_hidden = out.last_hidden_state
            masked = last_hidden.masked_fill(
                ~batch["attention_mask"][..., None].bool(), 0.0
            )
            pooled = masked.sum(dim=1) / batch["attention_mask"].sum(dim=1)[..., None]
        return pooled.squeeze(0).cpu().numpy()

    def _embed_with_openai(self, text: str) -> np.ndarray:
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        if response.usage:
            self._last_embed_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _embed_batch(self, texts: list) -> list:
        if not texts:
            return []
        if self.embed_provider in ("openai", "dashscope"):
            results = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = [self._truncate_head_tail(t, 8000) for t in texts[i : i + batch_size]]
                response = self.embed_client.embeddings.create(
                    model=self.embed_model,
                    input=batch,
                )
                for item in response.data:
                    results.append(np.array(item.embedding, dtype=np.float32))
            return results
        return [self._embed(t) for t in texts]

    def _rewrite_embeddings(self):
        with open(self.embed_path, "w", encoding="utf-8") as f:
            for i, emb in enumerate(self.embeddings):
                record = {
                    "id": self.memory_bank[i].get("task_id", str(i)),
                    "text": self.memory_bank[i].get("query", ""),
                    "embedding": emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ─── Retrieve ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple:
        empty_stats = {"latency_s": 0.0, "embed_usage": {}}
        if not self.memory_bank or not self.embeddings:
            return "", empty_stats

        t0 = time.time()
        instruct_query = _get_detailed_instruct(RETRIEVAL_INSTRUCTION, query)
        self._last_embed_usage = {}
        instruct_vec = self._embed(instruct_query)
        instruct_vec = self._l2_normalize(instruct_vec)
        embed_usage = self._last_embed_usage.copy()

        bank_matrix = np.stack(self.embeddings)
        scores = (bank_matrix @ instruct_vec) * 100.0

        k = min(self.top_k, len(self.memory_bank))
        top_indices = np.argsort(scores)[::-1][:k]

        workflow_items = []
        for idx in top_indices:
            entry = self.memory_bank[idx]
            for item in entry.get("workflow_items", []):
                workflow_items.append(item)

        stats = {"latency_s": time.time() - t0, "embed_usage": embed_usage}
        if not workflow_items:
            return "", stats
        return MEMORY_INJECTION_PREFIX + "\n\n".join(workflow_items), stats

    # ─── Extract (workflow induction + store) ────────────────────────────────

    def extract(
        self,
        content: str,
        task_id: str = "",
        query: str = "",
        context_category: str = "",
        sub_category: str = "",
    ) -> dict:
        t0 = time.time()

        # Step 1: Induce workflow from trajectory
        extract_messages = [
            {"role": "system", "content": SUCCESSFUL_WORKFLOW_PROMPT},
            {"role": "user", "content": content},
        ]
        response_text, error, llm_usage = call_openai_api(
            self.client, extract_messages, self.extract_model, temperature=0.7
        )

        if error or not response_text:
            log(f"   Workflow extraction failed: {error}")
            return {
                "latency_s": time.time() - t0,
                "llm_usage": llm_usage,
                "embed_usage": {},
            }

        workflow_items = _parse_workflow_items(response_text)
        if not workflow_items:
            return {
                "latency_s": time.time() - t0,
                "llm_usage": llm_usage,
                "embed_usage": {},
            }

        # Step 2: Embed and store
        embed_text = query if query else content
        self._last_embed_usage = {}
        embedding = self._l2_normalize(self._embed(embed_text))
        embed_usage = self._last_embed_usage.copy()

        entry = {
            "task_id": task_id,
            "query": self._truncate_head_tail(query, 8000),
            "workflow_items": workflow_items,
            "context_category": context_category,
            "sub_category": sub_category,
        }
        self.memory_bank.append(entry)
        self.embeddings.append(embedding)

        append_jsonl(entry, self.bank_path)
        embed_record = {
            "id": task_id,
            "text": self._truncate_head_tail(query, 8000),
            "embedding": embedding.tolist(),
        }
        append_jsonl(embed_record, self.embed_path)

        return {
            "latency_s": time.time() - t0,
            "llm_usage": llm_usage,
            "embed_usage": embed_usage,
        }
