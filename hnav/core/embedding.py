"""Embedder interface + implementations.  [T3]

One protocol, three implementations:

* :class:`HashEmbedder`     — deterministic pseudo-random unit vectors seeded by
  ``sha256(text)``. Test-only. No model, no GPU, no network. Every numeric test
  in ``hnav/tests/`` runs against this, which is what lets the whole numeric core
  be verified on a machine with no torch.
* :class:`HFEmbedder`       — the real one. Loads Qwen3-Embedding-4B in-process on
  the GPU. **torch/transformers are imported inside ``__init__``**, never at
  module import time, so importing this module on a torchless machine is safe.
* :class:`OpenAIEmbedder`   — an OpenAI-compatible ``/v1/embeddings`` endpoint,
  for the T4+ path where the benchmark itself is running against a served model.

:class:`HFEmbedder`'s pooling mirrors ``Qwen3Embedding4BEmbeddings`` at
``MemoryAgentBench/methods/embedding_retriever.py:58`` — mean pooling over the
attention mask, then L2 normalization — so H-Nav's geometry is the geometry the
benchmark's own retriever operates in.

The disk cache layout is **byte-compatible with T1's**
(``hnav/stage0/m1_geometry_calibration.py``): ``<sha256(model|dtype||text)>.npy``
under ``HNAV_CACHE_DIR/emb``. T2 and everything after therefore reuse T1's
~26k embeddings for free, as the brief requires.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["EmbedderProtocol", "HashEmbedder", "HFEmbedder", "OpenAIEmbedder",
           "DiskCachedEmbedder", "cache_key", "l2_normalize",
           "DEFAULT_MAX_LENGTH", "DEFAULT_MAX_BATCH_TOKENS",
           "expanded_gqa_attention"]

# Tokenizer truncation length for the real embedders.  [T12]
#
# The benchmark chunks context with `chunk_text_into_sentences(chunk_size=4096)`
# — 4096 *tiktoken* (gpt-4o-mini / o200k) tokens. Measured on the calibration
# split, the real chunks reach 4,333 tiktoken tokens / 17,675 characters, which
# is roughly 4.4–4.9k Qwen tokens. 8192 covers the worst chunk with ~1.7x
# headroom while staying under BOTH ceilings that bound the live path:
# Qwen3-Embedding-4B's 32768 native context and the served embed endpoint's
# `--max-model-len 16384` (hnav/deploy/serve_embeddings.sh).
#
# The previous value was 512, which captured ~12% of the largest chunk. See
# hnav/BUILD_NOTES.md §10 for the full correction record.
DEFAULT_MAX_LENGTH = 8192

# Memory bound for one padded forward, in tokens (count x longest).  [T13b]
# A batch of 9 real benchmark chunks is ~45k tokens, whose fp32 MLP activations
# ([tokens, 9728]) OOM a 24 GB card even with memory-efficient attention. 8192
# lets short facts batch freely (32 x ~16 = 512) while long chunks go one at a
# time (5,008 <= 8192 < 2 x 5,008). Purely a memory knob: it changes how texts
# share a padded forward, never a text's own vector.
DEFAULT_MAX_BATCH_TOKENS = 8192


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Everything H-Nav needs from an embedder.

    ``encode`` returns an ``(n, dim)`` float32 array of **L2-normalized** rows,
    in the same order as ``texts``.
    """

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization matching ``Qwen3EmbeddingMemory._l2_normalize``
    (``qwen3_embedding_memory.py:131``): a zero vector is returned unchanged."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / (n + 1e-10) if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.where(n > 0, v / (n + 1e-10), v)
    return out.astype(np.float32)


def cache_key(model: str, dtype: str, max_length: int) -> str:
    """The cache namespace: model, dtype AND the tokenizer truncation length.

    ``max_length`` is part of the namespace because it changes the vector
    (T12). Until 2026-08-15 the namespace was ``f"{model}|{dtype}"`` while
    every embedder truncated at 512 tokens; a caller who fixed the truncation
    would then have silently READ BACK the 512-token vectors for any text
    already cached, and the fix would have looked like a no-op. Encoding the
    length makes an incompatible vector a cache MISS instead of a wrong hit,
    and leaves the old ``|L512``-free entries on disk as provenance rather
    than overwriting them.

    ``max_length`` is REQUIRED, deliberately: this defect was born from a
    silently-defaulted truncation parameter, so no call site gets to omit it.
    """
    return f"{model}|{dtype}|L{int(max_length)}".replace("/", "_")


def _cache_path(cache_dir: Path, key: str, text: str) -> Path:
    h = hashlib.sha256((key + "||" + text).encode()).hexdigest()
    return cache_dir / f"{h}.npy"


# ── test embedder ────────────────────────────────────────────────────────────
class HashEmbedder:
    """Deterministic unit vectors seeded by the text hash. TEST USE ONLY.

    Properties the tests rely on:
      * ``encode`` is a pure function of the text — same text, same vector, in
        any process, on any platform;
      * distinct texts give near-orthogonal vectors in expectation, so nothing
        accidentally lands at cosine 1.0;
      * identical texts give cosine exactly 1.0, which is what makes the
        exact-duplicate and tie fixtures meaningful.

    It carries no semantics whatsoever. No threshold may be fit against it.
    """

    def __init__(self, dim: int = 64, salt: str = "hnav") -> None:
        self.dim = int(dim)
        self.salt = salt

    def _one(self, text: str) -> np.ndarray:
        seed = int.from_bytes(
            hashlib.sha256((self.salt + "||" + text).encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return l2_normalize(rng.standard_normal(self.dim).astype(np.float32))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts])


# ── disk cache wrapper ───────────────────────────────────────────────────────
class DiskCachedEmbedder:
    """Wrap any embedder with the shared on-disk cache.

    Kept separate from :class:`HFEmbedder` so that the cache can be exercised in
    tests against :class:`HashEmbedder`.

    ``persist=False`` makes the cache READ-ONLY: misses are still computed by
    the inner embedder but never written back. This is how the Stage-1 live
    stack protects the fp32-keyed cache — its fallback endpoint may serve a
    different serving dtype, and a vector from a different source must never be
    filed under the fp32 key (``hnav/_cache`` invariant: dtype drift moves
    every threshold). ``n_misses`` stays the tripwire either way.
    """

    def __init__(self, inner: EmbedderProtocol, cache_dir: Path, key: str,
                 persist: bool = True) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.persist = bool(persist)
        self.dim = getattr(inner, "dim", 0)
        self.n_hits = 0
        self.n_misses = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            p = _cache_path(self.cache_dir, self.key, t)
            if p.exists():
                out[i] = np.load(p)
                self.n_hits += 1
            else:
                todo.append(i)
        if todo:
            self.n_misses += len(todo)
            fresh = self.inner.encode([texts[i] for i in todo])
            for j, i in enumerate(todo):
                out[i] = fresh[j]
                if self.persist:
                    np.save(_cache_path(self.cache_dir, self.key, texts[i]), fresh[j])
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        stacked = np.stack([np.asarray(o, dtype=np.float32) for o in out])  # type: ignore[arg-type]
        self.dim = stacked.shape[1]
        return stacked


# ── attention memory: the GQA / fused-kernel interaction  [T13] ──────────────
@contextmanager
def expanded_gqa_attention(enabled: bool = True):
    """Make the fused (memory-efficient) SDPA kernels ELIGIBLE for this model.

    Why this exists — the measured root cause, not the assumed one. A single
    long text OOM'd a 24 GB card even at batch size 1, and the reason is a
    *grouped-query attention* interaction, not merely "SDPA picked the math
    path":

    Qwen3-Embedding-4B has 32 query heads and 8 KV heads. Transformers'
    ``sdpa_attention`` calls ``use_gqa_in_sdpa()``, which returns True when
    the attention mask is ``None`` (the single-text case, where an all-ones
    mask is optimized away). It then hands SDPA **un-expanded** K/V with
    ``enable_gqa=True``. PyTorch's fused kernels reject that for dense inputs
    ("both fused kernels require query, key and value to have the same
    num_heads. Query.sizes(): [1, 32, 4885, 128], Key sizes(): [1, 8, 4885,
    128]"), so it silently falls back to MATH, which materializes the full
    [1, 32, N, N] score matrix — 730 MiB per 8-head chunk at N=4885, on top of
    ~15.1 GiB of fp32 weights. Forcing ``EFFICIENT_ATTENTION`` directly does
    NOT help: it aborts with "No available kernel", because eligibility, not
    priority, is what is missing. (FLASH is fp16/bf16-only, so it is never an
    option under the fp32 pin.)

    This context manager makes ``use_gqa_in_sdpa`` return False, so
    transformers takes its own ``repeat_kv`` path and materializes K/V at 32
    heads. The fused kernel then becomes eligible and PyTorch selects it
    automatically.

    **Numerically neutral by construction**: ``repeat_kv`` broadcasts the same
    K/V that ``enable_gqa=True`` would have broadcast internally — identical
    attention, different materialization. Measured, not assumed: forcing
    EFFICIENT_ATTENTION on top of this produced BIT-IDENTICAL vectors
    (max|Δ| = 0.0), and re-encoding cached texts reproduces the committed
    fp32 cache (see ``hnav/BUILD_NOTES.md`` §11 for the equivalence table).

    Fails open: if the transformers internal is absent or renamed, the patch
    is skipped and the caller runs exactly as before. Yields whether it
    applied, so callers can record it rather than assume it.
    """
    if not enabled:
        yield False
        return
    try:
        import transformers.integrations.sdpa_attention as _sa  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — old/!transformers: nothing to patch
        yield False
        return
    original = getattr(_sa, "use_gqa_in_sdpa", None)
    if original is None:
        yield False
        return
    _sa.use_gqa_in_sdpa = lambda *a, **k: False
    try:
        yield True
    finally:
        _sa.use_gqa_in_sdpa = original


# ── real embedders (lazy heavy imports) ──────────────────────────────────────
class HFEmbedder:
    """In-process HuggingFace embedder, mean-pooled + L2-normalized.

    torch and transformers are imported in ``__init__``. Importing this module
    costs nothing and pulls in nothing; ``hnav/tests/test_no_torch_at_import.py``
    asserts exactly that.
    """

    def __init__(self, model_name: str, device: int = 1, dtype: str = "float32",
                 batch: int = 32, max_length: int = DEFAULT_MAX_LENGTH,
                 max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS) -> None:
        import torch  # noqa: PLC0415 — deliberate: keeps module import torch-free
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.model_name = model_name
        self.batch = batch
        self.max_length = max_length
        self.max_batch_tokens = int(max_batch_tokens)
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.dtype_name = dtype
        torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        # Whether the fused-kernel eligibility fix actually applied — recorded,
        # not assumed, so a transformers rename shows up as False in provenance
        # instead of as an OOM three hours into a run. See
        # :func:`expanded_gqa_attention`.
        self.gqa_expansion_applied: bool | None = None

    def _batches(self, texts: Sequence[str]) -> list[list[int]]:
        """Group indices into batches bounded by BOTH count and total tokens.

        A fixed count is the wrong unit when texts differ in length by two
        orders of magnitude (a fact is ~15 tokens, a benchmark chunk ~5,000).
        Padding makes a batch cost ``len(batch) × longest``, so 9 chunks in one
        batch is ~45k tokens and the MLP's [tokens, 9728] fp32 activations
        (~1.75 GiB each) exhaust the card even though attention is now
        memory-efficient (T13). This bounds that product.

        Order is preserved and every group is contiguous, so the returned
        vectors are in input order. Grouping changes only how many texts share
        a padded forward — never a text's own vector, because attention and
        mean-pooling are both masked. Verified against vectors cached from a
        different grouping (BUILD_NOTES §11b).
        """
        lengths = [len(self.tok(t, truncation=True,
                                max_length=self.max_length)["input_ids"])
                   for t in texts]
        groups: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i, n in enumerate(lengths):
            new_max = max(cur_max, n)
            if cur and ((len(cur) + 1) * new_max > self.max_batch_tokens
                        or len(cur) >= self.batch):
                groups.append(cur)
                cur, cur_max = [i], n
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            groups.append(cur)
        return groups

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks = []
        for group in self._batches(texts):
            batch_texts = [texts[i] for i in group]
            enc = self.tok(batch_texts, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            # K/V expanded so the fused SDPA kernel is eligible: without this a
            # single 4.9k-token text OOMs a 24 GB card at batch size 1, because
            # the math fallback materializes the [1, 32, N, N] score matrix.
            # Numerically neutral — verified against the committed fp32 cache.
            with self._torch.no_grad(), expanded_gqa_attention() as applied:
                self.gqa_expansion_applied = applied
                res = self.model(**enc)
            last = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
            mask = enc["attention_mask"].unsqueeze(-1).expand(last.size()).to(last.dtype)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())
        return np.vstack(chunks)


class OpenAIEmbedder:
    """An OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM on :8001).

    Used when the benchmark process itself needs embeddings; the ``openai``
    client is imported lazily for the same reason as torch above.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 batch: int = 32, dim: int = 0) -> None:
        from openai import OpenAI  # noqa: PLC0415

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.batch = batch
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        for s in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(
                model=self.model, input=list(texts[s: s + self.batch]))
            rows.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        out = l2_normalize(np.stack(rows))
        self.dim = out.shape[1]
        return out


def build_embedder(cfg, cached: bool = True) -> EmbedderProtocol:
    """Construct the configured real embedder, wrapped in the shared disk cache.

    Importing this module never triggers a model load; only calling this does.

    Every argument is passed BY KEYWORD (T12). The 512-token truncation defect
    survived unnoticed because this function passed four positional arguments
    and simply stopped before ``max_length``, so the parameter kept a default
    nobody had chosen for this benchmark. Keywords make an omission visible at
    the call site instead of silently correct-looking.
    """
    inner = HFEmbedder(model_name=cfg.embed_model, device=cfg.embed_device,
                       dtype=cfg.embed_dtype, batch=cfg.embed_batch,
                       max_length=cfg.embed_max_length,
                       max_batch_tokens=cfg.embed_max_batch_tokens)
    if not cached:
        return inner
    return DiskCachedEmbedder(
        inner, cfg.emb_cache_dir,
        cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length))
