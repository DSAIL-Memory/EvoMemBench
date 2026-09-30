"""The embedder must read a whole benchmark chunk, not its first 12%.  [T12]

The defect this pins (2026-08-15): ``HFEmbedder`` truncated at 512 tokens
while the benchmark chunks context at 4096 *tiktoken* tokens, so every
Stage-0 signal and every frozen threshold was computed from roughly the first
12% of each chunk — while the live path consumed the benchmark's own
full-length ranking. ``build_embedder`` never overrode it because it passed
four POSITIONAL arguments and stopped one short of ``max_length``.

Three independent things are checked, because each fails differently:

1. the configured length actually covers the largest real chunk on the
   calibration split (measured from the dataset, not asserted from memory);
2. the length reaches the tokenizer — verified behaviourally with a fake
   tokenizer, so a future refactor that drops the argument fails here even
   though no GPU is involved;
3. the cache NAMESPACE encodes the length, so a corrected embedder can never
   silently read back vectors produced by the truncated one. This is the part
   that would have made the fix look like a no-op on the box, where ~24k
   vectors were already cached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hnav import config as _config
from hnav.core.embedding import (DEFAULT_MAX_BATCH_TOKENS, DEFAULT_MAX_LENGTH,
                                 DiskCachedEmbedder, HFEmbedder, cache_key)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"

# The benchmark's chunker budget, from
# utils/eval_other_utils.py::chunk_text_into_sentences(chunk_size=4096).
BENCHMARK_CHUNK_TOKENS = 4096


def test_default_covers_a_full_benchmark_chunk():
    assert DEFAULT_MAX_LENGTH >= BENCHMARK_CHUNK_TOKENS, (
        f"max_length {DEFAULT_MAX_LENGTH} truncates the benchmark's "
        f"{BENCHMARK_CHUNK_TOKENS}-token chunks — the T12 defect, reintroduced")
    # Headroom: chunk_size is counted in tiktoken tokens, and the Qwen
    # tokenizer does not agree token-for-token with o200k on the same text.
    assert DEFAULT_MAX_LENGTH >= 2 * BENCHMARK_CHUNK_TOKENS, (
        "no headroom for tokenizer disagreement between tiktoken and Qwen")


def test_config_default_and_env_override():
    assert _config.HNavConfig().embed_max_length == DEFAULT_MAX_LENGTH
    cfg = _config.from_env({"HNAV_EMBED_MAX_LENGTH": "1234"}, repo=REPO)
    assert cfg.embed_max_length == 1234


@pytest.mark.skipif(not DATA.exists(), reason="benchmark data not present")
def test_configured_length_covers_the_real_calibration_chunks():
    """Measured, not assumed: build the real chunks and count real tokens."""
    tiktoken = pytest.importorskip("tiktoken")
    pytest.importorskip("nltk")
    sys.path.insert(0, str(REPO))
    from hnav.stage0.m2_retrieval_calibration import build_chunks, subset_name

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    data = json.load(open(DATA, encoding="utf-8"))
    worst = 0
    for item in data:
        if subset_name(item) not in ("sh_6k", "sh_32k"):   # calibration split
            continue
        chunks, fallback = build_chunks(item["context"], BENCHMARK_CHUNK_TOKENS)
        if fallback:
            pytest.skip("nltk/punkt missing — chunks are not the benchmark's")
        worst = max(worst, *(len(enc.encode(c, disallowed_special=())) for c in chunks))

    assert worst > 0, "no calibration chunks measured"
    assert worst <= DEFAULT_MAX_LENGTH, (
        f"largest real chunk is {worst} tiktoken tokens but max_length is "
        f"{DEFAULT_MAX_LENGTH}")
    # And the historical value must be genuinely inadequate — otherwise this
    # whole test file is guarding nothing.
    assert worst > 512, (
        "the largest calibration chunk fits in 512 tokens; the premise of the "
        "T12 correction no longer holds and this test needs revisiting")


# ── the length must actually reach the tokenizer ─────────────────────────────
class _FakeTokenizer:
    """Records the max_length it is called with; returns a minimal batch.

    Returns TORCH tensors, not numpy: the encode path calls ``.unsqueeze()``
    and ``.expand()`` on the attention mask. A numpy stub passes on a
    torchless machine (where this test is skipped) and fails on the box —
    which is exactly what happened, so the stub is now the real shape.
    """

    def __init__(self, torch) -> None:
        self.seen: list[int | None] = []
        self._torch = torch

    def __call__(self, *args, **kw):
        self.seen.append(kw.get("max_length"))
        n = len(args[0]) if args and isinstance(args[0], list) else 1
        torch = self._torch

        class _Enc(dict):
            def to(self, _device):
                return self

        return _Enc(input_ids=torch.zeros((n, 3), dtype=torch.long),
                    attention_mask=torch.ones((n, 3), dtype=torch.long))


def test_max_length_is_passed_through_to_the_tokenizer(monkeypatch):
    """Behavioural: no GPU, no weights — a stub model and tokenizer are
    installed on a bare instance, so this fails if a refactor drops the
    argument even though nothing here can load transformers."""
    torch = pytest.importorskip("torch", reason="torch-free box: covered by the "
                                                "config and namespace tests")
    emb = HFEmbedder.__new__(HFEmbedder)          # bypass __init__ (loads a model)
    tok = _FakeTokenizer(torch)
    emb.tok = tok
    emb._torch = torch
    emb.batch = 8
    emb.dim = 4
    emb.device = "cpu"
    emb.max_length = 4242
    # every field _batches() reads must be set here: bypassing __init__ means a
    # new grouping knob shows up as an AttributeError on the box and a skip here.
    emb.max_batch_tokens = DEFAULT_MAX_BATCH_TOKENS

    class _Model:
        def __call__(self, **enc):
            n = len(enc["attention_mask"])
            return (torch.ones((n, 3, 4)),)

    emb.model = _Model()
    emb.encode(["some text", "other text"])
    # _batches() length-probes each text before grouping, so the tokenizer is
    # called more than once. The property under test is that EVERY call carries
    # the configured length — asserting a fixed call count instead pins the
    # batching implementation, which is what made this assertion go stale.
    assert tok.seen, "tokenizer was never called"
    assert set(tok.seen) == {4242}, (
        f"tokenizer saw max_length={tok.seen}, want every call at 4242")


# ── the cache namespace must encode the length ───────────────────────────────
def test_cache_key_encodes_max_length_and_requires_it():
    a = cache_key("Qwen/Qwen3-Embedding-4B", "float32", 512)
    b = cache_key("Qwen/Qwen3-Embedding-4B", "float32", 8192)
    assert a != b, "512- and 8192-token vectors would share a cache entry"
    assert "L8192" in b and "/" not in b
    with pytest.raises(TypeError):
        cache_key("Qwen/Qwen3-Embedding-4B", "float32")   # must stay required


def test_changing_max_length_is_a_cache_miss_not_a_wrong_hit(tmp_path):
    """The scenario that would have hidden the fix on the box: a vector cached
    by the truncated embedder must NOT be served to the corrected one."""
    class _Const:
        dim = 4

        def __init__(self, fill): self.fill = fill

        def encode(self, texts):
            return np.full((len(texts), 4), self.fill, dtype=np.float32)

    text = "a chunk longer than five hundred and twelve tokens"
    old = DiskCachedEmbedder(_Const(1.0), tmp_path,
                             cache_key("m", "float32", 512))
    v_old = old.encode([text])
    assert old.n_misses == 1

    new = DiskCachedEmbedder(_Const(2.0), tmp_path,
                             cache_key("m", "float32", 8192))
    v_new = new.encode([text])
    assert new.n_hits == 0 and new.n_misses == 1, \
        "the corrected embedder read back a truncated vector"
    assert not np.array_equal(v_old, v_new)

    # ...and the old namespace still resolves, so prior runs stay reproducible.
    again = DiskCachedEmbedder(_Const(9.0), tmp_path,
                               cache_key("m", "float32", 512))
    assert np.array_equal(again.encode([text]), v_old) and again.n_hits == 1


def test_build_embedder_passes_length_and_namespace(monkeypatch):
    """``build_embedder`` must hand the configured length to BOTH the model and
    the cache namespace — the two halves of the original defect."""
    import hnav.core.embedding as E

    seen = {}

    class _Spy:
        dim = 4

        def __init__(self, **kw):
            seen.update(kw)

        def encode(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(E, "HFEmbedder", _Spy)
    cfg = _config.HNavConfig(embed_max_length=777, cache_dir=Path("/tmp/x"))
    wrapped = E.build_embedder(cfg)
    assert seen["max_length"] == 777, "configured length never reached the model"
    assert "L777" in wrapped.key, "cache namespace ignores the length"
