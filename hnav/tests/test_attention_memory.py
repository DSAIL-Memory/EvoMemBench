"""Long sequences must embed without exhausting the card.  [T13]

The defect: a SINGLE text of ~4.9k tokens OOM'd a 24 GB card at batch size 1,
so batch size was irrelevant by construction. Root cause is grouped-query
attention, not the "math path" alone — see
:func:`hnav.core.embedding.expanded_gqa_attention` for the mechanism.

Two layers, because they fail differently:

1. **torch-free** — the context manager's patch/restore/fail-open semantics,
   verifiable anywhere, so the contract is checked on every run rather than
   only on the box;
2. **GPU-only** — the actual regression: a >4.5k-token sequence encodes within
   the declared L8192 budget. Skipped without CUDA + the real weights.

Measured on the box (2026-08-15, RTX 4090 24 GB, fp32, torch 2.13):
every one of the 9 real calibration chunks at 4,885–5,008 tokens OOM'd
unpatched and succeeded patched at 17.2 GiB peak; at 2,455 tokens, where both
paths run, the vectors are bit-identical in cosine (max|1−cos| = 0.0,
max|Δcomponent| = 1.0e-07).
"""
from __future__ import annotations

import os

import pytest

from hnav.core.embedding import expanded_gqa_attention


# ── torch-free: the context manager's contract ───────────────────────────────
class _FakeSdpaModule:
    """Stand-in for transformers.integrations.sdpa_attention."""

    def __init__(self):
        self.use_gqa_in_sdpa = lambda mask, k, v: True   # the default that hurts


def _install(monkeypatch, module):
    """Install a fake ``transformers.integrations.sdpa_attention``.

    ``import a.b.c as x`` resolves through the parent packages, so the
    attribute chain has to exist too — not just the sys.modules entries.
    """
    import sys
    import types

    top = types.ModuleType("transformers")
    integrations = types.ModuleType("transformers.integrations")
    top.integrations = integrations
    integrations.sdpa_attention = module
    monkeypatch.setitem(sys.modules, "transformers", top)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.sdpa_attention", module)


def test_patch_applies_and_restores(monkeypatch):
    mod = _FakeSdpaModule()
    original = mod.use_gqa_in_sdpa
    _install(monkeypatch, mod)

    with expanded_gqa_attention() as applied:
        assert applied is True
        # inside: GQA is refused, so transformers takes its repeat_kv path and
        # the fused (memory-efficient) kernel becomes eligible
        assert mod.use_gqa_in_sdpa(None, None, None) is False

    assert mod.use_gqa_in_sdpa is original, "the patch must be reverted"
    assert mod.use_gqa_in_sdpa(None, None, None) is True


def test_patch_restores_even_on_exception(monkeypatch):
    mod = _FakeSdpaModule()
    original = mod.use_gqa_in_sdpa
    _install(monkeypatch, mod)

    with pytest.raises(RuntimeError):
        with expanded_gqa_attention():
            raise RuntimeError("forward blew up")
    assert mod.use_gqa_in_sdpa is original, "a failed forward must not leak the patch"


def test_disabled_is_a_no_op(monkeypatch):
    mod = _FakeSdpaModule()
    _install(monkeypatch, mod)
    with expanded_gqa_attention(enabled=False) as applied:
        assert applied is False
        assert mod.use_gqa_in_sdpa(None, None, None) is True


def test_fails_open_when_the_internal_is_absent(monkeypatch):
    """A transformers rename must degrade to 'unpatched', never to a crash —
    and must report False so provenance records it."""
    import sys

    class _NoSymbol:
        pass

    _install(monkeypatch, _NoSymbol())
    with expanded_gqa_attention() as applied:
        assert applied is False

    # module missing entirely
    monkeypatch.setitem(sys.modules, "transformers.integrations.sdpa_attention", None)
    with expanded_gqa_attention() as applied:
        assert applied is False


# ── GPU-only: the regression itself ──────────────────────────────────────────
LONG_TOKENS = 4600      # above every unpatched-OOM threshold measured on the box


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda_available(),
                    reason="needs CUDA + the real Qwen3-Embedding-4B weights")
@pytest.mark.skipif(os.environ.get("HNAV_GPU_TESTS") != "1",
                    reason="set HNAV_GPU_TESTS=1 to run (loads ~15 GiB of weights)")
def test_long_sequence_encodes_within_budget():
    """A single long text must embed without OOM, in the pinned fp32 dtype at
    the declared L8192 budget — the exact call that used to fail."""
    import numpy as np

    from hnav import config as _config
    from hnav.core.embedding import DEFAULT_MAX_LENGTH, HFEmbedder

    cfg = _config.get_config()
    emb = HFEmbedder(model_name=cfg.embed_model, device=cfg.embed_device,
                     dtype=cfg.embed_dtype, batch=1,
                     max_length=DEFAULT_MAX_LENGTH)
    text = ("Thomas Kyd was born in the city of London. "
            "The chairperson of Fatah is Mahmoud Abbas. ") * 260
    n = len(emb.tok(text)["input_ids"])
    assert n > LONG_TOKENS, f"fixture is only {n} tokens; not the failing regime"

    v = emb.encode([text])                      # must not raise OutOfMemoryError
    assert v.shape == (1, emb.dim)
    assert np.isfinite(v).all()
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5, "must stay L2-normalized"
    assert emb.gqa_expansion_applied is True, (
        "the fused-kernel eligibility fix did not apply — this run would OOM "
        "on a real chunk")
