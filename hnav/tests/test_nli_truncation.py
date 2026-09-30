"""The NLI cross-encoder must not silently halve its inputs.  [T12 note 5]

Same defect class as the embedder's 512-token truncation, in a second module:
``CrossEncoderNLI`` defaulted to ``max_length=256``, and premise and
hypothesis SHARE that budget. Harmless in the primary arena (a fact is one
short sentence) but not in CrossEp, where ``crossep_m5_write_headroom.run_nli``
pairs chunks truncated to 1200 chars each — so each side saw roughly its
first ~128 tokens.

The honest difference from the embedder case, pinned below: raising the knob
does NOT eliminate this truncation. 512 is DeBERTa-v3's own
``max_position_embeddings``, and CrossEp pairs exceed it anyway, so the
residue is structural and must be REPORTED. These tests therefore check both
that the budget is as large as the model allows AND that the engine counts
what it had to cut.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hnav import config as _config
from hnav.core.read_gate import NLI_MAX_LENGTH_DEFAULT, CrossEncoderNLI

REPO = Path(__file__).resolve().parents[2]
CLBENCH = REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl"

# crossep_m5_write_headroom.run_nli's per-side character cut.
RUN_NLI_CHAR_CAP = 1200
# DeBERTa-v3's documented position limit; the box re-checks it from config.json.
DEBERTA_V3_MAX_POSITIONS = 512


def test_default_is_the_models_documented_limit_not_the_old_256():
    assert NLI_MAX_LENGTH_DEFAULT == DEBERTA_V3_MAX_POSITIONS
    assert NLI_MAX_LENGTH_DEFAULT > 256, (
        "the T12 note-5 defect is back: premise+hypothesis share this budget, "
        "so 256 gives a 1200-char chunk pair ~128 tokens per side")


def test_config_default_and_env_override():
    assert _config.HNavConfig().nli_max_length == NLI_MAX_LENGTH_DEFAULT
    cfg = _config.from_env({"HNAV_NLI_MAX_LENGTH": "384"}, repo=REPO)
    assert cfg.nli_max_length == 384


def test_build_nli_passes_max_length_by_keyword(monkeypatch):
    """``build_nli`` must hand the configured budget to the engine — the exact
    omission that caused the embedder defect."""
    import hnav.core.read_gate as rg

    seen = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(rg, "CrossEncoderNLI", _Spy)
    rg.build_nli(_config.HNavConfig(nli_max_length=333))
    assert seen["max_length"] == 333, f"build_nli dropped max_length: {seen}"
    # and everything else stays keyword-passed
    assert set(seen) >= {"model_name", "device", "dtype", "batch", "max_length"}


def test_m5_call_site_passes_max_length():
    """The CrossEp caller is the one that actually feeds long pairs; assert by
    AST that it does not re-inherit the default."""
    import ast

    src = (REPO / "hnav/stage0/crossep_m5_write_headroom.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "CrossEncoderNLI"]
    assert calls, "no CrossEncoderNLI construction found in crossep_m5"
    for c in calls:
        kws = {k.arg for k in c.keywords}
        assert "max_length" in kws, (
            "crossep_m5 constructs CrossEncoderNLI without max_length — it "
            "would inherit the default while feeding 1200-char chunk pairs")
        assert not c.args, "positional args here are how the defect happened"


# ── truncation accounting (the structural residue) ───────────────────────────
class _Mask:
    """numpy-backed stand-in for the attention mask tensor: supports the two
    operations the counter uses (``.sum(dim=1)`` and ``>=``)."""

    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape

    def sum(self, dim=None):
        return _Mask(self.arr.sum(axis=dim)) if dim is not None else self.arr.sum()

    def __ge__(self, other):
        return _Mask(self.arr >= other)


class _FakeTok:
    """Returns an encoding whose attention_mask says how long the row is."""

    def __init__(self, lengths):
        self.lengths = lengths
        self.seen_max_length = None

    def __call__(self, prem, hyp, **kw):
        self.seen_max_length = kw.get("max_length")
        n = len(prem)
        cap = kw["max_length"]
        arr = np.zeros((n, cap), dtype=int)
        for i, L in enumerate(self.lengths[:n]):
            arr[i, : min(L, cap)] = 1

        class _Enc(dict):
            def to(self, _d):
                return self

        return _Enc(attention_mask=_Mask(arr))


class _TorchShim:
    """The three torch entry points ``score_pairs`` touches — no torch needed,
    so the counter logic is verifiable on a torchless machine like the rest of
    the numeric core."""

    @staticmethod
    def no_grad():
        import contextlib
        return contextlib.nullcontext()

    @staticmethod
    def softmax(x, dim=-1):
        e = np.exp(x - x.max(axis=dim, keepdims=True))
        return _Probs(e / e.sum(axis=dim, keepdims=True))


class _Probs:
    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


def _engine_without_loading(max_length, lengths):
    """A CrossEncoderNLI with the model/tokenizer stubbed — __init__ loads
    ~1.7 GB of weights, which no unit test may do."""
    eng = CrossEncoderNLI.__new__(CrossEncoderNLI)
    eng._torch = _TorchShim()
    eng.model_name = "stub"
    eng.batch = 16
    eng.max_length = max_length
    eng.device = "cpu"
    eng.n_scored = 0
    eng.n_truncated = 0
    eng.model_max_positions = DEBERTA_V3_MAX_POSITIONS
    eng.tok = _FakeTok(lengths)
    eng._idx = {"entailment": 0, "neutral": 1, "contradiction": 2}

    class _Logits:
        def __init__(self, arr): self.arr = arr
        def float(self): return self.arr

    class _Model:
        def __call__(self, **enc):
            n = enc["attention_mask"].shape[0]

            class _Out:
                logits = _Logits(np.zeros((n, 3)))

            return _Out()

    eng.model = _Model()
    return eng


def test_truncation_is_counted_not_assumed_away():
    eng = _engine_without_loading(512, lengths=[600, 100, 512, 30])
    eng.score_pairs([("a", "b")] * 4)
    assert eng.n_scored == 4
    # rows 0 (600 -> capped at 512) and 2 (exactly 512) hit the budget
    assert eng.n_truncated == 2
    assert eng.truncation_rate == 0.5
    rep = eng.truncation_report()
    assert rep["max_length"] == 512 and rep["n_truncated"] == 2
    assert rep["model_max_positions"] == 512


def test_short_pairs_are_never_counted_as_truncated():
    """The primary arena's shape: everything far below the budget. This is why
    raising 256 -> 512 cannot change any T11 primary-arena result."""
    eng = _engine_without_loading(512, lengths=[20, 35, 12])
    eng.score_pairs([("a", "b")] * 3)
    assert eng.n_truncated == 0 and eng.truncation_rate == 0.0


def test_max_length_reaches_the_tokenizer():
    eng = _engine_without_loading(512, lengths=[10])
    eng.score_pairs([("a", "b")])
    assert eng.tok.seen_max_length == 512


def test_exceeding_the_models_position_limit_is_refused():
    """Raising the budget past the checkpoint's limit must fail loudly rather
    than extrapolate silently — the whole point of T12. Exercises the REAL
    guard, not a copy of it."""
    from hnav.core.read_gate import check_position_limit

    assert check_position_limit(512, 512, "m") == 512      # exactly at the limit
    assert check_position_limit(256, 512, "m") == 512      # under
    assert check_position_limit(9999, 0, "m") == 0         # unknown limit: allowed
    with pytest.raises(ValueError, match="max_position_embeddings"):
        check_position_limit(1024, 512, "cross-encoder/nli-deberta-v3-large")
    with pytest.raises(ValueError, match="513"):
        check_position_limit(513, 512, "m")


# ── the measured premise: CrossEp pairs really are too long ──────────────────
@pytest.mark.skipif(not CLBENCH.exists(), reason="CL-bench data not present")
def test_crossep_pairs_structurally_exceed_even_the_raised_budget():
    """Measured from the real data with the real chunker: this is why the fix
    is 'raise to the model's limit AND report the residue', not 'raise it'."""
    tiktoken = pytest.importorskip("tiktoken")
    pytest.importorskip("nltk")
    sys.path.insert(0, str(REPO))
    from hnav.stage0.crossep_m5_write_headroom import build_chunks

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    sides = []
    with open(CLBENCH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 15:
                break
            for m in json.loads(line).get("messages", []):
                content = m.get("content") or ""
                if not content.strip():
                    continue
                chunks, fallback = build_chunks(content)
                if fallback:
                    pytest.skip("nltk/tiktoken missing — not the real chunker")
                sides.extend(len(enc.encode(c[:RUN_NLI_CHAR_CAP],
                                            disallowed_special=()))
                             for c in chunks)

    assert sides, "no CrossEp chunks measured"
    over_old = sum(1 for t in sides if t > 256 // 2) / len(sides)
    over_new = sum(1 for t in sides if 2 * t + 3 > NLI_MAX_LENGTH_DEFAULT) / len(sides)
    assert over_old > 0.5, (
        f"only {over_old:.1%} of sides exceeded the old 128-token half-budget; "
        f"the premise of this fix no longer holds")
    # The honest part: the raised budget still does not cover these pairs.
    assert over_new > 0.0, (
        "CrossEp pairs now fit in the budget — the 'structural residue' "
        "caveat in NLI_MAX_LENGTH_DEFAULT and M5's output should be revisited")
