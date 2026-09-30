"""Calibration harness: the load-bearing pieces, in closed form.  [T11]

The harness replays the REAL gate and the REAL rerank over prepass data, so
what needs pinning here is (a) the replay plumbing cannot silently diverge
from the live path — the QR cache equals the real residual, the replay NLI
refuses unknown pairs instead of guessing — and (b) the pre-registered
objective picks the cell a hand-worked example says it must.
"""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from hnav import config as _config
from hnav.core.geometry import qr_residual
from hnav.core.read_gate import StubNLI
from hnav.core.types import MemoryRecord, StoreView
from hnav.stage1 import calibrate_read_policy as cal

# ── _auc ─────────────────────────────────────────────────────────────────────
def test_auc_closed_form():
    assert cal._auc([3, 4, 1, 2], [True, True, False, False]) == 1.0
    assert cal._auc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    assert cal._auc([1, 1, 1, 1], [True, True, False, False]) == 0.5
    assert cal._auc([1, 2], [True, True]) is None
    # hand-computed: pos {2, 3}, neg {1, 4} -> pairs won 2+1=3 of 4, ties 0
    assert cal._auc([2, 3, 1, 4], [True, True, False, False]) == pytest.approx(0.5)


# ── ReplayNLI ────────────────────────────────────────────────────────────────
def test_replay_nli_roundtrip_and_refusal():
    p, h = "premise text", "hypothesis text"
    table = {f"{cal._sha(p)}|{cal._sha(h)}": (0.1, 0.2, 0.7)}
    nli = cal.ReplayNLI(table)
    (s,) = nli.score_pairs([(p, h)])
    assert (s.entailment, s.neutral, s.contradiction) == (0.1, 0.2, 0.7)
    with pytest.raises(KeyError, match="missing"):
        nli.score_pairs([(h, p)])           # direction matters


# ── the QR cache must be the real residual ───────────────────────────────────
def test_cached_qr_equals_real_qr_hit_and_miss():
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((6, 12))
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    cache = cal._CachedQR()
    cache.load(mat)
    for i in range(6):
        bank = np.delete(mat, i, axis=0)
        got, _ = cache(mat[i], bank)
        want, _ = qr_residual(mat[i], bank)
        assert got[0] == pytest.approx(want[0], abs=1e-12)
    # miss: a vector never loaded falls back to the real computation
    v = rng.standard_normal(12)
    v /= np.linalg.norm(v)
    got, _ = cache(v, mat)
    want, _ = qr_residual(v, mat)
    assert got[0] == pytest.approx(want[0], abs=1e-12)


# ── the grid is exactly the declared axes ────────────────────────────────────
def test_grid_is_the_declared_cross_product():
    cells = list(cal.grid_cells())
    assert len(cells) == 3 * 3 * 3 * 3 * 2 == 162
    seen = {(c["cos_pair"], c["r_min_label"], c["ambiguity_mode"],
             c["nli_contradiction"], c["pair_filter"]) for c in cells}
    assert len(seen) == 162
    assert {c["r_min"] for c in cells if c["r_min_label"] == "frozen"} \
        == {cal._rg.R_MIN_CAL}


# ── refusal of the confirmatory split ────────────────────────────────────────
def test_confirmatory_subsets_are_refused(monkeypatch):
    monkeypatch.setattr("sys.argv", ["calibrate_read_policy.py",
                                     "--subsets", "sh_64k", "--prepass"])
    assert cal.main() == 2


# ── end to end on a constructed subset ───────────────────────────────────────
# Real arena templates so the validated parser yields keys (fact_key must not
# be reimplemented and must not be fed made-up shapes it never saw).
T_OLD = "1. Thomas Kyd was born in the city of London."
T_NEW = "3. Thomas Kyd was born in the city of Madrid."
T_NOISE = "2. The chairperson of Fatah is Mahmoud Abbas."
CHUNK0 = f"{T_OLD} {T_NOISE}"          # stale + distractor
CHUNK1 = T_NEW                          # the superseder


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _e(i, dim=8):
    v = np.zeros(dim)
    v[i] = 1.0
    return v


class MapEmbedder:
    """Deterministic lookup embedder for the constructed subset."""

    def __init__(self, table, dim=8):
        self.table = {k: _unit(v) for k, v in table.items()}
        self.dim = dim

    def encode(self, texts):
        return np.stack([self.table[t] for t in texts])


def _built_state(cfg):
    a = _e(0)
    b = 0.99 * _e(0) + np.sqrt(1 - 0.99 ** 2) * _e(1)
    facts = [(1, T_OLD[3:].strip()), (2, T_NOISE[3:].strip()), (3, T_NEW[3:].strip())]
    # near-tie chunk ranking: chunk0 first, chunk1 a hair behind -> nmargin
    # flags ambiguity under mode "any"
    q_text = "Where was Thomas Kyd born?"
    message = cal.QUERY_TEMPLATE.format(question=q_text)
    rq = cal.RETRIEVAL_EXTRACT.search(message).group(1)
    emb_table = {
        facts[0][1]: a, facts[1][1]: _e(2), facts[2][1]: b,
        CHUNK0: 0.90 * _e(3) + np.sqrt(1 - 0.90 ** 2) * _e(4),
        CHUNK1: 0.8999 * _e(3) + np.sqrt(1 - 0.8999 ** 2) * _e(4),
        rq: _e(3),
    }
    emb = MapEmbedder(emb_table)

    from hnav.adapters.mab_adapter import fact_key
    records = []
    for s, t in facts:
        key, obj = fact_key(t)
        records.append(MemoryRecord(id=f"fact:{s}", text=t,
                                    vector=emb.encode([t])[0], version=s,
                                    metadata={"key": key, "object": obj}))
    chunks = [CHUNK0, CHUNK1]
    chunk_store = StoreView.from_records([
        MemoryRecord(id=f"chunk:{i}", text=t, vector=emb.encode([t])[0], version=i)
        for i, t in enumerate(chunks)])
    st = {"name": "sh_6k", "facts": facts, "records": records,
          "by_id": {r.id: r for r in records}, "chunks": chunks,
          "chunk_store": chunk_store, "fallback_chunker": False,
          "fact_chunk": {"fact:1": "chunk:0", "fact:2": "chunk:0",
                         "fact:3": "chunk:1"},
          "fact_of_chunk": {"chunk:0": ["fact:1", "fact:2"],
                            "chunk:1": ["fact:3"]},
          "qs": [{"index": 0, "question": q_text, "message": message,
                  "retrieval_query": rq, "truths": ["Madrid"]}]}
    return st, emb


def _first_fact_answer_fn(prompt: str) -> str:
    """An order-SENSITIVE stub model: answers with the object of the first
    fact in the prompt — the position bias the rerank exists to exploit."""
    from hnav.adapters.mab_adapter import FACT_RE_INLINE, fact_key
    for m in FACT_RE_INLINE.finditer(prompt):
        _, obj = fact_key(m.group(2).strip())
        if obj:
            return obj
    return "no idea"


def test_prepass_and_evaluate_pick_the_hand_worked_cell(monkeypatch, tmp_path):
    cfg = _config.HNavConfig(mode="shadow", out_dir=tmp_path, cache_dir=tmp_path)
    args = Namespace(top_k=10, chunk_size=4096, max_questions=0, no_llm=False,
                     stub_llm=True, stub_nli=True, smoke_embedder=True,
                     cache_only=False)
    st, emb = _built_state(cfg)

    pp = cal.prepass_subset(st, emb, StubNLI(), cfg, args)
    assert pp["n_chunks"] == 2 and len(pp["questions"]) == 1
    q = pp["questions"][0]
    assert q["top_ids"] == ["chunk:0", "chunk:1"], "stale chunk must rank first"
    assert q["nmargin"] < cal._rg.NMARGIN_CAL, "near-tie must flag ambiguity"
    # exactly one loose pair: the supersession pair, cos 0.99, same key
    (pair,) = q["pairs"]
    assert {pair["a"], pair["b"]} == {"fact:1", "fact:3"}
    assert pair["same_key"] is True and pair["same_object"] is False
    assert pair["cos"] == pytest.approx(0.99, abs=1e-4)
    assert len(pp["nli_table"]) == 2      # both directions of the one pair

    monkeypatch.setattr(cal, "make_answer_fn",
                        lambda a, c: _first_fact_answer_fn)
    res = cal.evaluate([pp], {"sh_6k": st}, cfg, args)

    assert res["n_questions"] == 1
    ch = res["chosen"]
    assert ch is not None, "the hand-worked example must yield a feasible cell"
    # Hand-worked winner: net=1 everywhere it fires; tie-breaks take the
    # highest cos_pair (0.94 < 0.99 still passes), then the highest verifying
    # NLI tau (StubNLI contra=0.90 -> 0.9), then grid stability gives the
    # frozen r_min (pair residual 0.141 < 0.1924) and filter=True. Ambiguity:
    # H_z never flags on a 2-chunk ranking, so mode "all" cells are quiet and
    # "any" wins over "none" by the declared preference order.
    assert ch["helped"] == 1 and ch["harmed"] == 0 and ch["net"] == 1
    assert ch["cos_pair"] == 0.94
    assert ch["nli_contradiction"] == 0.9
    assert ch["r_min_label"] == "frozen"
    assert ch["ambiguity_mode"] == "any"
    assert ch["pair_filter"] is True
    assert ch["n_true_supersession"] >= 1 and ch["false_verified_rate"] == 0.0

    # Note-1 exposure is measurable: filter-off cells exist and carry the
    # same counters (here the only pair IS same-key, so rates match).
    off_cells = [c for c in res["cells"] if not c["pair_filter"]
                 and c["cos_pair"] == 0.94 and c["nli_contradiction"] == 0.9
                 and c["r_min_label"] == "frozen" and c["ambiguity_mode"] == "any"]
    assert len(off_cells) == 1 and off_cells[0]["net"] == 1
