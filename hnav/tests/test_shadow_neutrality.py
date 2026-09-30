"""Shadow mode must be indistinguishable from off.  [T4]

The acceptance criterion in the brief is a *run-level* one: ``HNAV_MODE=off`` vs
``shadow`` on ``sh_6k``/``sh_32k`` at ``temperature=0`` must produce byte-identical
model output and identical per-question scores. That comparison needs the GPU box
and is listed in ``hnav/BUILD_NOTES.md`` as run-level-untested here.

What this file establishes locally is the *mechanism* that makes the run-level
claim true, at the three places where it could break:

  I1 return-value identity — every hooked entry point returns the caller's own
     object, checked with ``is``, not merely an equal one;
  I2 no store mutation — the adapter never writes to a ``StoreView`` it is given;
  I3 no extra model calls — the wrapper calls the inner backend exactly once per
     ``retrieve``/``extract`` and never calls an LLM itself.

Plus a source-level check that all four benchmark edits keep their native branch
intact and stay gated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from hnav import config as _config
from hnav.adapters import clbench_adapter as cl
from hnav.adapters import mab_adapter as mab
from hnav.core.types import Decision, StoreView
from hnav.tests.conftest import build_store, random_texts

REPO = Path(__file__).resolve().parents[2]

SHADOW = _config.HNavConfig(mode=_config.MODE_SHADOW)
OFF = _config.HNavConfig(mode=_config.MODE_OFF)

CHUNK = """Here is a list of facts:
0. Thomas Kyd was born in the city of London.
1. The chairperson of Fatah is Mahmoud Abbas.
2. Thomas Kyd was born in the city of Madrid.
3. Bengaluru is located in the continent of Asia.
"""


# ── I1: return-value identity ────────────────────────────────────────────────
@pytest.mark.parametrize("cfg", [OFF, SHADOW], ids=["off", "shadow"])
def test_before_memorize_returns_the_same_object(cfg, embedder):
    adapter = mab.MABAdapter(cfg=cfg, embedder=embedder)
    out = adapter.before_memorize(CHUNK, context_id=0, chunk_index=0)
    assert out is CHUNK, "the hook must return the caller's own string object"


def test_off_mode_records_nothing(embedder):
    """Default off means truly inert — no candidates, no index, no state."""
    adapter = mab.MABAdapter(cfg=OFF, embedder=embedder)
    adapter.before_memorize(CHUNK, context_id=0)
    assert adapter.facts == [] and adapter.n_write_decisions == 0
    assert adapter.write_index.n_observed == 0

    view_decision = adapter.on_retrieve("q", [CHUNK], [1.0], top_k=1)
    assert view_decision.action == "PASS" and view_decision.shadow
    assert adapter.n_read_decisions == 0


def test_shadow_mode_records_but_only_passes(embedder):
    adapter = mab.MABAdapter(cfg=SHADOW, embedder=embedder)
    adapter.before_memorize(CHUNK, context_id=0)

    assert adapter.n_write_decisions == 4
    assert [r.id for r in adapter.facts] == ["fact:0", "fact:1", "fact:2", "fact:3"]
    # fact 2 supersedes fact 0 on (born in the city of, Thomas Kyd)
    assert adapter._fact_by_id["fact:2"].metadata["object"] == "Madrid"

    decision = adapter.on_retrieve("Where was Thomas Kyd born?", [CHUNK], [88.0], top_k=1)
    assert decision.action == "PASS" and decision.shadow is True


def test_write_path_never_looks_ahead(embedder):
    """The predecessor of fact 2 is fact 0; fact 0 has no predecessor.

    If the write path used the whole-store index, fact 0 would resolve fact 2 as
    its "prior" — the exact look-ahead hard rule 2 forbids.
    """
    adapter = mab.MABAdapter(cfg=SHADOW, embedder=embedder)
    priors = {}
    for serial, text in mab.explode_facts(CHUNK):
        cand = adapter.candidate_for(serial, text)
        priors[serial] = cand.prior_id
        adapter._admit(cand, 0)

    assert priors[0] is None, "fact 0 saw a fact that did not exist yet"
    assert priors[2] == "fact:0"
    assert not hasattr(adapter.write_index, "latest"), \
        "WriteConflictIndex must not expose a whole-store lookup"


# ── I2: no store mutation ────────────────────────────────────────────────────
def test_adapter_never_mutates_a_given_store(embedder):
    adapter = mab.MABAdapter(cfg=SHADOW, embedder=embedder)
    store = build_store(random_texts(20, seed=11), embedder)
    before_ids, before_matrix = list(store.ids), store.matrix.copy()

    adapter.after_retrieval(
        _view(store, top_k=3), store, "some query")

    assert store.ids == before_ids
    assert np.array_equal(store.matrix, before_matrix)


def _view(store: StoreView, top_k: int):
    from hnav.core.types import RetrievalView
    n = len(store)
    return RetrievalView(query="q", query_vector=None, ids=list(store.ids),
                         scores=np.linspace(90, 10, n), top_k=top_k)


# ── the retriever hook returns the native page ───────────────────────────────
class _Doc:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


def test_native_page_is_the_k_limited_search(embedder):
    """Truncating the full ranking == searching with k=top_k.

    This is the equivalence the ``TextRetriever.retrieve`` edit relies on: the
    hooked branch asks FAISS for the whole ranking, and the value it returns is
    the same list the unhooked branch returned.
    """
    texts = random_texts(50, seed=12)
    q = embedder.encode(["a query"])[0]
    mat = embedder.encode(texts)
    d2 = 2.0 - 2.0 * (mat @ q)          # squared L2 on unit vectors
    order = np.argsort(d2, kind="stable")

    full = [(_Doc(texts[i]), float(d2[i])) for i in order]
    for top_k in (1, 3, 10):
        k_limited = [texts[i] for i in order[:top_k]]
        assert mab.native_page_from_scored(full, top_k) == k_limited


def test_l2sq_conversion_preserves_rank_order():
    d2 = np.array([0.0, 0.25, 1.0, 2.0, 3.5])
    sim = mab.to_similarity(d2, "l2sq")
    assert list(np.argsort(-sim, kind="stable")) == list(np.argsort(d2, kind="stable"))
    assert sim[0] == pytest.approx(100.0)     # d2=0 -> cos=1
    assert sim[3] == pytest.approx(0.0)       # d2=2 -> cos=0
    with pytest.raises(ValueError):
        mab.to_similarity(d2, "bogus")


# ── I1/I3 on the CrossEp-Know wrapper ────────────────────────────────────────
class _FakeMemory:
    """Counts calls so extra work is detectable."""

    def __init__(self) -> None:
        self.memory_bank = [{"task_id": "t1", "chunk_index": 0, "chunk_text": "a chunk"}]
        self.embeddings = [np.ones(4, dtype=np.float32) / 2.0]
        self.n_retrieve = 0
        self.n_extract = 0
        self.retrieve_result = ("memory text", {"latency_s": 0.1, "embed_usage": {"total_tokens": 7}})
        self.extract_result = {"latency_s": 0.2, "llm_usage": {"total_tokens": 11},
                               "embed_usage": {}}

    def retrieve(self, query):
        self.n_retrieve += 1
        return self.retrieve_result

    def extract(self, content, **kwargs):
        self.n_extract += 1
        return self.extract_result


def test_wrapper_delegates_verbatim_and_adds_no_calls():
    inner = _FakeMemory()
    adapter = cl.CLBenchAdapter(cfg=SHADOW)
    wrapper = cl.HNavMemoryWrapper(inner, adapter, mode=_config.MODE_SHADOW)

    got = wrapper.retrieve("a query")
    assert got is inner.retrieve_result          # the same tuple object
    assert inner.n_retrieve == 1                 # exactly one native call

    out = wrapper.extract("some trajectory", task_id="t1", query="q")
    assert out is inner.extract_result
    assert inner.n_extract == 1

    # I3: token accounting is the inner backend's, untouched
    assert out["llm_usage"]["total_tokens"] == 11
    assert adapter.n_read_decisions == 1 and adapter.n_write_decisions == 1


def test_wrapper_is_attribute_transparent():
    inner = _FakeMemory()
    wrapper = cl.HNavMemoryWrapper(inner, cl.CLBenchAdapter(cfg=SHADOW))
    assert wrapper.memory_bank is inner.memory_bank
    assert len(wrapper.embeddings) == 1


def test_wrap_memory_is_identity_when_off(monkeypatch):
    monkeypatch.setenv("HNAV_MODE", "off")
    _config.get_config(refresh=True)
    cl.reset_adapter()
    inner = _FakeMemory()
    assert cl.wrap_memory(inner) is inner
    assert cl.result_annotation() is None
    _config.get_config(refresh=True)


def test_stage0_scripts_still_refuse_live_but_the_adapter_permits_it(monkeypatch):
    """T11 protocol transition (STAGE1_PLAN.md §2 Faz B step 2, deliberate).

    ``config.require_not_live`` keeps guarding every Stage-0 measurement
    script; the ADAPTER no longer calls it, because live mode now legitimately
    arms the read-path rerank seam. Both halves are pinned here so neither can
    drift silently.
    """
    monkeypatch.setenv("HNAV_MODE", "live")
    cfg = _config.from_env()
    with pytest.raises(RuntimeError, match="live"):
        cfg.require_not_live()          # stage0 scripts: still refused
    adapter = mab.MABAdapter(cfg=cfg)   # stage1 adapter: permitted
    assert adapter.mode == _config.MODE_LIVE


def test_apply_read_decision_is_inert_outside_live(embedder):
    """In off/shadow the seam must return the native page byte-identically,
    whatever decision it is handed — defence in depth for S2 neutrality."""
    scored = [(_Doc("chunk one"), 0.1), (_Doc("chunk two"), 0.2)]
    armed = Decision(action="RERANK", payload={"order": ["x", "y"]}, shadow=False)
    for cfg in (OFF, SHADOW):
        adapter = mab.MABAdapter(cfg=cfg, embedder=embedder)
        page = adapter.apply_read_decision(armed, scored, top_k=2)
        assert page == ["chunk one", "chunk two"]
        assert adapter.n_reranks_applied == 0


# ── the four benchmark edits ─────────────────────────────────────────────────
EDITS = {
    "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/embedding_retriever.py":
        ["_hnav_adapter", "similarity_search_with_score_by_vector",
         "retrieved_docs[:top_k]", "apply_read_decision"],
    "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/agent.py":
        ["_hnav_adapter", "_dispatch_message", "on_send_message_enter"],
    "Cross-Episode-Knowledge/CROSSEP-KNOW/cl_bench_memory/registry.py":
        ["_hnav_wrap", "_build_memory", "wrap_memory"],
    "Cross-Episode-Knowledge/CROSSEP-KNOW/infer_context_memory.py":
        ["_hnav_annotate", "result_annotation"],
}


@pytest.mark.parametrize("path,markers", EDITS.items(), ids=lambda p: Path(str(p)).name
                         if isinstance(p, str) else "")
def test_benchmark_edits_present_and_parseable(path, markers):
    f = REPO / path
    assert f.exists(), f"missing {path}"
    src = f.read_text(encoding="utf-8")
    ast.parse(src, filename=str(f))          # a syntax error here breaks the whole suite
    assert "[HNAV]" in src, "the edit must be marked so it is findable in a diff"
    for m in markers:
        assert m in src, f"{path} lost marker {m!r}"


def test_hnav_import_failure_is_contained():
    """Every hook body must swallow import/runtime errors.

    If H-Nav is missing, mis-installed, or throws, the benchmark must run
    exactly as it did before. Structurally: each ``_hnav_*`` helper is a
    try/except whose handler returns the un-instrumented value.
    """
    for path in EDITS:
        src = (REPO / path).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=path)
        helpers = [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("_hnav")]
        assert helpers, f"{path} has no _hnav helper"
        for fn in helpers:
            assert any(isinstance(n, ast.Try) for n in ast.walk(fn)), \
                f"{path}:{fn.name} is not exception-guarded"


def test_retriever_native_branch_is_intact():
    """The un-hooked branch must still be the original three lines, and the
    hooked branch must START from the identical native truncation (T11: it now
    returns ``page``, which is initialized to that truncation and replaced
    only by the live-mode apply seam — never in off/shadow)."""
    src = (REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/methods/"
                  "embedding_retriever.py").read_text(encoding="utf-8")
    assert "results = self.vectorstore.similarity_search(query, k=initial_k)" in src
    assert "retrieved_docs = [doc.page_content for doc in results]" in src
    assert src.count("return retrieved_docs[:top_k]") == 1, \
        "the native branch must return the original expression"
    assert src.count("page = retrieved_docs[:top_k]") == 2, \
        "the hooked branch must initialize AND exception-restore the native page"
    assert 'if not getattr(decision, "shadow", True):' in src, \
        "the apply seam must be gated on an ARMED (non-shadow) decision"
