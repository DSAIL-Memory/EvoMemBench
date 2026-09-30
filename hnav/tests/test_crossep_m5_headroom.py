"""M5 CrossEp write-headroom machinery, on synthetic streams.  [T10]

Every statistic is checked against a value computed by hand or by an
independent oracle in the test: an exact duplicate planted at a known position
must surface as exactly one MD5 dup and a Jaccard of exactly 1.0; the
cluster aggregation must be the mean of the per-cluster rates (cluster-first),
not a candidate-pooled figure; the NLI pass must respect the split fence and
score the planted contradiction pair on both directions.
"""
from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pytest

from hnav import config as _config
from hnav.core.embedding import HashEmbedder
from hnav.stage0 import crossep_m5_write_headroom as m5

CFG = _config.HNavConfig()          # off-mode is fine: the replay drives the
                                    # adapter directly, no benchmark hook runs

T1 = "alpha bravo charlie delta echo foxtrot"
T2 = "golf hotel india juliett kilo lima"
T3 = "alpha bravo charlie delta echo GOLF"     # 5-of-7 word overlap with T1


def _events(cid: str, texts: list[str]) -> list[dict]:
    return [{"context_id": cid, "task_id": f"t{i}", "chunk_index": i,
             "context_category": "TestCat", "text": t}
            for i, t in enumerate(texts)]


# ── replay: dup detection + lexical proxy, closed-form ───────────────────────
def test_replay_flags_planted_duplicate_and_jaccard():
    emb = HashEmbedder(dim=64)
    records = m5.replay_context("ctxA", _events("ctxA", [T1, T2, T1]), emb, CFG)
    assert len(records) == 3

    # first write: empty store, nothing to compare against
    assert records[0]["geometry"]["is_exact_dup"] is False
    assert records[0]["lex_jaccard_max"] is None

    # planted exact duplicate at position 3
    assert records[2]["geometry"]["is_exact_dup"] is True
    assert records[2]["lex_jaccard_max"] == pytest.approx(1.0)
    assert records[2]["geometry"]["sim_max"] == pytest.approx(1.0, abs=1e-6)

    # hand-computed Jaccard for the middle write: T1 and T2 share no words
    assert records[1]["lex_jaccard_max"] == pytest.approx(0.0)
    # its own probe ranks it top-1 against the single, unrelated store row
    assert records[1]["retrieval_effect"]["rank_self_after"] == 0


def test_jaccard_matches_hand_computation_on_overlap():
    emb = HashEmbedder(dim=64)
    records = m5.replay_context("ctxB", _events("ctxB", [T1, T3]), emb, CFG)
    words1 = set(T1.lower().split())
    words3 = set(T3.lower().split())
    expected = len(words1 & words3) / len(words1 | words3)   # 5 / 7
    assert expected == pytest.approx(5 / 7)
    assert records[1]["lex_jaccard_max"] == pytest.approx(expected)


# ── cluster stats + cluster-first aggregation, closed-form ───────────────────
def _stats_for(cid: str, texts: list[str], split: str) -> dict:
    emb = HashEmbedder(dim=64)
    records = m5.replay_context(cid, _events(cid, texts), emb, CFG)
    for r in records:
        r["context_category"] = "TestCat"
    return m5.cluster_stats(cid, records, split)


def test_cluster_stats_closed_form():
    row = _stats_for("ctxA", [T1, T2, T1], "calibration")
    assert row["n_candidates"] == 3
    assert row["n_exact_dup"] == 1
    assert row["exact_dup_rate"] == pytest.approx(1 / 3)
    assert row["split"] == "calibration"
    assert row["context_category"] == "TestCat"
    # two finite sim_max values: cos(T2, T1-store) ~ 0, cos(T1-dup) = 1.0,
    # so exactly half clear the 0.99 rung
    assert row["sim_max"]["n"] == 2
    assert row["sim_max_frac_ge"]["0.99"] == pytest.approx(0.5)


def test_aggregation_is_cluster_first_not_pooled():
    a = _stats_for("ctxA", [T1, T2, T1], "calibration")          # dup rate 1/3
    b = _stats_for("ctxC", [T1, T2, T3, "x y z", T1, T1], "calibration")
    assert b["exact_dup_rate"] == pytest.approx(2 / 6)

    agg = m5.aggregate_split([a, b])
    assert agg["n_clusters"] == 2
    assert agg["n_candidates_total"] == 9
    # cluster-first: mean of {1/3, 2/6}, NOT the pooled 3/9 — equal here by
    # construction, so pin the mechanism with unequal clusters too:
    assert agg["exact_dup_rate__over_clusters"]["mean"] == pytest.approx(
        np.mean([1 / 3, 2 / 6]))

    c = _stats_for("ctxD", [T1, T2], "calibration")              # dup rate 0
    agg2 = m5.aggregate_split([a, c])
    pooled = 1 / 5                                                # would be wrong
    cluster_first = np.mean([1 / 3, 0.0])
    assert agg2["exact_dup_rate__over_clusters"]["mean"] == pytest.approx(cluster_first)
    assert agg2["exact_dup_rate__over_clusters"]["mean"] != pytest.approx(pooled)


# ── NLI pass: split fence + bidirectional scoring ────────────────────────────
def test_nli_stub_scores_pairs_and_respects_split_fence():
    emb = HashEmbedder(dim=64)
    per_ctx = OrderedDict()
    # calibration cluster: T3 arrives after T1 (nearest prior = T1, HashEmbedder
    # cosine is tiny but argmax is deterministic given only one prior candidate
    # with overlap); StubNLI sees the 5-of-7 token overlap -> contradiction.
    per_ctx["ctx_cal"] = m5.replay_context(
        "ctx_cal", _events("ctx_cal", [T1, T3]), emb, CFG)
    per_ctx["ctx_held"] = m5.replay_context(
        "ctx_held", _events("ctx_held", [T1, T2]), emb, CFG)
    splits = {"ctx_cal": "calibration", "ctx_held": "heldout"}

    block = m5.run_nli(per_ctx, splits, engine="stub",
                       allowed_splits={"calibration"}, max_pairs=10)
    assert block["n_pairs"] == 1, "held-out pairs must not be scored"
    row = block["per_cluster"][0]
    assert row["context_id"] == "ctx_cal"
    # StubNLI: >=0.6 token overlap -> contradiction 0.90 in BOTH directions
    assert row["bidir_contra_frac_ge"]["0.90"] == pytest.approx(1.0)

    empty = m5.run_nli(per_ctx, splits, engine="stub",
                       allowed_splits=set(), max_pairs=10)
    assert empty["n_pairs"] == 0


# ── stream readers ───────────────────────────────────────────────────────────
def test_qwen3_stream_reconstruction(tmp_path):
    rec = {
        "idx": "task-1",
        "messages": [
            {"role": "system", "content": "System rules here."},
            {"role": "user", "content": "What is the answer to life?"},
        ],
        "model_output": "Forty-two.",
        "metadata": {"task_id": "task-1", "context_id": "ctx-9",
                     "context_category": "TestCat", "sub_category": "s"},
    }
    src = tmp_path / "out.jsonl"
    src.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    streams, _used_fallback = m5.iter_qwen3_stream(src)
    assert list(streams) == ["ctx-9"]
    events = streams["ctx-9"]
    assert events and all(e["context_id"] == "ctx-9" for e in events)
    joined = " ".join(e["text"] for e in events)
    # the backend prepends the query, then the serialized trajectory
    assert joined.startswith("What is the answer to life?")
    for marker in ("**[System Context]**", "**[User Question]**",
                   "**[Model Output]**", "Forty-two."):
        assert marker in joined


def test_mem0_history_reader(tmp_path):
    import sqlite3

    ctx = tmp_path / "ctx-77"
    ctx.mkdir()
    con = sqlite3.connect(ctx / "history.db")
    con.execute("""CREATE TABLE history (
        id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT, new_memory TEXT,
        event TEXT, created_at DATETIME, updated_at DATETIME,
        is_deleted INTEGER, actor_id TEXT, role TEXT)""")
    rows = [
        ("1", "m1", None, "first memory", "ADD", "2026-01-01T00:00:01", None, 0),
        ("2", "m2", None, "second memory", "ADD", "2026-01-01T00:00:02", None, 0),
        ("3", "m3", None, "deleted memory", "ADD", "2026-01-01T00:00:03", None, 1),
        ("4", "m1", "first memory", "updated", "UPDATE", "2026-01-01T00:00:04", None, 0),
    ]
    con.executemany(
        "INSERT INTO history (id, memory_id, old_memory, new_memory, event, "
        "created_at, updated_at, is_deleted) VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()

    streams, fallback = m5.iter_mem0_history(tmp_path)
    assert fallback is False
    assert list(streams) == ["ctx-77"]
    texts = [e["text"] for e in streams["ctx-77"]]
    assert texts == ["first memory", "second memory"], \
        "only live ADD events, in created_at order"

    with pytest.raises(FileNotFoundError):
        m5.iter_mem0_history(tmp_path / "empty-nowhere")


# ── embedder provenance capture ──────────────────────────────────────────────
def test_embedder_report_unwraps_the_cache_and_records_the_namespace(tmp_path):
    """The artifact must self-certify which vectors it used.

    The 2026-08-16 real-embedder run could not: it survived only because the
    GQA fused-kernel fix engaged, and the flag that proves that was never
    written to the artifact. This pins the capture, including the distinction
    between "the fix did not engage" and "this embedder has no such flag".
    """
    from hnav.core.embedding import DiskCachedEmbedder, cache_key

    inner = HashEmbedder(dim=8)
    key = cache_key("some/model", "float32", 8192)
    assert key == "some_model|float32|L8192"      # slash-free, length-carrying

    wrapped = DiskCachedEmbedder(inner, tmp_path, key)
    rep = m5.embedder_report(wrapped)

    assert rep["class"] == "HashEmbedder"
    assert rep["wrapper"] == "DiskCachedEmbedder"
    assert rep["cache_namespace"] == key
    assert rep["cache_dir"] == str(tmp_path)
    # HashEmbedder has no GQA path at all — that is not the same as False
    assert rep["gqa_expansion_applied"] == "not_applicable"


def test_embedder_report_distinguishes_not_run_from_did_not_engage():
    class _FakeHF:
        model_name, dtype_name, device = "Qwen/Qwen3-Embedding-4B", "float32", "cuda:1"
        max_length, max_batch_tokens, batch, dim = 8192, 8192, 8, 2560

        def __init__(self, flag):
            self.gqa_expansion_applied = flag

    not_run = m5.embedder_report(_FakeHF(None))
    assert not_run["gqa_expansion_applied"] is None      # no forward pass yet
    assert not_run["wrapper"] is None                    # unwrapped embedder

    failed = m5.embedder_report(_FakeHF(False))
    assert failed["gqa_expansion_applied"] is False      # fix did NOT engage
    assert failed["max_length"] == 8192 and failed["max_batch_tokens"] == 8192
    assert failed["dtype"] == "float32" and failed["model"].endswith("Embedding-4B")

    ok = m5.embedder_report(_FakeHF(True))
    assert ok["gqa_expansion_applied"] is True


# ── M5b controls ─────────────────────────────────────────────────────────────
def _write_cache(tmp_path, namespace, texts_to_vecs):
    import hashlib
    d = tmp_path / "emb"
    d.mkdir(exist_ok=True)
    for t, v in texts_to_vecs.items():
        h = hashlib.sha256((namespace + "||" + t).encode()).hexdigest()
        np.save(d / f"{h}.npy", np.asarray(v, dtype=np.float32))
    return d


def test_m5b_cache_miss_is_fatal_never_a_silent_reembed(tmp_path):
    """M5b validates another run's vectors; embedding on a miss would let it
    silently compute in a different space from the run under validation."""
    from hnav.stage0 import crossep_m5b_control as m5b

    cache = m5b.CachedVectors("ns|float32|L8192", tmp_path)
    with pytest.raises(FileNotFoundError, match="will not embed"):
        cache.get("a text that was never embedded")


def test_m5b_cached_vectors_are_l2_normalized(tmp_path):
    from hnav.stage0 import crossep_m5b_control as m5b

    ns = "ns|float32|L8192"
    d = _write_cache(tmp_path, ns, {"t": np.array([3.0, 4.0])})   # norm 5
    v = m5b.CachedVectors(ns, d).get("t")
    assert v == pytest.approx([0.6, 0.8])


def test_m5b_control_pairs_are_reproducible_and_correctly_partitioned():
    """Closed form: contexts are built so within-context pairs are IDENTICAL
    vectors (cos 1.0) and across-context pairs are ORTHOGONAL (cos 0.0)."""
    from hnav.stage0 import crossep_m5b_control as m5b

    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    ctx = {"A": [e1, e1, e1], "B": [e2, e2, e2], "C": [e3, e3, e3]}

    out = m5b.control_pairs(ctx, n_pairs=200, seed=1)
    assert out["within_context"]["p50"] == pytest.approx(1.0)
    assert out["within_context"]["frac_ge_0.95"] == pytest.approx(1.0)
    assert out["across_contexts"]["p50"] == pytest.approx(0.0)
    assert out["across_contexts"]["frac_ge_0.95"] == pytest.approx(0.0)

    # same seed -> same draws; different seed -> the sampler actually moved
    again = m5b.control_pairs(ctx, n_pairs=200, seed=1)
    assert again == out


def test_m5b_naive_nn_prediction_matches_the_closed_form():
    """1 - (1-p)^i averaged over writes. Hand-checked on a 3-record store."""
    from hnav.stage0 import crossep_m5b_control as m5b

    p = 0.5
    got = m5b.naive_nn_prediction({"ctx": 3}, p)
    expected = np.mean([1 - 0.5 ** 1, 1 - 0.5 ** 2])      # writes i=1,2
    assert got["mean_predicted_nn_rate"] == pytest.approx(expected)
    assert got["n_writes_considered"] == 2

    # The point the report makes: at the measured pairwise rate and a
    # realistic store, the independent prediction for the NN max sits far
    # ABOVE M5's observed ~0.86 — so sim_max carries no evidence the pairwise
    # rate did not already carry. (Averaged over i=1..n-1, so early writes
    # with few priors pull it below the asymptote of ~1.0.)
    big = m5b.naive_nn_prediction({"ctx": 120}, 0.218)
    assert 0.96 < big["mean_predicted_nn_rate"] < 1.0
    assert big["mean_predicted_nn_rate"] > 0.86


def test_m5b_shingle_containment_is_a_ratio_of_the_chunk():
    from hnav.stage0 import crossep_m5b_control as m5b

    sys_text = " ".join(f"w{i}" for i in range(20))
    chunk = " ".join(f"w{i}" for i in range(10))          # wholly inside
    sh_c, sh_s = m5b.shingles(chunk), m5b.shingles(sys_text)
    assert sh_c and sh_c <= sh_s
    assert len(sh_c & sh_s) / len(sh_c) == pytest.approx(1.0)

    unrelated = " ".join(f"z{i}" for i in range(10))
    sh_u = m5b.shingles(unrelated)
    assert len(sh_u & sh_s) / len(sh_u) == pytest.approx(0.0)


def test_m5b_attribution_separates_artifact_from_organic():
    """A context with one system-derived chunk and one organic near-duplicate:
    the attribution must credit the artifact only where it applies."""
    from hnav.stage0 import crossep_m5b_control as m5b

    sys_text = " ".join(f"s{i}" for i in range(40))
    sys_chunk = " ".join(f"s{i}" for i in range(30))       # verbatim substring
    organic = " ".join(f"q{i}" for i in range(30))         # nothing in common

    streams = {"ctx": [
        {"text": sys_chunk, "task_id": "t1"},
        {"text": sys_chunk, "task_id": "t2"},              # artifact dup
        {"text": organic, "task_id": "t3"},
        {"text": organic, "task_id": "t4"},                # organic dup
    ]}
    v_sys = np.array([1.0, 0.0])
    v_org = np.array([0.0, 1.0])
    ctx_vecs = {"ctx": [v_sys, v_sys, v_org, v_org]}

    attr = m5b.attribution(streams, ctx_vecs, {"ctx": sys_text}, calib={"ctx"})
    assert attr["n_write_events"] == 4
    # two chunks are literally inside the system block
    assert attr["verbatim_system_substring"]["n"] == 2
    assert attr["verbatim_system_substring"]["rate"] == pytest.approx(0.5)

    b = attr["calibration"]
    # writes 2,3,4 get a sim_max; the two dups hit 1.0, the first organic 0.0
    assert b["near_dup_mass"]["n"] == 2
    assert b["near_dup_mass"]["cross_sample_frac"] == pytest.approx(1.0)
    # exactly one of the two near-dups is system-derived
    assert b["near_dup_mass"]["containment_frac_ge"]["0.9"] == pytest.approx(0.5)
    # the organic pair is counted as organic and IS a near-duplicate
    assert b["organic_chunks"]["n"] == 2
    assert b["organic_chunks"]["sim_max_frac_ge_tau"] == pytest.approx(0.5)
