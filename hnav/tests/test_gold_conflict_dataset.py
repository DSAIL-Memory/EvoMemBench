"""Gold conflict dataset — oracle checks.

The tier logic, the dual labels and the balanced eval set are each recomputed
here independently from the raw committed inputs and compared to what
``build_gold_conflict_dataset.py`` wrote. The builder's own EXPECTED table is
deliberately NOT the oracle — these tests re-derive the counts from
``audit_results_gpt5mini.jsonl`` so a builder bug and its expectation table
cannot agree by construction.

Skips (not fails) when the committed artifacts are absent, e.g. on a checkout
without ``stage0_results/``.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from collections import Counter, defaultdict

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR = REPO / "stage0_results" / "conflict_pairs"
DATASET = DIR / "gold_conflict_dataset.jsonl.gz"
SUMMARY = DIR / "gold_conflict_dataset.summary.json"
RESULTS = DIR / "audit_results_gpt5mini.jsonl"
RESULTS_GZ = DIR / "audit_results_gpt5mini.jsonl.gz"

BIN = 0.01
GOLD_TIERS = {"core", "update_only_fork"}

pytestmark = pytest.mark.skipif(
    not (DATASET.exists() and SUMMARY.exists()
         and (RESULTS.exists() or RESULTS_GZ.exists())),
    reason="committed gold-dataset artifacts not present")


def _jsonl(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@pytest.fixture(scope="module")
def dataset():
    return list(_jsonl(DATASET))


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def oracle_tiers():
    """Independent triage, straight from the judge's recorded flags."""
    tiers = {}
    for r in _jsonl(RESULTS if RESULTS.exists() else RESULTS_GZ):
        if r.get("status") != "ok":
            continue
        v = r["verdict"]
        if r["parser_tagged_conflict"]:
            if v["update_conflict"]:
                t = "core"
            elif v["same_referent"] and v["same_relation"] and v["context_overlap"]:
                t = "update_only_fork"
            else:
                t = "rejected"
        else:
            t = "discovered_unverified" if v["update_conflict"] else "negative"
        tiers[r["pair_id"]] = t
    return tiers


def test_every_audited_pair_appears_exactly_once_with_the_oracle_tier(dataset, oracle_tiers):
    seen = Counter(r["pair_id"] for r in dataset)
    assert set(seen) == set(oracle_tiers)
    assert seen.most_common(1)[0][1] == 1, "a pair_id appears twice"
    for r in dataset:
        assert r["tier"] == oracle_tiers[r["pair_id"]], r["pair_id"]


def test_dual_labels_are_consistent_with_tier_and_each_other(dataset):
    for r in dataset:
        assert r["gold_update"] == (r["tier"] in GOLD_TIERS)
        assert r["disputed_by_judge"] == (r["tier"] == "update_only_fork")
        if r["gold_strict"]:
            assert r["gold_update"], "gold_strict must imply gold_update"
            assert r["tier"] == "core" and r["judge"]["strict_conflict"]
        if r["tier"] == "core":
            assert r["gold_strict"] == bool(r["judge"]["strict_conflict"])


def test_eval_set_is_balanced_and_drawn_from_the_right_tiers(dataset):
    pos = defaultdict(int)
    neg = defaultdict(int)
    for r in dataset:
        if r["gold_update"]:
            assert r["in_eval_set"], "every gold positive belongs to the eval set"
        if not r["in_eval_set"]:
            continue
        assert r["tier"] in GOLD_TIERS | {"negative"}, (
            f"{r['tier']} may never enter the eval set")
        (pos if r["gold_update"] else neg)[r["subset"]] += 1
    assert pos and pos.keys() == neg.keys()
    for s in pos:
        assert pos[s] == neg[s], f"{s}: eval set unbalanced {pos[s]} vs {neg[s]}"


def test_eval_negatives_are_cosine_matched_or_carry_the_recorded_fallback(dataset):
    def cbin(c):
        return min(int(c / BIN), int(1.0 / BIN) - 1)

    pos_bins = defaultdict(set)
    for r in dataset:
        if r["in_eval_set"] and r["gold_update"]:
            pos_bins[r["subset"]].add(cbin(r["cosine_similarity"]))
    for r in dataset:
        if not (r["in_eval_set"] and r["tier"] == "negative"):
            continue
        b = cbin(r["cosine_similarity"])
        fb = r["provenance"]["cosine_bin_fallback"]
        if fb is None:
            assert b in pos_bins[r["subset"]], (
                f"{r['pair_id']}: claims exact bin match, bin {b} has no positive")
        else:
            assert any(abs(b - pb) == fb for pb in pos_bins[r["subset"]]), (
                f"{r['pair_id']}: recorded fallback {fb} matches no positive bin")


def test_summary_counts_and_cosine_auc_match_an_independent_recount(dataset, summary):
    tiers = Counter(r["tier"] for r in dataset)
    assert summary["tiers_total"] == {k: tiers[k] for k in sorted(tiers)}
    assert summary["core_strict"] == sum(1 for r in dataset if r["gold_strict"])
    assert summary["n_records"] == len(dataset)

    for s, blob in summary["per_subset"].items():
        pos = sorted(r["cosine_similarity"] for r in dataset
                     if r["subset"] == s and r["in_eval_set"] and r["gold_update"])
        neg = sorted(r["cosine_similarity"] for r in dataset
                     if r["subset"] == s and r["in_eval_set"] and r["tier"] == "negative")
        # quadratic-free independent AUC: count, for each positive, negatives
        # strictly below it (bisect) plus half the ties
        import bisect
        wins = 0.0
        for p in pos:
            lo = bisect.bisect_left(neg, p)
            hi = bisect.bisect_right(neg, p)
            wins += lo + (hi - lo) / 2.0
        auc = wins / (len(pos) * len(neg))
        assert abs(auc - blob["eval"]["cosine_only_auc"]) < 5e-4, s
        assert blob["eval"]["n_pos"] == len(pos) and blob["eval"]["n_neg"] == len(neg)


def test_quarantined_and_rejected_tiers_never_reach_gold_or_eval(dataset):
    for r in dataset:
        if r["tier"] in ("discovered_unverified", "rejected"):
            assert not r["gold_update"] and not r["gold_strict"]
            assert not r["in_eval_set"]
