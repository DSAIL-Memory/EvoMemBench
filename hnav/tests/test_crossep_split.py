"""The CrossEp calibration split is frozen, disjoint, complete.  [T10]

The committed artifact (``hnav/labeling/crossep_split.json``) is compared
field-for-field against a fresh derivation from the dataset — the split cannot
drift without this test failing — and its structural invariants are asserted
directly rather than trusted: disjointness, completeness over the dataset's
clusters, the ~40% cluster ratio, and at least one calibration cluster in
every category large enough to split.
"""
from __future__ import annotations

import json

import pytest

from hnav.labeling import crossep_split as cs


@pytest.fixture(scope="module")
def dataset_exists():
    if not cs.DATASET.exists():
        pytest.skip(f"missing {cs.DATASET}")


@pytest.fixture(scope="module")
def committed(dataset_exists) -> dict:
    if not cs.SPLIT_PATH.exists():
        pytest.fail(f"{cs.SPLIT_PATH} not committed — run crossep_split.py")
    return json.loads(cs.SPLIT_PATH.read_text(encoding="utf-8"))


def test_committed_artifact_matches_fresh_derivation(committed):
    assert cs.derive_split() == committed, \
        "crossep_split.json drifted from derive_split(dataset, SEED)"


def test_derivation_is_deterministic(dataset_exists):
    assert cs.derive_split() == cs.derive_split()


def test_disjoint_and_complete(committed):
    cal, held = set(committed["calibration"]), set(committed["heldout"])
    assert not cal & held, "a cluster appears on both sides"

    # completeness against the dataset itself, counted independently here
    seen = set()
    with open(cs.DATASET, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                seen.add(json.loads(line)["metadata"]["context_id"])
    assert cal | held == seen, "split does not cover exactly the dataset's clusters"
    assert committed["n_clusters"] == len(seen)


def test_cluster_ratio_and_stratification(committed):
    n = committed["n_clusters"]
    frac = committed["n_clusters_calibration"] / n
    assert 0.35 <= frac <= 0.45, f"cluster calibration fraction {frac:.3f}"

    for cat, row in committed["per_category"].items():
        if row["n_clusters"] >= 2:
            assert 1 <= row["n_calibration"] <= row["n_clusters"] - 1, \
                f"category {cat!r} is not genuinely split"
        else:
            assert row["n_calibration"] == 0, \
                f"lone-cluster category {cat!r} must be held-out"


def test_sample_counts_are_consistent(committed):
    assert committed["n_samples"] == (committed["n_samples_calibration"]
                                      + committed["n_samples_heldout"])
    per_cat_samples = sum(r["n_samples"] for r in committed["per_category"].values())
    assert per_cat_samples == committed["n_samples"]


def test_split_of_labels_every_cluster(committed):
    assert cs.split_of(committed["calibration"][0], committed) == "calibration"
    assert cs.split_of(committed["heldout"][0], committed) == "heldout"
    assert cs.split_of("no-such-context", committed) == "unknown"
