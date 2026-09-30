"""Shared fixtures.  [T3–T8]  No torch, no GPU, no network."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.core.embedding import HashEmbedder  # noqa: E402
from hnav.core.types import Candidate, MemoryRecord, StoreView  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
CLBENCH = REPO / "Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl"
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="session")
def conflict_data() -> list[dict]:
    if not DATA.exists():
        pytest.skip(f"missing {DATA}")
    return json.load(open(DATA, encoding="utf-8"))


@pytest.fixture(scope="session")
def sh_6k(conflict_data) -> dict:
    """The smallest single-hop subset — 455 facts, fast enough for every test."""
    for item in conflict_data:
        if item["metadata"]["qa_pair_ids"][0].startswith("factconsolidation_sh_6k_"):
            return item
    pytest.skip("factconsolidation_sh_6k not present")


def build_store(texts, embedder, versions=None, metadata=None) -> StoreView:
    """A StoreView over `texts`, embedded and vectorized in list order."""
    vecs = embedder.encode(list(texts))
    versions = list(versions) if versions is not None else list(range(len(texts)))
    metadata = metadata or [{} for _ in texts]
    return StoreView.from_records([
        MemoryRecord(id=f"rec:{i}", text=t, vector=vecs[i], version=versions[i],
                     metadata=metadata[i])
        for i, t in enumerate(texts)
    ])


def make_candidate(text: str, embedder, cid: str = "cand:1", version: int = 10**6,
                   **kw) -> Candidate:
    return Candidate(id=cid, text=text, vector=embedder.encode([text])[0],
                     version=version, **kw)


def random_texts(n: int, seed: int = 0, prefix: str = "fact") -> list[str]:
    rng = np.random.default_rng(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliett", "kilo", "lima"]
    return [f"{prefix} {i}: " + " ".join(rng.choice(words, size=6)) for i in range(n)]
