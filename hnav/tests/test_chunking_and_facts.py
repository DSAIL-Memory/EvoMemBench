"""Chunk -> fact recovery on real data.  [T4/T5]

The benchmark does not hand H-Nav the raw context. It hands it a chunk produced
by ``chunk_text_into_sentences``, which ends in ``" ".join(sentences)`` — so the
newlines between facts become spaces wherever punkt found a boundary, and the
line-anchored ``FACT_RE`` from the brief matches nothing at all.

Every write-side signal depends on exploding a chunk into facts, so this is
checked against the committed dataset, in both the raw and the joined form.
"""
from __future__ import annotations

import pytest

from hnav.adapters.mab_adapter import FACT_RE, explode_facts
from hnav.stage0.m2_retrieval_calibration import _fallback_chunks, build_chunks


def test_raw_context_explodes_completely(sh_6k):
    facts = explode_facts(sh_6k["context"])
    assert len(facts) == 455, "sh_6k is 455 facts; the dataset changed"
    assert [s for s, _ in facts] == list(range(455)), "serials must be exact"
    assert facts[0][1] == "Thomas Kyd was born in the city of London."


def test_line_anchored_regex_alone_fails_on_a_joined_chunk(sh_6k):
    """The failure mode this fallback exists for — asserted, not assumed."""
    joined = " ".join(sh_6k["context"].split("\n"))
    assert FACT_RE.findall(joined) == []
    assert len(explode_facts(joined)) == 455


def test_joined_and_raw_forms_give_identical_facts(sh_6k):
    raw = explode_facts(sh_6k["context"])
    joined = explode_facts(" ".join(sh_6k["context"].split("\n")))
    assert raw == joined


def test_every_fact_survives_chunking(sh_6k):
    """Fact recovery across chunk boundaries — with the boundary-loss bound.

    Discovered on the GPU box the first time the REAL chunker ran (2026-08-14;
    it was on BUILD_NOTES' untested list): ``chunk_text_into_sentences`` can
    split a fact so that its serial ends one chunk ("... 306. <text> 307.")
    while its text opens the next, serial-less. The serial-less head is
    unrecoverable from the chunk stream by any parser — this is a property of
    the benchmark's own chunker, not of ``explode_facts``. On sh_6k it eats
    fact 307, 1 of 455 (0.22%).

    So the honest invariant is NOT "nothing is lost". It is:
      - nothing is duplicated or reordered,
      - nothing is corrupted (recovered facts match the raw parse exactly),
      - at most one fact is lost per chunk boundary,
      - and every lost fact exhibits exactly the dangling-serial mechanism.
    The loss rate is measured and asserted, never silently absorbed.
    """
    chunks, used_fallback = build_chunks(sh_6k["context"], chunk_size=4096)
    assert chunks, "chunker produced nothing"

    recovered = [f for c in chunks for f in explode_facts(c)]
    expected = explode_facts(sh_6k["context"])

    if used_fallback:
        # line grouping never splits a line, so recovery must be perfect
        assert recovered == expected
        return

    rec_serials = [s for s, _ in recovered]
    exp_by_serial = dict(expected)

    # no duplicates, no reordering
    assert len(rec_serials) == len(set(rec_serials)), "duplicate serials"
    assert rec_serials == sorted(rec_serials), "serials out of order"

    # no corruption: every recovered fact is byte-identical to the raw parse
    for s, t in recovered:
        assert t == exp_by_serial[s], f"fact {s} corrupted by chunking"

    # boundary-loss bound: each of the (n_chunks - 1) boundaries can eat ≤ 1
    lost = sorted(set(exp_by_serial) - set(rec_serials))
    assert len(lost) <= len(chunks) - 1, (
        f"{len(lost)} facts lost but only {len(chunks) - 1} boundaries: "
        f"something beyond the boundary mechanism is eating facts: {lost[:10]}"
    )

    # every loss must be the dangling-serial mechanism, nothing else:
    # the lost fact's serial sits at the very end of some non-final chunk
    for s in lost:
        assert any(c.rstrip().endswith(f"{s}.") for c in chunks[:-1]), (
            f"fact {s} lost but no chunk ends with its dangling serial — "
            f"an unknown loss mechanism"
        )


def test_fallback_chunker_respects_its_budget():
    text = "\n".join(f"{i}. fact number {i} says something." for i in range(500))
    chunks = _fallback_chunks(text, chunk_size=64)     # 64 tokens ~ 256 chars
    assert len(chunks) > 1
    assert all(len(c) <= 256 + 80 for c in chunks)
    assert len([f for c in chunks for f in explode_facts(c)]) == 500


def test_unparseable_lines_are_skipped_not_guessed(sh_6k):
    """The preamble line is not a fact and must not become one."""
    facts = explode_facts(sh_6k["context"])
    assert not any("Here is a list of facts" in t for _, t in facts)


@pytest.mark.parametrize("blob,expected", [
    ("0. a fact.\n1. another fact.", [(0, "a fact."), (1, "another fact.")]),
    ("intro: 0. a fact. 1. another fact.", [(0, "a fact."), (1, "another fact.")]),
    ("", []),
    ("no facts here at all", []),
    ("7. only one.", [(7, "only one.")]),
    # dangling next-fact serial at a chunk boundary is stripped from the last
    # fact (the real chunker produces exactly this tail — sh_6k fact 307) ...
    ("0. a fact. 1. another fact. 2.", [(0, "a fact."), (1, "another fact.")]),
    # ... but a genuine number at the end of a fact's own sentence is not:
    # no terminal period precedes it, so DANGLING_SERIAL_TAIL cannot match
    ("0. a fact. 1. the answer is 42.", [(0, "a fact."), (1, "the answer is 42.")]),
])
def test_explode_facts_edge_cases(blob, expected):
    assert explode_facts(blob) == expected
