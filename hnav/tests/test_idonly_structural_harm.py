"""The E2E-4 structural claim, encoded as a test.  [E2E-4]

Claim: any arm whose every VERIFIED pair satisfies the parser's ``same_key``
test has ``n_suppressed_harmful == 0`` *by construction*, at any NLI
threshold — because ``same_key`` is an equivalence on (relation, subject), so
a connected component of verified edges lies inside one key, and ``suppress``
keeps each component's highest serial, so a key's newest member is never
dropped.

The test drives the REAL ``suppress_ids`` and the REAL ``classify_drops`` over
randomized stores, and — the part that makes it evidence rather than
decoration — it also breaks each load-bearing precondition deliberately and
asserts harm *does* appear. A property test that cannot fail proves nothing.

This is the safety difference between the symbolic and geometric screens: the
committed ``hnav_geo`` sh_64k run recorded 8 harmful suppressions (its own
void condition 4 failed) because a similarity threshold is not transitive and
its groups merged two keys.
"""
from __future__ import annotations

import numpy as np
import pytest

from hnav.core.read_gate import ConflictGroup, GateDecision
from hnav.core.read_policy import suppress_ids
from hnav.core.types import MemoryRecord
from hnav.stage1.detector_gap import classify_drops

RNG = np.random.default_rng(20260830)


def _store(n_keys: int, max_members: int):
    """(records, table) — a synthetic corpus with unique serials, in the exact
    shape ``classify_drops`` consumes."""
    recs, by_id, members, superseded = {}, {}, {}, set()
    serial = 0
    for k in range(n_keys):
        key = (f"| relation {k % 3} ", f"subject {k}")
        n = int(RNG.integers(1, max_members + 1))
        rows = []
        for j in range(n):
            fid = f"fact:{serial}"
            obj = f"value {j}"
            recs[fid] = MemoryRecord(id=fid, text=f"{key[1]} -> {obj}",
                                     vector=RNG.normal(size=4),
                                     version=serial,
                                     metadata={"key": key, "object": obj})
            by_id[fid] = (serial, recs[fid].text, key, obj)
            rows.append((serial, fid, obj))
            serial += 1
        members[key] = rows
        for s, fid, _o in rows[:-1]:
            superseded.add(fid)
    return recs, {"by_id": by_id, "members": members, "superseded": superseded}


def _groups_within_keys(table, recs, rng):
    """Actionable groups that each lie inside one key — what same_key forces."""
    groups = []
    for key, rows in table["members"].items():
        if len(rows) < 2 or rng.random() < 0.3:
            continue
        take = sorted(rng.choice(len(rows), size=int(rng.integers(2, len(rows) + 1)),
                                 replace=False))
        ids = [rows[i][1] for i in take]
        latest = max(ids, key=lambda f: table["by_id"][f][0])
        groups.append(ConflictGroup(
            member_ids=ids, latest_id=latest,
            stale_ids=[f for f in ids if f != latest],
            residuals={f: 0.0 for f in ids}))
    return groups


def _harm(groups, table) -> int:
    drop = suppress_ids(GateDecision(ambiguous=False, groups=groups))
    m = {"n_suppressed_harmful": 0, "n_suppressed_superseded": 0,
         "n_suppressed_same_value": 0}
    classify_drops(m, drop, table)
    return m["n_suppressed_harmful"]


# ── the claim ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("trial", range(40))
def test_same_key_groups_never_produce_a_harmful_suppression(trial):
    rng = np.random.default_rng(1000 + trial)
    recs, table = _store(n_keys=int(rng.integers(3, 12)), max_members=5)
    groups = _groups_within_keys(table, recs, rng)
    assert _harm(groups, table) == 0


# ── the negative controls: break a precondition, harm must appear ────────────
def test_a_group_spanning_two_keys_does_produce_harm():
    """What a non-transitive (similarity-threshold) screen can build — and
    exactly what the committed geo run did on sh_64k."""
    rng = np.random.default_rng(7)
    recs, table = _store(n_keys=4, max_members=3)
    keys = list(table["members"])
    a, b = table["members"][keys[0]], table["members"][keys[1]]
    ids = [r[1] for r in a] + [r[1] for r in b]
    latest = max(ids, key=lambda f: table["by_id"][f][0])
    merged = [ConflictGroup(member_ids=ids, latest_id=latest,
                            stale_ids=[f for f in ids if f != latest],
                            residuals={f: 0.0 for f in ids})]
    # the whole of whichever key does not own the maximum serial is erased
    assert _harm(merged, table) > 0


def test_dropping_a_groups_maximum_would_be_harmful():
    """Guards the other half of the proof: keep-the-max is load-bearing."""
    rng = np.random.default_rng(11)
    recs, table = _store(n_keys=3, max_members=4)
    key = next(k for k, rows in table["members"].items() if len(rows) >= 2)
    rows = table["members"][key]
    ids = [r[1] for r in rows]
    lowest = min(ids, key=lambda f: table["by_id"][f][0])
    # a hand-built (never produced by suppress_ids) group that keeps the OLDEST
    bad = [ConflictGroup(member_ids=ids, latest_id=lowest,
                         stale_ids=[f for f in ids if f != lowest],
                         residuals={f: 0.0 for f in ids})]
    assert _harm(bad, table) > 0


def test_unparsed_drops_count_as_harmful():
    """The other route into the harm counter, and why same_key (which rejects
    key=None) closes it."""
    recs, table = _store(n_keys=2, max_members=2)
    m = {"n_suppressed_harmful": 0, "n_suppressed_superseded": 0,
         "n_suppressed_same_value": 0}
    classify_drops(m, ["fact:99999"], table)      # absent from by_id
    assert m["n_suppressed_harmful"] == 1
