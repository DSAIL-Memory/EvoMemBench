"""(relation, subject) -> versions index.  [T4/T6]

**Two classes, on purpose.** This is the subtlest leakage risk in the whole
design (plan §11), so it is enforced by class boundary rather than by discipline:

``WriteConflictIndex``
    The ONLY index the write path may touch. It is built by *observation*:
    ``observe(record)`` is called as facts are admitted, and ``latest_before(key,
    serial)`` returns the newest observed member of ``key`` with a strictly
    smaller serial. It has no ``latest`` method and no way to see the whole
    store. At write time, facts with a higher serial have not been observed yet;
    using them would be look-ahead.

``ReadConflictIndex``
    Built from the whole store, exposes ``latest(key)``. Legitimate on the read
    path only — at read time every fact is genuinely present in the store.

Neither class reads benchmark questions or answer keys: a *key* is whatever the
caller supplies. The primary-arena adapter derives keys with
``conflict_analysis.parse``, which is validated at 99.5%+ coverage and must not
be reimplemented.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict
from typing import Iterable, Sequence

__all__ = ["Key", "Entry", "WriteConflictIndex", "ReadConflictIndex", "to_probe"]

Key = tuple[str, str]  # (relation_key, subject)


class Entry:
    """A minimal (version, id, text, object) tuple, orderable by version."""

    __slots__ = ("version", "id", "text", "object")

    def __init__(self, version: int, id: str, text: str = "", object: str | None = None) -> None:  # noqa: A002
        self.version = int(version)
        self.id = id
        self.text = text
        self.object = object

    def __lt__(self, other: "Entry") -> bool:
        return (self.version, self.id) < (other.version, other.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entry) and (self.version, self.id) == (other.version, other.id)

    def __hash__(self) -> int:
        return hash((self.version, self.id))

    def __repr__(self) -> str:
        return f"Entry(v={self.version}, id={self.id!r}, obj={self.object!r})"


class WriteConflictIndex:
    """Online, write-path index. Observation-ordered; no whole-store view."""

    def __init__(self) -> None:
        self._by_key: dict[Key, list[Entry]] = defaultdict(list)
        self.n_observed = 0

    def observe(self, key: Key | None, entry: Entry) -> None:
        """Record that ``entry`` has been admitted. Unparseable facts (``key is
        None``) are counted but not indexed — they cannot conflict with anything
        under a key-based definition of conflict."""
        self.n_observed += 1
        if key is None:
            return
        insort(self._by_key[key], entry)

    def latest_before(self, key: Key | None, serial: int) -> Entry | None:
        """Newest observed member of ``key`` with ``version < serial``.

        This is the write path's ONLY lookup. There is deliberately no
        ``latest()`` here: see the module docstring.
        """
        if key is None:
            return None
        members = self._by_key.get(key)
        if not members:
            return None
        i = bisect_left(members, Entry(serial, ""))
        return members[i - 1] if i > 0 else None

    def members_before(self, key: Key | None, serial: int) -> list[Entry]:
        if key is None:
            return []
        return [e for e in self._by_key.get(key, ()) if e.version < serial]

    def n_keys(self) -> int:
        return len(self._by_key)


class ReadConflictIndex:
    """Read-path index over the whole store. ``latest`` is legitimate here."""

    def __init__(self, items: Iterable[tuple[Key | None, Entry]] = ()) -> None:
        self._by_key: dict[Key, list[Entry]] = defaultdict(list)
        self._by_id: dict[str, tuple[Key, Entry]] = {}
        for key, entry in items:
            self.add(key, entry)

    def add(self, key: Key | None, entry: Entry) -> None:
        if key is None:
            return
        insort(self._by_key[key], entry)
        self._by_id[entry.id] = (key, entry)

    def latest(self, key: Key | None) -> Entry | None:
        """Highest-version member of ``key`` anywhere in the store."""
        if key is None:
            return None
        members = self._by_key.get(key)
        return members[-1] if members else None

    def members(self, key: Key | None) -> list[Entry]:
        return list(self._by_key.get(key, ())) if key else []

    def key_of(self, record_id: str) -> Key | None:
        hit = self._by_id.get(record_id)
        return hit[0] if hit else None

    def entry_of(self, record_id: str) -> Entry | None:
        hit = self._by_id.get(record_id)
        return hit[1] if hit else None

    def is_conflicted(self, key: Key | None) -> bool:
        """A key is conflicted when its members disagree on the object — the
        definition already used by ``conflict_analysis.analyze``. Two facts with
        the same object are a restatement, not a conflict."""
        members = self.members(key)
        return len({e.object for e in members}) > 1

    def conflicted_keys(self) -> list[Key]:
        return [k for k in self._by_key if self.is_conflicted(k)]

    def superseder_of(self, record_id: str) -> "Entry | None":
        """The newest member of this record's key, if it is newer than the
        record itself and carries a different object. ``None`` means the record
        is not stale."""
        hit = self._by_id.get(record_id)
        if hit is None:
            return None
        key, entry = hit
        newest = self.latest(key)
        if newest is None or newest.version <= entry.version:
            return None
        return newest if newest.object != entry.object else None

    def __len__(self) -> int:
        return len(self._by_key)


def to_probe(relation_key: str, subject: str) -> str:
    """Rebuild a natural-language probe from a parsed key.

    ``conflict_analysis`` encodes relations as ``"<prefix>|<middle>"`` — the
    prefix is empty for the ``"{s} MID {o}."`` family. So the probe is just the
    template with the object slot left open, e.g.::

        ("|" + " was born in the city of ", "Thomas Kyd")
            -> "Thomas Kyd was born in the city of"

    This uses **decision-time information only** (the candidate's own parsed
    fields). It must never be built from a benchmark question — see
    ``hnav/tests/test_leakage_audit.py``.
    """
    pre, _, mid = relation_key.partition("|")
    return f"{pre}{subject}{mid}".strip()


def build_read_index(records: Sequence, key_fn) -> ReadConflictIndex:
    """Build a read index from ``MemoryRecord``-likes using ``key_fn(record)``
    -> ``(key, object)``."""
    idx = ReadConflictIndex()
    for r in records:
        key, obj = key_fn(r)
        idx.add(key, Entry(version=r.version, id=r.id, text=r.text, object=obj))
    return idx
