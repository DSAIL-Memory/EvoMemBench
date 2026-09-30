"""MemoryAgentBench adapter — InEp-Know, the PRIMARY arena.  [T4]

Owns every benchmark-specific fact about this arena so that ``hnav/core/`` does
not have to know any of them:

* memorize messages are 4096-token *chunks* of numbered facts, so a chunk is
  exploded into per-fact candidates with ``FACT_RE``;
* the fact's serial number is the benchmark's own supersession key and becomes
  ``Candidate.version``;
* retrieval units are chunks, not facts, so read-side staleness is resolved by
  mapping retrieved chunks back to the facts they contain.

**Shadow mode returns the caller's own object, unchanged.** ``before_memorize``
returns the exact ``message`` it was handed (identity, not a copy);
``after_retrieval`` returns a ``Decision`` whose ``shadow`` flag the caller must
check before acting. Nothing here mutates a store, and nothing here makes an LLM
call (invariants I1/I2/I3).

**Live mode (Stage 1, T11) arms exactly one seam.** Under ``HNAV_MODE=live``
the read decision is returned with ``shadow=False`` and the retriever hook
routes it through :meth:`MABAdapter.apply_read_decision`, which reorders the
native top-k page and nothing else — same chunk set, same count, order only,
with a fail-open fallback to the native page on any irregularity. The write
path stays observation-only in every mode (``write_policy`` NO_GO forever,
``KAPI_KARARI.md``). Protocol trail for lifting ``require_not_live`` here:
``STAGE1_PLAN.md`` §2 Faz B step 2; Stage-0 measurement scripts still refuse
live via ``config.require_not_live``.

Leakage: this module reads fact text and serial numbers only. It never opens the
dataset JSON, so it cannot see the ``questions`` list or the answer list that
sit in the same file. ``hnav/tests/test_leakage_audit.py`` enforces that by AST
scan.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.core.audit import AuditLogger  # noqa: E402
from hnav.core.types import (Candidate, Decision, MemoryRecord,  # noqa: E402
                             RetrievalView, StoreView)
from hnav.labeling.conflict_analysis import parse as parse_fact  # noqa: E402
from hnav.labeling.conflict_index import (Entry, ReadConflictIndex,  # noqa: E402
                                          WriteConflictIndex, to_probe)

__all__ = ["FACT_RE", "FactSpan", "MABAdapter", "explode_facts", "fact_key",
           "fact_spans", "native_page_from_scored", "page_edit", "select_pool"]

# The benchmark's own fact numbering. Identical to the regex already used by
# conflict_analysis.analyze and by T1 — deliberately not a second dialect.
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)

# ...but a memorize chunk is NOT the raw context. `chunk_text_into_sentences`
# (utils/eval_other_utils.py:173) splits the context with nltk and rejoins with
# `" ".join(...)`, so wherever punkt found a sentence boundary the newline
# between two facts becomes a space and the line-anchored regex above stops
# matching. FACT_RE is still tried first — it is the prescribed regex and it is
# exact on the raw context — and this inline scanner takes over when the chunk
# has been through the joiner. Each fact runs to the next serial or to the end.
FACT_RE_INLINE = re.compile(r"(?:(?<=\s)|(?<=\A))(\d+)\.\s+(.+?)(?=\s+\d+\.\s|\Z)", re.S)

# The benchmark's chunker can end a chunk with a DANGLING SERIAL: punkt attaches
# the next fact's number to the previous sentence, the token budget cuts there,
# and the chunk tail reads "...Thomas Kyd was born in the city of Leeds. 307."
# while fact 307's text opens the next chunk. Discovered the first time the real
# chunker ran (sh_6k, fact 307). The inline scanner would otherwise glue that
# " 307." onto fact 306's text. The pattern requires the previous fact's
# terminal period before the number, so a genuine "...is 42." ending (no period
# before the number) is never touched. Applied to the LAST fact of a blob only —
# anywhere else the lookahead in FACT_RE_INLINE already stops at the next serial.
DANGLING_SERIAL_TAIL = re.compile(r"(?<=\.)\s+\d+\.\s*$")


def _text_id(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def fact_key(text: str) -> tuple[tuple[str, str] | None, str | None]:
    """``((relation, subject), object)`` for a fact, or ``(None, None)``.

    Delegates to the validated 99.5%-coverage parser. Do not reimplement.
    """
    p = parse_fact(text)
    if not p:
        return None, None
    rel, subj, obj = p
    return (rel, subj), obj


def explode_facts(message: str) -> list[tuple[int, str]]:
    """A memorize chunk -> ``[(serial, fact_text)]``.

    Tries the line-anchored ``FACT_RE`` first and falls back to
    ``FACT_RE_INLINE`` when the chunk has been through the benchmark's
    sentence-joining chunker and yields fewer facts. Serials are the
    benchmark's own and are preserved exactly; text is right-stripped only —
    except that a dangling next-fact serial on the last fact (see
    ``DANGLING_SERIAL_TAIL``) is removed, because it belongs to a fact whose
    text lives in the next chunk, not to this one.
    """
    primary = [(int(n), t.strip()) for n, t in FACT_RE.findall(message)]
    inline = [(int(n), t.strip()) for n, t in FACT_RE_INLINE.findall(message)]
    facts = inline if len(inline) > len(primary) else primary
    if facts:
        serial, text = facts[-1]
        cleaned = DANGLING_SERIAL_TAIL.sub("", text)
        if cleaned != text:
            facts[-1] = (serial, cleaned)
    return facts


# ── fact-granular page surgery (T13) ─────────────────────────────────────────
# The probe (hnav/stage1/stale_suppression_probe.py) measured that the arena's
# conflicted-stratum failure is POSITIONAL and per-FACT: deleting the stale
# sentence lifts conflicted accuracy 5.4% -> 89.2%, and the chunk-level version
# of the same idea is harmful (STAGE1_NULL_ANALIZI.md). read_policy therefore
# names FACTS; the page it must be applied to is a list of CHUNK texts. That
# splice lives here, because it needs the benchmark's serial numbering and its
# sentence-joining chunker — knowledge hard rule 2 keeps out of hnav/core/.
#
# Exactness contract, enforced by test_read_policy_facts.py against hand-built
# pages whose expected output is written out in full:
#   * surviving facts keep their ORIGINAL serials (no renumbering — the prompt
#     states its rule in terms of serial order);
#   * surviving text is spliced from the original bytes (no re-rendering, no
#     reflow, no whitespace normalisation);
#   * every byte outside the edited spans is unchanged.

class FactSpan(NamedTuple):
    """One fact's byte geometry inside a chunk.

    ``start``/``own_end``   the fact's OWN bytes ("<serial>. <text>"), which is
                            what a move re-inserts elsewhere.
    ``del_start``/``del_end``  what a deletion removes: the fact plus exactly
                            one adjacent separator, so the survivors stay
                            joined the way the chunker joined them. A fact owns
                            the separator that FOLLOWS it; the last fact of a
                            chunk has none, so it owns the one BEFORE it
                            instead (without that asymmetry, deleting the last
                            fact of a newline-joined context would leave a
                            blank line).
    """

    serial: int
    start: int
    own_end: int
    del_start: int
    del_end: int


def fact_spans(message: str) -> list[FactSpan]:
    """Byte spans of every fact in a chunk, in context order.

    Regex selection mirrors :func:`explode_facts` exactly — line-anchored
    ``FACT_RE`` first, ``FACT_RE_INLINE`` when it finds more, because a memorize
    chunk has been through the benchmark's sentence joiner. The dangling
    next-fact serial (``DANGLING_SERIAL_TAIL``) is excluded from the last fact's
    span: those bytes belong to a fact whose text lives in the NEXT chunk, and
    deleting them here would remove something this page never owned.
    """
    primary = list(FACT_RE.finditer(message))
    inline = list(FACT_RE_INLINE.finditer(message))
    ms = inline if len(inline) > len(primary) else primary
    if not ms:
        return []

    # End of content: trailing whitespace belongs to the CHUNK, not to the last
    # fact, so a move can re-insert before it and leave it in place.
    end_content = len(message.rstrip())
    last_end = min(ms[-1].end(), end_content)
    dang = DANGLING_SERIAL_TAIL.search(message[ms[-1].start(): last_end])
    if dang:
        last_end = ms[-1].start() + dang.start()

    out: list[FactSpan] = []
    for i, m in enumerate(ms):
        own_end = m.end() if i < len(ms) - 1 else last_end
        del_start = m.start()
        if i == len(ms) - 1:
            del_end = own_end
            if i > 0:                       # last fact owns the separator BEFORE it
                del_start = out[i - 1].own_end
        else:
            del_end = ms[i + 1].start()
        out.append(FactSpan(int(m.group(1)), m.start(), own_end, del_start, del_end))
    return out


def _content_tail(text: str) -> tuple[str, str]:
    """Split a chunk into ``(content, trailing_bytes)``.

    Trailing bytes are the trailing whitespace **and** a trailing dangling
    next-fact serial (``DANGLING_SERIAL_TAIL``). Nothing may ever be appended
    after either: the whitespace is the chunk's own framing, and the dangling
    serial belongs to a fact whose TEXT lives in the next chunk.

    This is not cosmetic. Appending "5. Pedro Pierluisi worked in ..." after a
    chunk that ends "...Leeds. 307." makes the inline parser read the whole
    appended sentence as the TEXT OF FACT 307 — the moved fact silently loses
    its own serial, which in this arena is the supersession key the entire
    mechanism turns on. Found by the retrieval-harness integrity check on the
    real sh_6k page, where chunk 0 ends exactly that way.
    """
    core = text.rstrip()
    dang = DANGLING_SERIAL_TAIL.search(core)
    if dang:
        core = core[: dang.start()]
    return core, text[len(core):]


def _fact_separator(spans_per_chunk: Sequence[Sequence[FactSpan]],
                    texts: Sequence[str]) -> str:
    """The bytes this page puts BETWEEN two consecutive facts.

    Read off the page rather than assumed: the raw dataset context joins facts
    with ``"\\n"``, a real memorize chunk joins most of them with ``" "``. The
    last page chunk is consulted first (that is where a moved fact lands), then
    any other chunk; a page with no consecutive pair at all falls back to the
    newline/space its last chunk actually contains.
    """
    order = list(range(len(texts) - 1, -1, -1))
    for ci in order:
        sp = spans_per_chunk[ci]
        if len(sp) >= 2:
            return texts[ci][sp[-2].own_end: sp[-1].start]
    tail = texts[-1] if texts else ""
    return "\n" if "\n" in tail else " "


def page_edit(page_texts: Sequence[str], drop_ids: Sequence[str] = (),
              move_last_ids: Sequence[str] = ()) -> tuple[list[str], dict]:
    """Apply a fact-level edit to a page of chunk texts.

    ``drop_ids``       facts to delete (``SUPPRESS``).
    ``move_last_ids``  facts to relocate to the very END of the assembled page
                       (``DEMOTE_LATE``): removed from their own chunk and
                       re-appended, in ascending serial order, to the end of the
                       LAST chunk's CONTENT — before its trailing whitespace and
                       before any dangling next-fact serial (see
                       :func:`_content_tail`) — so the highest serial ends up
                       last, which is the position the probe measured as
                       helpful.

    Returns ``(new_page, report)``. The page keeps its length and its chunk
    order; only chunk *contents* change. Raises ``LookupError`` when an id is
    not on the page and ``ValueError`` when the two id sets overlap — the caller
    (``MABAdapter.apply_read_decision``) turns either into a fall-back to the
    native page rather than a guess.

    Token accounting, stated exactly. A move always preserves the fact multiset;
    ``report["delta_chars"]`` is additionally **0** whenever the moved fact
    leaves a chunk that still holds another fact, because it takes one separator
    with it and acquires one on arrival. The single exception is a fact that was
    ALONE in its chunk: there was no separator to take, so the arrival separator
    is a net ``+len(sep)`` and the source chunk becomes empty (kept in place —
    the page count must never change). In this arena a chunk carries 230-260
    facts, so that corner cannot arise; it is reported rather than assumed away.
    """
    texts = list(page_texts)
    spans = [fact_spans(t) for t in texts]
    where: dict[str, tuple[int, FactSpan]] = {}
    for ci, sp in enumerate(spans):
        for s in sp:
            where.setdefault(f"fact:{s.serial}", (ci, s))

    drop, move = list(drop_ids), list(move_last_ids)
    both = sorted(set(drop) & set(move))
    if both:
        raise ValueError(f"ids in both drop and move sets: {both[:3]}")
    missing = [i for i in drop + move if i not in where]
    if missing:
        raise LookupError(f"{len(missing)} id(s) not on this page, "
                          f"first={missing[0]!r}")

    sep = _fact_separator(spans, texts)
    moved = sorted(((where[i][1].serial, i) for i in move))
    moved_bytes = [texts[where[i][0]][where[i][1].start: where[i][1].own_end]
                   for _, i in moved]

    cuts: dict[int, list[FactSpan]] = {}
    for i in drop + move:
        ci, s = where[i]
        cuts.setdefault(ci, []).append(s)

    out = list(texts)
    for ci, ss in cuts.items():
        t = out[ci]
        for s in sorted(ss, key=lambda x: x.del_start, reverse=True):
            t = t[: s.del_start] + t[s.del_end:]
        out[ci] = t

    if moved_bytes:
        core, trail = _content_tail(out[-1])
        out[-1] = core + "".join(sep + m for m in moved_bytes) + trail

    before = sum(len(t) for t in texts)
    after = sum(len(t) for t in out)
    report = {
        "n_chunks": len(texts),
        "n_dropped": len(drop), "n_moved": len(move),
        "dropped_serials": sorted(where[i][1].serial for i in drop),
        "moved_serials": [s for s, _ in moved],
        "separator": sep,
        "chars_before": before, "chars_after": after,
        "delta_chars": after - before,
    }
    return out, report


def select_pool(facts: Sequence[MemoryRecord], query_vector, cap: int) -> list[MemoryRecord]:
    """The read-gate candidate pool rule, in one importable place (T11).

    Facts without vectors are excluded (the gate would refuse them); when more
    than ``cap`` remain and a query vector exists, keep the ``cap`` most
    query-similar facts; the selected facts keep their original (chunk-rank,
    serial) order so the pool is deterministic. The Stage-1 calibration
    harness imports THIS function, so the offline grid measures exactly the
    pool the live adapter builds.
    """
    kept = [f for f in facts if f.vector is not None]
    cap = max(2, int(cap))
    if len(kept) <= cap:
        return kept
    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float64)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        sims = np.stack([np.asarray(f.vector, dtype=np.float64)
                         for f in kept]) @ q
        keep = sorted(np.argsort(-sims, kind="stable")[:cap].tolist())
        return [kept[i] for i in keep]
    return kept[:cap]


def native_page_from_scored(scored: Sequence[tuple[Any, float]], top_k: int) -> list[str]:
    """The exact list ``TextRetriever.retrieve`` returned before the hook.

    The native code was ``[doc.page_content for doc in
    similarity_search(query, k=top_k)][:top_k]``. The hooked code asks for the
    full ranking instead and truncates here. FAISS flat search is exact, so the
    first ``top_k`` entries of the full ranking are the same documents in the
    same order as a ``k=top_k`` search. This function is the single place that
    equivalence lives, so it can be tested (``test_shadow_neutrality.py``).
    """
    return [doc.page_content for doc, _ in scored[:top_k]]


def to_similarity(scores: Sequence[float], kind: str = "similarity") -> np.ndarray:
    """Normalize a score vector to the H-Nav convention (higher is better).

    ``"similarity"``  already cosine-like; passed through.
    ``"l2sq"``        FAISS squared-L2 distance -> ``cos × 100``, exact for unit
                      vectors and strictly decreasing, so ranks are preserved.
    """
    s = np.asarray(scores, dtype=np.float64)
    if kind == "l2sq":
        return (1.0 - s / 2.0) * 100.0
    if kind != "similarity":
        raise ValueError(f"unknown score_kind {kind!r}")
    return s


@dataclass
class ChunkRecord:
    """One memorize chunk, plus the facts it contains."""

    id: str
    text: str
    index: int
    fact_ids: list[str] = field(default_factory=list)


class MABAdapter:
    """Shadow-mode instrumentation for MemoryAgentBench.

    All components are injected and all are optional: with none of them the
    adapter still records candidate structure, conflicts and rankings, which is
    everything T4 needs. T5 supplies ``signals``; T6 supplies ``geometry`` and
    ``diff``.
    """

    def __init__(self, cfg=None, embedder=None, replica=None, audit: AuditLogger | None = None,
                 signals=None, geometry=None, diff=None, write_policy=None,
                 read_policy=None) -> None:
        self.cfg = cfg or _config.get_config()
        # Deliberate protocol transition (T11, STAGE1_PLAN.md §2 Faz B step 2):
        # the Stage-0 require_not_live() guard is lifted HERE and only here —
        # live mode arms the read-path rerank seam below. Stage-0 measurement
        # scripts keep calling config.require_not_live() themselves.
        self.mode = self.cfg.mode
        self.embedder = embedder
        self.replica = replica
        self.audit = audit
        self.signals = signals
        self.geometry = geometry
        self.diff = diff
        self.write_policy = write_policy
        self.read_policy = read_policy

        self.facts: list[MemoryRecord] = []
        self.chunks: list[ChunkRecord] = []
        self._chunk_by_text: dict[str, ChunkRecord] = {}
        self._fact_by_id: dict[str, MemoryRecord] = {}
        self._fact_chunk: dict[str, str] = {}
        self._chunk_vec: dict[str, np.ndarray] = {}
        self._pending_question_id: Any = None
        self.write_index = WriteConflictIndex()
        self._read_index: ReadConflictIndex | None = None
        self.context_id: Any = None
        self.n_write_decisions = 0
        self.n_read_decisions = 0
        self.n_reranks_applied = 0
        self.n_rerank_fallbacks = 0
        self.n_embed_errors = 0
        # T13 fact-level counters. chars_removed is the token-accounting
        # tripwire: SUPPRESS must show it positive, DEMOTE_LATE exactly zero.
        self.n_fact_edits_applied = 0
        self.n_facts_suppressed = 0
        self.n_facts_demoted = 0
        self.chars_removed = 0

    # ── embedding ────────────────────────────────────────────────────────────
    def embed(self, texts: Sequence[str]) -> np.ndarray | None:
        """Never raises: an unreachable embedding endpoint must degrade to
        "no vectors" (gate stays quiet, benchmark unharmed), not to a broken
        memorize pass. Failures are counted so they are visible in audit."""
        if self.embedder is None:
            return None
        try:
            return self.embedder.encode(list(texts))
        except Exception:  # noqa: BLE001
            self.n_embed_errors += 1
            return None

    def _embed_one(self, text: str) -> np.ndarray | None:
        v = self.embed([text])
        return None if v is None else v[0]

    # ── store views ──────────────────────────────────────────────────────────
    def store_view(self) -> StoreView:
        """Fact-level view of everything admitted so far, in admission order."""
        return StoreView.from_records(self.facts)

    def chunk_store_view(self) -> StoreView:
        """Chunk-level view — the unit the native retriever actually indexes."""
        return StoreView.from_records([
            MemoryRecord(id=c.id, text=c.text, vector=self._chunk_vec.get(c.id),
                         version=c.index, metadata={"fact_ids": c.fact_ids})
            for c in self.chunks
        ])

    # ── read index (whole store — read path only) ────────────────────────────
    def read_index(self) -> ReadConflictIndex:
        if self._read_index is None:
            idx = ReadConflictIndex()
            for rec in self.facts:
                idx.add(rec.metadata.get("key"),
                        Entry(rec.version, rec.id, rec.text, rec.metadata.get("object")))
            self._read_index = idx
        return self._read_index

    # ── write path ───────────────────────────────────────────────────────────
    def probe_generator(self, cand: Candidate) -> list[str]:
        """Deterministic probes from DECISION-TIME information only.

        Must not use benchmark questions, future facts or evaluator output. When
        the fact parses, the probe is its own relation template with the object
        slot left open; otherwise the candidate's own leading text.
        """
        key = cand.metadata.get("key")
        if key:
            return [to_probe(key[0], key[1])]
        return [cand.text[:200]]

    def candidate_for(self, serial: int, text: str, chunk_index: int = 0) -> Candidate:
        key, obj = fact_key(text)
        cand = Candidate(
            id=f"fact:{serial}",
            text=text,
            vector=self._embed_one(text),
            op="ADD",
            version=serial,
            metadata={"context_id": self.context_id, "chunk_index": chunk_index,
                      "key": key, "relation": key[0] if key else None,
                      "subject": key[1] if key else None, "object": obj},
        )
        # Predecessor resolution is ONLINE and must not look ahead: only facts
        # with a strictly smaller serial have been observed at this point.
        prior = self.write_index.latest_before(key, serial)
        if prior is not None:
            cand = cand.with_prior(MemoryRecord(id=prior.id, text=prior.text,
                                                version=prior.version))
        return cand

    def before_memorize(self, message: str, context_id: Any = None,
                        chunk_index: int = 0) -> str:
        """Hook at ``agent.py`` ``send_message(..., memorizing=True)``.

        Returns ``message`` itself — the same object, not a copy — so the caller
        is byte-identical to the unhooked path by construction.
        """
        if self.mode == _config.MODE_OFF:
            return message
        self.context_id = context_id
        store = self.store_view()

        for serial, text in explode_facts(message):
            cand = self.candidate_for(serial, text, chunk_index)
            decision = self._shadow_write_decision(cand, store)
            self.n_write_decisions += 1

            if self.audit is not None:
                self.audit.log_write(
                    cand, geometry=self._geometry_of(cand, store),
                    diff=self._diff_of(cand), retrieval_effect=self._effect_of(cand, store),
                    decision=decision, native_action="ADD", store_size=len(store),
                    context_id=context_id, chunk_index=chunk_index)

            self._admit(cand, chunk_index)
            store = store.with_provisional(cand)   # native behaviour: always admit

        self._register_chunk(message, chunk_index)
        return message

    def _shadow_write_decision(self, cand: Candidate, store: StoreView) -> Decision:
        """Stage 0 emits PASS and nothing else. The policies that could emit
        anything else are gated behind the T8 human decision and are not written
        yet — that is deliberate, not an omission."""
        reasons = {"native_action": "ADD", "has_prior": cand.prior_id is not None}
        if self.write_policy is not None:
            d = self.write_policy.decide(self._geometry_of(cand, store),
                                         self._diff_of(cand),
                                         self._effect_of(cand, store),
                                         ctx=dict(cand.metadata))
            d.shadow = self.mode != _config.MODE_LIVE
            return d
        return Decision(action="PASS", reasons=reasons, shadow=True)

    def _admit(self, cand: Candidate, chunk_index: int) -> None:
        rec = MemoryRecord(id=cand.id, text=cand.text, vector=cand.vector,
                           version=cand.version, metadata=dict(cand.metadata))
        self.facts.append(rec)
        self._fact_by_id[rec.id] = rec
        self.write_index.observe(cand.metadata.get("key"),
                                 Entry(cand.version, cand.id, cand.text,
                                       cand.metadata.get("object")))
        self._read_index = None   # invalidate; rebuilt lazily on the read path

    def _register_chunk(self, text: str, index: int) -> ChunkRecord:
        existing = self._chunk_by_text.get(text)
        if existing is not None:
            return existing
        chunk = ChunkRecord(id=_text_id("chunk", text), text=text, index=index)
        for serial, _ in explode_facts(text):
            fid = f"fact:{serial}"
            chunk.fact_ids.append(fid)
            self._fact_chunk[fid] = chunk.id
        self.chunks.append(chunk)
        self._chunk_by_text[text] = chunk
        return chunk

    # ── optional module plumbing ─────────────────────────────────────────────
    def _geometry_of(self, cand: Candidate, store: StoreView) -> Any:
        if self.geometry is None or cand.vector is None or len(store) == 0:
            return None
        return self.geometry.compute(cand, store)

    def _diff_of(self, cand: Candidate) -> Any:
        if self.diff is None:
            return None
        return self.diff.compute_if_update(cand.prior_text, cand.text,
                                           ctx=dict(cand.metadata))

    def _effect_of(self, cand: Candidate, store: StoreView) -> Any:
        if self.replica is None or self.signals is None or cand.vector is None:
            return None
        probes = self.probe_generator(cand)
        eff = self.replica.simulate_insert(store, cand, probes)
        return self.signals.compute_delta(eff["before"], eff["after"], self_id=cand.id)

    # ── read path ────────────────────────────────────────────────────────────
    def on_retrieve(self, query: str, ranked_texts: Sequence[str],
                    scores: Sequence[float], top_k: int,
                    query_vector: Sequence[float] | None = None,
                    question_id: str | None = None,
                    score_kind: str = "similarity") -> Decision:
        """Hook at ``TextRetriever.retrieve``. Receives the FULL pre-truncation
        ranking; returns a shadow ``Decision`` the caller must ignore.

        ``score_kind="l2sq"`` means the caller passed FAISS's squared-L2
        distances (lower is better). They are converted here — in one testable
        place — to ``cos × 100`` via ``cos = 1 - d²/2``, which is exact for the
        L2-normalized embeddings both shipped embedders produce, and is a
        strictly decreasing transform, so the ranking is untouched.
        """
        if self.mode == _config.MODE_OFF:
            return Decision(action="PASS", shadow=True)

        scores = to_similarity(scores, score_kind)
        ids = []
        for i, text in enumerate(ranked_texts):
            chunk = self._chunk_by_text.get(text) or self._register_chunk(text, i)
            ids.append(chunk.id)

        view = RetrievalView(query=query,
                             query_vector=np.asarray(query_vector, dtype=np.float32)
                             if query_vector is not None else None,
                             ids=ids, scores=np.asarray(scores, dtype=np.float64),
                             top_k=top_k)
        return self.after_retrieval(view, self.store_view(), query,
                                    question_id=question_id)

    def after_retrieval(self, view: RetrievalView, store: StoreView, query: str,
                        question_id: str | None = None) -> Decision:
        """Stale-supersession detection — the primary read-side target.

        Uses only the retrieved chunks and the store. At read time the whole
        store is legitimately visible, so ``ReadConflictIndex.latest`` is the
        correct API here (and ``latest_before`` would be wrong).
        """
        idx = self.read_index()
        retrieved_chunks = set(view.top_ids)
        retrieved_facts = self.facts_in(retrieved_chunks)

        stale_hits = []
        for rec in retrieved_facts:
            newest = idx.superseder_of(rec.id)
            if newest is None:
                continue
            superseder_chunk = self._fact_chunk.get(newest.id)
            stale_hits.append({
                "old_id": rec.id, "new_id": newest.id,
                "old_version": rec.version, "new_version": newest.version,
                "superseder_retrieved": superseder_chunk in retrieved_chunks,
                "rank_of_superseder": view.rank_of(superseder_chunk)
                if superseder_chunk else None,
            })

        conflicted_keys = {rec.metadata.get("key") for rec in retrieved_facts
                           if idx.is_conflicted(rec.metadata.get("key"))}
        conflict = {"stale_in_topk": stale_hits,
                    "n_conflicted_keys_in_topk": len(conflicted_keys - {None}),
                    "n_facts_in_topk": len(retrieved_facts)}

        sig = None
        if self.signals is not None:
            sig = self.signals.compute(view)

        if self.read_policy is not None:
            try:
                pool = self._read_candidates(view)
                decision = self.read_policy.decide(
                    pool, ordering=list(view.top_ids), signal_view=sig,
                    latest_key=self.latest_key, id_map=self._fact_chunk.get)
                decision.reasons["n_stale_in_topk"] = len(stale_hits)
                decision.reasons["n_read_candidates"] = len(pool)
            except Exception as e:  # noqa: BLE001 — fail OPEN, but audibly
                decision = Decision(action="PASS", reasons={
                    "read_policy_error": repr(e)[:300],
                    "n_stale_in_topk": len(stale_hits)})
            decision.shadow = self.mode != _config.MODE_LIVE
        else:
            decision = Decision(action="PASS", shadow=True,
                                reasons={"n_stale_in_topk": len(stale_hits)})

        # Campaign tripwire: fact vectors must come from the fp32 cache, so a
        # nonzero miss count in a live arm's audit trail flags dtype-mixed
        # geometry (see _build_components / serve_stage1_embed.sh).
        if self.embedder is not None and hasattr(self.embedder, "n_misses"):
            decision.reasons["embed_cache"] = {
                "hits": getattr(self.embedder, "n_hits", None),
                "misses": self.embedder.n_misses,
                "embed_errors": self.n_embed_errors,
            }

        self.n_read_decisions += 1
        if self.audit is not None:
            self.audit.log_read(view, decision, signals=sig, conflict=conflict,
                                store_size=len(store), question_id=question_id)
        return decision

    def facts_in(self, chunk_ids: Iterable[str]) -> list[MemoryRecord]:
        wanted = set(chunk_ids)
        out = []
        for chunk in self.chunks:
            if chunk.id in wanted:
                out.extend(self._fact_by_id[f] for f in chunk.fact_ids
                           if f in self._fact_by_id)
        return out

    # ── read-gate plumbing (Stage 1, T11) ────────────────────────────────────
    @staticmethod
    def latest_key(rec: MemoryRecord) -> int:
        """LATEST is the benchmark's own rule: the highest fact serial, which
        the adapter stored as ``version``. Supplied to the gate so the core
        never has to know what a serial is."""
        return rec.version

    @staticmethod
    def same_key_pair(a: MemoryRecord, b: MemoryRecord) -> bool:
        """Subject-identity screen for the gate (supervisor Note 1): a pair may
        reach the NLI only when BOTH facts parse and their ``(relation,
        subject)`` keys are equal. Unparseable facts are rejected rather than
        waved through — the NLI has been measured to rubber-stamp
        same-template/different-subject pairs, so absence of identity evidence
        must not default to trust. Parser coverage is 99.5%+, so the recall
        cost is bounded and the calibration artifact reports it."""
        ka = a.metadata.get("key")
        return ka is not None and ka == b.metadata.get("key")

    def _read_candidates(self, view: RetrievalView) -> list[MemoryRecord]:
        """The gate's candidate pool for one retrieval — see :func:`select_pool`
        for the rule; ``cap`` is ``cfg.top_m`` (the frozen Stage-0
        neighbourhood depth, 50). The cap is what keeps the gate tractable —
        a 64k page holds ~2,900 facts and the gate's leave-one-out residual is
        quadratic in pool size."""
        return select_pool(self.facts_in(view.top_ids), view.query_vector,
                           self.cfg.top_m)

    # ── agent.py send_message callbacks ──────────────────────────────────────
    def on_send_message_enter(self, message: str, memorizing: bool = False,
                              query_id: Any = None, context_id: Any = None) -> None:
        if self.mode == _config.MODE_OFF:
            return
        if context_id is not None and context_id != self.context_id:
            self.context_id = context_id
        if memorizing:
            self.before_memorize(message, context_id=context_id,
                                 chunk_index=len(self.chunks))
        else:
            self._pending_question_id = query_id

    def on_send_message_exit(self, result: Any, message: str, memorizing: bool = False,
                             query_id: Any = None, context_id: Any = None) -> Any:
        """Returns ``result`` unchanged. Present so that live mode has a seam;
        in shadow it exists only to close the audit record."""
        return result

    def apply_read_decision(self, decision: Decision, scored, top_k: int) -> list[str]:
        """Live-mode execution seam (T11 rerank; T13 fact-level edits).

        Returns the page of texts the retriever hands back. Three armed actions,
        one shared set of guards:

        ``RERANK``       same chunk set, same count, order only.
        ``SUPPRESS``     same chunk count and order; the named facts are spliced
                         out of the chunk text they live in. Tokens go DOWN.
        ``DEMOTE_LATE``  same chunk count and order; the named facts move to the
                         end of the last chunk's content. Token-neutral.

        Guards, in order — any of them failing returns the NATIVE page:

        * mode is ``live`` and the decision is armed (``shadow=False``) and
          carries a known action with a payload — anything else returns the
          native page byte-identically (off/shadow neutrality);
        * ``RERANK``'s order is exactly a permutation of the native page's chunk
          ids; a fact-level edit's ids are all locatable on the page and its
          drop/move sets are disjoint. Failures are counted in
          ``n_rerank_fallbacks``, never guessed at.
        """
        native = native_page_from_scored(scored, top_k)
        if (self.mode != _config.MODE_LIVE or decision is None or decision.shadow
                or decision.action not in ("RERANK", "SUPPRESS", "DEMOTE_LATE")
                or not decision.payload):
            return native
        if decision.action in ("SUPPRESS", "DEMOTE_LATE"):
            return self._apply_fact_edit(decision, native)
        order = list((decision.payload or {}).get("order") or [])
        text_of: dict[str, str] = {}
        for t in native:
            chunk = self._chunk_by_text.get(t)
            if chunk is None:                      # unknown page — refuse to act
                self.n_rerank_fallbacks += 1
                return native
            text_of[chunk.id] = t
        if sorted(order) != sorted(text_of) or len(order) != len(native):
            self.n_rerank_fallbacks += 1           # duplicate texts collapse here too
            return native
        self.n_reranks_applied += 1
        return [text_of[cid] for cid in order]

    def _apply_fact_edit(self, decision: Decision, native: list[str]) -> list[str]:
        """``SUPPRESS`` / ``DEMOTE_LATE`` execution. Fail-open by construction:
        any irregularity — an id the page does not carry, overlapping id sets,
        an unparseable chunk — returns the native page and increments
        ``n_rerank_fallbacks``, so a broken edit degrades to "no intervention"
        and is visible in the audit trail rather than silent."""
        payload = decision.payload or {}
        drop = list(payload.get("drop_ids") or ())
        move = list(payload.get("move_last_ids") or ())
        if not drop and not move:
            return native
        try:
            page, report = page_edit(native, drop_ids=drop, move_last_ids=move)
        except Exception as e:  # noqa: BLE001 — fail OPEN, but audibly
            self.n_rerank_fallbacks += 1
            decision.reasons["page_edit_error"] = repr(e)[:300]
            return native
        self.n_fact_edits_applied += 1
        self.n_facts_suppressed += report["n_dropped"]
        self.n_facts_demoted += report["n_moved"]
        self.chars_removed += -report["delta_chars"]
        decision.reasons["page_edit"] = report
        return page


_ADAPTER: MABAdapter | None = None


def _build_components(cfg) -> dict:
    """Best-effort default components for a hook-constructed adapter (T11).

    Everything degrades independently and silently to ``None`` — the benchmark
    must run whatever is broken — but every degradation is VISIBLE: a live arm
    without a ``read_policy`` produces zero RERANK records in the audit log,
    which the campaign's pre-registered analysis treats as a void run. Heavy
    imports stay inside this function (no torch at import time).

    Embeddings: shared disk cache first (the fp32 T1 cache — for the campaign
    subsets every fact text is already there), served endpoint as a
    NON-PERSISTING fallback so a differently-served dtype can never poison the
    fp32 cache namespace. ``n_misses`` on the wrapper is the tripwire.
    """
    out: dict = {}
    try:
        from hnav.core.retrieval_signals import RetrievalSignals
        out["signals"] = RetrievalSignals(top_m=cfg.top_m, k=cfg.churn_k)
    except Exception:  # noqa: BLE001
        pass
    try:
        from hnav.core.audit import audit_from_env
        out["audit"] = audit_from_env(cfg, kind="read",
                                      arm=f"stage1-{cfg.mode}")
    except Exception:  # noqa: BLE001
        pass
    try:
        from hnav.core.embedding import (DiskCachedEmbedder, OpenAIEmbedder,
                                         cache_key)
        out["embedder"] = DiskCachedEmbedder(
            OpenAIEmbedder(base_url=cfg.embed_base_url, model=cfg.embed_model),
            cfg.emb_cache_dir,
            cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length),
            persist=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        from hnav.core.read_gate import ReadGate, build_nli
        from hnav.core.read_policy import (FACT_MECHANISMS, ReadFactPolicy,
                                           ReadRerankPolicy, stage1_thresholds)
        gate = ReadGate(build_nli(cfg), stage1_thresholds(),
                        pair_filter=MABAdapter.same_key_pair)
        mech = getattr(cfg, "read_mechanism", "rerank")
        if mech in FACT_MECHANISMS:
            out["read_policy"] = ReadFactPolicy(gate, mechanism=mech)
        elif mech == "rerank":
            out["read_policy"] = ReadRerankPolicy(gate)
        else:
            raise ValueError(
                f"HNAV_READ_MECHANISM={mech!r} unknown; expected 'rerank' or "
                f"one of {FACT_MECHANISMS}")
    except Exception:  # noqa: BLE001
        pass
    return out


def get_adapter(**kw) -> MABAdapter | None:
    """Process-wide adapter, or ``None`` when ``HNAV_MODE=off``.

    The benchmark hooks call this. It must never raise: a misconfigured H-Nav
    has to degrade to "no instrumentation", never to "broken benchmark".
    Hook callers pass no kwargs; for them the default component stack
    (:func:`_build_components`) is constructed in shadow and live modes.
    Explicit kwargs (tests, measurement scripts) are used verbatim.
    """
    global _ADAPTER
    try:
        cfg = _config.get_config()
        if cfg.mode == _config.MODE_OFF:
            return None
        if _ADAPTER is None:
            _ADAPTER = MABAdapter(cfg=cfg, **(kw or _build_components(cfg)))
        return _ADAPTER
    except Exception:  # noqa: BLE001
        return None


def reset_adapter() -> None:
    global _ADAPTER
    _ADAPTER = None
