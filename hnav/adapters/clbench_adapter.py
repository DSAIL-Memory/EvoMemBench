"""CL-bench adapter — CrossEp-Know, the SECONDARY arena.  [T4, signal wiring T10]

``HNavMemoryWrapper`` wraps any ``cl_bench_memory`` backend behind the same
two-method ``Memory`` ABC (``retrieve`` / ``extract``). In shadow mode it
**delegates verbatim**: both methods return the inner backend's own return
value, the same objects, with no re-wrapping, so the caller cannot observe the
wrapper's presence.

Three things this arena needs that the primary one does not:

* there are no templates and no serial numbers, so ``Candidate.version`` is a
  monotone write counter and there is no key-based conflict index. The
  candidate's *predecessor* is therefore resolved geometrically: the nearest
  admitted record by cosine stands in for ``prior_id``/``prior_text``. That is
  decision-time information only — no benchmark questions, no future writes;
* the store is model-generated, so write-side effects can only ever be reported
  as *associational* — true counterfactual replay is unaffordable (plan §7.4);
* samples are clustered by ``context_id`` (ICC 0.346, design effect 3.20,
  effective N ≈ 276), which is recorded in every audit record so the analysis
  can cluster correctly. The runner does not pass ``context_id`` to
  ``extract``, so the wrapper recovers it from the backend's per-context
  ``memory_dir`` (the runner names that directory after the context).

Signal-space note. Native banks hold vectors from the benchmark's own
embedding API (DashScope text-embedding-v4, 1024-d); H-Nav's candidate vectors
come from its own embedder. Cosines across two spaces are meaningless, so the
adapter re-embeds the bank texts with its own embedder (memoized per text)
and computes every signal in that single space — the same methodology the
primary arena uses. Which space was used is recorded per write as
``bank_space`` so no reader has to guess.

Shadow legality: signals are computed and logged, nothing is acted on, no LLM
is ever called (the embedder is H-Nav's own, not a chat model), and off mode
remains an exact no-op — ``wrap_memory`` returns the backend itself.

Free-text probe strategy (Stage-0 CrossEp measurement, decided T10): with no
relation templates to derive a probe from, the candidate's own text is its
probe — ``simulate_insert``'s documented default — and the provisional-insert
effect is summarized by ``rank_self_after`` plus the neighbourhood displacement
stats (``churn@k``, ``rank_shift``, ``dH_self``). Nothing derived from
benchmark evaluation data may ever enter this path.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.core.audit import AuditLogger  # noqa: E402
from hnav.core.geometry import ABTTWhitening, GeometryModule, TauPolicy  # noqa: E402
from hnav.core.types import Candidate, Decision, MemoryRecord, StoreView  # noqa: E402

__all__ = ["CLBenchAdapter", "HNavMemoryWrapper", "wrap_memory", "result_annotation"]


def _context_key(memory: Any) -> str | None:
    """The context a backend instance belongs to, from its per-context
    ``memory_dir`` (the runner names that directory after the context_id).
    ``None`` when the backend does not expose one."""
    mdir = getattr(memory, "memory_dir", None)
    if not mdir:
        return None
    return os.path.basename(os.path.normpath(str(mdir)))


class CLBenchAdapter:
    """Signal computation for CrossEp-Know. Holds no benchmark objects."""

    def __init__(self, cfg=None, embedder=None, replica=None,
                 audit: AuditLogger | None = None, signals=None, geometry=None,
                 diff=None) -> None:
        self.cfg = cfg or _config.get_config()
        self.cfg.require_not_live()
        self.mode = self.cfg.mode
        self.embedder = embedder
        self.replica = replica
        self.audit = audit
        self.signals = signals
        self.geometry = geometry
        self.diff = diff
        self.write_counter = 0
        self.n_write_decisions = 0
        self.n_read_decisions = 0
        # Per-context state. The LIVE path shares ONE adapter (the module
        # singleton) across every context of a run, while the native banks are
        # per-context isolated — so any cross-context state would contaminate
        # the logged statistics: an MD5 dup-hash set spanning contexts flags a
        # chunk in context B as a "duplicate" of one admitted in context A,
        # which the native backend would never see. Geometry state (dup hashes
        # + adaptive tau) and the text->vector memo are therefore keyed by
        # context and LRU-bounded (a context, once finished, never returns;
        # the cap only needs to exceed the runner's worker count).
        self._memos: "OrderedDict[Any, dict[str, np.ndarray]]" = OrderedDict()
        self._geo_states: "OrderedDict[Any, GeometryModule]" = OrderedDict()

    MAX_CONTEXT_STATES = 64   # >> infer_context_memory's default 8 workers

    def _lru_get(self, table: OrderedDict, key: Any, factory):
        state = table.get(key)
        if state is None:
            state = factory()
            table[key] = state
        table.move_to_end(key)
        while len(table) > self.MAX_CONTEXT_STATES:
            table.popitem(last=False)
        return state

    def _memo_for(self, context_key: Any) -> dict[str, np.ndarray]:
        return self._lru_get(self._memos, context_key, dict)

    def _fresh_geometry(self) -> GeometryModule:
        """A new GeometryModule with the injected prototype's parameters but
        NONE of its accumulated state (dup hashes, tau history, whitening fit)."""
        proto = self.geometry
        whit = None
        if proto.whitening is not None:
            whit = ABTTWhitening(n_components=proto.whitening.n_components,
                                 min_fit_n=proto.whitening.min_fit_n)
        tp = proto.tau_policy
        return GeometryModule(
            whitening=whit,
            tau_policy=TauPolicy(init_tau=tp.init_tau, z=tp.z, min_n=tp.min_n,
                                 lo=tp.lo, hi=tp.hi),
            max_basis=proto.max_basis)

    def _geometry_for(self, context_key: Any) -> GeometryModule | None:
        """The injected module for keyless (single-context/offline) use; a
        per-context clone otherwise, so contexts never share dup/tau state."""
        if self.geometry is None:
            return None
        if context_key is None:
            return self.geometry
        return self._lru_get(self._geo_states, context_key, self._fresh_geometry)

    # ── store views built from the backend's own persisted state ─────────────
    def store_view(self, memory: Any) -> StoreView:
        """Build a StoreView from a live backend.

        ``Qwen3EmbeddingMemory`` keeps ``memory_bank`` (chunk entries) and
        ``embeddings`` (L2-normalized, same order) in lockstep and persists both
        to ``<memory_dir>/embeddings.jsonl``, so the view is exact and free —
        no re-embedding.
        """
        bank = getattr(memory, "memory_bank", None) or []
        vecs = getattr(memory, "embeddings", None) or []
        records = []
        for i, entry in enumerate(bank):
            vec = np.asarray(vecs[i], dtype=np.float32) if i < len(vecs) else None
            records.append(MemoryRecord(
                id=f"{entry.get('task_id', i)}_chunk{entry.get('chunk_index', i)}",
                text=entry.get("chunk_text", ""),
                vector=vec,
                version=i,
                metadata={"task_id": entry.get("task_id"),
                          "chunk_index": entry.get("chunk_index"),
                          "context_category": entry.get("context_category"),
                          "sub_category": entry.get("sub_category")},
            ))
        return StoreView.from_records(records)

    def to_candidate(self, content: str, **kwargs) -> Candidate:
        self.write_counter += 1
        vec = None
        if self.embedder is not None:
            memo = self._memo_for(kwargs.get("context_id"))
            vec = memo.get(content)
            if vec is None:
                vec = np.asarray(self.embedder.encode([content])[0],
                                 dtype=np.float32)
                memo[content] = vec
        return Candidate(
            id=f"{kwargs.get('task_id', 'task')}_w{self.write_counter}",
            text=content, vector=vec, op="ADD", version=self.write_counter,
            metadata={k: v for k, v in kwargs.items() if k != "content"},
        )

    # ── signal-space plumbing ────────────────────────────────────────────────
    def _signal_store(self, store: StoreView,
                      context_key: Any = None) -> StoreView | None:
        """The store re-embedded into the adapter's own space, or ``None``.

        ``None`` means signals cannot be computed (no embedder, or empty
        store) — a different statement from "the signals were zero", and the
        two are never collapsed: ``on_extract`` records which case occurred.
        The memo is per-context (see ``__init__``): banks are per-context
        isolated natively, and a shared memo would grow without bound across
        a live multi-context run.
        """
        if self.embedder is None or len(store) == 0:
            return None
        memo = self._memo_for(context_key)
        missing = [r.text for r in store.records if r.text not in memo]
        if missing:
            fresh = self.embedder.encode(missing)
            for t, v in zip(missing, fresh):
                memo[t] = np.asarray(v, dtype=np.float32)
        return StoreView.from_records([
            MemoryRecord(id=r.id, text=r.text, vector=memo[r.text],
                         version=r.version, metadata=r.metadata)
            for r in store.records
        ])

    # ── decisions (shadow: always PASS) ──────────────────────────────────────
    def on_extract(self, cand: Candidate, store: StoreView) -> Decision:
        """Compute and log write-side signals; decide nothing.

        Geometry (sim_max / QR novelty / MD5 exact-dup), marginal-diff against
        the nearest-neighbour predecessor, and the provisional-insert effect
        (candidate-as-its-own-probe) — all in shadow, all logged, none acted
        on. The decision is PASS unconditionally; the native path admits.
        """
        self.n_write_decisions += 1
        decision = Decision(action="PASS", shadow=True,
                            reasons={"native_action": "ADD"})

        ctx_key = cand.metadata.get("context_id")
        sig_store = self._signal_store(store, ctx_key)
        bank_space = "hnav" if sig_store is not None else (
            "empty_store" if len(store) == 0 else "unavailable_no_embedder")
        target = sig_store if sig_store is not None else store

        geo = None
        geometry = self._geometry_for(ctx_key)   # per-context dup/tau state
        if geometry is not None:
            geo = geometry.compute(cand, target)
            # Nearest-neighbour predecessor: no keys/serials in this arena, so
            # the closest already-admitted record stands in as the prior. Only
            # records admitted before this write are visible — no look-ahead.
            if cand.prior_id is None and geo.argmax_id is not None:
                cand = cand.with_prior(target.get(geo.argmax_id))
            geometry.observe(cand.text)        # the native path always admits

        diff = None
        if self.diff is not None:
            diff = self.diff.compute_if_update(cand.prior_text, cand.text,
                                               ctx=dict(cand.metadata))

        effect = None
        if (self.replica is not None and self.signals is not None
                and cand.vector is not None and sig_store is not None):
            views = self.replica.simulate_insert(sig_store, cand, [cand.text])
            effect = self.signals.compute_delta(views["before"], views["after"],
                                                self_id=cand.id)

        decision.reasons.update({
            "bank_space": bank_space,
            "has_prior": cand.prior_id is not None,
            "is_exact_dup": bool(geo.is_exact_dup) if geo is not None else None,
        })
        if self.audit is not None:
            self.audit.log_write(cand, geometry=geo, diff=diff,
                                 retrieval_effect=effect, decision=decision,
                                 native_action="ADD", store_size=len(store),
                                 context_id=cand.metadata.get("context_id")
                                 or cand.metadata.get("context_category"),
                                 bank_space=bank_space)
        return decision

    def on_retrieve(self, query: str, store: StoreView,
                    native_text: str = "", context_id: Any = None) -> Decision:
        self.n_read_decisions += 1
        decision = Decision(action="PASS", shadow=True)
        if self.replica is None or len(store) == 0:
            return decision
        # Rank in the adapter's own embedding space: the native bank vectors
        # come from the benchmark's embedding API and are not comparable with
        # the H-Nav query vector the replica produces.
        sig_store = self._signal_store(store, context_id)
        if sig_store is None:
            return decision
        view = self.replica.rank(sig_store, query)
        sig = self.signals.compute(view) if self.signals is not None else None
        decision.reasons["n_ranked"] = len(view.ids)
        if self.audit is not None:
            self.audit.log_read(view, decision, signals=sig, store_size=len(store),
                                native_chars=len(native_text))
        return decision

    def annotation(self) -> dict:
        """The additive ``"hnav"`` field for the per-sample result record."""
        return {"mode": self.mode, "run_id": self.cfg.run_id,
                "n_read_decisions": self.n_read_decisions,
                "n_write_decisions": self.n_write_decisions,
                "action": "PASS"}


class HNavMemoryWrapper:
    """Wraps a ``cl_bench_memory`` backend. Shadow mode delegates verbatim.

    Not declared as a subclass of ``Memory``: importing the benchmark ABC from
    ``hnav/`` would invert the dependency the architecture depends on. Duck
    typing is sufficient — ``build_memory``'s callers only ever call
    ``retrieve`` and ``extract`` — and attribute access falls through to the
    inner backend, so ``memory.memory_bank`` and friends keep working.
    """

    def __init__(self, inner: Any, adapter: CLBenchAdapter | None = None,
                 mode: str = _config.MODE_SHADOW) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_mode", mode)

    # ── Memory interface ─────────────────────────────────────────────────────
    def retrieve(self, query: str) -> tuple:
        result = self._inner.retrieve(query)
        if self._mode == _config.MODE_SHADOW and self._adapter is not None:
            try:
                text = result[0] if isinstance(result, tuple) and result else ""
                self._adapter.on_retrieve(query, self._adapter.store_view(self._inner),
                                          native_text=text,
                                          context_id=_context_key(self._inner))
            except Exception:  # noqa: BLE001 — instrumentation must never break a run
                pass
        return result   # the inner backend's own object, unchanged

    def extract(self, content: str, **kwargs) -> dict:
        if self._mode == _config.MODE_SHADOW and self._adapter is not None:
            try:
                store = self._adapter.store_view(self._inner)
                # The runner does not pass context_id to extract; the backend's
                # per-context memory_dir is named after it. Recovered into the
                # CANDIDATE's kwargs only — the inner call below receives the
                # caller's kwargs untouched.
                cand_kwargs = dict(kwargs)
                if "context_id" not in cand_kwargs:
                    ctx = _context_key(self._inner)
                    if ctx is not None:
                        cand_kwargs["context_id"] = ctx
                cand = self._adapter.to_candidate(content, **cand_kwargs)
                decision = self._adapter.on_extract(cand, store)
                assert decision.shadow, "shadow mode produced an actionable decision"
            except AssertionError:
                raise
            except Exception:  # noqa: BLE001
                pass
        return self._inner.extract(content, **kwargs)

    # ── transparency ─────────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def __repr__(self) -> str:
        return f"HNavMemoryWrapper({self._inner!r}, mode={self._mode})"


_ADAPTER: CLBenchAdapter | None = None


def get_adapter(**kw) -> CLBenchAdapter | None:
    global _ADAPTER
    try:
        cfg = _config.get_config()
        if cfg.mode == _config.MODE_OFF:
            return None
        if _ADAPTER is None:
            _ADAPTER = CLBenchAdapter(cfg=cfg, **kw)
        return _ADAPTER
    except Exception:  # noqa: BLE001
        return None


def reset_adapter() -> None:
    global _ADAPTER
    _ADAPTER = None


def wrap_memory(memory: Any) -> Any:
    """``registry.build_memory`` hook. Returns ``memory`` itself when off."""
    adapter = get_adapter()
    if adapter is None:
        return memory
    return HNavMemoryWrapper(memory, adapter, mode=adapter.mode)


def result_annotation() -> dict | None:
    """``infer_context_memory`` hook. ``None`` when off, so the caller adds
    nothing at all to the record."""
    adapter = get_adapter()
    return None if adapter is None else adapter.annotation()
