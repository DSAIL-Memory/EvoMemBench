"""Audit logging.  [T3]

One JSONL record per decision, schemas ``hnav.write.v1`` / ``hnav.read.v1``
(plan §6). Two separate writers, deliberately:

``AuditLogger``   — the ONLINE path. Writes what the decision saw. It has no
                    access to, and no field for, evaluator output.
``OutcomeLogger`` — the OFFLINE post-hoc pass. Writes to a *different file*,
                    joined back by ``run_id + candidate_id``. The online record
                    is closed before grading runs (plan §11).

Keeping these apart is the mechanical enforcement of "no evaluator output
online": there is no code path by which an outcome can reach an online record,
because the online writer cannot express one.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Decision, RetrievalView

__all__ = ["AuditLogger", "OutcomeLogger", "jsonable"]

WRITE_SCHEMA = "hnav.write.v1"
READ_SCHEMA = "hnav.read.v1"


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and dataclasses to JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return str(obj)


class _JsonlWriter:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict) -> dict:
        record = jsonable(record)
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "_JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AuditLogger:
    """Online decision log. ``path=None`` keeps records in memory (tests)."""

    def __init__(self, path: Path | str | None = None, run_id: str = "dev",
                 arm: str = "A0-shadow", mode: str = "shadow", suite: str = "",
                 subset: str = "") -> None:
        self.writer = _JsonlWriter(path)
        self.run_id = run_id
        self.arm = arm
        self.mode = mode
        self.suite = suite
        self.subset = subset

    @property
    def records(self) -> list[dict]:
        return self.writer.records

    def _envelope(self, schema: str) -> dict:
        return {"schema": schema, "run_id": self.run_id, "arm": self.arm,
                "mode": self.mode, "suite": self.suite, "subset": self.subset}

    def log_write(self, cand, geometry: Any, diff: Any, retrieval_effect: Any,
                  decision: Decision, native_action: str = "ADD",
                  store_size: int = 0, **ctx) -> dict:
        rec = self._envelope(WRITE_SCHEMA)
        rec.update({
            "context_id": ctx.pop("context_id", None),
            "step": ctx.pop("step", None),
            "chunk_index": ctx.pop("chunk_index", None),
            "candidate_id": cand.id,
            "operation": cand.op,
            "candidate_text": cand.text,
            "candidate_version": cand.version,
            "prior_id": cand.prior_id,
            "prior_text": cand.prior_text,
            "store_size": store_size,
            "geometry": geometry,
            "diff": diff,
            "retrieval_effect": retrieval_effect,
            "native_action": native_action,
            "hnav_action": decision.action,
            "hnav_reasons": decision.reasons,
            "shadow": decision.shadow,
        })
        rec.update(ctx)
        return self.writer.write(rec)

    def log_read(self, view: RetrievalView, decision: Decision, signals: Any = None,
                 conflict: Any = None, native_topk_ids: list[str] | None = None,
                 store_size: int = 0, question_id: str | None = None,
                 max_ranking: int = 200, **ctx) -> dict:
        """``max_ranking`` truncates the persisted ranking; 18,332-entry rankings
        would otherwise dominate the log. Signals are computed on the FULL
        ranking before truncation — only the stored copy is shortened."""
        rec = self._envelope(READ_SCHEMA)
        rec.update({
            "question_id": question_id,
            "query": view.query,
            "store_size": store_size,
            "top_k": view.top_k,
            "ranking": {"ids": view.ids[:max_ranking],
                        "scores": view.scores[:max_ranking],
                        "truncated_at": view.top_k,
                        "n_total": len(view.ids)},
            "signals": signals,
            "conflict": conflict,
            "native_topk_ids": native_topk_ids if native_topk_ids is not None else view.top_ids,
            "hnav_action": decision.action,
            "hnav_payload_ids": decision.payload,
            "hnav_reasons": decision.reasons,
            "shadow": decision.shadow,
        })
        rec.update(ctx)
        return self.writer.write(rec)

    def close(self) -> None:
        self.writer.close()


class OutcomeLogger:
    """Offline post-hoc labels. A separate file, written after grading.

    The join key is ``(run_id, candidate_id)`` for write records and
    ``(run_id, question_id)`` for read records.
    """

    def __init__(self, path: Path | str | None = None, run_id: str = "dev") -> None:
        self.writer = _JsonlWriter(path)
        self.run_id = run_id

    @property
    def records(self) -> list[dict]:
        return self.writer.records

    def log(self, key: dict, outcome: dict) -> dict:
        rec = {"schema": "hnav.outcome.v1", "run_id": self.run_id}
        rec.update(key)
        rec["outcome"] = outcome
        return self.writer.write(rec)

    def close(self) -> None:
        self.writer.close()


def default_audit_path(out_dir: Path, run_id: str, kind: str = "write") -> Path:
    return Path(out_dir) / "audit" / f"{run_id}.{kind}.jsonl"


def audit_from_env(cfg, kind: str = "write", **kw) -> AuditLogger:
    path = None
    if os.environ.get("HNAV_AUDIT", "1") != "0":
        path = default_audit_path(cfg.out_dir, cfg.run_id, kind)
    return AuditLogger(path, run_id=cfg.run_id, mode=cfg.mode, **kw)
