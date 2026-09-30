"""``H_raw`` is logged and never decided on.  [T5, hard rule 6; T9 transition]

Three independent checks, because one of them alone is easy to route around:

1. the policy-facing accessor does not contain ``H_raw`` at all;
2. no ``Decision.reasons`` produced by a real shadow pass contains it, at any
   nesting depth;
3. an AST scan of every policy module that exists — and of ``read_gate.py``,
   which feeds the read policy — finds no reference to it.

PROTOCOL TRANSITION (T9, deliberate). Stage 0 shipped no policy modules and
this file asserted that. The T8 verdict (``KAPI_KARARI.md`` §3/§6:
instruments GO, write_policy NO_GO, read_policy CONDITIONAL) plus the user's
Stage-1 decision of 2026-08-15 (``STAGE1_PLAN.md`` §0, read-path rerank only)
changed the rule to:

* ``hnav/core/write_policy.py`` is FORBIDDEN **forever** — the NO_GO verdict
  is a measured result (no write-path headroom), not a deferral;
* ``hnav/core/read_policy.py`` is now PERMITTED (built in Faz B), and remains
  subject to the ``H_raw`` scan like everything else here;
* no other ``*policy*`` module may appear under ``hnav/core/``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from hnav import config as _config
from hnav.adapters import mab_adapter as mab
from hnav.core.retrieval_signals import POLICY_FORBIDDEN, RetrievalSignals
from hnav.core.types import RetrievalView

REPO = Path(__file__).resolve().parents[2]
SHADOW = _config.HNavConfig(mode=_config.MODE_SHADOW)

CHUNK = """Here is a list of facts:
0. Thomas Kyd was born in the city of London.
1. The chairperson of Fatah is Mahmoud Abbas.
2. Thomas Kyd was born in the city of Madrid.
"""


def test_policy_view_excludes_raw_entropy():
    sig = RetrievalSignals(top_m=4).compute(
        RetrievalView(query="q", query_vector=None, ids=list("abcd"),
                      scores=np.array([90.0, 80.0, 70.0, 60.0]), top_k=3))
    assert "H_raw" in sig.to_dict(), "H_raw must still be LOGGED"
    for name in POLICY_FORBIDDEN:
        assert name not in sig.for_policy(), f"{name} leaked into the policy view"


def _walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_no_decision_reasons_mention_raw_entropy(embedder):
    adapter = mab.MABAdapter(cfg=SHADOW, embedder=embedder,
                             signals=RetrievalSignals(top_m=50, k=3))
    adapter.before_memorize(CHUNK, context_id=0)
    decision = adapter.on_retrieve("Where was Thomas Kyd born?", [CHUNK], [88.0], top_k=1)

    reasons = list(_walk_strings(decision.reasons))
    for name in POLICY_FORBIDDEN:
        assert not any(name in r for r in reasons), f"{name} reached Decision.reasons"


def test_write_policy_is_forbidden_forever_and_read_policy_is_the_only_exception():
    """T9 protocol transition — see the module docstring for the full trail.

    write_policy NO_GO is a *measured verdict* (KAPI_KARARI.md §2: post-veto
    write-path headroom 0–1.6%, could-change-correctness 0.00 on the
    confirmation subset), so ``write_policy.py`` may never exist. read_policy
    was CONDITIONAL and the condition was met by the user's 2026-08-15
    decision (STAGE1_PLAN.md §0): ``read_policy.py`` is permitted, and nothing
    else matching ``*policy*`` is.
    """
    present = sorted(p.name for p in (REPO / "hnav/core").glob("*policy*.py"))
    forbidden = [n for n in present if n != "read_policy.py"]
    assert forbidden == [], (
        f"forbidden policy modules under hnav/core/: {forbidden}. "
        "write_policy.py is barred FOREVER by the T8 NO_GO verdict "
        "(KAPI_KARARI.md); only read_policy.py is permitted post-T8."
    )


def test_policy_modules_and_the_read_gate_never_reference_raw_entropy():
    """The H_raw scan: any ``*policy*`` module that exists, plus ``read_gate.py``
    (the module that feeds the read policy), must not reference H_raw.

    Kept live rather than added later, so the constraint outlives whoever
    remembers it.
    """
    paths = sorted((REPO / "hnav/core").glob("*policy*.py"))
    gate = REPO / "hnav/core/read_gate.py"
    if gate.exists():
        paths.append(gate)
    assert paths, "read_gate.py missing — the T9 gate should exist"
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            if name and any(f in str(name) for f in POLICY_FORBIDDEN):
                raise AssertionError(f"{path.name}:{node.lineno} references {name!r}")
