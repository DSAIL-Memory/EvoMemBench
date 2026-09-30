"""Every Stage-0 measurement script refuses HNAV_MODE=live.  [T11]

Supervisor audit note (T11): the T11 commit message claimed "every
hnav/stage0/ script still refuses live" while two of nine
(``m1_geometry_calibration.py``, ``report.py``) carried no guard at all. The
practical risk was nil — neither touches the adapters or the benchmark hooks —
but an invariant asserted in prose and enforced nowhere decays. So it is
enforced here instead of claimed.

The rule, post-T8: live mode exists for the Stage-1 READ-PATH campaign only
(``STAGE1_PLAN.md`` §2 Faz B step 2). ``MABAdapter`` deliberately does not
call the guard — that transition is pinned in ``test_shadow_neutrality.py``.
Everything under ``hnav/stage0/`` must, because a Stage-0 number measured
under intervention is not a Stage-0 number.

Two guard forms are accepted, because one module is deliberately standalone:
``cfg.require_not_live()`` (the config method) or a module-level
``require_not_live(env)`` (the stdlib-only inline twin in
``m1_geometry_calibration.py``). Both are checked BY BEHAVIOUR — the inline
twin is imported and called with a live env — not merely by grepping for the
name, so a guard that exists but does not raise fails this test.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STAGE0 = REPO / "hnav/stage0"


def stage0_scripts() -> list[Path]:
    return sorted(p for p in STAGE0.glob("*.py") if p.name != "__init__.py")


def test_there_are_scripts_to_check():
    names = [p.name for p in stage0_scripts()]
    assert len(names) >= 8, f"only found {names} — wrong tree?"


@pytest.mark.parametrize("path", stage0_scripts(), ids=lambda p: p.name)
def test_every_stage0_script_calls_a_live_guard_in_main(path: Path):
    """``main()`` must invoke a live guard before doing any work.

    AST, not substring: a mention inside a docstring or a comment does not
    count, and the call must be reachable from ``main`` rather than sitting in
    a helper nobody calls.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mains = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert mains, f"{path.name} has no main()"

    calls = []
    for node in ast.walk(mains[0]):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else \
            (f.id if isinstance(f, ast.Name) else None)
        if name:
            calls.append(name)
    assert "require_not_live" in calls, (
        f"{path.name}: main() never calls require_not_live(). Stage-0 "
        f"measurements must refuse HNAV_MODE=live (STAGE1_PLAN.md §2 Faz B "
        f"step 2); only hnav/adapters/mab_adapter.py is exempt."
    )


def _main_calls(source: str) -> list[str]:
    """The scan under test, factored out so it can be aimed at a fixture."""
    tree = ast.parse(source)
    mains = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "main"]
    out: list[str] = []
    for node in ast.walk(mains[0]):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else \
                (f.id if isinstance(f, ast.Name) else None)
            if name:
                out.append(name)
    return out


UNGUARDED = '''
"""A module whose docstring may discuss require_not_live freely."""
def main():
    # require_not_live() in a comment must not count either
    return run()
'''

GUARD_IN_DEAD_HELPER = '''
def helper(cfg):
    cfg.require_not_live()
def main():
    return run()
'''

GUARDED_METHOD = '''
def main():
    cfg = get_config()
    cfg.require_not_live()
    return run()
'''

GUARDED_INLINE = '''
def main():
    env = load_env()
    require_not_live(env)
    return run()
'''


@pytest.mark.parametrize("src,name", [
    (UNGUARDED, "docstring/comment only"),
    (GUARD_IN_DEAD_HELPER, "guard in an uncalled helper"),
])
def test_the_scan_catches_deliberately_unguarded_mains(src, name):
    assert "require_not_live" not in _main_calls(src), \
        f"the scan failed to catch the deliberate {name} violation"


@pytest.mark.parametrize("src,name", [
    (GUARDED_METHOD, "cfg.require_not_live()"),
    (GUARDED_INLINE, "inline require_not_live(env)"),
])
def test_the_scan_accepts_both_guard_forms(src, name):
    assert "require_not_live" in _main_calls(src), f"{name} was not recognized"


def test_the_config_guard_actually_raises():
    from hnav import config as _config
    live = _config.HNavConfig(mode=_config.MODE_LIVE)
    with pytest.raises(RuntimeError, match="live"):
        live.require_not_live()
    for mode in (_config.MODE_OFF, _config.MODE_SHADOW):
        _config.HNavConfig(mode=mode).require_not_live()   # must not raise


def test_the_inline_twin_actually_raises():
    """``m1_geometry_calibration`` is stdlib-only by design and reproduces the
    guard rather than importing it. Behaviour, not shape, is what matters, so
    the twin is loaded and exercised on both paths."""
    path = STAGE0 / "m1_geometry_calibration.py"
    spec = importlib.util.spec_from_file_location("_m1_guard_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1_guard_probe"] = mod
    try:
        spec.loader.exec_module(mod)
        with pytest.raises(RuntimeError, match="live"):
            mod.require_not_live({"HNAV_MODE": "live"})
        with pytest.raises(RuntimeError, match="live"):
            mod.require_not_live({"HNAV_MODE": " LIVE "})   # normalized
        mod.require_not_live({})                            # default off
        mod.require_not_live({"HNAV_MODE": "shadow"})
    finally:
        sys.modules.pop("_m1_guard_probe", None)
