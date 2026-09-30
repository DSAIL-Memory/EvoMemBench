"""Every hnav module must import cleanly with no torch installed.  [T3]

The GPU box has torch; this laptop does not, and the benchmark processes that
load H-Nav in shadow mode should not pay an 8 GB import either. Every heavy
import (torch, transformers, faiss, openai, langchain) therefore lives *inside*
a function or method.

The check runs in a subprocess with a fresh interpreter, so it is unaffected by
whatever the pytest session has already imported.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HEAVY = ("torch", "transformers", "sentence_transformers", "faiss", "openai",
         "langchain", "langchain_community", "vllm")

# stage0 scripts are entry points, not libraries, but they must still be
# importable — a syntax error there would only surface on the GPU box.
SKIP_DIRS = {"__pycache__", "_cache", "_out", "deploy", "prompts", "tests"}


def _module_names() -> list[str]:
    mods = []
    for p in sorted((REPO / "hnav").rglob("*.py")):
        rel = p.relative_to(REPO)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    return [m for m in mods if m]


SCRIPT = """
import importlib, sys
mods = %r
heavy = %r
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, type(e).__name__ + ': ' + str(e)))
leaked = sorted(h for h in heavy if h in sys.modules)
print('FAILED=' + repr(failed))
print('LEAKED=' + repr(leaked))
"""


def test_all_modules_import_without_heavy_deps():
    mods = _module_names()
    assert len(mods) > 8, f"module discovery found only {mods}"

    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT % (mods, HEAVY)],
        capture_output=True, text=True, cwd=REPO, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr

    out = dict(line.split("=", 1) for line in proc.stdout.strip().splitlines()
               if "=" in line)
    failed = eval(out["FAILED"])  # noqa: S307 — our own repr, from our own subprocess
    leaked = eval(out["LEAKED"])  # noqa: S307

    assert not failed, "modules failed to import:\n" + "\n".join(f"  {m}: {e}" for m, e in failed)
    assert not leaked, (
        f"importing hnav pulled in {leaked}. Move the import inside the function "
        "that needs it — this suite must run on a machine with no torch."
    )


def test_heavy_imports_are_function_local():
    """Belt and braces: no top-level `import torch` textually, either."""
    import ast

    offenders = []
    for p in sorted((REPO / "hnav").rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in tree.body:  # top level only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in HEAVY:
                    offenders.append(f"{p.relative_to(REPO)}:{node.lineno} imports {n}")
    assert not offenders, "top-level heavy imports:\n" + "\n".join(offenders)
