"""Research-integrity audit: no gold-answer leakage into any online path.  [T4]

The benchmark's questions and its answer list sit in the *same JSON file* as the
fact contexts. Reading either one from a module that computes a signal or a
decision would invalidate every result in the thesis, and it is a two-character
mistake to make. So it is checked mechanically, by AST scan, over every module
under ``hnav/core/`` and ``hnav/adapters/``:

* no identifier, attribute, keyword argument, dict key or subscript naming a
  forbidden concept;
* no import of ``hnav.labeling.counterfactual`` (the module that legitimately
  holds evaluator output).

**Docstrings are exempt, comments are invisible to the AST.** Prose has to be
able to say what the rule is; only executable references are scanned. That
exemption is the one hole in the check and is stated here rather than hidden.

The scanner is exercised against a deliberately violating source below, so this
test can actually fail — a leakage audit that cannot fail is decoration.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ONLINE_DIRS = ["hnav/core", "hnav/adapters"]

# Substring-matched against lowercased identifiers and non-docstring strings.
FORBIDDEN = ("answers", "answer_key", "gold", "ground_truth", "rubric", "questions")
FORBIDDEN_IMPORTS = ("hnav.labeling.counterfactual", "labeling.counterfactual",
                     "counterfactual")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids() of the Constant nodes that are docstrings."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def scan_source(source: str, filename: str = "<src>") -> list[str]:
    """Return a list of leakage violations. Empty list == clean."""
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    violations: list[str] = []

    def flag(node, what: str, text: str) -> None:
        low = text.lower()
        for tok in FORBIDDEN:
            if tok in low:
                violations.append(
                    f"{filename}:{getattr(node, 'lineno', '?')}: {what} {text!r} matches {tok!r}")
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            flag(node, "name", node.id)
        elif isinstance(node, ast.Attribute):
            flag(node, "attribute", node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            flag(node, "keyword", node.arg)
        elif isinstance(node, ast.arg):
            flag(node, "argument", node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flag(node, "definition", node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                flag(node, "string", node.value[:80])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if any(m in a.name for m in FORBIDDEN_IMPORTS):
                    violations.append(f"{filename}:{node.lineno}: imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(m in mod for m in FORBIDDEN_IMPORTS) or \
                    any(any(m in a.name for m in FORBIDDEN_IMPORTS) for a in node.names):
                names = ", ".join(a.name for a in node.names)
                violations.append(f"{filename}:{node.lineno}: from {mod} import {names}")

    return violations


def online_modules() -> list[Path]:
    files = []
    for d in ONLINE_DIRS:
        files.extend(sorted((REPO / d).rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def test_online_modules_are_clean():
    files = online_modules()
    assert len(files) >= 6, f"only found {[str(f) for f in files]}"
    violations: list[str] = []
    for f in files:
        violations.extend(scan_source(f.read_text(encoding="utf-8"), str(f.relative_to(REPO))))
    assert not violations, (
        "LEAKAGE: online modules reference benchmark ground truth:\n  "
        + "\n  ".join(violations))


def test_online_modules_do_not_import_the_benchmark():
    """`hnav/core/` and `hnav/adapters/` must not import benchmark code.

    (Adapters may import ``hnav.labeling.conflict_analysis`` and
    ``conflict_index`` — those parse fact text and index serial numbers, and
    contain no evaluator output.)
    """
    benchmark_roots = ("methods", "utils.eval", "cl_bench_memory", "agent",
                       "infer_context_memory", "eval_other_utils")
    bad = []
    for f in online_modules():
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                if any(m == b or m.startswith(b + ".") for b in benchmark_roots):
                    bad.append(f"{f.relative_to(REPO)}:{node.lineno}: imports {m}")
    assert not bad, "core/adapters must stay benchmark-free:\n  " + "\n  ".join(bad)


# ── the scanner must be able to fail ─────────────────────────────────────────
VIOLATION_IDENTIFIER = '''
"""A module whose docstring may discuss gold answers freely."""
def decide(item):
    gold = item["answers"][0]          # <- fatal: evaluator ground truth online
    return gold
'''

VIOLATION_SUBSCRIPT = '''
def probes(item):
    return item["questions"]           # <- fatal: benchmark questions as probes
'''

VIOLATION_IMPORT = '''
from hnav.labeling.counterfactual import label_write
'''

CLEAN = '''
"""Docstrings may mention gold answers, rubrics and questions."""
def decide(candidate, store):
    # comments may too
    return candidate.text[:200]
'''


@pytest.mark.parametrize("src,name", [
    (VIOLATION_IDENTIFIER, "identifier+subscript"),
    (VIOLATION_SUBSCRIPT, "questions subscript"),
    (VIOLATION_IMPORT, "counterfactual import"),
])
def test_scanner_catches_deliberate_violations(src, name):
    found = scan_source(src, name)
    assert found, f"scanner failed to catch the deliberate {name} violation"


def test_scanner_does_not_flag_prose():
    assert scan_source(CLEAN, "clean") == []
