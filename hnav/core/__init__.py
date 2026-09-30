"""H-Nav core: benchmark-agnostic types, signals and replicas.  [T3–T6]

Nothing in this package may import from a benchmark, and nothing here may read
benchmark ``questions``/``answers`` — see ``hnav/tests/test_leakage_audit.py``,
which enforces both by AST scan.

Submodules are deliberately NOT imported here: ``import hnav.core`` must stay
cheap and side-effect-free so that a stray import can never change a benchmark
number (hard rule 4).
"""
