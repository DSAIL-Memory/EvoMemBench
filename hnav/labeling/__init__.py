"""Offline labeling and analysis.

This is one of the two packages (with ``hnav/stage0/``) where benchmark
``questions``/``answers`` and evaluator output are permitted. Nothing here may
be imported by ``hnav/core/`` or ``hnav/adapters/`` — except
``conflict_index``, which exposes two deliberately separate classes:
``WriteConflictIndex.latest_before`` (online, write path) and
``ReadConflictIndex.latest`` (online, read path only). ``counterfactual`` is
hard-barred from the online path by ``hnav/tests/test_leakage_audit.py``.
"""
