"""Benchmark adapters.  [T4]

Adapters own **all** benchmark-specific knowledge — chunk parsing, serial
numbers, prompt formats, store layout — so that ``hnav/core/`` can stay free of
any benchmark import. They are also subject to the leakage audit: no adapter may
read ``questions``/answer keys or import the counterfactual module.
"""
