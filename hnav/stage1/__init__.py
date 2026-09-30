"""Stage-1 read-path offline tooling.  [T11 rerank; T12 probe; T13 detector gap]

``calibrate_read_policy``     the chunk-rerank grid (T11) — the search that
                              returned the directional null.
``stale_suppression_probe``   the ORACLE ceiling (T12): what a perfect detector
                              would buy, per mechanism.
``detector_gap``              the same design with DETECTOR-driven arms (T13),
                              and the ratio between the two.


Like ``hnav/stage0/``, everything here is OFFLINE and may read benchmark
questions and answers; nothing under ``hnav/core/`` or ``hnav/adapters/`` may
import from it (the leakage audit scans those trees). Calibration is bound to
the sh_6k + sh_32k split — the confirmatory subsets are refused by the
scripts themselves.
"""
