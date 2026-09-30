# Gold conflict dataset — summary

- records: 54,569 (audited pairs only); eval set: 5,340 (balanced 1:1 per subset, cosine-matched, seed 20260824)
- gold positives: 2,670 (core 2,388, of which strict 1,966; update-only fork 282) — rejected 12, discovered-unverified 105 (quarantined)

| subset | split | core | fork | rejected | discovered | negatives | eval pos | eval neg | exact-bin | pos cos p50 | neg cos p50 | cos-only AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sh_6k | calibration | 141 | 19 | 0 | 3 | 368 | 160 | 160 | 26 | 0.9637 | 0.8592 | 0.9598 |
| sh_32k | calibration | 740 | 89 | 6 | 27 | 9,364 | 829 | 829 | 238 | 0.9638 | 0.8991 | 0.9109 |
| sh_64k | confirmatory | 1507 | 174 | 6 | 75 | 42,050 | 1681 | 1681 | 537 | 0.9639 | 0.9094 | 0.893 |

Conventions and caveats: see `gold_conflict_dataset.summary.json`.
Dual labels: `gold_update` (benchmark update convention, default) and `gold_strict` (logical incompatibility). The `update_only_fork` tier is gold under update semantics only and carries `disputed_by_judge`.
