# hnav_abtt_noparser — ABTT cosine with NO identity screen (the danger arm)

Identical to `hnav_abtt` except `pair_filter` is removed entirely: the screen
is ABTT-space cosine alone, then bidirectional NLI. **Preregistered
expectation (2026-08-27): this arm may degrade.** The Faz A audit measured the
NLI cross-encoder rubber-stamping same-template/different-subject pairs as
bidirectional contradiction (0.999 both ways), and the T11 calibration showed
~86% spurious verifications with the identity screen off. This arm exists to
measure what that does to end-to-end accuracy when the *strongest* cosine
screen (ABTT, pair-level AUROC 0.999) is the only thing in front of the NLI —
i.e. whether "cosine without the parser" is viable at all. A negative result
is the reportable answer, not a failure of the run.

Reuses the committed `_abtt` prepasses unchanged (cos_loose 0.30 covers a pure
superset of what any cell here admits). `cos_pair` is re-selected on the
calibration split from the preregistered grid {0.30, 0.45, 0.60}; the
`n_suppressed_harmful == 0` selection requirement stays hard — if no cell
reaches it, that IS the result and no operating point is frozen.

Before the first run (once, on the box; prepasses assumed present from the
ABTT campaign, rebuild per ../hnav_abtt/README.md if not):

    python hnav/stage1/detector_gap.py --select --subsets sh_6k sh_32k \
           --geometry-space abtt --pair-screen none \
           --whitening-artifact stage0_results/abtt/abtt_whitening_D128.json \
           --cos-grid 0.30 0.45 0.60
    # then pin the new operating point: put its sha256 into pipeline.json
    # (operating_point_sha256) in the same commit that adds the file.

Run:

    python pipelines/hnav_abtt_noparser/run.py --llm-model <served-name> [--llm-base-url URL] [--dry-run]

Everything else: ../README.md.
