# pipelines/ — the two frozen H-Nav configurations, runnable on new answering models

Two pipelines, identical everywhere except geometry:

| folder | geometry before the cosine screen | operating point (frozen) | committed result, conflicted sh_64k |
|---|---|---|---|
| `hnav_raw/` | none (raw cosine) | `cos_pair 0.90, r_min 0.44` | 17/66 → **37/66** vs native |
| `hnav_abtt/` | ABTT whitening, D=128, fit on sh_6k+sh_32k | `cos_pair 0.30, r_min 0.954` | 17/66 → **37/66** (exact null vs raw) |

Both then run the same detector: cosine pair screen → regex `pair_filter`
(subject+relation identity) → bidirectional NLI → suppression, through
`hnav/stage1/detector_gap.py`. **No detection logic lives in this folder** —
each pipeline is a frozen configuration plus a driver
(`_shared/runner.py`); the science stays in `hnav/`, covered by its tests.

## What is frozen and what varies

**Frozen, verified before every run** (sha256 / fingerprint pinned in each
`pipeline.json`; the runner refuses on any mismatch):
- the embedding model — `Qwen/Qwen3-Embedding-4B`, fp32, L8192. The
  thresholds and the whitening are *coordinates in that encoder's space*; G1
  measured that they do not transfer. A new embedder means a new calibration
  campaign, not this runner.
- both operating-point artifacts and the ABTT whitening artifact.
- the per-subset prepasses (`hnav/_out/stage1_prepass_<subset>_benchmarkpage
  [_abtt].json`) — chunking, candidate pairs, NLI table. These are
  **LLM-independent**: built once per subset, reused for every future model.

**What varies: the answering LLM only**, passed per run:

```bash
# on the GPU box, with the model served OpenAI-compatible (e.g. vLLM :8003)
python pipelines/hnav_raw/run.py  --llm-model <served-name> --llm-base-url http://localhost:8003/v1
python pipelines/hnav_abtt/run.py --llm-model <served-name> --llm-base-url http://localhost:8003/v1

# always first: budget + guard pre-flight, sends nothing
python pipelines/hnav_raw/run.py --llm-model <name> --dry-run
```

Each run writes `results/<model-tag>/`: the raw `detector_gap_*.json` per
subset, a `run_manifest.json` (model, endpoint, git head, pinned hashes), the
sh_64k dry-run log, and `REPORT.md` with the stratified table. Results you
want to keep must be committed deliberately (repo convention).

## Subsets, and how to present them in the thesis

All three subsets run by default — sh_6k, sh_32k, sh_64k — reported
**separately, never pooled** (store sizes span 455 → 4,580 facts; a pooled
number would average incomparable populations). Per subset, report the
committed strata (`question_strata.json`): **conflicted** is the primary
endpoint — it is the population the mechanism exists for — and **unique** is
the do-no-harm check. Give native vs H-Nav counts, the paired net, and the
McNemar exact p; `REPORT.md` emits exactly this table.

Label the roles: sh_6k and sh_32k are the **calibration split** (the
operating point was selected on their fact geometry — detection quality only,
no LLM, no gold answers in the objective), sh_64k is **held-out**. For a *new
answering model* all three are honest accuracy measurements — the selection
objective never saw any LLM's behaviour — but the labels should still appear,
and sh_64k remains the headline.

**RULE (user decision, 2026-08-24): every subsequent model runs this same
protocol — sh_6k + sh_32k + sh_64k, stratified, one shot each. sh_262k is
excluded permanently**, for two reasons: its contexts would most likely
exceed the context window of the smaller answering models this folder exists
to test, so including it would silently turn a memory comparison into a
context-length comparison; and it was never part of the registered campaign
(`detector_gap` refuses it, no prepass, no NLI table). The runner's
`ALLOWED_SUBSETS` enforces this — do not widen it per model. The thesis
states sh_262k was not measured rather than extrapolating to it.

One shot per model per subset. The runner refuses to overwrite an existing
results folder; a void run is reported as void, not re-rolled.

## Cost per new model

5 arms × 100 questions × 3 subsets = **1,500 LLM calls per pipeline**
(3,000 for both). The duplicated `native` arms are not waste: raw-vs-ABTT ran
against the same substrate twice and agreeing native arms are the proof the
contrast carries only the geometry (the campaign's native arms agreed
500/500).

## If something refuses

The runner's refusal messages name the fix. The common ones:
- **prepass missing** — build once per subset:
  `python hnav/stage1/confirmatory_prepass.py --subset <s>` (add
  `--geometry-space abtt --whitening-artifact stage0_results/abtt/abtt_whitening_D128.json`
  for the ABTT pipeline). Needs the embedding cache and the NLI model; no LLM.
- **operating point sha mismatch** — someone re-froze thresholds. That is a
  method change; re-pin the hash in `pipeline.json` only in the same commit
  that justifies it.
- **embed model mismatch** — `HNAV_EMBED_MODEL` is set to something other
  than the frozen encoder. Unset it; this runner is not the tool for a new
  embedder.
