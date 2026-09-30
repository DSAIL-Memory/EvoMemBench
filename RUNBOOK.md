# RUNBOOK — ozonderlab2

The copy-paste path for running H-Nav Stage 0 on the 2× RTX 4090 box. Written
because there is no coding agent on that machine: every script you need is
already in the repo, and this file is the order to run them in.

**Authority order.** `HNAV_AGENT_BRIEF.md` is the spec. This file is the
operational path for *this box* and supersedes `hnav/NEXT_STEPS.md` where they
differ (two of NEXT_STEPS' own "corrections" turned out to be wrong; see §6).

**The one rule:** stop at every **GATE** and report. Do not continue on your own
judgment — that is what the gates are for.

---

## 0. Lab rules — these override anything else in this repo

**There is no workload manager (no Slurm).** Nothing stops two jobs landing on
the same card, so the discipline is manual:

1. **Always `nvidia-smi` before running anything.** The launcher scripts do this
   for you and *refuse to start* if the target GPU already has a compute process
   (`hnav/deploy/gpu_guard.sh`). Override only deliberately, with
   `HNAV_FORCE_GPU=1`.
2. **Monitor in a second terminal:** `watch -n 1 nvidia-smi`
3. **Always target an explicit device.** `HNAV_EMBED_DEVICE` for H-Nav,
   `CUDA_VISIBLE_DEVICES` for the benchmark. Never let something pick "auto".

**Conda: do not install your own.** A system-wide install lives at
`/opt/anaconda3`. `conda init` is deliberately *not* run at login, so `conda` is
not on `PATH` until you run:

```bash
source activate_conda
conda activate hnav
```

Every H-Nav launcher does this internally via `hnav/deploy/_activate.sh`, so
`nohup`/`tmux` jobs work without an interactive shell. Setup **creates an env
inside the system conda** — it never installs a conda of its own.

### Box facts this runbook assumes

| | |
|---|---|
| GPU0 | **busy** — the vLLM server, PID 52520, ~15.6 GB. Never allocate on it. |
| GPU1 | idle, ~24.5 GB free → embedder goes here, `float32` fits (needs ~17 GB) |
| Work dir | `/mnt/nvmes/nvme1/egekutlu` |
| HF cache | `/mnt/nvmes/nvme1/egekutlu/hf_cache` — already exists, reused |

Lab machines have device numbers `0, 1, 2`; **this** box showed two. Re-check
with `nvidia-smi` — GPU1 being free is an assumption with a shelf life, and the
guard will stop you if it has expired.

---

## 1. Get the code

```bash
cd /mnt/nvmes/nvme1/egekutlu
git clone https://github.com/kutluege/EvoMemBench.git
cd EvoMemBench
git checkout claude/evomembench-hnav-analysis-nfwl9z
git log --oneline -1        # should be at or after the "M0 harness" commit
```

---

## 2. Setup — once

```bash
nvidia-smi                              # rule 1 — look before you touch
bash hnav/deploy/setup_ozonderlab2.sh
```

Creates the conda env `hnav` (python 3.11) **inside the system-wide
`/opt/anaconda3`**, installs torch cu124 + deps, pre-caches nltk/punkt and
tiktoken, pulls the ~8 GB embedder into `HF_HOME` on the nvme, and writes both
`.env` files. It allocates no GPU memory. Idempotent.

Then, in any new shell:

```bash
source activate_conda
conda activate hnav
export HF_HOME=/mnt/nvmes/nvme1/egekutlu/hf_cache
```

**Edit `.env`** (repo root) and set `HNAV_LLM_MODEL` to whatever `:8000` actually
serves. Then:

```bash
python hnav/deploy/check_env.py     # ← GATE. Must exit 0.
pytest hnav/tests/ -q               # ← must be 153 passed, WITH torch installed
```

`check_env.py` prints the model id `:8000` reports and warns if it disagrees with
your `.env`. Fix the `.env`, not the server.

If it says GPU1 lacks room for `float32`: set `HNAV_EMBED_DTYPE=float16` and
**pin it for every later task**, and record the change in the report. Dtype drift
changes cosines and moves every threshold.

---

## 2b. UNATTENDED MODE — run T1→T8 overnight with one command

There is no coding agent on this box, so the whole Stage-0 campaign is scripted:

```bash
tmux new -s stage0
bash hnav/deploy/run_stage0.sh
# detach: Ctrl-b d        reattach later: tmux attach -t stage0
```

(or `nohup bash hnav/deploy/run_stage0.sh > hnav/_out/pipeline/console.log 2>&1 &`)

What it does, in order: preflight (check_env + 153 tests, auto-syncs
`HNAV_LLM_MODEL` to what `:8000` serves) → T1 smoke → **T1 full (gate S3)** →
T2 → starts the `:8001` embed server out of your existing `vllm_0.9.1` env on
GPU1 → **M0 (gate S1)** → **T4 both arms + diff (gate S2)** → kills the server,
waits for VRAM to drain → M2 (with the fallback-chunker tripwire) → M3 (the big
LLM item) → M4 → T8 report.

**Gate semantics:** a gate that *passes* lets the pipeline continue — that is
what a human would order. A gate that *fires* **halts everything downstream
instantly**; the status file names the gate and a human decides. The T8 GO/NO_GO
gate is never evaluated by the pipeline at all — it only writes the report and
stops, as the brief requires.

**If M0/T4's benchmark dependencies are missing** (langchain etc. — they need
`pip install -r In-Episode-Knowledge/INEP-KNOW/requirements.txt`), those two
stages are **SKIPped with the reason recorded** and the pipeline continues:
M2–M4 use H-Nav's own offline replica and do not depend on them, and `report.py`
renders missing measurements as NOT RUN by design.

**Monitoring and control:**

```bash
tail -f hnav/_out/pipeline/*.log          # everything logs per-stage
cat hnav/_out/pipeline/SUMMARY.md         # state of every stage
watch -n 1 nvidia-smi                     # rule 2
```

**Resume:** interrupted for any reason → run the same command again. Completed
stages (PASS/SKIP) are skipped; a FAILed stage re-runs. Force one stage over:
`bash hnav/deploy/run_stage0.sh --redo m2`. T1's embedding cache
(`hnav/_cache/emb/`) survives interruption, so a resumed T1 is much cheaper.

When it finishes: `STAGE0_REPORT.md` is in the repo root, per-stage outputs in
`hnav/_out/`. Both are gitignored except the report — **commit what you want to
keep**, then bring the report and `SUMMARY.md` back for the human gate decision.

---

## 3. T0 — confirm the data is intact

```bash
python hnav/labeling/conflict_analysis.py
python hnav/labeling/gold_rule.py
```

Expected, and already verified to reproduce exactly:

- `sh_262k`: **11,037 keys / 7,197 conflicted (65.2%)**, all groups size 2
- `sh_262k`: **77%** of questions on a conflicted key, **73/77** gold-is-LATEST

Different numbers mean the data file changed. Stop and report.

---

## 4. T1 — the kill switch  ← **GATE**

Nothing after this matters until it passes. T1 needs only the embedder on GPU1;
no LLM, no embedding server.

```bash
# 2-minute smoke FIRST. This is the first time HFEmbedder is ever constructed —
# no torch existed on the machine the code was written on, so a model load has
# never once been exercised. Cheapest possible test of the riskiest path.
python hnav/stage0/m1_geometry_calibration.py --subsets sh_6k --max-pairs 50

# then the full run, under nohup
bash hnav/deploy/run_t1.sh
tail -f hnav/_out/m1.log
```

- **exit 0** → the near-duplicate premise holds. Continue.
- **exit 2** → **S3 fired**: median `whole_blob_sim` < 0.70. Conflict pairs are
  not near-duplicates, the geometry half of H-Nav is dead on this benchmark.
  **Stop. Report. Do not run anything below.** A human re-scopes first.

Record the median `whole_blob_sim` per subset either way.

---

## 5. T2 — grouping ablation (decides attribution)

```bash
python hnav/stage0/m1b_grouping_ablation.py
```

Reuses T1's embedding cache. **If it starts embedding from scratch, stop** — the
cache key changed, meaning the model or dtype changed, and your calibration is no
longer comparable to T1's.

Record the F1 at the equal-coverage point. This decides whether any downstream
gain is attributable to H-Nav's geometry or merely to the benchmark's regex-able
templates and serial numbers. **Report it prominently whatever it says.**

---

## 6. The embedding server — needed from here on

> **`NEXT_STEPS.md` §6 "Correction 2" is wrong and has been struck through.** It
> claims `TextRetriever` loads the embedder in-process. It does not.
> `Qwen3Embedding4BEmbeddings` exists at `embedding_retriever.py:51` but
> `TextRetriever.__init__` never selects it — line 176 routes
> `"Qwen/Qwen3-Embedding-4B"` to LangChain `OpenAIEmbeddings` against
> `OPENAI_BASE_URL`. **You need `:8001`.**

```bash
tmux new -s embed
bash hnav/deploy/serve_embeddings.sh     # Qwen3-Embedding-4B on :8001, GPU1
```

### The two-endpoint split — read this before running anything below

Embeddings and the LLM both key off the name `OPENAI_BASE_URL`, but they read it
from **different sources**, which is exactly what lets them point at different
ports:

| | reads from | must be |
|---|---|---|
| embeddings | `dotenv.dotenv_values()` → **`MemoryAgentBench/.env`** | `:8001` |
| LLM | bare `OpenAI()` → **`os.environ` only** | `:8000` |

`setup_ozonderlab2.sh` already wrote `MemoryAgentBench/.env` with `:8001`. In your
shell you export `:8000`. **Do not export `:8001`** — that sends the LLM calls to
the embedding server.

---

## 7. M0 — live-index fidelity

Newly written (`hnav/stage0/m0_live_fidelity.py`); this was the one Stage-0
measurement with no script.

```bash
python hnav/stage0/m0_live_fidelity.py --subsets sh_6k --max-pairs 50   # smoke
python hnav/stage0/m0_live_fidelity.py                                  # full
```

Writes `hnav/_out/m0_replica_fidelity.json`. **Exit 2 = S1 fired** (top-k
agreement < 0.999): `rank_self`, `margin`, `dH_self`, `dH_neighbor` and `churn`
are invalid — report the maximum achievable fidelity and state which signals die.
Do not re-base them on a different retriever.

Until this file exists, `report.py` prints M0 as NOT RUN and treats those signals
as provisional. **That is correct behaviour — never hand-write the JSON.**

---

## 8. T4 — shadow neutrality  ← **GATE (S2)**

```bash
source activate_conda && conda activate hnav
nvidia-smi                             # rule 1
cd In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench

export CUDA_VISIBLE_DEVICES=1          # keep everything off GPU0
export OPENAI_BASE_URL=http://localhost:8000/v1   # LLM. NOT 8001.
export OPENAI_API_KEY=EMPTY

CFG=configs/agent_conf/RAG_Agents/local-qwen/Embedding_rag_local-qwen-qwen3_embedding_4b.yaml
DS=configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml

HNAV_MODE=off    python main.py --agent_config $CFG --dataset_config $DS
HNAV_MODE=shadow python main.py --agent_config $CFG --dataset_config $DS
# diff the two result JSONs, ignoring the additive "hnav" key
```

The agent config is newly written. Three things in it are load-bearing and are
documented in its own comments: `agent_name` must contain
`rag_qwen3_embedding_4b` (substring match at `agent.py:1710`), `model` must
**not** contain `deepseek` (or `agent.py:1730` routes you through the Volcengine
Ark batch API), and `temperature` must be `0`.

Set `model:` in that file to what `:8000` serves. Use a distinct `output_dir` per
arm so the runs cannot overwrite each other.

**Acceptance:** byte-identical model output, identical per-question
`substring_exact_match`, unchanged token counts. Repeat for `sh_32k`.
**Any difference is a bug, not noise** — at `temperature=0` with a deterministic
evaluator there is no legitimate source of variation. Hard stop until fixed.

---

## 9. T5 / T6 / T7 — the measurements

```bash
python hnav/stage0/m2_retrieval_calibration.py      # T5
python hnav/stage0/m3_headroom.py                   # T6 — real LLM calls, use tmux
python hnav/stage0/m4_marginal_diff_test.py         # T7 — calibration split only
```

- M2: **check `"fallback_chunker": false` in the output.** `true` means nltk or
  punkt is missing and the chunking is not the benchmark's — every read-side
  number would then be computed over the wrong units.
- M2: record the raw-entropy degeneracy verdict. Either answer is publishable.
- M3: the big compute item, order of a couple thousand LLM calls. Run it under
  tmux, **not** with `--no-llm`. `--max-counterfactuals` bounds it; whatever
  value you use goes in the report, because a bounded run is a sampled run.
- M4: record the key-clustered CI and the cross-validated ΔAUC too, not only the
  subset-clustered CI — the calibration split has only two subsets, so that
  bootstrap resamples from two clusters.

---

## 10. T8 — the report  ← **HARD STOP**

```bash
python hnav/stage0/report.py --strict
```

`--strict` exits non-zero and lists what is missing if any measurement is absent.
**Do not evaluate the gate on a partial report.** Every NO_GO must be tagged
**benchmark** (class too rare) / **detection** (signal doesn't predict it) /
**policy** (intervention doesn't repair it) — these must never be conflated.

Then **stop**. Do not implement `write_policy.py` or `read_policy.py`, do not run
live arms, do not set `HNAV_MODE=live`. A human evaluates the §4 gate in
`EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md` first.

---

## 11. Operational reminders

- Everything long-running goes in `tmux` or `nohup`. SSH will drop.
- **No scheduler.** `nvidia-smi` before every launch; `watch -n 1 nvidia-smi` in
  a second pane while things run. The launchers refuse a busy card — if you
  override with `HNAV_FORCE_GPU=1`, know whose process you are sharing with.
- **Never install conda.** Use `source activate_conda` + `conda activate hnav`.
  If `conda` is missing after that, ask the admin; do not work around it.
- **Do not delete `hnav/_cache/emb/`.** T1 pays for it once; everything else is
  then free. Do not copy it between machines — the key is
  `sha256(model|dtype||text)` and a dtype mismatch is silent.
- OOM on GPU1 → lower `HNAV_EMBED_BATCH` **before** touching dtype. Dtype is
  pinned once chosen.
- `hnav/_out/` is gitignored. Anything you want to keep, commit deliberately or
  copy off the box.
- Nothing is tuned on `sh_64k` / `sh_262k`. `m3_headroom.py` refuses to fit
  thresholds without a calibration subset; `m4` refuses a non-calibration split.
