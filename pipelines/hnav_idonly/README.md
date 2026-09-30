# hnav_idonly — symbolic identity alone, semantic gate waived (E2E-4)

The question: after E2E-3 showed geometry cannot replace the parser, *what is
actually still limiting the parser arm?* The answer, measured: not identity —
the **bidirectional-NLI contradiction gate**.

Of the parser arm's 29 conflicted failures on sh_64k, 22 are retrieval misses
(gold fact never on the page) and 5 are parametric-knowledge failures (the
page is already correct and the model answers from its weights anyway). Only
**2** are detection misses, and both are same-key pairs the NLI blocks:

| q | pair | bidirectional contradiction |
| --- | --- | --- |
| 23 | `racing video game … sport of racing` → `… Australian rules football` | **0.854** (threshold 0.90) |
| 98 | `The Kinks was founded in the city of London` → `… of England` | **0.0002** |

q98 is the instructive one: the NLI is *right* that London and England are
compatible, and *wrong* for this store, whose convention is that a later
serial supersedes the same key. The semantic gate vetoes the store's own
recency rule.

**This arm removes that gate and nothing else.** Same shipped `same_key`
screen, same cosine/r_min/ambiguity axes, `nli_contradiction = 0.0`.

**Safety is structural, not empirical.** `suppress` keeps each group's
highest serial; `same_key` is an equivalence on (relation, subject), so a
group never spans two keys; therefore a key's newest member is never dropped
and `classify_drops` can never fire. The selection run confirmed it: harm 0
in **every** cell of the grid, including the fully-open one.

**Frozen operating point** (calibration sh_6k + sh_32k only): `cos_pair 0.90,
nli_contradiction 0.0, r_min 0.44, ambiguity none` — pair precision **1.000**,
pool recall **0.9952** (shipped parser arm 0.9784), conflicted-question
recall **137/139**, 0 harmful of 2,719 suppressions.

Preregistration and gates: `stage0_results/stage1/IDONLY_PREREG.md`
(GI1 passed; GI2 = sh_64k overall > 64/100, one shot, paired vs the committed
parser records).

**Scope caveat, stated up front:** waiving semantic verification is safe *for
a single-valued-relation store* like this benchmark. Where a relation is
genuinely multi-valued, the NLI gate is doing real work and this result does
not transfer.

## Run against a new answering model

```bash
python pipelines/hnav_idonly/run.py --llm-model <served-name> --dry-run   # always first
python pipelines/hnav_idonly/run.py --llm-model <served-name> --llm-base-url http://localhost:8003/v1
```

Uses the base `stage1_prepass_<subset>_benchmarkpage.json` prepasses — the
same ones the parser arm uses, no rebuild, no extra prepass.
