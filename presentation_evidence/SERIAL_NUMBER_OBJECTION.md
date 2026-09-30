# Does the model just read the serial numbers?

**The objection.** If the page shows `91.` and `259.` and the prompt says "larger
serial = newer", picking the newer fact is trivial. Is this a reasoning task at
all — and what is left for H-Nav to do?

**Short answer: the objection is correct on its premises and wrong in its
conclusion.** Precedence *is* trivially available. The model still gets it wrong
~95% of the time, and H-Nav does not solve precedence either — it solves
*grouping*, then executes the trivial rule outside the model.

---

### 1. Are serials in the prompt? **Yes, verbatim.**

`hnav/stage1/stale_suppression_probe.py:163-171` — `render_context`:

```python
body = "\n".join(f"{s}. {t}" for s, t in facts)
```

Each fact is rendered as `"<serial>. <text>"`. The docstring notes this
reproduces the dataset's `context` field **exactly** (pinned by
`test_stale_suppression_probe.py`), so the `native` arm is the untouched
benchmark input. Chunks are then wrapped as `Memory i:` blocks by
`build_user_prompt` (`hnav/stage1/calibrate_read_policy.py:477-480`). Serials
survive every edit: `suppress()` (line 174) keeps every surviving fact's original
serial, and `test_read_policy_facts.py` enforces no renumbering.

### 2. Is the rule stated? **Yes, twice, in the system prompt.**

`In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/utils/templates.py:40-42`
(`long_context_agent`, `rag_agent`, `agentic_memory_agent` — all three):

> "Each fact in the knowledge pool is provided with a serial number at the
> beginning, **and the newer fact has larger serial number.** You need to solve
> the conflicts of facts in the knowledge pool by **finding the newest fact with
> larger serial number.**"

So the task is fully specified: both operands and the rule are on the page.

### 3. Then why does the model fail?

It does — overwhelmingly. `presentation_evidence/data/item02_error_classes.csv`,
eight independent sh_6k runs:

| | conflicted correct | **errors returning the stale value of the correct key** |
|---|---|---|
| across 8 runs | 0–5 of 74 | **572 of ~575** |

The failure is **not** retrieval and **not** confusion. The model locates the
right `(relation, subject)` slot, has both members and both serials in front of
it, and returns the **smaller-serial** one. This is a *compliance* failure, not
an information failure — the instruction is present and ignored. That is itself
a reportable result (`HNAV_FINAL_REPORT.md` §9, finding 2: explicit precedence
instructions are ~95% ineffective at this scale).

So: yes, the task reduces to a comparison — and the model does not perform it.

### 4. What H-Nav adds — **not** precedence

H-Nav uses **the same trivial rule**. `mab_adapter.py:704-709`:

```python
@staticmethod
def latest_key(rec: MemoryRecord) -> int:
    """LATEST is the benchmark's own rule: the highest fact serial..."""
    return rec.version
```

and `read_gate.py:639-657` (`_finalize`) simply takes `max(vals)`. **H-Nav does
not out-reason the model about which fact is newer.** Claiming otherwise would be
wrong.

The work is the *other* half of the problem, which serials cannot help with:

> A serial number gives a **total order** over facts. It says nothing about
> **which facts are in contention for the same slot.**

On sh_64k, 4,580 facts carry 4,580 distinct serials; nothing in `259 > 91`
indicates that #91 and #259 are about *Nobuhiro Watsuki* while #92 is about
someone else. Identifying the contending group is the entire detector pipeline
(§2–3 of `DETECTOR_MECHANICS.md`): cosine screen → subject-identity screen →
bidirectional NLI. Only once a group exists is `max(serial)` meaningful.

H-Nav's contribution is therefore: **group the contenders, apply the stated rule
outside the model, and delete the loser from the page** — converting an
instruction the model demonstrably ignores into an edit it cannot ignore.

### 5. How the claim must be framed

This **narrows** the thesis claim; it does not void it. Three constraints:

**(a) Do not claim improved revision *reasoning*.** The precedence rule is given
and H-Nav merely executes it. The defensible claim is about **context
engineering**: removing the distractor, not resolving the conflict more cleverly.

**(b) The arms already separate these two hypotheses.** If the failure were about
serial ordering or position, reordering would fix it. It does not
(`stale_suppression_probe_sh6k.json`, conflicted stratum):

| arm | conflicted accuracy |
|---|---|
| `native` | 4/74 (5.4%) |
| `oracle_recency` — newest fact **moved to the end**, both kept | 20/74 (27.0%) |
| `oracle_suppress` — stale fact **deleted** | **66/74 (89.2%)** |

Presence of the stale value, not its position, is what breaks the answer. That
contrast is the finding.

**(c) State the external-validity limit.** Precedence is trivial *here* because
the benchmark is synthetic and hands out serials. On a real substrate the
recency key must come from timestamps or version metadata — which is why
`latest_key` is an adapter-supplied callable and the core falls back to
`MemoryRecord.version`, returning `latest_id = None` (suppressing nothing) when
the key is missing or tied. **The detection half transfers; the precedence half
must be supplied by the host system.**

---

**One-sentence answer.** Yes, the model sees the serials and is told the rule —
and returns the stale value in 572 of ~575 conflicted errors anyway; H-Nav does
not improve that comparison, it identifies *which facts are competing* (the part
serials cannot express) and then enforces the rule by editing the page, so the
claim is about making an ignored instruction unignorable, not about better
reasoning over revisions.
