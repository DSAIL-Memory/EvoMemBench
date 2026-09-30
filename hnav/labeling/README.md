# Stage-0 measurement scripts

Reproducible scripts backing the numbers in `EVOMEMBENCH_HNAV_REPO_ANALYSIS.md`. Pure Python
(stdlib only) — no API keys, no GPU, no benchmark run required. Read from the repository's own
`Conflict_Resolution.json`.

```bash
python3 hnav/labeling/conflict_analysis.py   # template induction + conflict census (analysis §4.2)
python3 hnav/labeling/gold_rule.py           # gold == latest-serial + question headroom (§4.2)
python3 hnav/labeling/marginal_diff.py       # whole-blob vs marginal-diff separation (§9)
```

**Caveat on `marginal_diff.py`:** it uses a char-3gram cosine as a *lexical proxy* for whole-blob
embedding similarity, because no embedder was available in the analysis environment. It establishes
that conflict pairs are near-identical in surface form and that the changed span is disjoint. It is
**not** a substitute for real embeddings — Stage-0 measurement **M1** replaces it, and no threshold
may be set from the proxy. See the Stage-0 protocol §2/M1.

These become `hnav/labeling/fact_templates.py` + `conflict_index.py` in the implementation plan's
step 1; they are kept here in their as-measured form so the reported numbers stay auditable.
