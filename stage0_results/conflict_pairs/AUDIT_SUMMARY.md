# LLM semantic audit — summary

- model: `openai/gpt-5-mini`, results: 54,569, total cost $19.49
- disagreements (tagged, judge says not conflict): 294
- discoveries (untagged, judge says update conflict): 105

## sh_32k
- **tagged** n=835: update_conflict=740 strict=616 errors=0
  - reasons: direct_replacement 599, multi_valued_relation 160, relation_paraphrase 27, subsumption 15, different_referent 15, different_relation 6
- **untagged** n=9,391: update_conflict=27 strict=26 errors=0
  - reasons: different_referent 8,810, different_relation 403, relation_paraphrase 130, context_mismatch 18, direct_replacement 10, multi_valued_relation 6

## sh_64k
- **tagged** n=1,687: update_conflict=1,507 strict=1,231 errors=0
  - reasons: direct_replacement 1,204, multi_valued_relation 352, relation_paraphrase 59, subsumption 32, different_referent 14, other 12
- **untagged** n=42,125: update_conflict=75 strict=70 errors=0
  - reasons: different_referent 40,517, different_relation 1,142, relation_paraphrase 358, context_mismatch 37, direct_replacement 22, multi_valued_relation 17

## sh_6k
- **tagged** n=160: update_conflict=141 strict=119 errors=0
  - reasons: direct_replacement 115, multi_valued_relation 32, other 5, subsumption 3, relation_paraphrase 3, different_relation 1
- **untagged** n=371: update_conflict=3 strict=2 errors=0
  - reasons: different_referent 311, different_relation 47, relation_paraphrase 11, direct_replacement 1, alias_equivalent 1

## Coverage and protocol

- Candidate universe: 87,102 pairs (all cos >= 0.80, exact campaign embeddings).
- Audited: 54,569 (62.6%) under a $20 hard cap; total spend $19.49, 0 errors,
  0 parse failures. Budget stop fired at $19.35 (run) after 54,269 calls.
- Priority coverage at the stop (deterministic, seed 20260824):
  - slice 1 — parser-tagged conflict pairs: 2,682/2,682 (100%)
  - slice 2 — unparsed-fact / same-key untagged pairs: 77/77 (100%)
  - slice 3 — structural channels (same-subject cross-template, alias
    candidates): 30,695/30,695 (100%)
  - slice 4 — seeded random bulk sample: 4,000/4,000 (100%)
  - slice 5 — shuffled bulk tail: 17,115/49,648 (34.5%, unbiased because
    shuffled before truncation)
- Parser precision under the update-conflict convention
  (same_referent AND same_relation AND context_overlap AND values_incompatible):
  sh_6k 141/160 (88.1%), sh_32k 740/835 (88.6%), sh_64k 1,507/1,687 (89.3%).
  Disagreements are dominated by multi_valued_relation and subsumption — the
  documented definitional fork, re-derivable from the recorded per-pair flags.
- Judge: openai/gpt-5-mini via OpenRouter, reasoning effort minimal
  (0 reasoning tokens observed), strict json_schema output, single call per
  pair with seeded A/B order randomization. System prompt fixed verbatim
  (sha256 a3632d8b7b29e97c103abcf881be192f17c5f1f4de48c62b44101ec23c342a30).
