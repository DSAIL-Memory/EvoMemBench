# E2E-4 — `hnav_idonly`: is the NLI semantic gate the binding constraint?

Preregistration, written 2026-08-30 BEFORE any selection cell for this arm was
scored. Motivated by the E2E-3 post-hoc diagnosis (below), which used an
already-spent shot for *attribution only*; no threshold here is fitted to
sh_64k.

## Why this arm exists (the diagnosis that produced it)

The complementarity analysis mandated by the user was run first and came back
NULL, so the preregistered hybrid was **not** built (see
`E2E4_COMPLEMENTARITY.md`): geo's suppressions are 99% a subset of the
parser's (527 shared / 208 parser-only / **5** geo-only), the answer-level
oracle union is 65/100 vs the parser's 64, and that +1 (q30) is answering
noise — neither arm cut q30's target serial. A geo-primary/parser-fallback
hybrid would therefore converge to the parser arm.

The same analysis localized the parser arm's own residual. Of its 29
conflicted failures on sh_64k:

- **22** are retrieval misses — the gold-valued fact never reaches the
  retrieved page, so no page edit can fix them;
- **5** are parametric-knowledge failures — the page is already correct after
  the edit (q6, q30, q73, q88) or holds only the gold fact (q90), and the
  model still answers from its own weights;
- **2** are genuine detection misses, **both blocked by the bidirectional-NLI
  gate, not by identity**: q23 (`racing video game … sport of racing` →
  `… Australian rules football`, min contradiction **0.854**, just under the
  0.90 threshold) and q98 (`The Kinks was founded in the city of London` →
  `… of England`, min contradiction **0.0002** — the NLI judges the two
  compatible and vetoes the store's own supersession convention).

Calibration confirms the mechanism is systematic, not anecdotal: **2.2 % of
true supersession pairs (13/536 at cos ≥ 0.90) are blocked by NLI on
sh_6k + sh_32k** (11 on sh_64k), which is exactly the parser arm's missing
pool recall (0.978).

## The claim under test

**H1.** Once symbolic identity is certified, the NLI contradiction gate is
redundant under this benchmark's supersession convention, and removing it
raises detection recall to ~1.0 at unchanged (zero) harm.

**H2 (safety, structural).** Any arm whose every verified pair satisfies
`same_key` has *n_suppressed_harmful = 0 by construction*, at any NLI
threshold: `suppress_ids` keeps each group's highest serial; `same_key` is an
equivalence on (relation, subject) so a group cannot span two keys; therefore
the key's newest corpus member is never dropped and
`survivors[-1] == rows[-1]` in `classify_drops`; and `same_key` requires both
facts to parse, so the unparsed-drop channel is empty. This is asserted here
as a *prediction*: the selection run must observe harm = 0 in every cell.

## The arm

`--pair-screen idonly` — the shipped `MABAdapter.same_key_pair` screen pinned
on, as its own arm with its own operating point
(`stage0_results/stage1/idonly_operating_point.json`). Nothing about the
committed parser arm changes; `stage1_operating_point.json` is untouched.

- Grid: shipped cosine axis {0.90, 0.92, 0.94} × r_min {frozen, loose, off} ×
  ambiguity {all, any, none} × **NLI axis {0.0, 0.5, 0.90}** (0.0 = gate off;
  0.90 = the shipped value, present so the arm can decline the relaxation).
- Selection rule: unchanged — `n_suppressed_harmful == 0` hard, maximize
  `pair_recall_pool`, shipped tie-breaks (higher cos_pair, **higher
  nli_contradiction**, tighter r_min, stricter ambiguity). The tie-break
  keeps the stricter gate whenever relaxing buys nothing, so the arm can only
  move if the relaxation actually adds recall.
- Fit split: sh_6k + sh_32k only. sh_64k untouched until one wet shot.

## Gates

- **GI1 (proceed to wet run).** The selected cell must have
  `n_suppressed_harmful == 0` **and** pool recall strictly greater than the
  shipped parser arm's 0.9784. Otherwise: null result, no LLM spent.
- **GI2 (primary endpoint, sh_64k, one shot).** Overall accuracy **> 64/100**
  (the committed hnav_raw/hnav_abtt result), paired per-question McNemar
  against the committed parser records. Reported regardless of outcome.
- **GI3 (mechanism check).** Report how many of the two NLI-blocked questions
  {q23, q98} are recovered, and whether any question the parser arm answered
  correctly is lost (a degradation list, named).
- **GI4 (do-no-harm).** Standard void conditions and harm classes unchanged;
  the unique stratum is the protected stratum. Detection-level harm must be 0
  on sh_64k as well (H2's out-of-sample test).
- One shot; a void is reported, not re-rolled. sh_262k excluded.

## What each outcome means

- **GI2 passes** → the binding constraint on H-Nav's detector was the
  semantic gate, not identity; the ceiling analysis (22 retrieval + 5
  parametric) says the remaining headroom is ~1 question, so the arm should
  land at 65–66 and *saturate the mechanism*.
- **GI2 fails while detection recall rises** → extra suppressions cost as
  much as they buy; the detector was already at its accuracy ceiling and the
  residual is entirely retrieval + parametric. Equally publishable, and it
  closes the question.

## Honest scope

Waiving the semantic check is safe *under this benchmark's convention* —
single-valued relations where a later serial supersedes the same key. In a
store with genuinely multi-valued relations the NLI gate would be doing real
work, and this result must not be read as "semantic verification is
unnecessary in general". The claim is exactly: *on a single-valued-relation
store, symbolic identity plus recency is sufficient, and the semantic gate is
a recall tax.*
