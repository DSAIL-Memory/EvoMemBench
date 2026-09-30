# E2E-4 · Part 1 — Is symbolic and geometric identity complementary? (2026-08-30)

Mandated analysis before any hybrid arm was built. Inputs: the committed
per-question records of `hnav_raw` (`stage0_results/abtt/abtt_arm_A1_raw_sh64k.json`)
and `hnav_geo` (`pipelines/hnav_geo/results/Qwen_Qwen3-4B-Instruct-2507_2026-08-29/
detector_gap_sh_64k.json`), paired on question index; the calibration
prepasses; and the frozen artifacts of both arms.

**Answer: no. The two signals are not complementary — geometric identity is
very nearly a subset of symbolic identity, and the measured union buys one
question, which is noise.** The preregistered `hnav_geo_parser_fallback` arm
was therefore **not built**; a geo-primary/parser-fallback hybrid provably
converges to the parser arm. What the analysis did produce is a different,
better-supported arm (Part 2, `IDONLY_PREREG.md`).

## 1. Answer-level 2×2 (sh_64k, n = 100, paired)

| | geo correct | geo wrong | total |
| --- | ---: | ---: | ---: |
| **parser correct** | 55 | **9** | 64 |
| **parser wrong** | **1** | 35 | 36 |
| total | 56 | 44 | 100 |

Per stratum — conflicted (n = 66): both 28, parser-only 9, geo-only 1,
neither 28. Unique (n = 34): both 27, parser-only 0, geo-only 0, neither 7.

- parser-only: **{9, 11, 26, 35, 54, 69, 70, 91, 93}**
- geo-only: **{30}**

## 2. Oracle upper bound

Counting a question correct if **either** arm answers it:

| | oracle union | parser alone | geo alone | native |
| --- | ---: | ---: | ---: | ---: |
| overall | **65**/100 | 64 | 56 | 45 |
| conflicted | 38/66 | 37 | 29 | 17 |
| unique | 27/34 | 27 | 27 | 28 |

**Headroom above the parser arm: +1 question.** Even a perfect per-question
selector between the two detectors — which no deployable system could be —
gains one question. That is the entire complementarity budget.

## 3. The +1 is noise, not signal

On q30 (`Os Lusíadas was written in the language of …`, gold *english*)
**neither** arm suppressed the question's own target key: the pair
(2615 Portuguese, 2829 English) is cut by both arms' plans identically. The
two arms differ only in *unrelated* groups elsewhere on the page (parser 9
groups, geo 8), and the answering model returned `Portuguese` under one page
and `English` under the other. The gain is a page-composition side effect of
suppressions that have nothing to do with the queried fact — i.e. answering
noise, not a detection win. The honest reading of the oracle bound is
therefore **65 − 1 = 64: no complementarity at all.**

## 4. Detection-level containment — the structural reason

Across all 100 sh_64k questions, comparing the two arms' suppression plans
serial by serial:

| | count |
| --- | ---: |
| suppressed by both | 527 |
| parser only | 208 |
| **geo only** | **5** |

The 5 geo-only suppressions are one each on questions {19, 48, 58, 73, 83} —
none of them a question the parser fails. Geometric identity certifies
**0.9 %** of what the parser does not; the parser certifies 28 % of what
geometry does not.

**And those five are exactly the harmful ones.** The geo artifact's own
`void_conditions.4_no_harmful_suppression` failed with
`n_suppressed_harmful = 8`, and recomputing them from the corpus shows all
eight are *key erasures* on precisely {19, 48, 58, 73, 83}: both members of
`official language of Italy` (1035, 1075), the only member of
`type of music that The Game plays` (3286), and both members of
`outfielder · associated with the sport` (103, 978) are deleted from the
page, because a geometric group merged two different keys and suppression
kept only the merged group's newest serial. **The set of geometry-only
suppressions and the set of harmful suppressions are identical — 5 for 5.**
Geometry's entire unique contribution over symbolic identity was information
loss, so the answer-level complementarity of +1 is not merely noise: the
detection-level complementarity is *negative*. On the nine questions the parser alone answers, geo's
plans are strict subsets (34 suppressions vs the parser's 70, with **zero**
geo-only serials).

Why: the parser's blind spot is unparseable facts, and the pools are almost
fully parseable — 1 of 421 pooled facts on sh_6k, 0 of 1,250 on sh_32k, 1 of
1,621 on sh_64k. There is essentially no region where geometry can see an
identity the parser cannot, and the repo's own harm rule forbids suppressing
unparsed facts anyway (`classify_drops`: "unparsed — no defensible
reasoning"). **The additive direction is structurally empty.**

## 5. Why the prescribed hybrid could not have worked

A geo-primary + parser-fallback screen verifies `geo ∪ parser` pairs. Since
`geo ⊆ parser` to 99 %, that union is the parser's set — the hybrid *is* the
parser arm, and would spend a one-shot run reproducing 64/100. The converse
composition (parser ∧ geo, geometry as a second witness) is a conjunction:
it can only *remove* verified pairs. Measured directly on the two questions
the parser misses for NLI reasons, geo's margins certify q98
(m_w +0.937, m_p +0.366 — passes) but reject q23 (m_p −0.384 — fails), so
the dual-witness rule is strictly weaker than the single symbolic witness.
Both compositions are dominated; neither warranted a shot.

## 6. Where the parser arm's own residual actually is

Decomposing its 29 conflicted failures by whether the gold-valued fact is
even on the retrieved page:

| cause | n | fixable by any page edit? |
| --- | ---: | --- |
| retrieval miss (gold fact not on the page) | **22** | no — retrieval bound, arm-independent |
| parametric-knowledge failure (page already correct after the edit, or holds only the gold fact; model answers from its weights) | **5** | no — q6, q30, q73, q88, q90 |
| detection miss | **2** | yes — q23, q98 |

The same decomposition for geo: 22 retrieval (identical — it is arm-independent)
and 15 detection misses. Unique-stratum ceiling is the native 28/34.

So the absolute ceiling for *any* suppression-only detector on this substrate
is **44/66 conflicted + 28/34 unique = 72/100**, and the parser arm at 64 is
within 2 detection misses of everything its mechanism can reach.

## 7. What the two detection misses are — and the arm they motivated

Both are same-key pairs the **bidirectional NLI gate** rejects, not the
identity screen:

| q | pair | min contradiction | verdict |
| --- | --- | ---: | --- |
| 23 | `racing video game … sport of racing` → `… Australian rules football` | 0.854 | below the 0.90 threshold |
| 98 | `The Kinks was founded in the city of London` → `… of England` | 0.0002 | NLI judges the values compatible |

q98 is the diagnostic case: the NLI is *factually right* (London is in
England) and *wrong for this store*, whose convention is that a later serial
supersedes the same key. Calibration shows the mechanism is systematic:
**2.2 % of true supersession pairs (13 of 536 at cos ≥ 0.90) are blocked by
the NLI gate** on sh_6k + sh_32k, and 11 on sh_64k — precisely the parser
arm's missing pool recall (0.978).

That finding, not a hybrid, is what Part 2 tests.
