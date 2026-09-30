#!/usr/bin/env python3
"""LLM semantic audit of the cos>=0.80 conflict candidate set.  [offline]

Sends every record of ``stage0_results/conflict_pairs/audit_candidates_cos080.jsonl``
to ``openai/gpt-5-mini`` via OpenRouter and records a structured verdict per
pair: the five slot-alignment checks, ``strict_conflict`` / ``update_conflict``,
a reason code and a short explanation. The system prompt is the user's text
verbatim (2026-08-24) — do not edit it.

Money-handling rules (user decision 2026-08-24, hard cap $20):
  * every response's ``usage`` is priced and accumulated; the runner stops
    issuing new calls once spend reaches ``--budget`` (default $19.50);
  * calls are issued in priority order (tagged pairs first, shuffled bulk
    last) so a budget stop loses only the least informative tail;
  * results are append-only JSONL; on restart, completed pair_ids are skipped.

The API key comes from ``OPENROUTER_API_KEY`` in the process env or repo-root
``.env`` (gitignored). It is never written to any output or log line.

Usage:
    python hnav/labeling/audit_runner.py --dry-run            # no network
    python hnav/labeling/audit_runner.py --limit 300          # pilot, then GATE
    python hnav/labeling/audit_runner.py                      # full run
    python hnav/labeling/audit_runner.py --analyze            # post-run tables

Stdlib only (urllib) — no torch, no third-party client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.config import load_env  # noqa: E402

DIR = REPO / "stage0_results" / "conflict_pairs"
CANDIDATES = DIR / "audit_candidates_cos080.jsonl"
RESULTS = DIR / "audit_results_gpt5mini.jsonl"

MODEL = "openai/gpt-5-mini"
BASE_URL = "https://openrouter.ai/api/v1"
SEED = 20260824
BULK_SAMPLE_N = 4000

# openai/gpt-5-mini list prices (USD per token), 2026-08-24. Used only as a
# fallback estimate — the accountant prefers the provider-reported usage.cost.
PRICE_IN = 0.25e-6
PRICE_OUT = 2.00e-6
PRICE_CACHED = 0.03e-6

# ── the judge prompt: USER'S TEXT, VERBATIM (2026-08-24). Do not edit. ───────
SYSTEM_PROMPT = """You are a semantic auditor for fact-pair conflict labels.

Your task is not to check whether either sentence is factually true in the real world. Your task is to determine whether the two sentences express conflicting propositions.

Conflict detection must be treated as a semantic slot-alignment problem before making a contradiction decision. First identify what proposition each sentence expresses; only then compare them.

For each sentence, identify:

* subject: the entity the proposition is about
* relation: the canonical semantic attribute or predicate being asserted
* object: the asserted value
* polarity: positive or negative
* context: any relevant time, location, population, condition, organization, modality, or other validity qualifier

Different surface strings may refer to the same entity. You may use ordinary semantic and entity knowledge to recognize aliases, abbreviations, and equivalent names.

Different phrasings may express the same relation. Normalize paraphrases to the same canonical relation whenever their meaning is equivalent.

Do NOT use world knowledge to verify whether the asserted object value itself is factually correct.

Then evaluate the following:

1. same_referent
   True if both propositions are about the same real-world entity, even if the subject strings differ.

2. same_relation
   True if both propositions assert the same semantic attribute or relation, even if phrased differently.

3. context_overlap
   True if the propositions apply under compatible or overlapping validity conditions. Different times, populations, organizations, locations, conditions, or modalities may make otherwise similar propositions non-conflicting.

4. values_incompatible
   True only if the asserted values cannot both occupy the same semantic slot under the same context.

Do not mark values incompatible merely because their strings differ.

Examples of values that may be compatible:

* paraphrases or aliases
* one value subsuming or containing another
* values that may coexist simultaneously

5. relation_allows_multiple_values
   True if the semantic relation naturally permits multiple simultaneous values, such as multiple citizenships, children, co-founders, languages spoken, memberships, or similar multi-valued relations.

Important distinction:

A strict real-world conflict requires the same referent, same relation, overlapping context, incompatible values, and a relation for which both values cannot simultaneously hold.

A benchmark update conflict uses a memory-store convention in which each (entity, relation) slot stores one current value and a new incompatible value replaces the previous value. Therefore a pair is an update conflict when:

same_referent AND same_relation AND context_overlap AND values_incompatible

Even if the relation is naturally multi-valued, such a pair can still be an update conflict under this convention.

Always make a definite decision. Do not return "unsure".

Use one of these reason codes:

* direct_replacement
* different_referent
* different_relation
* context_mismatch
* alias_equivalent
* relation_paraphrase
* value_paraphrase
* subsumption
* multi_valued_relation
* negation
* other

Keep the explanation extremely short.

Return only the required JSON object and no additional text."""

REASON_CODES = ["direct_replacement", "different_referent", "different_relation",
                "context_mismatch", "alias_equivalent", "relation_paraphrase",
                "value_paraphrase", "subsumption", "multi_valued_relation",
                "negation", "other"]

VERDICT_FIELDS = ["same_referent", "same_relation", "context_overlap",
                  "values_incompatible", "relation_allows_multiple_values",
                  "strict_conflict", "update_conflict", "reason_code",
                  "explanation"]

SCHEMA = {
    "type": "object",
    "properties": {
        "same_referent": {"type": "boolean"},
        "same_relation": {"type": "boolean"},
        "context_overlap": {"type": "boolean"},
        "values_incompatible": {"type": "boolean"},
        "relation_allows_multiple_values": {"type": "boolean"},
        "strict_conflict": {"type": "boolean"},
        "update_conflict": {"type": "boolean"},
        "reason_code": {"type": "string", "enum": REASON_CODES},
        "explanation": {"type": "string", "maxLength": 80},
    },
    "required": VERDICT_FIELDS,
    "additionalProperties": False,
}

# Few-shot turns (fictional entities). Each answer follows the schema exactly
# and demonstrates one reason-code regime, including the two decisions that
# matter most: subsumption is NOT incompatible, and a naturally multi-valued
# relation can still be an update conflict (update true, strict false).
FEWSHOT = [
    ("A: The capital of Freedonia is Arto.\n"
     "B: The capital of Freedonia is Melk.",
     {"same_referent": True, "same_relation": True, "context_overlap": True,
      "values_incompatible": True, "relation_allows_multiple_values": False,
      "strict_conflict": True, "update_conflict": True,
      "reason_code": "direct_replacement",
      "explanation": "One capital slot, two different cities."}),
    ("A: The capital of Freedonia is Arto.\n"
     "B: The capital of Sylvania is Arto.",
     {"same_referent": False, "same_relation": True, "context_overlap": True,
      "values_incompatible": False, "relation_allows_multiple_values": False,
      "strict_conflict": False, "update_conflict": False,
      "reason_code": "different_referent",
      "explanation": "Different countries; no shared slot."}),
    ("A: Jan Novak was born in the city of Brno.\n"
     "B: Jan Novak works in the field of biology.",
     {"same_referent": True, "same_relation": False, "context_overlap": True,
      "values_incompatible": False, "relation_allows_multiple_values": False,
      "strict_conflict": False, "update_conflict": False,
      "reason_code": "different_relation",
      "explanation": "Birthplace vs occupation; different attributes."}),
    ("A: Karel Dvorak plays the sport of rugby union.\n"
     "B: Karel Dvorak plays the sport of rugby.",
     {"same_referent": True, "same_relation": True, "context_overlap": True,
      "values_incompatible": False, "relation_allows_multiple_values": True,
      "strict_conflict": False, "update_conflict": False,
      "reason_code": "subsumption",
      "explanation": "Rugby subsumes rugby union; compatible values."}),
    ("A: W. H. Ostry was born in the city of Tarnow.\n"
     "B: William Henry Ostry was born in the city of Ostrava.",
     {"same_referent": True, "same_relation": True, "context_overlap": True,
      "values_incompatible": True, "relation_allows_multiple_values": False,
      "strict_conflict": True, "update_conflict": True,
      "reason_code": "alias_equivalent",
      "explanation": "Same person via alias; one birthplace, two cities."}),
    ("A: Mira Vance holds citizenship of Norway.\n"
     "B: Mira Vance holds citizenship of Chile.",
     {"same_referent": True, "same_relation": True, "context_overlap": True,
      "values_incompatible": True, "relation_allows_multiple_values": True,
      "strict_conflict": False, "update_conflict": True,
      "reason_code": "multi_valued_relation",
      "explanation": "Distinct values compete for one slot; dual is possible."}),
]


# ── prompt building ──────────────────────────────────────────────────────────

def ab_order(pair_id: str) -> str:
    """Deterministic per-pair A/B assignment: 'ab' keeps (fact_a, fact_b)."""
    h = hashlib.sha256(f"{SEED}|{pair_id}".encode()).digest()
    return "ab" if h[0] % 2 == 0 else "ba"


def build_messages(rec: dict) -> tuple[list[dict], str]:
    order = ab_order(rec["pair_id"])
    a, b = ((rec["fact_a"], rec["fact_b"]) if order == "ab"
            else (rec["fact_b"], rec["fact_a"]))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, ans in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"A: {a}\nB: {b}"})
    return msgs, order


# ── priority ordering ────────────────────────────────────────────────────────

def _slice_of(rec: dict) -> int:
    """1 tagged | 2 same-key-untagged or unparsed | 3 structural channels |
    4/5 bulk (split later)."""
    if rec["parser_tagged_conflict"]:
        return 1
    m = rec["parser_metadata"]
    if not m["both_parse"] or m["same_key"]:
        return 2
    if m["same_subject"] and not m["same_key"]:
        return 3                       # cross-template / relation-paraphrase
    if m["same_relation"] and m["same_object"] and not m["same_subject"]:
        return 3                       # alias-candidate channel
    return 4


def priority_order(records: list[dict], seed: int = SEED,
                   bulk_sample_n: int = BULK_SAMPLE_N) -> list[dict]:
    """Deterministic audit order: tagged -> parser blind spots -> structural
    channels -> a seeded bulk sample -> remaining bulk, shuffled."""
    rng = random.Random(seed)
    slices: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        slices[_slice_of(r)].append(r)
    bulk = slices.pop(4, [])
    rng.shuffle(bulk)
    sample, rest = bulk[:bulk_sample_n], bulk[bulk_sample_n:]
    out = []
    for k in (1, 2, 3):
        out.extend(sorted(slices.get(k, []), key=lambda r: r["pair_id"]))
    out.extend(sample)
    out.extend(rest)
    return out


def is_subsumption_candidate(rec: dict) -> bool:
    """Tagged pair whose objects are substring-nested — the known FP channel."""
    if not rec["parser_tagged_conflict"]:
        return False
    m = rec["parser_metadata"]
    oa = m["fact_a_parsed"]["object"].lower()
    ob = m["fact_b_parsed"]["object"].lower()
    return oa != ob and (oa in ob or ob in oa)


def pilot_selection(ordered: list[dict], limit: int) -> list[dict]:
    """Stratified pilot: half tagged (subsumption candidates force-included),
    half untagged in priority order."""
    tagged = [r for r in ordered if r["parser_tagged_conflict"]]
    untag = [r for r in ordered if not r["parser_tagged_conflict"]]
    n_tag = min((limit + 1) // 2, len(tagged))
    forced = [r for r in tagged if is_subsumption_candidate(r)][:n_tag]
    forced_ids = {r["pair_id"] for r in forced}
    pick_t = forced + [r for r in tagged if r["pair_id"] not in forced_ids]
    return pick_t[:n_tag] + untag[:limit - n_tag]


# ── cost accounting ──────────────────────────────────────────────────────────

def price(usage: dict) -> float:
    """Fallback list-price estimate from a usage dict (provider cost wins)."""
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return (pt - cached) * PRICE_IN + cached * PRICE_CACHED + ct * PRICE_OUT


class BudgetTracker:
    """Thread-safe spend accumulator with a hard stop."""

    def __init__(self, stop_usd: float):
        self.stop_usd = float(stop_usd)
        self._spent = 0.0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        with self._lock:
            self._spent += float(cost)

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def allow(self) -> bool:
        return self.spent < self.stop_usd


# ── OpenRouter client ────────────────────────────────────────────────────────

RETRY_STATUS = {408, 429, 500, 502, 503, 524}


class Judge:
    def __init__(self, api_key: str, model: str = MODEL,
                 base_url: str = BASE_URL, timeout: float = 120.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.send_reasoning = True     # dropped if the provider rejects it

    def _body(self, messages: list[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "conflict_audit", "strict": True, "schema": SCHEMA}},
            "max_tokens": 600,
            "usage": {"include": True},
        }
        if self.send_reasoning:
            body["reasoning"] = {"effort": "minimal"}
        return body

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://localhost/hnav-audit",
                     "X-Title": "hnav conflict audit"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def call(self, messages: list[dict], max_tries: int = 5) -> dict:
        """One judged pair. Returns {"verdict"|None, "usage", "cost_usd",
        "status", "error"?}. Retries transport errors; retries a malformed
        JSON body once."""
        json_retry = 1
        delay = 1.0
        for attempt in range(max_tries):
            try:
                resp = self._post(self._body(messages))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code == 400 and self.send_reasoning and "reasoning" in detail.lower():
                    self.send_reasoning = False
                    continue           # same attempt budget, new body
                if e.code in (401, 402, 403):
                    raise RuntimeError(f"auth/credit failure HTTP {e.code}: "
                                       f"{detail}") from e
                if e.code in RETRY_STATUS and attempt < max_tries - 1:
                    time.sleep(delay + random.random())
                    delay = min(delay * 2, 30)
                    continue
                return {"verdict": None, "usage": {}, "cost_usd": 0.0,
                        "status": "error", "error": f"HTTP {e.code}: {detail}"}
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < max_tries - 1:
                    time.sleep(delay + random.random())
                    delay = min(delay * 2, 30)
                    continue
                return {"verdict": None, "usage": {}, "cost_usd": 0.0,
                        "status": "error", "error": f"transport: {e}"}

            usage = resp.get("usage") or {}
            cost = usage.get("cost")
            cost = float(cost) if cost is not None else price(usage)
            try:
                content = resp["choices"][0]["message"]["content"]
                verdict = json.loads(content)
                missing = [f for f in VERDICT_FIELDS if f not in verdict]
                if missing or verdict.get("reason_code") not in REASON_CODES:
                    raise ValueError(f"bad verdict: missing={missing}")
            except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
                if json_retry > 0:
                    json_retry -= 1
                    continue
                return {"verdict": None, "usage": usage, "cost_usd": cost,
                        "status": "parse_error", "error": str(e)[:200]}
            return {"verdict": verdict, "usage": usage, "cost_usd": cost,
                    "status": "ok"}
        return {"verdict": None, "usage": {}, "cost_usd": 0.0,
                "status": "error", "error": "retries exhausted"}


# ── run orchestration ────────────────────────────────────────────────────────

def load_candidates(path: pathlib.Path = CANDIDATES) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def load_done(path: pathlib.Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["pair_id"])
    return done


def run(args) -> int:
    env = load_env()
    api_key = env.get("OPENROUTER_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        print("OPENROUTER_API_KEY missing from environment/.env — refusing.",
              file=sys.stderr)
        return 2

    records = load_candidates(pathlib.Path(args.candidates))
    ordered = priority_order(records)
    if args.limit:
        ordered = pilot_selection(ordered, args.limit)

    out = pathlib.Path(args.out)
    done = load_done(out)
    todo = [r for r in ordered if r["pair_id"] not in done]
    n_tagged = sum(r["parser_tagged_conflict"] for r in todo)
    print(f"candidates={len(records):,}  selected={len(ordered):,}  "
          f"already done={len(done):,}  to send={len(todo):,} "
          f"(tagged={n_tagged:,})")

    if args.dry_run:
        chars = [sum(len(m["content"]) for m in build_messages(r)[0])
                 for r in todo[:2000]] or [0]
        avg_chars = sum(chars) / len(chars)
        est_in = avg_chars / 3.8                       # rough chars->tokens
        est_out = 95
        n = len(todo)
        worst = n * (est_in * PRICE_IN + est_out * PRICE_OUT)
        cached = n * (300 * PRICE_IN + (est_in - 300) * PRICE_CACHED
                      + est_out * PRICE_OUT)
        print(f"dry-run: ~{est_in:,.0f} in-tok/call (chars/3.8), "
              f"{est_out} out-tok/call assumed")
        print(f"  projected, no caching : ${worst:,.2f}")
        print(f"  projected, cached pfx : ${cached:,.2f}")
        print("  (pilot replaces both numbers with measured ones)")
        return 0

    judge = Judge(api_key, model=args.model, base_url=args.base_url)
    budget = BudgetTracker(args.budget)
    write_lock = threading.Lock()
    tally = Counter()
    t0 = time.time()
    n_done = 0

    def work(rec: dict) -> None:
        nonlocal n_done
        msgs, order = build_messages(rec)
        res = judge.call(msgs)
        budget.add(res["cost_usd"])
        line = {
            "pair_id": rec["pair_id"], "subset": rec["subset"],
            "parser_tagged_conflict": rec["parser_tagged_conflict"],
            "judge_input_order": order, "status": res["status"],
            "verdict": res["verdict"], "model": args.model,
            "usage": {k: res["usage"].get(k) for k in
                      ("prompt_tokens", "completion_tokens", "total_tokens",
                       "cost")} | {"cached_tokens":
                                   (res["usage"].get("prompt_tokens_details")
                                    or {}).get("cached_tokens"),
                                   "reasoning_tokens":
                                   (res["usage"].get("completion_tokens_details")
                                    or {}).get("reasoning_tokens")},
            "cost_usd": res["cost_usd"], "ts": time.time(),
        }
        if res.get("error"):
            line["error"] = res["error"]
        with write_lock:
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_done += 1
            tally[res["status"]] += 1
            if res["verdict"]:
                tally["update_conflict" if res["verdict"]["update_conflict"]
                      else "not_conflict"] += 1
            if n_done % 100 == 0:
                dt = time.time() - t0
                rate = n_done / dt
                eta = (len(todo) - n_done) / rate / 60 if rate else 0
                print(f"  {n_done:>6,}/{len(todo):,}  ${budget.spent:6.2f}  "
                      f"{rate:5.1f}/s  eta {eta:5.1f} min  "
                      f"ok={tally['ok']} err={tally['error']} "
                      f"parse={tally['parse_error']} "
                      f"conflict={tally['update_conflict']} "
                      f"not={tally['not_conflict']}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    stopped_at = None
    pending = set()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, rec in enumerate(todo):
            if not budget.allow():
                stopped_at = i
                break
            pending.add(ex.submit(work, rec))
            if len(pending) >= args.concurrency * 2:
                done_f, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done_f:
                    f.result()         # surface auth failures immediately
        for f in pending:
            f.result()

    print(f"\nfinished: sent={n_done:,}  spent=${budget.spent:.4f}  "
          f"statuses={dict(tally)}")
    if stopped_at is not None:
        print(f"BUDGET STOP at ${budget.stop_usd}: {len(todo) - stopped_at:,} "
              f"pairs not sent (priority tail). Results file records "
              f"everything completed.")
    return 0


# ── analysis ─────────────────────────────────────────────────────────────────

def analyze(args) -> int:
    cand = {r["pair_id"]: r for r in load_candidates(pathlib.Path(args.candidates))}
    out = pathlib.Path(args.out)
    results = []
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                results.append(json.loads(line))
    unknown = [r for r in results if r["pair_id"] not in cand]
    assert not unknown, f"{len(unknown)} result pair_ids not in candidates"

    per = defaultdict(lambda: defaultdict(Counter))
    disagreements, discoveries = [], []
    total_cost = sum(r.get("cost_usd") or 0.0 for r in results)
    for r in results:
        s, tagged = r["subset"], r["parser_tagged_conflict"]
        grp = "tagged" if tagged else "untagged"
        per[s][grp]["n"] += 1
        if r["status"] != "ok":
            per[s][grp][r["status"]] += 1
            continue
        v = r["verdict"]
        per[s][grp]["update_conflict"] += v["update_conflict"]
        per[s][grp]["strict_conflict"] += v["strict_conflict"]
        per[s][grp][f"reason:{v['reason_code']}"] += 1
        c = cand[r["pair_id"]]
        row = {"pair_id": r["pair_id"], "subset": s,
               "fact_a": c["fact_a"], "fact_b": c["fact_b"],
               "cosine_similarity": c["cosine_similarity"],
               "parser_metadata": c["parser_metadata"], "verdict": v}
        if tagged and not v["update_conflict"]:
            disagreements.append({"kind": "tagged_not_conflict", **row})
        if not tagged and v["update_conflict"]:
            discoveries.append({"kind": "untagged_conflict", **row})

    review = out.with_name("audit_review_disagreements.jsonl")
    with review.open("w", encoding="utf-8", newline="\n") as fh:
        for row in disagreements + discoveries:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["# LLM semantic audit — summary", "",
             f"- model: `{args.model}`, results: {len(results):,}, "
             f"total cost ${total_cost:.2f}",
             f"- disagreements (tagged, judge says not conflict): "
             f"{len(disagreements)}",
             f"- discoveries (untagged, judge says update conflict): "
             f"{len(discoveries)}", ""]
    for s in sorted(per):
        lines.append(f"## {s}")
        for grp in ("tagged", "untagged"):
            c = per[s][grp]
            if not c["n"]:
                continue
            lines.append(f"- **{grp}** n={c['n']:,}: update_conflict="
                         f"{c['update_conflict']:,} strict={c['strict_conflict']:,} "
                         f"errors={c['error'] + c['parse_error']}")
            reasons = {k.split(':', 1)[1]: v for k, v in c.items()
                       if k.startswith("reason:")}
            top = sorted(reasons.items(), key=lambda kv: -kv[1])[:6]
            lines.append("  - reasons: " +
                         ", ".join(f"{k} {v:,}" for k, v in top))
        lines.append("")
    md = out.with_name("AUDIT_SUMMARY.md")
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    summary = {"model": args.model, "n_results": len(results),
               "total_cost_usd": round(total_cost, 4),
               "n_disagreements": len(disagreements),
               "n_discoveries": len(discoveries),
               "per_subset": {s: {g: dict(per[s][g]) for g in per[s]}
                              for s in per}}
    js = out.with_name("AUDIT_SUMMARY.json")
    js.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                  encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"wrote {review}\nwrote {md}\nwrote {js}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--limit", type=int, default=0,
                    help="pilot: stratified selection of this many pairs")
    ap.add_argument("--budget", type=float, default=19.50,
                    help="hard stop in USD (cap $20)")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
