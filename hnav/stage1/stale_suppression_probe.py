#!/usr/bin/env python3
"""Oracle probe: is the conflicted-stratum failure POSITIONAL or PARAMETRIC?  [T12]

A **measurement**, not a policy. Nothing here can ship: every arm needs the
expected answer to decide which fact to cut or move, and the module lives under
``hnav/stage1/`` precisely because that is where reading the answer list is
permitted. It exists to separate two explanations that the native run confounds.

The thing being explained
-------------------------
``hnav/labeling/question_strata.py`` (T12) splits the arena's questions by
whether the queried ``(relation, subject)`` key carries a conflict at all:

* unique-key questions: **26/26 correct in all eight committed ``sh_6k`` runs**;
* conflicted-key questions: **0–5 of 74**, and 572 of the 575 errors are the
  *stale value of the correct key*.

So the model reliably finds the right key and reliably picks the superseded
value from it, under a prompt that states "the newer fact has larger serial
number". Two explanations fit that equally well from the native run alone:

**H-position/presence** — the stale fact is present, and earlier; the model
anchors on the first (or the most numerous) mention it sees. Remove the stale
fact and the answer becomes correct; move the newest to the end and it improves.

**H-parametric** — the contexts are counterfactual rewrites of world facts, so
the stale value is usually the *world-true* one ("the official language of Japan
is Japanese" while gold is "Swedish"). The model is overriding the context with
its parameters. Removing the stale sentence changes nothing, because the model
was never reading it: it recognises the subject and answers from memory.

The two make opposite predictions about ``oracle_suppress``, so one probe
separates them. That is the whole design.

Arms (per question, all graded with the benchmark's own evaluator)
-----------------------------------------------------------------
``native``          the untouched context.
``native_repeat``   the same prompt, a second independent call. This is the
                    in-harness **A/A floor**: every other arm's effect must be
                    read against it, not against zero.
``oracle_suppress`` every fact of the queried key whose value is not the
                    expected answer is deleted. Serials are NOT renumbered —
                    the prompt's rule is stated in terms of serial order, and
                    renumbering would silently change the task.
``oracle_recency``  nothing is deleted; the highest-serial fact of the queried
                    key is moved to the END of the context. Tests placement
                    alone — the direction the only positive result in this
                    corpus used, and the direction ``read_policy.rerank_order``
                    structurally cannot take (it reorders retrieved chunks, not
                    facts within one).
``anti``            the mirror of ``oracle_recency``: the highest-serial fact
                    goes to the FRONT and the most recent stale fact goes last.
                    Whichever end a model privileges, these two arms disagree
                    about which value sits there — so if placement is the
                    mechanism at all, they must move the number in *opposite*
                    directions. Both doing nothing rules placement out; both
                    helping means the effect is "the surgery touched this key",
                    not "the newest fact ended up here".

Readouts, all stratified by unique vs conflicted
------------------------------------------------
The unique stratum is natively 26/26 with zero observed flips across eight runs,
so it is a near-noiseless control: *any* drop there is an artifact of the
surgery, not sampling. Per arm: accuracy overall and per stratum, the A/A floor
alongside, and the paired McNemar cells (b = native right / arm wrong,
c = native wrong / arm right) with an exact two-sided binomial p.

Split discipline
----------------
``sh_6k`` + ``sh_32k`` only. ``sh_64k`` and ``sh_262k`` are the confirmatory
arena and are refused outright (exit 2), as in ``m4_marginal_diff_test.py`` and
``calibrate_read_policy.py``. ``HNAV_MODE=live`` is refused via
``config.require_not_live()``: this is a measurement, and a measurement taken
under intervention is not a measurement.

Harness, and its one deliberate deviation
-----------------------------------------
System message, query template and the ``Memory i:`` user-prompt shape are
imported from ``hnav/stage1/calibrate_read_policy.py``, which transcribes them
verbatim from ``utils/templates.py`` and ``RAGSystem.answer_query``. Grading is
``labeling.counterfactual.substring_exact_match``, the transcription of the
benchmark's own ``substring_exact_match``. ``generation_max_length`` is 10
tokens, as in the dataset config.

The deviation: the campaign feeds the model ``top_k`` *chunks in similarity-rank
order*. On the calibration split every chunk is retrieved (sh_6k: 2 chunks,
sh_32k: 9, against ``retrieve_num=10``), so the model already sees the whole
context — but the block order is query-dependent, which makes "the end of the
context" undefined for a placement experiment. This probe therefore presents the
context as a single ``Memory 1:`` block in context order, identically for every
arm. The native arm's accuracy is reported next to the campaign's recorded
number so the size of that difference is visible rather than assumed.

Running it
----------
The endpoint is ``HNAV_LLM_BASE_URL`` (default ``:8000``). The Stage-1 frozen
substrate is the **:8003** chat server (``hnav/deploy/serve_stage1_chat.sh``:
Qwen3-4B-Instruct-2507 bf16, ``--max-model-len 65536``,
``--no-enable-prefix-caching --max-num-seqs 1 --enforce-eager``), so point the
probe at it explicitly and expect prefill, not decode, to dominate: with prefix
caching disabled every call re-reads its whole context. sh_6k is ~3.2M prompt
tokens over 473 calls; sh_32k is ~16M over 465. Run sh_6k first — it is the
subset the strata finding was measured on, and it answers the question on its
own.

    python hnav/stage1/stale_suppression_probe.py --dry-run     # budget, no calls
    python hnav/stage1/stale_suppression_probe.py --smoke-llm   # offline stub
    HNAV_LLM_BASE_URL=http://localhost:8003/v1 \
        python hnav/stage1/stale_suppression_probe.py --subsets sh_6k
    HNAV_LLM_BASE_URL=http://localhost:8003/v1 \
        python hnav/stage1/stale_suppression_probe.py           # sh_6k + sh_32k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402
from hnav.adapters.mab_adapter import fact_key  # noqa: E402
from hnav.labeling.counterfactual import substring_exact_match  # noqa: E402
from hnav.labeling.question_strata import (FACT_RE_LINE, STRATA,  # noqa: E402
                                           classify_questions, key_members)
# Imported, never re-transcribed: one copy of the benchmark's prompt shape.
from hnav.stage1.calibrate_read_policy import (CALIBRATION,  # noqa: E402
                                               GENERATION_MAX_TOKENS,
                                               QUERY_TEMPLATE, RETRIEVAL_EXTRACT,
                                               SYSTEM_MESSAGE, build_user_prompt)

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
STRATA_JSON = REPO / "stage0_results/question_strata.json"
CONFIRMATORY = ("sh_64k", "sh_262k")
ARMS = ("native", "native_repeat", "oracle_suppress", "oracle_recency", "anti")
# STRATA is imported from question_strata, not redeclared: the probe must
# stratify by exactly the labels the T12 record uses, including ``ambiguous``
# (zero on the calibration split, but the tuple has to stay in step).


# ── context surgery (pure; the only part that touches the fact list) ─────────
def split_context(context: str) -> tuple[str, list[tuple[int, str]]]:
    """``(preamble, [(serial, fact_text)])``.

    The preamble is everything before the first numbered line ("Here is a list
    of facts:"). It is carried through every arm untouched — dropping it would
    change the prompt in a way none of the arms are about.
    """
    facts = [(int(n), t.strip()) for n, t in FACT_RE_LINE.findall(context)]
    first = FACT_RE_LINE.search(context)
    preamble = context[: first.start()].rstrip("\n") if first else context.rstrip("\n")
    return preamble, facts


def render_context(preamble: str, facts: list[tuple[int, str]]) -> str:
    """The raw dataset form: preamble, then one ``"<serial>. <fact>"`` per line.

    With the native fact list this reproduces the dataset's ``context`` field
    exactly (pinned by ``test_stale_suppression_probe.py``), so the ``native``
    arm is the untouched context and not an approximation of it.
    """
    body = "\n".join(f"{s}. {t}" for s, t in facts)
    return (preamble + "\n" + body + "\n") if preamble else body + "\n"


def suppress(facts: list[tuple[int, str]], serials) -> list[tuple[int, str]]:
    """Delete exactly the listed serials; everything else keeps its serial,
    its text and its relative order."""
    drop = set(serials)
    return [(s, t) for s, t in facts if s not in drop]


def move_to_end(facts: list[tuple[int, str]], serial: int) -> list[tuple[int, str]]:
    moved = [(s, t) for s, t in facts if s == serial]
    return [(s, t) for s, t in facts if s != serial] + moved


def move_to_front(facts: list[tuple[int, str]], serial: int) -> list[tuple[int, str]]:
    moved = [(s, t) for s, t in facts if s == serial]
    return moved + [(s, t) for s, t in facts if s != serial]


def surgery_plan(record: dict, members: list[tuple[int, str, str]]) -> dict:
    """What each arm will do to this question's key.

    ``gold_serials``  members whose value is the expected answer;
    ``stale_serials`` members whose value is not — what ``oracle_suppress``
                      deletes;
    ``latest``        highest serial on the key — what the benchmark's own rule
                      names as the answer, and what the two placement arms move;
    ``stale_anchor``  highest-serial stale member other than ``latest`` — the
                      fact ``anti`` puts last. ``None`` when the key has no
                      stale member (the unique stratum) or when the only stale
                      member IS the latest (gold is not the newest fact, 4/77
                      of sh_262k's conflicted questions; 0 on sh_6k).
    """
    truths = set(record["truths"])
    gold = [s for s, _, o in members if o.lower() in truths]
    stale = [s for s, _, o in members if o.lower() not in truths]
    latest = max((s for s, _, _ in members), default=None)
    anchor = max([s for s in stale if s != latest], default=None)
    return {"gold_serials": sorted(gold), "stale_serials": sorted(stale),
            "latest": latest, "stale_anchor": anchor,
            "gold_is_latest": bool(gold) and latest in gold}


def apply_arm(facts: list[tuple[int, str]], arm: str,
              plan: dict) -> list[tuple[int, str]]:
    """The fact list this arm shows the model. ``native``/``native_repeat`` and
    any arm with nothing to operate on return the list unchanged."""
    if arm in ("native", "native_repeat") or not plan:
        return list(facts)
    if arm == "oracle_suppress":
        return suppress(facts, plan["stale_serials"])
    if arm == "oracle_recency":
        return facts if plan["latest"] is None else move_to_end(facts, plan["latest"])
    if arm == "anti":
        out = facts if plan["latest"] is None else move_to_front(facts, plan["latest"])
        if plan["stale_anchor"] is not None:
            out = move_to_end(out, plan["stale_anchor"])
        return out
    raise ValueError(f"unknown arm {arm!r}")


# ── prompting ────────────────────────────────────────────────────────────────
def build_prompt(context_text: str, question: str) -> str:
    """The benchmark's user prompt with the whole context as ``Memory 1``."""
    return build_user_prompt([context_text], QUERY_TEMPLATE.format(question=question))


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def make_answer_fn(args, cfg):
    if args.smoke_llm:
        return primacy_stub()
    from openai import OpenAI  # noqa: PLC0415 — lazy: no network at import

    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def answer(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                      {"role": "user", "content": prompt}],
            temperature=cfg.llm_temperature,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    return answer


def primacy_stub():
    """SMOKE ONLY. A reader that anchors on the FIRST fact whose subject occurs
    in the question — the failure mode the native runs actually exhibit.

    It measures nothing about any model. It exists so the arm construction,
    budgeting, caching, grading and the paired statistics can be exercised end
    to end with known answers: under this stub ``native`` and ``oracle_recency``
    must miss every conflicted question and ``oracle_suppress`` and ``anti``
    must get every one of them right.
    """
    def stub(prompt: str) -> str:
        memory = prompt.split("Pretend you are a knowledge management system")[0]
        m = RETRIEVAL_EXTRACT.search(prompt)
        question = (m.group(1) if m else prompt).lower()
        for _, text in FACT_RE_LINE.findall(memory):
            key, obj = fact_key(text)
            if key and obj and key[1].lower() in question:
                return obj
        return "no idea"
    return stub


# ── statistics ───────────────────────────────────────────────────────────────
def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs.

    The exact form rather than the chi-square approximation because b + c here
    is small by design (the unique stratum should produce zero discordants).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_cells(native: list[bool], arm: list[bool]) -> dict:
    b = sum(1 for x, y in zip(native, arm) if x and not y)
    c = sum(1 for x, y in zip(native, arm) if y and not x)
    return {"n": len(native), "b_native_only": b, "c_arm_only": c,
            "net": c - b, "p_exact": mcnemar_exact_p(b, c)}


def _acc(flags: list[bool]) -> dict:
    return {"n": len(flags), "correct": sum(flags),
            "accuracy": (sum(flags) / len(flags)) if flags else None}


# ── per-subset planning (no LLM involved) ────────────────────────────────────
def plan_subset(item: dict, name: str, max_questions: int | None = None) -> dict:
    preamble, facts = split_context(item["context"])
    members = key_members(item)
    records = classify_questions(item)
    if max_questions:
        records = records[:max_questions]

    questions = []
    for rec in records:
        key = tuple(rec["key"]) if rec["key"] else None
        plan = surgery_plan(rec, members.get(key, [])) if key else {}
        arms = {}
        for arm in ARMS:
            arm_facts = apply_arm(facts, arm, plan)
            text = render_context(preamble, arm_facts)
            arms[arm] = {"prompt": build_prompt(text, rec["question"]),
                         "n_facts": len(arm_facts)}
        native_prompt = arms["native"]["prompt"]
        questions.append({
            "index": rec["index"], "question": rec["question"],
            "truths": rec["truths"], "stratum": rec["stratum"],
            "key": list(rec["key"]) if rec["key"] else None,
            "plan": plan,
            "arms": arms,
            "identical_to_native": {a: arms[a]["prompt"] == native_prompt
                                    for a in ARMS},
        })

    return {"subset": name, "preamble": preamble, "n_facts": len(facts),
            "questions": questions,
            "strata_counts": {s: sum(1 for q in questions if q["stratum"] == s)
                              for s in STRATA}}


def campaign_reference(subset: str, path: Path = STRATA_JSON) -> dict | None:
    """What the committed benchmark runs scored on this subset, for comparison.

    Read from ``stage0_results/question_strata.json`` (T12, task 1) so the
    probe's ``native`` arm can be placed next to the campaign's own native
    number rather than assumed equal to it — the two harnesses differ in how the
    context is blocked (see the module docstring). ``None`` when no committed run
    exists for the subset (sh_32k has none).
    """
    if not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [r for r in payload.get("runs", []) if r["subset"] == subset]
    if not runs:
        return None
    counts = next((s["counts"] for s in payload.get("subsets", [])
                   if s["subset"] == subset), None)
    def _rng(field):
        vals = [r["strata"][field]["accuracy"] for r in runs
                if r["strata"][field]["n"]]
        return [min(vals), max(vals)] if vals else None
    return {
        "source": "stage0_results/question_strata.json",
        "n_runs": len(runs),
        "runs": [r["run"] for r in runs],
        "strata_counts": counts,
        "accuracy_overall_range": [min(r["accuracy_overall"] for r in runs),
                                   max(r["accuracy_overall"] for r in runs)],
        "unique_accuracy_range": _rng("unique"),
        "conflicted_accuracy_range": _rng("conflicted"),
    }


def budget(plans: list[dict]) -> dict:
    """The EXACT number of chat completions the run will issue.

    Arms whose prompt is byte-identical to ``native`` (an ``oracle_suppress``
    on a key with no stale fact, i.e. every unique-stratum question) share the
    cached native answer; ``native_repeat`` deliberately does not, because it is
    the noise floor. Counted here, before anything is sent.
    """
    rows, total_calls, total_chars = [], 0, 0
    for p in plans:
        prompts: set[str] = set()
        chars = 0
        for q in p["questions"]:
            for arm in ARMS:
                if arm == "native_repeat":
                    continue
                text = q["arms"][arm]["prompt"]
                if text not in prompts:
                    prompts.add(text)
                    chars += len(text)
        repeats = len(p["questions"])          # one extra native call each
        calls = len(prompts) + repeats
        chars += sum(len(q["arms"]["native"]["prompt"]) for q in p["questions"])
        rows.append({"subset": p["subset"], "n_questions": len(p["questions"]),
                     "n_distinct_prompts": len(prompts), "n_aa_repeats": repeats,
                     "n_calls": calls,
                     "approx_prompt_tokens": chars // 4,
                     "n_identical_to_native": {
                         a: sum(1 for q in p["questions"]
                                if q["identical_to_native"][a])
                         for a in ARMS if a not in ("native", "native_repeat")}})
        total_calls += calls
        total_chars += chars
    return {"per_subset": rows, "total_calls": total_calls,
            "approx_total_prompt_tokens": total_chars // 4}


# ── execution ────────────────────────────────────────────────────────────────
def run_subset(plan: dict, answer_fn) -> dict:
    cache: dict[str, str] = {}
    n_calls = 0
    per_question = []

    for q in plan["questions"]:
        row = {"index": q["index"], "stratum": q["stratum"], "key": q["key"],
               "truths": q["truths"], "plan": q["plan"], "arms": {}}
        for arm in ARMS:
            prompt = q["arms"][arm]["prompt"]
            if arm == "native_repeat":
                out = answer_fn(prompt)          # deliberately uncached: A/A
                n_calls += 1
            else:
                if prompt not in cache:
                    cache[prompt] = answer_fn(prompt)
                    n_calls += 1
                out = cache[prompt]
            row["arms"][arm] = {
                "output": out,
                "correct": bool(substring_exact_match(out, q["truths"])),
                "n_facts": q["arms"][arm]["n_facts"],
                "prompt_sha": _sha(prompt),
                "identical_to_native": q["identical_to_native"][arm],
            }
        per_question.append(row)

    flags = {a: [r["arms"][a]["correct"] for r in per_question] for a in ARMS}
    by_stratum = {}
    for s in STRATA:
        idx = [i for i, r in enumerate(per_question) if r["stratum"] == s]
        if not idx:
            continue
        by_stratum[s] = {
            "arms": {a: _acc([flags[a][i] for i in idx]) for a in ARMS},
            "paired_vs_native": {
                a: paired_cells([flags["native"][i] for i in idx],
                                [flags[a][i] for i in idx])
                for a in ARMS if a != "native"},
        }

    return {
        "subset": plan["subset"],
        "n_facts": plan["n_facts"],
        "strata_counts": plan["strata_counts"],
        "campaign_reference": campaign_reference(plan["subset"]),
        "n_llm_calls": n_calls,
        "arms": {a: _acc(flags[a]) for a in ARMS},
        "paired_vs_native": {a: paired_cells(flags["native"], flags[a])
                             for a in ARMS if a != "native"},
        "aa_floor": paired_cells(flags["native"], flags["native_repeat"]),
        "by_stratum": by_stratum,
        "per_question": per_question,
    }


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def format_report(results: list[dict]) -> str:
    lines = []
    for res in results:
        lines += ["", "=" * 88,
                  f" {res['subset']}  ({res['n_facts']} facts, "
                  f"{res['strata_counts']} )", "=" * 88,
                  f"{'arm':<18}{'overall':>18}{'unique':>18}{'conflicted':>18}"
                  f"{'b/c vs native':>16}"]
        for arm in ARMS:
            cells = ""
            if arm != "native":
                pc = res["paired_vs_native"][arm]
                cells = f"{pc['b_native_only']}/{pc['c_arm_only']}"
            row = f"{arm:<18}"
            for scope in ("overall", "unique", "conflicted"):
                if scope == "overall":
                    a = res["arms"][arm]
                else:
                    st = res["by_stratum"].get(scope)
                    a = st["arms"][arm] if st else None
                cell = "-" if not a or a["accuracy"] is None else \
                    "{}/{} {:.3f}".format(a["correct"], a["n"], a["accuracy"])
                row += f"{cell:>18}"
            lines.append(row + f"{cells:>16}")
        ref = res.get("campaign_reference")
        if ref:
            lines.append(
                "  campaign reference ({} committed runs): overall {:.3f}-{:.3f}, "
                "unique {:.3f}-{:.3f}, conflicted {:.3f}-{:.3f}".format(
                    ref["n_runs"], *ref["accuracy_overall_range"],
                    *ref["unique_accuracy_range"], *ref["conflicted_accuracy_range"]))
        aa = res["aa_floor"]
        lines.append(f"  A/A floor: {aa['b_native_only']}/{aa['c_arm_only']} "
                     f"discordant of {aa['n']} (p={aa['p_exact']:.3f}) "
                     "- every effect below must clear this")
        for arm in ARMS:
            if arm in ("native", "native_repeat"):
                continue
            pc = res["paired_vs_native"][arm]
            lines.append(f"  {arm:<16} net {pc['net']:+d}  "
                         f"(b={pc['b_native_only']} c={pc['c_arm_only']}, "
                         f"exact p={pc['p_exact']:.4g})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="*", default=list(CALIBRATION))
    ap.add_argument("--max-questions", type=int, default=None,
                    help="cap per subset (smoke / budget trimming)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every arm, print the exact call budget, send nothing")
    ap.add_argument("--smoke-llm", action="store_true",
                    help="SMOKE ONLY: deterministic primacy-anchored stub instead "
                         "of the answer model. Writes *_SMOKE.json. Results are "
                         "MEANINGLESS as evidence about any model.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()

    bad = [s for s in args.subsets if s not in CALIBRATION]
    if bad:
        print(f" REFUSED: {bad} is outside the calibration split {CALIBRATION}.\n"
              f" {CONFIRMATORY} is the confirmatory arena; a probe run there would\n"
              " void the campaign (STAGE1_PLAN.md §0). Nothing was sent.")
        return 2

    data = json.loads(DATA.read_text(encoding="utf-8"))
    plans = []
    for item in data:
        full = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        name = full.replace("factconsolidation_", "")
        if name not in args.subsets:
            continue
        plans.append(plan_subset(item, name, args.max_questions))
    if not plans:
        print(" no subsets selected")
        return 1

    bud = budget(plans)
    print("=" * 88)
    print(" STALE-SUPPRESSION ORACLE PROBE - call budget (exact, pre-flight)")
    print("=" * 88)
    for row in bud["per_subset"]:
        print(f"  {row['subset']:<9} questions={row['n_questions']:<5}"
              f" distinct prompts={row['n_distinct_prompts']:<5}"
              f" A/A repeats={row['n_aa_repeats']:<5}"
              f" calls={row['n_calls']:<6} ~{row['approx_prompt_tokens']:,} prompt tokens")
        print(f"             arms byte-identical to native: "
              f"{row['n_identical_to_native']}")
    print(f"  TOTAL {bud['total_calls']} calls, "
          f"~{bud['approx_total_prompt_tokens']:,} prompt tokens "
          f"(max_tokens={GENERATION_MAX_TOKENS} out)")
    print(f"  server: {'STUB (no endpoint)' if args.smoke_llm else cfg.llm_base_url}"
          f"   model: {'stub' if args.smoke_llm else cfg.llm_model}"
          f"   mode: {cfg.mode}")
    longest = max((row["approx_prompt_tokens"] // max(1, row["n_calls"])
                   for row in bud["per_subset"]), default=0)
    if longest > 60000:
        print(f"  WARNING: ~{longest:,} tokens per prompt is close to the frozen "
              "substrate's --max-model-len 65536")

    if args.dry_run:
        print("\n --dry-run: nothing sent.")
        return 0

    answer_fn = make_answer_fn(args, cfg)
    if args.smoke_llm:
        print("\n *** SMOKE RUN - primacy-anchored stub, not a model. ***")
    results = [run_subset(p, answer_fn) for p in plans]
    print(format_report(results))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "mode": cfg.mode,
        "smoke_llm": bool(args.smoke_llm),
        "harness": {
            "llm_model": None if args.smoke_llm else cfg.llm_model,
            "llm_base_url": None if args.smoke_llm else cfg.llm_base_url,
            "temperature": cfg.llm_temperature,
            "max_tokens": GENERATION_MAX_TOKENS,
            "system_message": SYSTEM_MESSAGE,
            "prompt_shape": "RAGSystem: 'Memory 1:\\n<whole context>\\n' + templated query",
            "prompt_source": "hnav.stage1.calibrate_read_policy (imported verbatim)",
            "grader": "hnav.labeling.counterfactual.substring_exact_match",
            "deviation_from_campaign": (
                "the campaign presents top_k CHUNKS in similarity-rank order; on "
                "the calibration split all chunks are retrieved (2 and 9 vs "
                "retrieve_num=10) so the same facts are shown, but block order is "
                "query-dependent. This probe uses one Memory block in context "
                "order, identically for every arm, because placement arms need a "
                "defined 'end of context'."),
            "aa_floor_caveat": (
                "the A/A floor is two calls in one process; the eight committed "
                "t4_s2 runs give the wider cross-run floor (conflicted stratum "
                "0-5 of 74). Treat the in-harness floor as a LOWER bound on noise."),
        },
        "arms": {
            "native": "untouched context",
            "native_repeat": "same prompt, independent second call (A/A floor)",
            "oracle_suppress": "delete every non-expected-value fact of the queried key; serials NOT renumbered",
            "oracle_recency": "move the highest-serial fact of the queried key to the END",
            "anti": "highest-serial fact to the FRONT, most recent stale fact LAST",
        },
        "split": {"calibration": list(CALIBRATION), "refused": list(CONFIRMATORY)},
        "budget": bud,
        "results": results,
    }
    out = Path(args.out) if args.out else (
        cfg.out_dir / ("stale_suppression_probe_SMOKE.json" if args.smoke_llm
                       else "stale_suppression_probe.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
