#!/usr/bin/env python3
"""Question strata: does the queried key carry a conflict at all?  [T12]

**Offline only.** Reads the benchmark's ``questions``/``answers`` and the
committed run artifacts under ``stage0_results/t4_s2_evidence/``. Nothing under
``hnav/core/`` or ``hnav/adapters/`` may import this module; the AST scan in
``hnav/tests/test_leakage_audit.py`` enforces that for the whole tree.

Why this exists
---------------
The primary arena's headline accuracy (``m3_headroom.json``: 0.33 / 0.47 / 0.44
/ 0.20) is an average over two populations that behave nothing alike:

* **unique** — the queried ``(relation, subject)`` key has exactly one distinct
  value in the context. There is no conflict to resolve; the question is a
  plain lookup.
* **conflicted** — the key carries two or more distinct values, so answering
  requires applying the rule the prompt states ("the newer fact has larger
  serial number").
* **ambiguous** — two or more keys qualify and they *disagree* about whether the
  question is conflicted. Named rather than silently first-won. Zero on the
  calibration split, zero on sh_64k, zero on sh_262k under the rule below.
* **unmatched** — no key whose subject occurs in the question also carries the
  expected answer among its values. Two questions in ``sh_262k``; none on the
  calibration split.

Splitting the eight committed ``sh_6k`` runs this way shows the average is a
mixture: the unique stratum is answered perfectly and the conflicted stratum is
answered almost never. The error taxonomy says *how* it fails — overwhelmingly
by emitting the superseded value **of the right key**, not by retrieving the
wrong key and not by refusing.

Classification rule
-------------------
The base rule is the committed one from ``gold_rule.py`` (and its packaged form
``counterfactual.map_questions_to_keys``): *the key whose subject appears in the
question and whose object set intersects the expected answer*. Facts are parsed
by the validated ``conflict_analysis.parse`` (99.5%+ template coverage), which is
imported and never re-derived.

Two refinements, both forced by a real mis-assignment found in supervisor audit
(sh_262k q26, "What is the country of citizenship of Arthur Miller?"), where a
bare substring match let the key ``(was created in the country of, "Arthur")``
qualify and, being earlier in traversal, silently win over the correct
``(is a citizen of, "Arthur Miller")`` — turning a unique question into a
conflicted one:

1. **word-boundary subject matching**, so a subject cannot match inside a longer
   word ("Arthur" no longer matches "Arthurian"). Note this does *not* by itself
   fix the audit case — "Arthur" is followed by a space in "Arthur Miller" and
   so is a legitimate boundary match;
2. **all candidates are collected, none is first-won.** This is what fixes it: a
   candidate whose subject is properly contained in another candidate's subject
   is dropped as strictly less specific, so "Arthur" loses to "Arthur Miller".
   If candidates still disagree about conflictedness, the question is labelled
   ``ambiguous`` rather than assigned.

Effect on the counts: sh_262k is **22 unique / 76 conflicted** (was 21/77); the
other three subsets are unchanged. ``test_question_strata.py`` pins the
difference from ``map_questions_to_keys`` at exactly this one question out of
400, keeps the Arthur Miller case as a fixture, and asserts zero ambiguity on
the calibration split.

Independence from the 512-token truncation defect
-------------------------------------------------
Nothing here embeds anything. The strata come from the dataset's own text and
the committed run outputs — no embedder, no retriever, no vector. So this result
does **not** move when the truncation defect is re-fit, unlike most numbers in
this project, which are provisional pending that re-fit.

    python hnav/labeling/question_strata.py             # table + JSON
    python hnav/labeling/question_strata.py --no-write  # table only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.conflict_analysis import parse as parse_fact  # noqa: E402
from hnav.labeling.counterfactual import (normalize_answer,  # noqa: E402
                                          substring_exact_match)

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
EVIDENCE = REPO / "stage0_results/t4_s2_evidence"
M3 = REPO / "stage0_results/final/m3_headroom.json"
OUT = REPO / "stage0_results/question_strata.json"

STRATA = ("unique", "conflicted", "ambiguous", "unmatched")
ERROR_CLASSES = ("stale_value", "off_list", "empty")
SINGLE_HOP_PREFIX = "factconsolidation_sh_"

FACT_RE_LINE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)


# ── parsing ──────────────────────────────────────────────────────────────────
def key_members(item: dict) -> dict[tuple[str, str], list[tuple[int, str, str]]]:
    """``(relation, subject) -> [(serial, fact_text, object)]``, context order.

    The raw ``context`` is newline-separated ("Here is a list of facts:\\n0. ..."),
    so the line-anchored form is exact here — this is the raw dataset field, not
    a memorize chunk that has been through the benchmark's sentence joiner.
    """
    groups: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    for num, txt in FACT_RE_LINE.findall(item.get("context", "")):
        p = parse_fact(txt)
        if p is None:
            continue
        rel, subj, obj = p
        groups.setdefault((rel, subj), []).append((int(num), txt, obj))
    return groups


_SUBJECT_RE: dict[str, re.Pattern] = {}


def _subject_pattern(subject: str) -> re.Pattern:
    """Compiled once per distinct subject. ``sh_262k`` has ~11k of them and the
    stdlib pattern cache holds 512, so without this the scan thrashes."""
    pat = _SUBJECT_RE.get(subject)
    if pat is None:
        pat = _SUBJECT_RE[subject] = re.compile(
            r"(?<!\w)" + re.escape(subject) + r"(?!\w)", re.IGNORECASE)
    return pat


def subject_matches(subject: str, question: str) -> bool:
    """Does ``subject`` occur in ``question`` on token boundaries?

    A bare ``in`` test is what produced the one known mis-assignment (see the
    module docstring): "Arthur" is inside "Arthur Miller". Lookarounds rather
    than ``\\b`` so that subjects starting or ending with a non-word character
    ("H. P. Lovecraft", "(band)") still anchor correctly.
    """
    if not subject:
        return False
    return _subject_pattern(subject).search(question) is not None


def candidate_keys(question: str, truths: set[str],
                   groups: dict) -> list[tuple[str, str]]:
    """Every key that qualifies for this question, most specific first.

    Qualifying = the subject matches on token boundaries **and** the key carries
    the expected answer among its values. A candidate whose subject is properly
    contained (again on token boundaries) in another candidate's subject is
    dropped: it is a strictly less specific reading of the same span. Whatever
    survives is returned longest-subject-first; the caller decides what to do
    when the survivors disagree.

    The cheap ``in`` test in front of the regex is a pure prefilter: a
    token-boundary match is always a substring match, so nothing that would
    qualify is skipped. It matters because the scan is |keys| x |questions| and
    sh_262k has 11,037 keys.
    """
    low = question.lower()
    cands = [key for key, members in groups.items()
             if key[1] and key[1].lower() in low
             and subject_matches(key[1], question)
             and any(o.lower() in truths for _, _, o in members)]
    kept = [k for k in cands
            if not any(other is not k and len(other[1]) > len(k[1])
                       and subject_matches(k[1], other[1]) for other in cands)]
    return sorted(kept, key=lambda k: (-len(k[1]), k[0]))


def stratum_of(record: dict) -> str:
    """The stratum recorded on an already-classified question."""
    return record["stratum"]


def classify_questions(item: dict) -> list[dict]:
    """Every question of one subset, stratified, with the key's value inventory.

    Carries what the error taxonomy needs — ``other_values`` (the key's values
    that are *not* the expected answer) and ``gold_is_latest`` (whether the
    expected answer sits on the highest serial, i.e. whether the benchmark's own
    stated rule would produce it) — plus the candidate audit trail
    (``n_candidates`` / ``candidate_subjects``) that makes an ambiguous
    assignment visible instead of silently resolved.

    ``ambiguous`` questions deliberately get an EMPTY ``other_values``: without
    a settled key there is no defensible "superseded value of the same key", so
    every error on them scores ``off_list``. That is the direction that works
    against the headline finding, which is the direction an unresolved case
    should push.
    """
    groups = key_members(item)
    out = []
    for qi, (question, answers) in enumerate(zip(item.get("questions", []),
                                                 item.get("answers", []))):
        truths = {str(a).lower() for a in
                  (answers if isinstance(answers, (list, tuple)) else [answers])}
        cands = candidate_keys(question, truths, groups)
        conflicted_of = {k: len({o for _, _, o in groups[k]}) > 1 for k in cands}

        if not cands:
            stratum, key, members = "unmatched", None, []
        elif len(set(conflicted_of.values())) > 1:
            stratum, key, members = "ambiguous", None, []
        else:
            key = cands[0]
            members = sorted(groups[key])
            stratum = "conflicted" if conflicted_of[key] else "unique"

        matching = [s for s, _, o in members if o.lower() in truths]
        other: list[str] = []
        for _, _, obj in members:
            if obj.lower() not in truths and obj not in other:
                other.append(obj)
        latest = max((s for s, _, _ in members), default=None)
        target = max(matching) if matching else None

        out.append({
            "index": qi, "question": question, "truths": sorted(truths),
            "stratum": stratum,
            "key": key, "conflicted": stratum == "conflicted",
            "target_serial": target,
            "member_serials": [s for s, _, _ in members],
            "other_values": other,
            "n_members": len(members),
            "latest_serial": latest,
            "gold_is_latest": bool(members and target is not None
                                   and target == latest),
            "n_candidates": len(cands),
            "candidate_subjects": [k[1] for k in cands],
        })
    return out


# ── error taxonomy ───────────────────────────────────────────────────────────
def error_class(output: str, record: dict) -> str:
    """Classify one **incorrect** model output for one question.

    ``empty``        nothing survives the evaluator's normalization;
    ``stale_value``  the output contains a non-expected value **of the same
                     key** — the model found the right key and picked a
                     superseded value from it;
    ``off_list``     anything else: a value that is not on the queried key's
                     list at all (a different key's object, or a world-knowledge
                     answer that never appears under this key).

    Substring containment with the benchmark's own ``normalize_answer`` is used
    on purpose: it is the same comparison ``substring_exact_match`` makes for
    the expected answer, so "correct" and "stale" are decided on one scale.
    """
    pred = normalize_answer(output or "")
    if not pred.strip():
        return "empty"
    for value in record.get("other_values", ()):
        norm = normalize_answer(value)
        if norm and norm in pred:
            return "stale_value"
    return "off_list"


# ── run artifacts ────────────────────────────────────────────────────────────
def subset_of_run(payload: dict) -> str:
    """Subset short name, taken from the run's own ``dataset_config``."""
    full = payload.get("dataset_config", {}).get("sub_dataset", "")
    return full.replace("factconsolidation_", "") or "unknown"


def score_run(payload: dict, records: list[dict]) -> dict:
    """Per-stratum accuracy and error taxonomy for one committed run.

    Grades are **recomputed** with H-Nav's transcription of the evaluator and
    compared against the ``substring_exact_match`` field the benchmark itself
    wrote; the agreement count is reported rather than assumed.
    """
    per = {s: {"n": 0, "correct": 0, "errors": Counter()} for s in STRATA}
    disagree, id_mismatch, off_list_samples = 0, 0, []
    sub = payload.get("dataset_config", {}).get("sub_dataset", "")

    for row in payload.get("data", []):
        qi = row.get("query_id")
        if qi is None or qi >= len(records):
            id_mismatch += 1
            continue
        if row.get("qa_pair_id") not in (None, f"{sub}_no{qi}"):
            id_mismatch += 1
        rec = records[qi]
        output = row.get("parsed_output", row.get("output", "")) or ""
        ok = substring_exact_match(output, rec["truths"])
        if bool(row.get("substring_exact_match")) != ok:
            disagree += 1
        bucket = per[rec["stratum"]]
        bucket["n"] += 1
        if ok:
            bucket["correct"] += 1
        else:
            cls = error_class(output, rec)
            bucket["errors"][cls] += 1
            if cls == "off_list":
                off_list_samples.append(
                    {"index": qi, "output": output, "truths": rec["truths"],
                     "other_values": rec["other_values"]})

    strata = {}
    for s in STRATA:
        b = per[s]
        strata[s] = {
            "n": b["n"], "correct": b["correct"],
            "accuracy": (b["correct"] / b["n"]) if b["n"] else None,
            "errors": {c: b["errors"].get(c, 0) for c in ERROR_CLASSES},
        }
    n = sum(per[s]["n"] for s in STRATA)
    correct = sum(per[s]["correct"] for s in STRATA)
    return {
        "subset": subset_of_run(payload),
        "n_rows": n,
        "accuracy_overall": (correct / n) if n else None,
        "strata": strata,
        "grade_check": {"n_compared": n, "n_disagreements": disagree,
                        "n_id_mismatches": id_mismatch},
        "off_list_outputs": off_list_samples,
        "recorded_averaged_substring_exact_match":
            payload.get("averaged_metrics", {}).get("substring_exact_match"),
    }


def load_runs(evidence_dir: Path = EVIDENCE) -> list[tuple[str, Path, dict]]:
    """``[(run_name, path, payload)]`` for every ``*_results.json`` present."""
    runs = []
    for path in sorted(Path(evidence_dir).glob("*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append((path.name[: -len("_results.json")], path, payload))
    return runs


# ── conflicted-only accuracy: measured estimate, or assumption-free bound ────
M3_CAVEAT = ("m3_headroom.py grades with H-Nav's own counterfactual prompt "
             "(labeling.counterfactual.build_prompt) over retrieved chunks, NOT "
             "the benchmark's templated query used by the t4_s2 evidence runs. "
             "Every number derived from accuracy_native is an m3-harness number "
             "and must be labelled as one wherever it is exported.")


def accuracy_bounds(n_correct: int, n_unique: int, n_conflicted: int,
                    n_other: int = 0) -> dict:
    """Assumption-free interval for conflicted-only accuracy, and the ceiling
    the pooled count puts on unique-stratum accuracy.

    Pure counting, no premise about either stratum. With ``k`` correct answers
    in total, the conflicted stratum can have absorbed at fewest
    ``max(0, k - n_unique - n_other)`` of them and at most ``min(k, n_conflicted)``;
    the unique stratum cannot have more than ``min(k, n_unique)`` correct. When
    that last ratio is below 1 the "unique questions are always correct" premise
    is not merely unverified for the subset, it is **refuted** by arithmetic —
    which is the honest way to say what a negative point estimate was trying to.
    """
    if not n_conflicted:
        return {}
    lo = max(0, n_correct - n_unique - n_other) / n_conflicted
    hi = min(n_correct, n_conflicted) / n_conflicted
    unique_ceiling = (min(n_correct, n_unique) / n_unique) if n_unique else None
    return {"lo": lo, "hi": hi, "unique_accuracy_upper_bound": unique_ceiling,
            "assumption_refuted": bool(unique_ceiling is not None
                                       and unique_ceiling < 1.0)}


def conflicted_only_accuracy(acc_pooled: float | None, counts: dict,
                             n_total: int, measured_unique: dict | None) -> dict:
    """The conflicted-stratum accuracy implied by a pooled accuracy.

    Emits a **point estimate only where the premise it needs has been measured**
    — i.e. where committed per-question runs exist for the subset and show the
    unique stratum at 1.000 (sh_6k, 26/26 in all eight). Everywhere else the
    field is a two-sided ``bound`` and ``estimate`` is ``None``: a point estimate
    resting on an unverified premise invites exactly the objection that killed
    the first version of this table, where sh_262k produced a *negative
    probability*.

    The bound is always emitted, including for sh_6k — where the point estimate
    turns out to coincide with the bound's lower end, because assuming the
    unique stratum is perfect is precisely the assumption that leaves the
    conflicted stratum the fewest correct answers to claim.
    """
    n_u, n_c = counts.get("unique", 0), counts.get("conflicted", 0)
    n_other = n_total - n_u - n_c
    out = {"kind": "unavailable", "estimate": None, "bound": None,
           "pooled_accuracy": acc_pooled, "n_correct_pooled": None,
           "unique_accuracy": measured_unique,
           "harness": "m3_headroom.py accuracy_native", "harness_caveat": M3_CAVEAT}
    if acc_pooled is None or not n_c:
        return out

    k = int(round(acc_pooled * n_total))
    out["n_correct_pooled"] = k
    out["bound"] = accuracy_bounds(k, n_u, n_c, n_other)
    if measured_unique and measured_unique.get("accuracy") == 1.0:
        out["kind"] = "estimate"
        out["estimate"] = (k - n_u) / n_c
        out["basis"] = (f"unique stratum measured at 1.000 in "
                        f"{measured_unique['n_runs']} committed runs "
                        f"({measured_unique['source']})")
    else:
        out["kind"] = "bound"
        out["basis"] = ("no committed per-question run for this subset, so the "
                        "unique-stratum premise is unmeasured; counting bound only")
    return out


def m3_native_accuracy(path: Path = M3) -> dict[str, float]:
    if not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["subset"]: row.get("read", {}).get("accuracy_native")
            for row in payload if isinstance(row, dict) and "subset" in row}


# ── provenance ───────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _producer_uncommitted() -> bool | None:
    """Was *this module* uncommitted when the record was written?

    If it was, ``git_head`` names the parent of the producing code, not the
    producing code — worth stating rather than leaving the reader to assume.
    Scoped to this one file on purpose: the working tree is shared with other
    agents and a global dirty check would report their work as ours.
    """
    try:
        rel = str(Path(__file__).resolve().relative_to(REPO)).replace("\\", "/")
        out = subprocess.run(["git", "status", "--porcelain", "--", rel],
                             cwd=REPO, capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


# ── assembly ─────────────────────────────────────────────────────────────────
def build(data_path: Path = DATA, evidence_dir: Path = EVIDENCE,
          m3_path: Path = M3, subsets: list[str] | None = None) -> dict:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    m3 = m3_native_accuracy(m3_path)

    by_subset: dict[str, list[dict]] = {}
    order: list[str] = []
    for item in data:
        full = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        if not full.startswith(SINGLE_HOP_PREFIX):
            continue
        name = full.replace("factconsolidation_", "")
        if subsets and name not in subsets:
            continue
        by_subset[name] = classify_questions(item)
        order.append(name)

    # runs first: whether a subset gets a point estimate or a bound depends on
    # whether its unique stratum was MEASURED, which only the runs can say.
    runs = []
    for name, path, payload in load_runs(evidence_dir):
        sub = subset_of_run(payload)
        if sub not in by_subset:
            continue
        scored = score_run(payload, by_subset[sub])
        scored["run"] = name
        scored["file"] = str(Path(path).relative_to(REPO)).replace("\\", "/")
        runs.append(scored)

    measured_unique: dict[str, dict] = {}
    for sub in order:
        rows = [r for r in runs if r["subset"] == sub and r["strata"]["unique"]["n"]]
        if not rows:
            continue
        accs = {r["strata"]["unique"]["accuracy"] for r in rows}
        measured_unique[sub] = {
            "accuracy": accs.pop() if len(accs) == 1 else None,
            "accuracy_range": [min(r["strata"]["unique"]["accuracy"] for r in rows),
                               max(r["strata"]["unique"]["accuracy"] for r in rows)],
            "n_runs": len(rows),
            "source": "stage0_results/t4_s2_evidence",
        }

    subset_rows = []
    for name in order:
        records = by_subset[name]
        counts = Counter(r["stratum"] for r in records)
        subset_rows.append({
            "subset": name,
            "n_questions": len(records),
            "counts": {s: counts.get(s, 0) for s in STRATA},
            "indices": {s: [r["index"] for r in records if r["stratum"] == s]
                        for s in STRATA},
            "n_multi_candidate": sum(1 for r in records if r["n_candidates"] > 1),
            "n_conflicted_gold_is_latest":
                sum(1 for r in records
                    if r["stratum"] == "conflicted" and r["gold_is_latest"]),
            "m3_accuracy_native": m3.get(name),
            "conflicted_only_accuracy": conflicted_only_accuracy(
                m3.get(name), {s: counts.get(s, 0) for s in STRATA},
                len(records), measured_unique.get(name)),
        })

    totals = Counter()
    for r in runs:
        for s in STRATA:
            for c in ERROR_CLASSES:
                totals[c] += r["strata"][s]["errors"][c]
    uniq_acc = [r["strata"]["unique"]["accuracy"] for r in runs
                if r["strata"]["unique"]["n"]]
    conf_acc = [r["strata"]["conflicted"]["accuracy"] for r in runs
                if r["strata"]["conflicted"]["n"]]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "producer_uncommitted_at_generation": _producer_uncommitted(),
        "provenance": {
            "dataset_file": str(Path(data_path).relative_to(REPO)).replace("\\", "/"),
            "dataset_sha256": _sha256(data_path),
            "evidence_dir": str(Path(evidence_dir).relative_to(REPO)).replace("\\", "/"),
            "n_run_files": len(runs),
            "fact_parser": "hnav.labeling.conflict_analysis.parse (imported; validated 99.5%+ coverage)",
            "question_to_key_rule": (
                "gold_rule.py's rule (subject-in-question AND "
                "expected-answer-in-key-values) with two audit-forced "
                "refinements: word-boundary subject matching, and "
                "all-candidate collection where a properly-contained subject "
                "loses to a longer one and a residual disagreement is labelled "
                "'ambiguous' instead of first-won. Differs from "
                "counterfactual.map_questions_to_keys on exactly 1 of 400 "
                "questions (sh_262k q26, Arthur Miller), pinned by "
                "hnav/tests/test_question_strata.py"),
            "grader": "hnav.labeling.counterfactual.substring_exact_match (recomputed, cross-checked against each run's own field)",
            "m3_file": str(Path(m3_path).relative_to(REPO)).replace("\\", "/")
                       if Path(m3_path).exists() else None,
            "m3_caveat": M3_CAVEAT,
            "embedding_independence": (
                "no embedder, retriever or vector is used anywhere in this "
                "analysis, so the result is unaffected by the 512-token "
                "truncation defect and does not move when that is re-fit"),
        },
        "definitions": {
            "unique": "queried (relation, subject) key has exactly one distinct value in the context",
            "conflicted": "queried key has >= 2 distinct values",
            "ambiguous": "several keys qualify and disagree about conflictedness; not assigned",
            "unmatched": "no key with the question's subject carries the expected answer",
            "stale_value": "incorrect output contains a non-expected value OF THE SAME KEY",
            "off_list": "incorrect output contains no value of the queried key",
            "empty": "output is empty after the evaluator's normalization",
            "conflicted_only_accuracy.kind": (
                "'estimate' = the unique-stratum premise is MEASURED for this "
                "subset and a point estimate is defensible; 'bound' = it is not, "
                "so only the assumption-free counting interval is emitted"),
        },
        "subsets": subset_rows,
        "runs": runs,
        "aggregate": {
            "n_runs": len(runs),
            "unique_accuracy_min": min(uniq_acc) if uniq_acc else None,
            "unique_accuracy_max": max(uniq_acc) if uniq_acc else None,
            "conflicted_accuracy_min": min(conf_acc) if conf_acc else None,
            "conflicted_accuracy_max": max(conf_acc) if conf_acc else None,
            "errors_total": {c: totals[c] for c in ERROR_CLASSES},
            "n_grade_disagreements":
                sum(r["grade_check"]["n_disagreements"] for r in runs),
        },
    }


# ── reporting ────────────────────────────────────────────────────────────────
def format_table(payload: dict) -> str:
    lines = ["=" * 92,
             " QUESTION STRATA - does the queried key carry a conflict at all?",
             "=" * 92, "",
             f"{'subset':<10}{'n':>5}{'unique':>9}{'confl':>8}{'ambig':>7}"
             f"{'unmatch':>9}{'gold=latest':>13}{'m3 acc':>9}"
             f"{'conflicted-only':>22}"]
    notes = []
    for row in payload["subsets"]:
        c = row["counts"]
        acc = row["m3_accuracy_native"]
        co = row["conflicted_only_accuracy"]
        if co["kind"] == "estimate":
            cell = "{:.3f} (est)".format(co["estimate"])
        elif co["kind"] == "bound":
            cell = "[{:.3f}, {:.3f}]".format(co["bound"]["lo"], co["bound"]["hi"])
        else:
            cell = "-"
        lines.append(
            f"{row['subset']:<10}{row['n_questions']:>5}{c['unique']:>9}"
            f"{c['conflicted']:>8}{c['ambiguous']:>7}{c['unmatched']:>9}"
            f"{row['n_conflicted_gold_is_latest']:>13}"
            f"{(f'{acc:.3f}' if acc is not None else '-'):>9}"
            f"{cell:>22}")
        if co.get("bound", {}).get("assumption_refuted"):
            notes.append(
                "   {}: the pooled count REFUTES the 'unique stratum is always "
                "correct' premise here - with {} correct of {}, unique accuracy "
                "is at most {:.3f}. Bound only; no point estimate.".format(
                    row["subset"], co["n_correct_pooled"], row["n_questions"],
                    co["bound"]["unique_accuracy_upper_bound"]))
    if notes:
        lines += [""] + notes
    lines += ["", "   'est' = the unique-stratum premise is measured for that "
                  "subset; brackets = assumption-free counting bound.",
              "   All conflicted-only figures are m3-harness numbers "
              "(m3_caveat in the JSON), not the benchmark's templated query."]

    lines += ["", "-" * 92,
              " COMMITTED RUNS (stage0_results/t4_s2_evidence) - accuracy by stratum,"
              " taxonomy of errors", "-" * 92,
              f"{'run':<16}{'subset':<9}{'unique':>14}{'conflicted':>15}"
              f"{'stale':>8}{'off_list':>10}{'empty':>7}{'grade d':>9}"]
    for r in payload["runs"]:
        u, c = r["strata"]["unique"], r["strata"]["conflicted"]
        errs = Counter()
        for s in STRATA:
            for k, v in r["strata"][s]["errors"].items():
                errs[k] += v
        u_cell = "{}/{}".format(u["correct"], u["n"])
        c_cell = "{}/{}".format(c["correct"], c["n"])
        u_acc = "-" if u["accuracy"] is None else "{:.2f}".format(u["accuracy"])
        c_acc = "-" if c["accuracy"] is None else "{:.2f}".format(c["accuracy"])
        lines.append(
            f"{r['run']:<16}{r['subset']:<9}{u_cell:>9}{u_acc:>5}"
            f"{c_cell:>10}{c_acc:>5}"
            f"{errs['stale_value']:>8}{errs['off_list']:>10}{errs['empty']:>7}"
            f"{r['grade_check']['n_disagreements']:>9}")

    agg = payload["aggregate"]
    if agg["n_runs"]:
        lines += ["", " {} runs. unique accuracy {}-{}, conflicted accuracy "
                      "{:.3f}-{:.3f}".format(
                          agg["n_runs"], agg["unique_accuracy_min"],
                          agg["unique_accuracy_max"],
                          agg["conflicted_accuracy_min"],
                          agg["conflicted_accuracy_max"])]
    lines.append(f" errors across all runs: {agg['errors_total']}   "
                 f"(grade disagreements vs the benchmark's own field: "
                 f"{agg['n_grade_disagreements']})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="*", default=None)
    ap.add_argument("--evidence-dir", default=str(EVIDENCE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    payload = build(evidence_dir=Path(args.evidence_dir), subsets=args.subsets)
    print(format_table(payload))
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
