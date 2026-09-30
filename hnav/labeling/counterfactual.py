"""Counterfactual replay and outcome labeling.  [T6]

**Offline only.** This module reads the benchmark's questions and its answer
list, and it is the single module in the package that is allowed to.
``hnav/tests/test_leakage_audit.py`` hard-bars importing it from anywhere under
``hnav/core/`` or ``hnav/adapters/``.

Why replay is affordable here — the methodological advantage over BFCL:

* the store is append-only and the retriever is a pure function of the store, so
  ``S_without`` is ``S_with`` minus one row of the bank matrix. The memorize
  phase never re-runs.
* grading is ``substring_exact_match``: deterministic, offline, free.

So one counterfactual costs exactly **one answer call** and nothing else.

Every external dependency is injected: ``answer_fn(prompt) -> str`` and
``grade_fn(prediction, truths) -> bool``. Tests pass stubs; no test in this
repository opens a socket.
"""
from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .conflict_analysis import parse as parse_fact

__all__ = ["normalize_answer", "substring_exact_match", "map_questions_to_keys",
           "CounterfactualLabeler", "WriteOutcome", "ReadOutcome",
           "build_prompt", "local_answer_fn", "CLASSES"]

CLASSES = ("must_write", "must_suppress", "may_suppress", "inert_superseded",
           "uncertain")


# ── grading: transcribed from the benchmark's own evaluator ──────────────────
def normalize_answer(answer_text: str) -> str:
    """Verbatim transcription of ``utils/eval_other_utils.py:28``.

    Copied rather than imported so that Stage 0 does not depend on the benchmark
    package's import side effects. ``hnav/tests/test_counterfactual.py`` pins the
    behaviour on fixtures; if the benchmark's evaluator ever changes, that test
    is where the divergence shows up.
    """
    text = answer_text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def substring_exact_match(prediction: str, truths: Sequence[str] | str) -> bool:
    """``max`` over ground truths of "normalized truth is a substring of
    normalized prediction" — ``substring_exact_match_score`` at
    ``eval_other_utils.py:101``, wrapped by ``drqa_metric_max_over_ground_truths``.
    """
    if isinstance(truths, str):
        truths = [truths]
    flat: list[str] = []
    for t in truths:
        flat.extend(t if isinstance(t, (list, tuple)) else [t])
    pred = normalize_answer(prediction or "")
    return any(normalize_answer(str(t)) in pred for t in flat)


# ── question -> key mapping (offline) ────────────────────────────────────────
def map_questions_to_keys(item: dict) -> list[dict]:
    """Map each single-hop question to the ``(relation, subject)`` key it turns on.

    Same rule as the committed ``gold_rule.py``: the key whose subject appears in
    the question and whose object set intersects the expected answer. Requires
    the expected answer, which is why this is offline and why nothing that runs
    online may import it.

    Returns one record per question with ``key``, ``target_serial`` (the member
    whose object matches, highest serial first) and ``conflicted``.
    """
    facts = re.findall(r"^\s*(\d+)\.\s+(.*)$", item.get("context", ""), re.M)
    groups: dict[tuple, list] = defaultdict(list)
    for num, txt in facts:
        p = parse_fact(txt)
        if p:
            rel, subj, obj = p
            groups[(rel, subj)].append((int(num), txt, obj))

    out = []
    for qi, (question, truths) in enumerate(zip(item.get("questions", []),
                                                item.get("answers", []))):
        expected = {str(a).lower() for a in
                    (truths if isinstance(truths, (list, tuple)) else [truths])}
        hit = None
        for key, members in groups.items():
            if key[1].lower() in question.lower() and \
                    any(o.lower() in expected for _, _, o in members):
                hit = (key, members)
                break
        record = {"index": qi, "question": question, "truths": sorted(expected),
                  "key": None, "target_serial": None, "conflicted": False,
                  "member_serials": []}
        if hit is not None:
            key, members = hit
            members = sorted(members)
            matching = [n for n, _, o in members if o.lower() in expected]
            record.update({
                "key": key,
                "target_serial": max(matching) if matching else None,
                "conflicted": len({o for _, _, o in members}) > 1,
                "member_serials": [n for n, _, _ in members],
            })
        out.append(record)
    return out


# ── prompting ────────────────────────────────────────────────────────────────
PROMPT = ("Answer the question using only the facts below. "
          "Reply with the answer alone, no explanation.\n\n"
          "Facts:\n{context}\n\nQuestion: {question}\nAnswer:")


def build_prompt(question: str, retrieved: Iterable[str]) -> str:
    """H-Nav's own counterfactual harness prompt.

    Deliberately not the benchmark's answering template. A counterfactual
    isolates the *retrieval* bottleneck by holding the model and the prompt
    fixed and varying only the retrieved set, so what matters is that the two
    arms are identical apart from the retrieved facts — not that the absolute
    accuracy matches a full benchmark run. Any comparison to benchmark accuracy
    must say which harness produced it.
    """
    return PROMPT.format(context="\n".join(retrieved), question=question)


def local_answer_fn(cfg, max_tokens: int | None = None) -> Callable[[str], str]:
    """An ``answer_fn`` backed by the local vLLM endpoint. No API key, t=0.

    The ``openai`` client is imported lazily, so this module still imports on a
    machine that has neither the package nor a network.
    """
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def answer(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.llm_temperature,
            max_tokens=max_tokens or cfg.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""

    return answer


# ── outcomes ─────────────────────────────────────────────────────────────────
@dataclass
class WriteOutcome:
    candidate_id: str
    counterfactual_class: str
    n_affected: int = 0
    correct_with: list[bool] = field(default_factory=list)
    correct_without: list[bool] = field(default_factory=list)
    question_indices: list[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"candidate_id": self.candidate_id,
                "counterfactual_class": self.counterfactual_class,
                "n_affected": self.n_affected,
                "correct_with": self.correct_with,
                "correct_without": self.correct_without,
                "question_indices": self.question_indices,
                "note": self.note}


@dataclass
class ReadOutcome:
    question_index: int
    correct_native: bool
    correct_repaired: bool
    n_injected: int = 0

    @property
    def repair_helped(self) -> bool:
        return self.correct_repaired and not self.correct_native

    @property
    def repair_harmed(self) -> bool:
        return self.correct_native and not self.correct_repaired

    def to_dict(self) -> dict:
        return {"question_index": self.question_index,
                "correct_native": self.correct_native,
                "correct_repaired": self.correct_repaired,
                "n_injected": self.n_injected,
                "repair_helped": self.repair_helped,
                "repair_harmed": self.repair_harmed}


class CounterfactualLabeler:
    """Replays retrieval with and without a single record and grades both.

    ``answer_fn`` and ``grade_fn`` are injected; ``cache`` memoizes on the exact
    prompt string, so the ``S_with`` arm of one candidate and the ``S_with`` arm
    of the next never pay twice for the same question.
    """

    def __init__(self, replica, answer_fn: Callable[[str], str],
                 grade_fn: Callable[[str, Sequence[str]], bool] = substring_exact_match,
                 top_k: int = 10, prompt_fn: Callable[[str, Iterable[str]], str] = build_prompt,
                 ) -> None:
        self.replica = replica
        self.answer_fn = answer_fn
        self.grade_fn = grade_fn
        self.top_k = top_k
        self.prompt_fn = prompt_fn
        self.cache: dict[str, str] = {}
        self.n_calls = 0

    # ── one graded answer ────────────────────────────────────────────────────
    def answer_and_grade(self, question: str, retrieved: Sequence[str],
                         truths: Sequence[str]) -> bool:
        prompt = self.prompt_fn(question, retrieved)
        if prompt not in self.cache:
            self.cache[prompt] = self.answer_fn(prompt)
            self.n_calls += 1
        return bool(self.grade_fn(self.cache[prompt], truths))

    def retrieve_texts(self, store, question: str) -> list[str]:
        view = self.replica.rank(store, question, top_k=self.top_k)
        by_id = {r.id: r.text for r in store.records}
        return [by_id[i] for i in view.top_ids]

    # ── write side ───────────────────────────────────────────────────────────
    def label_write(self, store, record_id: str, affected: Sequence[dict],
                    is_newest_for_key: bool = True) -> WriteOutcome:
        """``S_with`` = the store; ``S_without`` = the store minus ``record_id``.

        Classes follow plan §7.2 exactly:
          must_write        ∃q: correct_with ∧ ¬correct_without
          must_suppress     ∃q: ¬correct_with ∧ correct_without
          may_suppress      ∀q equal, and the record is not the newest for its key
          inert_superseded  superseded and no question maps to it
          uncertain         no affected question
        """
        if not affected:
            cls = "inert_superseded" if not is_newest_for_key else "uncertain"
            return WriteOutcome(record_id, cls, note="no affected question")

        without = store.without(record_id)
        with_ok, without_ok, idxs = [], [], []
        for q in affected:
            truths = q["truths"]
            with_ok.append(self.answer_and_grade(q["question"],
                                                 self.retrieve_texts(store, q["question"]),
                                                 truths))
            without_ok.append(self.answer_and_grade(q["question"],
                                                    self.retrieve_texts(without, q["question"]),
                                                    truths))
            idxs.append(q["index"])

        if any(w and not wo for w, wo in zip(with_ok, without_ok)):
            cls = "must_write"
        elif any(wo and not w for w, wo in zip(with_ok, without_ok)):
            cls = "must_suppress"
        elif not is_newest_for_key:
            cls = "may_suppress"
        else:
            cls = "uncertain"

        return WriteOutcome(record_id, cls, n_affected=len(affected),
                            correct_with=with_ok, correct_without=without_ok,
                            question_indices=idxs)

    # ── read side ────────────────────────────────────────────────────────────
    def label_read(self, question: dict, native_texts: Sequence[str],
                   injected_texts: Sequence[str]) -> ReadOutcome:
        """Grade the native top-k against the same top-k with the missing
        superseding facts injected. No store variant is needed, so this isolates
        the retrieval bottleneck exactly, holding model and store fixed."""
        repaired = list(native_texts) + [t for t in injected_texts
                                         if t not in set(native_texts)]
        return ReadOutcome(
            question_index=question["index"],
            correct_native=self.answer_and_grade(question["question"], native_texts,
                                                 question["truths"]),
            correct_repaired=self.answer_and_grade(question["question"], repaired,
                                                   question["truths"]),
            n_injected=len(repaired) - len(native_texts),
        )
