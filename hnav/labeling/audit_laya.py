#!/usr/bin/env python3
"""Second reader over the cos>=0.80 conflict candidate set: Laya.  [offline]

The gold conflict dataset carries one machine judge (``openai/gpt-5-mini``,
``audit_runner.py``). This module scores the **same** 87,102 candidates
(``stage0_results/conflict_pairs/audit_candidates_cos080.jsonl.gz``) with
Laya, a non-generative encoder classifier (``convaiinnovations/laya``,
ModernBERT-large backbone, Apache-2.0). It answers seven ``choice``
questions per pair in one forward pass; the keys mirror the judge's
``VERDICT_FIELDS`` so the two readers can be compared flag by flag
(``audit_compare_laya.py``) and their disagreements handed to humans
(``audit_human_review.py``).

What this pass is and is not:
  * a second, architecturally different reader — its agreement with the
    judge is reported, its disagreements are reviewed by people;
  * NOT a relabelling. The gold dataset and its frozen tier counts are not
    touched. Laya is not fine-tuned and no temperature is refitted on the
    judge's labels: either would make the agreement circular.

Design choices (2026-09-30):
  * every pair is scored in BOTH input orders (``ab`` and ``ba``); the judge
    used seeded order randomisation, Laya is order-sensitive, and the flip
    rate is itself a reported quantity;
  * ``choice`` questions with neutral option keys ``A``/``B`` — the model card
    documents a label-following bias in the ``noul`` primitive; ``A`` maps to
    True for every flag;
  * English root checkpoint; the longest pair is 207 characters, well inside
    the 512-token budget; ``head_max_len`` is per question (two options).

Runbook (GPU box; this container has no GPU and cannot reach Hugging Face):
    source .venv-hnav/bin/activate
    pip install laya                                   # torch already present
    python hnav/labeling/audit_laya.py --dry-run       # no model, no network
    USE_TF=0 python hnav/labeling/audit_laya.py --device cuda   # ~1-3 h on a T4
    python hnav/labeling/audit_laya.py --compress      # -> audit_results_laya.jsonl.gz
then commit ``stage0_results/conflict_pairs/audit_results_laya.jsonl.gz``
(``Audit:`` prefix). Results are append-only JSONL keyed by (pair_id, order);
a restart skips completed rows.

Heavy imports (``laya``, ``torch``) live inside ``load_agent`` so the module
imports on a machine without them (``test_no_torch_at_import.py``).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.labeling.audit_runner import priority_order  # noqa: E402  (stdlib only)

DIR = REPO / "stage0_results" / "conflict_pairs"
CANDIDATES = DIR / "audit_candidates_cos080.jsonl"
RESULTS = DIR / "audit_results_laya.jsonl"

CHECKPOINT = "convaiinnovations/laya"
ORDERS = ("ab", "ba")
TRUE_KEY = "A"           # option key that maps to True for every flag
FLAGS = ["same_referent", "same_relation", "context_overlap",
         "values_incompatible", "relation_allows_multiple_values",
         "strict_conflict", "update_conflict"]

# ── the seven questions: definitions paraphrase the judge's system prompt ────
# (audit_runner.SYSTEM_PROMPT, 2026-08-24). Option A is always the True side.
QUESTIONS = {
    "same_referent": {
        "type": "choice",
        "instructions": ("Are the two sentences about the same real-world entity? "
                         "Different surface strings, aliases and abbreviations can "
                         "name the same entity."),
        "criteria": {"A": "both sentences are about the same entity",
                     "B": "the sentences are about different entities"},
    },
    "same_relation": {
        "type": "choice",
        "instructions": ("Do both sentences assert the same semantic attribute or "
                         "relation of their subject, even if it is phrased differently?"),
        "criteria": {"A": "the same attribute or relation",
                     "B": "different attributes or relations"},
    },
    "context_overlap": {
        "type": "choice",
        "instructions": ("Do the two propositions apply under compatible or overlapping "
                         "validity conditions? Different times, places, populations, "
                         "organizations, conditions or modalities can make similar "
                         "propositions non-conflicting."),
        "criteria": {"A": "compatible or overlapping conditions",
                     "B": "different conditions, so both could hold"},
    },
    "values_incompatible": {
        "type": "choice",
        "instructions": ("Can the two asserted values both occupy the same semantic slot "
                         "under the same context? Paraphrases, aliases, one value that "
                         "contains the other, and values that can coexist are compatible. "
                         "Do not treat values as incompatible merely because the strings differ."),
        "criteria": {"A": "the values cannot both occupy the slot",
                     "B": "the values are compatible"},
    },
    "relation_allows_multiple_values": {
        "type": "choice",
        "instructions": ("Does the relation naturally permit several simultaneous values, "
                         "such as citizenships, children, co-founders, languages spoken "
                         "or memberships?"),
        "criteria": {"A": "naturally multi-valued",
                     "B": "one value at a time"},
    },
    "strict_conflict": {
        "type": "choice",
        "instructions": "Can both sentences be true at the same time in the real world?",
        "criteria": {"A": "no, they cannot both be true",
                     "B": "yes, both can be true"},
    },
    "update_conflict": {
        "type": "choice",
        "instructions": ("A memory store keeps one current value per (entity, attribute) "
                         "slot, and a new incompatible value replaces the previous one. "
                         "Under that convention, does one sentence replace the other "
                         "sentence's value for the same slot?"),
        "criteria": {"A": "yes, one sentence replaces the other's value for the same slot",
                     "B": "no, they do not compete for one slot, or the values are compatible"},
    },
}
assert list(QUESTIONS) == FLAGS


# ── I/O ──────────────────────────────────────────────────────────────────────

def _open_text(path: pathlib.Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else open(path, "rt", encoding="utf-8")


def resolve_candidates(path: pathlib.Path = CANDIDATES) -> pathlib.Path:
    """The plain .jsonl is gitignored; fall back to the committed .gz."""
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    raise SystemExit(f"candidate file missing: {path} (or .gz)")


def load_candidates(path: pathlib.Path = CANDIDATES) -> list[dict]:
    with _open_text(resolve_candidates(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: pathlib.Path) -> set[tuple[str, str]]:
    """(pair_id, order) keys already written — the resume set."""
    done: set[tuple[str, str]] = set()
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.exists():
            with _open_text(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        done.add((r["pair_id"], r["order"]))
    return done


# ── state construction ───────────────────────────────────────────────────────

def build_state(rec: dict, order: str) -> str:
    a, b = ((rec["fact_a"], rec["fact_b"]) if order == "ab"
            else (rec["fact_b"], rec["fact_a"]))
    return f"Sentence A: {a}\nSentence B: {b}"


def work_items(records: list[dict], orders=ORDERS,
               done: set[tuple[str, str]] | None = None) -> list[tuple[dict, str]]:
    done = done or set()
    return [(r, o) for r in records for o in orders if (r["pair_id"], o) not in done]


# ── result parsing ───────────────────────────────────────────────────────────

def parse_answers(answers: dict) -> tuple[dict, dict]:
    """Laya's ``answers`` -> (per-question record, boolean verdict).

    ``p_true`` is the probability mass on option A; ``choice`` is Laya's argmax
    label; the verdict flag is ``choice == "A"``. Missing probabilities are
    tolerated (``p_true`` None) so a runtime that omits them still records the
    choice.
    """
    out, verdict = {}, {}
    for q in FLAGS:
        a = answers[q]
        probs = a.get("probabilities") or {}
        p_true = probs.get(TRUE_KEY)
        choice = a.get("choice")
        if choice is None and p_true is not None:
            choice = TRUE_KEY if p_true >= 0.5 else "B"
        out[q] = {"choice": choice,
                  "p_true": None if p_true is None else float(p_true),
                  "confidence": a.get("confidence"),
                  "answer_confidence": a.get("answer_confidence")}
        verdict[q] = choice == TRUE_KEY
    return out, verdict


# ── model (heavy imports stay in here) ───────────────────────────────────────

def snapshot_sha(repo_id: str = CHECKPOINT) -> str | None:
    """Commit hash of the cached Hugging Face snapshot, if resolvable."""
    home = pathlib.Path(os.environ.get("HF_HUB_CACHE")
                        or pathlib.Path(os.environ.get("HF_HOME",
                                                       pathlib.Path.home() / ".cache" / "huggingface")) / "hub")
    ref = home / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def load_agent(device: str | None, head_max_len: int | None, max_len: int | None):
    os.environ.setdefault("USE_TF", "0")       # transformers' TF probe can deadlock
    import laya                                # noqa: WPS433  heavy, deliberate
    import torch                               # noqa: WPS433
    kwargs = {}
    if device:
        kwargs["device"] = device
    agent = laya.load(CHECKPOINT, **kwargs)
    cfg = getattr(agent, "cfg", None)
    if isinstance(cfg, dict):
        if head_max_len:
            cfg["head_max_len"] = int(head_max_len)
        if max_len:
            cfg["max_len"] = int(max_len)
    meta = {
        "reader": "laya",
        "laya_version": getattr(laya, "__version__", None),
        "checkpoint": CHECKPOINT,
        "snapshot_sha": snapshot_sha(),
        "torch_version": torch.__version__,
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "head_max_len": (cfg or {}).get("head_max_len") if isinstance(cfg, dict) else head_max_len,
        "max_len": (cfg or {}).get("max_len") if isinstance(cfg, dict) else max_len,
        "questions_sha256": hashlib.sha256(
            json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest(),
    }
    return agent, meta


def predict_states(agent, states: list[str], batch_size: int) -> list[dict]:
    if hasattr(agent, "predict_batch"):
        return agent.predict_batch(states, QUESTIONS, batch_size=batch_size,
                                   sort_by_length=True)
    return [agent.predict(s, QUESTIONS) for s in states]


# ── main ─────────────────────────────────────────────────────────────────────

def compress(out: pathlib.Path) -> int:
    gz = out.with_suffix(out.suffix + ".gz")
    if not out.exists():
        print(f"nothing to compress: {out} missing", file=sys.stderr)
        return 2
    n_in = 0
    with open(out, "rt", encoding="utf-8") as f, gzip.open(gz, "wt", encoding="utf-8") as g:
        for line in f:
            if line.strip():
                g.write(line if line.endswith("\n") else line + "\n")
                n_in += 1
    with _open_text(gz) as g:
        n_out = sum(1 for line in g if line.strip())
    if n_in != n_out:
        raise SystemExit(f"compress mismatch: {n_in} rows in, {n_out} rows out")
    print(f"wrote {gz} ({n_out:,} rows); the plain .jsonl is gitignored and kept")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: laya's choice)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--orders", default=",".join(ORDERS),
                    help="comma-separated subset of ab,ba (default both)")
    ap.add_argument("--limit", type=int, default=0,
                    help="pilot: first N pairs in audit priority order")
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build states and questions, load nothing")
    ap.add_argument("--compress", action="store_true",
                    help="gzip the results file and exit")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    if args.compress:
        return compress(out)

    orders = tuple(o for o in args.orders.split(",") if o)
    bad = [o for o in orders if o not in ORDERS]
    if bad or not orders:
        raise SystemExit(f"--orders must be a subset of {ORDERS}, got {args.orders!r}")

    records = load_candidates(pathlib.Path(args.candidates))
    ordered = priority_order(records)
    if args.limit:
        ordered = ordered[:args.limit]
    done = load_done(out)
    items = work_items(ordered, orders, done)
    print(f"candidates={len(records):,}  selected={len(ordered):,}  orders={orders}  "
          f"already done={len(done):,}  to score={len(items):,}")

    if args.dry_run:
        chars = [len(build_state(r, o)) for r, o in items[:5000]] or [0]
        print(f"dry-run: state length mean {sum(chars)/len(chars):.0f} chars, "
              f"max {max(chars)} chars (~{max(chars)/3.8:.0f} tokens); "
              f"{len(QUESTIONS)} choice questions, {sum(len(q['criteria']) for q in QUESTIONS.values())} options")
        for r, o in items[:3]:
            print("---", r["pair_id"], o)
            print(build_state(r, o))
        print("--- questions")
        print(json.dumps(QUESTIONS, indent=1, ensure_ascii=False))
        return 0

    if not items:
        print("nothing to do")
        return 0

    agent, meta = load_agent(args.device, args.head_max_len, args.max_len)
    print("model:", json.dumps(meta))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0
    with open(out, "at", encoding="utf-8") as f:
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            states = [build_state(r, o) for r, o in chunk]
            results = predict_states(agent, states, args.batch_size)
            if len(results) != len(chunk):
                raise SystemExit(f"predict returned {len(results)} results for {len(chunk)} states")
            ts = time.time()
            for (r, o), res in zip(chunk, results):
                answers, verdict = parse_answers(res["answers"])
                row = {"pair_id": r["pair_id"], "subset": r["subset"],
                       "parser_tagged_conflict": r["parser_tagged_conflict"],
                       "order": o, "answers": answers, "verdict": verdict,
                       "meta": meta, "ts": ts}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            f.flush()
            if (start // args.batch_size) % 50 == 0:
                el = time.time() - t0
                rate = n_written / el if el else 0.0
                left = (len(items) - n_written) / rate if rate else float("inf")
                print(f"  {n_written:,}/{len(items):,}  {rate:.1f} rows/s  "
                      f"eta {left/60:.0f} min", flush=True)
    print(f"done: {n_written:,} rows appended to {out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
