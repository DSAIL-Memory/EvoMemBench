#!/usr/bin/env python3
"""Per-model preflight gate for the multi-model campaign.  [E2E-5]

Runs BEFORE any of the 4,500 completions a model costs, and refuses the model
rather than let it produce a number that looks valid and is not.

Why each check exists — every one of these is a way a substituted answering
model silently ruins the experiment:

1. ``served_name``   vLLM advertises whatever ``--served-model-name`` says. Get
                     it wrong and the artifacts carry the *reference* model's
                     name; nothing downstream would ever notice.
2. ``prompt_fits``   The frozen harness sends the benchmark's own RAG prompt —
                     up to 169,810 characters on sh_64k. Tokens per character
                     differ per tokenizer, so "42.5k tokens" is a Qwen fact,
                     not a universal one. This measures the real longest prompt
                     with THIS model's tokenizer. A prompt over the served
                     context aborts mid-campaign with an API error.
3. ``generates``     The longest prompt, through the exact harness call.
4. ``no_reasoning``  ``GENERATION_MAX_TOKENS = 10`` (the benchmark's own
                     ``generation_max_length``). A thinking model spends all ten
                     tokens thinking and scores ~0 — a model-capability claim
                     that is really a serving-mode artifact. Serve such models
                     non-thinking, or exclude them; never report the 0.
5. ``deterministic`` The same prompt twice must give the same string. The
                     campaign's A/A floor (native vs native_repeat) must be 0
                     discordant or the run is void; catching non-determinism
                     here costs one call instead of 1,500.
6. ``short_prompt``  An sh_6k question end-to-end, so a failure that only shows
                     on small contexts is caught too.
7. ``one_shot``      No results folder for this model may exist yet.
8. ``no_degenerate`` A model whose numerics are broken still emits fluent,
                     non-empty, deterministic, reasoning-free text. gemma-3-4b
                     under ``--kv-cache-dtype fp8`` answered "United States of
                     United States of United States of United" and passed
                     checks 1-7 unharmed. Repetition is the signature.
9. ``answer_sanity`` The same failure also shows as an accuracy floor: on the
                     UNIQUE stratum - single-fact retrieval, no conflict,
                     nothing for H-Nav to do - a working model of this class
                     scores near ceiling (Phi-4-mini: 26/26). The fp8 gemma-3
                     scored 4/26. This probes ten unique-stratum questions and
                     requires 3. It is an instrument-health tripwire, not a
                     capability gate: a genuinely weak model can be run anyway
                     with HNAV_PREFLIGHT_ALLOW_LOW_ACCURACY=1, which records
                     that the override was used.

Tool-call formatting is deliberately NOT checked: this benchmark is plain
question answering, the harness never sends a ``tools`` array, and no tool-call
parser is enabled for any model. Enabling one could only add a parse failure
mode. That is recorded in the preflight record rather than left implicit.

Usage:
    python hnav/deploy/preflight_model.py --model-key phi4_mini \
        --model-path /mnt/nvmes/nvme1/models/Phi-4-mini-instruct \
        --served-name microsoft/Phi-4-mini-instruct \
        --base-url http://localhost:8003/v1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import types

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.config import get_config                                     # noqa: E402
from hnav.stage1.calibrate_read_policy import (GENERATION_MAX_TOKENS,  # noqa: E402
                                               SYSTEM_MESSAGE)

ARMS_DIRS = ("hnav_raw", "hnav_idonly", "hnav_geo")
# The reasoning markers a chat model may emit into `content` when it is served
# in thinking mode. Any of them means the ten-token budget is being spent on
# reasoning, not on the answer.
# Tag-shaped reasoning, plus the PROSE openings a thinking model emits when it
# has no tags. The measured Qwen3.5 control (thinking left on) answered
# "Thinking Process:\n\n1.  **Analyze" - no tag, empty reasoning_content, so
# the marker list below passed it. It was the `answer_sanity` FLOOR that caught
# it, at 0/10. Keep that in mind before trusting this list: markers are a
# convenience, the accuracy floor is the guard that actually holds.
THINK_MARKERS = ("<think>", "</think>", "<reasoning>", "<|thinking|>",
                 "<seed:think>", "◁think▷",
                 "Thinking Process", "Thought Process", "Let me think",
                 "Let's think", "First, I need to", "Step 1:")

# Instrument-health thresholds. Set after watching gemma-3-4b fail under fp8 KV
# while passing every other check; they are deliberately far from any plausible
# model's real behaviour, so they separate a broken server from a weak model
# rather than ranking models.
VOID_MARK = "_VOID"     # in a results dir name: preserved evidence, not a shot
SANITY_N = 10           # unique-stratum questions probed
SANITY_FLOOR = 3        # a working model of this class scores 9-10
REPEAT_NGRAM = 2        # an n-gram repeated this many times is degenerate
REPEAT_LIMIT = 3


def degenerate(text: str) -> list[str]:
    """N-grams repeated to the point of looping. Ten output tokens is not much
    room, so a 2-gram appearing three times is already pathological."""
    w = text.lower().split()
    out = []
    for i in range(len(w) - REPEAT_NGRAM + 1):
        g = tuple(w[i:i + REPEAT_NGRAM])
        n = sum(1 for j in range(len(w) - REPEAT_NGRAM + 1)
                if tuple(w[j:j + REPEAT_NGRAM]) == g)
        if n >= REPEAT_LIMIT and " ".join(g) not in out:
            out.append(" ".join(g))
    return out


def _plan(subset: str) -> dict:
    """The REAL plan for one subset, built through the campaign's own code —
    not a reimplementation. Uses the parser cell, whose native prompts are the
    longest arm and are operating-point independent."""
    from hnav.stage1.detector_gap import (ReplayNLI, decide_all,  # noqa: PLC0415
                                          frozen_cell, load_context,
                                          plan_subset)
    cfg = get_config()
    args = types.SimpleNamespace(
        page_source="benchmark", geometry_space="raw", prepass_tag="",
        allow_unstamped_prepass=False, _whitener=None, _ces=None,
        harness="retrieval", max_questions=None, pair_screen="parser",
        operating_point=None, subsets=[subset])
    ctx = load_context(cfg, [subset], args)
    cell = frozen_cell("raw", "parser")
    pp, table, recs = (ctx["prepasses"][subset], ctx["tables"][subset],
                       ctx["records"][subset])
    decisions = decide_all(pp, recs, [cell], ReplayNLI(pp["nli_table"]), None)
    return plan_subset(ctx["items"][subset], subset, pp, decisions, 0, table,
                       None, harness="retrieval", page_source="benchmark")


def longest_prompt(plan: dict) -> tuple[str, int]:
    best, n = "", 0
    for q in plan["questions"]:
        for arm in q["arms"].values():
            p = arm["prompt"]
            if len(p) > n:
                best, n = p, len(p)
    return best, n


# A prompt cannot plausibly tokenize to fewer than one token per 8 characters
# of English. The first version of this function returned 2 for a 169,810-char
# prompt, because transformers v5's apply_chat_template(tokenize=True) returns
# a BatchEncoding and len() counted its KEYS. That would have sized every
# server's context window from a number three orders of magnitude too small, and
# nothing downstream would have caught it until the campaign died mid-run.
MIN_CHARS_PER_TOKEN = 8.0


def count_tokens(model_path: str, prompt: str) -> dict:
    """Tokens for the full chat-formatted request, with this model's own
    tokenizer AND its own chat template.

    Renders the template to text and then encodes it, rather than asking for
    ids directly: that is what the server does, and it is the one form whose
    return type has not changed across transformers versions.
    """
    from transformers import AutoTokenizer                    # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]

    def encode(text: str) -> int:
        # the rendered template already carries its own control tokens as text
        return len(tok.encode(text, add_special_tokens=False))

    try:
        out = {"n_tokens": encode(tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)),
            "method": "chat_template", "exact": True}
    except Exception as exc:                                  # noqa: BLE001
        # Some templates reject a system turn; merging it into the user turn is
        # what those templates do internally anyway, so the count stays honest.
        try:
            out = {"n_tokens": encode(tok.apply_chat_template(
                [{"role": "user", "content": SYSTEM_MESSAGE + "\n\n" + prompt}],
                add_generation_prompt=True, tokenize=False)),
                "method": "chat_template_merged_system", "exact": True,
                "note": str(exc)[:200]}
        except Exception as exc2:                             # noqa: BLE001
            out = {"n_tokens": encode(SYSTEM_MESSAGE + "\n\n" + prompt) + 64,
                   "method": "raw_encode_plus_64", "exact": False,
                   "note": f"{str(exc)[:120]} | {str(exc2)[:120]}"}

    ratio = len(prompt) / max(out["n_tokens"], 1)
    if ratio > MIN_CHARS_PER_TOKEN:
        raise RuntimeError(
            f"implausible token count for {model_path}: {out['n_tokens']} "
            f"tokens for {len(prompt):,} chars ({ratio:.1f} chars/token). The "
            f"tokenizer call returned something that is not a token sequence.")
    return out


def ask(client, model: str, prompt: str) -> dict:
    """The EXACT call the harness makes — same messages, temperature and
    max_tokens. Anything else here would preflight a different experiment."""
    cfg = get_config()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                  {"role": "user", "content": prompt}],
        temperature=cfg.llm_temperature, max_tokens=GENERATION_MAX_TOKENS)
    msg = resp.choices[0].message
    return {"content": msg.content or "",
            "reasoning_content": getattr(msg, "reasoning_content", None) or "",
            "finish_reason": resp.choices[0].finish_reason,
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "latency_s": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--base-url", default="http://localhost:8003/v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-token-count", action="store_true",
                    help="advisory only; the served endpoint still reports its "
                         "own prompt_tokens for the longest prompt")
    ap.add_argument("--probe-only", action="store_true",
                    help="serving-health probe only (served name, generation, "
                         "repetition, sh_6k accuracy floor). Skips the sh_64k "
                         "plan build and the one-shot check, so it can be run "
                         "repeatedly against a model whose campaign already "
                         "exists - for diagnosing a serving configuration.")
    args = ap.parse_args(argv)

    cfg = get_config()
    rec: dict = {"model_key": args.model_key, "model_path": args.model_path,
                 "served_name": args.served_name, "base_url": args.base_url,
                 "harness": {"system_message": SYSTEM_MESSAGE,
                             "max_tokens": GENERATION_MAX_TOKENS,
                             "temperature": cfg.llm_temperature,
                             "sends_tools": False,
                             "tool_parser_enabled": False,
                             "tool_check": "N/A - this benchmark is plain QA; "
                                           "the harness never sends a tools "
                                           "array, so no tool-call parser is "
                                           "enabled for any model"},
                 "checks": {}}
    checks = rec["checks"]

    def gate(name, ok, **extra):
        checks[name] = dict(ok=bool(ok), **extra)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}"
              + ("  " + json.dumps(extra, default=str)[:220] if extra else ""))
        return bool(ok)

    print(f"== preflight {args.model_key} ==")

    # ── 7. one shot ──────────────────────────────────────────────────────────
    # Match on the tag the RUNNER builds from the served name, not on
    # model_key: 'gemma3_4b' is not a substring of
    # 'google_gemma-3-4b-it_2026-08-30', so keying on model_key made this
    # guard incapable of firing - decoration, not a check.
    tag_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", args.served_name).strip("_")
    existing, voided = [], []
    if not args.probe_only:
        for arm in ARMS_DIRS:
            res = REPO / "pipelines" / arm / "results"
            if not res.is_dir():
                continue
            for p in res.iterdir():
                if not (p.is_dir() and any(p.glob("detector_gap_*.json"))
                        and p.name.startswith(tag_stem)):
                    continue
                # A run VOIDED for a documented instrument defect is not a spent
                # shot at a valid cell - it is preserved evidence that the
                # instrument was broken, and re-running is the correct response.
                # The marker must be EXPLICIT and in the directory name, so a
                # real measured result can never be skipped silently: renaming a
                # results folder is a deliberate act, not something this script
                # can do to itself. (gemma-3's fp8 run is the case in point.)
                (voided if VOID_MARK in p.name.upper() else existing).append(
                    str(p.relative_to(REPO).as_posix()))
        gate("one_shot", not existing, tag_stem=tag_stem, existing=existing,
             ignored_voided=voided)

    # ── 1. the served name ───────────────────────────────────────────────────
    from openai import OpenAI                                 # noqa: PLC0415
    client = OpenAI(base_url=args.base_url, api_key=cfg.llm_api_key)
    try:
        models = client.models.list()
        served = [m.id for m in models.data]
        ctx_len = next((getattr(m, "max_model_len", None) for m in models.data
                        if m.id == args.served_name), None)
    except Exception as exc:                                  # noqa: BLE001
        return 1 if not gate("served_name", False, error=str(exc)[:300]) else 0
    gate("served_name", args.served_name in served, served=served,
         served_max_model_len=ctx_len)

    # ── 2. does the longest real prompt fit ──────────────────────────────────
    a = None
    if not args.probe_only:
        plan = _plan("sh_64k")
        prompt64, chars = longest_prompt(plan)
        tok = ({"n_tokens": None, "method": "skipped", "exact": False}
               if args.skip_token_count
               else count_tokens(args.model_path, prompt64))
        budget_ok = (ctx_len is None or tok["n_tokens"] is None
                     or tok["n_tokens"] + GENERATION_MAX_TOKENS <= ctx_len)
        gate("prompt_fits", budget_ok, longest_prompt_chars=chars,
             tokens=tok, served_max_model_len=ctx_len,
             headroom=(None if (ctx_len is None or tok["n_tokens"] is None)
                       else ctx_len - tok["n_tokens"] - GENERATION_MAX_TOKENS))

        # ── 3/4/5. the longest prompt, through the harness call, twice ───────
        try:
            a = ask(client, args.served_name, prompt64)
            b = ask(client, args.served_name, prompt64)
        except Exception as exc:                              # noqa: BLE001
            gate("generates", False, error=str(exc)[:400])
            _write(rec, args)
            return 1
        gate("generates", bool(a["content"].strip()), output=a["content"][:120],
             finish_reason=a["finish_reason"], latency_s=a["latency_s"],
             server_prompt_tokens=a["prompt_tokens"])
        leaked = [m for m in THINK_MARKERS if m in a["content"]]
        gate("no_reasoning", not leaked and not a["reasoning_content"],
             markers=leaked, reasoning_content=a["reasoning_content"][:120])
        gate("deterministic", a["content"] == b["content"],
             first=a["content"][:80], second=b["content"][:80])

    # ── 6. a short prompt too ────────────────────────────────────────────────
    small = _plan("sh_6k")
    q0 = small["questions"][0]
    s = ask(client, args.served_name, q0["arms"]["native"]["prompt"])
    gate("short_prompt", bool(s["content"].strip()),
         output=s["content"][:120], truths=q0["truths"],
         latency_s=s["latency_s"])

    # ── 8. degenerate generation ─────────────────────────────────────────────
    loops = {"short": degenerate(s["content"])}
    if a is not None:
        loops["longest"] = degenerate(a["content"])
    gate("no_degenerate", not any(loops.values()), repeats=loops)

    # ── 4b. reasoning leakage, ALSO in probe-only mode ───────────────────────
    # The full preflight checks this on the longest prompt, inside the block
    # --probe-only skips. That left the probe unable to see reasoning leakage -
    # exactly what a probe of a thinking-by-default model exists to catch.
    if args.probe_only:
        leaked_s = [m for m in THINK_MARKERS if m in s["content"]]
        gate("no_reasoning", not leaked_s and not s["reasoning_content"],
             markers=leaked_s, reasoning_content=s["reasoning_content"][:120])

    # ── 9. the accuracy floor on questions with no conflict in them ──────────
    from hnav.labeling.counterfactual import substring_exact_match  # noqa: PLC0415
    probe = [q for q in small["questions"] if q["stratum"] == "unique"][:SANITY_N]
    hits, sample = 0, []
    for q in probe:
        r = ask(client, args.served_name, q["arms"]["native"]["prompt"])
        ok = any(substring_exact_match(r["content"], t) for t in q["truths"])
        hits += bool(ok)
        sample.append({"index": q["index"], "out": r["content"][:40],
                       "truth": q["truths"][0], "ok": bool(ok)})
    override = os.environ.get("HNAV_PREFLIGHT_ALLOW_LOW_ACCURACY") == "1"
    rec["sanity_probe"] = sample
    gate("answer_sanity", hits >= SANITY_FLOOR or override,
         correct=hits, n=len(probe), floor=SANITY_FLOOR,
         override_used=bool(override and hits < SANITY_FLOOR))

    rec["ok"] = all(c["ok"] for c in checks.values())
    _write(rec, args)
    print(f"\n  PREFLIGHT {'PASSED' if rec['ok'] else 'FAILED'} "
          f"for {args.model_key}")
    return 0 if rec["ok"] else 1


def _write(rec: dict, args) -> None:
    name = "probe.json" if getattr(args, "probe_only", False) else "preflight.json"
    out = pathlib.Path(args.out) if args.out else (
        get_config().out_dir / "campaign" / args.model_key / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8",
                   newline="\n")
    print(f"  written: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
