#!/usr/bin/env python3
"""How long is the benchmark's longest prompt — in EACH model's tokens? [E2E-5]

The runbook's "max prompt ≈ 42.5k tokens" is a *Qwen3* fact: it was measured
with one tokenizer. Serving a different model at 48k on the strength of it is
an assumption, and if it is wrong the campaign dies mid-run with a context
error after hours of completions.

This measures it, once, for every candidate answering model, from the real
sh_64k plan built through the campaign's own code — including each model's own
chat template, since the template's tokens count against the same budget.

    python hnav/deploy/measure_prompt_tokens.py \
        --models /mnt/nvmes/nvme1/models/Phi-4-mini-instruct ...

Writes hnav/_out/campaign/prompt_tokens.json. No GPU, no server, no LLM calls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.config import get_config                                   # noqa: E402
from hnav.stage1.calibrate_read_policy import GENERATION_MAX_TOKENS  # noqa: E402

# hnav/deploy/ is deliberately not a package (it is scripts, and the import
# audit skips it), so the sibling is loaded by path rather than by making one.
_spec = importlib.util.spec_from_file_location(
    "_hnav_preflight", pathlib.Path(__file__).with_name("preflight_model.py"))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)
count_tokens, longest_prompt, _plan = _pf.count_tokens, _pf.longest_prompt, _pf._plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="local weight directories")
    ap.add_argument("--subsets", nargs="+", default=["sh_64k", "sh_32k"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out: dict = {"generation_max_tokens": GENERATION_MAX_TOKENS, "subsets": {}}
    for s in args.subsets:
        plan = _plan(s)
        prompt, chars = longest_prompt(plan)
        row = {"longest_prompt_chars": chars,
               "n_questions": len(plan["questions"]), "models": {}}
        print(f"\n== {s}: longest prompt {chars:,} chars ==")
        for path in args.models:
            name = pathlib.Path(path).name
            try:
                tok = count_tokens(path, prompt)
                tok["chars_per_token"] = round(chars / tok["n_tokens"], 3)
                # what the server must be given, with the answer budget and a
                # 4% margin for the questions this longest one does not cover
                tok["min_max_model_len"] = int(
                    (tok["n_tokens"] + GENERATION_MAX_TOKENS) * 1.04)
            except Exception as exc:                          # noqa: BLE001
                tok = {"error": str(exc)[:300]}
            row["models"][name] = tok
            print(f"   {name:26s} {json.dumps(tok, default=str)[:180]}")
        out["subsets"][s] = row

    dest = pathlib.Path(args.out) if args.out else (
        get_config().out_dir / "campaign" / "prompt_tokens.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8",
                    newline="\n")
    print(f"\n written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
