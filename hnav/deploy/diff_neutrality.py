#!/usr/bin/env python3
"""T4 shadow-neutrality comparator.

    python hnav/deploy/diff_neutrality.py <off_result.json> <shadow_result.json>

Exit 0  = arms are equivalent → shadow mode is neutral.
Exit 1  = arms differ → S2 FIRED. At temperature=0 with a deterministic
          evaluator there is no legitimate source of variation; any difference
          is a bug in the instrumentation, not sampling noise.
Exit 2  = usage / unreadable file.

What is compared: everything in the two result JSONs EXCEPT fields that
legitimately differ between two runs of identical behaviour:

  - the additive "hnav" instrumentation blob (shadow arm only, by design)
  - wall-clock fields (any key containing "time", e.g. time_cost_list,
    memory_construction_time, query_time_len, start_time)
  - the per-arm output paths (output_dir / output_path / agent_save_folder),
    which are deliberately distinct so the arms cannot overwrite each other

Token counts (input_len / output_len) are NOT excluded — the acceptance
criterion says they must be unchanged, so they are part of the comparison.
"""
from __future__ import annotations

import json
import sys

DROP_EXACT = {"hnav", "output_dir", "output_path", "agent_save_folder", "date"}
DROP_SUBSTring = ("time",)  # case-insensitive


def scrub(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            kl = str(k).lower()
            if kl in DROP_EXACT or any(s in kl for s in DROP_SUBSTring):
                continue
            out[k] = scrub(v)
        return out
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def walk_diff(a, b, path="$", out=None, limit=20):
    """Collect up to `limit` leaf-level differences with their JSON paths."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in shadow")
            elif k not in b:
                out.append(f"{path}.{k}: only in off")
            else:
                walk_diff(a[k], b[k], f"{path}.{k}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} (off) vs {len(b)} (shadow)")
        for i, (x, y) in enumerate(zip(a, b)):
            walk_diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                break
    elif a != b:
        ra, rb = repr(a), repr(b)
        out.append(f"{path}:\n      off    = {ra[:160]}\n      shadow = {rb[:160]}")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        off = json.load(open(sys.argv[1], encoding="utf-8"))
        sha = json.load(open(sys.argv[2], encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"cannot read inputs: {e}")
        return 2

    off_s, sha_s = scrub(off), scrub(sha)
    if off_s == sha_s:
        print("NEUTRAL: arms identical after excluding hnav/timing/path fields.")
        print("(token counts and per-question metrics were part of the comparison)")
        return 0

    print("S2 FIRED — arms differ. First differences:")
    for d in walk_diff(off_s, sha_s):
        print(f"   {d}")
    print("\nAt temperature=0 with a deterministic evaluator there is no legitimate")
    print("source of variation. This is a bug in the instrumentation. HARD STOP.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
