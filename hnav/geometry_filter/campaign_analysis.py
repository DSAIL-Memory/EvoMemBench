"""Model-agnostic campaign analysis for the multi-model H-Nav runs.

`e2e3_analysis` compares one arm against the *committed Qwen3-4B* records —
right for E2E-3, useless for a new answering model. This module instead
discovers whatever arms were run for a given model tag and reports:

  * per arm, per stratum: native vs detector, exact McNemar (within-model,
    so every comparison is paired on the same questions and the same model);
  * cross-arm pairing on the detector mechanism, with the native-agreement
    check that makes the pairing legitimate;
  * validity: every preregistered run-voiding condition, with condition 2
    (the model-specific native band) reported as a warning, matching
    `pipelines/_shared/runner.py`;
  * the campaign's structural question: which questions each arm alone
    answers, so the parser-only set can be compared across models.

Usage:
    python -m hnav.geometry_filter.campaign_analysis <model-tag> [--subset sh_64k]
    python -m hnav.geometry_filter.campaign_analysis --list
"""
from __future__ import annotations

import argparse
import json
import pathlib

from hnav.stage1.stale_suppression_probe import mcnemar_exact_p

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPELINES = REPO / "pipelines"
MECHANISM = "detector_suppress"
STRATA = ("all", "conflicted", "unique")


def discover(model_tag: str | None = None) -> dict[str, dict[str, pathlib.Path]]:
    """{arm: {subset: artifact path}} for one model tag (or every tag)."""
    out: dict[str, dict[str, pathlib.Path]] = {}
    for arm_dir in sorted(PIPELINES.iterdir()):
        res = arm_dir / "results"
        if not res.is_dir():
            continue
        for tag_dir in sorted(res.iterdir()):
            if not tag_dir.is_dir():
                continue
            if model_tag and tag_dir.name != model_tag:
                continue
            for art in sorted(tag_dir.glob("detector_gap_*.json")):
                subset = art.stem.replace("detector_gap_", "")
                out.setdefault(arm_dir.name, {})[subset] = art
    return out


def load(path: pathlib.Path) -> dict:
    art = json.loads(path.read_text(encoding="utf-8"))
    res = art["results"][0]
    return {q["index"]: {"stratum": q["stratum"],
                         "native": bool(q["arms"]["native"]["correct"]),
                         "arm": bool(q["arms"][MECHANISM]["correct"])}
            for q in res["per_question"]}, res


def validity(res: dict) -> tuple[list[str], list[str]]:
    """(fails, warnings) — same policy as the runner: condition 2's native
    band is model-specific, so it warns rather than voids."""
    fails, warns = [], []
    for name, vc in (res.get("void_conditions") or {}).items():
        if not (isinstance(vc, dict) and vc.get("voids") == "run"
                and vc.get("status") == "fail"):
            continue
        obs = json.dumps(vc.get("observed", {}))
        (warns if name.startswith("2_") else fails).append(f"{name} {obs}")
    return fails, warns


def stratum_rows(flags: dict) -> dict:
    rows = {}
    for st in STRATA:
        sel = [v for v in flags.values()
               if st == "all" or v["stratum"] == st]
        if not sel:
            continue
        b = sum(1 for v in sel if v["native"] and not v["arm"])
        c = sum(1 for v in sel if v["arm"] and not v["native"])
        rows[st] = {"n": len(sel),
                    "native": sum(v["native"] for v in sel),
                    "arm": sum(v["arm"] for v in sel),
                    "net": c - b, "p_exact": mcnemar_exact_p(b, c)}
    return rows


def cross_arm(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    a_only = [i for i in common if a[i]["arm"] and not b[i]["arm"]]
    b_only = [i for i in common if b[i]["arm"] and not a[i]["arm"]]
    native_disagree = [i for i in common if a[i]["native"] != b[i]["native"]]
    return {"n_paired": len(common), "a_only": a_only, "b_only": b_only,
            "net_a_minus_b": len(a_only) - len(b_only),
            "p_exact": mcnemar_exact_p(len(b_only), len(a_only)),
            "native_disagreements": native_disagree,
            "pairing_valid": not native_disagree}


def run(model_tag: str, subset: str) -> dict:
    found = discover(model_tag)
    arms = {a: s[subset] for a, s in found.items() if subset in s}
    if not arms:
        raise SystemExit(f" no artifacts for model tag {model_tag!r} "
                         f"subset {subset!r}. Try --list.")
    flags, out = {}, {"model_tag": model_tag, "subset": subset, "arms": {}}
    for arm, path in arms.items():
        f, res = load(path)
        flags[arm] = f
        fails, warns = validity(res)
        out["arms"][arm] = {"path": str(path.relative_to(REPO).as_posix()),
                            "strata": stratum_rows(f),
                            "void": fails, "warnings": warns,
                            "valid": not fails}
    names = sorted(flags)
    out["cross_arm"] = {f"{x}_vs_{y}": cross_arm(flags[x], flags[y])
                        for i, x in enumerate(names) for y in names[i + 1:]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_tag", nargs="?")
    ap.add_argument("--subset", default="sh_64k")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.list or not args.model_tag:
        seen = {}
        for arm, subs in discover().items():
            for _s, p in subs.items():
                seen.setdefault(p.parent.name, set()).add(arm)
        for tag, arms in sorted(seen.items()):
            print(f"  {tag}: {', '.join(sorted(arms))}")
        return 0
    res = run(args.model_tag, args.subset)
    print(f"\n=== {res['model_tag']} · {res['subset']} ===")
    for arm, r in res["arms"].items():
        st = r["strata"]
        head = f"{arm:22s} {'VALID' if r['valid'] else 'VOID'}"
        print(f" {head}")
        for s in STRATA:
            if s in st:
                x = st[s]
                print(f"    {s:11s} native {x['native']:3d} -> arm {x['arm']:3d}"
                      f" /{x['n']:<4d} net {x['net']:+3d}  p={x['p_exact']:.4g}")
        for v in r["void"]:
            print(f"    VOID: {v}")
        for w in r["warnings"]:
            print(f"    warn: {w}")
    print("\n cross-arm (detector mechanism, paired on question index):")
    for k, c in res["cross_arm"].items():
        flag = "" if c["pairing_valid"] else "  [NATIVE DISAGREES - CHECK]"
        print(f"   {k:44s} net {c['net_a_minus_b']:+3d}  p={c['p_exact']:.4g}"
              f"  a_only={c['a_only']}  b_only={c['b_only']}{flag}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1),
                                          encoding="utf-8")
        print("\n written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
