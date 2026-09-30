"""E2E campaign analysis — the four-arm accuracy comparison.  [Phase C]

Reads detector-gap artifacts (per-question records) and produces the paired
comparison the campaign exists for:

  arms      hnav_raw (committed A1), hnav_abtt (committed A2),
            hnav_ces (new), hnav_abtt_noparser (new)
  metrics   overall /100 (primary), conflicted /66, unique /34 — per subset,
            never pooled; native vs detector_suppress with exact McNemar;
            cross-arm sh_64k comparison paired on question index.

Accuracy is recomputed here from ``per_question`` records — never read from a
runner's own summary (the repo's independent-recount rule).

Usage:  python -m hnav.geometry_filter.e2e_analysis \
            [--ces-dir pipelines/hnav_ces/results/<tag>] \
            [--noparser-dir pipelines/hnav_abtt_noparser/results/<tag>]
        (default: the single results dir present under each pipeline)
"""
from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
MECHANISM = "detector_suppress"
BASELINES_64K = {
    "hnav_raw": REPO / "stage0_results/abtt/abtt_arm_A1_raw_sh64k.json",
    "hnav_abtt": REPO / "stage0_results/abtt/abtt_arm_A2_abtt_sh64k.json",
}
BASELINES_CAL = {  # committed calibration retrieval-harness runs (raw parser arm)
    "sh_6k": REPO / "stage0_results/stage1/detector_gap_retrieval_sh6k.json",
    "sh_32k": REPO / "stage0_results/stage1/detector_gap_retrieval_sh32k.json",
}


def mcnemar_exact_p(b: int, c: int) -> float:
    from hnav.stage1.stale_suppression_probe import mcnemar_exact_p as f
    return f(b, c)


def load_result(path: pathlib.Path) -> dict:
    art = json.loads(path.read_text(encoding="utf-8"))
    return art["results"][0]


def flags(res: dict, arm: str, stratum: str | None = None) -> dict[int, bool]:
    out = {}
    for q in res["per_question"]:
        if stratum is None or q["stratum"] == stratum:
            out[q["index"]] = bool(q["arms"][arm]["correct"])
    return out


def arm_rows(res: dict) -> dict:
    rows = {}
    for st in ("all", "conflicted", "unique"):
        nat = flags(res, "native", None if st == "all" else st)
        sup = flags(res, MECHANISM, None if st == "all" else st)
        b = sum(1 for i in nat if nat[i] and not sup[i])
        c = sum(1 for i in nat if sup[i] and not nat[i])
        rows[st] = {"n": len(nat), "native": sum(nat.values()),
                    "detector": sum(sup.values()), "net": c - b,
                    "mcnemar_p": mcnemar_exact_p(b, c)}
    return rows


def guards(res: dict) -> list[str]:
    bad = [f"{k}={res[k]}" for k in ("n_page_edit_mismatch",
                                     "n_containment_violations",
                                     "n_page_edit_errors") if res.get(k, 0)]
    # older committed artifacts predate these fields; absence is not a void
    pc = res.get("positive_control")
    if pc is not None and not pc["ok"]:
        bad.append("positive control did not fire")
    aa = res.get("aa_floor")
    if aa is not None and aa["b_native_only"] + aa["c_arm_only"]:
        bad.append(f"A/A floor non-zero: {aa}")
    return bad


def cross_arm(results_64k: dict[str, dict]) -> dict:
    """Pairwise sh_64k comparisons of the SUPPRESS arms, question-paired."""
    out = {}
    names = list(results_64k)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, bname = names[i], names[j]
            fa = flags(results_64k[a], MECHANISM)
            fb = flags(results_64k[bname], MECHANISM)
            assert fa.keys() == fb.keys(), (a, bname)
            b = sum(1 for k in fa if fa[k] and not fb[k])
            c = sum(1 for k in fa if fb[k] and not fa[k])
            out[f"{a}_vs_{bname}"] = {
                "a_only_correct": b, "b_only_correct": c,
                "net_b_minus_a": c - b, "mcnemar_p": mcnemar_exact_p(b, c)}
    # native arms must agree across artifacts (same prompts, same model)
    natives = {n: flags(r, "native") for n, r in results_64k.items()}
    ref = natives[names[0]]
    out["native_agreement"] = {
        n: sum(1 for k in ref if ref[k] != natives[n][k]) for n in names[1:]}
    return out


def _default_dir(pipeline: str) -> pathlib.Path | None:
    root = REPO / "pipelines" / pipeline / "results"
    dirs = [d for d in root.iterdir() if d.is_dir()
            and any(d.glob("detector_gap_*.json"))] if root.exists() else []
    return dirs[0] if len(dirs) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ces-dir", default=None)
    ap.add_argument("--noparser-dir", default=None)
    ap.add_argument("--out", default=str(REPO / "stage0_results" /
                                         "geometry_filter" / "e2e_comparison.json"))
    args = ap.parse_args()

    ces_dir = pathlib.Path(args.ces_dir) if args.ces_dir else _default_dir("hnav_ces")
    nop_dir = (pathlib.Path(args.noparser_dir) if args.noparser_dir
               else _default_dir("hnav_abtt_noparser"))
    if not ces_dir or not nop_dir:
        raise SystemExit("could not resolve results dirs; pass --ces-dir/--noparser-dir")

    blob: dict = {"per_arm": {}}
    results_64k: dict[str, dict] = {}

    for name, path in BASELINES_64K.items():
        res = load_result(path)
        blob["per_arm"].setdefault(name, {})["sh_64k"] = {
            "rows": arm_rows(res), "guards": guards(res)}
        results_64k[name] = res
    for name, d in (("hnav_ces", ces_dir), ("hnav_abtt_noparser", nop_dir)):
        for sub in ("sh_6k", "sh_32k", "sh_64k"):
            p = d / f"detector_gap_{sub}.json"
            if not p.exists():
                continue
            res = load_result(p)
            blob["per_arm"].setdefault(name, {})[sub] = {
                "rows": arm_rows(res), "guards": guards(res)}
            if sub == "sh_64k":
                results_64k[name] = res
    for sub, path in BASELINES_CAL.items():
        if path.exists():
            res = load_result(path)
            blob["per_arm"].setdefault("hnav_raw", {})[sub] = {
                "rows": arm_rows(res), "guards": guards(res)}

    blob["cross_arm_sh_64k"] = cross_arm(results_64k)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print("written:", out)

    for name, subs in blob["per_arm"].items():
        for sub, entry in subs.items():
            r = entry["rows"]
            g = " GUARDS:" + ";".join(entry["guards"]) if entry["guards"] else ""
            print(f"{name:20s} {sub:7s} overall {r['all']['native']:>3}->"
                  f"{r['all']['detector']:>3}/{r['all']['n']}  conflicted "
                  f"{r['conflicted']['native']:>2}->{r['conflicted']['detector']:>2}"
                  f"/{r['conflicted']['n']} (p={r['conflicted']['mcnemar_p']:.2g}){g}")
    print()
    for k, v in blob["cross_arm_sh_64k"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
