"""E2E-3 — paired analysis of the hnav_geo wet run against the committed arms.

Recomputes every accuracy from per_question records (never from summaries),
pairs on question index, and reports exact McNemar per stratum — the same
conventions as e2e_analysis.py, extended with the GG2 verdict of
GEO_PREREG.md (primary endpoint: sh_64k overall > 64/100 vs the committed
parser arm).

Usage:  python -m hnav.geometry_filter.e2e3_analysis <geo_results_dir>
"""
from __future__ import annotations

import json
import pathlib
import sys

from hnav.stage1.stale_suppression_probe import mcnemar_exact_p

REPO = pathlib.Path(__file__).resolve().parents[2]
COMMITTED = {
    "hnav_raw": REPO / "stage0_results/abtt/abtt_arm_A1_raw_sh64k.json",
    "hnav_abtt": REPO / "stage0_results/abtt/abtt_arm_A2_abtt_sh64k.json",
    "hnav_ces": REPO / "pipelines/hnav_ces/results/"
                       "Qwen_Qwen3-4B-Instruct-2507_2026-08-27/"
                       "detector_gap_sh_64k.json",
    "hnav_abtt_noparser": REPO / "pipelines/hnav_abtt_noparser/results/"
                                 "Qwen_Qwen3-4B-Instruct-2507_2026-08-27/"
                                 "detector_gap_sh_64k.json",
}


def flags(path: pathlib.Path, arm: str = "detector_suppress") -> dict:
    art = json.loads(path.read_text(encoding="utf-8"))
    res = art["results"][0]
    out = {}
    for q in res["per_question"]:
        out[q["index"]] = {
            "stratum": q["stratum"],
            "native": bool(q["arms"]["native"]["correct"]),
            "arm": bool(q["arms"][arm]["correct"]),
        }
    return out


def stratum_table(f: dict) -> dict:
    t = {}
    for name, keep in (("all", lambda s: True),
                       ("conflicted", lambda s: s == "conflicted"),
                       ("unique", lambda s: s == "unique")):
        rows = [v for v in f.values() if keep(v["stratum"])]
        b = sum(1 for v in rows if v["native"] and not v["arm"])
        c = sum(1 for v in rows if v["arm"] and not v["native"])
        t[name] = {"n": len(rows),
                   "native": sum(v["native"] for v in rows),
                   "arm": sum(v["arm"] for v in rows),
                   "b_native_only": b, "c_arm_only": c, "net": c - b,
                   "p_exact": mcnemar_exact_p(b, c)}
    return t


def cross(fa: dict, fb: dict) -> dict:
    assert fa.keys() == fb.keys()
    a_only = [i for i in fa if fa[i]["arm"] and not fb[i]["arm"]]
    b_only = [i for i in fa if fb[i]["arm"] and not fa[i]["arm"]]
    # native flags must agree: same prompts, same model, greedy decode
    disagree = [i for i in fa if fa[i]["native"] != fb[i]["native"]]
    return {"a_only": sorted(a_only), "b_only": sorted(b_only),
            "net_a_minus_b": len(a_only) - len(b_only),
            "p_exact": mcnemar_exact_p(len(b_only), len(a_only)),
            "native_disagreements": sorted(disagree)}


def main() -> int:
    geo_dir = pathlib.Path(sys.argv[1])
    geo64 = geo_dir / "detector_gap_sh_64k.json"
    fg = flags(geo64)
    out = {"geo_vs_native": stratum_table(fg), "cross_arm": {}}
    for name, path in COMMITTED.items():
        if path.exists():
            out["cross_arm"][f"geo_vs_{name}"] = cross(fg, flags(path))
    parser = out["cross_arm"].get("geo_vs_hnav_raw", {})
    geo_all = out["geo_vs_native"]["all"]["arm"]
    out["GG2"] = {
        "geo_overall_sh64k": geo_all,
        "hnav_raw_overall_sh64k": 64,
        "pass": bool(geo_all > 64),
        "paired_vs_parser": {k: parser.get(k) for k in
                             ("net_a_minus_b", "p_exact")},
        "conflicted": out["geo_vs_native"]["conflicted"]["arm"],
        "unique": out["geo_vs_native"]["unique"]["arm"],
    }
    dst = geo_dir / "e2e3_comparison.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("written:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
