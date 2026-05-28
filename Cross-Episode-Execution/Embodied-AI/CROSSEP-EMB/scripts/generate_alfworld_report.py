#!/usr/bin/env python3
"""Generate final report for ALFWorld mem0 batch evaluation.

Usage:
    python generate_alfworld_report.py <output_base> [--title "..."]
"""

import argparse
import glob
import json
import os
from datetime import datetime

SHORT = {
    "pick_and_place_simple":          "pick_and_place",
    "pick_two_obj_and_place":         "pick_two_obj",
    "pick_clean_then_place_in_recep": "pick_clean",
    "pick_cool_then_place_in_recep":  "pick_cool",
    "pick_heat_then_place_in_recep":  "pick_heat",
    "look_at_obj_in_light":           "look_at_obj",
}


def shorten(name):
    for long, short in SHORT.items():
        name = name.replace(long, short)
    return name


def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_row(s):
    n = s.get("num_samples", 0)
    return {
        "n":       n,
        "success": s.get("success_rate", 0),
        "steps":   s.get("avg_rounds_per_sample", 0),
        "tokens":  s.get("avg_tokens_per_sample", 0),
        "lat":     s.get("avg_sample_latency_per_sample", 0),
        "in_tok":  s.get("total_prompt_tokens", 0) // max(n, 1),
        "out_tok": s.get("total_completion_tokens", 0) // max(n, 1),
        "emb":     (s.get("memory_inject", {}).get("total_embedding_tokens", 0)
                    + s.get("memory_update", {}).get("total_embedding_tokens", 0)),
        "inj_lat": s.get("memory_inject", {}).get("avg_latency_per_sample", 0),
        "upd_lat": s.get("memory_update", {}).get("avg_latency_per_sample", 0),
    }


def weighted_avg(rows):
    total_n = sum(r["n"] for r in rows)
    if total_n == 0:
        return None

    def wavg(key):
        return sum(r[key] * r["n"] for r in rows) / total_n

    return {
        "n":       total_n,
        "success": wavg("success"),
        "steps":   wavg("steps"),
        "tokens":  wavg("tokens"),
        "lat":     wavg("lat"),
        "in_tok":  int(wavg("in_tok")),
        "out_tok": int(wavg("out_tok")),
        "emb":     int(sum(r["emb"] for r in rows) / total_n),
        "inj_lat": wavg("inj_lat"),
        "upd_lat": wavg("upd_lat"),
    }


def fmt_in_env(label, r):
    return (
        f"{label:<38} {r['n']:>3}  {r['success']:>8.4f}  {r['steps']:>8.2f}  "
        f"{r['tokens']:>6}  {r['lat']:>6.2f}s  {r['in_tok']:>6}  {r['out_tok']:>6}  "
        f"{r['emb']:>7}  {r['inj_lat']:>6.2f}s  {r['upd_lat']:>6.2f}s"
    )


def fmt_cross(label, r):
    return (
        f"{label:<48} {r['n']:>3}  {r['success']:>8.4f}  {r['steps']:>8.2f}  "
        f"{r['tokens']:>6}  {r['lat']:>6.2f}s  {r['in_tok']:>6}  {r['out_tok']:>6}  "
        f"{r['emb']:>7}  {r['inj_lat']:>6.2f}s  {r['upd_lat']:>6.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_base", help="Base output directory of an experiment run")
    parser.add_argument("--title", default="ALFWorld Mem0 Batch Final Report",
                        help="Report title line")
    args = parser.parse_args()

    output_base = args.output_base
    report_file = os.path.join(output_base, "final_report.txt")

    SEP = "=" * 76
    HDR = "-" * 76
    H_IN = (
        f"{'Task':<38} {'N':>3}  {'SuccRate':>8}  {'Avg.Step':>8}  "
        f"{'Tkns/s':>6}  {'AvgLat':>7}  {'InTok':>6}  {'OutTok':>6}  "
        f"{'Emb':>7}  {'InjLat':>7}  {'UpdLat':>7}"
    )
    H_CRS = (
        f"{'Source -> Target':<48} {'N':>3}  {'SuccRate':>8}  {'Avg.Step':>8}  "
        f"{'Tkns/s':>6}  {'AvgLat':>7}  {'InTok':>6}  {'OutTok':>6}  "
        f"{'Emb':>7}  {'InjLat':>7}  {'UpdLat':>7}"
    )

    in_env_rows = []
    for p in sorted(glob.glob(os.path.join(output_base, "in_env", "*", "summary.json"))):
        task = os.path.basename(os.path.dirname(p))
        s = load_summary(p)
        if s:
            in_env_rows.append((task, extract_row(s)))

    cross_rows = []
    for p in sorted(glob.glob(os.path.join(output_base, "cross_env", "*", "summary.json"))):
        pair = os.path.basename(os.path.dirname(p))
        s = load_summary(p)
        if s:
            cross_rows.append((pair, extract_row(s)))

    lines = [
        SEP,
        args.title,
        f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output : {output_base}",
        SEP,
        "",
        "[In-Env Results]",
        H_IN, HDR,
    ]

    if in_env_rows:
        for task, r in in_env_rows:
            lines.append(fmt_in_env(shorten(task), r))
        wa = weighted_avg([r for _, r in in_env_rows])
        if wa:
            lines.append(HDR)
            lines.append(fmt_in_env("[Weighted Avg]", wa))
    else:
        lines.append("  (no results found)")

    lines += ["", "[Cross-Env Results]", H_CRS, HDR]
    if cross_rows:
        for pair, r in cross_rows:
            label = shorten(pair.replace("__to__", " -> "))
            lines.append(fmt_cross(label, r))
    else:
        lines.append("  (no results found)")

    lines += ["", SEP, f"Report saved to: {report_file}", ""]
    report = "\n".join(lines)
    print(report)
    with open(report_file, "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
