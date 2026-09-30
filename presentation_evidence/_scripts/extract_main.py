# Extraction for items 2, 3, 5(arms), 6, 7, 9, 10, 11, 12, 15.
# Every number is read from the JSON artifacts; derived numbers are recomputed here.
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "presentation_evidence", "data")

def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)

def save_json(name, obj):
    with open(os.path.join(DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

def save_csv(name, header, rows):
    with open(os.path.join(DATA_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

print("#" * 30, "ITEMS 2 & 3 — question_strata.json")
qs = load("stage0_results/question_strata.json")

# Recompute aggregate error totals from the per-run breakdown (never trust the summary field).
tot = {"stale_value": 0, "off_list": 0, "empty": 0}
rows2, rows3 = [], []
for r in qs["runs"]:
    for stratum in ("unique", "conflicted", "ambiguous", "unmatched"):
        for k in tot:
            tot[k] += r["strata"][stratum]["errors"][k]
    c = r["strata"]["conflicted"]
    u = r["strata"]["unique"]
    rows2.append([r["run"], c["n"], c["correct"], c["errors"]["stale_value"],
                  c["errors"]["off_list"], c["errors"]["empty"],
                  u["errors"]["stale_value"], u["errors"]["off_list"], u["errors"]["empty"]])
    # recompute overall accuracy from strata counts
    n_all = sum(r["strata"][s]["n"] for s in ("unique", "conflicted", "ambiguous", "unmatched"))
    c_all = sum(r["strata"][s]["correct"] for s in ("unique", "conflicted", "ambiguous", "unmatched"))
    overall_recomputed = c_all / n_all
    assert abs(overall_recomputed - r["accuracy_overall"]) < 1e-12, r["run"]
    rows3.append([r["run"], r["file"], u["n"], u["correct"], round(u["accuracy"], 6),
                  c["n"], c["correct"], round(c["accuracy"], 6),
                  round(r["accuracy_overall"], 6)])
print("recomputed errors_total:", tot, "| artifact says:", qs["aggregate"]["errors_total"])
assert tot == qs["aggregate"]["errors_total"]
print("sum of error classes:", sum(tot.values()))
save_csv("item02_error_classes.csv",
         ["run", "conflicted_n", "conflicted_correct", "conflicted_stale_value",
          "conflicted_off_list", "conflicted_empty",
          "unique_stale_value", "unique_off_list", "unique_empty"], rows2)
save_csv("item03_strata_accuracy.csv",
         ["run", "file", "unique_n", "unique_correct", "unique_acc",
          "conflicted_n", "conflicted_correct", "conflicted_acc", "overall_acc"], rows3)
save_json("item02_definitions.json", {
    "definitions": qs["definitions"], "aggregate": qs["aggregate"],
    "recomputed_errors_total": tot,
    "arithmetic": "sum over runs[].strata.{unique,conflicted,ambiguous,unmatched}.errors",
})
print("unique acc min/max:", qs["aggregate"]["unique_accuracy_min"], qs["aggregate"]["unique_accuracy_max"])
print("conflicted acc min/max:", qs["aggregate"]["conflicted_accuracy_min"], qs["aggregate"]["conflicted_accuracy_max"])

# Independent recount of ONE run from its raw evidence file, using subsets[].indices.
raw = load("stage0_results/t4_s2_evidence/sh_6k_off_results.json")
sub = next(s for s in qs["subsets"] if s["subset"] == "sh_6k")
raw_rows = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
uniq_ix = set(sub["indices"]["unique"]); conf_ix = set(sub["indices"]["conflicted"])
uc = sum(1 for i, row in enumerate(raw_rows) if i in uniq_ix and row["substring_exact_match"])
cc = sum(1 for i, row in enumerate(raw_rows) if i in conf_ix and row["substring_exact_match"])
run_off = next(r for r in qs["runs"] if r["run"] == "sh_6k_off")
print(f"raw recount sh_6k_off: unique {uc}/{len(uniq_ix)}, conflicted {cc}/{len(conf_ix)}",
      "| artifact:", run_off["strata"]["unique"]["correct"], run_off["strata"]["conflicted"]["correct"])
assert uc == run_off["strata"]["unique"]["correct"] and cc == run_off["strata"]["conflicted"]["correct"]

print("#" * 30, "ITEM 5 — probe arms (sh_6k, sh_32k) + confirmatory mirror (sh_64k)")
item5 = {}
for tag, path in [("sh_6k", "stage0_results/stage1/stale_suppression_probe_sh6k.json"),
                  ("sh_32k", "stage0_results/stage1/stale_suppression_probe_sh32k.json")]:
    p = load(path)
    r0 = p["results"][0]
    conf = r0["by_stratum"]["conflicted"]
    item5[tag] = {
        "file": path, "harness": p["harness"], "arm_descriptions": p["arms"],
        "conflicted_arms": conf["arms"], "paired_vs_native": conf["paired_vs_native"],
    }
    # recompute accuracies
    for a, v in conf["arms"].items():
        assert abs(v["correct"] / v["n"] - v["accuracy"]) < 1e-12
conf64 = load("stage0_results/stage1/detector_gap_confirmatory_sh64k.json")
c64 = conf64["results"][0]["by_stratum"]["conflicted"]
item5["sh_64k_confirmatory_mirror"] = {
    "file": "stage0_results/stage1/detector_gap_confirmatory_sh64k.json",
    "harness_note": conf64["harness"].get("comparability_to_oracle_probe", ""),
    "arm_descriptions": conf64["arms"],
    "conflicted_arms": c64["arms"], "paired_vs_native": c64["paired_vs_native"],
}
save_json("item05_arms.json", item5)
for tag in ("sh_6k", "sh_32k"):
    a = item5[tag]["conflicted_arms"]
    print(tag, {k: f"{v['correct']}/{v['n']}={v['accuracy']:.3f}" for k, v in a.items()})
print("sh_64k", {k: f"{v['correct']}/{v['n']}={v['accuracy']:.3f}" for k, v in c64["arms"].items()})

print("#" * 30, "ITEM 6 — m1 geometry percentiles")
m1 = load("stage0_results/final/m1_geometry_calibration.json")
rows6 = []
for e in m1:
    for series in ("whole_blob_sim", "control_whole_blob_sim", "diff_sim", "qr_residual_new_vs_old"):
        d = e[series]
        rows6.append([e["subset"], series, d["mean"], d["p10"], d["p50"], d["p90"],
                      e["n_conflict_pairs"], e["n_control_pairs"],
                      e["separation_auc_conflict_vs_control"], e["model"], e["dtype"],
                      e["gate_pass"], e["parse_coverage"]])
    print(e["subset"], "conflict p50:", e["whole_blob_sim"]["p50"],
          "control p50:", e["control_whole_blob_sim"]["p50"],
          "AUC:", e["separation_auc_conflict_vs_control"],
          "n:", e["n_conflict_pairs"], "/", e["n_control_pairs"],
          "gate_pass:", e["gate_pass"], "parse_coverage:", e["parse_coverage"])
save_csv("item06_geometry_percentiles.csv",
         ["subset", "series", "mean", "p10", "p50", "p90", "n_conflict_pairs",
          "n_control_pairs", "separation_auc", "model", "dtype", "gate_pass",
          "parse_coverage"], rows6)

print("#" * 30, "ITEM 7 — m1b PR sweep")
m1b = load("stage0_results/final/m1b_grouping_ablation.json")
rows7, meta7 = [], {}
for e in m1b:
    for pt in e["pr_curve"]:
        rows7.append([e["subset"], pt["tau"], pt["precision"], pt["recall"], pt["f1"]])
    bf = e["best_f1"]
    # recompute F1 from precision/recall at the best point
    pr, rc = bf["precision"], bf["recall"]
    f1r = 2 * pr * rc / (pr + rc)
    assert abs(f1r - bf["f1"]) < 1e-9
    meta7[e["subset"]] = {"best_f1": bf, "equal_coverage": e["equal_coverage"],
                          "recall_ceiling_from_knn": e["recall_ceiling_from_knn"],
                          "n_truth_pairs": e["n_truth_pairs"],
                          "n_candidate_pairs": e["n_candidate_pairs"],
                          "parse_coverage_pct": e["parse_coverage_pct"],
                          "interpretation": e["interpretation"],
                          "pr_curve_len": len(e["pr_curve"])}
    print(e["subset"], "best F1", round(bf["f1"], 4), "at tau", bf["tau"],
          "| equal-coverage F1", round(e["equal_coverage"].get("f1", float("nan")), 4),
          "| knn recall ceiling", e["recall_ceiling_from_knn"],
          "| curve pts", len(e["pr_curve"]))
save_csv("item07_pr_curves.csv", ["subset", "tau", "precision", "recall", "f1"], rows7)
save_json("item07_summary.json", meta7)

print("#" * 30, "ITEM 9 — operating point thresholds")
op = load("stage0_results/stage1_operating_point.json")
th = op["thresholds"]
val = math.sqrt(1 - 0.44 ** 2)
print("thresholds:", th)
print("sqrt(1-0.44^2) =", val)
save_json("item09_thresholds.json", {
    "source": "stage0_results/stage1_operating_point.json -> thresholds",
    "thresholds": th, "r_min_label": op["r_min_label"],
    "sqrt(1-r_min^2)": val,
    "reading": "a pair passing r >= 0.44 implies in-span cosine <= 0.898, just under the cos_pair=0.90 screen: the two screens meet at the same geometric point rather than one overriding the other",
})

print("#" * 30, "ITEMS 10 & 11 — stage1_calibration cells")
cal = load("stage0_results/stage1/stage1_calibration.json")
cells = cal["cells"]
print("n cells:", len(cells), "| n_questions:", cal["n_questions"])
rows10 = []
fv_on, fv_off = [], []
net_pos = {"true": 0, "false": 0}
help_harm = {"true": [0, 0], "false": [0, 0]}
for c in cells:
    nv = c["n_verified"]
    fv = (c["n_fv_diff_key"] + c["n_fv_same_object"]) / nv if nv else None
    if fv is not None:
        assert abs(fv - c["false_verified_rate"]) < 1e-12
    (fv_on if c["pair_filter"] else fv_off).append((fv, nv))
    key = "true" if c["pair_filter"] else "false"
    net = c["helped"] - c["harmed"]
    if net > 0:
        net_pos[key] += 1
    help_harm[key][0] += c["helped"]
    help_harm[key][1] += c["harmed"]
    rows10.append([c["cos_pair"], c["r_min_label"],
                   (round(c["r_min"], 6) if c["r_min"] is not None else ""), c["ambiguity_mode"],
                   c["nli_contradiction"], c["pair_filter"], nv, c["n_true_supersession"],
                   c["n_fv_diff_key"], c["n_fv_same_object"],
                   (round(fv, 6) if fv is not None else ""), c["helped"], c["harmed"], net])
save_csv("item10_11_cells.csv",
         ["cos_pair", "r_min_label", "r_min", "ambiguity_mode", "nli_contradiction",
          "pair_filter", "n_verified", "n_true_supersession", "n_fv_diff_key",
          "n_fv_same_object", "fv_rate_recomputed", "helped", "harmed", "net"], rows10)

def rng(vals):
    xs = [v for v, n in vals if v is not None]
    zero_nv = sum(1 for v, n in vals if v is None)
    return {"n_cells": len(vals), "n_cells_with_verified_pairs": len(xs),
            "n_cells_zero_verified": zero_nv,
            "min": min(xs) if xs else None, "max": max(xs) if xs else None}

r_on, r_off = rng(fv_on), rng(fv_off)
print("pair_filter=True  FV rate:", r_on)
print("pair_filter=False FV rate:", r_off)
print("net-positive cells: screen-on", net_pos["true"], "of", len(fv_on),
      "| screen-off", net_pos["false"], "of", len(fv_off))
print("helped/harmed totals: screen-on", help_harm["true"], "screen-off", help_harm["false"])
save_json("item10_summary.json", {
    "grid": cal["provenance"]["grid"], "nli_model": cal["provenance"]["nli_model"],
    "n_questions_calibration": cal["n_questions"],
    "fv_rate_pair_filter_true": r_on, "fv_rate_pair_filter_false": r_off,
    "fv_rate_formula": "(n_fv_diff_key + n_fv_same_object) / n_verified per cell",
    "operating_point_metrics": op["metrics"],
    "operating_point_note": "metrics.n_questions=200 -> CALIBRATION split (sh_6k+sh_32k), not held-out",
    "kyd_marlowe_provenance": "TEZ_BULGULARI.md lines 265-267 ONLY; scores 0.99949 / 0.99983 in the two directions; no JSON artifact behind it",
})
save_json("item11_summary.json", {
    "net_positive_cells": net_pos,
    "helped_harmed_totals": {"pair_filter_true": {"helped": help_harm["true"][0], "harmed": help_harm["true"][1]},
                             "pair_filter_false": {"helped": help_harm["false"][0], "harmed": help_harm["false"][1]}},
    "narrative": "STAGE1_NULL_ANALIZI.md",
})
# verify the 'numbers to state carefully' claim: every net-positive cell has pair_filter false
bad = [c for c in cells if (c["helped"] - c["harmed"]) > 0 and c["pair_filter"]]
print("net-positive cells with pair_filter=True (expect 0):", len(bad))
print("net-positive cells across all 162 (expect 21):", net_pos["true"] + net_pos["false"])

print("#" * 30, "ITEM 12 — detector vs oracle")
item12 = {}
for tag, path in [("sh_6k", "stage0_results/stage1/detector_gap_sh6k.json"),
                  ("sh_32k", "stage0_results/stage1/detector_gap_sh32k.json")]:
    g = load(path)
    dvo = g["detector_vs_oracle"][tag]
    m = dvo["by_mechanism"]["detector_suppress"]
    ratio = m["detector_net"] / m["oracle_net"]
    assert abs(ratio - m["net_ratio"]) < 1e-12
    item12[tag] = {"file": path, "source_probe": dvo["source"],
                   "same_harness": dvo["same_harness"],
                   "native_cross_run": dvo["native_cross_run"],
                   "suppress": m,
                   "demote_late": dvo["by_mechanism"]["detector_demote_late"],
                   "identical_to_oracle_probe": g["harness"].get("identical_to_oracle_probe", "")}
    print(tag, "suppress: detector_net", m["detector_net"], "oracle_net", m["oracle_net"],
          "ratio", round(ratio, 4), "| native cross-run identical:",
          dvo["native_cross_run"]["identical"])
# the retrieval-path files exist but carry harness_match false — record the caveat
ret = load("stage0_results/stage1/detector_gap_retrieval_sh6k.json")
item12["retrieval_files_caveat"] = ret["detector_vs_oracle"]["sh_6k"]["harness_caveat"]
item12["sh_64k_correction"] = load(
    "stage0_results/stage1/detector_gap_confirmatory_sh64k.json")["corrections"][0]["items"][4]
# cross-run per-question comparison: oracle fixed but detector missed (and vice versa)
for tag, gap_path, probe_path in [
        ("sh_6k", "stage0_results/stage1/detector_gap_sh6k.json",
         "stage0_results/stage1/stale_suppression_probe_sh6k.json"),
        ("sh_32k", "stage0_results/stage1/detector_gap_sh32k.json",
         "stage0_results/stage1/stale_suppression_probe_sh32k.json")]:
    g = load(gap_path); p = load(probe_path)
    gq = {q["index"]: q for q in g["results"][0]["per_question"]}
    pq = {q["index"]: q for q in p["results"][0]["per_question"]}
    nat_same = sum(1 for i in gq if gq[i]["arms"]["native"]["output"] == pq[i]["arms"]["native"]["output"])
    oracle_only = [i for i in gq if gq[i]["stratum"] == "conflicted"
                   and pq[i]["arms"]["oracle_suppress"]["correct"]
                   and not gq[i]["arms"]["detector_suppress"]["correct"]]
    det_only = [i for i in gq if gq[i]["stratum"] == "conflicted"
                and not pq[i]["arms"]["oracle_suppress"]["correct"]
                and gq[i]["arms"]["detector_suppress"]["correct"]]
    item12[tag]["per_question_cross_run"] = {
        "native_output_identical_questions": nat_same, "of": len(gq),
        "conflicted_oracle_right_detector_wrong": oracle_only,
        "conflicted_detector_right_oracle_wrong": det_only,
        "caveat": "cross-RUN comparison (two separate LLM passes); native outputs identical on "
                  f"{nat_same}/{len(gq)} questions",
    }
    print(tag, "cross-run: oracle-right/detector-wrong", oracle_only,
          "| detector-right/oracle-wrong", det_only, "| native identical", nat_same, "/", len(gq))
save_json("item12_detector_vs_oracle.json", item12)

print("#" * 30, "ITEM 15 — t4_s2 trials")
t4 = load("stage0_results/t4_s2_trials_summary.json")
off = t4["off_sem_per_run"]; sh = t4["shadow_sem_per_run"]
mean = lambda xs: sum(xs) / len(xs)
print("off runs:", off, "mean", mean(off), "| artifact mean", t4["off_sem_mean"])
print("shadow runs:", sh, "mean", mean(sh), "| artifact mean", t4["shadow_sem_mean"])
assert abs(mean(off) - t4["off_sem_mean"]) < 1e-9
assert abs(mean(sh) - t4["shadow_sem_mean"]) < 1e-9
save_json("item15_trials.json", {
    "source": "stage0_results/t4_s2_trials_summary.json",
    "off_sem_per_run_pct": off, "shadow_sem_per_run_pct": sh,
    "off_mean_recomputed": mean(off), "shadow_mean_recomputed": mean(sh),
    "noise_floor": t4["noise_floor"], "tost": t4["tost"], "permutation": t4["permutation"],
    "decision_rule_result": t4["decision_rule_result"],
    "protocol": t4["protocol"],
    "scale_reference_confirmatory_sh64k": {
        "conflicted_native": 17 / 66, "conflicted_suppress": 37 / 66,
        "delta_points": (37 - 17) / 66 * 100,
        "note": "sh_64k conflicted stratum, different subset from the sh_6k noise runs; drawn for scale only"},
})
print("noise floor:", t4["noise_floor"])
print("tost:", t4["tost"])
print("permutation:", t4["permutation"])
print("DONE")
