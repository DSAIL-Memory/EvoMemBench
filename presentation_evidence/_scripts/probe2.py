import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)

def dump(label, obj, n=3500):
    print("=" * 100)
    print(label)
    print(json.dumps(obj, indent=1, default=str)[:n])

probe = load("stage0_results/stage1/stale_suppression_probe_sh6k.json")
r0 = probe["results"][0]
print("results[0] keys:", list(r0.keys()))
dump("probe sh6k results[0].by_stratum.conflicted (minus per-arm details)",
     {k: (v if k != "arms" else v) for k, v in r0["by_stratum"]["conflicted"].items()})
dump("probe sh6k per_question[1]", r0["per_question"][1])

conf = load("stage0_results/stage1/detector_gap_confirmatory_sh64k.json")
c0 = conf["results"][0]
print("confirmatory results[0] keys:", list(c0.keys()))
dump("confirmatory by_stratum.conflicted", c0["by_stratum"]["conflicted"], 4000)
dump("confirmatory tokens", c0.get("tokens"))
dump("confirmatory harm", c0.get("harm"))
dump("confirmatory void_conditions", c0.get("void_conditions"), 6000)
dump("confirmatory corrections", conf["corrections"], 4000)
dump("confirmatory per_question[0]", c0["per_question"][0], 4000)

gap6 = load("stage0_results/stage1/detector_gap_sh6k.json")
dump("detector_gap_sh6k detector_vs_oracle", gap6["detector_vs_oracle"], 4000)
gap32 = load("stage0_results/stage1/detector_gap_sh32k.json")
dump("detector_gap_sh32k detector_vs_oracle", gap32["detector_vs_oracle"], 4000)
ret6 = load("stage0_results/stage1/detector_gap_retrieval_sh6k.json")
dump("retrieval_sh6k detector_vs_oracle", ret6.get("detector_vs_oracle"), 3000)

cal = load("stage0_results/stage1/stage1_calibration.json")
dump("stage1_calibration cells[0]", cal["cells"][0])
dump("stage1_calibration provenance.grid", cal["provenance"]["grid"])
print("nli_model:", cal["provenance"]["nli_model"])

qs = load("stage0_results/question_strata.json")
dump("question_strata runs[0]", qs["runs"][0])
dump("question_strata subsets[0] (keys + first indices)", {k: (v[:12] if isinstance(v, list) else v) for k, v in qs["subsets"][0].items()})
dump("question_strata definitions", qs["definitions"])
dump("question_strata aggregate", qs["aggregate"])

t4 = load("stage0_results/t4_s2_trials_summary.json")
dump("t4 summary full", t4, 4000)

op = load("stage0_results/stage1_operating_point.json")
dump("operating_point thresholds+metrics", {"thresholds": op["thresholds"], "metrics": op["metrics"]})
