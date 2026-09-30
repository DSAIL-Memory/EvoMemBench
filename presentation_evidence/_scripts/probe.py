# Probe the structure of every artifact the evidence pack needs.
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)

def shape(x, depth=0, max_depth=2):
    if depth >= max_depth:
        return type(x).__name__
    if isinstance(x, dict):
        return {k: shape(v, depth + 1, max_depth) for k, v in x.items()}
    if isinstance(x, list):
        return [f"list[{len(x)}]", shape(x[0], depth + 1, max_depth) if x else None]
    return repr(x)[:80]

FILES = [
    "stage0_results/question_strata.json",
    "stage0_results/stage1/stale_suppression_probe_sh6k.json",
    "stage0_results/stage1/detector_gap_confirmatory_sh64k.json",
    "stage0_results/stage1/stage1_calibration.json",
    "stage0_results/stage1/detector_gap_sh6k.json",
    "stage0_results/stage1_operating_point.json",
    "stage0_results/final/m1_geometry_calibration.json",
    "stage0_results/final/m1b_grouping_ablation.json",
    "stage0_results/t4_s2_trials_summary.json",
]

for p in FILES:
    d = load(p)
    print("=" * 100)
    print(p)
    print(json.dumps(shape(d), indent=1, default=str)[:3000])
