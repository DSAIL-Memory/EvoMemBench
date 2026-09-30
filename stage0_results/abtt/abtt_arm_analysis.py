#!/usr/bin/env python3
"""ABTT Phase 5 — analysis of the sh_64k replication.  [registered]

Committed with ``abtt_preregistration.md`` and not modified between
registration and reporting. Reads the two arm artifacts, checks every
registered void condition, and reports the registered primary outcome.

The primary outcome is the PAIRED difference A2 - A1 on the conflicted
stratum, with a 95% CI. The study is registered as estimation, not as a
superiority test, because calibration predicted a ~+1 question effect on
n = 66 and no test at that n resolves it.

    python stage0_results/abtt/abtt_arm_analysis.py \\
        hnav/_out/abtt_arm_A1_raw_sh64k.json \\
        hnav/_out/abtt_arm_A2_abtt_sh64k.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.stage1.stale_suppression_probe import mcnemar_exact_p  # noqa: E402

MECHANISM = "detector_suppress"     # the shipped mechanism; demote_late is secondary


def load(p: Path) -> dict:
    # encoding is explicit: this repo has been bitten by a cp1254 default
    return json.load(io.open(p, encoding="utf-8"))


def paired(a: list[bool], b: list[bool]) -> dict:
    """b = A-only correct, c = B-only correct; net favours B."""
    nb = sum(1 for x, y in zip(a, b) if x and not y)
    nc = sum(1 for x, y in zip(a, b) if y and not x)
    return {"n": len(a), "b_A1_only": nb, "c_A2_only": nc, "net": nc - nb,
            "p_exact": mcnemar_exact_p(nb, nc)}


def wilson_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for the paired proportion difference (c-b)/n.

    Agresti-Min style normal approximation on discordant pairs; adequate here
    and honest about being an approximation at these counts.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    d = (c - b) / n
    var = ((b + c) - (c - b) ** 2 / n) / (n * n)
    se = max(var, 0.0) ** 0.5
    return (d - z * se, d + z * se)


def stratum_flags(art: dict, stratum: str, arm: str) -> tuple[list[int], list[bool]]:
    r = art["results"][0]
    idx, flags = [], []
    for q in r["per_question"]:
        if stratum != "all" and q["stratum"] != stratum:
            continue
        idx.append(q["index"])
        flags.append(bool(q["arms"][arm]["correct"]))
    return idx, flags


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    a1, a2 = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    r1, r2 = a1["results"][0], a2["results"][0]

    print("=" * 78)
    print(" ABTT sh_64k REPLICATION — registered analysis")
    print("=" * 78)
    op1, op2 = a1["operating_point"]["thresholds"], a2["operating_point"]["thresholds"]
    print(f" A1 raw   cos_pair={op1['cos_pair']} r_min={op1['r_min']:.4f} "
          f"space={a1['operating_point']['provenance'].get('geometry_space', 'raw')}")
    print(f" A2 abtt  cos_pair={op2['cos_pair']} r_min={op2['r_min']:.4f} "
          f"space={a2['operating_point']['provenance'].get('geometry_space')}")

    # ---- void conditions -------------------------------------------------
    print("\n VOID CONDITIONS")
    fails = []
    for tag, r in (("A1", r1), ("A2", r2)):
        v = r["void_conditions"]
        for k in ("n_page_edit_mismatch", "n_containment_violations",
                  "n_page_edit_errors"):
            if r.get(k, 0):
                fails.append(f"{tag}: {k}={r[k]}")
        if not r["positive_control"]["ok"]:
            fails.append(f"{tag}: positive control did not fire")
        if r["aa_floor"]["net"] != 0 or (r["aa_floor"]["b_native_only"]
                                         + r["aa_floor"]["c_arm_only"]) != 0:
            fails.append(f"{tag}: A/A floor non-zero {r['aa_floor']}")
        print(f"  {tag}: mismatch={r.get('n_page_edit_mismatch')} "
              f"containment={r.get('n_containment_violations')} "
              f"edit_errors={r.get('n_page_edit_errors')} "
              f"pos_control={r['positive_control']['ok']} "
              f"A/A discordant="
              f"{r['aa_floor']['b_native_only'] + r['aa_floor']['c_arm_only']}"
              f"/{r['aa_floor']['n']}  harness_verdict={v['verdict']['run_void']}")

    # condition 3: the two native arms must agree question-for-question
    i1, n1 = stratum_flags(a1, "all", "native")
    i2, n2 = stratum_flags(a2, "all", "native")
    same_idx = i1 == i2
    disagree = [i for i, x, y in zip(i1, n1, n2) if x != y] if same_idx else None
    print(f"  native-vs-native across runs: "
          f"{'indices differ!' if not same_idx else f'{len(disagree)} disagreement(s)'}")
    if not same_idx or disagree:
        fails.append(f"native arms disagree on {disagree}")

    print(f"\n  => RUN {'VOID: ' + '; '.join(fails) if fails else 'VALID'}")

    # ---- accuracy ---------------------------------------------------------
    print("\n ACCURACY (arm = detector_suppress)")
    print(f"  {'stratum':14s} {'n':>4s} {'A0 native':>10s} {'A1 raw':>10s} "
          f"{'A2 abtt':>10s}")
    for st in ("all", "conflicted", "unique"):
        _, nat = stratum_flags(a1, st, "native")
        _, s1 = stratum_flags(a1, st, MECHANISM)
        _, s2 = stratum_flags(a2, st, MECHANISM)
        if not nat:
            continue
        print(f"  {st:14s} {len(nat):4d} {sum(nat):5d}/{len(nat):<4d} "
              f"{sum(s1):5d}/{len(s1):<4d} {sum(s2):5d}/{len(s2):<4d}")

    # ---- the registered primary outcome ----------------------------------
    print("\n PRIMARY OUTCOME — A2 (abtt) vs A1 (raw), paired, conflicted stratum")
    _, c1 = stratum_flags(a1, "conflicted", MECHANISM)
    _, c2 = stratum_flags(a2, "conflicted", MECHANISM)
    pc = paired(c1, c2)
    lo, hi = wilson_diff_ci(pc["b_A1_only"], pc["c_A2_only"], pc["n"])
    print(f"  n={pc['n']}  A1-only correct={pc['b_A1_only']}  "
          f"A2-only correct={pc['c_A2_only']}  net={pc['net']:+d}")
    print(f"  difference {pc['net'] / pc['n']:+.4f} "
          f"(95% CI {lo:+.4f} .. {hi:+.4f})   McNemar exact p={pc['p_exact']:.4g}")

    superiority = pc["net"] >= 5 and pc["p_exact"] < 0.01
    noninferior = -pc["net"] <= 3
    print(f"  registered SUPERIORITY (net>=+5 and p<0.01): "
          f"{'MET' if superiority else 'not met'}")
    print(f"  registered NON-INFERIORITY (net>=-3):        "
          f"{'MET' if noninferior else 'NOT MET'}")

    # both arms must clear the no-H-Nav baseline
    _, cn = stratum_flags(a1, "conflicted", "native")
    print(f"  vs A0 baseline: A0={sum(cn)}/{len(cn)}  "
          f"A1={sum(c1)}/{len(c1)}  A2={sum(c2)}/{len(c2)}  "
          f"{'both clear A0' if sum(c1) > sum(cn) and sum(c2) > sum(cn) else 'CHECK'}")

    # ---- side-predictions -------------------------------------------------
    print("\n SIDE-PREDICTIONS")
    s1n = r1["positive_control"]["n_facts_suppressed"]
    s2n = r2["positive_control"]["n_facts_suppressed"]
    print(f"  P1 ABTT suppresses fewer facts : {s2n} < {s1n} -> "
          f"{'HIT' if s2n < s1n else 'MISS'}")
    print(f"  P2 net in [-2,+3] of 66        : net={pc['net']:+d} -> "
          f"{'HIT' if -2 <= pc['net'] <= 3 else 'MISS'}")
    h1, h2 = r1.get("harm", {}).get(MECHANISM, {}), r2.get("harm", {}).get(MECHANISM, {})
    print(f"  P3 no new harm class in A2     : A1={json.dumps(h1)}")
    print(f"                                   A2={json.dumps(h2)}")
    print(f"  P4 A2 conflicted coverage >= A1: "
          f"{sum(c2)} vs {sum(c1)} -> {'HIT' if sum(c2) >= sum(c1) else 'MISS'}")

    # ---- token cost -------------------------------------------------------
    t1, t2 = r1.get("tokens", {}), r2.get("tokens", {})
    if t1 and t2:
        print("\n TOKEN COST vs native (registered: must be <= 0)")
        for tag, t in (("A1", t1), ("A2", t2)):
            arm = t.get(MECHANISM, {})
            pct = arm.get("delta_pct")
            if pct is None:
                continue
            print(f"  {tag}: {pct:+.4f}%  "
                  f"({arm.get('delta_chars_vs_native'):+d} chars)  "
                  f"{'OK' if pct <= 0 else 'EXCEEDS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
