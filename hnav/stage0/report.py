#!/usr/bin/env python3
"""The Stage-0 report.  [T8 — GATE, HARD STOP]

Collects every measurement written under ``HNAV_OUT_DIR`` and emits
``STAGE0_REPORT.md``: M0 fidelity, M1 real-embedding distributions (superseding
the lexical proxy), the M1b grouping-ablation verdict, M2 distributions and the
raw-entropy verdict, M3 headroom per component, the M4/H2 verdict, and a
GO/NO_GO decision per component against the frozen gate in
``EVOMEMBENCH_HNAV_STAGE0_PROTOCOL.md`` §4.

Every NO_GO is tagged with **which of the three verdicts** it is. They must never
be conflated:

  1. **benchmark** — the class is too rare or absent. "EvoMemBench does not
     generate this failure class." Not evidence about H-Nav.
  2. **detection** — the class is abundant but the signal does not predict it.
     Genuine evidence against H-Nav.
  3. **policy** — the signal predicts, the intervention does not repair.

A missing measurement is reported as **NOT RUN**, never as a zero and never as a
pass. ``--strict`` exits non-zero if anything required is absent.

**This script does not unlock anything.** After it runs, a human evaluates the
gate. ``write_policy.py`` and ``read_policy.py`` do not exist and must not be
written before that happens.

    python hnav/stage0/report.py
    python hnav/stage0/report.py --strict
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav import config as _config  # noqa: E402

CALIBRATION = ("sh_6k", "sh_32k")

# Frozen gate — protocol §4. Primary arena values.
GATE = {
    "min_positives": 40,
    "min_coverage": 0.10,
    "min_precision": 0.90,
    "min_precision_wilson_lb": 0.80,
    "max_harm_wilson_ub": 0.03,
    "min_delta_acc_pp": 3.0,
}

NA = "—"


# ── helpers ──────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used rather than the normal approximation because
    the counts here are small and the proportions live near 0 and 1."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(out_dir: Path, name: str):
    f = out_dir / name
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def fmt(x, spec: str = ".3f") -> str:
    if x is None:
        return NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return NA if v != v else format(v, spec)


def section(title: str) -> str:
    return f"\n## {title}\n\n"


# ── M0 ───────────────────────────────────────────────────────────────────────
def render_m0(out_dir: Path) -> tuple[str, bool]:
    data = load(out_dir, "m0_replica_fidelity.json")
    s = section("M0 — retriever-replica fidelity")
    if data is None:
        s += (
            "**NOT RUN against a live index.**\n\n"
            "What *is* established, by `hnav/tests/test_replica_fidelity.py` "
            "(1,200 sampled (store, query) pairs):\n\n"
            "- `NumpyCosineReplica` reproduces `(bank_matrix @ query_vec) * 100.0` "
            "and `np.argsort(scores)[::-1]` **bit-for-bit** — 100% exact top-k and "
            "full-ranking identity, zero score error, including on a fixture with "
            "deliberate exact ties.\n"
            "- `FaissFlatReplica`'s ordering is identical to ascending squared-L2, "
            "the quantity LangChain FAISS returns.\n\n"
            "What is **not** established here: agreement with a live LangChain FAISS "
            "index (FAISS's internal tie order is not reproducible from outside it), "
            "and agreement when vectors come from the real embedder rather than the "
            "test embedder. Neither can be measured without the GPU box.\n\n"
            "**No signal is declared valid on this basis.** Until the live check runs, "
            "`rank_self`, `margin`, `dH_self`, `dH_neighbor` and `churn` are "
            "provisional on the primary arena.\n")
        return s, False

    s += "| arena | pairs | top-1 | top-k | Kendall τ | max |Δscore| | tie rate |\n"
    s += "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    ok = True
    for row in data:
        s += (f"| {row.get('arena', NA)} | {row.get('n_pairs', NA)} | "
              f"{fmt(row.get('top1_agreement'), '.4f')} | "
              f"{fmt(row.get('topk_agreement'), '.4f')} | "
              f"{fmt(row.get('kendall_tau'), '.4f')} | "
              f"{fmt(row.get('max_abs_score_error'), '.2e')} | "
              f"{fmt(row.get('tie_rate'), '.4f')} |\n")
        if (row.get("topk_agreement") or 0) < 0.999:
            ok = False
    if not ok:
        s += ("\n**S1 FIRED.** Fidelity below 99.9%. `rank_self`, `margin`, "
              "`dH_self`, `dH_neighbor` and `churn` are INVALID and the mechanisms "
              "depending on them are dropped — not silently re-based on a different "
              "retriever.\n")
    return s, ok


# ── M1 / M1b / M2 ────────────────────────────────────────────────────────────
def render_m1(out_dir: Path) -> tuple[str, bool | None]:
    data = load(out_dir, "m1_geometry_calibration.json")
    s = section("M1 — geometry calibration with real embeddings")
    s += ("*Supersedes the char-3gram lexical proxy in analysis §9. No threshold "
          "anywhere in H-Nav is derived from that proxy.*\n\n")
    if data is None:
        return s + "**NOT RUN.** The Stage-0 gate has not been evaluated.\n", None

    s += ("| subset | pairs | whole p50 | diff p50 | control p50 | AUC | crit-Δ % | "
          "gate |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | :-- |\n")
    all_pass = True
    for r in data:
        gate_pass = bool(r.get("gate_pass"))
        all_pass &= gate_pass
        s += (f"| {r.get('subset')} | {r.get('n_conflict_pairs')} | "
              f"{fmt((r.get('whole_blob_sim') or {}).get('p50'))} | "
              f"{fmt((r.get('diff_sim') or {}).get('p50'))} | "
              f"{fmt((r.get('control_whole_blob_sim') or {}).get('p50'))} | "
              f"{fmt(r.get('separation_auc_conflict_vs_control'))} | "
              f"{fmt(r.get('critical_delta_pct'), '.1f')} | "
              f"{'PASS' if gate_pass else '**FAIL**'} |\n")
    s += ("\n**S3 gate:** median `whole_blob_sim` ≥ 0.70 unwhitened. "
          + ("Passed on every subset — the near-duplicate premise holds.\n"
             if all_pass else
             "**FAILED.** Conflict pairs are not near-duplicates in this embedding "
             "space. The geometry half of the thesis is dead; only read-side repair "
             "remains viable.\n"))
    return s, all_pass


def render_m1b(out_dir: Path) -> str:
    data = load(out_dir, "m1b_grouping_ablation.json")
    s = section("M1b — regex-vs-geometry grouping ablation")
    if data is None:
        return s + ("**NOT RUN.** Without it, no downstream gain can be attributed "
                    "to geometry rather than to the benchmark's metadata.\n")

    s += ("| subset | facts | truth pairs | knn recall ceiling | best F1 | τ* | "
          "equal-coverage F1 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in data:
        best, eq = r.get("best_f1", {}), r.get("equal_coverage", {})
        s += (f"| {r.get('subset')} | {r.get('n_facts')} | {r.get('n_truth_pairs')} | "
              f"{fmt(r.get('recall_ceiling_from_knn'))} | {fmt(best.get('f1'))} | "
              f"{fmt(best.get('tau'), '.2f')} | {fmt(eq.get('f1'))} |\n")
    s += "\n> " + data[0].get("interpretation", "") + "\n"
    return s


def render_m2(out_dir: Path) -> str:
    data = load(out_dir, "m2_retrieval_calibration.json")
    s = section("M2 — retrieval calibration")
    if data is None:
        return s + "**NOT RUN.**\n"

    s += ("| subset | questions | top1 p50 | margin p50 | nmargin p50 | H_raw p50 | "
          "H_z p50 | H_vn p50 | eff size | tie rate |\n"
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in data:
        s += (f"| {r.get('subset')} | {r.get('n_questions')} | "
              f"{fmt((r.get('score_scale') or {}).get('p50'), '.2f')} | "
              f"{fmt((r.get('margin') or {}).get('p50'))} | "
              f"{fmt((r.get('nmargin') or {}).get('p50'), '.4f')} | "
              f"{fmt(r.get('median_H_raw'), '.4f')} | "
              f"{fmt(r.get('median_H_z'))} | "
              f"{fmt((r.get('H_vn') or {}).get('p50'))} | "
              f"{fmt((r.get('eff_size') or {}).get('p50'), '.2f')} | "
              f"{fmt(r.get('tie_rate_top_m'))} |\n")

    verdicts = {r.get("raw_entropy_verdict") for r in data}
    s += "\n### Raw-score entropy verdict\n\n"
    if verdicts == {"DEGENERATE"}:
        s += ("**Confirmed degenerate** on `cosine × 100` scores, on every subset. "
              "`H_raw` remains logged and remains barred from every decision path; "
              "`H_z` is primary. This replicates the prior BFCL finding on a "
              "different substrate.\n")
    elif "DEGENERATE" not in verdicts:
        s += ("**Refuted.** Raw-score entropy is *not* degenerate here. This revises "
              "the prior BFCL finding and is a result in its own right. `H_raw` stays "
              "barred from decisions regardless — the bar is pre-registered, and "
              "relaxing it on the strength of the same data that tested it would be "
              "exactly the fishing the protocol forbids.\n")
    else:
        s += ("**Mixed across subsets** — see the table. A verdict that depends on "
              "store size is itself the finding and must be reported stratified.\n")
    s += ("\nAlso reported per subset: `dH_self`, `churn@k` and `rank_shift` across "
          "simulated provisional inserts (see `m2_retrieval_calibration.json`).\n")
    return s


# ── M3 / M4 ──────────────────────────────────────────────────────────────────
def render_m3(out_dir: Path):
    data = load(out_dir, "m3_headroom.json")
    s = section("M3 — headroom")
    if data is None:
        return s + "**NOT RUN.**\n", None

    s += "### Write side\n\n"
    s += ("| subset | split | decisions | conflict | crit-Δ | stale | dup | redundant "
          "| would intervene | after vetoes | could change correctness |\n"
          "| --- | :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in data:
        w = r.get("write", {})
        s += (f"| {r.get('subset')} | {r.get('split')} | {w.get('n_write_decisions')} | "
              f"{fmt(w.get('conflict_rate'))} | {fmt(w.get('critical_delta_rate'))} | "
              f"{fmt(w.get('stale_supersede_rate'))} | {fmt(w.get('duplicate_rate'))} | "
              f"{fmt(w.get('redundant_rate'))} | {fmt(w.get('would_intervene_rate'))} | "
              f"{fmt(w.get('would_intervene_after_vetoes_rate'))} | "
              f"{fmt(w.get('could_change_correctness_rate'))} |\n")

    s += "\n### Counterfactual classes (write side)\n\n"
    s += "| subset | " + " | ".join(
        ["must_write", "must_suppress", "may_suppress", "inert_superseded",
         "uncertain"]) + " |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for r in data:
        c = (r.get("write", {}) or {}).get("counterfactual_classes", {}) or {}
        s += (f"| {r.get('subset')} | " + " | ".join(
            str(c.get(k, 0)) for k in ["must_write", "must_suppress", "may_suppress",
                                       "inert_superseded", "uncertain"]) + " |\n")

    s += "\n### Read side\n\n"
    s += ("| subset | reads | native acc | repaired acc | Δ (pp) | helped | harmed |\n"
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in data:
        rd = r.get("read", {})
        na, ra = rd.get("accuracy_native"), rd.get("accuracy_repaired")
        delta = (ra - na) * 100 if (na is not None and ra is not None) else None
        s += (f"| {r.get('subset')} | {rd.get('n_reads')} | {fmt(na)} | {fmt(ra)} | "
              f"{fmt(delta, '+.2f')} | {rd.get('repair_helped', NA)} | "
              f"{rd.get('repair_harmed', NA)} |\n")

    s += "\n### Read strata\n\n"
    s += ("| subset | label | n | rate | native failure rate | helped | harmed |\n"
          "| --- | :-- | ---: | ---: | ---: | ---: | ---: |\n")
    for r in data:
        for label, st in sorted((r.get("read", {}).get("strata") or {}).items()):
            s += (f"| {r.get('subset')} | {label} | {st.get('n')} | "
                  f"{fmt(st.get('rate'))} | {fmt(st.get('failure_rate_native'))} | "
                  f"{st.get('repair_helped', NA)} | {st.get('repair_harmed', NA)} |\n")

    s += ("\n*`READ_DISTRACTOR` and `READ_MISSING` are computed from the offline "
          "question→key mapping, so their rates are an upper bound on what an online "
          "detector — which would need a query-side parser H-Nav does not build — "
          "could reach.*\n")
    return s, data


def render_m4(out_dir: Path):
    data = load(out_dir, "m4_marginal_diff_test.json")
    s = section("M4 — H2: does marginal-diff geometry add information?")
    if data is None:
        return s + "**NOT RUN.**\n", None
    if data.get("verdict") == "UNTESTABLE":
        return s + (f"**UNTESTABLE.** {data.get('reason')} "
                    f"(n={data.get('n')}, positives={data.get('n_positive')}). "
                    "Report as NO_GO (benchmark), not as a failure of the signal.\n"), data

    b = data.get("bootstrap_subset_clustered", {})
    k = data.get("bootstrap_key_clustered", {})
    cv = data.get("cross_validated", {})
    s += (f"Fit on **{', '.join(data.get('split', []))}** only — the calibration split.\n\n"
          f"| quantity | value |\n| --- | ---: |\n"
          f"| n / positives | {data.get('n')} / {data.get('n_positive')} "
          f"({fmt(data.get('base_rate'))}) |\n"
          f"| AUC base | {fmt(data.get('auc_base'), '.4f')} |\n"
          f"| AUC +diff | {fmt(data.get('auc_diff'), '.4f')} |\n"
          f"| ΔAUC | {fmt(data.get('delta_auc'), '+.4f')} |\n"
          f"| ΔAUC 95% CI (subset-clustered, {b.get('n_clusters', NA)} clusters) | "
          f"[{fmt(b.get('lo'), '+.4f')}, {fmt(b.get('hi'), '+.4f')}] |\n"
          f"| ΔAUC 95% CI (key-clustered, {k.get('n_clusters', NA)} clusters) | "
          f"[{fmt(k.get('lo'), '+.4f')}, {fmt(k.get('hi'), '+.4f')}] |\n"
          f"| ΔAUC cross-validated | {fmt(cv.get('delta_auc'), '+.4f')} |\n"
          f"| LRT χ²({data.get('lrt_df')}) | {fmt(data.get('lrt_statistic'), '.3f')} |\n"
          f"| LRT p | {fmt(data.get('lrt_p'), '.3e')} |\n")
    s += (f"\n**H2 {'PASSES' if data.get('h2_pass') else 'DOES NOT PASS'}** — criterion: "
          f"{data.get('criterion')}\n")
    if (b.get("n_clusters") or 0) < 5:
        s += ("\n*The calibration split has two subsets, so the subset-clustered CI "
              "rests on two clusters and is not a reliable interval alone. Read it "
              "with the key-clustered CI and the cross-validated ΔAUC.*\n")
    return s, data


# ── GO / NO_GO ───────────────────────────────────────────────────────────────
def component_row(name: str, positives, coverage, precision, harm, delta_pp,
                  note: str = "") -> dict:
    p_lb = wilson(int(round((precision or 0) * (positives or 0))),
                  int(positives or 0))[0] if positives else float("nan")
    h_ub = wilson(int(round((harm or 0) * (positives or 0))),
                  int(positives or 0))[1] if positives else float("nan")
    checks = {
        "positives": (positives is not None and positives >= GATE["min_positives"]),
        "coverage": (coverage is not None and coverage >= GATE["min_coverage"]),
        "precision": (precision is not None and precision >= GATE["min_precision"]),
        "precision_lb": (p_lb == p_lb and p_lb >= GATE["min_precision_wilson_lb"]),
        "harm_ub": (h_ub == h_ub and h_ub <= GATE["max_harm_wilson_ub"]),
        "delta_acc": (delta_pp is not None and delta_pp >= GATE["min_delta_acc_pp"]),
    }
    # A criterion that was never measured is NOT a criterion that failed. Keeping
    # the two apart is the difference between "H-Nav's signal does not work" and
    # "we did not run the measurement", which the protocol forbids conflating.
    unmeasured = [k for k, v in (("positives", positives), ("coverage", coverage),
                                 ("precision", precision), ("harm", harm),
                                 ("delta_acc", delta_pp)) if v is None]
    return {"component": name, "positives": positives, "coverage": coverage,
            "precision": precision, "precision_wilson_lb": p_lb,
            "harm": harm, "harm_wilson_ub": h_ub, "delta_acc_pp": delta_pp,
            "checks": checks, "unmeasured": unmeasured,
            "pass": all(checks.values()), "note": note}


def derive_components(m3, m4) -> list[dict]:
    """Compute gate inputs for A1 / A2 / A3 from the calibration split only."""
    rows = []
    if not m3:
        return rows
    cal = [r for r in m3 if r.get("subset") in CALIBRATION] or m3

    # A1 — geometry-only write admission
    n_dec = sum((r.get("write", {}) or {}).get("n_write_decisions", 0) for r in cal)
    cov = [(r.get("write", {}) or {}).get("would_intervene_after_vetoes_rate")
           for r in cal]
    cov = [c for c in cov if c is not None and c == c]
    coverage = sum(cov) / len(cov) if cov else None
    positives = int(round((coverage or 0) * n_dec)) if coverage is not None else None
    graded = [(r.get("write", {}) or {}).get("could_change_correctness_rate")
              for r in cal]
    graded = [g for g in graded if g is not None]
    rows.append(component_row(
        "A1 — geometry-only write admission", positives, coverage,
        precision=None, harm=None, delta_pp=None,
        note=("precision / harm / ΔAcc need the counterfactual labels for the "
              "candidates the gate fires on; " +
              ("no such candidate was graded" if not graded else
               f"could-change-correctness = {graded[0]:.3f}"))))

    # A3 — read-side stale repair
    n_reads = sum((r.get("read", {}) or {}).get("n_reads", 0) for r in cal)
    helped = sum((r.get("read", {}) or {}).get("repair_helped", 0) for r in cal)
    harmed = sum((r.get("read", {}) or {}).get("repair_harmed", 0) for r in cal)
    target = 0
    for r in cal:
        st = (r.get("read", {}) or {}).get("strata", {}) or {}
        target += (st.get("READ_RELEVANT_BELOW_K", {}) or {}).get("n", 0)
    changed = helped + harmed
    na = [(r.get("read", {}) or {}).get("accuracy_native") for r in cal]
    ra = [(r.get("read", {}) or {}).get("accuracy_repaired") for r in cal]
    na = [v for v in na if v is not None]
    ra = [v for v in ra if v is not None]
    delta_pp = ((sum(ra) / len(ra)) - (sum(na) / len(na))) * 100 if na and ra else None
    rows.append(component_row(
        # a measured zero is a measured value: 0 targeted reads is a benchmark
        # finding, not a missing measurement
        "A3 — read-side stale repair", positives=(target if n_reads else None),
        coverage=(target / n_reads) if n_reads else None,
        precision=(helped / changed) if changed else None,
        harm=(harmed / target) if target else None,
        delta_pp=delta_pp,
        note=f"{helped} helped / {harmed} harmed out of {target} targeted reads"))

    # A2 — marginal-diff-aware geometry, gated on H2
    if m4 is None:
        rows.append({"component": "A2 — marginal-diff critical-delta veto",
                     "pass": False, "checks": {}, "note": "M4 not run",
                     "positives": None, "coverage": None, "precision": None,
                     "precision_wilson_lb": float("nan"), "harm": None,
                     "harm_wilson_ub": float("nan"), "delta_acc_pp": None})
    else:
        a2 = component_row("A2 — marginal-diff critical-delta veto",
                           positives=m4.get("n_positive"),
                           coverage=m4.get("base_rate"),
                           precision=None, harm=None, delta_pp=None,
                           note="runs only if H2 passes; "
                                f"H2 {'passed' if m4.get('h2_pass') else 'did not pass'}")
        a2["pass"] = bool(a2["pass"] and m4.get("h2_pass"))
        rows.append(a2)
    return rows


def verdict_short(c: dict) -> str:
    if c.get("pass"):
        return "GO"
    checks = c.get("checks", {})
    measured_absence = (c.get("positives") is not None and c.get("coverage") is not None
                        and (not checks.get("positives") or not checks.get("coverage")))
    if c.get("unmeasured") and not measured_absence:
        return "NOT ASSESSABLE"
    return "NO_GO"


def classify_no_go(row: dict, m3, m4) -> str:
    """Tag a NO_GO with verdict type 1 (benchmark), 2 (detection) or 3 (policy)."""
    checks = row.get("checks", {})
    if not checks:
        return "NOT ASSESSABLE — the measurement it depends on was not run"

    unmeasured = row.get("unmeasured", [])
    # Base rate is measurable without any counterfactual, so a genuine
    # benchmark-side absence can still be declared even when the rest is missing.
    if row.get("positives") is not None and row.get("coverage") is not None and (
            not checks.get("positives") or not checks.get("coverage")):
        return ("**NO_GO (benchmark)** — the class is too rare or absent here. "
                "*EvoMemBench does not meaningfully generate this failure class.* "
                "This is not evidence against H-Nav.")
    if unmeasured:
        return ("**NOT ASSESSABLE** — " + ", ".join(unmeasured) + " was not measured. "
                "This is not a NO_GO of any kind: an unmeasured criterion must never "
                "be reported as a failed one.")
    if not checks.get("positives") or not checks.get("coverage"):
        return ("**NO_GO (benchmark)** — the class is too rare or absent here. "
                "*EvoMemBench does not meaningfully generate this failure class.* "
                "This is not evidence against H-Nav.")
    if not checks.get("precision") or not checks.get("precision_lb"):
        return ("**NO_GO (detection)** — the class is abundant but the signal does "
                "not predict it. This IS genuine evidence against H-Nav.")
    if not checks.get("delta_acc") or not checks.get("harm_ub"):
        return ("**NO_GO (policy)** — detection succeeds, the proposed intervention "
                "does not repair. Detection and policy must be reported separately.")
    return "GO"


def render_gate(m3, m4) -> str:
    s = section("GO / NO_GO — against the frozen gate (protocol §4)")
    s += ("Thresholds, frozen before any measurement: positives ≥ "
          f"{GATE['min_positives']}, coverage ≥ {GATE['min_coverage']:.0%}, "
          f"precision ≥ {GATE['min_precision']:.2f}, Wilson LB ≥ "
          f"{GATE['min_precision_wilson_lb']:.2f}, harm 95% UB ≤ "
          f"{GATE['max_harm_wilson_ub']:.2f}, ΔAcc ≥ {GATE['min_delta_acc_pp']:.0f} pp. "
          "Evaluated on the **calibration split**.\n\n")

    comps = derive_components(m3, m4)
    if not comps:
        return s + "**NOT ASSESSABLE.** M3 has not been run.\n"

    s += ("| component | positives | coverage | precision | Wilson LB | harm | "
          "harm UB | ΔAcc (pp) | verdict |\n"
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-- |\n")
    for c in comps:
        s += (f"| {c['component']} | {c['positives'] if c['positives'] is not None else NA} | "
              f"{fmt(c['coverage'])} | {fmt(c['precision'])} | "
              f"{fmt(c['precision_wilson_lb'])} | {fmt(c['harm'])} | "
              f"{fmt(c['harm_wilson_ub'])} | {fmt(c['delta_acc_pp'], '+.2f')} | "
              f"{verdict_short(c)} |\n")

    s += "\n### Verdict per component\n\n"
    for c in comps:
        s += f"**{c['component']}** — {classify_no_go(c, m3, m4)}\n\n"
        if c.get("note"):
            s += f"> {c['note']}\n\n"
    return s


# ── main ─────────────────────────────────────────────────────────────────────
def build_report(out_dir: Path) -> tuple[str, list[str]]:
    missing: list[str] = []
    parts = ["# EvoMemBench × H-Nav — Stage-0 Report\n",
             "\nGenerated by `hnav/stage0/report.py` from the measurement files in "
             "`HNAV_OUT_DIR`. Every number here is read from disk; nothing is "
             "restated from a document.\n",
             "\n**This report ends Stage 0.** A human evaluates the gate below. "
             "`hnav/core/write_policy.py` and `hnav/core/read_policy.py` do not exist "
             "and must not be written until that evaluation has happened.\n"]

    m0, _ = render_m0(out_dir)
    parts.append(m0)
    if load(out_dir, "m0_replica_fidelity.json") is None:
        missing.append("M0 (live-index fidelity)")

    m1, _ = render_m1(out_dir)
    parts.append(m1)
    if load(out_dir, "m1_geometry_calibration.json") is None:
        missing.append("M1")

    parts.append(render_m1b(out_dir))
    if load(out_dir, "m1b_grouping_ablation.json") is None:
        missing.append("M1b")

    parts.append(render_m2(out_dir))
    if load(out_dir, "m2_retrieval_calibration.json") is None:
        missing.append("M2")

    m3_text, m3 = render_m3(out_dir)
    parts.append(m3_text)
    if m3 is None:
        missing.append("M3")

    m4_text, m4 = render_m4(out_dir)
    parts.append(m4_text)
    if m4 is None:
        missing.append("M4")

    parts.append(render_gate(m3, m4))

    parts.append(section("Status"))
    if missing:
        parts.append("**INCOMPLETE.** Not run: " + ", ".join(missing) +
                     ". No component may be taken to a live arm on a partial "
                     "report.\n")
    else:
        parts.append("All Stage-0 measurements are present. The gate above is ready "
                     "for human evaluation.\n")
    parts.append("\n---\n\n**STOP.** Do not implement write/read policies and do not "
                 "run live arms before this report's gate has been evaluated by a "
                 "human (brief §3 T8, protocol §7).\n")
    return "".join(parts), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="report path (default: repo root)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any measurement is missing")
    args = ap.parse_args()

    cfg = _config.get_config()
    cfg.require_not_live()   # uniform Stage-0 guard; see test_stage0_refuses_live.py
    text, missing = build_report(cfg.out_dir)
    path = Path(args.out) if args.out else REPO / "STAGE0_REPORT.md"
    path.write_text(text, encoding="utf-8")

    print(f"wrote {path}")
    if missing:
        print("MISSING: " + ", ".join(missing))
        return 1 if args.strict else 0
    print("all Stage-0 measurements present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
