"""Pipeline driver — Stages 0-7, fixed seeds, gate rules from PREREG.md.

Usage:
    python -m hnav.qda_filter.run_all              # everything but the null
    python -m hnav.qda_filter.run_all --only-null  # 200-repeat pipeline null
                                                   # (hours; merges into
                                                   # eval.json when done)
    python -m hnav.qda_filter.run_all --quick      # smoke: tiny perms/boots,
                                                   # writes *_SMOKE.json

Every stage writes its JSON under stage0_results/qda_filter/ with the same
provenance block the geometry_filter results carry. Gates are decided by the
preregistered rules only; a failed gate is reported, never patched around.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hnav.geometry_filter import data as gfdata
from hnav.geometry_filter.metrics import auroc, paired_bootstrap_delta_auc

from . import adapt as ad
from . import calibrate as cb
from . import eval as ev
from . import fit as ft
from . import preprocess as pp
from . import score as sc
from . import spectrum as sp

OUT = sc.OUT_DIR
SEED = 20260824
N_PERM_PA = 200
N_FLIPS = 2000
N_NULL_REPEATS = 200
N_PERM_PA_NULL = 50


def _write(name: str, payload: dict, quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (name.replace(".json", "_SMOKE.json") if quick else name)
    p.write_text(json.dumps(payload, indent=1, default=float),
                 encoding="utf-8")
    print(f"  wrote {p.name}")


def _prov(**extra) -> dict:
    return gfdata.provenance(pipeline="qda_filter", **extra)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-null", action="store_true")
    ap.add_argument("--null-repeats", type=int, default=N_NULL_REPEATS)
    args = ap.parse_args(argv)
    quick = args.quick
    n_pa = 20 if quick else N_PERM_PA
    n_flips = 100 if quick else N_FLIPS
    n_stab = 5 if quick else 50

    t0 = time.time()
    print("Stage 0/1 — bundle + discovery")
    b = pp.Bundle()

    if args.only_null:
        print(f"pipeline permutation null: {args.null_repeats} repeats "
              f"(inner PA perms {N_PERM_PA_NULL})")
        null = ev.pipeline_permutation_null(
            b, n_repeats=args.null_repeats, n_perm_inner=N_PERM_PA_NULL)
        path = OUT / "eval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pipeline_permutation_null"] = null
        path.write_text(json.dumps(payload, indent=1, default=float),
                        encoding="utf-8")
        print("  merged pipeline_permutation_null into eval.json")
        print(f"  p_balanced={null['p_balanced']:.4g} "
              f"p_hard={null['p_hard']:.4g}")
        return 0

    disc = pp.discovery(b)
    disc["provenance"] = _prov(stage="discovery")
    _write("discovery.json", disc, quick)

    # ── Stage 2 — whitening + spectrum ──────────────────────────────────────
    print("Stage 2 — Ledoit-Wolf whitening + gold spectrum + parallel "
          "analysis")
    idx_a = np.flatnonzero(b.neg_half_a)
    idx_b = np.flatnonzero(b.neg_half_b)
    idx_g = np.flatnonzero(b.fit_gold)
    W0, lw_info = sp.ledoit_wolf_whitener(b.D_t[idx_a].astype(np.float64))
    Z_gold = b.whiten(W0, idx_g)
    Z_halfb = b.whiten(W0, idx_b)
    spec = sp.fit_spectrum(Z_gold, Z_halfb, n_perm=n_pa, seed=SEED)
    lam_b, _ = sp.nontrivial_spectrum(Z_halfb)
    n1, dim = Z_gold.shape
    spec_json = {
        "provenance": _prov(stage="spectrum", n_perm=n_pa,
                            n1_gold=n1, n_prime=dim),
        "ledoit_wolf": lw_info,
        "k_obj": spec["k_obj"], "k_subj": spec["k_subj"],
        "sigma1_sq": spec["sigma1sq"],
        "nontrivial_rank": int(len(spec["lam"])),
        "rank_deficiency_note": "n1 < N': spectra compared on the nontrivial "
                                "part only (PREREG deviation, data-forced)",
        "lam_top64": spec["lam"][:64].tolist(),
        "lam_bottom64": spec["lam"][-64:].tolist(),
        "null95_top64": spec["null95_top"][:64].tolist(),
        "null05_bot64": spec["null05_bot"][:64].tolist(),
        "mp_edge_reference": spec["mp_edge_reference"],
        "halfb_whitened_sanity": {
            "n": int(len(idx_b)),
            "top5": lam_b[:5].tolist(), "bottom5": lam_b[-5:].tolist(),
            "median": float(np.median(lam_b)),
            "note": "should straddle 1 within the Marchenko-Pastur bulk; "
                    "gross departure means the whitening failed"},
    }
    _write("spectrum.json", spec_json, quick)
    print(f"  k_obj={spec['k_obj']} k_subj={spec['k_subj']} "
          f"sigma1^2={spec['sigma1sq']:.4f}")

    # ── Stage 3 — ordered term ──────────────────────────────────────────────
    print("Stage 3 — sign-flip test of the ordered term")
    test1 = ft.signflip_test(Z_gold, n_flips=n_flips, seed=SEED)
    Z_negAB = np.vstack([b.whiten(W0, idx_a), Z_halfb])
    test0 = ft.signflip_test(Z_negAB, n_flips=n_flips, seed=SEED)
    ordered_candidate = test1["p_signflip"] < 0.01
    _write("ordered_term.json", {
        "provenance": _prov(stage="ordered_term", n_flips=n_flips),
        "mu1_test": test1, "mu0_test": test0,
        "mu0_source": "fit negatives, serial-oriented (order IS available "
                      "for negatives — see discovery.json)",
        "g3_signflip_condition_met": bool(ordered_candidate),
        "note": "G3's second condition (V3 held-out >= V2 held-out) is "
                "decided in fit-stage output"}, quick)
    print(f"  p_signflip(mu1)={test1['p_signflip']:.4g} "
          f"p_signflip(mu0)={test0['p_signflip']:.4g}")

    # ── Stage 4 — variants ──────────────────────────────────────────────────
    print("Stage 4 — score variants (whitening all pairs)")
    mu1 = Z_gold.mean(axis=0)
    mu0 = Z_negAB.mean(axis=0)
    model = ft.QDAModel(W0, spec, mu1, mu0, ordered_on=ordered_candidate)

    n_all = len(b.records)
    Z_all = np.empty((n_all, dim), dtype=np.float32)
    for s in range(0, n_all, 4096):
        e = min(s + 4096, n_all)
        Z_all[s:e] = (b.D_t[s:e].astype(np.float64) @ W0.T).astype(np.float32)
    lognorm = np.log(np.maximum(b.norm_dt, 1e-12))

    # V0 — CES exactly as run_dimension_ideas fits it (imported)
    from hnav.geometry_filter.methods import fit_training_edits
    from hnav.geometry_filter.run_dimension_ideas import ContrastiveSubspace
    V = b.V_raw
    D_pos, rel_pos = fit_training_edits(b.records, b.pv, V, b.y & b.cal)
    pv_hn = b.pv.subset(b.hardneg & b.cal)
    D_hn = pv_hn.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace(k=20).fit(D_pos, rel_pos, D_hn,
                                        list(pv_hn.relation))
    scores: dict[str, np.ndarray] = {"V0": ces.score(b.pv, V)}

    Dhat = b.D_t.astype(np.float64) / b.norm_dt[:, None]
    scores["V1"] = ft.QDAModel.v1_quantized(Dhat, spec["U_obj"],
                                            spec["U_subj"])
    del Dhat
    core_all = model.core(Z_all)
    ordered_all = model.ordered(Z_all)
    scores["V2"] = core_all

    # G3 second condition: V3 held-out (balanced sh_64k) >= V2
    m64 = b.in_eval & (b.subset == "sh_64k")
    y64 = b.y[m64]
    v3_all = core_all + ordered_all
    g3_holdout_ok = (auroc(v3_all[m64][y64], v3_all[m64][~y64])
                     >= auroc(core_all[m64][y64], core_all[m64][~y64]))
    g3_pass = bool(ordered_candidate and g3_holdout_ok)
    scores["V3"] = v3_all if g3_pass else core_all.copy()
    if not g3_pass:
        model.ordered_on = False
        ordered_all = np.zeros_like(ordered_all)   # V5's feature honors G3

    fit_m = b.fit_gold | b.fit_neg
    beta_info = ft.fit_beta_norm(scores["V3"][fit_m], lognorm[fit_m],
                                 b.y[fit_m])
    scores["V4"] = scores["V3"] + beta_info["beta"] * lognorm
    v5_model, v5_info = ft.fit_v5(core_all[fit_m], ordered_all[fit_m],
                                  lognorm[fit_m], b.y[fit_m], seed=0)
    X5 = np.column_stack([core_all, ordered_all, lognorm])
    scores["V5"] = v5_model.decision_function(X5)

    # continuity reference rows
    spaces = gfdata.build_spaces(b.V_raw)
    scores["abtt_cos"] = b.pv.cos(spaces["abtt"])
    scores["campaign_cos"] = b.cos.copy()

    # G1 — quantized core vs CES on balanced sh_64k. The pooled global CES
    # (same subspace count, no relation identity) is the apples-to-apples
    # reference for the discrepancy report the gate's fallback requires.
    ii64 = np.flatnonzero(m64)
    Dh64 = V[b.pv.ib[ii64]] - V[b.pv.ia[ii64]]      # CES lives in RAW space
    Dh64 /= np.maximum(np.linalg.norm(Dh64, axis=1, keepdims=True), 1e-12)
    ces_g64 = (np.linalg.norm(Dh64 @ ces.U_obj_g, axis=1) ** 2
               - np.linalg.norm(Dh64 @ ces.U_subj_g, axis=1) ** 2)
    del Dh64
    a_v1 = auroc(scores["V1"][m64][y64], scores["V1"][m64][~y64])
    a_v0 = auroc(scores["V0"][m64][y64], scores["V0"][m64][~y64])
    g1 = {"V1_balanced_sh64k": a_v1, "V0_balanced_sh64k": a_v0,
          "V0_global_pooled_balanced_sh64k": auroc(ces_g64[y64],
                                                   ces_g64[~y64]),
          "abs_diff": abs(a_v1 - a_v0), "pass": bool(abs(a_v1 - a_v0) <= 0.010),
          "note": "V1 is the POOLED whitened-space quantization; V0 is the "
                  "relation-aware raw-space CES — the preregistered "
                  "comparison as stated in the task. The pooled-CES row "
                  "shows how much of any V1-V0 gap is relation identity, "
                  "not whitening."}

    # G2 — V2 vs CES, balanced + band
    band = (b.cos >= 0.87) & (b.cos <= 0.97)
    mb = m64 & band
    yb = b.y[mb]
    d_bal = paired_bootstrap_delta_auc(
        scores["V2"][m64][y64], scores["V2"][m64][~y64],
        scores["V0"][m64][y64], scores["V0"][m64][~y64], 1000, SEED)
    band_v2 = auroc(scores["V2"][mb][yb], scores["V2"][mb][~yb])
    band_v0 = auroc(scores["V0"][mb][yb], scores["V0"][mb][~yb])
    half = (d_bal["hi"] - d_bal["lo"]) / 2.0
    g2 = {"delta_balanced": d_bal, "ci_half_width": half,
          "band_V2": band_v2, "band_V0": band_v0,
          "pass": bool(d_bal["delta"] > half and band_v2 >= band_v0)}

    # save the frozen artifact
    arrays = {
        "M": (W0 @ b.Q).astype(np.float32),
        "A": None,  # filled below
        "U_obj": spec["U_obj"].astype(np.float32),
        "U_subj": spec["U_subj"].astype(np.float32),
        "w_obj": model.w_obj.astype(np.float32),
        "w_subj": model.w_subj.astype(np.float32),
        "ordered_coef": model.ordered_coef.astype(np.float32),
    }
    from hnav.core.geometry import ABTTWhitening
    blob = json.loads(gfdata.ABTT_ARTIFACT.read_text(encoding="utf-8"))
    arrays["A"] = np.asarray(
        ABTTWhitening.from_dict(blob["whitening"]).components,
        dtype=np.float32)
    scalars = {"w_perp": model.w_perp, "beta": beta_info["beta"],
               "ordered_on": bool(model.ordered_on),
               "k_obj": spec["k_obj"], "k_subj": spec["k_subj"],
               "sigma1_sq": spec["sigma1sq"], "dtype": "float32"}
    if not quick:
        sc.save_artifact(arrays, scalars, _prov(stage="weights"))
        print("  wrote weights.npz + weights_manifest.json")

    _write("fit.json", {
        "provenance": _prov(stage="fit"),
        "gates": {"G1": g1, "G2": g2,
                  "G3": {"signflip": bool(ordered_candidate),
                         "holdout_ok": bool(g3_holdout_ok),
                         "pass": g3_pass,
                         "rule": "V3 := V2 when failed"}},
        "beta": beta_info, "v5": v5_info,
        "ordered_term_in_score": g3_pass,
    }, quick)
    print(f"  G1 pass={g1['pass']} (|diff|={g1['abs_diff']:.4f})  "
          f"G2 pass={g2['pass']} (delta={d_bal['delta']:+.4f})  "
          f"G3 pass={g3_pass}")

    # ── Stage 4b — relation-gated mixture ───────────────────────────────────
    print("Stage 4b — relation-gated mixture")
    eligible = [s for s in ("sh_6k", "sh_32k", "sh_64k")
                if disc["per_subset"][s].get("stage4b_eligible")]
    rg_json = {"provenance": _prov(stage="relation_gate"),
               "eligible_subsets": eligible}
    if eligible and not quick:
        rel_gold = [b.relation[i] for i in idx_g]
        gate_idx = np.flatnonzero(
            (b.fit_gold | b.fit_neg)
            & np.array([r is not None for r in b.relation]))
        M_gate = b.midpoints_t(gate_idx).astype(np.float64) @ W0.T
        rel_gate = [b.relation[i] for i in gate_idx]
        mix = ft.RelationMixture(model, min_rel_gold=max(50, 3 * spec["k_obj"]))
        mix.fit(Z_gold, rel_gold, M_gate, rel_gate, seed=0)
        rg_json.update({
            "n_relations_own_sigma": len(mix.rel_low),
            "own_sigma_threshold": max(50, 3 * spec["k_obj"]),
            "n_gate_classes": len(mix.classes),
            "gate_kind": mix.gate_kind, "gate_cv_acc": mix.gate_cv_acc})
        v4b = {}
        for sub in eligible:
            m = b.in_eval & (b.subset == sub)
            ii = np.flatnonzero(m)
            Zi = b.whiten(W0, ii)
            Mi = b.midpoints_t(ii).astype(np.float64) @ W0.T
            s4b = mix.score(Zi, Mi)
            y = b.y[ii]
            bd = (b.cos[ii] >= 0.87) & (b.cos[ii] <= 0.97)
            row = {"auroc": auroc(s4b[y], s4b[~y]),
                   "band_auroc": auroc(s4b[bd][y[bd]], s4b[bd][~y[bd]])}
            if sub == "sh_64k":
                d = paired_bootstrap_delta_auc(
                    s4b[y], s4b[~y],
                    scores["V2"][ii][y], scores["V2"][ii][~y], 1000, SEED)
                db = paired_bootstrap_delta_auc(
                    s4b[bd][y[bd]], s4b[bd][~y[bd]],
                    scores["V2"][ii][bd][y[bd]], scores["V2"][ii][bd][~y[bd]],
                    1000, SEED)
                row["delta_vs_V2"] = d
                row["band_delta_vs_V2"] = db
            v4b[sub] = row
        rg_json["balanced"] = v4b
        # G5 decided after relation-disjoint eval below (needs both parts)
        rg_json["note"] = ("scored on eval sets only; a full-pool scoring "
                           "is not needed for the gates")
        b._mix = mix  # stashed for the relation-disjoint stage
    else:
        rg_json["skipped"] = ("quick mode" if quick else
                              "no subset with n1 >= 200")
        b._mix = None
    _write("relation_gate.json", rg_json, quick)

    # ── Stage 5 — evaluation ────────────────────────────────────────────────
    print("Stage 5 — evaluation")
    eval_json = {"provenance": _prov(stage="eval", n_boot=ev.N_BOOT),
                 "balanced": ev.balanced_eval(b, scores),
                 "hard_confirmatory": ev.hard_task_eval(b, scores),
                 "tail_seen_unseen": ev.tail_eval(b, scores),
                 "cosine_strata_sh64k": ev.cosine_strata(b, scores),
                 "subspace_stability": ev.subspace_stability(
                     Z_gold, spec["U_obj"], spec["k_obj"], n_boot=n_stab),
                 "relation_disjoint": ev.relation_disjoint(
                     b, n_perm_inner=10 if quick else 50),
                 "pipeline_permutation_null": {
                     "status": "run separately: "
                               "python -m hnav.qda_filter.run_all --only-null"}}

    # 4b on the relation-disjoint protocol (refit inside folds is what the
    # pooled relation_disjoint() already measures for V2/V3; the mixture's
    # transfer is approximated by scoring the frozen fit-split mixture on
    # the disjoint-fold CALIBRATION pairs it never saw a subspace for)
    if b._mix is not None:
        own = set(b._mix.rel_low)
        disjoint_m = (b.cal & (b.y | b.hardneg)
                      & np.array([r is not None and r not in own
                                  for r in b.relation]))
        ii = np.flatnonzero(disjoint_m)
        if len(ii):
            Zi = b.whiten(W0, ii)
            Mi = b.midpoints_t(ii).astype(np.float64) @ W0.T
            s4b = b._mix.score(Zi, Mi)
            y = b.y[ii]
            v2s = scores["V2"][ii]
            d = paired_bootstrap_delta_auc(s4b[y], s4b[~y],
                                           v2s[y], v2s[~y], 1000, SEED)
            eval_json["relation_disjoint_4b"] = {
                "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                "note": "cal pairs whose relation has no own-Sigma in the "
                        "mixture (gate must route them from midpoints alone)",
                "auroc_4b": auroc(s4b[y], s4b[~y]),
                "auroc_V2": auroc(v2s[y], v2s[~y]),
                "delta_vs_V2": d}
    _write("eval.json", eval_json, quick)

    # G5 verdict now that both halves exist
    if b._mix is not None and "balanced" in rg_json:
        r64 = rg_json["balanced"].get("sh_64k", {})
        d = r64.get("delta_vs_V2", {})
        db = r64.get("band_delta_vs_V2", {})
        hw = (d.get("hi", 0) - d.get("lo", 0)) / 2 if d else None
        hwb = (db.get("hi", 0) - db.get("lo", 0)) / 2 if db else None
        rd = eval_json.get("relation_disjoint_4b", {})
        rd_d = rd.get("delta_vs_V2", {})
        rd_hw = (rd_d.get("hi", 0) - rd_d.get("lo", 0)) / 2 if rd_d else 0
        g5_pass = bool(d and db
                       and d["delta"] > hw and db["delta"] > hwb
                       and (not rd_d or rd_d["delta"] > -rd_hw))
        rg_json["G5"] = {"pass": g5_pass,
                         "seen_delta": d, "seen_band_delta": db,
                         "relation_disjoint_delta": rd_d}
        _write("relation_gate.json", rg_json, quick)
        print(f"  G5 pass={g5_pass}")

    # ── Stage 6 — calibration ───────────────────────────────────────────────
    print("Stage 6 — conformal thresholds + chi^2 tail")
    conf_idx = np.flatnonzero(b.conformal_neg)
    Z_conf = b.whiten(W0, conf_idx)
    obj_e = np.einsum("ij,ij->i", Z_conf @ spec["U_obj"],
                      Z_conf @ spec["U_obj"])
    cal_json = {"provenance": _prov(stage="calibration",
                                    n0_cal=int(len(conf_idx)))}
    for var in ("V2", "V4"):
        cal_json[f"conformal_{var}"] = cb.conformal_thresholds(
            scores[var][conf_idx])
    cal_json["chi2_check"] = cb.chi2_tail_check(obj_e, spec["k_obj"])
    _write("calibration.json", cal_json, quick)

    # ── Stage 7 — label-free adaptation ─────────────────────────────────────
    print("Stage 7 — adaptation (sh_32k -> sh_64k)")
    tgt64 = b.subset == "sh_64k"
    adn = ad.adapt_nuisance(b, W0, tgt64, b.neg_half_b)
    ii64 = np.flatnonzero(m64)
    Za = ad.whiten_adapted(b, W0, ii64, adn)
    v2_ad = model.core(Za) + model.ordered(Za)
    v2_base = scores["V3"][ii64]
    y = b.y[ii64]
    d_ad = paired_bootstrap_delta_auc(v2_ad[y], v2_ad[~y],
                                      v2_base[y], v2_base[~y], 1000, SEED)
    hw_ad = (d_ad["hi"] - d_ad["lo"]) / 2
    g6_nuis = bool(d_ad["delta"] >= -hw_ad)

    m32 = b.subset == "sh_32k"
    em = ad.PrevalenceEM(scores["V2"][m32 & b.negative],
                         scores["V2"][m32 & b.y])
    pi_all = em.estimate(scores["V2"][tgt64])
    true_pi = float(b.y[tgt64].mean())
    val = ad.validate_prevalence(em, scores["V2"][b.y & tgt64],
                                 scores["V2"][b.hardneg & tgt64])

    pool = np.flatnonzero(tgt64 & (b.y | b.hardneg))
    M_pool = (b.midpoints_t(pool).astype(np.float64) @ W0.T).astype(np.float32)
    Dh_pool = (b.D_t[pool].astype(np.float64)
               / b.norm_dt[pool][:, None]).astype(np.float32)
    coh = ad.local_coherence(M_pool, Dh_pool, K=16)
    y_p = b.y[pool]
    seen_p = np.array([b.trans_keys[i] in b.cal_transitions
                       if b.trans_keys[i] is not None else False
                       for i in pool])
    v2_p = scores["V2"][pool]
    fused = ad.rank_average(coh, v2_p)

    def _sl(mask_pos):
        sel = mask_pos | ~y_p
        return {"coherence": auroc(coh[sel & y_p], coh[sel & ~y_p]),
                "V2": auroc(v2_p[sel & y_p], v2_p[sel & ~y_p]),
                "fused": auroc(fused[sel & y_p], fused[sel & ~y_p]),
                "n_pos": int((sel & y_p).sum())}

    rels = sorted({r for r in b.relation if r is not None})
    fold_of = gfdata.relation_fold(rels, n_folds=2)
    rel_fold_p = np.array([fold_of.get(b.relation[i], -1) for i in pool])
    coh_slices = {
        "seen_transition": _sl(y_p & seen_p),
        "unseen_transition": _sl(y_p & ~seen_p),
        "relation_fold_0": None, "relation_fold_1": None,
        "all": {"coherence": auroc(coh[y_p], coh[~y_p]),
                "V2": auroc(v2_p[y_p], v2_p[~y_p]),
                "fused": auroc(fused[y_p], fused[~y_p])},
    }
    for f in (0, 1):
        sel = rel_fold_p == f
        if (y_p[sel].any()) and ((~y_p[sel]).any()):
            coh_slices[f"relation_fold_{f}"] = {
                "coherence": auroc(coh[sel & y_p], coh[sel & ~y_p]),
                "V2": auroc(v2_p[sel & y_p], v2_p[sel & ~y_p]),
                "fused": auroc(fused[sel & y_p], fused[sel & ~y_p])}

    # G6 for coherence: paired deltas of fused vs V2 on the named slices
    def _delta(mask_pos):
        sel = mask_pos | ~y_p
        return paired_bootstrap_delta_auc(
            fused[sel & y_p], fused[sel & ~y_p],
            v2_p[sel & y_p], v2_p[sel & ~y_p], 1000, SEED)

    d_unseen = _delta(y_p & ~seen_p)
    d_seen = _delta(y_p & seen_p)
    hw_u = (d_unseen["hi"] - d_unseen["lo"]) / 2
    hw_s = (d_seen["hi"] - d_seen["lo"]) / 2
    g6_coh = bool(d_unseen["delta"] > hw_u and d_seen["delta"] >= -hw_s)

    _write("adaptation.json", {
        "provenance": _prov(stage="adaptation"),
        "nuisance": {**{k: v for k, v in adn.items() if k != "mu_t"},
                     "mu_t_norm": float(np.linalg.norm(adn["mu_t"])),
                     "target_auroc_adapted": auroc(v2_ad[y], v2_ad[~y]),
                     "target_auroc_frozen": auroc(v2_base[y], v2_base[~y]),
                     "delta": d_ad, "G6_keep": g6_nuis},
        "prevalence": {"pi_hat_sh64k_all_records": pi_all,
                       "dataset_frame_true_pi": true_pi,
                       "f1_in_sample_note": "f0/f1 are sh_32k KDEs; f1 is "
                                            "in-sample for the fit split",
                       "validation": val},
        "local_coherence": {"K": 16, "slices": coh_slices,
                            "fused_delta_unseen": d_unseen,
                            "fused_delta_seen": d_seen,
                            "G6_fuse": g6_coh},
    }, quick)
    print(f"  G6 nuisance keep={g6_nuis}  G6 coherence fuse={g6_coh}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
