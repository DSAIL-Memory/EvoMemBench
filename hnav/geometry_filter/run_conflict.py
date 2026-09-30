"""Experiment 3 — conflict detection: RCED / RCESP vs cosine, held out properly.

Fit discipline (hard): every estimated quantity — mu_r, U_r, k, thresholds —
is fit on CALIBRATION gold pairs only (sh_6k + sh_32k). Confirmatory (sh_64k)
numbers are computed once, with everything frozen. Calibration-side metrics on
the same pairs the fit saw are labeled ``in_sample`` and are diagnostics, not
results.

Tasks
  balanced      the committed cosine-matched balanced eval set, per subset,
                plus the 0.87–0.97 overlap band where cosine is uninformative
  hard_negative gold_update positives vs same-relation different-subject
                verified non-conflicts (the adversary class), incl. the
                cosine-inverted comparison set where cosine's win rate is 0
  operating     threshold = best F1 on the calibration hard-negative task,
                frozen, applied to confirmatory
  generalization
                transition-disjoint / subject-disjoint confirmatory positives;
                relation-disjoint two-fold within calibration (global variants
                only — relation-conditioned scoring is undefined on unseen
                relations); transition-deduped refit ablation

Methods per space (raw / centered / abtt)
  cos_space     in-space cosine (in raw this IS the campaign cosine; delta
                norm = sqrt(2-2cos) is a monotone transform of it and is
                therefore not a separate method — stated, not duplicated)
  rced          |d_hat · mu_r|, global-mu fallback counted
  rced_max      max_r |d_hat · mu_r| (no relation identity at inference)
  rced_global   |d_hat · mu_global|
  rcesp         ||U_r^T d||/||d||, k selected on calibration
  rcesp_global  global subspace, same k selection
  lda_global    LDA on the k_global RCESP coordinates (covariance-aware,
                only meaningful if RCESP itself shows signal)

Parser baseline: ``same_key`` (relation+subject match) is binary — reported as
precision/recall on each task, not as an AUC.

Usage:  python -m hnav.geometry_filter.run_conflict
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .methods import RCED, RCESP, fit_training_edits
from .metrics import (auprc, auroc, best_f1_threshold, bootstrap_ci,
                      inverted_win_rate, paired_bootstrap_delta_auc, prf_at)

K_GRID = (1, 3, 5, 10, 20)
BAND = (0.87, 0.97)
N_BOOT = 1000


def _task_metrics(scores, is_pos, seed, with_ci=("auroc",)) -> dict:
    pos, neg = scores[is_pos], scores[~is_pos]
    out = {"n_pos": int(is_pos.sum()), "n_neg": int((~is_pos).sum()),
           "auroc": auroc(pos, neg), "auprc": auprc(pos, neg)}
    if "auroc" in with_ci:
        out["auroc_ci95"] = bootstrap_ci(auroc, pos, neg, N_BOOT, seed)
    return out


def run() -> dict:
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    pv = data.PairView(records, index)

    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    in_eval = np.array([r["in_eval_set"] for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])
    camp_cos = np.array([r["cosine_similarity"] for r in records])
    subset = np.array([r["subset"] for r in records], object)
    trans_keys = [data.transition_key(r) for r in records]

    cal_subjects, cal_transitions = data.calibration_positive_sets(records)

    def unseen_transition(i):
        t = trans_keys[i]
        return t is not None and t not in cal_transitions

    def unseen_subject(i):
        p = records[i]["parser"]
        return (p["both_parse"]
                and p["fact_a_parsed"]["subject"] not in cal_subjects
                and p["fact_b_parsed"]["subject"] not in cal_subjects)

    out = {"provenance": data.provenance(experiment="conflict_detection",
                                         k_grid=list(K_GRID), band=list(BAND),
                                         n_boot=N_BOOT),
           "n_train_edits": int((gold & cal).sum()),
           "spaces": {}}

    for space, V in spaces.items():
        D_tr, rel_tr = fit_training_edits(records, pv, V, gold & cal)
        rced = RCED().fit(D_tr, rel_tr)

        # k selection on the calibration balanced eval task, honestly in-sample
        cal_eval = in_eval & cal
        pv_cal_eval = pv.subset(cal_eval)
        y_cal_eval = gold[cal_eval]
        k_scores = {}
        rcesp_by_k = {}
        for k in K_GRID:
            m = RCESP(k=k).fit(D_tr, rel_tr)
            rcesp_by_k[k] = m
            s, _ = m.score(pv_cal_eval, V)
            k_scores[k] = auroc(s[y_cal_eval], s[~y_cal_eval])
        k_star = max(k_scores, key=k_scores.get)
        rcesp = rcesp_by_k[k_star]

        def all_scores(view: data.PairView) -> dict[str, np.ndarray]:
            rc, rc_info = rced.score(view, V)
            rp, rp_info = rcesp.score(view, V)
            return {
                "cos_space": view.cos(V),
                "rced": rc, "rced_max": rced.score_max(view, V),
                "rced_global": rced.score_global(view, V),
                "rcesp": rp, "rcesp_global": rcesp.score_global(view, V),
                "_fallbacks": {"rced": rc_info, "rcesp": rp_info},
            }

        # LDA on global-subspace coordinates, fit on calibration pos vs hardneg
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        Ug = rcesp.U_global

        def lda_coords(view):
            out_c = np.empty((len(view), Ug.shape[1]))
            for s in range(0, len(view), 8192):
                e = min(s + 8192, len(view))
                d = V[view.ib[s:e]] - V[view.ia[s:e]]
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                out_c[s:e] = np.abs(d @ Ug)  # sign-invariant coordinates
            return out_c
        lda_fit_mask = (gold | hardneg) & cal
        pv_lf = pv.subset(lda_fit_mask)
        lda = LinearDiscriminantAnalysis()
        lda.fit(lda_coords(pv_lf), gold[lda_fit_mask])
        pos_col = list(lda.classes_).index(True)

        def with_lda(view, scores):
            scores["lda_global"] = lda.predict_proba(lda_coords(view))[:, pos_col]
            return scores

        blob: dict = {"k_selection": {"grid_auroc_cal": {str(k): v for k, v in
                                                         k_scores.items()},
                                      "k_star": k_star},
                      "n_relations_fit": len(rced.mu)}

        # ── balanced eval task, per subset ─────────────────────────────────
        balanced = {}
        for sub in ("sh_6k", "sh_32k", "sh_64k"):
            mask = in_eval & (subset == sub)
            view = pv.subset(mask)
            y = gold[mask]
            sc = with_lda(view, all_scores(view))
            entry = {"in_sample_fit": bool(sub != "sh_64k"),
                     "methods": {}, "fallbacks": sc.pop("_fallbacks")}
            band_mask = (camp_cos[mask] >= BAND[0]) & (camp_cos[mask] <= BAND[1])
            for name, s in sc.items():
                entry["methods"][name] = _task_metrics(s, y, data.SEED)
                yb, sb = y[band_mask], s[band_mask]
                entry["methods"][name]["band_auroc"] = (
                    auroc(sb[yb], sb[~yb]) if yb.any() and (~yb).any() else None)
            entry["band_n_pos"] = int(y[band_mask].sum())
            entry["band_n_neg"] = int((~y[band_mask]).sum())
            balanced[sub] = entry
        blob["balanced_eval"] = balanced

        # ── hard-negative task ─────────────────────────────────────────────
        hard = {}
        for split_name, split_mask in (("calibration", cal), ("confirmatory", ~cal)):
            mask = (gold | hardneg) & split_mask
            view = pv.subset(mask)
            y = gold[mask]
            cc = camp_cos[mask]
            sc = with_lda(view, all_scores(view))
            sc.pop("_fallbacks")
            entry = {"in_sample_fit": split_name == "calibration", "methods": {}}
            for name, s in sc.items():
                entry["methods"][name] = _task_metrics(s, y, data.SEED)
                entry["methods"][name]["inverted_vs_campaign_cos"] = \
                    inverted_win_rate(s[y], cc[y], s[~y], cc[~y])
            hard[split_name] = entry
        blob["hard_negative"] = hard

        # paired bootstrap on the confirmatory hard-negative task
        mask = (gold | hardneg) & ~cal
        view = pv.subset(mask)
        y = gold[mask]
        sc = with_lda(view, all_scores(view))
        blob["paired_bootstrap_confirmatory_hard"] = {
            f"{a}_minus_{b}": paired_bootstrap_delta_auc(
                sc[a][y], sc[a][~y], sc[b][y], sc[b][~y], N_BOOT, data.SEED)
            for a, b in (("rcesp", "cos_space"), ("rcesp", "rced"),
                         ("rced", "cos_space"), ("lda_global", "rcesp"))}

        # ── operating point: best-F1 on calibration hard task, frozen ──────
        op = {}
        mask_c = (gold | hardneg) & cal
        view_c = pv.subset(mask_c)
        y_c = gold[mask_c]
        sc_c = with_lda(view_c, all_scores(view_c))
        for name in ("cos_space", "rced", "rcesp", "rcesp_global", "lda_global"):
            t = best_f1_threshold(sc_c[name][y_c], sc_c[name][~y_c])
            op[name] = {"calibration": prf_at(t, sc_c[name][y_c], sc_c[name][~y_c]),
                        "confirmatory": prf_at(t, sc[name][y], sc[name][~y])}
        blob["operating_point"] = op

        # parser same_key baseline on the same two tasks (binary)
        def parser_prf(m):
            sk = np.array([bool(records[i]["parser"].get("same_key"))
                           for i in np.flatnonzero(m)])
            yy = gold[m]
            return prf_at(0.5, sk[yy].astype(float), sk[~yy].astype(float))
        blob["parser_same_key_baseline"] = {
            "hard_confirmatory": parser_prf((gold | hardneg) & ~cal),
            "balanced_sh_64k": parser_prf(in_eval & (subset == "sh_64k"))}

        # ── generalization ─────────────────────────────────────────────────
        gen: dict = {}
        conf_hard = hardneg & ~cal
        for tag, keep_pos in (("transition_disjoint", unseen_transition),
                              ("subject_disjoint", unseen_subject)):
            pos_idx = [i for i in np.flatnonzero(gold & ~cal) if keep_pos(i)]
            m = np.zeros(len(records), bool)
            m[pos_idx] = True
            m |= conf_hard
            view_g = pv.subset(m)
            y_g = gold[m]
            sc_g = with_lda(view_g, all_scores(view_g))
            sc_g.pop("_fallbacks")
            gen[tag] = {"n_pos": int(y_g.sum()), "n_neg": int((~y_g).sum()),
                        "auroc": {n: auroc(s[y_g], s[~y_g])
                                  for n, s in sc_g.items()}}

        # relation-disjoint (calibration, two folds, global variants only)
        fold = data.relation_fold([r for r in pv.relation if r])
        pair_fold = np.array([fold.get(r, -1) for r in pv.relation])
        rd = []
        for tr_f in (0, 1):
            D_f, rel_f = fit_training_edits(
                records, pv, V, gold & cal & (pair_fold == tr_f))
            rced_f = RCED().fit(D_f, rel_f)
            rcesp_f = RCESP(k=k_star).fit(D_f, rel_f)
            m = (gold | hardneg) & cal & (pair_fold == 1 - tr_f)
            view_f = pv.subset(m)
            y_f = gold[m]
            entry = {"n_pos": int(y_f.sum()), "n_neg": int((~y_f).sum()),
                     "auroc": {
                         "cos_space": auroc(*_split(view_f.cos(V), y_f)),
                         "rced_max": auroc(*_split(rced_f.score_max(view_f, V), y_f)),
                         "rced_global": auroc(*_split(rced_f.score_global(view_f, V), y_f)),
                         "rcesp_global": auroc(*_split(rcesp_f.score_global(view_f, V), y_f)),
                     }}
            rd.append(entry)
        gen["relation_disjoint_cal"] = {"fold0_to_fold1": rd[0],
                                        "fold1_to_fold0": rd[1]}

        # transition-deduped refit: does performance depend on repeated edits?
        D_dd, rel_dd = fit_training_edits(records, pv, V, gold & cal,
                                          dedupe_transitions=True,
                                          transition_keys=trans_keys)
        rced_dd = RCED().fit(D_dd, rel_dd)
        rcesp_dd = RCESP(k=k_star).fit(D_dd, rel_dd)
        m = (gold | hardneg) & ~cal
        view_dd = pv.subset(m)
        y_dd = gold[m]
        gen["transition_deduped_refit_confirmatory"] = {
            "n_train_after_dedupe": int(len(D_dd)),
            "auroc": {"rced": auroc(*_split(rced_dd.score(view_dd, V)[0], y_dd)),
                      "rcesp": auroc(*_split(rcesp_dd.score(view_dd, V)[0], y_dd))}}
        blob["generalization"] = gen
        out["spaces"][space] = blob

    data.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = data.OUT_DIR / "conflict_detection.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written:", dst)
    return out


def _split(scores, y):
    return scores[y], scores[~y]


if __name__ == "__main__":
    r = run()
    for space, blob in r["spaces"].items():
        h = blob["hard_negative"]["confirmatory"]["methods"]
        print(f"\n[{space}] k*={blob['k_selection']['k_star']}  "
              f"confirmatory hard-negative task:")
        for name, m in h.items():
            inv = m["inverted_vs_campaign_cos"]["win_rate"]
            print(f"  {name:13s} AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}"
                  f"  inverted-win {inv:.3f}" if inv is not None else
                  f"  {name:13s} AUROC {m['auroc']:.4f}")
