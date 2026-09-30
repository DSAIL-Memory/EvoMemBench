"""Experiment 5 — the geometry of CES's failure, and the nuisance-complement fix.

The additive local model the earlier measurements imply: within one relation
template r, a fact embedding decomposes approximately as

    v(s, o) ~ mu_r + S(s) + O(o) + noise

(subject content + object content around the template mean). Then an
object-edit difference is d = O(o2) - O(o1) (subject-invariant — the measured
0.86 cross-subject coherence) and a subject-change difference is
d = S(s2) - S(s1). Four consequences, each tested here:

  Q1  the subject-variation subspace is estimable UNSUPERVISED: pairs with
      same relation + same object + different subject isolate S-differences
      without any conflict label. If span(unsupervised) ~ span(labeled U_subj)
      (principal angles), the nuisance model needs no gold and can even be
      re-fit on a deployment store — the transfer problem dissolves.
  Q2  CES's U_obj estimates span{seen transitions}, NOT the object-variation
      space: different transitions are near-orthogonal (0.067 coherence), so
      unseen-transition edits should have LOW energy in U_obj. Direct energy
      accounting on confirmatory gold pairs, seen vs unseen transitions.
  Q3  therefore the detector should be INVERTED: test non-membership in the
      well-estimated nuisance span rather than membership in the
      under-estimated positive span:  score = -||U_nuis^T d_hat||^2.
      Compare pure subject-veto and an expanded nuisance basis
      (+ relation-change edits) against CES on held-out slices.
  Q4  at pool prevalence ~0.8%, precision 0.99 needs FPR ~1e-4 — the extreme
      negative TAIL decides, not AUROC. Report TPR at fixed low FPR, and the
      calibration-chosen zero-FP conjunction (ABTT-cos AND subject-veto)
      applied frozen to the confirmatory hard task.

All fitting on calibration; sh_64k slices evaluated once, frozen.

Usage:  python -m hnav.geometry_filter.run_nuisance_analysis
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .methods import fit_training_edits
from .metrics import auroc
from .run_dimension_ideas import ContrastiveSubspace

K = 20
_EPS = 1e-12


def _hat(D: np.ndarray) -> np.ndarray:
    return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), _EPS)


def _top(D: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(D, full_matrices=False)
    return vt[: min(k, vt.shape[0])].T


def principal_angle_cosines(U: np.ndarray, W: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(U.T @ W, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def slot_mask(records, want, tier=None):
    out = np.zeros(len(records), bool)
    for i, r in enumerate(records):
        if tier is not None and r["tier"] != tier:
            continue
        if data.slot_class(r) == want:
            out[i] = True
    return out


def energies(view, V, U, chunk=8192):
    out = np.empty(len(view))
    for s in range(0, len(view), chunk):
        e = min(s + chunk, len(view))
        d = _hat(V[view.ib[s:e]] - V[view.ia[s:e]])
        out[s:e] = np.linalg.norm(d @ U, axis=1) ** 2
    return out


def tpr_at_fpr(pos, neg, fpr: float) -> float:
    thr = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thr))


def run() -> dict:
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    V, V_abtt = spaces["raw"], spaces["abtt"]
    pv = data.PairView(records, index)

    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])
    trans_keys = [data.transition_key(r) for r in records]
    _, cal_transitions = data.calibration_positive_sets(records)
    in_eval = np.array([r["in_eval_set"] for r in records])
    subset = np.array([r["subset"] for r in records], object)

    out: dict = {"provenance": data.provenance(experiment="nuisance_analysis", k=K)}

    # ── fitted bases (calibration only) ──────────────────────────────────────
    D_pos, rel_pos = fit_training_edits(records, pv, V, gold & cal)
    pv_hn = pv.subset(hardneg & cal)
    D_hn = pv_hn.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace(k=K).fit(D_pos, rel_pos, D_hn, list(pv_hn.relation))
    U_obj_g, U_subj = ces.U_obj_g, ces.U_subj_g

    # unsupervised subject basis: same relation+object, different subject,
    # verified negatives — observable in ANY store without conflict labels
    m_unsup = slot_mask(records, "subject_only", tier="negative") & cal
    D_unsup = pv.subset(m_unsup).diff(V, normalize=True, oriented=False)
    U_subj_unsup = _top(D_unsup, K)

    # relation-change nuisance: same subject, different relation (both
    # object kept and object changed), verified negatives
    m_rel = (slot_mask(records, "relation_object", tier="negative")
             | slot_mask(records, "subject_relation", tier="negative")
             | slot_mask(records, "all_change", tier="negative")) & cal
    # 'relation changed' is the common factor; keep the same-subject ones
    m_rel &= np.array([r["parser"]["both_parse"]
                       and r["parser"]["same_subject"] for r in records])
    D_rel = pv.subset(m_rel).diff(V, normalize=True, oriented=False)
    U_rel = _top(D_rel, K) if len(D_rel) >= K else _top(D_rel, max(1, len(D_rel)))
    out["fit_sizes"] = {"gold_edits": int(len(D_pos)), "hardneg": int(len(D_hn)),
                        "unsup_subject": int(len(D_unsup)),
                        "relation_change_same_subject": int(len(D_rel))}

    # Q1 — do labeled and unsupervised subject bases agree?
    pa = principal_angle_cosines(U_subj, U_subj_unsup)
    out["q1_subject_basis_agreement"] = {
        "principal_angle_cosines_top10": [round(float(x), 4) for x in pa[:10]],
        "mean_sq_cos": float(np.mean(pa ** 2)),
        "null_mean_sq_cos_random_k20": float(K / V.shape[1]),
    }

    # Q2 — energy accounting on confirmatory gold, seen vs unseen transitions
    conf_gold_idx = np.flatnonzero(gold & ~cal)
    seen = np.array([trans_keys[i] in cal_transitions for i in conf_gold_idx])
    view_g = pv.subset(np.isin(np.arange(len(pv)), conf_gold_idx))
    e_obj = energies(view_g, V, U_obj_g)
    e_subj = energies(view_g, V, U_subj)
    view_hn_conf = pv.subset(hardneg & ~cal)
    out["q2_energy_accounting_conf"] = {
        "gold_seen_transition": {"n": int(seen.sum()),
                                 "obj_energy_mean": float(e_obj[seen].mean()),
                                 "subj_energy_mean": float(e_subj[seen].mean())},
        "gold_unseen_transition": {"n": int((~seen).sum()),
                                   "obj_energy_mean": float(e_obj[~seen].mean()),
                                   "subj_energy_mean": float(e_subj[~seen].mean())},
        "hardneg": {"n": len(view_hn_conf),
                    "obj_energy_mean": float(energies(view_hn_conf, V, U_obj_g).mean()),
                    "subj_energy_mean": float(energies(view_hn_conf, V, U_subj).mean())},
    }

    # ── scores to compare ────────────────────────────────────────────────────
    U_nuis = np.linalg.qr(np.hstack([U_subj, U_rel]))[0]

    def scores(view) -> dict[str, np.ndarray]:
        rc, _ = None, None
        s = {
            "ces": ces.score(view, V),
            "subj_veto": -energies(view, V, U_subj),
            "subj_veto_unsup": -energies(view, V, U_subj_unsup),
            "nuis_veto": -energies(view, V, U_nuis),
            "abtt_cos": view.cos(V_abtt),
        }
        return s

    def eval_task(mask, name) -> dict:
        view = pv.subset(mask)
        y = gold[mask]
        sc = scores(view)
        entry = {}
        for n, s in sc.items():
            entry[n] = {"auroc": auroc(s[y], s[~y]),
                        "tpr_at_fpr_1e-3": tpr_at_fpr(s[y], s[~y], 1e-3),
                        "tpr_at_fpr_1e-4": tpr_at_fpr(s[y], s[~y], 1e-4)}
        entry["_n"] = {"pos": int(y.sum()), "neg": int((~y).sum())}
        return entry

    # Q3 — held-out comparisons
    out["q3_conf_hard"] = eval_task((gold | hardneg) & ~cal, "hard")
    m_td = np.zeros(len(records), bool)
    m_td[[i for i in conf_gold_idx if trans_keys[i] not in cal_transitions]] = True
    m_td |= hardneg & ~cal
    out["q3_transition_disjoint"] = eval_task(m_td, "td")
    out["q3_balanced_sh64k"] = eval_task(in_eval & (subset == "sh_64k"), "bal")

    # Q4 — calibration-chosen zero-FP conjunction, applied frozen to conf
    def conj(mask):
        view = pv.subset(mask)
        return (view.cos(V_abtt), -energies(view, V, U_subj), gold[mask])

    ca, sa, ya = conj((gold | hardneg) & cal)
    best = None
    # for each cosine floor, the veto threshold is set just above the WORST
    # surviving calibration negative — zero FP on calibration by construction
    for q in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        c_thr = float(np.quantile(ca[~ya], q))
        surv = sa[~ya][ca[~ya] >= c_thr]
        s_thr = float(surv.max()) + 1e-9 if len(surv) else -np.inf
        tpr = float(np.mean((ca[ya] >= c_thr) & (sa[ya] > s_thr)))
        if best is None or tpr > best["cal_tpr"]:
            best = {"abtt_cos_thr": c_thr, "subj_veto_thr": s_thr,
                    "cal_tpr": tpr}
    cc, sscore, yc = conj((gold | hardneg) & ~cal)
    admit_pos = (cc[yc] >= best["abtt_cos_thr"]) & (sscore[yc] > best["subj_veto_thr"])
    admit_neg = (cc[~yc] >= best["abtt_cos_thr"]) & (sscore[~yc] > best["subj_veto_thr"])
    best["conf_tpr"] = float(admit_pos.mean())
    best["conf_fp"] = int(admit_neg.sum())
    best["conf_n_neg"] = int((~yc).sum())
    out["q4_zero_fp_conjunction"] = best

    # ── Q5: nuisance rank — how much subject-change energy does U_subj(k)
    # capture, and where does gold-edit bleed begin? ──────────────────────────
    q5 = {}
    view_hn = view_hn_conf
    for kk in (20, 50, 100, 200, 400):
        Uk = _top(D_hn, kk)
        q5[str(kk)] = {
            "hardneg_energy_mean": float(energies(view_hn, V, Uk).mean()),
            "gold_energy_mean": float(energies(view_g, V, Uk).mean()),
        }
    out["q5_subject_rank_sweep"] = q5

    # ── Q6: store-adaptive object space — the additive-model estimator.
    # O_r = top-k of within-relation fact variation AFTER projecting out the
    # (label-free) subject subspace. Fit on the sh_64k STORE FACTS ONLY (no
    # labels, no questions, no pair information) — the same store-side fitting
    # precedent as GeometryModule.fit_whitening. ─────────────────────────────
    facts_64k: dict[str, list[int]] = {}
    seen_fact = set()
    for r in records:
        if r["subset"] != "sh_64k" or not r["parser"]["both_parse"]:
            continue
        for side, parsed in (("fact_a", "fact_a_parsed"), ("fact_b", "fact_b_parsed")):
            t = r[side]
            if t in seen_fact:
                continue
            seen_fact.add(t)
            facts_64k.setdefault(r["parser"][parsed]["relation"], []).append(index[t])
    U_obj_store: dict[str, np.ndarray] = {}
    for rel, idxs in facts_64k.items():
        if len(idxs) < 5:
            continue
        F = V[np.array(idxs)]
        C = F - F.mean(axis=0)
        R = C - (C @ U_subj_unsup) @ U_subj_unsup.T   # project out S, cheaply
        U_obj_store[rel] = _top(R, K)
    out["q6_store_fit"] = {"n_relations": len(U_obj_store),
                           "facts_used": len(seen_fact)}

    def store_ces(view) -> np.ndarray:
        s = np.empty(len(view))
        for i in range(len(view)):
            rel = view.relation[i]
            d = _hat((V[view.ib[i]] - V[view.ia[i]])[None, :])[0]
            Uo = U_obj_store.get(rel)
            e_o = float(np.linalg.norm(d @ Uo) ** 2) if Uo is not None else 0.0
            s[i] = e_o - float(np.linalg.norm(d @ U_subj_unsup) ** 2)
        return s

    def eval_store(mask) -> dict:
        view = pv.subset(mask)
        y = gold[mask]
        s = store_ces(view)
        return {"auroc": auroc(s[y], s[~y]),
                "tpr_at_fpr_1e-3": tpr_at_fpr(s[y], s[~y], 1e-3),
                "tpr_at_fpr_1e-4": tpr_at_fpr(s[y], s[~y], 1e-4)}

    out["q6_store_adaptive_ces"] = {
        "conf_hard": eval_store((gold | hardneg) & ~cal),
        "transition_disjoint": eval_store(m_td),
        "balanced_sh64k": eval_store(in_eval & (subset == "sh_64k")),
    }

    data.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = data.OUT_DIR / "nuisance_analysis.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written:", dst)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "provenance"}, indent=1))
