"""GEO identity screen — pair-level evaluation on the gold conflict dataset.

The same arenas the CES rows in REPORT.md sec.7 were measured on (balanced
cosine-matched eval per subset + 0.87-0.97 band, confirmatory hard task with
inverted-win, transition/subject-disjoint slices, tail TPRs), so the GEO
score is directly comparable to CES 0.9756 / ABTT-cos 0.9648 / raw cos
0.8930 on balanced sh_64k. Also carries the PCA investigation rows
(PCA-compressed probe variants) with their calibration-fit provenance.

Scores evaluated (all parser-free at inference):
    geo        min-margin score of the frozen artifact (the screen itself)
    probe      the slot-probe logit alone
    abtt_cos   whitened cosine (equals the artifact's cos_w axis)
    campaign_cos
    ces        committed CES refit (parser-routed) — the reference row

Usage:  python -m hnav.geometry_filter.run_geo_pairlevel
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .geo_artifact import ARTIFACT_JSON, GeoIdentityScreen
from .methods import fit_training_edits
from .metrics import auprc, auroc, bootstrap_ci, inverted_win_rate, \
    paired_bootstrap_delta_auc
from .run_dimension_ideas import ContrastiveSubspace
from .run_nuisance_analysis import tpr_at_fpr

N_BOOT = 1000


def run() -> dict:
    art, man = GeoIdentityScreen.load(ARTIFACT_JSON)
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    V, V_abtt = spaces["raw"], spaces["abtt"]
    pv = data.PairView(records, index)

    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    in_eval = np.array([r["in_eval_set"] for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])
    camp_cos = np.array([r["cosine_similarity"] for r in records])
    subset = np.array([r["subset"] for r in records], object)
    trans_keys = [data.transition_key(r) for r in records]
    _, cal_transitions = data.calibration_positive_sets(records)

    # vectorized artifact scores over every record
    D = V[pv.ib] - V[pv.ia]
    nrm = np.linalg.norm(D, axis=1)
    Dh = D / np.maximum(nrm[:, None], 1e-12)
    probe = np.abs(Dh) @ art.probe_w + art.probe_b
    cos_w = pv.cos(V_abtt)          # same committed whitening as the artifact
    geo = np.minimum((cos_w - art.T_w) / art.s_w, (probe - art.T_p) / art.s_p)

    # CES reference (refit exactly as run_dimension_ideas)
    D_pos, rel_pos = fit_training_edits(records, pv, V, gold & cal)
    pv_hn = pv.subset(hardneg & cal)
    D_hn = pv_hn.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace(k=20).fit(D_pos, rel_pos, D_hn,
                                        list(pv_hn.relation))
    scores = {"geo": geo, "probe": probe, "abtt_cos": cos_w,
              "campaign_cos": camp_cos, "ces": ces.score(pv, V)}

    # PCA investigation: PCA-compressed probe variants (calibration-fit)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    Xg = np.abs(np.vstack([D_pos, D_hn])).astype(np.float32)
    yg = np.r_[np.ones(len(D_pos)), np.zeros(len(D_hn))]
    pca_rows = {}
    for ncomp in (64, 256):
        p = PCA(n_components=ncomp, random_state=0).fit(Xg)
        lr = LogisticRegression(max_iter=3000, C=1.0,
                                class_weight="balanced").fit(p.transform(Xg), yg)
        s = lr.decision_function(p.transform(np.abs(Dh).astype(np.float32)))
        scores[f"probe_pca{ncomp}"] = s
        pca_rows[str(ncomp)] = "calibration-fit PCA of |d_hat| before the probe"

    out = {"provenance": data.provenance(
        experiment="geo_pairlevel", artifact_fingerprint=art.fingerprint(),
        n_boot=N_BOOT), "pca_note": pca_rows}

    bal = {}
    for sub in ("sh_6k", "sh_32k", "sh_64k"):
        m = in_eval & (subset == sub)
        y = m & gold
        band = m & (camp_cos >= 0.87) & (camp_cos <= 0.97)
        row = {"n_pos": int(y.sum()), "n_neg": int((m & ~gold).sum()),
               "in_sample_for_fit": sub != "sh_64k", "methods": {}}
        for name, s in scores.items():
            e = {"auroc": auroc(s[y], s[m & ~gold]),
                 "band_auroc": auroc(s[band & gold], s[band & ~gold])}
            if sub == "sh_64k":
                e["auroc_ci95"] = bootstrap_ci(auroc, s[y], s[m & ~gold],
                                               N_BOOT, data.SEED)
                if name != "ces":
                    e["delta_vs_ces"] = paired_bootstrap_delta_auc(
                        s[y], s[m & ~gold],
                        scores["ces"][y], scores["ces"][m & ~gold],
                        N_BOOT, data.SEED)
            row["methods"][name] = e
        bal[sub] = row
    out["balanced"] = bal

    mh = (gold | hardneg) & ~cal
    hard = {"n_pos": int((mh & gold).sum()), "n_neg": int((mh & ~gold).sum()),
            "methods": {}}
    conf_gold_idx = np.flatnonzero(gold & ~cal)
    seen = np.array([trans_keys[i] in cal_transitions for i in conf_gold_idx])
    neg_idx = np.flatnonzero(hardneg & ~cal)
    for name, s in scores.items():
        e = {"auroc": auroc(s[mh & gold], s[mh & ~gold]),
             "auprc": auprc(s[mh & gold], s[mh & ~gold]),
             "inverted_vs_campaign_cos": inverted_win_rate(
                 s[mh & gold], camp_cos[mh & gold],
                 s[mh & ~gold], camp_cos[mh & ~gold])}
        for tag, gidx in (("seen", conf_gold_idx[seen]),
                          ("unseen", conf_gold_idx[~seen])):
            for f in (1e-3, 1e-4):
                e[f"tpr_{tag}_fpr_{f:g}"] = tpr_at_fpr(s[gidx], s[neg_idx], f)
        hard["methods"][name] = e
    out["hard_confirmatory"] = hard

    dst = data.OUT_DIR / "geo_pairlevel.json"
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written:", dst)
    return out


if __name__ == "__main__":
    r = run()
    print("\nbalanced sh_64k:")
    for n, m in r["balanced"]["sh_64k"]["methods"].items():
        print(f"  {n:14s} auroc {m['auroc']:.4f}  band {m['band_auroc']:.4f}")
    print("\nhard confirmatory:")
    for n, m in r["hard_confirmatory"]["methods"].items():
        print(f"  {n:14s} auroc {m['auroc']:.4f}  auprc {m['auprc']:.4f}  "
              f"inv {m['inverted_vs_campaign_cos']['win_rate']:.3f}  "
              f"tail seen/unseen@1e-4 {m['tpr_seen_fpr_0.0001']:.3f}/"
              f"{m['tpr_unseen_fpr_0.0001']:.3f}")
