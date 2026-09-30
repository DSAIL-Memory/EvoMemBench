"""Experiment 2 — slot localization: does d_hat direction say WHICH slot changed?

Classes come from the parser's deterministic slot comparison (these are slot
labels, not conflict labels — exact for this question):

    object_only, subject_only, subject_object, relation_object,
    subject_relation, all_change        (relation_only n=29 → dropped)

A candidate pair is unordered, so the probe must not see true orientation
(tagged pairs have one; unordered classes don't — using it only where it
exists would hand the probe a class-correlated cue). Three feature variants:

    signed  d_hat with a deterministic random sign per pair. This is a
            *designed negative control*: ±d_hat is symmetrically distributed
            within every class, so no linear model can separate it — measured
            chance here confirms the sign randomization actually bites.
    abs     |d_hat| elementwise — fully sign-invariant axis-energy profile.
    canon   d_hat flipped so its largest-|coordinate| entry is positive — a
            label-free canonicalization that retains directional structure.

Probes (multinomial logistic regression, balanced class weights, per-class cap):
    main               train calibration pairs → test confirmatory pairs
    relation-disjoint  within calibration, 2 relation folds (both directions);
                       cross-relation pairs must have both relations in the
                       same fold or they are dropped (count recorded)
    binary             object_only vs subject_only AUROC, cal → conf

Usage:  python -m hnav.geometry_filter.run_slot_probe
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from . import data
from .metrics import auroc

CLASSES = ["object_only", "subject_only", "subject_object",
           "relation_object", "subject_relation", "all_change"]
CAP_PER_CLASS = 3000
# "centered" tracked "raw" to within 0.005 macro-F1 in the first pass and is
# dropped from the loop to keep the 3-variant sweep tractable.
PROBE_SPACES = ("raw", "abtt")


def _probe_sign(pair_id: str) -> int:
    h = hashlib.sha256(f"{data.SEED}|probe|{pair_id}".encode()).digest()
    return 1 if h[0] % 2 == 0 else -1


FEATURE_VARIANTS = ("signed", "abs", "canon")


def _features(recs, index, V, variant: str = "signed") -> np.ndarray:
    ia = np.array([index[r["fact_a"]] for r in recs])
    ib = np.array([index[r["fact_b"]] for r in recs])
    sign = np.array([_probe_sign(r["pair_id"]) for r in recs])
    D = (V[ib] - V[ia]) * sign[:, None]
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    if variant == "abs":
        D = np.abs(D)
    elif variant == "canon":
        flip = np.sign(D[np.arange(len(D)), np.abs(D).argmax(axis=1)])
        D = D * flip[:, None]
    return D.astype(np.float32)


def _cap(recs, rng) -> list[dict]:
    by_cls = defaultdict(list)
    for r in recs:
        by_cls[r["_slot"]].append(r)
    out = []
    for c, lst in by_cls.items():
        if len(lst) > CAP_PER_CLASS:
            keep = rng.choice(len(lst), CAP_PER_CLASS, replace=False)
            lst = [lst[i] for i in keep]
        out.extend(lst)
    return out


def _fit_eval(train, test, index, V, variant):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score

    Xtr, ytr = _features(train, index, V, variant), [r["_slot"] for r in train]
    Xte, yte = _features(test, index, V, variant), [r["_slot"] for r in test]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels = [c for c in CLASSES if c in set(ytr) | set(yte)]
    return {
        "n_train": len(train), "n_test": len(test),
        "train_class_counts": dict(Counter(ytr)),
        "test_class_counts": dict(Counter(yte)),
        "macro_f1": float(f1_score(yte, pred, labels=labels, average="macro",
                                   zero_division=0)),
        "per_class_f1": dict(zip(labels, [float(x) for x in f1_score(
            yte, pred, labels=labels, average=None, zero_division=0)])),
        "confusion_matrix": {"labels": labels,
                             "rows_true_cols_pred": confusion_matrix(
                                 yte, pred, labels=labels).tolist()},
    }


def run() -> dict:
    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    spaces = data.build_spaces(V_raw)
    rng = np.random.default_rng(data.SEED)

    labeled = []
    for r in records:
        c = data.slot_class(r)
        if c in CLASSES:
            r = dict(r)
            r["_slot"] = c
            labeled.append(r)
    n_dropped_relation_only = sum(1 for r in records
                                  if data.slot_class(r) == "relation_only")

    cal = [r for r in labeled if r["split"] == "calibration"]
    conf = [r for r in labeled if r["split"] == "confirmatory"]
    fold = data.relation_fold([data.relation_of(r) for r in labeled])

    out = {"provenance": data.provenance(experiment="slot_probe",
                                         cap_per_class=CAP_PER_CLASS,
                                         classes=CLASSES,
                                         n_dropped_relation_only=n_dropped_relation_only),
           "spaces": {}}

    def fold_of(r):
        pa = r["parser"]["fact_a_parsed"]["relation"]
        pb = r["parser"]["fact_b_parsed"]["relation"]
        fa, fb = fold[pa], fold[pb]
        return fa if fa == fb else None

    for name in PROBE_SPACES:
        V = spaces[name]
        out["spaces"][name] = {}
        for variant in FEATURE_VARIANTS:
            blob = {}
            blob["cal_to_conf"] = _fit_eval(_cap(cal, rng), _cap(conf, rng),
                                            index, V, variant)

            if variant != "signed":  # signed is the chance-level control only
                keyed = [(r, fold_of(r)) for r in cal]
                n_cross_dropped = sum(1 for _, f in keyed if f is None)
                rd = []
                for tr_f in (0, 1):
                    train = [r for r, f in keyed if f == tr_f]
                    test = [r for r, f in keyed if f == 1 - tr_f]
                    rd.append(_fit_eval(_cap(train, rng), _cap(test, rng),
                                        index, V, variant))
                blob["relation_disjoint_cal"] = {
                    "n_cross_relation_pairs_dropped": n_cross_dropped,
                    "fold0_to_fold1": rd[0], "fold1_to_fold0": rd[1],
                    "mean_macro_f1": float(np.mean([x["macro_f1"] for x in rd])),
                }

            # binary object_only vs subject_only AUROC
            from sklearn.linear_model import LogisticRegression
            btr = [r for r in cal if r["_slot"] in ("object_only", "subject_only")]
            bte = [r for r in conf if r["_slot"] in ("object_only", "subject_only")]
            btr, bte = _cap(btr, rng), _cap(bte, rng)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(_features(btr, index, V, variant), [r["_slot"] for r in btr])
            s = clf.decision_function(_features(bte, index, V, variant))
            if clf.classes_.tolist().index("object_only") == 0:
                s = -s
            y = np.array([r["_slot"] == "object_only" for r in bte])
            blob["object_vs_subject_auroc"] = {
                "auroc": auroc(s[y], s[~y]),
                "n_object": int(y.sum()), "n_subject": int((~y).sum())}

            out["spaces"][name][variant] = blob

    data.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = data.OUT_DIR / "slot_probe.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written:", dst)
    return out


if __name__ == "__main__":
    r = run()
    for name, variants in r["spaces"].items():
        for variant, b in variants.items():
            rd = b.get("relation_disjoint_cal")
            print(f"{name:9s} {variant:7s} cal->conf macroF1 "
                  f"{b['cal_to_conf']['macro_f1']:.3f} | rel-disjoint "
                  f"{rd['mean_macro_f1']:.3f} | " if rd else
                  f"{name:9s} {variant:7s} cal->conf macroF1 "
                  f"{b['cal_to_conf']['macro_f1']:.3f} | ",
                  f"obj-vs-subj AUROC {b['object_vs_subject_auroc']['auroc']:.3f}")
