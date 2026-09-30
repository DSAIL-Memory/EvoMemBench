"""Stage 0 + Stage 1 — discovery and preprocessing.  [QDA]

Builds the pair bundle every later stage consumes:

    d_t     = Q (v_later - v_earlier)      ABTT-complement difference, R^2432
    m_t     = Q ((v_a + v_b)/2 - mean)     ABTT-complement midpoint (the ABTT
                                           mean does NOT cancel in midpoints,
                                           so it is subtracted; it cancels
                                           exactly in differences)
    norm_dt = ||d_t||
    cos     = the record's campaign cosine (raw space, as committed)

``Q`` (2432 x 2560) is an orthonormal basis of the orthogonal complement of
the frozen ABTT D=128 components — the ``P_abtt`` of the protocol realized as
an explicit basis so that N' = 2432 is the actual ambient dimension and no
structurally-zero directions can leak into covariance spectra. Built
deterministically from the committed artifact via SVD.

Orientation. ``fact_a`` is the earlier serial on every one of the 54,569
records (fact ids ARE serials; verified against the explicit ``serial_*``
fields on all 2,682 tagged pairs — see discovery.json), so ``d`` is always
the oriented old->new difference for BOTH classes. No pseudo-random signs.

Created splits (seed 0, recorded in discovery.json): the fit-split verified
negatives are partitioned 20% conformal-calibration / 40% half A / 40% half B
by a seeded permutation. Conformal negatives touch nothing fit-side.
"""
from __future__ import annotations

import numpy as np

from hnav.geometry_filter import data as gfdata

SPLIT_SEED = 0
CONFORMAL_FRAC = 0.20
N_ABTT = 128
DIM_RAW = 2560
DIM = DIM_RAW - N_ABTT  # N' = 2432

_EPS = 1e-12


def serial_of(fact_id: str) -> int:
    """``fact:N`` -> N. The fact index within its context IS the serial."""
    return int(fact_id.split(":", 1)[1])


def complement_basis() -> tuple[np.ndarray, np.ndarray]:
    """(Q, mean): orthonormal basis of the ABTT-component complement plus the
    artifact's mean. Deterministic function of the committed artifact."""
    import json

    from hnav.core.geometry import ABTTWhitening

    blob = json.loads(gfdata.ABTT_ARTIFACT.read_text(encoding="utf-8"))
    w = ABTTWhitening.from_dict(blob["whitening"])
    assert w.fitted and w.components.shape == (N_ABTT, DIM_RAW)
    # full SVD of the component stack: right singular vectors 128..2559 span
    # the orthogonal complement exactly
    _, _, vt = np.linalg.svd(w.components, full_matrices=True)
    Q = vt[N_ABTT:]
    assert Q.shape == (DIM, DIM_RAW)
    # closed-form checks: Q orthonormal, Q ⟂ components
    assert np.allclose(Q @ Q.T, np.eye(DIM), atol=1e-10)
    assert np.abs(Q @ w.components.T).max() < 1e-10
    return Q, np.asarray(w.mean, dtype=np.float64)


class Bundle:
    """Everything Stage 2+ needs, in memory once.

    Arrays are float32 (508 MB for D_t at 54,569 x 2432); every covariance or
    spectrum computation upcasts its own chunk to float64.
    """

    def __init__(self) -> None:
        records = gfdata.load_records()
        index, V_raw = gfdata.fact_matrix(records)
        self.records = records
        self.pv = gfdata.PairView(records, index)
        self.index = index
        self.V_raw = V_raw

        self.Q, self.abtt_mean = complement_basis()

        n = len(records)
        self.y = np.array([r["gold_update"] for r in records])
        self.cal = np.array([r["split"] == "calibration" for r in records])
        self.in_eval = np.array([r["in_eval_set"] for r in records])
        self.negative = np.array([r["tier"] == "negative" for r in records])
        self.hardneg = np.array([gfdata.is_hard_negative(r) for r in records])
        self.cos = np.array([r["cosine_similarity"] for r in records])
        self.subset = np.array([r["subset"] for r in records], object)
        self.relation = [gfdata.relation_of(r) for r in records]
        self.trans_keys = [gfdata.transition_key(r) for r in records]
        (self.cal_subjects,
         self.cal_transitions) = gfdata.calibration_positive_sets(records)

        # orientation: a is the earlier serial on every record (asserted)
        sa = np.array([serial_of(r["fact_a_id"]) for r in records])
        sb = np.array([serial_of(r["fact_b_id"]) for r in records])
        assert (sa < sb).all(), "fact_a must be the earlier serial everywhere"
        for r in records:
            if "serial_earlier" in r:
                assert r["serial_earlier"] == serial_of(r["fact_a_id"])
                assert r["serial_later"] == serial_of(r["fact_b_id"])
        self.serial_a, self.serial_b = sa, sb

        # d_t = Q(v_b - v_a), oriented old->new; chunked float32
        self.D_t = np.empty((n, DIM), dtype=np.float32)
        chunk = 4096
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = V_raw[self.pv.ib[s:e]] - V_raw[self.pv.ia[s:e]]
            self.D_t[s:e] = (d @ self.Q.T).astype(np.float32)
        self.norm_dt = np.linalg.norm(self.D_t.astype(np.float64), axis=1)
        n_zero = int((self.norm_dt < _EPS).sum())
        assert n_zero == 0, f"{n_zero} zero-difference pairs (dedup violated?)"

        # created splits over fit negatives (seed 0)
        fit_neg_idx = np.flatnonzero(self.negative & self.cal)
        rng = np.random.default_rng(SPLIT_SEED)
        perm = fit_neg_idx[rng.permutation(len(fit_neg_idx))]
        n_conf = int(round(CONFORMAL_FRAC * len(perm)))
        conf_idx = perm[:n_conf]
        rest = perm[n_conf:]
        half = len(rest) // 2
        self.conformal_neg = np.zeros(n, bool)
        self.conformal_neg[conf_idx] = True
        self.neg_half_a = np.zeros(n, bool)
        self.neg_half_a[rest[:half]] = True
        self.neg_half_b = np.zeros(n, bool)
        self.neg_half_b[rest[half:]] = True
        assert not (self.conformal_neg & (self.neg_half_a | self.neg_half_b)).any()

        # fit material: gold + non-conformal negatives, calibration split only
        self.fit_gold = self.y & self.cal
        self.fit_neg = (self.neg_half_a | self.neg_half_b)

    # ── views ────────────────────────────────────────────────────────────────
    def midpoints_t(self, idx: np.ndarray) -> np.ndarray:
        """m_t = Q((v_a + v_b)/2 - abtt_mean) for the given record indices."""
        out = np.empty((len(idx), DIM), dtype=np.float32)
        chunk = 4096
        for s in range(0, len(idx), chunk):
            e = min(s + chunk, len(idx))
            ii = idx[s:e]
            m = 0.5 * (self.V_raw[self.pv.ia[ii]] + self.V_raw[self.pv.ib[ii]])
            out[s:e] = ((m - self.abtt_mean) @ self.Q.T).astype(np.float32)
        return out

    def whiten(self, W0: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """z = W0 d_t for the given record indices, float64, chunked."""
        out = np.empty((len(idx), DIM), dtype=np.float64)
        chunk = 4096
        for s in range(0, len(idx), chunk):
            e = min(s + chunk, len(idx))
            out[s:e] = self.D_t[idx[s:e]].astype(np.float64) @ W0.T
        return out


def discovery(b: Bundle) -> dict:
    """Stage 0 — counts, availability, split provenance. Pure bookkeeping."""
    from collections import Counter

    per_subset = {}
    for sub in ("sh_6k", "sh_32k", "sh_64k", "sh_262k"):
        m = b.subset == sub
        if not m.any():
            per_subset[sub] = {"present": False}
            continue
        gold_idx = np.flatnonzero(b.y & m)
        rels = Counter(b.relation[i] for i in gold_idx)
        trans = Counter(b.trans_keys[i] for i in gold_idx
                        if b.trans_keys[i] is not None)
        per_subset[sub] = {
            "present": True,
            "n1_gold": int(len(gold_idx)),
            "n0_negative": int((b.negative & m).sum()),
            "n0_hard_negative": int((b.hardneg & m).sum()),
            "n_relations_gold": len(rels),
            "n_transitions_gold": len(trans),
            "gold_per_relation_top10": rels.most_common(10),
            "gold_per_transition_top10": [
                [list(k), v] for k, v in trans.most_common(10)],
            "split": "calibration" if sub in ("sh_6k", "sh_32k")
            else "confirmatory",
            "stage4b_eligible": bool(len(gold_idx) >= 200),
        }
    n_tagged = sum(1 for r in b.records if "serial_earlier" in r)
    return {
        "per_subset": per_subset,
        "serial_order": {
            "available_for_gold": True,
            "available_for_negatives": True,
            "source": "fact_a_id/fact_b_id (fact index == serial); fact_a is "
                      "the earlier serial on all records — verified in code",
            "n_records_checked": len(b.records),
            "n_explicitly_tagged": n_tagged,
            "explicit_tags_all_consistent": True,
            "mu0_fit": "estimated from fit negatives (order IS available; the "
                       "protocol's mu0:=0 fallback was not needed)",
        },
        "context_structure": {
            "note": "each subset is a single benchmark context; pairs within "
                    "a subset share facts, so splits are by subset "
                    "(calibration = sh_6k+sh_32k, confirmatory = sh_64k), "
                    "exactly the committed geometry_filter convention",
            "sh_262k": "absent from the gold conflict dataset by construction "
                       "(audit selection frame); skipped",
        },
        "existing_splits_reused": [
            "record['split'] calibration/confirmatory",
            "record['in_eval_set'] balanced cosine-matched eval set",
            "gfdata.calibration_positive_sets seen-subject/-transition lists",
            "gfdata.relation_fold deterministic 2-fold relation split",
        ],
        "created_splits": {
            "seed": SPLIT_SEED,
            "rule": "fit-split tier=='negative' records permuted with "
                    "default_rng(0); first 20% conformal-calibration, "
                    "remainder halved into half A (whitening) / half B "
                    "(parallel-analysis pool)",
            "n_conformal_neg": int(b.conformal_neg.sum()),
            "n_half_a": int(b.neg_half_a.sum()),
            "n_half_b": int(b.neg_half_b.sum()),
        },
        "abtt": {
            "artifact": "stage0_results/abtt/abtt_whitening_D128.json",
            "applied_as": "explicit orthonormal complement basis Q "
                          f"({DIM} x {DIM_RAW}); artifact never re-fit",
            "n_prime": DIM,
        },
        "exclusions": {
            "n_records_dropped": 0,
            "note": "no pair is dropped anywhere; quarantined tiers "
                    "(discovered_unverified, rejected) are simply never gold "
                    "and never negatives, matching the dataset rules",
        },
    }
