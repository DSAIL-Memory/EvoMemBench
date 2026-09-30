"""GEO identity screen — parser-free same-key identity from geometry.  [E2E-3]

The parser arm wins end-to-end because its identity screen answers one
question with near-perfect precision on cos >= 0.90 pool pairs: is this a
same-key pair (object slot changed -> candidate supersession) or a cross-key
pair (subject slot changed -> the NLI rubber-stamp adversary)? This artifact
answers the same question from EMBEDDINGS ONLY — no parser field is read at
inference:

    probe(a, b)  = w . |d_hat| + b          the slot probe: logistic on the
                                            absolute axis profile of the unit
                                            difference (sign-invariant; fit on
                                            calibration gold vs hard-negative
                                            edits of the gold conflict dataset)
    cos_w(a, b)  = whitened cosine          under the frozen committed ABTT
                                            D=128 artifact (copied in, source
                                            fingerprint pinned)
    score(a, b)  = min( (cos_w - T_w)/s_w , (probe - T_p)/s_p )

The two anchors (T_w, T_p) are the joint zero-false-positive staircase point
on the CALIBRATION pool pairs (both benchmarkpage prepasses, raw cos >= 0.88,
NLI-passing pairs only), and (s_w, s_p) are the feature scales over the same
pairs — so ``score >= tau`` sweeps a 1-parameter family of NESTED conjunction
regions along the diagonal through the anchor, which is exactly the shape
``detector_gap --select`` explores with its tau grid. tau = 0 is the
pair-level zero-FP conjunction; negative tau explores the slack the key-level
harm rule allows (a false pair is harmless when the older fact it would drop
is itself superseded); positive tau is stricter.

Storage/fingerprint discipline is CESArtifact's: JSON manifest + sibling
``.npz``, fingerprint over ARRAY bytes, bit-identical round-trip asserted by
the fitter. Fitter: ``python -m hnav.geometry_filter.geo_artifact`` — refuses
fit data outside sh_6k + sh_32k.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_JSON = REPO / "stage0_results" / "geometry_filter" / "geo_identity_screen.json"
ABTT_ARTIFACT = REPO / "stage0_results" / "abtt" / "abtt_whitening_D128.json"
FIT_SUBSETS = ("sh_6k", "sh_32k")
POOL_COS_MIN = 0.88          # pool-pair frame used for anchors and scales
POOL_NLI_MIN = 0.90          # anchors are set among NLI-passing pairs only

_EPS = 1e-12
_ARRAYS = ("probe_w", "abtt_mean", "abtt_components")
_SCALARS = ("probe_b", "T_w", "T_p", "s_w", "s_p")


class GeoIdentityScreen:
    def __init__(self, probe_w: np.ndarray, probe_b: float,
                 abtt_mean: np.ndarray, abtt_components: np.ndarray,
                 T_w: float, T_p: float, s_w: float, s_p: float) -> None:
        self.probe_w = np.ascontiguousarray(probe_w, dtype=np.float64)
        self.probe_b = float(probe_b)
        self.abtt_mean = np.ascontiguousarray(abtt_mean, dtype=np.float64)
        self.abtt_components = np.ascontiguousarray(abtt_components,
                                                    dtype=np.float64)
        self.T_w, self.T_p = float(T_w), float(T_p)
        self.s_w, self.s_p = float(s_w), float(s_p)
        self._cache: dict[tuple, float] = {}

    # ── identity ─────────────────────────────────────────────────────────────
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.probe_w, self.abtt_mean, self.abtt_components):
            h.update(a.tobytes())
        for s in (self.probe_b, self.T_w, self.T_p, self.s_w, self.s_p):
            h.update(np.float64(s).tobytes())
        return h.hexdigest()

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, json_path: pathlib.Path, provenance: dict) -> None:
        json_path = pathlib.Path(json_path)
        npz_path = json_path.with_suffix(".npz")
        np.savez_compressed(npz_path, probe_w=self.probe_w,
                            abtt_mean=self.abtt_mean,
                            abtt_components=self.abtt_components)
        manifest = {
            "artifact": "GEO identity screen (slot probe x whitened cosine)",
            "dim": int(self.probe_w.shape[0]),
            "scalars": {"probe_b": self.probe_b, "T_w": self.T_w,
                        "T_p": self.T_p, "s_w": self.s_w, "s_p": self.s_p},
            "matrices_npz": npz_path.name,
            "fingerprint": self.fingerprint(),
            "provenance": provenance,
        }
        json_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, json_path: pathlib.Path) -> tuple["GeoIdentityScreen", dict]:
        json_path = pathlib.Path(json_path)
        man = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(json_path.parent / man["matrices_npz"]) as z:
            s = man["scalars"]
            art = cls(z["probe_w"], s["probe_b"], z["abtt_mean"],
                      z["abtt_components"], s["T_w"], s["T_p"],
                      s["s_w"], s["s_p"])
        got = art.fingerprint()
        if got != man["fingerprint"]:
            raise ValueError(
                f"GEO artifact fingerprint mismatch: manifest says "
                f"{man['fingerprint']}, loaded parameters hash to {got} — "
                "the npz and the manifest are out of step; refuse to score.")
        return art, man

    # ── scoring ──────────────────────────────────────────────────────────────
    def _whiten(self, v: np.ndarray) -> np.ndarray:
        c = np.asarray(v, dtype=np.float64) - self.abtt_mean
        c = c - (c @ self.abtt_components.T) @ self.abtt_components
        return c / max(float(np.linalg.norm(c)), _EPS)

    def score_pair(self, va, vb) -> float:
        d = np.asarray(vb, dtype=np.float64) - np.asarray(va, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < _EPS:
            return -np.inf                     # identical vectors: no edit
        dh = d / n
        probe = float(self.probe_w @ np.abs(dh)) + self.probe_b
        cw = float(self._whiten(va) @ self._whiten(vb))
        return min((cw - self.T_w) / self.s_w, (probe - self.T_p) / self.s_p)

    def margins_pair(self, va, vb) -> tuple[float, float]:
        """(whitened-cos margin, probe margin) in anchored scale units."""
        d = np.asarray(vb, dtype=np.float64) - np.asarray(va, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < _EPS:
            return -np.inf, -np.inf             # identical vectors: no edit
        dh = d / n
        probe = float(self.probe_w @ np.abs(dh)) + self.probe_b
        cw = float(self._whiten(va) @ self._whiten(vb))
        return (cw - self.T_w) / self.s_w, (probe - self.T_p) / self.s_p

    @staticmethod
    def parse_tau(tau) -> tuple[float, float]:
        """A float tau is the diagonal family (both margins >= tau); a
        'tw:tp' string is a rectangle (per-axis offsets) — the E2E-3
        calibration-only grid amendment."""
        import math
        if isinstance(tau, str) and ":" in tau:
            a, b = tau.split(":", 1)
            tw, tp = float(a), float(b)
        else:
            tw = tp = float(tau)
        if math.isnan(tw) or math.isnan(tp):
            raise ValueError(f"geo tau {tau!r} parses to NaN — a NaN "
                             "threshold silently rejects every pair.")
        return tw, tp

    def pair_filter(self, tau):
        """The gate-shaped callable: True iff both anchored margins clear
        their thresholds. Margins depend only on the two vectors, so they are
        cached across the many (cell, question) combinations one --select
        pass replays. The cache key is the pair of fact TEXTS, not ids —
        fact ids ('fact:<serial>') repeat across subsets with different
        texts, and one artifact instance survives a multi-subset invocation
        (the review caught the id-keyed version serving sh_32k margins to
        sh_64k pairs)."""
        tw, tp = self.parse_tau(tau)

        def _filter(a, b) -> bool:
            key = (a.text, b.text) if a.text <= b.text else (b.text, a.text)
            m = self._cache.get(key)
            if m is None:
                m = self.margins_pair(a.vector, b.vector)
                self._cache[key] = m
            return m[0] >= tw and m[1] >= tp
        return _filter


# ── fitter ───────────────────────────────────────────────────────────────────
def _pool_pairs(subset: str):
    """Unique calibration pool pairs at raw cos >= POOL_COS_MIN with their
    (texts, cos, same_key, same_object, bidirectional NLI contradiction)."""
    import re

    data_file = (REPO / "In-Episode-Knowledge" / "INEP-KNOW" /
                 "MemoryAgentBench" / "data" / "Conflict_Resolution.json")
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    sub_idx = {"sh_6k": 0, "sh_32k": 1}[subset]
    fact_re = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$", re.M)
    facts = {int(m.group(1)): m.group(2)
             for m in fact_re.finditer(raw[sub_idx]["context"])}
    pre = json.loads((REPO / "hnav" / "_out" /
                      f"stage1_prepass_{subset}_benchmarkpage.json"
                      ).read_text(encoding="utf-8"))
    nli = pre["nli_table"]
    sha16 = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]
    seen = {}
    for q in pre["questions"]:
        for p in q["pairs"]:
            if p["cos"] >= POOL_COS_MIN:
                seen.setdefault((p["a"], p["b"]), p)
    rows = []
    for p in seen.values():
        ta = facts[int(p["a"].split(":")[1])]
        tb = facts[int(p["b"].split(":")[1])]
        v1 = nli.get(sha16(ta) + "|" + sha16(tb))
        v2 = nli.get(sha16(tb) + "|" + sha16(ta))
        nli_bi = min(v1[2], v2[2]) if (v1 and v2) else 0.0
        rows.append((ta, tb, float(p["cos"]), bool(p["same_key"]),
                     bool(p["same_object"]), nli_bi))
    return rows


def main() -> int:
    from hnav.core.geometry import ABTTWhitening
    from sklearn.linear_model import LogisticRegression

    from . import data as gfdata
    from .methods import fit_training_edits

    # 1. slot probe on the gold conflict dataset, calibration split only
    records = gfdata.load_records()
    fit_recs = [r for r in records
                if (r["gold_update"] or gfdata.is_hard_negative(r))
                and r["split"] == "calibration"]
    bad = sorted({r["subset"] for r in fit_recs} - set(FIT_SUBSETS))
    if bad:
        raise SystemExit(f" REFUSED: fit data contains {bad}; the GEO artifact "
                         f"may be fit on {FIT_SUBSETS} only.")
    index, V_raw = gfdata.fact_matrix(records)
    pv = gfdata.PairView(records, index)
    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    hard = np.array([gfdata.is_hard_negative(r) for r in records])
    D_pos, _ = fit_training_edits(records, pv, V_raw, gold & cal)
    D_hn = pv.subset(hard & cal).diff(V_raw, normalize=True, oriented=False)
    X = np.abs(np.vstack([D_pos, D_hn])).astype(np.float32)
    y = np.r_[np.ones(len(D_pos)), np.zeros(len(D_hn))]
    lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    lr.fit(X, y)
    probe_w = lr.coef_[0].astype(np.float64)
    probe_b = float(lr.intercept_[0])

    # 2. the frozen ABTT whitening, copied in with its fingerprint pinned
    blob = json.loads(ABTT_ARTIFACT.read_text(encoding="utf-8"))
    wh = ABTTWhitening.from_dict(blob["whitening"])
    assert wh.fitted and blob["embed_cache_namespace"] == gfdata.NAMESPACE

    # 3. anchors + scales from the calibration POOL pairs (deployment frame)
    art0 = GeoIdentityScreen(probe_w, probe_b, wh.mean, wh.components,
                             0.0, 0.0, 1.0, 1.0)
    feats, labels, nli_ok = [], [], []
    per_subset = {}
    for sub in FIT_SUBSETS:
        rows = _pool_pairs(sub)
        per_subset[sub] = len(rows)
        for ta, tb, _cos, same_key, same_obj, nli_bi in rows:
            va = gfdata._load_vec(ta); va = va / np.linalg.norm(va)
            vb = gfdata._load_vec(tb); vb = vb / np.linalg.norm(vb)
            d = vb - va
            dh = d / max(np.linalg.norm(d), _EPS)
            probe = float(probe_w @ np.abs(dh)) + probe_b
            cw = float(art0._whiten(va) @ art0._whiten(vb))
            feats.append((cw, probe))
            labels.append(bool(same_key and not same_obj))
            nli_ok.append(nli_bi >= POOL_NLI_MIN)
    F = np.asarray(feats)
    yy = np.asarray(labels)
    nn = np.asarray(nli_ok)
    s_w = float(F[:, 0].std(ddof=1))
    s_p = float(F[:, 1].std(ddof=1))
    # joint zero-FP staircase over NLI-passing pairs: sweep T_w, T_p = FP max
    best = (-1.0, None)
    for t_w in np.unique(np.quantile(F[:, 0], np.linspace(0, 1, 200))):
        m = nn & (F[:, 0] >= t_w)
        neg = F[m & ~yy, 1]
        t_p = float(neg.max()) + 1e-12 if len(neg) else -np.inf
        rec = float(((F[:, 1] > t_p) & m & yy).sum() / max(yy.sum(), 1))
        if rec > best[0]:
            best = (rec, (float(t_w), t_p))
    T_w, T_p = best[1]
    art = GeoIdentityScreen(probe_w, probe_b, wh.mean, wh.components,
                            T_w, T_p, s_w, s_p)

    provenance = {
        "fit_subsets": list(FIT_SUBSETS),
        "probe": {"n_pos_edits": int(len(D_pos)), "n_neg_edits": int(len(D_hn)),
                  "features": "|d_hat| (2560), LogisticRegression C=1.0 "
                              "class_weight=balanced",
                  "fit_frame": "gold conflict dataset, calibration gold vs "
                               "calibration hard negatives"},
        "anchors": {"pool_cos_min": POOL_COS_MIN, "pool_nli_min": POOL_NLI_MIN,
                    "n_pool_pairs": {k: int(v) for k, v in per_subset.items()},
                    "anchor_zero_fp_recall_unique_pairs": best[0],
                    "rule": "joint zero-FP staircase over NLI-passing "
                            "calibration pool pairs; T_p = max FP probe at "
                            "T_w; scales = feature std over the same pairs"},
        "abtt_source": ABTT_ARTIFACT.relative_to(REPO).as_posix(),
        "abtt_source_fingerprint": blob["whitening"]["fingerprint"],
        "dataset": gfdata.DATASET.relative_to(REPO).as_posix(),
        "dataset_sha256": hashlib.sha256(gfdata.DATASET.read_bytes()).hexdigest(),
        "embed_cache_namespace": gfdata.NAMESPACE,
        "geometry_space": "raw (probe) + committed ABTT D=128 (cos_w)",
        "seed": gfdata.SEED,
        "no_parser_at_inference": "score_pair reads a.vector/b.vector only; "
                                  "no metadata, no parser field",
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    art.save(ARTIFACT_JSON, provenance)
    art2, man = GeoIdentityScreen.load(ARTIFACT_JSON)
    assert art2.fingerprint() == art.fingerprint()
    assert np.array_equal(art2.probe_w, art.probe_w)

    print(f" wrote {ARTIFACT_JSON.name} + .npz")
    print(f"   probe: {len(D_pos)} gold vs {len(D_hn)} hard-neg edits")
    print(f"   anchors: T_w={T_w:.4f} T_p={T_p:.4f} scales s_w={s_w:.4f} "
          f"s_p={s_p:.4f}")
    print(f"   anchor zero-FP recall (unique cal pool pairs): {best[0]:.4f}")
    print(f"   fingerprint {art.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
