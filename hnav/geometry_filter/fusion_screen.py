"""Fusion screen — CES + ABTT-cosine, logistic, calibration-fit.  [E2E-2]

Experiment 5 measured the two signals to be tail-complementary with disjoint
failure regimes (CES owns seen-transition tails, ABTT-cosine owns unseen);
the fusion fit chose the *evidence-accumulating* form over OR-fusion on
calibration (TPR@1e-4 0.942 vs 0.547 in-sample; 0.748 vs 0.468 held-out):

    score(a, b) = w1·z(ces) + w2·z(abtt_cos) + b        (a logit)

with z-standardization statistics fit on calibration hard negatives and the
weights by balanced logistic regression on calibration gold vs hard
negatives. Everything the screen needs at decision time is shipped as
constants: the fusion parameters here, the CES subspaces and the ABTT
whitening by pinned fingerprint. The gate-facing contract is identical to
CES: ``score_pair(v_a_raw, v_b_raw, relation)`` and ``pair_filter(theta)``.

Fitter: ``python -m hnav.geometry_filter.fusion_screen`` — calibration only,
bit-identical round-trip assert, sklearn used only inside the fitter.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from .ces_artifact import CESArtifact

REPO = pathlib.Path(__file__).resolve().parents[2]
FUSION_JSON = REPO / "stage0_results" / "geometry_filter" / "fusion_screen.json"
CES_JSON = REPO / "stage0_results" / "geometry_filter" / "ces_subspaces_k20.json"
WHITENING_JSON = REPO / "stage0_results" / "abtt" / "abtt_whitening_D128.json"

_EPS = 1e-12


class FusionScreen:
    def __init__(self, ces: CESArtifact, wh_mean: np.ndarray,
                 wh_components: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                 w: np.ndarray, b: float) -> None:
        self.ces = ces
        self.wh_mean = np.ascontiguousarray(wh_mean, dtype=np.float64)
        self.wh_components = np.ascontiguousarray(wh_components, dtype=np.float64)
        self.mu = np.ascontiguousarray(mu, dtype=np.float64)      # (2,) ces, cos
        self.sd = np.ascontiguousarray(sd, dtype=np.float64)
        self.w = np.ascontiguousarray(w, dtype=np.float64)
        self.b = float(b)

    # ── identity ─────────────────────────────────────────────────────────────
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.mu, self.sd, self.w, np.array([self.b])):
            h.update(a.tobytes())
        h.update(self.ces.fingerprint().encode())
        h.update(self.wh_mean.tobytes())
        h.update(self.wh_components.tobytes())
        return h.hexdigest()

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: pathlib.Path, provenance: dict) -> None:
        blob = {
            "artifact": "CES + ABTT-cosine fusion screen (logistic)",
            "mu": self.mu.tolist(), "sd": self.sd.tolist(),
            "w": self.w.tolist(), "b": self.b,
            "ces_artifact": CES_JSON.relative_to(REPO).as_posix(),
            "ces_fingerprint": self.ces.fingerprint(),
            "whitening_artifact": WHITENING_JSON.relative_to(REPO).as_posix(),
            "whitening_fingerprint": self._wh_fingerprint(),
            "fingerprint": self.fingerprint(),
            "provenance": provenance,
        }
        pathlib.Path(path).write_text(json.dumps(blob, indent=1), encoding="utf-8")

    def _wh_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.wh_mean.tobytes())
        h.update(self.wh_components.tobytes())
        return h.hexdigest()

    @classmethod
    def load(cls, path: pathlib.Path = FUSION_JSON) -> tuple["FusionScreen", dict]:
        blob = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        # tolerate manifests written before path normalization
        for k in ("ces_artifact", "whitening_artifact"):
            blob[k] = blob[k].replace("\\", "/")
        ces, _ = CESArtifact.load(REPO / blob["ces_artifact"])
        if ces.fingerprint() != blob["ces_fingerprint"]:
            raise ValueError("fusion screen: CES artifact fingerprint mismatch")
        from hnav.core.geometry import ABTTWhitening
        wh_blob = json.loads((REPO / blob["whitening_artifact"])
                             .read_text(encoding="utf-8"))
        wh = ABTTWhitening.from_dict(wh_blob["whitening"])
        f = cls(ces, wh.mean, wh.components, np.asarray(blob["mu"]),
                np.asarray(blob["sd"]), np.asarray(blob["w"]), blob["b"])
        if f._wh_fingerprint() != blob["whitening_fingerprint"]:
            raise ValueError("fusion screen: whitening fingerprint mismatch")
        got = f.fingerprint()
        if got != blob["fingerprint"]:
            raise ValueError(
                f"fusion screen fingerprint mismatch: file says "
                f"{blob['fingerprint']}, loaded parameters hash to {got}.")
        return f, blob

    # ── scoring ──────────────────────────────────────────────────────────────
    def _whiten(self, v: np.ndarray) -> np.ndarray:
        u = np.asarray(v, dtype=np.float64) - self.wh_mean
        u = u - (u @ self.wh_components.T) @ self.wh_components
        n = np.linalg.norm(u)
        return u / max(n, _EPS)

    def score_pair(self, va, vb, relation: str | None) -> float:
        ces = self.ces.score_pair(va, vb, relation)
        acos = float(self._whiten(va) @ self._whiten(vb))
        z = (np.array([ces, acos]) - self.mu) / self.sd
        return float(z @ self.w + self.b)

    def pair_filter(self, theta: float):
        def _filter(a, b) -> bool:
            return self.score_pair(a.vector, b.vector,
                                   self.ces.pair_relation(a, b)) > theta
        return _filter


# ── fitter ───────────────────────────────────────────────────────────────────
def main() -> int:
    import hashlib as _hl
    import subprocess

    from . import data

    ces, ces_man = CESArtifact.load(CES_JSON)
    from hnav.core.geometry import ABTTWhitening
    wh_blob = json.loads(WHITENING_JSON.read_text(encoding="utf-8"))
    wh = ABTTWhitening.from_dict(wh_blob["whitening"])

    records = data.load_records()
    index, V_raw = data.fact_matrix(records)
    V = V_raw
    pv = data.PairView(records, index)
    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])
    bad = sorted({r["subset"] for r, m in zip(records, (gold | hardneg) & cal)
                  if m} - {"sh_6k", "sh_32k"})
    if bad:
        raise SystemExit(f" REFUSED: fusion fit data contains {bad}")

    mask = (gold | hardneg) & cal
    view = pv.subset(mask)
    y = gold[mask]
    ces_s = np.array([ces.score_pair(V[view.ia[i]], V[view.ib[i]],
                                     view.relation[i]
                                     if view.relation[i] in ces.relations
                                     else None) for i in range(len(view))])
    Vw = wh.transform(V)
    acos = np.einsum("ij,ij->i", Vw[view.ia], Vw[view.ib])

    mu = np.array([ces_s[~y].mean(), acos[~y].mean()])
    sd = np.array([ces_s[~y].std(), acos[~y].std()])
    Z = np.c_[(ces_s - mu[0]) / sd[0], (acos - mu[1]) / sd[1]]
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Z, y)

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = "unknown"
    f = FusionScreen(ces, wh.mean, wh.components, mu, sd,
                     lr.coef_[0], float(lr.intercept_[0]))
    f.save(FUSION_JSON, {
        "fit_subsets": ["sh_6k", "sh_32k"],
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
        "form": "balanced logistic on z-standardized (ces, abtt_cos); "
                "chosen over max-z on CALIBRATION tails only "
                "(tpr@1e-4 0.942 vs 0.547 in-sample)",
        "dataset_sha256": _hl.sha256(data.DATASET.read_bytes()).hexdigest(),
        "git_head": head, "seed": data.SEED,
    })
    f2, blob = FusionScreen.load(FUSION_JSON)
    assert f2.fingerprint() == f.fingerprint()
    assert np.allclose(f2.w, f.w) and f2.b == f.b
    print(f" wrote {FUSION_JSON.name}")
    print(f"   w={f.w.round(4).tolist()} b={f.b:.4f}  mu={mu.round(4).tolist()} "
          f"sd={sd.round(4).tolist()}")
    print(f"   fingerprint {f.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
