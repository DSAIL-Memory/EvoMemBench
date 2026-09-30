"""Artifact-only scoring API.  [QDA]

The one module in this package a shadow/live arm may call: it loads the
committed ``weights.npz`` + manifest, verifies the fingerprint, and scores
raw embedding pairs. It never reads the gold dataset, the benchmark files,
or anything else offline-tier — keep it that way.

    score_pairs(v1, v2, serial1, serial2) -> float          (single pair)
    score_batch(V1, V2, serials1, serials2) -> np.ndarray   (vectorized)

Orientation is taken from the serials (write-time information, available
online): d = v(later) - v(earlier). The symmetric core is invariant to a
swap; the ordered term flips sign with it, which is exactly why the serials
are part of the signature.

The norm term needs ||Q d|| without shipping Q: ||Q d||^2 =
||d||^2 - ||A d||^2 with A the 128 frozen ABTT components (A ⟂ Q spans the
rest of R^2560), so only the small A matrix is stored.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "stage0_results" / "qda_filter"
WEIGHTS_NPZ = OUT_DIR / "weights.npz"
MANIFEST = OUT_DIR / "weights_manifest.json"

_EPS = 1e-12
_ARRAY_KEYS = ("M", "A", "U_obj", "U_subj", "w_obj", "w_subj",
               "ordered_coef")


def fingerprint(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in _ARRAY_KEYS:
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def save_artifact(arrays: dict, scalars: dict, provenance: dict) -> None:
    np.savez_compressed(WEIGHTS_NPZ, **arrays)
    MANIFEST.write_text(json.dumps({
        "artifact": "QDA conflict scorer weights",
        "scalars": scalars,
        "fingerprint": fingerprint(arrays),
        "provenance": provenance,
    }, indent=1), encoding="utf-8")


class QDAScorer:
    """Loads the frozen artifact; ``variant`` picks V2 / V3 / V4."""

    def __init__(self, variant: str = "V4",
                 weights: pathlib.Path = WEIGHTS_NPZ,
                 manifest: pathlib.Path = MANIFEST) -> None:
        man = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8"))
        with np.load(weights) as z:
            raw = {k: z[k] for k in _ARRAY_KEYS}
        got = fingerprint(raw)   # over the stored bytes, before any cast
        self.arrays = {k: v.astype(np.float64) for k, v in raw.items()}
        if got != man["fingerprint"]:
            raise ValueError(
                f"QDA artifact fingerprint mismatch: manifest says "
                f"{man['fingerprint']}, arrays hash to {got} — refuse to "
                f"score with an artifact that is out of step.")
        s = man["scalars"]
        self.w_perp = float(s["w_perp"])
        self.beta = float(s["beta"])
        self.ordered_on = bool(s["ordered_on"])
        self.variant = variant
        assert variant in ("V2", "V3", "V4")
        self.manifest = man

    def score_batch(self, V1, V2, serials1, serials2) -> np.ndarray:
        V1 = np.asarray(V1, dtype=np.float64)
        V2 = np.asarray(V2, dtype=np.float64)
        s1 = np.asarray(serials1)
        s2 = np.asarray(serials2)
        if V1.ndim == 1:
            V1, V2 = V1[None], V2[None]
            s1, s2 = np.atleast_1d(s1), np.atleast_1d(s2)
        sign = np.where(s2 >= s1, 1.0, -1.0)
        d = (V2 - V1) * sign[:, None]            # oriented old -> new
        a = self.arrays
        z = d @ a["M"].T
        po = z @ a["U_obj"]
        ps = z @ a["U_subj"]
        sq = np.einsum("ij,ij->i", z, z)
        perp = sq - np.einsum("ij,ij->i", po, po) \
            - np.einsum("ij,ij->i", ps, ps)
        score = 0.5 * ((po ** 2) @ a["w_obj"] + (ps ** 2) @ a["w_subj"]
                       + self.w_perp * perp)
        if self.variant in ("V3", "V4") and self.ordered_on:
            score = score + z @ a["ordered_coef"]
        if self.variant == "V4":
            n2 = (np.einsum("ij,ij->i", d, d)
                  - np.einsum("ij,ij->i", d @ a["A"].T, d @ a["A"].T))
            score = score + self.beta * 0.5 * np.log(np.maximum(n2, _EPS))
        return score

    def score_pairs(self, v1, v2, serial1, serial2) -> float:
        return float(self.score_batch(v1, v2, [serial1], [serial2])[0])


def score_pairs(v1, v2, serial1, serial2, variant: str = "V4") -> float:
    """Convenience one-shot wrapper (loads the artifact each call — use
    :class:`QDAScorer` for anything hot)."""
    return QDAScorer(variant=variant).score_pairs(v1, v2, serial1, serial2)
