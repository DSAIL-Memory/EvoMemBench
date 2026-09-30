"""CES artifact — the frozen contrastive-edit-subspace screen.  [E2E campaign]

Persists the subspaces the pair-level experiments fit (`REPORT.md` §7) in the
same shape the ABTT whitening artifact uses (`hnav/core/geometry.py`): fitted
offline on the CALIBRATION split only, shipped as constants, self-fingerprinted
so a consumer can prove it is scoring with the artifact it thinks it is.

Storage is a JSON manifest + a sibling ``.npz`` holding the matrices (the
per-relation subspaces are ~3.7M floats — too heavy for JSON). The fingerprint
is computed over the ARRAY bytes (float64, sorted relation keys), never over
the npz file bytes, so zip timestamps cannot perturb it.

Scoring (relation-aware, the arm the pair-level gate passed):

    score(a, b) = ||U_obj_r^T d_hat||^2 - ||U_subj_r^T d_hat||^2

with ``r`` = the pair's shared relation template when both facts parse to the
same relation and the artifact holds subspaces for it; the GLOBAL subspaces
otherwise. The score is sign-invariant in (a, b). ``pair_filter(tau)`` wraps
this as the opaque ``(MemoryRecord, MemoryRecord) -> bool`` callable the read
gate accepts — relation identity comes from ``metadata["key"][0]`` (the
parser's relation template); the parser's SUBJECT equality is deliberately not
consulted: replacing that decision with the subject-edit subspace is the
experiment.

Fitter: ``python -m hnav.geometry_filter.ces_artifact`` — refuses fit data
outside sh_6k + sh_32k, asserts a bit-identical round-trip after writing.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

K_DEFAULT = 20
MIN_PAIRS_DEFAULT = 5

REPO = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_JSON = REPO / "stage0_results" / "geometry_filter" / "ces_subspaces_k20.json"
FIT_SUBSETS = ("sh_6k", "sh_32k")

_EPS = 1e-12


class CESArtifact:
    def __init__(self, k: int, min_pairs: int, U_obj_g: np.ndarray,
                 U_subj_g: np.ndarray, relations: dict[str, dict]) -> None:
        self.k = int(k)
        self.min_pairs = int(min_pairs)
        self.U_obj_g = np.ascontiguousarray(U_obj_g, dtype=np.float64)
        self.U_subj_g = np.ascontiguousarray(U_subj_g, dtype=np.float64)
        # relation -> {"U_obj": (dim,k), "U_subj": (dim,k)}; only relations
        # with >= min_pairs examples in BOTH classes are present.
        self.relations = {
            r: {"U_obj": np.ascontiguousarray(m["U_obj"], dtype=np.float64),
                "U_subj": np.ascontiguousarray(m["U_subj"], dtype=np.float64)}
            for r, m in relations.items()}

    @property
    def dim(self) -> int:
        return int(self.U_obj_g.shape[0])

    # ── identity ─────────────────────────────────────────────────────────────
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.U_obj_g.tobytes())
        h.update(self.U_subj_g.tobytes())
        for r in sorted(self.relations):
            h.update(r.encode("utf-8"))
            h.update(self.relations[r]["U_obj"].tobytes())
            h.update(self.relations[r]["U_subj"].tobytes())
        return h.hexdigest()

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, json_path: pathlib.Path, provenance: dict) -> None:
        json_path = pathlib.Path(json_path)
        npz_path = json_path.with_suffix(".npz")
        arrays = {"U_obj_g": self.U_obj_g, "U_subj_g": self.U_subj_g}
        rels = sorted(self.relations)
        for i, r in enumerate(rels):
            arrays[f"rel{i}_U_obj"] = self.relations[r]["U_obj"]
            arrays[f"rel{i}_U_subj"] = self.relations[r]["U_subj"]
        np.savez_compressed(npz_path, **arrays)
        manifest = {
            "artifact": "CES contrastive edit subspaces",
            "k": self.k, "min_pairs": self.min_pairs, "dim": self.dim,
            "n_relations": len(rels), "relation_order": rels,
            "matrices_npz": npz_path.name,
            "fingerprint": self.fingerprint(),
            "provenance": provenance,
        }
        json_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, json_path: pathlib.Path) -> tuple["CESArtifact", dict]:
        json_path = pathlib.Path(json_path)
        man = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(json_path.parent / man["matrices_npz"]) as z:
            rels = {r: {"U_obj": z[f"rel{i}_U_obj"], "U_subj": z[f"rel{i}_U_subj"]}
                    for i, r in enumerate(man["relation_order"])}
            art = cls(man["k"], man["min_pairs"], z["U_obj_g"], z["U_subj_g"], rels)
        got = art.fingerprint()
        if got != man["fingerprint"]:
            raise ValueError(
                f"CES artifact fingerprint mismatch: manifest says "
                f"{man['fingerprint']}, loaded matrices hash to {got}. The "
                f"npz and the manifest are out of step — refuse to score.")
        return art, man

    # ── scoring ──────────────────────────────────────────────────────────────
    def score_pair(self, va, vb, relation: str | None) -> float:
        d = np.asarray(vb, dtype=np.float64) - np.asarray(va, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < _EPS:
            return 0.0  # identical vectors: no edit, neither subspace wins
        d = d / n
        m = self.relations.get(relation) if relation is not None else None
        Uo = m["U_obj"] if m is not None else self.U_obj_g
        Us = m["U_subj"] if m is not None else self.U_subj_g
        return float(np.linalg.norm(d @ Uo) ** 2 - np.linalg.norm(d @ Us) ** 2)

    @staticmethod
    def pair_relation(a, b) -> str | None:
        """The shared relation template of two gate records, or None.

        ``metadata["key"]`` is the parser's ``(relation, subject)``; only the
        RELATION half is read — subject identity is what CES replaces."""
        ka, kb = a.metadata.get("key"), b.metadata.get("key")
        if ka is not None and kb is not None and ka[0] == kb[0]:
            return ka[0]
        return None

    def pair_filter(self, tau: float):
        """The gate-shaped callable: True iff score > tau. Pairs whose relation
        is unknown or unseen fall back to the global subspaces (weaker,
        measured 0.87 AUROC vs 0.98 — counted upstream via the gate's
        ``n_pairs_filter_rejected`` only in aggregate)."""
        def _filter(a, b) -> bool:
            return self.score_pair(a.vector, b.vector,
                                   self.pair_relation(a, b)) > tau
        return _filter


# ── fitter ───────────────────────────────────────────────────────────────────
def main() -> int:
    # heavy/offline imports stay function-local: importing this module must
    # remain free for the pipeline runtime and the no-torch import test.
    import subprocess

    from . import data
    from .methods import fit_training_edits
    from .run_dimension_ideas import ContrastiveSubspace

    records = data.load_records()
    fit_recs = [r for r in records
                if (r["gold_update"] or data.is_hard_negative(r))
                and r["split"] == "calibration"]
    bad = sorted({r["subset"] for r in fit_recs} - set(FIT_SUBSETS))
    if bad:
        raise SystemExit(f" REFUSED: fit data contains {bad}; the CES artifact "
                         f"may be fit on {FIT_SUBSETS} only.")

    index, V_raw = data.fact_matrix(records)
    V = V_raw  # raw space: the space the pair-level result was measured in
    pv = data.PairView(records, index)
    gold = np.array([r["gold_update"] for r in records])
    cal = np.array([r["split"] == "calibration" for r in records])
    hardneg = np.array([data.is_hard_negative(r) for r in records])

    D_pos, rel_pos = fit_training_edits(records, pv, V, gold & cal)
    pv_neg = pv.subset(hardneg & cal)
    D_neg = pv_neg.diff(V, normalize=True, oriented=False)
    ces = ContrastiveSubspace(k=K_DEFAULT, min_pairs=MIN_PAIRS_DEFAULT).fit(
        D_pos, rel_pos, D_neg, list(pv_neg.relation))

    both = sorted(set(ces.U_obj) & set(ces.U_subj))
    art = CESArtifact(K_DEFAULT, MIN_PAIRS_DEFAULT, ces.U_obj_g, ces.U_subj_g,
                      {r: {"U_obj": ces.U_obj[r], "U_subj": ces.U_subj[r]}
                       for r in both})

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        head = "unknown"
    provenance = {
        "fit_subsets": list(FIT_SUBSETS),
        "n_pos_edits": int(len(D_pos)), "n_neg_edits": int(len(D_neg)),
        "n_relations_obj": len(ces.U_obj), "n_relations_subj": len(ces.U_subj),
        "n_relations_both": len(both),
        "dataset": str(data.DATASET.relative_to(REPO)),
        "dataset_sha256": hashlib.sha256(data.DATASET.read_bytes()).hexdigest(),
        "embed_cache_namespace": data.NAMESPACE,
        "geometry_space": "raw",
        "git_head": head,
        "seed": data.SEED,
        "pair_level_evidence": "stage0_results/geometry_filter/REPORT.md sec.7 "
                               "(relation-aware 0.981 hard AUROC; global-only "
                               "FAILED its gate at 0.8725 - 2026-08-27)",
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    art.save(ARTIFACT_JSON, provenance)

    # round-trip must be bit-identical — same discipline as fit_abtt_artifact
    art2, man = CESArtifact.load(ARTIFACT_JSON)
    assert art2.fingerprint() == art.fingerprint()
    assert np.array_equal(art2.U_obj_g, art.U_obj_g)
    for r in both:
        assert np.array_equal(art2.relations[r]["U_obj"], art.relations[r]["U_obj"])
        assert np.array_equal(art2.relations[r]["U_subj"], art.relations[r]["U_subj"])

    print(f" wrote {ARTIFACT_JSON.name} + .npz")
    print(f"   k={art.k} dim={art.dim} relations={len(both)} "
          f"(obj-only {len(ces.U_obj) - len(both)}, "
          f"subj-only {len(ces.U_subj) - len(both)} dropped to global)")
    print(f"   fit: {len(D_pos)} gold edits / {len(D_neg)} hard-negative edits "
          f"on {list(FIT_SUBSETS)}")
    print(f"   fingerprint {art.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
