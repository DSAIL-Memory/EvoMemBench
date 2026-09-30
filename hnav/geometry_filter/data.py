"""Data layer for the geometry-filter experiments.

Loads the committed gold conflict dataset, resolves every fact text to its
cached campaign embedding (Qwen3-Embedding-4B float32 L8192 unit vectors), and
exposes the three embedding *spaces* the experiments compare:

    raw       the cached unit vectors, untouched
    centered  mean-subtracted then re-normalized; the mean is the ABTT
              artifact's mean, i.e. estimated on the 2,765 calibration facts
              (sh_6k + sh_32k) only — no confirmatory data touches the fit
    abtt      the frozen committed ABTT whitening (mean + top-128 principal
              directions removed, re-normalized), same calibration-only fit

A pitfall stated once, here, and asserted in the tests: mean-centering
*without* re-normalization is exactly a no-op on difference vectors, because
``(v2 - mu) - (v1 - mu) = v2 - v1``. The "centered" space is only distinct
from "raw" because each embedding is re-normalized after the subtraction.

Orientation. For parser-tagged pairs ``fact_a`` is always the earlier serial
(verified: 0 violations across all tagged records), so ``d = v_b - v_a`` is the
oriented old→new edit. Untagged pairs have no direction; where a signed
``d`` is needed for them, :func:`orientation_sign` supplies a deterministic
per-pair pseudo-random sign so that no artifact of pair-id ordering can leak
into a signed statistic.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import subprocess
from collections import defaultdict

import numpy as np

SEED = 20260824
NAMESPACE = "Qwen_Qwen3-Embedding-4B|float32|L8192"

REPO = pathlib.Path(__file__).resolve().parents[2]
PAIR_DIR = REPO / "stage0_results" / "conflict_pairs"
DATASET = PAIR_DIR / "gold_conflict_dataset.jsonl.gz"
ABTT_ARTIFACT = REPO / "stage0_results" / "abtt" / "abtt_whitening_D128.json"
CACHE = REPO / "hnav" / "_cache" / "emb"
OUT_DIR = REPO / "stage0_results" / "geometry_filter"

GOLD_TIERS = ("core", "update_only_fork")
SPACES = ("raw", "centered", "abtt")

_EPS = 1e-12


# ── records ──────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    recs = []
    with gzip.open(DATASET, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def relation_of(rec: dict) -> str | None:
    """The pair's relation template, from fact_a (fact_b when only b parses)."""
    p = rec["parser"]
    for side in ("fact_a_parsed", "fact_b_parsed"):
        if p.get(side):
            return p[side]["relation"]
    return None


def slot_class(rec: dict) -> str | None:
    """Which slots changed, from the parser's deterministic string comparison.

    Returns None for pairs where either fact failed to parse. These are slot
    labels, not conflict labels: the parser compares strings, so the label is
    exact for the slot question even where the *conflict* question is disputed.
    """
    p = rec["parser"]
    if not p["both_parse"]:
        return None
    ds, dr, do = (not p["same_subject"]), (not p["same_relation"]), (not p["same_object"])
    return {
        (False, False, True): "object_only",
        (True, False, False): "subject_only",
        (False, True, False): "relation_only",
        (True, False, True): "subject_object",
        (False, True, True): "relation_object",
        (True, True, False): "subject_relation",
        (True, True, True): "all_change",
    }.get((ds, dr, do))  # (F,F,F) cannot occur: identical facts are deduped


def orientation_sign(rec: dict) -> int:
    """+1 for tagged pairs (a=earlier). Deterministic ±1 otherwise."""
    if "serial_earlier" in rec:
        return 1
    h = hashlib.sha256(f"{SEED}|{rec['pair_id']}".encode()).digest()
    return 1 if h[0] % 2 == 0 else -1


def is_hard_negative(rec: dict) -> bool:
    """Verified non-conflict with the same relation template, different subject
    — the cosine-matched adversary class for a conflict detector."""
    p = rec["parser"]
    return (rec["tier"] == "negative" and p["both_parse"]
            and p["same_relation"] and not p["same_subject"])


def transition_key(rec: dict) -> tuple | None:
    p = rec["parser"]
    if not p["both_parse"]:
        return None
    return (p["fact_a_parsed"]["relation"],
            p["fact_a_parsed"]["object"], p["fact_b_parsed"]["object"])


# ── embeddings ───────────────────────────────────────────────────────────────
def _load_vec(text: str) -> np.ndarray:
    h = hashlib.sha256((NAMESPACE + "||" + text).encode()).hexdigest()
    return np.load(CACHE / f"{h}.npy")


def fact_matrix(records: list[dict]) -> tuple[dict[str, int], np.ndarray]:
    """(text → row index, unit-vector matrix) over every distinct fact text."""
    index: dict[str, int] = {}
    rows = []
    for r in records:
        for t in (r["fact_a"], r["fact_b"]):
            if t not in index:
                index[t] = len(rows)
                rows.append(_load_vec(t).astype(np.float64))
    V = np.asarray(rows)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), _EPS)
    return index, V


def build_spaces(V_raw: np.ndarray) -> dict[str, np.ndarray]:
    """The three unit-vector spaces. ABTT parameters come from the committed
    calibration-fit artifact and are applied, never re-fit."""
    from hnav.core.geometry import ABTTWhitening  # lazy: keeps import light

    blob = json.loads(ABTT_ARTIFACT.read_text(encoding="utf-8"))
    w = ABTTWhitening.from_dict(blob["whitening"])
    assert w.fitted and blob["embed_cache_namespace"] == NAMESPACE

    centered = V_raw - np.asarray(w.mean)[None, :]
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), _EPS)
    return {"raw": V_raw, "centered": centered, "abtt": w.transform(V_raw)}


# ── vectorized pair views ────────────────────────────────────────────────────
class PairView:
    """Index-based view of a list of records against a fact matrix.

    Everything downstream works from ``ia``/``ib`` gathers so that no
    (n_pairs × dim) matrix is materialized unless an experiment asks for the
    difference vectors explicitly.
    """

    def __init__(self, records: list[dict], index: dict[str, int]) -> None:
        self.records = records
        self.ia = np.array([index[r["fact_a"]] for r in records])
        self.ib = np.array([index[r["fact_b"]] for r in records])
        self.sign = np.array([orientation_sign(r) for r in records])
        self.relation = [relation_of(r) for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def subset(self, mask) -> "PairView":
        pv = object.__new__(PairView)
        idx = np.flatnonzero(np.asarray(mask))
        pv.records = [self.records[i] for i in idx]
        pv.ia, pv.ib = self.ia[idx], self.ib[idx]
        pv.sign = self.sign[idx]
        pv.relation = [self.relation[i] for i in idx]
        return pv

    def cos(self, V: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", V[self.ia], V[self.ib])

    def diff(self, V: np.ndarray, normalize: bool = True,
             oriented: bool = True) -> np.ndarray:
        """d = v_b - v_a (times the orientation sign when ``oriented``)."""
        D = V[self.ib] - V[self.ia]
        if oriented:
            D = D * self.sign[:, None]
        if normalize:
            D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), _EPS)
        return D


# ── provenance ───────────────────────────────────────────────────────────────
def provenance(**extra) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    return {
        "git_commit": commit,
        "dataset": str(DATASET.relative_to(REPO)),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "embed_cache_namespace": NAMESPACE,
        "abtt_artifact": str(ABTT_ARTIFACT.relative_to(REPO)),
        "seed": SEED,
        **extra,
    }


def relation_fold(relations: list[str], n_folds: int = 2) -> dict[str, int]:
    """Deterministic relation → fold assignment for relation-disjoint splits."""
    out = {}
    for r in sorted(set(relations)):
        h = hashlib.sha256(f"{SEED}|relfold|{r}".encode()).digest()
        out[r] = int.from_bytes(h[:4], "big") % n_folds
    return out


def calibration_positive_sets(records: list[dict]) -> tuple[set, set]:
    """(subjects, oriented transition keys) seen among calibration gold pairs —
    the exclusion lists for subject-/transition-disjoint confirmatory evals."""
    subjects, transitions = set(), set()
    for r in records:
        if r["gold_update"] and r["split"] == "calibration":
            p = r["parser"]
            if p["both_parse"]:
                subjects.add(p["fact_a_parsed"]["subject"])
                subjects.add(p["fact_b_parsed"]["subject"])
                t = transition_key(r)
                transitions.add(t)
                transitions.add((t[0], t[2], t[1]))  # both orientations
    return subjects, transitions
