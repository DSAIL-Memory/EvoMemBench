#!/usr/bin/env python3
"""M7 - directed difference-vector geometry of supersession pairs.  [exploratory]

Question
--------
A supersession pair is two facts sharing a ``(relation, subject)`` key and
disagreeing about the object; the later serial supersedes the earlier. Write the
**directed difference**

    delta = v_later - v_earlier

Do these deltas occupy a characteristic *region*, *magnitude range* or
*direction* of the embedding space, relative to non-conflict pairs that are
matched on the things that would otherwise explain any difference?

This is exploratory geometry. **No classifier is trained and no threshold is
derived here**, so the script is safe to run on held-out subsets: nothing it
emits can be tuned on. Every number is descriptive.

The magnitude identity that governs the whole analysis
------------------------------------------------------
Both raw and ABTT-whitened vectors are L2-normalized, so for any pair

    ||delta||^2 = ||v_l||^2 + ||v_e||^2 - 2 v_l.v_e = 2 (1 - cos)

``||delta||`` is therefore a strictly decreasing function of the pair cosine and
carries **no information beyond it**. Reporting "conflict pairs have small
delta norms" and "conflict pairs have high cosine" as two findings would be
counting one fact twice. The script reports both because they were asked for,
labels them as one degree of freedom, and puts the weight on the *direction* of
delta, which is genuinely independent of the cosine.

A prediction this makes about ABTT, stated before measuring
-----------------------------------------------------------
ABTT subtracts a common mean vector ``mu`` from every embedding. The difference
of two whitened vectors is

    (v_l - mu) - (v_e - mu) = v_l - v_e      (before the renormalization step)

so the mean-removal that dominates ABTT's effect on *cosine* geometry cancels
exactly in the *difference* operator. The difference operator is already a
mean-centering. Whitening should therefore move delta geometry far less than it
moves pair cosines. Whether the residual principal-direction removal (D>0) still
matters is the open part, and is measured.

Control sets
------------
A conflict pair is: SAME relation, SAME subject, DIFFERENT object. Each control
varies one factor, so a difference can be attributed:

``same_relation``  same relation, different subject - isolates subject identity.
``same_subject``   same subject, different relation - isolates the relation
                   template. These are the high-cosine "near miss" pairs the
                   cosine screen actually has to reject.
``cos_matched``    arbitrary non-conflict pairs nearest-matched to the conflict
                   cosine distribution. Because of the identity above this
                   equalises ``||delta||`` *by construction*, so any surviving
                   difference is directional. This is the decisive control.
``random``         uniform pairs. Weakest; included as the reference that a
                   naive analysis would stop at.

Every control is drawn at **exactly the conflict count** ``m``, because the
mean-resultant-length null is ``1/sqrt(m)`` and comparing resultants computed at
different ``m`` would be an artefact. Every pair, conflict or control, is
oriented earlier-serial -> later-serial, so direction means the same thing
everywhere.

Usage
-----
    python hnav/stage0/m7_delta_geometry.py --subsets sh_6k sh_64k
    python hnav/stage0/m7_delta_geometry.py --subsets sh_6k --no-figures
    python hnav/stage0/m7_delta_geometry.py --subsets sh_6k sh_32k sh_64k --embed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hnav.adapters.mab_adapter import explode_facts            # noqa: E402
from hnav.config import get_config                             # noqa: E402
from hnav.core.geometry import ABTTWhitening                   # noqa: E402
from hnav.core.embedding import cache_key                      # noqa: E402
from hnav.labeling.conflict_analysis import parse              # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
CALIBRATION = ("sh_6k", "sh_32k")
OUT = REPO / "stage0_results" / "delta_geometry"

# Plot order; "conflict" first so it is the highlighted series everywhere.
# "conflict_matched" is the subset of conflict pairs that a cosine-matched
# control could actually be found for. It is the only fair partner for
# "cos_matched": comparing all 160 conflicts against 50 matched controls would
# compare different pairs, not different geometries.
PAIR_SETS = ("conflict", "conflict_matched", "cos_matched", "same_relation",
             "same_subject", "random")
COLORS = {"conflict": "#c0392b", "conflict_matched": "#e67e22",
          "cos_matched": "#2c3e50", "same_relation": "#2980b9",
          "same_subject": "#16a085", "random": "#95a5a6"}
SPACES = ("raw", "abtt")


# ── vectors ──────────────────────────────────────────────────────────────────
def namespaces(cfg) -> list[str]:
    """Cache namespaces to try, current first.

    The legacy ``model|dtype`` namespace (no ``|L``) predates the 2026-08-15
    truncation fix. For these facts - one short sentence each, far under either
    limit - no truncation occurs, so the two namespaces hold identical vectors;
    ``--verify-namespaces`` proves that rather than assuming it. Vectors are
    never MIXED inside one subset: that would be the T12 wrong-hit failure.
    """
    current = cache_key(cfg.embed_model, cfg.embed_dtype, cfg.embed_max_length)
    legacy = f"{cfg.embed_model}|{cfg.embed_dtype}".replace("/", "_")
    return [current, legacy]


def _path(cache_dir: pathlib.Path, ns: str, text: str) -> pathlib.Path:
    return cache_dir / f"{hashlib.sha256((ns + '||' + text).encode()).hexdigest()}.npy"


def load_vectors(texts: list[str], cfg, embed: bool) -> tuple[np.ndarray, str, dict]:
    """Return (matrix, namespace_used, report). Refuses a partial namespace."""
    cache_dir = pathlib.Path(cfg.emb_cache_dir)
    tried = {}
    for ns in namespaces(cfg):
        hit = sum(1 for t in texts if _path(cache_dir, ns, t).exists())
        tried[ns] = hit
        if hit == len(texts):
            mat = np.stack([np.load(_path(cache_dir, ns, t)) for t in texts])
            return mat.astype(np.float64), ns, {"cache": tried, "embedded": 0}
    if not embed:
        raise RuntimeError(
            f"no cache namespace covers all {len(texts)} texts (best: {tried}). "
            f"Re-run with --embed to compute the missing vectors, or point "
            f"HNAV_EMB_CACHE_DIR at a cache that has them.")
    from hnav.core.embedding import DiskCachedEmbedder, build_embedder
    ns = namespaces(cfg)[0]
    emb = DiskCachedEmbedder(build_embedder(cfg), cache_dir, ns)
    mat = np.asarray(emb.encode(texts), dtype=np.float64)
    return mat, ns, {"cache": tried, "embedded": len(texts) - tried.get(ns, 0)}


# ── facts ────────────────────────────────────────────────────────────────────
class Store:
    """Parsed facts of one subset plus their vectors, in serial order."""

    def __init__(self, name, serial, text, rel, subj, obj, vec, ns, report):
        self.name, self.serial, self.text = name, serial, text
        self.rel, self.subj, self.obj = rel, subj, obj
        self.V, self.namespace, self.load_report = vec, ns, report
        self.key = [(r, s) for r, s in zip(rel, subj)]
        self.n = len(text)

    @property
    def conflicts(self) -> list[tuple[int, int]]:
        """(earlier_idx, later_idx) for each key whose objects disagree."""
        groups = defaultdict(list)
        for i, k in enumerate(self.key):
            groups[k].append(i)
        out = []
        for idxs in groups.values():
            if len({self.obj[i] for i in idxs}) < 2:
                continue
            idxs = sorted(idxs, key=lambda i: self.serial[i])
            out.append((idxs[0], idxs[-1]))
        return sorted(out)


def load_store(name: str, item: dict, cfg, embed: bool, max_facts: int | None,
               exclude: set[str] | None = None) -> Store:
    facts = explode_facts(item["context"])
    n_before = len(facts)
    if exclude:
        # sh_6k is a strict subset of sh_64k, so "sh_6k vs sh_64k" is nested,
        # not independent. Dropping the shared facts leaves the slice of the
        # larger store that the smaller one never saw.
        facts = [(i, t) for i, t in facts if t not in exclude]
    if max_facts:
        facts = facts[:max_facts]
    keep = []
    for serial, text in facts:
        p = parse(text)
        if p is not None:
            keep.append((serial, text) + p)
    texts = [k[1] for k in keep]
    V, ns, rep = load_vectors(texts, cfg, embed)
    rep["n_facts_raw"] = len(facts)
    rep["n_unparsed"] = len(facts) - len(keep)
    rep["n_excluded_as_shared"] = n_before - len(facts)
    return Store(name, [k[0] for k in keep], texts, [k[2] for k in keep],
                 [k[3] for k in keep], [k[4] for k in keep], V, ns, rep)


# ── control construction ─────────────────────────────────────────────────────
def _codes(values: list[str]) -> np.ndarray:
    lut: dict[str, int] = {}
    return np.array([lut.setdefault(v, len(lut)) for v in values], dtype=np.int32)


def _orient(serial: list[int], i: int, j: int) -> tuple[int, int]:
    """Every pair points earlier-serial -> later-serial, so that the SIGN of a
    delta means the same thing for a control as it does for a supersession."""
    return (int(i), int(j)) if serial[i] <= serial[j] else (int(j), int(i))


def caliper_match(pool_cos: np.ndarray, target: np.ndarray, caliper: float
                  ) -> np.ndarray:
    """1:1 nearest-cosine matching without replacement, within ``caliper``.

    Targets are consumed in DESCENDING cosine order: the highest-cosine
    conflicts are the scarcest to match, so they must choose first. Greedy
    matching in an arbitrary order would spend the few available high-cosine
    controls on easy targets and then fail on the hard ones.

    Returns an index into ``pool_cos`` per target, or -1 where no unused
    candidate lies within the caliper.
    """
    order = np.argsort(pool_cos, kind="stable")
    pc = pool_cos[order]
    avail = np.ones(pc.size, dtype=bool)
    out = np.full(target.size, -1, dtype=np.int64)
    for t_idx in np.argsort(-target, kind="stable"):
        t = float(target[t_idx])
        p = int(np.searchsorted(pc, t))
        lo, hi = p - 1, p
        while True:
            d_lo = abs(float(pc[lo]) - t) if lo >= 0 else np.inf
            d_hi = abs(float(pc[hi]) - t) if hi < pc.size else np.inf
            if min(d_lo, d_hi) > caliper:
                break
            if d_lo <= d_hi:
                if avail[lo]:
                    avail[lo] = False
                    out[t_idx] = order[lo]
                    break
                lo -= 1
            else:
                if avail[hi]:
                    avail[hi] = False
                    out[t_idx] = order[hi]
                    break
                hi += 1
    return out


def build_controls(store: Store, conflict: list[tuple[int, int]], gram: np.ndarray,
                   rng: np.random.Generator, caliper: float
                   ) -> tuple[dict[str, list[tuple[int, int]]], dict]:
    """All control sets, drawn from the EXHAUSTIVE pair space of the subset.

    Enumerating every pair rather than sampling matters for ``cos_matched``:
    only a few hundred non-conflict pairs in the whole store reach the conflict
    cosine range, so a sampled pool would silently fail to match and the
    "matched" control would quietly be a lower-cosine control instead.

    ``~same_key`` is the eligibility mask for every control. It removes the
    conflict pairs and also any same-key duplicate, so no control can be a
    supersession in disguise.
    """
    n, m = store.n, len(conflict)
    ser = store.serial
    i, j = np.triu_indices(n, 1)
    cos = gram[i, j].astype(np.float64)
    rel, sub, obj = _codes(store.rel), _codes(store.subj), _codes(store.obj)
    same_rel = rel[i] == rel[j]
    same_sub = sub[i] == sub[j]
    same_key = same_rel & same_sub
    eligible = ~same_key

    target = np.array([gram[a, b] for a, b in conflict], dtype=np.float64)
    diag = {
        "n_pairs_total": int(i.size),
        "n_eligible": int(eligible.sum()),
        "n_same_key": int(same_key.sum()),
        "n_same_key_same_object": int((same_key & (obj[i] == obj[j])).sum()),
        # how much of the conflict cosine range the control space can even reach
        "nonconflict_cos_max": float(cos[eligible].max()) if eligible.any() else None,
        "conflict_cos_min": float(target.min()) if target.size else None,
        "n_eligible_above_conflict_min": int(
            (cos[eligible] >= target.min()).sum()) if target.size else 0,
    }

    def take(mask: np.ndarray, want: int) -> list[tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        pick = rng.choice(idx, size=min(want, idx.size), replace=False)
        return [_orient(ser, i[k], j[k]) for k in pick]

    out: dict[str, list[tuple[int, int]]] = {
        "random": take(eligible, m),
        "same_relation": take(eligible & same_rel, m),
        "same_subject": take(eligible & same_sub, m),
    }

    pool_idx = np.flatnonzero(eligible)
    matched = caliper_match(cos[pool_idx], target, caliper)
    hit = matched >= 0
    out["cos_matched"] = [_orient(ser, i[pool_idx[k]], j[pool_idx[k]])
                          for k in matched[hit]]
    gap = np.abs(cos[pool_idx[matched[hit]]] - target[hit]) if hit.any() \
        else np.zeros(0)
    diag["cos_match"] = {
        "caliper": caliper, "n_target": int(target.size),
        "matched_target_idx": [int(k) for k in np.flatnonzero(hit)],
        "n_matched": int(hit.sum()), "n_unmatched": int((~hit).sum()),
        "mean_abs_gap": float(gap.mean()) if gap.size else None,
        "max_abs_gap": float(gap.max()) if gap.size else None,
        # the conflicts no control could reach are themselves a result
        "unmatched_target_cos": describe(target[~hit]) if (~hit).any() else {},
    }
    diag["n_control_pairs"] = {k: len(v) for k, v in out.items()}
    return out, diag


# ── statistics ───────────────────────────────────────────────────────────────
def deltas(V: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, ...]:
    if not pairs:
        return np.zeros(0), np.zeros(0), np.zeros((0, V.shape[1]))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    cos = np.einsum("ij,ij->i", V[a], V[b])
    d = V[b] - V[a]
    nrm = np.linalg.norm(d, axis=1)
    return cos, nrm, d / np.maximum(nrm, 1e-12)[:, None]


def describe(v: np.ndarray) -> dict:
    if v.size == 0:
        return {}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
            "min": float(v.min()), "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())}


def directional(U: np.ndarray, rng: np.random.Generator,
                max_pairs: int = 300_000, n_perm: int = 200) -> dict:
    """Direction-only statistics of a set of unit delta vectors.

    The null is a **sign-flip permutation**: each delta is randomly re-oriented
    (later-earlier or earlier-later) while everything else is held fixed. That
    destroys any shared *signed* direction but preserves the axis structure, the
    norms and the pair identities, so a statistic that survives it is telling us
    the deltas agree on which way an update points - not merely that they lie
    along a common axis. The analytic ``1/sqrt(m)`` and ``1/sqrt(d)`` values are
    reported alongside as the isotropic reference.
    """
    if U.shape[0] < 3:
        return {}
    m, d = U.shape
    resultant = float(np.linalg.norm(U.mean(axis=0)))
    g = U @ U.T
    iu = np.triu_indices(m, 1)
    c = g[iu]
    n_pairs = c.size
    if c.size > max_pairs:
        c = rng.choice(c, max_pairs, replace=False)
    sv = np.linalg.svd(U, compute_uv=False)
    lam = sv ** 2
    frac = lam / lam.sum()

    # sign-flip null for both statistics, from the same draws
    r_null, a_null = np.empty(n_perm), np.empty(n_perm)
    tr = float(np.trace(g))
    for t in range(n_perm):
        sgn = rng.choice((-1.0, 1.0), size=m)
        r_null[t] = np.linalg.norm((U * sgn[:, None]).mean(axis=0))
        a_null[t] = (float(sgn @ g @ sgn) - tr) / 2.0 / n_pairs
    align_mean = float(c.mean())

    def z(obs, null):
        sd = float(null.std())
        return float((obs - float(null.mean())) / sd) if sd > 0 else float("nan")

    return {
        "n": m, "dim": d,
        # ‖mean direction‖. For m independent isotropic unit vectors the
        # expectation of the squared resultant is exactly 1/m.
        "resultant": resultant, "resultant_null_isotropic": float(1.0 / np.sqrt(m)),
        "resultant_ratio": float(resultant * np.sqrt(m)),
        "resultant_null_signflip_mean": float(r_null.mean()),
        "resultant_null_signflip_sd": float(r_null.std()),
        "resultant_signflip_z": z(resultant, r_null),
        # pairwise alignment. Isotropic reference for ONE cosine: sd 1/sqrt(d).
        "align_mean": align_mean, "align_abs_mean": float(np.abs(c).mean()),
        "align_sd": float(c.std()),
        "align_null_sd_isotropic": float(1.0 / np.sqrt(d)),
        "align_mean_over_null_sd": float(align_mean * np.sqrt(d)),
        "align_null_signflip_sd": float(a_null.std()),
        "align_signflip_z": z(align_mean, a_null),
        # concentration of the direction cloud
        "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "var_top1": float(frac[0]), "var_top10": float(frac[:10].sum()),
        "var_top50": float(frac[:50].sum()),
        "_align_sample": c[rng.choice(c.size, min(20000, c.size), replace=False)],
    }


def heldout_energy(U_fit: np.ndarray, targets: dict[str, np.ndarray],
                   ks: tuple[int, ...]) -> dict:
    """Fraction of each target's squared norm captured by a subspace fitted on
    OTHER data. Rows are unit, so the value is directly a fraction, and the
    structureless baseline is exactly k/d."""
    if U_fit.shape[0] < 4:
        return {}
    _, _, vt = np.linalg.svd(U_fit, full_matrices=False)
    d = U_fit.shape[1]
    out: dict = {"ks": list(ks), "baseline": [k / d for k in ks], "curves": {}}
    for name, X in targets.items():
        if X.size == 0:
            continue
        out["curves"][name] = [float(((X @ vt[:k].T) ** 2).sum(1).mean())
                               for k in ks]
    return out


def relation_decomposition(U: np.ndarray, rels: list[str]) -> dict:
    """Split conflict-delta alignment by whether the two pairs share a relation.

    If alignment is high only within a relation, the structure is a relation
    template effect, not a general 'update direction'."""
    if U.shape[0] < 4:
        return {}
    g = U @ U.T
    iu = np.triu_indices(U.shape[0], 1)
    same = np.array([rels[i] == rels[j] for i, j in zip(*iu)])
    c = g[iu]
    per_rel = {}
    for r in sorted(set(rels)):
        sel = [i for i, x in enumerate(rels) if x == r]
        if len(sel) >= 5:
            sub = g[np.ix_(sel, sel)]
            k = np.triu_indices(len(sel), 1)
            per_rel[r] = {"n_pairs": len(sel), "align_mean": float(sub[k].mean())}
    return {"within_relation": describe(c[same]) if same.any() else {},
            "across_relation": describe(c[~same]) if (~same).any() else {},
            "n_relations": len(set(rels)),
            "per_relation": dict(sorted(per_rel.items(),
                                        key=lambda kv: -kv[1]["align_mean"])[:12])}


# ── per subset × space ───────────────────────────────────────────────────────
def analyse(store: Store, space: str, V: np.ndarray, rng: np.random.Generator,
            ks: tuple[int, ...], caliper: float) -> tuple[dict, dict]:
    gram = (V @ V.T).astype(np.float32)
    np.fill_diagonal(gram, -2.0)
    conflict = store.conflicts
    sets = {"conflict": conflict}
    controls, pair_space = build_controls(store, conflict, gram, rng, caliper)
    sets.update(controls)
    sets["conflict_matched"] = [conflict[k] for k in
                                pair_space["cos_match"]["matched_target_idx"]]

    res: dict = {"space": space, "n_facts": store.n, "pair_space": pair_space,
                 "sets": {}}
    arrays: dict = {}
    for name in PAIR_SETS:
        pairs = sets.get(name, [])
        cos, nrm, U = deltas(V, pairs)
        dstat = directional(U, rng)
        sample = dstat.pop("_align_sample", np.zeros(0))
        res["sets"][name] = {"n_pairs": len(pairs), "cos": describe(cos),
                             "delta_norm": describe(nrm), "direction": dstat}
        arrays[name] = {"cos": cos, "norm": nrm, "U": U, "align": sample}

    # held-out subspace: fit on half the conflict deltas, score the other half
    U = arrays["conflict"]["U"]
    if U.shape[0] >= 8:
        perm = rng.permutation(U.shape[0])
        half = U.shape[0] // 2
        fit, held = U[perm[:half]], U[perm[half:]]
        # "conflict_matched" is EXCLUDED as a target: it is a subset of the
        # conflict pairs, so half of it sat in the fitting half and its curve
        # would be measuring memorisation, not generalisation.
        targets = {"conflict_heldout": held}
        targets.update({k: arrays[k]["U"] for k in PAIR_SETS
                        if k not in ("conflict", "conflict_matched")})
        res["heldout_energy"] = heldout_energy(fit, targets, ks)

    rels = [store.rel[i] for i, _ in conflict]
    res["relation_decomposition"] = relation_decomposition(U, rels)

    return res, arrays


# ── figures ──────────────────────────────────────────────────────────────────
def figures(store_arrays: dict, results: dict, outdir: pathlib.Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    subsets = list(store_arrays.keys())
    written = []
    MARK = {"sh_6k": "o", "sh_32k": "s", "sh_64k": "^"}

    def grid(nrow, ncol, w=3.4, h=2.7):
        f, ax = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
        return f, ax

    def save(fig, name, legend_rows=1):
        fig.tight_layout(rect=(0, 0.055 * legend_rows, 1, 0.965))
        fig.savefig(outdir / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    def legend(fig, ax, ncol=6):
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))

    # 1 — pair cosine, and the delta norm it determines ---------------------
    fig, ax = grid(len(subsets), 4, w=3.0, h=2.4)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            for col, quant, lab in ((0, "cos", "pair cosine"), (1, "norm", r"$\|\Delta\|$")):
                a = ax[r][c * 2 + col]
                for name in PAIR_SETS:
                    v = store_arrays[s][space][name][quant]
                    if v.size:
                        a.hist(v, bins=45, density=True, histtype="step", lw=1.3,
                               color=COLORS[name], label=name)
                a.set_title(f"{s} · {space} · {lab}", fontsize=8)
                a.set_xlabel(lab)
    legend(fig, ax[0][0], ncol=6)
    fig.suptitle(r"Pair cosine and $\|\Delta\|$ are ONE degree of freedom:  "
                 r"$\|\Delta\|^2 = 2(1-\cos)$", fontsize=9)
    save(fig, "fig1_magnitude.png")

    # 2 — alignment distributions, zoomed to where the mass is --------------
    fig, ax = grid(len(subsets), 2)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            null = results[s][space]["sets"]["conflict"]["direction"].get(
                "align_null_sd_isotropic", 0.02)
            for name in PAIR_SETS:
                v = store_arrays[s][space][name]["align"]
                if not v.size:
                    continue
                a.hist(v, bins=np.linspace(-0.5, 0.5, 121), density=True,
                       histtype="step", lw=1.3, color=COLORS[name], label=name)
                a.axvline(float(v.mean()), color=COLORS[name], lw=0.9, ls=":")
            a.axvspan(-2 * null, 2 * null, color="k", alpha=0.08, zorder=0)
            a.set_yscale("log")
            a.set_xlim(-0.5, 0.5)
            a.set_title(f"{s} · {space}", fontsize=8)
            a.set_xlabel(r"$\cos(\hat\Delta_i,\hat\Delta_j)$")
    legend(fig, ax[0][0])
    fig.suptitle(r"Pairwise alignment of $\hat\Delta$ (log density; dotted = set mean; "
                 r"grey band = $\pm2/\sqrt{d}$)", fontsize=9)
    save(fig, "fig2_alignment.png")

    # 3 — spectrum ----------------------------------------------------------
    fig, ax = grid(len(subsets), 2)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            for name in PAIR_SETS:
                U = store_arrays[s][space][name]["U"]
                if U.shape[0] < 3:
                    continue
                lam = np.linalg.svd(U, compute_uv=False) ** 2
                a.plot(np.arange(1, len(lam) + 1), np.cumsum(lam) / lam.sum(),
                       color=COLORS[name], lw=1.3, label=name)
            n = min(store_arrays[s][space]["conflict"]["U"].shape)
            a.plot(np.arange(1, n + 1), np.arange(1, n + 1) / n, "k--", lw=0.8,
                   label="isotropic")
            a.set_xscale("log")
            a.set_title(f"{s} · {space}", fontsize=8)
            a.set_xlabel("component"); a.set_ylabel("cumulative variance")
    legend(fig, ax[0][0], ncol=7)
    fig.suptitle(r"Spectrum of the $\hat\Delta$ cloud. Following the dashed line "
                 r"means no preferred direction", fontsize=9)
    save(fig, "fig3_spectrum.png")

    # 4 — projection onto the conflict principal plane ----------------------
    fig, ax = grid(len(subsets), 2, w=3.4, h=3.0)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            U = store_arrays[s][space]["conflict"]["U"]
            if U.shape[0] < 3:
                continue
            _, _, vt = np.linalg.svd(U, full_matrices=False)
            P = vt[:2].T
            for name in ("random", "same_subject", "same_relation", "cos_matched",
                         "conflict"):
                X = store_arrays[s][space][name]["U"]
                if X.size:
                    p = X @ P
                    a.scatter(p[:, 0], p[:, 1], s=5, alpha=0.5, lw=0,
                              color=COLORS[name], label=name)
            a.set_title(f"{s} · {space}", fontsize=8)
            a.set_xlabel(r"PC1 of conflict $\hat\Delta$"); a.set_ylabel("PC2")
    legend(fig, ax[0][0], ncol=5)
    fig.suptitle("Controls projected into the plane fitted to the CONFLICT deltas — "
                 "the projection most favourable to a conflict cluster", fontsize=9)
    save(fig, "fig4_projection.png")

    # 5 — held-out subspace energy, as a multiple of the random-subspace value
    fig, ax = grid(len(subsets), 2)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            he = results[s][space].get("heldout_energy") or {}
            if not he:
                continue
            ks, base = np.array(he["ks"]), np.array(he["baseline"])
            for name, curve in he["curves"].items():
                a.plot(ks, np.array(curve) / base, lw=1.5, marker="o", ms=2.8,
                       color=COLORS.get(name.replace("conflict_heldout", "conflict")),
                       label=name)
            a.axhline(1.0, color="k", ls="--", lw=0.9, label="random subspace")
            a.set_xscale("log"); a.set_yscale("log")
            a.set_title(f"{s} · {space}", fontsize=8)
            a.set_xlabel("subspace rank k")
            a.set_ylabel(r"captured energy $\div$ $k/d$")
    legend(fig, ax[0][0])
    fig.suptitle("Subspace fitted on HALF the conflict deltas, scored on the held-out "
                 "half and on controls", fontsize=9)
    save(fig, "fig5_heldout_energy.png")

    # 6 — relation decomposition -------------------------------------------
    fig, ax = grid(len(subsets), 3, w=3.2, h=2.6)
    for r, s in enumerate(subsets):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            rd = results[s][space].get("relation_decomposition") or {}
            w_, x_ = rd.get("within_relation") or {}, rd.get("across_relation") or {}
            ctrl = results[s][space]["sets"]["cos_matched"]["direction"].get(
                "align_mean")
            if not w_ or not x_:
                continue
            names = ["same\nrelation", "different\nrelation", "cos-matched\ncontrol"]
            vals = [w_["mean"], x_["mean"], ctrl if ctrl is not None else 0.0]
            errs = [w_["sd"], x_["sd"], 0.0]
            a.bar(names, vals, color=["#8e44ad", "#7f8c8d", COLORS["cos_matched"]],
                  alpha=0.85, yerr=errs, capsize=3)
            nullsd = results[s][space]["sets"]["conflict"]["direction"].get(
                "align_null_sd_isotropic", 0.02)
            a.axhspan(-2 * nullsd, 2 * nullsd, color="k", alpha=0.08, zorder=0)
            a.axhline(0, color="k", lw=0.6)
            a.set_title(f"{s} · {space}", fontsize=8)
            a.set_ylabel(r"mean $\cos(\hat\Delta_i,\hat\Delta_j)$")
        # third column: which relations carry it (raw space)
        a = ax[r][2]
        per = (results[s]["raw"].get("relation_decomposition") or {}).get(
            "per_relation", {})
        if per:
            items = list(per.items())[:10][::-1]
            lbl = [k.replace("|", " __ ").strip()[:34] for k, _ in items]
            a.barh(lbl, [v["align_mean"] for _, v in items], color="#8e44ad",
                   alpha=0.85)
            a.set_title(f"{s} · raw · top relations", fontsize=8)
            a.set_xlabel(r"within-relation mean $\cos(\hat\Delta_i,\hat\Delta_j)$")
            a.tick_params(axis="y", labelsize=6)
    fig.suptitle("Is the shared direction a general 'update' direction, or one "
                 "direction per relation template?", fontsize=9)
    save(fig, "fig6_relation_split.png", legend_rows=0)

    # 7 — effect summary, everything on the sign-flip null scale ------------
    fig, ax = grid(2, 2, w=4.2, h=3.0)
    stats = (("resultant_signflip_z", r"shared-direction strength  $z$"),
             ("align_signflip_z", r"pairwise alignment  $z$"))
    xs = np.arange(len(PAIR_SETS))
    for r, (key, lab) in enumerate(stats):
        for c, space in enumerate(SPACES):
            a = ax[r][c]
            for k, s in enumerate(subsets):
                off = (k - (len(subsets) - 1) / 2) * 0.22
                y = [results[s][space]["sets"][n]["direction"].get(key, np.nan)
                     for n in PAIR_SETS]
                a.scatter(xs + off, y, s=34, marker=MARK.get(s, "o"),
                          color=[COLORS[n] for n in PAIR_SETS],
                          edgecolor="k", linewidth=0.4, label=s if r == 0 and c == 0
                          else None, zorder=3)
            a.axhspan(-2, 2, color="k", alpha=0.09, zorder=0)
            a.axhline(0, color="k", lw=0.6)
            a.set_yscale("symlog", linthresh=2)
            a.set_xticks(xs)
            a.set_xticklabels([p.replace("_", "\n") for p in PAIR_SETS], fontsize=7)
            a.set_title(f"{space}", fontsize=9)
            a.set_ylabel(lab)
    h = [plt.Line2D([], [], ls="", marker=MARK[s], color="0.35", label=s)
         for s in subsets if s in MARK]
    h.append(plt.Line2D([], [], color="k", alpha=0.3, lw=6,
                        label=r"$|z|<2$: indistinguishable from the null"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Effect summary against the sign-flip null (each delta randomly "
                 "re-oriented). Marker colour = pair set", fontsize=9)
    save(fig, "fig7_effect_summary.png")
    return written


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=["sh_6k"])
    ap.add_argument("--whiten-d", type=int, default=128,
                    help="ABTT components removed (campaign value: 128)")
    ap.add_argument("--whiten-regime", choices=("frozen_global", "per_store"),
                    default="frozen_global",
                    help="frozen_global fits on the calibration split only")
    ap.add_argument("--whitening-artifact",
                    default=str(REPO / "stage0_results/abtt/abtt_whitening_D128.json"),
                    help="load the campaign's frozen (mu, C) instead of refitting; "
                         "'-' to refit")
    ap.add_argument("--embed", action="store_true",
                    help="compute missing vectors instead of failing")
    ap.add_argument("--max-facts", type=int, default=0)
    ap.add_argument("--novel-vs", nargs="*", default=[],
                    help="drop facts that also occur in these subsets, leaving "
                         "only the part of the store they do not share")
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--caliper", type=float, default=0.02,
                    help="max cosine gap allowed when matching a control")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    cfg = get_config()
    cfg.require_not_live()          # Stage-0 measurement never runs against live
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = {}
    for item in data:
        nm = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0] \
            .replace("factconsolidation_", "")
        items[nm] = item

    stores: dict[str, Store] = {}
    for s in args.subsets:
        if s not in items:
            print(f"  unknown subset {s}", file=sys.stderr)
            return 1
        shared: set[str] = set()
        for other in args.novel_vs:
            if other in items and other != s:
                shared.update(t for _, t in explode_facts(items[other]["context"]))
        stores[s] = load_store(s, items[s], cfg, args.embed,
                               args.max_facts or None, shared or None)
        st = stores[s]
        print(f"{s}: {st.n} parsed facts ({st.load_report['n_unparsed']} unparsed), "
              f"{len(st.conflicts)} conflict pairs, ns={st.namespace}")

    # -- whitener ---------------------------------------------------------
    whit_meta: dict = {"regime": args.whiten_regime, "d": args.whiten_d}
    art = pathlib.Path(args.whitening_artifact) if args.whitening_artifact != "-"         else None
    if art is not None and art.exists():
        # Prefer the campaign's own artifact over refitting: it was fitted on
        # all 2,765 calibration facts including the ones this script's parser
        # drops, so a refit here would silently be a DIFFERENT space.
        blob = json.loads(art.read_text(encoding="utf-8"))
        w = ABTTWhitening.from_dict(blob["whitening"])
        held = [x for x in blob.get("fit_subsets", []) if x not in CALIBRATION]
        if held:
            print(f"  REFUSED: artifact was fitted on held-out {held}",
                  file=sys.stderr)
            return 2
        ns_used = {st.namespace for st in stores.values()}
        whit_meta.update(
            source=str(art.relative_to(REPO)), regime=blob.get("regime"),
            d=w.n_components, n_fit=w.n_fit, fitted=w.fitted,
            fit_subsets=list(blob.get("fit_subsets", [])),
            fingerprint=w.fingerprint(),
            artifact_namespace=blob.get("embed_cache_namespace"),
            vector_namespaces=sorted(ns_used),
            namespace_match=blob.get("embed_cache_namespace") in ns_used)
        print(f"ABTT loaded {art.name}: {blob.get('regime')} D={w.n_components} "
              f"n_fit={w.n_fit} fit_on={list(blob.get('fit_subsets', []))} "
              f"fp={str(w.fingerprint())[:16]}")
        if not whit_meta["namespace_match"]:
            print(f"  NOTE: artifact namespace {blob.get('embed_cache_namespace')!r} "
                  f"vs vectors {sorted(ns_used)} - equivalent only if verified",
                  file=sys.stderr)
    elif args.whiten_regime == "frozen_global":
        cal = [stores[s].V for s in stores if s in CALIBRATION]
        if not cal:
            print("  frozen_global needs a calibration subset (sh_6k/sh_32k) in "
                  "--subsets; falling back to per_store", file=sys.stderr)
            args.whiten_regime = "per_store"
            whit_meta["regime"] = "per_store"
            whit_meta["fallback_reason"] = "no calibration subset loaded"
        else:
            w = ABTTWhitening(args.whiten_d, cfg.whiten_min_fit_n).fit(np.vstack(cal))
            whit_meta.update(fit_subsets=[s for s in stores if s in CALIBRATION],
                             n_fit=w.n_fit, fitted=w.fitted,
                             fingerprint=w.fingerprint())
            print(f"ABTT frozen_global D={args.whiten_d} fitted on "
                  f"{whit_meta['fit_subsets']} n={w.n_fit} "
                  f"fp={str(w.fingerprint())[:16]}")

    results: dict = {}
    arrays: dict = {}
    ks = tuple(k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k < 2560)
    loaded_artifact = "source" in whit_meta
    for s, st in stores.items():
        if args.whiten_regime == "per_store" and not loaded_artifact:
            w = ABTTWhitening(args.whiten_d, cfg.whiten_min_fit_n).fit(st.V)
            whit_meta.setdefault("per_store", {})[s] = {
                "n_fit": w.n_fit, "fitted": w.fitted,
                "fingerprint": w.fingerprint()}
        results[s], arrays[s] = {}, {}
        for space in SPACES:
            V = st.V if space == "raw" else w.transform(st.V)
            V = np.asarray(V, dtype=np.float64)
            r, a = analyse(st, space, V, np.random.default_rng(args.seed), ks,
                           args.caliper)
            results[s][space], arrays[s][space] = r, a
            cs = r["sets"]["conflict"]
            print(f"  {s:8s} {space:4s} conflict n={cs['n_pairs']:5d} "
                  f"cos={cs['cos']['mean']:.4f} |D|={cs['delta_norm']['mean']:.4f} "
                  f"R={cs['direction']['resultant']:.4f} "
                  f"(ratio {cs['direction']['resultant_ratio']:.2f}, "
                  f"signflip z={cs['direction']['resultant_signflip_z']:+.1f}) "
                  f"align={cs['direction']['align_mean']:+.4f} "
                  f"PR={cs['direction']['participation_ratio']:.1f}")

    payload = {
        "measurement": "M7 delta-vector geometry of supersession pairs (exploratory)",
        "no_classifier": True,
        "no_threshold_derived": True,
        "config": {"embed_model": cfg.embed_model, "embed_dtype": cfg.embed_dtype,
                   "embed_max_length": cfg.embed_max_length, "seed": args.seed,
                   "namespaces": {s: st.namespace for s, st in stores.items()},
                   "load_report": {s: st.load_report for s, st in stores.items()}},
        "whitening": whit_meta,
        "identity_note": "||delta||^2 = 2(1-cos) for unit vectors: the norm and "
                         "the cosine are one degree of freedom, not two.",
        "subsets": results,
    }
    (outdir / "m7_delta_geometry.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {outdir / 'm7_delta_geometry.json'}")

    if not args.no_figures:
        names = figures(arrays, results, outdir)
        print("figures: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
