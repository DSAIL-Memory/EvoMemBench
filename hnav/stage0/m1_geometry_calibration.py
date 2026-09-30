#!/usr/bin/env python3
"""M1 — geometry calibration with real embeddings.  [STAGE-0 GATE / kill switch]

Replaces the char-3gram LEXICAL PROXY in hnav/labeling/marginal_diff.py with real
embedding cosines. No threshold anywhere in H-Nav may be derived from the proxy.

Question this answers:
    Are the (stale, superseding) fact pairs in EvoMemBench's Conflict_Resolution
    actually near-duplicates in the embedding space the retriever uses, while the
    changed span is semantically disjoint?

    If yes  -> the marginal-diff premise holds; geometry modules are worth building.
    If no   -> STOP S3 fires. Median whole_blob_sim < 0.70 means near-duplicate
               geometry cannot detect supersession here, and the geometry half of
               the thesis is dead. Only read-side repair would remain viable.

Runs in-process on the GPU. Deliberately requires NO served endpoint, so the gate
is reachable before any serving infrastructure exists.

    python hnav/stage0/m1_geometry_calibration.py                 # all subsets
    python hnav/stage0/m1_geometry_calibration.py --subsets sh_6k # quick smoke

Leakage: reads ONLY `context` from the dataset. Never touches `questions`/`answers`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hnav" / "labeling"))
from conflict_analysis import parse as parse_fact  # noqa: E402  (validated, 99.5%+ coverage)

# The ONE shared piece of package code this otherwise-standalone script uses:
# the fused-kernel eligibility fix (T13). Imported, never re-implemented, so
# the standalone and package embedders cannot drift apart on attention memory.
# hnav/core/embedding.py imports no torch at module level, so this stays cheap.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from hnav.core.embedding import expanded_gqa_attention as _expanded_gqa_attention  # noqa: E402

DATA = REPO / "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json"
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)


# ── env ──────────────────────────────────────────────────────────────────────
def load_env() -> dict:
    env = dict(os.environ)
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.split("#")[0].strip())
    return env


def require_not_live(env: dict) -> None:
    """Same refusal as ``hnav.config.HNavConfig.require_not_live``, inlined.

    This script is near-standalone (stdlib + numpy; it imports the validated
    parser by path). It does import ``hnav.core.embedding`` for the shared
    attention-memory fix (T13) — a torch-free module import — but deliberately
    not ``hnav.config``, so the guard is reproduced rather than imported and
    the script keeps working from a bare checkout. Post-T8 the LIVE mode exists for
    the Stage-1 read-path campaign only; a Stage-0 measurement taken under
    intervention is not a Stage-0 measurement. Enforced uniformly across
    ``hnav/stage0/`` by ``hnav/tests/test_stage0_refuses_live.py``.
    """
    if env.get("HNAV_MODE", "off").strip().lower() == "live":
        raise RuntimeError(
            "HNAV_MODE=live but this code path is a Stage-0 measurement, "
            "which permits off/shadow only. Live is reserved for the "
            "Stage-1 read-path campaign (STAGE1_PLAN.md)."
        )


# ── embedder ─────────────────────────────────────────────────────────────────
class Embedder:
    """In-process HF embedder, mean-pooled + L2-normalized.

    Pooling mirrors Qwen3Embedding4BEmbeddings at
    MemoryAgentBench/methods/embedding_retriever.py:58 so that M1's geometry is
    the geometry the benchmark's own retriever would operate in.
    """

    def __init__(self, model_name: str, device: int, dtype: str, batch: int, cache: Path,
                 max_length: int = 8192):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch = batch
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_length = int(max_length)
        # Namespace includes the truncation length — the twin of
        # hnav.core.embedding.cache_key, kept byte-identical to it so this
        # standalone script and the package share cached vectors (T12).
        self.key = f"{model_name}|{dtype}|L{self.max_length}".replace("/", "_")

        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.dtype = {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[dtype]
        print(f"  loading {model_name} on {self.device} ({dtype}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        self.gqa_expansion_applied = None
        print(f"  loaded. hidden_size = {self.dim}", flush=True)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha256((self.key + "||" + text).encode()).hexdigest()
        return self.cache / f"{h}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                out[i] = np.load(p)
            else:
                todo.append(i)

        for s in range(0, len(todo), self.batch):
            idx = todo[s: s + self.batch]
            batch_texts = [texts[i] for i in idx]
            enc = self.tok(batch_texts, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            # Same fused-kernel eligibility fix as hnav/core/embedding.py (T13):
            # at L8192 a single MAB chunk is ~4.9k tokens, and without this the
            # math fallback's [1, 32, N, N] score matrix OOMs the card. Imported
            # rather than re-implemented so the two paths cannot drift.
            with self.torch.no_grad(), _expanded_gqa_attention() as applied:
                self.gqa_expansion_applied = applied
                res = self.model(**enc)
            last = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
            mask = enc["attention_mask"].unsqueeze(-1).expand(last.size()).to(last.dtype)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs = pooled.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = vecs[j]
                np.save(self._cache_path(texts[i]), vecs[j])
            if s % (self.batch * 20) == 0 and todo:
                print(f"    embedded {min(s + self.batch, len(todo))}/{len(todo)} new", flush=True)

        return np.stack([o for o in out])  # type: ignore[misc]


# ── data ─────────────────────────────────────────────────────────────────────
def build_pairs(context: str, rng: random.Random):
    """-> (conflict_pairs, control_pairs, n_facts, n_parsed)

    conflict pair : the two facts sharing (relation, subject) with different objects.
                    old = lower serial, new = higher serial.
    control pair  : two facts with DIFFERENT (relation, subject), sampled to match count.
    """
    facts = FACT_RE.findall(context)
    groups: dict[tuple, list] = defaultdict(list)
    parsed_all = []
    for num, txt in facts:
        p = parse_fact(txt)
        if p:
            rel, subj, obj = p
            groups[(rel, subj)].append((int(num), txt, obj))
            parsed_all.append((int(num), txt, obj, rel, subj))

    conflicts = []
    for (rel, subj), v in groups.items():
        if len({o for _, _, o in v}) > 1:
            v = sorted(v)
            old, new = v[0], v[-1]
            conflicts.append({"relation": rel, "subject": subj,
                              "old_serial": old[0], "old_text": old[1], "old_obj": old[2],
                              "new_serial": new[0], "new_text": new[1], "new_obj": new[2]})

    controls = []
    if len(parsed_all) >= 2:
        for _ in range(len(conflicts)):
            a, b = rng.sample(parsed_all, 2)
            if (a[3], a[4]) == (b[3], b[4]):
                continue
            controls.append({"old_text": a[1], "old_obj": a[2],
                             "new_text": b[1], "new_obj": b[2]})

    return conflicts, controls, len(facts), len(parsed_all)


# ── metrics ──────────────────────────────────────────────────────────────────
def pcts(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    return {"mean": float(x.mean()), "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg), ties counted as 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def qr_residual(new_vecs: np.ndarray, bank: np.ndarray, max_basis: int = 512) -> np.ndarray:
    """Novelty of each new vector against the span of `bank`.

    r = ||v - P_bank v||, where P_bank projects onto the column space of the bank.
    Bank is subsampled to max_basis columns to keep the QR tractable; the residual
    is then an UPPER bound on true novelty. Documented, not hidden.
    """
    if bank.shape[0] == 0:
        return np.ones(new_vecs.shape[0], dtype=np.float32)
    if bank.shape[0] > max_basis:
        idx = np.random.default_rng(0).choice(bank.shape[0], max_basis, replace=False)
        bank = bank[idx]
    q, _ = np.linalg.qr(bank.T)                     # (d, k)
    proj = new_vecs @ q                             # (n, k)
    recon = proj @ q.T                              # (n, d)
    return np.linalg.norm(new_vecs - recon, axis=1)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="e.g. sh_6k sh_32k (default: all single-hop)")
    ap.add_argument("--include-mh", action="store_true",
                    help="also run multi-hop subsets (exploratory only)")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap conflict pairs per subset (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = load_env()
    require_not_live(env)
    model = env.get("HNAV_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
    device = int(env.get("HNAV_EMBED_DEVICE", "1"))
    dtype = env.get("HNAV_EMBED_DTYPE", "float32")
    batch = int(env.get("HNAV_EMBED_BATCH", "32"))
    max_length = int(env.get("HNAV_EMBED_MAX_LENGTH", "8192"))
    cache = REPO / env.get("HNAV_CACHE_DIR", "hnav/_cache") / "emb"
    outdir = REPO / env.get("HNAV_OUT_DIR", "hnav/_out")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(" M1 — geometry calibration  [STAGE-0 GATE]")
    print(f"   model={model}  device=cuda:{device}  dtype={dtype}  max_length={max_length}")
    print("=" * 74)

    data = json.load(open(DATA, encoding="utf-8"))
    rng = random.Random(args.seed)
    emb = Embedder(model_name=model, device=device, dtype=dtype, batch=batch,
                   cache=cache, max_length=max_length)

    results = []
    for item in data:
        name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0].replace("factconsolidation_", "")
        if not args.include_mh and name.startswith("mh_"):
            continue
        if args.subsets and name not in args.subsets:
            continue

        conflicts, controls, n_facts, n_parsed = build_pairs(item["context"], rng)
        if args.max_pairs:
            conflicts, controls = conflicts[: args.max_pairs], controls[: args.max_pairs]
        print(f"\n── {name}: {n_facts} facts, {n_parsed} parsed "
              f"({100*n_parsed/n_facts:.1f}%), {len(conflicts)} conflict pairs", flush=True)

        texts = []
        for c in conflicts:
            texts += [c["old_text"], c["new_text"], c["old_obj"], c["new_obj"]]
        for c in controls:
            texts += [c["old_text"], c["new_text"], c["old_obj"], c["new_obj"]]
        vecs = emb.encode(texts)

        n_c = len(conflicts)
        C = vecs[: 4 * n_c].reshape(n_c, 4, -1)
        K = vecs[4 * n_c:].reshape(len(controls), 4, -1)

        whole_c = np.einsum("ij,ij->i", C[:, 0], C[:, 1])
        diff_c = np.einsum("ij,ij->i", C[:, 2], C[:, 3])
        whole_k = np.einsum("ij,ij->i", K[:, 0], K[:, 1]) if len(controls) else np.array([])

        r = qr_residual(C[:, 1], C[:, 0])
        crit = int(((whole_c >= 0.80) & (diff_c <= 0.30)).sum())

        res = {
            "subset": name, "model": model, "dtype": dtype,
            "n_facts": n_facts, "parse_coverage": round(100 * n_parsed / n_facts, 2),
            "n_conflict_pairs": n_c, "n_control_pairs": len(controls),
            "whole_blob_sim": pcts(whole_c),
            "diff_sim": pcts(diff_c),
            "control_whole_blob_sim": pcts(whole_k),
            "qr_residual_new_vs_old": pcts(r),
            "separation_auc_conflict_vs_control": round(auc(whole_c, whole_k), 4),
            "critical_delta_n": crit,
            "critical_delta_pct": round(100 * crit / n_c, 2) if n_c else 0.0,
            "gate_median_whole_blob": round(float(np.percentile(whole_c, 50)), 4),
            "gate_pass": bool(np.percentile(whole_c, 50) >= 0.70),
        }
        results.append(res)

        print(f"   whole_blob_sim : mean={res['whole_blob_sim']['mean']:.3f} "
              f"p10={res['whole_blob_sim']['p10']:.3f} p50={res['whole_blob_sim']['p50']:.3f} "
              f"p90={res['whole_blob_sim']['p90']:.3f}")
        print(f"   diff_sim       : mean={res['diff_sim']['mean']:.3f} "
              f"p50={res['diff_sim']['p50']:.3f}")
        print(f"   control whole  : mean={res['control_whole_blob_sim'].get('mean', float('nan')):.3f}"
              f"   separation AUC={res['separation_auc_conflict_vs_control']:.3f}")
        print(f"   critical-delta : {crit}/{n_c} = {res['critical_delta_pct']:.1f}%")
        print(f"   GATE (p50 >= 0.70): {'PASS' if res['gate_pass'] else 'FAIL'}")

    out = outdir / "m1_geometry_calibration.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print(" M1 SUMMARY")
    print("=" * 74)
    print(f" {'subset':<10} {'pairs':>7} {'whole p50':>10} {'diff p50':>9} {'AUC':>6} {'crit%':>7}  gate")
    for r_ in results:
        print(f" {r_['subset']:<10} {r_['n_conflict_pairs']:>7} "
              f"{r_['whole_blob_sim']['p50']:>10.3f} {r_['diff_sim']['p50']:>9.3f} "
              f"{r_['separation_auc_conflict_vs_control']:>6.3f} "
              f"{r_['critical_delta_pct']:>6.1f}%  {'PASS' if r_['gate_pass'] else 'FAIL'}")
    print(f"\n wrote {out.relative_to(REPO)}")

    failed = [r_["subset"] for r_ in results if not r_["gate_pass"]]
    print("\n" + "=" * 74)
    if failed:
        print(" GATE S3 FIRED on: " + ", ".join(failed))
        print(" Median whole-blob similarity < 0.70 — conflict pairs are NOT")
        print(" near-duplicates in this embedding space. The geometry premise fails.")
        print(" STOP. Do not build the geometry modules. Report to a human.")
        print(" Read-side stale repair may still be viable; the framing changes.")
        print("=" * 74)
        return 2
    print(" GATE PASSED on all subsets. Geometry premise holds.")
    print(" Proceed to T2 (regex-vs-geometry grouping ablation) — reuses this cache,")
    print(" no new embedding compute.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
