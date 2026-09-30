"""Whole-blob vs marginal-diff separation on real EvoMemBench conflict pairs.

Uses a LEXICAL proxy (char-3gram cosine + token Jaccard) for whole-blob similarity.
This is a proxy for embedding cosine, not a substitute: it establishes that the two
facts are near-identical in surface form and differ only in the answer-bearing span.
"""
import json, re, sys, pathlib
from collections import Counter, defaultdict
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conflict_analysis import parse, D

d = json.load(open(D, encoding="utf-8"))


def ngrams(s, n=3):
    s = s.lower()
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def cos(a, b):
    common = set(a) & set(b)
    num = sum(a[g] * b[g] for g in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def jac(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / len(A | B) if A | B else 0.0


for item in d:
    name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
    if name != "factconsolidation_sh_262k":
        continue
    facts = re.findall(r"^\s*(\d+)\.\s+(.*)$", item["context"], re.M)
    groups = defaultdict(list)
    for num, txt in facts:
        p = parse(txt)
        if p:
            groups[(p[0], p[1])].append((int(num), txt, p[2]))
    conflicts = [v for v in groups.values() if len({o for _, _, o in v}) > 1]

    whole, diffsim, objjac = [], [], []
    for v in conflicts:
        v = sorted(v)
        (n1, t1, o1), (n2, t2, o2) = v[0], v[-1]
        whole.append(cos(ngrams(t1), ngrams(t2)))
        diffsim.append(cos(ngrams(o1), ngrams(o2)))
        objjac.append(jac(o1, o2))

    def pct(xs, p):
        xs = sorted(xs)
        return xs[int(p * (len(xs) - 1))]

    print(f"=== {name}: {len(conflicts)} conflict pairs (old fact vs superseding fact)")
    print(f"  WHOLE-BLOB similarity (char-3gram cosine of the two full facts)")
    print(f"     mean={sum(whole)/len(whole):.3f}  p10={pct(whole,.10):.3f} "
          f"p50={pct(whole,.50):.3f}  p90={pct(whole,.90):.3f}")
    print(f"  MARGINAL-DIFF similarity (same metric on the differing OBJECT span only)")
    print(f"     mean={sum(diffsim)/len(diffsim):.3f}  p10={pct(diffsim,.10):.3f} "
          f"p50={pct(diffsim,.50):.3f}  p90={pct(diffsim,.90):.3f}")
    print(f"  object token Jaccard: mean={sum(objjac)/len(objjac):.3f} "
          f"(fraction of pairs with ZERO token overlap: "
          f"{100*sum(1 for x in objjac if x==0)/len(objjac):.1f}%)")
    hi = sum(1 for w, ds in zip(whole, diffsim) if w >= 0.80 and ds <= 0.30)
    print(f"  CRITICAL-DELTA cases (whole>=0.80 AND diff<=0.30): {hi} "
          f"= {100*hi/len(conflicts):.1f}% of conflict pairs")
    print()
    print("  worked examples (highest whole-blob sim, lowest diff sim):")
    ranked = sorted(zip(whole, diffsim, conflicts), key=lambda x: (-x[0], x[1]))
    for w, ds, v in ranked[:4]:
        v = sorted(v)
        print(f"    whole={w:.3f} diff={ds:.3f}")
        print(f"       old #{v[0][0]}: {v[0][1]}")
        print(f"       new #{v[-1][0]}: {v[-1][1]}")
