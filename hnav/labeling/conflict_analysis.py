"""Rigorous conflict-structure analysis of MemoryAgentBench Conflict_Resolution (factconsolidation).

Groups facts by (relation, subject) using induced templates, then reports:
  - template coverage
  - conflict-group counts / sizes
  - how many QUESTIONS target a conflicted (relation, subject)
  - whether ground-truth answers follow a "latest wins" rule
"""
import json, re, sys, pathlib
from collections import Counter, defaultdict

D = str(pathlib.Path(__file__).resolve().parents[2] /
        "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json")
d = json.load(open(D, encoding="utf-8"))

# ── relation templates ───────────────────────────────────────────────────────
# "{s} MID {o}." family
SUFFIX_RELS = [
    " is affiliated with the religion of ", " is associated with the sport of ",
    " was created in the country of ", " was written in the language of ",
    " is located in the continent of ", " was born in the city of ",
    " was founded in the city of ", " speaks the language of ",
    " died in the city of ", " worked in the city of ", " plays the position of ",
    " works in the field of ", " is a citizen of ", " was performed by ",
    " was developed by ", " was founded by ", " was created by ",
    " is employed by ", " is married to ", " is famous for ", "'s child is ",
]
# "PRE {s} MID {o}." family
PREFIX_RELS = [
    ("The name of the current head of the ", " government is "),
    ("The name of the current head of state in ", " is "),
    ("The headquarters of ", " is located in the city of "),
    ("The univeristy where ", " was educated is "),
    ("The type of music that ", " plays is "),
    ("The chief executive officer of ", " is "),
    ("The origianl broadcaster of ", " is "),
    ("The company that produced ", " is "),
    ("The original language of ", " is "),
    ("The official language of ", " is "),
    ("The chairperson of ", " is "),
    ("The head coach of ", " is "),
    ("The Governor of ", " is "),
    ("The director of ", " is "),
    ("The capital of ", " is "),
    ("The author of ", " is "),
    ("The mother of ", " is "),
    ("The father of ", " is "),
]
SUFFIX_RELS.sort(key=len, reverse=True)
PREFIX_RELS.sort(key=lambda x: -(len(x[0]) + len(x[1])))


def parse(fact):
    """-> (relation_key, subject, object) or None."""
    f = fact.rstrip()
    if f.endswith("."):
        f = f[:-1]
    for pre, mid in PREFIX_RELS:
        if f.startswith(pre):
            rest = f[len(pre):]
            i = rest.find(mid)
            if i > 0:
                return (pre + "|" + mid, rest[:i], rest[i + len(mid):])
    for mid in SUFFIX_RELS:
        i = f.find(mid)
        if i > 0:
            return ("|" + mid, f[:i], f[i + len(mid):])
    return None


def analyze(item):
    facts = re.findall(r"^\s*(\d+)\.\s+(.*)$", item["context"], re.M)
    parsed, unparsed = [], []
    for num, txt in facts:
        p = parse(txt)
        (parsed.append((int(num), txt) + p) if p else unparsed.append(txt))

    groups = defaultdict(list)  # (rel, subj) -> [(num, txt, obj)]
    for num, txt, rel, subj, obj in parsed:
        groups[(rel, subj)].append((num, txt, obj))

    conflicts = {k: v for k, v in groups.items() if len({o for _, _, o in v}) > 1}
    n_conf_facts = sum(len(v) for v in conflicts.values())
    return facts, parsed, unparsed, groups, conflicts, n_conf_facts


def main() -> None:
    print("=" * 78)
    print("FACT-STORE CONFLICT STRUCTURE (Conflict_Resolution / factconsolidation)")
    print("=" * 78)
    rows = []
    for item in d:
        name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
        facts, parsed, unparsed, groups, conflicts, n_conf = analyze(item)
        cov = 100 * len(parsed) / len(facts)
        sizes = Counter(len(v) for v in conflicts.values())
        rows.append((name, len(facts), cov, len(groups), len(conflicts), n_conf,
                     100 * n_conf / len(facts), dict(sorted(sizes.items()))))
        print(f"\n{name}")
        print(f"  facts={len(facts):>6}  template_coverage={cov:5.1f}%  unparsed={len(unparsed)}")
        print(f"  distinct (relation,subject) keys : {len(groups)}")
        print(f"  CONFLICTED keys (>1 distinct obj): {len(conflicts)} "
              f"({100*len(conflicts)/len(groups):.1f}% of keys)")
        print(f"  facts inside conflicted keys     : {n_conf} ({100*n_conf/len(facts):.1f}% of facts)")
        print(f"  conflict group-size histogram    : {dict(sorted(sizes.items()))}")
        if name == "factconsolidation_mh_6k":
            print("  unparsed sample:", unparsed[:5])
            print("  example conflicts:")
            for k, v in list(conflicts.items())[:5]:
                print(f"    key={k[1]!r} rel={k[0]!r}")
                for num, txt, obj in sorted(v):
                    print(f"       #{num:>5}  obj={obj}")


if __name__ == "__main__":
    main()
