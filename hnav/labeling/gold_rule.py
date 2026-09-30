"""Determine the ground-truth conflict-resolution rule and question-level headroom."""
import json, re, sys, pathlib
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conflict_analysis import parse, D  # reuse templates

import json as _j
d = _j.load(open(D, encoding="utf-8"))

for item in d:
    name = item["metadata"]["qa_pair_ids"][0].rsplit("_no", 1)[0]
    if not name.startswith("factconsolidation_sh"):
        continue  # single-hop: answer maps directly to one fact
    facts = re.findall(r"^\s*(\d+)\.\s+(.*)$", item["context"], re.M)
    groups = defaultdict(list)
    obj_index = defaultdict(list)  # object string -> [fact numbers]
    for num, txt in facts:
        p = parse(txt)
        if p:
            rel, subj, obj = p
            groups[(rel, subj)].append((int(num), obj))
            obj_index[obj.lower()].append(int(num))

    conflicts = {k: v for k, v in groups.items() if len({o for _, o in v}) > 1}

    stats = Counter()
    for q, ans in zip(item["questions"], item["answers"]):
        gold = {a.lower() for a in ans}
        # find which conflict group this question targets: the group whose
        # object set intersects gold and whose subject appears in the question
        hit = None
        for (rel, subj), vals in conflicts.items():
            if subj.lower() in q.lower() and any(o.lower() in gold for _, o in vals):
                hit = (rel, subj, vals)
                break
        if hit is None:
            # maybe targets a non-conflicted key
            nonconf = False
            for (rel, subj), vals in groups.items():
                if subj.lower() in q.lower() and any(o.lower() in gold for _, o in vals):
                    nonconf = True
                    break
            stats["q_on_unique_fact" if nonconf else "q_unmatched"] += 1
            continue
        rel, subj, vals = hit
        vals_sorted = sorted(vals)
        golds = [n for n, o in vals_sorted if o.lower() in gold]
        stats["q_on_conflicted_key"] += 1
        if golds and golds[0] == vals_sorted[-1][0]:
            stats["gold_is_LATEST"] += 1
        elif golds and golds[0] == vals_sorted[0][0]:
            stats["gold_is_EARLIEST"] += 1
        else:
            stats["gold_is_MIDDLE"] += 1

    print(f"\n{name}  (n_questions={len(item['questions'])})")
    for k, v in stats.most_common():
        print(f"   {k:<24} {v:>4}  ({100*v/len(item['questions']):.0f}%)")
