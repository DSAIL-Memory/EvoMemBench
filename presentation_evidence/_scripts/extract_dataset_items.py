# Extraction for items 1, 5 (paired examples), 8, 13, 14 — everything that needs
# the dataset file or the confirmatory per-question records.
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "presentation_evidence", "data")
sys.path.insert(0, ROOT)

from hnav.labeling.conflict_analysis import parse  # noqa: E402  validated parser
from hnav.stage1.stale_suppression_probe import (  # noqa: E402
    split_context, render_context, build_prompt)

DATASET = os.path.join(ROOT, "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json")
FACT_RE_LINE = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)

def load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)

def save_json(name, obj):
    with open(os.path.join(DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

with open(DATASET, encoding="utf-8") as f:
    dataset = json.load(f)

def entry_for(subset):
    for i, item in enumerate(dataset):
        if item["metadata"]["qa_pair_ids"][0].startswith(subset):
            return i, item
    raise KeyError(subset)

print("#" * 30, "ITEM 1 — the Nobuhiro exhibit + 2 more examples")
i6, e6 = entry_for("factconsolidation_sh_6k")
facts6 = {int(n): t.strip() for n, t in FACT_RE_LINE.findall(e6["context"])}
print("dataset entry index:", i6, "| qa_pair_ids[0]:", e6["metadata"]["qa_pair_ids"][0])
print("serial 91 :", facts6[91])
print("serial 259:", facts6[259])
print("questions[1]:", e6["questions"][1])
print("answers[1]:", e6["answers"][1])

probe6 = load("stage0_results/stage1/stale_suppression_probe_sh6k.json")
pq6 = probe6["results"][0]["per_question"]
q1 = next(q for q in pq6 if q["index"] == 1)
print("per_question[index=1]: native =", q1["arms"]["native"]["output"],
      "| oracle_suppress =", q1["arms"]["oracle_suppress"]["output"],
      "| plan:", q1["plan"])

# candidate extra examples: conflicted, native wrong, oracle_suppress right,
# exactly one gold and one stale serial (clean two-fact story)
cands = []
for q in pq6:
    if q["stratum"] != "conflicted":
        continue
    a = q["arms"]
    if a["native"]["correct"] or not a["oracle_suppress"]["correct"]:
        continue
    p = q["plan"]
    if len(p["gold_serials"]) == 1 and len(p["stale_serials"]) == 1:
        g, s = p["gold_serials"][0], p["stale_serials"][0]
        cands.append({
            "index": q["index"], "subject": q["key"][1], "relation": q["key"][0],
            "question": e6["questions"][q["index"]], "gold_answer": e6["answers"][q["index"]],
            "old_serial": s, "old_text": facts6[s],
            "new_serial": g, "new_text": facts6[g],
            "native_output": a["native"]["output"],
            "suppressed_output": a["oracle_suppress"]["output"],
        })
print(f"\n{len(cands)} clean candidates (conflicted, native wrong, suppress right, 1 gold + 1 stale):")
for c in cands[:20]:
    print(f"  idx {c['index']:>2}  subj={c['subject']!r:40} old#{c['old_serial']} new#{c['new_serial']}"
          f"  native={c['native_output']!r} suppressed={c['suppressed_output']!r}")
save_json("item01_examples.json", {
    "dataset_file": "In-Episode-Knowledge/INEP-KNOW/MemoryAgentBench/data/Conflict_Resolution.json",
    "dataset_entry_index": i6, "qa_pair_id_0": e6["metadata"]["qa_pair_ids"][0],
    "nobuhiro": {
        "index": 1, "question": e6["questions"][1], "gold_answer": e6["answers"][1],
        "serial_91": facts6[91], "serial_259": facts6[259],
        "native_output": q1["arms"]["native"]["output"],
        "oracle_suppress_output": q1["arms"]["oracle_suppress"]["output"],
        "plan": q1["plan"],
    },
    "all_clean_candidates": cands,
})

# byte-exact page reproduction check + excerpt
preamble, fact_list = split_context(e6["context"])
rendered = render_context(preamble, fact_list)
byte_exact = rendered == e6["context"]
print("\nrender_context reproduces dataset context byte-exactly:", byte_exact)
prompt = build_prompt(rendered, e6["questions"][1])
lines = rendered.split("\n")
def around(serial, k=3):
    idx = next(i for i, ln in enumerate(lines) if ln.startswith(f"{serial}. "))
    return lines[max(0, idx - k): idx + k + 1]
excerpt = []
excerpt.append("REPRODUCED PAGE EXCERPT - factconsolidation_sh_6k, question index 1")
excerpt.append("Reproduced offline with hnav/stage1/stale_suppression_probe.py::render_context;")
excerpt.append(f"byte-identical to the dataset 'context' field: {byte_exact}")
excerpt.append(f"full prompt: {len(prompt)} chars; context: {len(rendered)} chars, {len(fact_list)} facts")
excerpt.append("=" * 78)
excerpt.append("--- prompt head (first 12 lines of the full prompt) ---")
excerpt.extend(prompt.split("\n")[:12])
excerpt.append("...")
excerpt.append("--- around serial 91 (the stale fact) ---")
excerpt.extend(around(91))
excerpt.append("...")
excerpt.append("--- around serial 259 (the superseding fact) ---")
excerpt.extend(around(259))
excerpt.append("...")
excerpt.append("--- prompt tail (last 6 lines: template + the question) ---")
excerpt.extend(prompt.split("\n")[-6:])
with open(os.path.join(DATA_DIR, "item01_page_excerpt.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(excerpt) + "\n")
print("wrote item01_page_excerpt.txt")

print("#" * 30, "ITEM 5 — paired examples where native and oracle_recency disagree")
pairs5 = []
for q in pq6:
    if q["stratum"] != "conflicted":
        continue
    a = q["arms"]
    if a["native"]["output"] != a["oracle_recency"]["output"]:
        pairs5.append({
            "subset": "sh_6k", "index": q["index"], "subject": q["key"][1],
            "question": e6["questions"][q["index"]], "gold": e6["answers"][q["index"]],
            "plan": q["plan"],
            "answers_by_arm": {arm: {"output": a[arm]["output"], "correct": a[arm]["correct"]}
                               for arm in a},
        })
print(f"sh_6k: {len(pairs5)} conflicted questions where oracle_recency changed the answer")
for p in pairs5[:12]:
    ab = p["answers_by_arm"]
    print(f"  idx {p['index']:>2} {p['subject']!r:38} native={ab['native']['output']!r}"
          f" recency={ab['oracle_recency']['output']!r} gold={p['gold']!r}")
save_json("item05_paired_examples.json", pairs5)

print("#" * 30, "ITEM 8 — parser demo (run for real)")
demo = parse("Nobuhiro Watsuki is famous for Rurouni Kenshin.")
print('parse("Nobuhiro Watsuki is famous for Rurouni Kenshin.") ->', demo)
save_json("item08_parser.json", {
    "function": "hnav/labeling/conflict_analysis.py::parse, lines 53-68",
    "demo_input": "Nobuhiro Watsuki is famous for Rurouni Kenshin.",
    "demo_output": {"relation_key": demo[0], "subject": demo[1], "object": demo[2]},
    "parse_coverage_m1_pct": {e["subset"]: e["parse_coverage"]
                              for e in load("stage0_results/final/m1_geometry_calibration.json")},
    "parse_coverage_m1b_pct": {e["subset"]: e["parse_coverage_pct"]
                               for e in load("stage0_results/final/m1b_grouping_ablation.json")},
})

print("#" * 30, "ITEM 13 — confirmatory sh_64k")
conf = load("stage0_results/stage1/detector_gap_confirmatory_sh64k.json")
r0 = conf["results"][0]
arms13 = {}
for arm, v in r0["arms"].items():
    arms13[arm] = {"overall": v}
for stratum in ("unique", "conflicted"):
    for arm, v in r0["by_stratum"][stratum]["arms"].items():
        arms13[arm][stratum] = v
# recompute overall = unique + conflicted
for arm in arms13:
    u, c = arms13[arm].get("unique"), arms13[arm].get("conflicted")
    o = arms13[arm]["overall"]
    if u and c:
        assert u["correct"] + c["correct"] == o["correct"], arm
        assert u["n"] + c["n"] == o["n"], arm
print("overall arms:", {k: f"{v['overall']['correct']}/{v['overall']['n']}" for k, v in arms13.items()})
print("unique arms:", {k: f"{v['unique']['correct']}/{v['unique']['n']}" for k, v in arms13.items()})
mc = r0["by_stratum"]["conflicted"]["paired_vs_native"]["detector_suppress"]
print("McNemar suppress (conflicted):", mc)
print("tokens delta_pct suppress:", r0["tokens"]["detector_suppress"]["delta_pct"])
save_json("item13_summary.json", {
    "file": "stage0_results/stage1/detector_gap_confirmatory_sh64k.json",
    "preregistration": conf["preregistration"],
    "arms_by_stratum": arms13,
    "mcnemar_conflicted": r0["by_stratum"]["conflicted"]["paired_vs_native"],
    "tokens": r0["tokens"],
    "harm": {k: {kk: vv for kk, vv in v.items() if kk != "harms"} | {"harms": v["harms"]}
             for k, v in r0["harm"].items()},
    "void_conditions": r0["void_conditions"],
    "corrections": conf["corrections"],
})
rows13 = []
arm_names = list(r0["arms"].keys())
for q in r0["per_question"]:
    row = [q["index"], q["stratum"], q["key"][0], q["key"][1], "; ".join(q["truths"])]
    for arm in arm_names:
        row += [q["arms"][arm]["output"], q["arms"][arm]["correct"]]
    rows13.append(row)
hdr = ["index", "stratum", "relation", "subject", "truths"]
for arm in arm_names:
    hdr += [f"{arm}_output", f"{arm}_correct"]
with open(os.path.join(DATA_DIR, "item13_per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows13)
print(f"wrote item13_per_question.csv ({len(rows13)} rows)")

print("#" * 30, "ITEM 14 — the 735 deletions, independently re-verified")
i64, e64 = entry_for("factconsolidation_sh_64k")
facts64 = {int(n): t.strip() for n, t in FACT_RE_LINE.findall(e64["context"])}
# key index over the FULL context via the validated parser
key_members = {}
unparsed = 0
for s, t in facts64.items():
    p = parse(t)
    if p is None:
        unparsed += 1
        continue
    key_members.setdefault((p[0], p[1]), []).append((s, p[2]))
print(f"sh_64k context: {len(facts64)} facts, {unparsed} unparsed by the validated parser")

rows14 = []
n_total = 0
n_not_latest = 0
n_is_latest = 0
n_unparsed_supp = 0
n_value_is_queried_gold = 0
for q in r0["per_question"]:
    truths = [t.lower() for t in q["truths"]]
    for s in q["plan"]["suppress_serials"]:
        n_total += 1
        text = facts64.get(s, "")
        p = parse(text) if text else None
        if p is None:
            n_unparsed_supp += 1
            rows14.append([q["index"], s, text, "", "", "", "", "PARSE_FAIL"])
            continue
        members = key_members[(p[0], p[1])]
        latest_serial = max(m[0] for m in members)
        latest_value = next(v for m, v in members if m == latest_serial)
        is_latest = (s == latest_serial)
        n_is_latest += is_latest
        n_not_latest += (not is_latest)
        gold_flag = p[2].lower() in truths and (p[0], p[1]) == (q["key"][0], q["key"][1])
        n_value_is_queried_gold += gold_flag
        rows14.append([q["index"], s, text, p[2], latest_serial, latest_value,
                       is_latest, "GOLD_VALUE_OF_QUERIED_KEY" if gold_flag else ""])
print(f"total suppressed (multiplicity over 100 question-pages): {n_total}")
print(f"  not the key's latest serial (superseded): {n_not_latest}")
print(f"  IS the key's latest serial: {n_is_latest}")
print(f"  parse failures: {n_unparsed_supp}")
print(f"  suppressed fact carried the QUERIED question's gold value: {n_value_is_queried_gold}")
vc4 = r0["void_conditions"]["4_no_harmful_suppression"]["observed"]
print("artifact VC4 observed:", vc4)
pc = r0["void_conditions"]["8_guards_and_positive_control"]["observed"]["positive_control"]
print("positive control n_facts_suppressed:", pc["n_facts_suppressed"],
      "| n_fact_edits_applied:", pc["n_fact_edits_applied"])
with open(os.path.join(DATA_DIR, "item14_deletions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question_index", "suppressed_serial", "fact_text", "fact_value",
                "key_latest_serial", "key_latest_value", "suppressed_is_key_latest", "flag"])
    w.writerows(rows14)
print(f"wrote item14_deletions.csv ({len(rows14)} rows)")
save_json("item14_summary.json", {
    "artifact_vc4": vc4,
    "recomputed": {"n_suppressed_total": n_total, "n_not_key_latest": n_not_latest,
                   "n_is_key_latest": n_is_latest, "n_parse_fail": n_unparsed_supp,
                   "n_carrying_queried_gold_value": n_value_is_queried_gold},
    "method": "sum of len(per_question[i].plan.suppress_serials); each serial joined to the "
              "full sh_64k context, keyed by the validated parser, and compared to the key's "
              "highest serial in the full context",
    "positive_control": pc,
    "gold_cut_correction": conf["corrections"][0]["items"][2],
})
print("DONE")
