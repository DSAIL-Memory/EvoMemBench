"""Extract trajectory data for selected context_ids from memory vs nomemory JSONL files.

Usage:
    python extract_trajectories.py --memory-infer <path> --memory-graded <path> \
        --nomemory-infer <path> --nomemory-graded <path> \
        --context-ids <id1> <id2> ... --output <output.json>
"""

import argparse
import json
import sys


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_task_id(record):
    meta = record.get("metadata", {})
    return meta.get("task_id") or record.get("idx", "")


def get_context_id(record):
    meta = record.get("metadata", {})
    return meta.get("context_id", "")


def get_last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_system_message(messages):
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            return c[:3000] + ("\n...[truncated]..." if len(c) > 3000 else "")
    return ""


def truncate(text, max_chars=2000):
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total {len(text)} chars]..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-infer", required=True)
    parser.add_argument("--memory-graded", required=True)
    parser.add_argument("--nomemory-infer", required=True)
    parser.add_argument("--nomemory-graded", required=True)
    parser.add_argument("--context-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Support both full UUIDs and 8-char prefixes
    raw_ids = args.context_ids
    target_prefixes = {cid[:8].lower() for cid in raw_ids}
    target_ids = set(raw_ids)  # kept for exact match fallback

    def matches(cid):
        return cid in target_ids or cid[:8].lower() in target_prefixes

    print(f"Loading memory infer: {args.memory_infer}", file=sys.stderr)
    mem_infer = load_jsonl(args.memory_infer)
    print(f"Loading memory graded: {args.memory_graded}", file=sys.stderr)
    mem_graded = load_jsonl(args.memory_graded)
    print(f"Loading nomemory infer: {args.nomemory_infer}", file=sys.stderr)
    nom_infer = load_jsonl(args.nomemory_infer)
    print(f"Loading nomemory graded: {args.nomemory_graded}", file=sys.stderr)
    nom_graded = load_jsonl(args.nomemory_graded)

    # Build task_id → score maps
    mem_scores = {get_task_id(r): r.get("score", r.get("overall_score", None)) for r in mem_graded}
    nom_scores = {get_task_id(r): r.get("score", r.get("overall_score", None)) for r in nom_graded}

    # Build prefix→full_uuid mapping from actual data
    prefix_to_full = {}
    for r in mem_infer:
        cid = get_context_id(r)
        p = cid[:8].lower()
        if p in target_prefixes:
            prefix_to_full[p] = cid
    for r in nom_infer:
        cid = get_context_id(r)
        p = cid[:8].lower()
        if p in target_prefixes:
            prefix_to_full[p] = cid

    # Normalize target_ids to full UUIDs
    normalized_ids = set()
    for rid in raw_ids:
        p = rid[:8].lower()
        if p in prefix_to_full:
            normalized_ids.add(prefix_to_full[p])
        else:
            normalized_ids.add(rid)

    # Group by context_id (using full UUIDs)
    mem_by_ctx = {}
    for r in mem_infer:
        cid = get_context_id(r)
        if cid in normalized_ids:
            mem_by_ctx.setdefault(cid, []).append(r)

    nom_by_ctx = {}
    for r in nom_infer:
        cid = get_context_id(r)
        if cid in normalized_ids:
            nom_by_ctx.setdefault(cid, []).append(r)

    # Build prefix→full_uuid for output keying
    prefix_to_full_final = {}
    for full_cid in normalized_ids:
        p = full_cid[:8].lower()
        prefix_to_full_final[p] = full_cid

    results = {}
    for raw_cid in args.context_ids:
        p = raw_cid[:8].lower()
        cid = prefix_to_full_final.get(p, raw_cid)  # resolve to full UUID
        mem_tasks = mem_by_ctx.get(cid, [])
        nom_tasks = nom_by_ctx.get(cid, [])

        # Build task_id → nomemory record map for matching
        nom_by_task = {get_task_id(r): r for r in nom_tasks}

        tasks_out = []
        for r in mem_tasks:
            tid = get_task_id(r)
            messages = r.get("messages", [])
            nom_r = nom_by_task.get(tid, {})
            nom_msgs = nom_r.get("messages", [])

            task_info = {
                "task_id": tid,
                "metadata": r.get("metadata", {}),
                "system_context": get_system_message(messages),
                "user_question": get_last_user_message(messages),
                "rubrics": r.get("rubrics", []),
                "memory_retrieved": truncate(r.get("memory_retrieved", ""), 3000),
                "memory_output": truncate(r.get("model_output", ""), 2000),
                "memory_score": mem_scores.get(tid),
                "nomemory_output": truncate(nom_r.get("model_output", ""), 2000),
                "nomemory_score": nom_scores.get(tid),
            }
            tasks_out.append(task_info)

        # Compute context-level scores
        mem_ctx_score = sum(1 for t in tasks_out if t["memory_score"] == 1)
        nom_ctx_score = sum(1 for t in tasks_out if t["nomemory_score"] == 1)
        n = len(tasks_out)

        delta = round((mem_ctx_score - nom_ctx_score) / n, 3) if n else 0
        results[raw_cid] = {
            "context_id": cid,
            "n_tasks": n,
            "memory_solved": mem_ctx_score,
            "nomemory_solved": nom_ctx_score,
            "memory_rate": round(mem_ctx_score / n, 3) if n else 0,
            "nomemory_rate": round(nom_ctx_score / n, 3) if n else 0,
            "delta": delta,
            "tasks": tasks_out,
        }
        print(f"  {cid[:8]}: n={n}, nomem={nom_ctx_score}/{n}, mem={mem_ctx_score}/{n}, delta={delta:+.1%}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
