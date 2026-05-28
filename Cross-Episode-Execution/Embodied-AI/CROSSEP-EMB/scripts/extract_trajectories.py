"""
Extract ALFWorld trajectory JSON files into readable Markdown format.
Highlights memory injection sections for verification.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output/alfworld_mem0_20260425_233108/readable"

TARGETS = [
    (
        "output/alfworld_mem0_20260425_233108/in_env/pick_and_place_simple/alfworld_2423.json",
        "pick_and_place_simple",
        "in_env",
        "alfworld_2423_no_memory.md",
    ),
    (
        "output/alfworld_mem0_20260425_233108/in_env/pick_and_place_simple/alfworld_2618.json",
        "pick_and_place_simple",
        "in_env",
        "alfworld_2618_with_memory.md",
    ),
    (
        "output/alfworld_mem0_20260425_233108/cross_env/pick_and_place_simple__to__pick_two_obj_and_place/alfworld_2421.json",
        "pick_and_place_simple → pick_two_obj_and_place",
        "cross_env",
        "alfworld_2421_cross_env.md",
    ),
]


def parse_memories(content: str) -> list:
    m = re.search(r"--- Retrieved Memories ---\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return []
    return [line.lstrip("- ").strip() for line in m.group(1).strip().splitlines() if line.strip()]


def strip_memories_from_prompt(content: str) -> str:
    return re.sub(r"\n\n--- Retrieved Memories ---\n.*?\n---", "", content, flags=re.DOTALL).strip()


def format_trajectory(path: Path, task_type: str, env_type: str) -> str:
    data = json.loads(path.read_text())

    item_id = data["item_id"]
    success = data["success"]
    reward = data["reward"]
    num_rounds = data["num_rounds"]
    prompt_tokens = data["prompt_tokens"]
    completion_tokens = data["completion_tokens"]
    total_tokens = data["total_tokens"]
    inject_emb = data["inject_embedding_tokens"]
    inject_in = data["inject_input_tokens"]
    inject_out = data["inject_output_tokens"]
    inject_lat = data["inject_latency"]
    update_in = data["update_input_tokens"]
    update_out = data["update_output_tokens"]
    update_emb = data["update_embedding_tokens"]
    update_lat = data["update_latency"]

    success_icon = "✅" if success else "❌"
    conversations = data["conversations"]

    memories = []
    if conversations:
        memories = parse_memories(conversations[0]["content"])

    lines = []
    lines.append(f"# Trajectory: {item_id} — {task_type} ({env_type})\n")

    # --- Metadata ---
    lines.append("## Metadata\n")
    lines.append("| 字段 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| success | {success_icon} {success} (reward={reward}) |")
    lines.append(f"| num_rounds | {num_rounds} |")
    lines.append(f"| prompt_tokens / completion_tokens | {prompt_tokens} / {completion_tokens} (total {total_tokens}) |")
    lines.append(f"| inject_embedding_tokens | {inject_emb} |")
    lines.append(f"| inject_input_tokens / inject_output_tokens | {inject_in} / {inject_out} |")
    lines.append(f"| inject_latency | {inject_lat:.3f}s |")
    lines.append(f"| update_input_tokens / update_output_tokens | {update_in} / {update_out} |")
    lines.append(f"| update_embedding_tokens | {update_emb} |")
    lines.append(f"| update_latency | {update_lat:.3f}s |")
    lines.append("")

    # --- Retrieved Memories ---
    if memories:
        lines.append(f"## Retrieved Memories ({len(memories)} items)\n")
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. {mem}")
        lines.append("")
    else:
        lines.append("## Retrieved Memories\n")
        lines.append("> **(none)** — 数据库为空或未检索到相关记忆（inject_embedding_tokens 反映已发起查询但无结果）\n")

    # --- Conversation ---
    lines.append(f"## Conversation ({num_rounds} rounds)\n")

    for idx, turn in enumerate(conversations):
        role = turn["role"]
        content = turn["content"]

        if idx == 0 and role == "user":
            # System prompt: strip memory block (already shown above), truncate if long
            clean = strip_memories_from_prompt(content)
            lines.append(f"### Turn {idx} — **{role}** (系统指令)\n")
            lines.append("```")
            lines.append(clean[:800] + ("..." if len(clean) > 800 else ""))
            lines.append("```\n")
        else:
            lines.append(f"### Turn {idx} — **{role}**\n")
            lines.append("```")
            lines.append(content)
            lines.append("```\n")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, task_type, env_type, out_name in TARGETS:
        src = BASE_DIR / rel_path
        if not src.exists():
            print(f"[WARN] not found: {src}")
            continue
        md = format_trajectory(src, task_type, env_type)
        out_path = OUTPUT_DIR / out_name
        out_path.write_text(md)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
