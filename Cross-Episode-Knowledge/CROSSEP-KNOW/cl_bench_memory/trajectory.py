"""Format messages for memory extraction and inject retrieved memory into chat."""

import copy


def format_trajectory(messages, model_output):
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"**[System Context]**\n{content}")
        elif role == "user":
            parts.append(f"**[User Question]**\n{content}")
        elif role == "assistant":
            parts.append(f"**[Assistant Response]**\n{content}")
    if model_output:
        parts.append(f"**[Model Output]**\n{model_output}")
    return "\n\n".join(parts)


def get_last_user_message(messages):
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def inject_memory_into_messages(messages, memory_text):
    if not memory_text:
        return messages
    augmented = copy.deepcopy(messages)
    for i, msg in enumerate(augmented):
        if msg["role"] == "system":
            augmented[i]["content"] += memory_text
            return augmented
    augmented.insert(0, {"role": "system", "content": memory_text.strip()})
    return augmented
