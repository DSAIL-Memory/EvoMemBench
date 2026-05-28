import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Context-budget / in-episode memory helpers
# ---------------------------------------------------------------------------

_TIKTOKEN_ENC = None


def _get_tiktoken_enc():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken as _tiktoken
            _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
    return _TIKTOKEN_ENC


def _count_tokens(obj) -> int:
    """Rough token count for any JSON-serialisable object (falls back to chars/4)."""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = str(obj)
    enc = _get_tiktoken_enc()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _extract_user_text_from_messages(messages: list) -> str:
    """Return the text of the last user-role message in the list."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(parts)
        return str(content)
    return ""


def _append_memory_to_last_user_message(messages: list, snippet: str) -> None:
    """Append memory snippet in-place to the last user-role message."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if isinstance(content, str):
            msg["content"] = content + snippet
        elif isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = (part.get("text") or "") + snippet
                    return
            content.append({"type": "text", "text": snippet})
        return

from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.utils import *
from overrides import final

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )
    from bfcl_eval.memory.base import Memory

_COMPRESSION_MEMORY_TYPES = frozenset({"memagent", "memobrain"})


class BaseHandler:
    model_name: str
    is_fc_model: bool
    registry_name: str
    temperature: float
    registry_dir_name: str
    model_name_underline_replaced: str
    model_style: ModelStyle

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, **kwargs
    ) -> None:
        """
        Args:
            model_name: The name of the model as used in the vendor API or on Hugging Face.
            temperature: The temperature of the model.
            registry_name: The name of the model as used internally in BFCL, used for result directory naming.
            is_fc_model: Whether the model is a function calling model.
            **kwargs: Additional attributes passed via kwargs.
        """
        self.model_name = model_name
        self.is_fc_model = is_fc_model
        self.registry_name = registry_name

        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        # The directory name for the model
        # Replace the slash with underscore to avoid creating subdirectories
        self.registry_dir_name = registry_name.replace("/", "_")
        self.temperature = temperature

        # Set any additional attributes passed via kwargs
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @staticmethod
    def _truncate_messages_to_budget(
        messages: list[dict], tools, budget: int
    ) -> list[dict]:
        """Return a (possibly shortened) copy of messages that fits within budget tokens.

        In prompting mode messages[0] is a system message and is always pinned.
        In FC mode there is no system message (function docs go in tools), so
        n_pin == 0.

        The last user message is always preserved.  In FC mode at step k≥1, only
        the *current step* (the last assistant tool-call message + its tool results)
        is pinned alongside the user message.  Earlier intra-turn steps — tool-call/
        result pairs from steps 0..k-2 within the same turn — are treated as
        droppable history and removed in whole-step chunks *after* exhausting
        cross-turn drops.

        Drop order (oldest first):
          1. Cross-turn history  (messages before the last user message).
          2. Intra-turn steps    (assistant + tool-result pairs from earlier steps
                                  within the current turn), one complete step at a
                                  time to preserve tool_call_id pairing invariants.

        The effective budget is: budget - _RESPONSE_MARGIN - tool_tokens.
        The original list is never mutated.
        """
        if not messages or budget <= 0:
            return list(messages)
        _RESPONSE_MARGIN = 512
        tool_tokens = _count_tokens(tools) if tools else 0
        effective = budget - _RESPONSE_MARGIN - tool_tokens
        if effective <= 0:
            return list(messages)

        if _count_tokens(messages) <= effective:
            return list(messages)

        # Pin the system message in prompting mode; pin nothing in FC mode.
        n_pin = 1 if messages[0].get("role") == "system" else 0

        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if last_user_idx is None or last_user_idx < n_pin:
            return list(messages)

        head = messages[:n_pin]

        # In FC mode at step k≥1 there are assistant+tool messages after the last
        # user message.  Find the last assistant message — it starts the "current
        # step" that must be preserved.
        last_asst_after_user = next(
            (i for i in range(len(messages) - 1, last_user_idx, -1)
             if messages[i].get("role") == "assistant"),
            None,
        )

        if last_asst_after_user is not None:
            # ── FC mode, step k ≥ 1 ─────────────────────────────────────────────
            # Four segments (only pre_turn and intra_turn are droppable):
            #   head | pre_turn | [user_msg] | intra_turn | current_step
            pre_turn     = list(messages[n_pin:last_user_idx])
            user_msg     = [messages[last_user_idx]]
            intra_turn   = list(messages[last_user_idx + 1:last_asst_after_user])
            current_step = list(messages[last_asst_after_user:])

            def _total():
                return _count_tokens(head + pre_turn + user_msg + intra_turn + current_step)

            # 1. Drop cross-turn history in whole-turn chunks (up to the next user-message
            #    boundary) so the remaining sequence always starts at a valid user boundary.
            #    Dropping one message at a time can leave orphaned tool/assistant messages
            #    at the front, producing invalid role sequences the Ark FC API rejects.
            _pre_turn_orig = len(pre_turn)
            while pre_turn and _total() > effective:
                next_user = next(
                    (i for i in range(1, len(pre_turn)) if pre_turn[i].get("role") == "user"),
                    len(pre_turn),
                )
                del pre_turn[:next_user]
            if len(pre_turn) < _pre_turn_orig:
                _n = _pre_turn_orig - len(pre_turn)
                _marker = {"role": "user", "content": f"[CONTEXT TRUNCATED: {_n} message(s) removed from conversation history]"}
                pre_turn.insert(0, _marker)

            # 2. Drop intra-turn steps in whole-step chunks (asst + its tool results)
            #    to keep tool_call_id pairing valid.
            _intra_orig = len(intra_turn)
            while intra_turn and _total() > effective:
                drop_end = 1
                while drop_end < len(intra_turn) and intra_turn[drop_end].get("role") == "tool":
                    drop_end += 1
                intra_turn = intra_turn[drop_end:]
            if len(intra_turn) < _intra_orig:
                _n = _intra_orig - len(intra_turn)
                _marker = {"role": "user", "content": f"[CONTEXT TRUNCATED: {_n} intra-turn message(s) from earlier steps removed]"}
                intra_turn.insert(0, _marker)

            # 3. Trim tool result content in current_step from the front (keep suffix)
            #    if still over budget after exhausting all droppable history.
            if _total() > effective:
                for i, msg in enumerate(current_step):
                    if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                        content = msg["content"]
                        _orig_len = len(content)
                        while content and _total() > effective:
                            trim = max(1, len(content) // 5)
                            content = content[trim:]
                            current_step[i] = {**msg, "content": content}
                        if len(content) < _orig_len:
                            _label = f"[CONTEXT TRUNCATED: tool result trimmed from {_orig_len} to {len(content)} chars, leading content omitted]\n"
                            current_step[i] = {**current_step[i], "content": _label + content}
                        if _total() <= effective:
                            break

            result = head + pre_turn + user_msg + intra_turn + current_step

        else:
            # ── Step 0 or prompting mode ─────────────────────────────────────────
            tail   = messages[last_user_idx:]
            middle = list(messages[n_pin:last_user_idx])

            _middle_orig = len(middle)
            while middle and _count_tokens(head + middle + tail) > effective:
                next_user = next(
                    (i for i in range(1, len(middle)) if middle[i].get("role") == "user"),
                    len(middle),
                )
                del middle[:next_user]
            if len(middle) < _middle_orig:
                _n = _middle_orig - len(middle)
                _marker = {"role": "user", "content": f"[CONTEXT TRUNCATED: {_n} message(s) removed from conversation history]"}
                if middle:
                    middle.insert(0, _marker)
                else:
                    tail = [_marker] + list(tail)

            result = head + middle + tail

        # Merge any adjacent user messages that appear at the join point after
        # dropping cross-turn history (can only happen in FC mode where n_pin == 0).
        merged: list[dict] = []
        for msg in result:
            if (merged and merged[-1].get("role") == "user"
                    and msg.get("role") == "user"):
                prev = merged[-1]
                sep = "\n\n" if prev.get("content") else ""
                merged[-1] = {**prev, "content": (prev.get("content") or "") + sep + (msg.get("content") or "")}
            else:
                merged.append(msg)
        return merged

    @staticmethod
    def _compress_history(
        previous_memory: str,
        new_messages: list[dict],
        current_user_prompt: str,
        target_tokens: int,
        memory_type: str = "",
    ) -> str:
        """Compress conversation history into a text summary within target_tokens.

        Only new_messages (the delta since the last compression) are fed into the
        compressor together with previous_memory (the accumulated summary so far)
        and the current turn's question.
        """
        if memory_type == "memagent":
            from bfcl_eval.memory.memagent import MemAgentMemory
            return MemAgentMemory.compress(previous_memory, new_messages, current_user_prompt, target_tokens)
        # Fallback stub: prepend previous_memory then truncated new_messages.
        prefix = ("[PREVIOUS MEMORY]\n" + previous_memory + "\n\n") if previous_memory.strip() else ""
        parts = []
        for msg in new_messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if content:
                parts.append(f"[{role}]: {str(content)[:300]}")
        return prefix + "[CONVERSATION HISTORY SUMMARY]\n" + "\n".join(parts)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        external_memory: "Memory | None" = None,
        context_budget: int = 128000,
        long_context: bool = False,
    ):
        # This method is used to retrive model response for each model.

        # FC model
        # TODO: Let all models have the is_fc_model attribute and remove the "FC" check
        if "FC" in self.registry_name or self.is_fc_model:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log,
                    external_memory=external_memory,
                    context_budget=context_budget,
                    long_context=long_context,
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log)
        # Prompting model
        else:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log,
                    external_memory=external_memory,
                    context_budget=context_budget,
                    long_context=long_context,
                )
            else:
                return self.inference_single_turn_prompting(test_entry, include_input_log)

    @final
    def inference_multi_turn_FC(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        external_memory: "Memory | None" = None,
        context_budget: int = 128000,
        long_context: bool = False,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list] = (
            []
        )  # The model response that will be used for later evaluation
        # Decoded function-call lists per turn × step (used by external metrics)
        all_decoded_responses: list[list[list[str]]] = []
        all_inference_log: list[list[dict]] = (
            []
        )  # The debugging log for human to understand
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        all_reasoning_content: list[list] = []
        # Shape: list[turn][step][message]. Only populated when use_compression=True.
        all_sent_context_snapshots: list[list[list[dict]]] = []

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=(long_context or "long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data["context_budget"] = context_budget

        # Per-sample memory: reset usage metrics at the start of each sample.
        if external_memory is not None:
            external_memory.begin_sample()

        use_compression = (
            external_memory is not None
            and getattr(external_memory, "memory_type", "") in _COMPRESSION_MEMORY_TYPES
        )

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                # Since we have added new functions, we need to recompile the tools
                inference_data = self._compile_tools(inference_data, test_entry)
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                # TODO: Move this to before pre_query_processing_FC.
                # Shouldn't be happening in the inference loop.
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            # Record start index before this turn's messages are added.
            turn_start_idx = len(inference_data.get("message", []))

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_FC(
                    inference_data, current_turn_message
                )
                if use_compression:
                    inference_data["_compressed_messages"] = list(inference_data["message"])
                    inference_data["_compressed_summary"] = ""
                    if getattr(external_memory, "memory_type", "") == "memobrain":
                        _task_text = current_turn_message[0].get("content", "") if current_turn_message else ""
                        external_memory.init_brain(_task_text)
            else:
                inference_data = self._add_next_turn_user_message_FC(
                    inference_data, current_turn_message
                )
                if use_compression and current_turn_message:
                    new_user_msg = current_turn_message[0]
                    _compressed = inference_data["_compressed_messages"]
                    # Merge into the last user message if adjacent, to avoid
                    # consecutive user-role messages that some APIs reject.
                    if _compressed and _compressed[-1].get("role") == "user":
                        prev = _compressed[-1]
                        sep = "\n\n" if prev.get("content") else ""
                        _compressed[-1] = {**prev, "content": (prev.get("content") or "") + sep + (new_user_msg.get("content") or "")}
                    else:
                        _compressed.append(new_user_msg)

            # Per-turn memory utilize: only for non-compression memory types.
            if external_memory is not None and not use_compression:
                try:
                    user_text = _extract_user_text_from_messages(inference_data["message"])
                    snippet = external_memory.utilize(user_text)
                    if snippet:
                        _append_memory_to_last_user_message(
                            inference_data["message"], snippet)
                except Exception as _mem_exc:
                    all_inference_log.append(
                        [{"role": "memory_log", "content": f"memory_utilize failed: {type(_mem_exc).__name__}: {_mem_exc}"}]
                    )

            current_turn_response = []
            current_turn_decoded_steps: list[list[str]] = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content = []
            current_turn_sent_snapshots: list[list[dict]] = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                if use_compression:
                    current_turn_sent_snapshots.append(
                        list(inference_data["_compressed_messages"])
                    )

                api_response, query_latency = self._query_FC(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # In compression mode, always record the actual messages sent to the LLM
                # (_compressed_messages is still the sent snapshot before _add_assistant_message_FC extends it).
                if use_compression:
                    current_step_inference_log.append(
                        {
                            "role": "compressed_context_log",
                            "content": list(inference_data["_compressed_messages"]),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_FC(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                _prev_msg_len = len(inference_data["message"])
                inference_data = self._add_assistant_message_FC(
                    inference_data, model_response_data
                )
                if use_compression:
                    inference_data["_compressed_messages"].extend(
                        inference_data["message"][_prev_msg_len:]
                    )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)

                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        long_context or "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                _prev_msg_len = len(inference_data["message"])
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )
                if use_compression:
                    _new_msgs = inference_data["message"][_prev_msg_len:]
                    inference_data["_compressed_messages"].extend(_new_msgs)
                    _mem_type = getattr(external_memory, "memory_type", "")
                    if _mem_type == "memobrain" and _new_msgs:
                        _asst_msg = (
                            inference_data["message"][_prev_msg_len - 1]
                            if _prev_msg_len > 0
                            and inference_data["message"][_prev_msg_len - 1].get("role") == "assistant"
                            else None
                        )
                        _to_memorize = ([_asst_msg] if _asst_msg else []) + list(_new_msgs)
                        external_memory.memorize(_to_memorize)
                    _step_tools = inference_data.get("tools", [])
                    _step_effective = context_budget - 512 - (_count_tokens(_step_tools) if _step_tools else 0)
                    if _count_tokens(inference_data["_compressed_messages"]) > _step_effective:
                        if _mem_type == "memobrain":
                            _step_summary = external_memory.recall()
                            _summary_msg = [{"role": "user", "content": _step_summary}]
                            if _count_tokens(_summary_msg) > _step_effective:
                                lo, hi = 0, len(_step_summary)
                                while lo < hi:
                                    mid = (lo + hi + 1) // 2
                                    if _count_tokens([{"role": "user", "content": _step_summary[:mid]}]) <= _step_effective:
                                        lo = mid
                                    else:
                                        hi = mid - 1
                                _step_summary = _step_summary[:lo]
                        else:
                            _prev_summary = inference_data.get("_compressed_summary", "")
                            _delta_start = 1 if _prev_summary else 0
                            _new_msgs_for_summary = inference_data["_compressed_messages"][_delta_start:]
                            _current_q = ""
                            if current_turn_message:
                                _qcontent = current_turn_message[0].get("content")
                                if isinstance(_qcontent, str):
                                    _current_q = _qcontent
                                elif isinstance(_qcontent, list):
                                    _current_q = " ".join(
                                        p.get("text", "") for p in _qcontent if isinstance(p, dict)
                                    )
                            _step_summary = BaseHandler._compress_history(
                                previous_memory=_prev_summary,
                                new_messages=_new_msgs_for_summary,
                                current_user_prompt=_current_q,
                                target_tokens=_step_effective,
                                memory_type=_mem_type,
                            )
                        inference_data["_compressed_summary"] = _step_summary
                        inference_data["_compressed_messages"] = [
                            {"role": "user", "content": _step_summary}
                        ]

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                # Track decoded calls for this completed step before advancing.
                current_turn_decoded_steps.append(list(decoded_model_responses))

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )

                    break

            # Per-turn memory update: ingest this turn's message delta.
            if external_memory is not None and not use_compression:
                try:
                    turn_delta = inference_data["message"][turn_start_idx:]
                    if turn_delta:
                        external_memory.update(turn_delta)
                except Exception as _mem_exc:
                    all_inference_log.append(
                        [{"role": "memory_log", "content": f"memory_update failed: {type(_mem_exc).__name__}: {_mem_exc}"}]
                    )

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_decoded_responses.append(current_turn_decoded_steps)
            all_inference_log.append(current_turn_inference_log)
            all_reasoning_content.append(current_turn_reasoning_content)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)
            if use_compression:
                all_sent_context_snapshots.append(current_turn_sent_snapshots)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        # Per-turn updates are done inside the turn loop; just collect usage metrics here.
        external_memory_usage = None
        if external_memory is not None:
            external_memory_usage = external_memory.drain_usage()

        # Build per-turn and sample-level aggregates for richer downstream logging.
        per_turn_totals = []
        for turn_in, turn_out, turn_lat in zip(
            total_input_token_count, total_output_token_count, total_latency
        ):
            per_turn_totals.append(
                {
                    "step_count": len(turn_lat),
                    "latency_s": round(sum(turn_lat), 4),
                    "input_tokens": int(sum(turn_in)),
                    "output_tokens": int(sum(turn_out)),
                    "input_tokens_per_step": [int(x) for x in turn_in],
                    "output_tokens_per_step": [int(x) for x in turn_out],
                }
            )
        sample_totals = {
            "step_count": sum(t["step_count"] for t in per_turn_totals),
            "latency_s": round(sum(t["latency_s"] for t in per_turn_totals), 4),
            "input_tokens": sum(t["input_tokens"] for t in per_turn_totals),
            "output_tokens": sum(t["output_tokens"] for t in per_turn_totals),
        }
        sample_totals["total_tokens"] = sample_totals["input_tokens"] + sample_totals["output_tokens"]

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
            # Decoded function-call lists — used by external metrics (compute_success_rate etc.)
            "decoded_responses": all_decoded_responses,
            # Per-turn and sample-level aggregates
            "per_turn_totals": per_turn_totals,
            "sample_totals": sample_totals,
            "force_quit": force_quit,
            # Full flat message history; JSON-serialisable when handler stores plain dicts.
            "full_message_history": inference_data.get("message", []),
        }

        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        if external_memory_usage is not None:
            metadata["memory_usage"] = {
                "latency_s": round(external_memory_usage.latency_s, 4),
                "input_tokens": external_memory_usage.input_tokens,
                "output_tokens": external_memory_usage.output_tokens,
                "embedding_tokens": external_memory_usage.embedding_tokens,
                "n_llm_calls": external_memory_usage.n_llm_calls,
                "n_embed_calls": external_memory_usage.n_embed_calls,
                "n_utilize": external_memory_usage.n_utilize,
                "n_update": external_memory_usage.n_update,
            }

        if use_compression and all_sent_context_snapshots:
            metadata["sent_context_snapshots"] = all_sent_context_snapshots

        return all_model_response, metadata

    @final
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        external_memory: "Memory | None" = None,
        context_budget: int = 128000,
        long_context: bool = False,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # Decoded function-call lists per turn × step (used by external metrics)
        all_decoded_responses: list[list[list[str]]] = []
        # Only for reasoning models, reasoning content will be stored as part of metadata and in inference log
        all_reasoning_content: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=(long_context or "long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data["context_budget"] = context_budget

        # Per-sample memory: reset usage metrics at the start of each sample.
        if external_memory is not None:
            external_memory.begin_sample()

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=holdout_function[str(turn_idx)]
                        ),
                    }
                ]

            # Record start index before this turn's messages are added.
            turn_start_idx = len(inference_data.get("message", []))

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            # Per-turn memory utilize: query with this turn's user message text.
            if external_memory is not None:
                try:
                    user_text = _extract_user_text_from_messages(inference_data["message"])
                    snippet = external_memory.utilize(user_text)
                    if snippet:
                        _append_memory_to_last_user_message(
                            inference_data["message"], snippet)
                except Exception as _mem_exc:
                    all_inference_log.append(
                        [{"role": "memory_log", "content": f"memory_utilize failed: {type(_mem_exc).__name__}: {_mem_exc}"}]
                    )

            current_turn_response = []
            current_turn_reasoning_content = []
            current_turn_decoded_steps: list[list[str]] = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_prompting(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_prompting(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)
                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    model_response_data["model_responses_decoded"] = decoded_model_responses
                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        current_turn_decoded_steps.append(
                            list(decoded_model_responses) if decoded_model_responses else []
                        )
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    current_turn_decoded_steps.append([])
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        long_context or "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                # Track decoded calls for this completed step before advancing.
                current_turn_decoded_steps.append(list(decoded_model_responses))

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            # Per-turn memory update: ingest this turn's message delta.
            if external_memory is not None and not use_compression:
                try:
                    turn_delta = inference_data["message"][turn_start_idx:]
                    if turn_delta:
                        external_memory.update(turn_delta)
                except Exception as _mem_exc:
                    all_inference_log.append(
                        [{"role": "memory_log", "content": f"memory_update failed: {type(_mem_exc).__name__}: {_mem_exc}"}]
                    )

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_decoded_responses.append(current_turn_decoded_steps)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        # Per-turn updates are done inside the turn loop; just collect usage metrics here.
        external_memory_usage = None
        if external_memory is not None:
            external_memory_usage = external_memory.drain_usage()

        # Build per-turn and sample-level aggregates for richer downstream logging.
        per_turn_totals = []
        for turn_in, turn_out, turn_lat in zip(
            total_input_token_count, total_output_token_count, total_latency
        ):
            per_turn_totals.append(
                {
                    "step_count": len(turn_lat),
                    "latency_s": round(sum(turn_lat), 4),
                    "input_tokens": int(sum(turn_in)),
                    "output_tokens": int(sum(turn_out)),
                    "input_tokens_per_step": [int(x) for x in turn_in],
                    "output_tokens_per_step": [int(x) for x in turn_out],
                }
            )
        sample_totals = {
            "step_count": sum(t["step_count"] for t in per_turn_totals),
            "latency_s": round(sum(t["latency_s"] for t in per_turn_totals), 4),
            "input_tokens": sum(t["input_tokens"] for t in per_turn_totals),
            "output_tokens": sum(t["output_tokens"] for t in per_turn_totals),
        }
        sample_totals["total_tokens"] = sample_totals["input_tokens"] + sample_totals["output_tokens"]

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
            # Decoded function-call lists — used by external metrics (compute_success_rate etc.)
            "decoded_responses": all_decoded_responses,
            # Per-turn and sample-level aggregates
            "per_turn_totals": per_turn_totals,
            "sample_totals": sample_totals,
            "force_quit": force_quit,
            # Full flat message history; JSON-serialisable when handler stores plain dicts.
            "full_message_history": inference_data.get("message", []),
        }
        # We only include reasoning content if it exists and is not empty
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content
        if external_memory_usage is not None:
            metadata["memory_usage"] = {
                "latency_s": round(external_memory_usage.latency_s, 4),
                "input_tokens": external_memory_usage.input_tokens,
                "output_tokens": external_memory_usage.output_tokens,
                "embedding_tokens": external_memory_usage.embedding_tokens,
                "n_llm_calls": external_memory_usage.n_llm_calls,
                "n_embed_calls": external_memory_usage.n_embed_calls,
                "n_utilize": external_memory_usage.n_utilize,
                "n_update": external_memory_usage.n_update,
            }

        return all_model_response, metadata

    @final
    def inference_single_turn_FC(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_FC(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_FC(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    @final
    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_prompting(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_prompting(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard AST checker input.
        """
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard execute checker input.
        """
        raise NotImplementedError

    @final
    def write(self, result, result_dir, update_mode=False):
        # Use the internal registry name to decide the result directory to avoid
        # collisions between different variants that share the same API model name.
        model_result_dir = result_dir / self.registry_dir_name

        if isinstance(result, dict):
            result = [result]

        # Collect and format each entry for JSON compatibility
        entries_to_write = [make_json_serializable(entry) for entry in result]

        # Group entries by their `test_category` for efficient file handling
        file_entries = {}
        for entry in entries_to_write:
            test_category = extract_test_category_from_id(entry["id"])
            # Determine the high-level grouping folder (non_live, live, etc.)
            group_dir_name = get_directory_structure_by_id(entry["id"])
            group_dir_path = model_result_dir / group_dir_name
            group_dir_path.mkdir(parents=True, exist_ok=True)

            file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result.json"
            file_entries.setdefault(file_path, []).append(entry)

        for file_path, entries in file_entries.items():
            if update_mode:
                # Load existing entries from the file
                existing_entries = {}
                if file_path.exists():
                    existing_entries = {
                        entry["id"]: entry for entry in load_file(file_path)
                    }

                # Update existing entries with new data
                for entry in entries:
                    existing_entries[entry["id"]] = entry

                # Sort entries by `id` and write them back to ensure order consistency
                sorted_entries = sorted(existing_entries.values(), key=sort_key)
                with open(file_path, "w") as f:
                    for entry in sorted_entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

            else:
                # Normal mode: Append to the end of the file
                # Note: We will sort all the entries at the end of the generation pipeline to ensure the order is consistent
                entries.sort(key=sort_key)
                with open(file_path, "a") as f:
                    for entry in entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the `_parse_query_response_FC` method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_FC` and `_add_next_turn_user_message_FC` methods, but it's usually cleaner to do it here.
        The inference_data dict is updated in place and returned.

        Note: This method has different signature from its Prompting version.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        [Only for FC mode]
        This method is used to prepare/compile the tools from the test entry and add them to the inference data to use for model query in FC mode.
        Function docs usually need to be transformed to the format expected by the model, done through the `convert_to_tool` function from `model_handler/utils.py`.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn. For most entry, this role will not appear.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_FC` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_prompting` and `_add_next_turn_user_message_prompting` methods, but it's usually cleaner to do it here.
        The function docs are usually supplied to the prompting models as part of the system prompt, done via the `system_prompt_pre_processing_chat_model` function from `model_handler/utils.py`, unless the model has a different way of handling it.
        Returns a dict that contains all the necessary information for the query method.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.

        Note: This method has different signature from its FC version.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_prompting` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        By default, execution results are added back as a `user` role message, as most models don't support the `tool` role in prompting mode.
        """
        raise NotImplementedError
