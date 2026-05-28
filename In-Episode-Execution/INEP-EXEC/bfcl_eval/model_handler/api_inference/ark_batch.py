"""Volcengine Ark batch endpoint handlers for prompting and native function-calling modes."""
from __future__ import annotations

import os
from typing import Any

from overrides import override

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.memory.ark_client import ArkBatchClient
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    system_prompt_pre_processing_chat_model,
)


class ArkBatchPromptingHandler(OpenAICompletionsHandler):
    """Prompting-mode handler that calls the Volcengine Ark batch endpoint.

    Inherits all message-mutation methods, decode_execute, and the base
    _parse_query_response_prompting from OpenAICompletionsHandler.  The Ark
    batch endpoint returns OpenAI-shape ChatCompletion responses, so almost
    nothing needs to change.

    Key differences from the plain OpenAI handler:
    - Uses ArkBatchClient (Volcengine SDK + tenacity retry) instead of openai.OpenAI.
    - combine_consecutive_user_prompts applied to every turn in pre-processing
      (Ark / DeepSeek-class models reject back-to-back user messages).
    - _add_assistant_message_prompting stores a plain dict so that
      inference_data["message"] is JSON-serialisable and safe for memory.update().
    - _parse_query_response_prompting also captures reasoning_content for
      DeepSeek-R1-class models served on Ark.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool = False,
        ark_api_key: str | None = None,
        request_timeout: int = 1200,
        retry_attempts: int = 30,
        **kwargs,
    ) -> None:
        # Bypass OpenAICompletionsHandler.__init__ which builds an OpenAI() client
        # that we do not need here.
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self._ark = ArkBatchClient(
            api_key=ark_api_key or os.getenv("DEEPSEEK_API_KEY") or "",
            timeout=request_timeout,
            retry_attempts=retry_attempts,
        )

    # ------------------------------------------------------------------ #
    # Prompting methods                                                    #
    # ------------------------------------------------------------------ #

    @override
    def _query_prompting(self, inference_data: dict):
        budget = inference_data.get("context_budget", 128000)
        message = self._truncate_messages_to_budget(
            inference_data["message"], None, budget
        )
        inference_data["inference_input_log"] = {"message": repr(message)}
        return self._ark.chat_completions(
            model=self.model_name, messages=message
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # Ark / DeepSeek-class models reject consecutive user-role messages.
        for turn_idx in range(len(test_entry["question"])):
            test_entry["question"][turn_idx] = combine_consecutive_user_prompts(
                test_entry["question"][turn_idx]
            )

        return {"message": []}

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        # Store a plain dict so that inference_data["message"] stays JSON-serialisable
        # and can be passed directly to memory.update(trajectory).
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": str(model_response_data["model_responses"]),
            }
        )
        return inference_data

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_prompting(api_response)
        # Capture reasoning_content for DeepSeek-R1-class models served on Ark.
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data


class ArkBatchFCHandler(OpenAICompletionsHandler):
    """Function-calling handler for the Volcengine Ark batch endpoint.

    Identical setup to ArkBatchPromptingHandler but routes through
    inference_multi_turn_FC / inference_single_turn_FC, passing the
    ``tools`` list to the Ark batch API.  Inherits _compile_tools,
    _pre_query_processing_FC, add_first_turn_message_FC,
    _add_next_turn_user_message_FC, _add_execution_results_FC,
    decode_ast, and decode_execute from OpenAICompletionsHandler.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool = True,
        ark_api_key: str | None = None,
        request_timeout: int = 1200,
        retry_attempts: int = 30,
        **kwargs,
    ) -> None:
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self._ark = ArkBatchClient(
            api_key=ark_api_key or os.getenv("DEEPSEEK_API_KEY") or "",
            timeout=request_timeout,
            retry_attempts=retry_attempts,
        )

    # ------------------------------------------------------------------ #
    # FC methods                                                           #
    # ------------------------------------------------------------------ #

    @override
    def _query_FC(self, inference_data: dict):
        tools = inference_data["tools"]
        budget = inference_data.get("context_budget", 128000)
        if "_compressed_messages" in inference_data:
            # Compression mode: history already managed by inference loop.
            message = inference_data["_compressed_messages"]
        else:
            message = self._truncate_messages_to_budget(
                inference_data["message"], tools, budget
            )
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        return self._ark.chat_completions(
            model=self.model_name,
            messages=message,
            tools=tools if tools else None,
        )

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        response_data = super()._parse_query_response_FC(api_response)
        # Strip reasoning_content from chat history (R1-class models on Ark).
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        # Ensure the stored message is a plain dict so that inference_data["message"]
        # stays JSON-serialisable and safe for memory.update(trajectory).
        msg = model_response_data["model_responses_message_for_chat_history"]
        if not isinstance(msg, dict):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                msg = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            else:
                msg = {"role": "assistant", "content": msg.content or ""}
        inference_data["message"].append(msg)
        return inference_data
