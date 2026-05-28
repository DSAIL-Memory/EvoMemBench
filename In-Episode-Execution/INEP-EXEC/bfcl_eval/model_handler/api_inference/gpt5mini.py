"""gpt-5-mini handler for the IE no-memory baseline.

Inherits OpenAICompletionsHandler but overrides _query_FC and _query_prompting
to apply context-budget truncation, matching the behaviour of ArkBatchFCHandler.
OPENAI_API_KEY / OPENAI_BASE_URL must be set in the environment before use
(the shell script exports them from GPT5MINI_API_KEY / GPT5MINI_BASE_URL).
"""
from __future__ import annotations

from overrides import override

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler


class Gpt5MiniPromptingHandler(OpenAICompletionsHandler):
    @override
    def _query_prompting(self, inference_data: dict):
        budget = inference_data.get("context_budget", 128000)
        message = self._truncate_messages_to_budget(
            inference_data["message"], None, budget
        )
        inference_data["inference_input_log"] = {"message": repr(message)}
        return self.generate_with_backoff(
            messages=message,
            model=self.model_name,
            temperature=self.temperature,
            store=False,
        )


class Gpt5MiniFCHandler(OpenAICompletionsHandler):
    @override
    def _query_FC(self, inference_data: dict):
        tools = inference_data["tools"]
        budget = inference_data.get("context_budget", 128000)
        if "_compressed_messages" in inference_data:
            message = inference_data["_compressed_messages"]
        else:
            message = self._truncate_messages_to_budget(
                inference_data["message"], tools, budget
            )
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        kwargs = {
            "messages": message,
            "model": self.model_name,
            "temperature": self.temperature,
            "store": False,
        }
        if tools:
            kwargs["tools"] = tools
        return self.generate_with_backoff(**kwargs)

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        # Convert ChatCompletionMessage object to plain dict so that
        # _truncate_messages_to_budget can call .get() on every message.
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
