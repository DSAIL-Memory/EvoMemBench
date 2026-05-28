#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import InferenceClient
from huggingface_hub.utils import is_torch_available
from PIL import Image

from openai import (
    BadRequestError,
    APIStatusError,
    APIConnectionError,
    OpenAIError,
)
import time

from .tools import Tool
from .utils import encode_image_base64, make_image_url


logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
}


class EmptyContentError(Exception):
    def __init__(self, response):
        self.response = response
        super().__init__(f"Empty content in response: {response}")



def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls]
        return cls(role=message.role, content=message.content, tool_calls=tool_calls, reasoning_content=message.reasoning_content, raw=raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            # ** ** ** ** ** ** ** *0307hhy ** ** ** ** *
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(
                        **{k: v for k, v in tc["function"].items() if k != "parameters"}
                    ),
                    id=tc["id"],
                    type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


class Model:
    def __init__(self, **kwargs):
        # Last call statistics (single API call)
        self.last_input_token_count = None
        self.last_output_token_count = None
        
        # Task-level cumulative statistics (multiple API calls in one task)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        
        self.kwargs = kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def reset_total_counts(self):
        """Reset cumulative token counts for a new task"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
    
    def get_token_counts(self) -> Dict[str, int]:
        """Get token counts from the last API call"""
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }
    
    def get_total_counts(self) -> Dict[str, int]:
        """Get cumulative token counts for the current task"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_api_calls": self.total_api_calls,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!

    def to_dict(self) -> Dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: Dict[str, Any]) -> "Model":
        model_instance = cls(
            **{
                k: v
                for k, v in model_dictionary.items()
                if k not in ["last_input_token_count", "last_output_token_count"]
            }
        )
        model_instance.last_input_token_count = model_dictionary.pop("last_input_token_count", None)
        model_instance.last_output_token_count = model_dictionary.pop("last_output_token_count", None)
        return model_instance


class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        self.custom_role_conversions = custom_role_conversions

    @staticmethod
    def truncate_content_based_on_stop_sequences(content: str, stop_sequences: List[str]) -> str:
        if not stop_sequences:
            return content
        # 在stop_seq之后截断content
        for stop_seq in stop_sequences:
            index = content.find(stop_seq)
            if index != -1:
                content = content[:index + len(stop_seq)]
                break  # Only keep the first match
        return content
    
    @staticmethod
    def remove_think_tags(content: str) -> str:
        """移除content中的think相关标签（<think>、<think>等）"""
        if not content:
            return content
        import re
        # 移除<think>...</think>标签及其内容（支持多行）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 移除<think>...</think>标签及其内容（支持多行）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 移除自闭合标签
        content = re.sub(r'<think\s*/>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<redacted_reasoning\s*/>', '', content, flags=re.IGNORECASE)
        return content.strip()
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Check if model_id contains 'o3' or 'o4'
        if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
            # Remove stop_sequences from completion_kwargs
            completion_kwargs.pop('stop', None)

        # response = self.client.chat.completions.create(**completion_kwargs)

        max_retries = 5
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**completion_kwargs)

                # Update last call counts
                self.last_input_token_count = response.usage.prompt_tokens
                self.last_output_token_count = response.usage.completion_tokens
                
                # Accumulate total counts for the current task
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.total_api_calls += 1

                if not response.choices[0].message.content and not getattr(response.choices[0].message, 'tool_calls', None):  # o1 o3-mini
                    raise EmptyContentError(response)

                message = ChatMessage.from_dict(
                    response.choices[0].message.model_dump(include={"role", "content", "tool_calls", "reasoning_content"})
                )
                message.raw = response

                # If model_id contains 'o3' or 'o4', manually truncate content based on stop_sequences
                if 'o3' in self.model_id.lower() or 'o4' in self.model_id.lower():
                    message.content = self.truncate_content_based_on_stop_sequences(message.content, stop_sequences)

                # 移除content中的think标签（适用于所有带think标签的模型）
                if message.content:
                    message.content = self.remove_think_tags(message.content)

                if tools_to_call_from is not None:
                    return parse_tool_args_if_needed(message)
                return message
            except BadRequestError as e:
                logger.error(f"Bad Request Error: {e}")
                raise
            except APIConnectionError as e:
                if attempt < max_retries:
                    logging.warning(f"Network error occurred: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except (APIStatusError, EmptyContentError) as e:
                if attempt < max_retries:
                    logging.warning(f"API status error occurred: {e}. Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    logging.error(f"Failed to complete request after {max_retries} retries.")
                    raise
            except OpenAIError as e:
                logging.error(f"API error occurred: {e}.")
                raise
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}.")
                raise

class ArkModel(Model):
    """Volcengine Ark batch inference model.

    Uses volcenginesdkarkruntime.Ark with client.batch.chat.completions.create().
    Credentials and endpoint are read from env vars DEEPSEEK_API_KEY and BATCH_MODEL
    by default, and can be overridden via constructor arguments.

    Parameters:
        model_id (`str`, *optional*): Ark endpoint ID (e.g. "ep-bi-...").
            Defaults to os.environ["BATCH_MODEL"].
        api_key (`str`, *optional*): Ark API key.
            Defaults to os.environ["DEEPSEEK_API_KEY"].
        custom_role_conversions (`dict`, *optional*): Role conversion mapping.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            from volcenginesdkarkruntime import Ark
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'volcengine-python-sdk[ark]' to use ArkModel: "
                "`pip install 'volcengine-python-sdk[ark]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id or os.environ["BATCH_MODEL"]
        self.client = Ark(api_key=api_key or os.environ["DEEPSEEK_API_KEY"])
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        # volcenginesdkarkruntime does not accept 'stop' kwarg
        completion_kwargs.pop("stop", None)

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.batch.chat.completions.create(**completion_kwargs)

                self.last_input_token_count = getattr(response.usage, "prompt_tokens", None)
                self.last_output_token_count = getattr(response.usage, "completion_tokens", None)
                if self.last_input_token_count:
                    self.total_input_tokens += self.last_input_token_count
                if self.last_output_token_count:
                    self.total_output_tokens += self.last_output_token_count
                self.total_api_calls += 1

                msg = response.choices[0].message
                content = getattr(msg, "content", None) or ""
                if not content:
                    raise EmptyContentError(response)
                content = OpenAIServerModel.remove_think_tags(content)
                reasoning_content = getattr(msg, "reasoning_content", None)

                message = ChatMessage(
                    role=getattr(msg, "role", "assistant"),
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=None,
                    raw=response,
                )
                if tools_to_call_from is not None:
                    return parse_tool_args_if_needed(message)
                return message

            except BadRequestError as e:
                # 400 错误(如内容违规、参数非法)不重试
                logging.error(f"[ArkModel] bad request: {e}")
                raise
            except APIConnectionError as e:
                # 网络错误,短暂等待后重试
                if attempt < max_retries - 1:
                    logging.warning(f"[ArkModel] network error (attempt {attempt + 1}): {e}. "
                                    f"Retrying in 5s...")
                    time.sleep(5)
                else:
                    logging.error(f"[ArkModel] failed after {max_retries} retries.")
                    raise
            except (APIStatusError, EmptyContentError) as e:
                # 限速/余额不足/空响应,等待60s后重试
                if attempt < max_retries - 1:
                    logging.warning(f"[ArkModel] API status error (attempt {attempt + 1}): {e}. "
                                    f"Retrying in 60s...")
                    time.sleep(60)
                else:
                    logging.error(f"[ArkModel] failed after {max_retries} retries.")
                    raise
            except Exception as e:
                # 其他未预期错误,直接抛出不重试
                logging.error(f"[ArkModel] unexpected error: {e}")
                raise


class _ArkBatchCompletions:
    def __init__(self, ark_batch_completions, model_id):
        self._inner = ark_batch_completions
        self._model_id = model_id

    def create(self, model=None, **kwargs):
        return self._inner.create(model=self._model_id, **kwargs)


class _ArkBatchChat:
    def __init__(self, ark_batch_completions, model_id):
        self.completions = _ArkBatchCompletions(ark_batch_completions, model_id)


class ArkBatchServerModel(OpenAIServerModel):
    """Drop-in replacement for OpenAIServerModel using Ark batch inference.

    Reads DEEPSEEK_API_KEY and BATCH_MODEL from environment.
    When model_id == OPENAI_AGENT_MODEL env var, degrades to plain OpenAIServerModel
    using AIGCBEST_API_KEY / AIGCBEST_BASE_URL (for gpt-5-mini / gemini baselines).
    """

    # Ark batch endpoint hard limit is 98304 tokens; use 90000 chars as a
    # conservative proxy (DeepSeek tokenizer: ~1 char ≈ 1 token for CJK text).
    MAX_CONTEXT_CHARS = 90_000

    def __init__(self, model_id=None, **kwargs):
        # OpenAI 直连分支：当 model_id 与 OPENAI_AGENT_MODEL 相符时，
        # 退化为标准 OpenAIServerModel，不依赖 Ark SDK 也不读 BATCH_MODEL。
        openai_agent_model = os.environ.get("OPENAI_AGENT_MODEL", "")
        if openai_agent_model and model_id == openai_agent_model:
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            api_key  = kwargs.pop("api_key",  None) or os.environ.get("AIGCBEST_API_KEY")
            api_base = (kwargs.pop("api_base", None)
                        or os.environ.get("AIGCBEST_BASE_URL"))
            super().__init__(model_id=model_id, api_key=api_key, api_base=api_base, **kwargs)
            return
        # Ark batch 分支（原始逻辑）
        batch_model = os.environ.get("BATCH_MODEL", model_id)
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        # Ark batch API only accepts max_tokens, not max_completion_tokens
        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        super().__init__(
            model_id=batch_model,
            api_key=os.environ.get("OPENAI_API_KEY", "unused"),
            api_base=os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            **kwargs,
        )
        from volcenginesdkarkruntime import Ark
        _ark = Ark(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        self.client = type("_ArkBatchClientShim", (), {
            "chat": _ArkBatchChat(_ark.batch.chat.completions, batch_model)
        })()

    _MEMORY_MARKER = "————Memory System Guidance————"

    @staticmethod
    def _msg_chars(msg) -> int:
        """Return character length of a message's content (handles str and list blocks)."""
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        else:
            content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            return sum(len(str(block.get("text", "") if isinstance(block, dict) else block)) for block in content)
        return len(str(content))

    @classmethod
    def _is_memory_msg(cls, msg) -> bool:
        """Return True if this message is a memory guidance block."""
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        else:
            content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            return any(
                cls._MEMORY_MARKER in str(block.get("text", "") if isinstance(block, dict) else block)
                for block in content
            )
        return cls._MEMORY_MARKER in str(content)

    def _truncate_messages(self, messages):
        """Shorten memory-guidance content to fit within MAX_CONTEXT_CHARS."""
        total = sum(self._msg_chars(m) for m in messages)
        if total <= self.MAX_CONTEXT_CHARS:
            return messages

        result = list(messages)
        non_memory_total = sum(
            self._msg_chars(m) for m in result if not self._is_memory_msg(m)
        )
        memory_budget = self.MAX_CONTEXT_CHARS - non_memory_total

        if memory_budget <= 0:
            result = [m for m in result if not self._is_memory_msg(m)]
            logger.warning(
                f"[ArkBatchServerModel] Context truncated: {total} → {non_memory_total} chars, "
                f"memory guidance dropped (non-memory alone exceeds limit)."
            )
            return result

        for i, msg in enumerate(result):
            if not self._is_memory_msg(msg):
                continue
            msg_chars = self._msg_chars(msg)
            if msg_chars <= memory_budget:
                memory_budget -= msg_chars
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                new_blocks = []
                remaining = memory_budget
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        new_blocks.append({**block, "text": text[:remaining]})
                        remaining = max(0, remaining - len(text))
                    else:
                        new_blocks.append(block)
                result[i] = {**msg, "content": new_blocks}
            else:
                result[i] = {**msg, "content": str(content)[:memory_budget]}
            memory_budget = 0

        new_total = sum(self._msg_chars(m) for m in result)
        logger.warning(
            f"[ArkBatchServerModel] Context truncated: {total} → {new_total} chars, "
            f"memory guidance content shortened to {self.MAX_CONTEXT_CHARS - non_memory_total} chars budget."
        )
        return result

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> "ChatMessage":
        messages = self._truncate_messages(messages)
        return super().__call__(
            messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )


_ARK_PROVIDERS = {
    "bm25", "qwen3_embedding_4b", "graphrag",
    "agent_kb", "agent_workflow_memory", "skillweaver", "lightweight_memory",
    "mem0", "ace", "reasoning_bank",
    "a_mem", "memos", "memory_os",
}


def build_model(memory_provider=None, model_id=None, api_key=None, api_base=None,
                custom_role_conversions=None, **kwargs) -> Model:
    """Factory: returns ArkModel for Ark-batch providers, else OpenAIServerModel.

    For ArkModel, model_id/api_key/api_base are ignored — it reads BATCH_MODEL
    and DEEPSEEK_API_KEY from the environment. All remaining kwargs (e.g.
    max_completion_tokens) are forwarded to the chosen model class.
    """
    if memory_provider in _ARK_PROVIDERS:
        return ArkModel(custom_role_conversions=custom_role_conversions, **kwargs)
    return OpenAIServerModel(
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
        custom_role_conversions=custom_role_conversions,
        **kwargs,
    )


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "OpenAIServerModel",
    "ArkModel",
    "ArkBatchServerModel",
    "build_model",
    "ChatMessage",
]
