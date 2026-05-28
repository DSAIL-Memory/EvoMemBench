import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from volcenginesdkarkruntime import Ark

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class ArkLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        if config is None:
            config = BaseLlmConfig()
        elif isinstance(config, dict):
            config = BaseLlmConfig(**config)
        super().__init__(config)

        api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = Ark(api_key=api_key)
        self.batch_model = os.getenv("BATCH_MODEL") or self.config.model

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(extract_json(tool_call.function.arguments)),
                    })
            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = {
            "model": self.batch_model,
            "messages": messages,
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.batch.chat.completions.create(**params)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** min(attempt, 5) * 5
                    logger.warning(
                        f"Ark API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

        parsed_response = self._parse_response(response, tools)
        if response.usage:
            self._last_token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return parsed_response
