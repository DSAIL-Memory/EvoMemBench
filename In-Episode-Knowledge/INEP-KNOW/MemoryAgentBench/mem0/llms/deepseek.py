import json
import logging
import os
import time
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import httpx
from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.deepseek import DeepSeekConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class DeepSeekLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, DeepSeekConfig, Dict]] = None):
        # Convert to DeepSeekConfig if needed
        if config is None:
            config = DeepSeekConfig()
        elif isinstance(config, dict):
            config = DeepSeekConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, DeepSeekConfig):
            # Convert BaseLlmConfig to DeepSeekConfig
            config = DeepSeekConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "deepseek-chat"

        api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = self.config.deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(1200.0, connect=30.0),
            max_retries=5,
        )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

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
        """
        Generate a response based on the given messages using DeepSeek.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional DeepSeek-specific parameters.

        Returns:
            str: The generated response.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** min(attempt, 5) * 5  # 5, 10, 20, 40, 80, 160s (cap at 160s)
                    logger.warning(f"DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
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
