"""Minimal OpenRouter free-model adapter for the existing one-tool Agent Loop."""
import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelResponse
from coding_agent.openai_compatible import parse_chat_response
from coding_agent.deepseek import to_deepseek_messages, to_deepseek_tools
from coding_agent.tools.base import ToolSpec

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def parse_openrouter_response(raw: dict[str, Any]) -> ModelResponse:
    return parse_chat_response(raw, provider="openrouter", require_zero_cost=True)

class OpenRouterModel:
    provider = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        tool_specs: list[ToolSpec],
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_output_tokens: int = 8192,
    ):
        if not model.endswith(":free"):
            raise ValueError("OpenRouter adapter only permits explicit :free models")
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        if timeout <= 0 or max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")
        self._api_key = api_key
        self._tools = to_deepseek_tools(tool_specs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def build_payload(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": to_deepseek_messages(history),
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        if self._tools:
            payload["tools"] = self._tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        return payload

    def generate(self, history: list[dict[str, Any]]) -> ModelResponse:
        return parse_openrouter_response(self._post(self.build_payload(history)))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        # HTTP 429 is propagated to Agent's existing MODEL_ERROR path. No retries.
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
