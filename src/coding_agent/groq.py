"""Groq Free Plan adapter for the existing single-tool Agent Loop."""
import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelResponse
from coding_agent.deepseek import to_deepseek_messages, to_deepseek_tools
from coding_agent.openai_compatible import parse_chat_response
from coding_agent.tools.base import ToolSpec

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

# Narrow model allowlist for the free-model experiment: no arbitrary paid IDs.
ALLOWED_MODELS = frozenset({DEFAULT_MODEL})


def parse_groq_response(raw: dict[str, Any]) -> ModelResponse:
    return parse_chat_response(raw, provider="groq")


class GroqModel:
    provider = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        tool_specs: list[ToolSpec],
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_output_tokens: int = 4096,
    ):
        if model not in ALLOWED_MODELS:
            raise ValueError("Groq free-tier adapter only supports verified model IDs")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
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
            "max_completion_tokens": self.max_output_tokens,
            "stream": False,
        }
        if self._tools:
            payload["tools"] = self._tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        return payload

    def generate(self, history: list[dict[str, Any]]) -> ModelResponse:
        return parse_groq_response(self._post(self.build_payload(history)))

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
        # No automated retry on free-tier 429. The caller records MODEL_ERROR.
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
