"""Magpie adapter for the existing OpenAI-compatible single-tool Agent Loop."""

import json
from typing import Any
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from coding_agent.contracts import ModelResponse
from coding_agent.deepseek import to_deepseek_messages, to_deepseek_tools
from coding_agent.openai_compatible import parse_chat_response
from coding_agent.tools.base import ToolSpec

# Verified from the user's live Magpie /v1/models and tool-calling preflight.
DEFAULT_MODEL = "workbuddy/deepseek-v4.1-flash"


class MagpieModel:
    provider = "magpie"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        tool_specs: list[ToolSpec],
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_output_tokens: int = 8192,
    ):
        if not api_key:
            raise ValueError("MAGPIE_API_KEY is required")
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MAGPIE_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("MAGPIE_BASE_URL must not contain credentials or query parameters")
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("MAGPIE_BASE_URL must end in /v1")
        if not model or timeout <= 0 or max_output_tokens <= 0:
            raise ValueError("model, timeout and max_output_tokens must be valid")

        self._api_key = api_key
        self._tools = to_deepseek_tools(tool_specs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        # Local/Tailscale Magpie must bypass global HTTP(S) proxy settings.
        self._opener = build_opener(ProxyHandler({}))

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
        return parse_chat_response(self._post(self.build_payload(history)), provider=self.provider)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "coding-agent/0.1",
            },
            method="POST",
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.load(response)
