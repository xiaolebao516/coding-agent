"""Minimal OpenRouter free-model adapter for the existing one-tool Agent Loop."""
import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall, Usage
from coding_agent.deepseek import to_deepseek_messages, to_deepseek_tools
from coding_agent.tools.base import ToolSpec

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def parse_openrouter_response(raw: dict[str, Any]) -> ModelResponse:
    """Keep provider stop reasons and counted tokens, never execute malformed tool JSON."""
    raw_usage = raw.get("usage")
    usage = Usage()
    if isinstance(raw_usage, dict):
        cost = raw_usage.get("cost")
        # This adapter is intentionally free-only; do not silently accept a charge.
        if cost is not None and (not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost != 0):
            raise ValueError("OpenRouter returned nonzero or invalid cost for free model")
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens"),
            output_tokens=raw_usage.get("completion_tokens"),
            cache_read_tokens=(raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            cost_usd=float(cost) if cost is not None else None,
        )

    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ModelResponse(None, None, usage, ModelFinishReason.PROVIDER_STOPPED, "openrouter:missing_choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return ModelResponse(None, None, usage, ModelFinishReason.PROVIDER_STOPPED, "openrouter:missing_message")
    content = message.get("content")
    reason = choice.get("finish_reason")
    if reason == "length":
        return ModelResponse(content, None, usage, ModelFinishReason.OUTPUT_TRUNCATED, "openrouter:length")
    if reason not in ("stop", "tool_calls", None):
        return ModelResponse(content, None, usage, ModelFinishReason.PROVIDER_STOPPED, f"openrouter:{reason}")

    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or len(calls) > 1:
        return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, "openrouter:unsupported_tool_calls")
    if calls:
        try:
            call = calls[0]
            arguments = json.loads(call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            call_id, name = call["id"], call["function"]["name"]
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                raise ValueError("missing tool call id/name")
            tool_call = ToolCall(id=call_id, name=name, arguments=arguments)
        except (KeyError, TypeError, ValueError, AttributeError):
            return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, "openrouter:invalid_tool_call")
        return ModelResponse(content, tool_call, usage)
    # A tool_calls finish without a tool call is not a successful completion.
    if reason == "tool_calls":
        return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, "openrouter:missing_tool_call")
    return ModelResponse(content, None, usage)


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
