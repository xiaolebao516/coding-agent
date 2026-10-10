"""Shared parser for OpenAI-compatible chat-completions responses.

Tool JSON validation is identical across OpenRouter and Groq. Provider-specific
cost enforcement stays at the adapter boundary.
"""
import json
from typing import Any

from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall, Usage


def parse_chat_response(
    raw: dict[str, Any],
    *,
    provider: str,
    require_zero_cost: bool = False,
) -> ModelResponse:
    raw_usage = raw.get("usage")
    usage = Usage()
    if isinstance(raw_usage, dict):
        cost = raw_usage.get("cost")
        if require_zero_cost and cost is not None and (
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or cost != 0
        ):
            raise ValueError(f"{provider} returned nonzero or invalid cost for free model")
        prompt_details = raw_usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens"),
            output_tokens=raw_usage.get("completion_tokens"),
            cache_read_tokens=prompt_details.get("cached_tokens"),
            # Groq Free Plan does not provide a USD cost field: unknown, not zero.
            cost_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        )

    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ModelResponse(None, None, usage, ModelFinishReason.PROVIDER_STOPPED, f"{provider}:missing_choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return ModelResponse(None, None, usage, ModelFinishReason.PROVIDER_STOPPED, f"{provider}:missing_message")
    content = message.get("content")
    reason = choice.get("finish_reason")
    if reason == "length":
        return ModelResponse(content, None, usage, ModelFinishReason.OUTPUT_TRUNCATED, f"{provider}:length")
    if reason not in ("stop", "tool_calls", None):
        return ModelResponse(content, None, usage, ModelFinishReason.PROVIDER_STOPPED, f"{provider}:{reason}")

    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or len(calls) > 1:
        return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, f"{provider}:unsupported_tool_calls")
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
            return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, f"{provider}:invalid_tool_call")
        return ModelResponse(content, tool_call, usage)
    if reason == "tool_calls":
        return ModelResponse(content, None, usage, ModelFinishReason.INVALID_ACTION, f"{provider}:missing_tool_call")
    return ModelResponse(content, None, usage)
