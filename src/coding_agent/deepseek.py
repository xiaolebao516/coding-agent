import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall, Usage
from coding_agent.tools.base import ToolSpec

DEFAULT_BASE_URL = "https://api.deepseek.com"


def to_deepseek_tools(
    tool_specs: list[ToolSpec],
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in tool_specs
    ]


def to_deepseek_messages(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    for item in history:
        role = item["role"]

        if role == "user":
            messages.append(
                {
                    "role": "user",
                    "content": item["content"],
                }
            )
            continue

        if role == "assistant":
            tool_call = item.get("tool_call")

            if not isinstance(tool_call, ToolCall):
                raise ValueError("assistant history item must contain ToolCall")

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    tool_call.arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
            continue

        if role == "tool":
            observation = {
                "return_code": item.get("return_code"),
                "stdout": item.get("stdout", ""),
                "stderr": item.get("stderr", ""),
                "timed_out": item.get("timed_out", False),
                "error": item.get("error"),
            }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["tool_call_id"],
                    "content": json.dumps(
                        observation,
                        ensure_ascii=False,
                    ),
                }
            )
            continue

        raise ValueError(f"unsupported history role: {role}")

    return messages


# ---------------------------------------------------------------------------
# Pricing (LiteLLM price table)
# ---------------------------------------------------------------------------


def _price_key(model: str) -> str:
    return f"deepseek/{model}"


def _require_price_entry(model: str) -> None:
    import litellm

    if _price_key(model) not in litellm.model_cost:
        raise ValueError(
            f"no LiteLLM price entry for {_price_key(model)!r}; "
            "refusing to run without a way to compute cost"
        )


def _cost_usd(
    model: str,
    input_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one call from the LiteLLM price table.

    May raise, or return <= 0, if the price table cannot price this call.
    """
    import litellm

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=_price_key(model),
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        cache_read_input_tokens=cache_read_tokens,
    )
    return prompt_cost + completion_cost


# ---------------------------------------------------------------------------
# Response -> ModelResponse / Usage
# ---------------------------------------------------------------------------


def usage_from_deepseek(raw_usage: dict[str, Any] | None, model: str) -> Usage:
    """Map DeepSeek `usage` to our Usage.

    Rules (decided 2026-10-08):
    1. prompt_tokens -> input_tokens, prompt_cache_hit_tokens -> cache_read_tokens,
       completion_tokens -> output_tokens; cost_usd from _cost_usd(...).
    2. raw_usage is None -> every field None.
    3. prompt_cache_hit_tokens missing -> cache_read_tokens None and cost_usd None
       (input/output tokens still filled).
    5. _cost_usd raises, or returns <= 0 -> tokens still filled, cost_usd None.

    TODO(Wu): implement. Spec = tests/test_deepseek.py::test_usage_*
    """
    if raw_usage is None:
        return Usage()
    
    input_tokens = raw_usage.get("prompt_tokens")
    cache_read_tokens = raw_usage.get("prompt_cache_hit_tokens")
    output_tokens = raw_usage.get("completion_tokens")
    usage = Usage(
                input_tokens=input_tokens,
                cache_read_tokens=cache_read_tokens,
                output_tokens=output_tokens,
                cost_usd=None,
            )
    
    if cache_read_tokens is None or input_tokens is None or output_tokens is None:
        return usage

    try:
        cost_usd = _cost_usd(model, input_tokens, cache_read_tokens, output_tokens)
        if cost_usd > 0:
            usage.cost_usd = cost_usd

    except Exception:
        pass
    return usage

def parse_deepseek_response(raw: dict[str, Any], model: str) -> ModelResponse:
    choice = raw["choices"][0]
    message = choice.get("message") or {}
    usage = usage_from_deepseek(raw.get("usage"), model)

    # The model may have returned incomplete JSON in a tool call.
    # Detect truncation BEFORE parsing any tool arguments.
    if choice.get("finish_reason") == "length":
        return ModelResponse(
            content=message.get("content"),
            tool_call=None,
            usage=usage,
            finish_reason=ModelFinishReason.OUTPUT_TRUNCATED,
        )

    tool_calls = message.get("tool_calls") or []

    if len(tool_calls) > 1:
        raise ValueError(f"expected at most one tool call, got {len(tool_calls)}")

    tool_call = None
    if tool_calls:
        call = tool_calls[0]
        tool_call = ToolCall(
            id=call["id"],
            name=call["function"]["name"],
            arguments=json.loads(call["function"]["arguments"]),
        )

    return ModelResponse(
        content=message.get("content"),
        tool_call=tool_call,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DeepSeekModel:
    provider = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        tool_specs: list[ToolSpec],
        model: str = "deepseek-flash",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_output_tokens: int | None = None,
    ):
        _require_price_entry(model)
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._tools = to_deepseek_tools(tool_specs)
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        self.max_output_tokens = max_output_tokens

    def build_payload(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": to_deepseek_messages(history),
            "tools": self._tools,
            # Thinking mode off for now: cheaper, and avoids having to echo
            # reasoning_content back during tool-call turns.
            "thinking": {"type": "disabled"},
        }
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens

        return payload

    def generate(self, history: list[dict[str, Any]]) -> ModelResponse:
        raw = self._post(self.build_payload(history))
        return parse_deepseek_response(raw, self.model)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
