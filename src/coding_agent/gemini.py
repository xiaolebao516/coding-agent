import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall, Usage
from coding_agent.tools.base import ToolSpec


def _merge_consecutive_user_turns(messages: list[dict]) -> list[dict]:
    """Gemini expects alternating turns; fold adjacent user turns into one.

    Adjacent user turns appear when a tool result (sent as role "user") is
    followed by agent feedback, e.g. after an invalid action.
    """
    merged: list[dict] = []
    for message in messages:
        if merged and message["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1] = {"role": "user", "parts": merged[-1]["parts"] + message["parts"]}
        else:
            merged.append(message)
    return merged


class GeminiModel:
    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        tool_specs: list[ToolSpec],
        model: str = "gemini-3.8-flash",
        timeout: float = 60.0,
        max_output_tokens: int = 512,
    ):
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.max_output_tokens = max_output_tokens

        self._tools = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in tool_specs
        ]

        # 保留模型返回的原始工具调用及思考签名
        self._pending_calls: dict[str, dict[str, Any]] = {}

    def _messages(self, history: list[dict]) -> list[dict]:
        messages = []

        for item in history:
            role = item["role"]

            if role == "user":
                messages.append(
                    {
                        "role": "user",
                        "parts": [{"text": item["content"]}],
                    }
                )

            elif role == "assistant":
                call = item["tool_call"]
                original = self._pending_calls.get(call.id)

                if original is None:
                    raise ValueError(f"missing Gemini tool-call state: {call.id}")

                messages.append(original)

            elif role == "tool":
                call_id = item["tool_call_id"]
                original = self._pending_calls.get(call_id)

                if original is None:
                    raise ValueError(f"unknown Gemini tool_call_id: {call_id}")

                function_call = next(
                    part["functionCall"]
                    for part in original["parts"]
                    if "functionCall" in part
                    and part["functionCall"].get("id") == call_id
                )

                observation = {
                    "return_code": item.get("return_code"),
                    "stdout": item.get("stdout", ""),
                    "stderr": item.get("stderr", ""),
                    "timed_out": item.get("timed_out", False),
                    "error": item.get("error"),
                }

                messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": call_id,
                                    "name": function_call["name"],
                                    "response": observation,
                                }
                            }
                        ],
                    }
                )

            else:
                raise ValueError(f"unsupported role: {role}")

        return _merge_consecutive_user_turns(messages)

    def generate(self, history: list[dict]) -> ModelResponse:
        payload = {
            "contents": self._messages(history),
            "tools": [{"functionDeclarations": self._tools}],
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": "low"},
            },
        }

        raw = self._post(payload)

        # Preserve token usage even when the provider has hit its output limit.
        usage_metadata = raw.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=usage_metadata.get("promptTokenCount"),
            output_tokens=(
                usage_metadata["candidatesTokenCount"]
                + usage_metadata.get("thoughtsTokenCount", 0)
                if "candidatesTokenCount" in usage_metadata
                else None
            ),
            cost_usd=None,
        )

        candidates = raw.get("candidates") or []
        if not candidates:
            prompt_feedback = raw.get("promptFeedback") or {}
            detail = prompt_feedback.get("blockReason") or "missing_candidates"
            return ModelResponse(
                content=None,
                tool_call=None,
                usage=usage,
                finish_reason=ModelFinishReason.PROVIDER_STOPPED,
                stop_detail=f"gemini:{detail}",
            )

        candidate = candidates[0]
        original = candidate.get("content") or {}
        parts = original.get("parts", [])
        content = (
            "\n".join(
                part["text"]
                for part in parts
                if "text" in part and not part.get("thought", False)
            )
            or None
        )

        # A truncated tool call may lack id/args; it must never execute.
        if candidate.get("finishReason") == "MAX_TOKENS":
            return ModelResponse(
                content=content,
                tool_call=None,
                usage=usage,
                finish_reason=ModelFinishReason.OUTPUT_TRUNCATED,
                stop_detail="gemini:MAX_TOKENS",
            )

        reason = candidate.get("finishReason")
        if reason == "MALFORMED_FUNCTION_CALL":
            # The model tried to call a tool but produced an invalid call: recoverable.
            return ModelResponse(
                content=content,
                tool_call=None,
                usage=usage,
                finish_reason=ModelFinishReason.INVALID_ACTION,
                stop_detail="gemini:MALFORMED_FUNCTION_CALL",
            )

        if reason not in (None, "STOP"):
            return ModelResponse(
                content=content,
                tool_call=None,
                usage=usage,
                finish_reason=ModelFinishReason.PROVIDER_STOPPED,
                stop_detail=f"gemini:{reason}",
            )

        calls = [part["functionCall"] for part in parts if "functionCall" in part]

        if len(calls) > 1:
            return ModelResponse(
                content=content,
                tool_call=None,
                usage=usage,
                finish_reason=ModelFinishReason.INVALID_ACTION,
                stop_detail="gemini:unsupported_tool_calls",
            )

        tool_call = None

        if calls:
            call = calls[0]
            if (not isinstance(call, dict)
                    or not call.get("id")
                    or not call.get("name")
                    or not isinstance(call.get("args"), dict)):
                return ModelResponse(
                    content=content,
                    tool_call=None,
                    usage=usage,
                    finish_reason=ModelFinishReason.INVALID_ACTION,
                    stop_detail="gemini:invalid_tool_call",
                )
            call_id = call["id"]

            # Preserve the raw Content (including the thoughtSignature).
            self._pending_calls[call_id] = original

            tool_call = ToolCall(
                id=call_id,
                name=call["name"],
                arguments=call["args"],
            )

        return ModelResponse(
            content=content,
            tool_call=tool_call,
            usage=usage,
        )

    def _post(self, payload: dict) -> dict:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
