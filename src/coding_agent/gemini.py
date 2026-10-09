import json
from typing import Any
from urllib.request import Request, urlopen

from coding_agent.contracts import ModelResponse, ToolCall, Usage
from coding_agent.tools.base import ToolSpec


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

        return messages

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
        candidate = raw["candidates"][0]
        original = candidate["content"]
        parts = original.get("parts", [])

        calls = [part["functionCall"] for part in parts if "functionCall" in part]

        if len(calls) > 1:
            raise ValueError("parallel tool calls not supported")

        tool_call = None

        if calls:
            call = calls[0]
            call_id = call["id"]

            # 必须保存整个原始 Content，不能丢 thoughtSignature
            self._pending_calls[call_id] = original

            tool_call = ToolCall(
                id=call_id,
                name=call["name"],
                arguments=call["args"],
            )

        content = (
            "\n".join(
                part["text"]
                for part in parts
                if "text" in part and not part.get("thought", False)
            )
            or None
        )

        usage = raw.get("usageMetadata") or {}

        return ModelResponse(
            content=content,
            tool_call=tool_call,
            usage=Usage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=(
                    usage["candidatesTokenCount"] + usage.get("thoughtsTokenCount", 0)
                    if "candidatesTokenCount" in usage
                    else None
                ),
                cost_usd=None,
            ),
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
