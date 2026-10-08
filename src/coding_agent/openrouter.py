import json
from typing import Any

from coding_agent.contracts import ToolCall
from coding_agent.tools.base import ToolSpec


def to_openrouter_tools(
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


def to_openrouter_messages(
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
