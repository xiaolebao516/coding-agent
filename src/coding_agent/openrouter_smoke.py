"""Two-turn *live* OpenRouter tool-call smoke, without Docker or shell execution.

Run with OPENROUTER_API_KEY set; stdout contains no credential or raw response.
"""
import os

from coding_agent.contracts import ModelFinishReason
from coding_agent.openrouter import OpenRouterModel
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry

MARKER = "openrouter_tool_ok"


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    model = OpenRouterModel(
        api_key=key,
        tool_specs=ToolRegistry([BashTool()]).specs(),
        max_output_tokens=4096,
    )
    history = [{
        "role": "user",
        "content": (
            "This is a tool-use test. You MUST use the bash tool once with the command "
            "'printf openrouter_tool_ok'. Do not give a final answer until you "
            "receive the tool response. Then report the exact output."
        ),
    }]
    first = model.generate(history)
    call = first.tool_call
    if first.finish_reason != ModelFinishReason.FINISHED or call is None:
        raise SystemExit("FAIL: model did not emit a valid tool call")
    command = call.arguments.get("command")
    if call.name != "bash" or not isinstance(command, str) or MARKER not in command:
        raise SystemExit("FAIL: model emitted an unexpected tool call")
    # Intentionally do NOT execute the model's generated shell command.
    history.extend([
        {"role": "assistant", "tool_call": call},
        {
            "role": "tool", "tool_call_id": call.id, "return_code": 0,
            "stdout": MARKER, "stderr": "", "timed_out": False, "error": None,
        },
    ])
    second = model.generate(history)
    if (
        second.finish_reason != ModelFinishReason.FINISHED
        or second.tool_call is not None
        or MARKER not in (second.content or "")
    ):
        raise SystemExit("FAIL: model did not complete after receiving the tool result")
    print("PASS: live tool call -> synthetic tool result -> final answer")
    costs = (first.usage.cost_usd, second.usage.cost_usd)
    print("cost_usd:", sum(costs) if all(c is not None for c in costs) else "unknown")


if __name__ == "__main__":
    main()
