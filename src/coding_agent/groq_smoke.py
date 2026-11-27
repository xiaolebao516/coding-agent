"""Safe live two-turn Groq function-calling smoke (never executes model code)."""
import json
import os
from urllib.error import HTTPError

from coding_agent.contracts import ModelFinishReason
from coding_agent.groq import GroqModel
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry

MARKER = "groq_tool_ok"


def _generate_with_diagnostics(model: GroqModel, history: list[dict], key: str):
    """Report safe Groq HTTP error details; never dump HTML or credentials."""
    try:
        return model.generate(history)
    except HTTPError as exc:
        body = exc.read(4096)
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            error = data["error"]
            parts = []
            for field in ("type", "code", "message"):
                value = error.get(field)
                if isinstance(value, str) and value:
                    value = value.replace(key, "[REDACTED]")
                    parts.append(f"{field}={value[:400]}")
            detail = "; ".join(parts) if parts else "error JSON has no public fields"
        else:
            detail = (
                "non-JSON response"
                f"; content-type={exc.headers.get('Content-Type', 'unknown')}"
                f"; server={exc.headers.get('Server', 'unknown')}"
            )
        raise SystemExit(f"FAIL: Groq HTTP {exc.code}: {detail}") from None


def main() -> None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is missing")
    model = GroqModel(
        api_key=key,
        tool_specs=ToolRegistry([BashTool()]).specs(),
        max_output_tokens=1024,
    )
    history = [{
        "role": "user",
        "content": (
            "Use the bash tool with command 'printf groq_tool_ok' exactly once. "
            "Do not reply with a final answer until the tool output arrives. "
            "Then report the tool's output verbatim."
        ),
    }]
    first = _generate_with_diagnostics(model, history, key)
    call = first.tool_call
    if first.finish_reason != ModelFinishReason.FINISHED or call is None:
        raise SystemExit("FAIL: Groq did not emit a valid tool call")
    command = call.arguments.get("command")
    if call.name != "bash" or command != "printf groq_tool_ok":
        raise SystemExit("FAIL: Groq emitted an unexpected tool command")
    # Synthetic result only: no shell execution, no Docker.
    history.extend([
        {"role": "assistant", "tool_call": call},
        {"role": "tool", "tool_call_id": call.id, "return_code": 0,
         "stdout": MARKER, "stderr": "", "timed_out": False, "error": None},
    ])
    second = _generate_with_diagnostics(model, history, key)
    if (
        second.finish_reason != ModelFinishReason.FINISHED
        or second.tool_call is not None
        or MARKER not in (second.content or "")
    ):
        raise SystemExit("FAIL: Groq did not complete after the tool result")
    print("PASS: Groq tool call -> synthetic result -> final answer")
    print("tokens:", (first.usage.input_tokens, first.usage.output_tokens),
          (second.usage.input_tokens, second.usage.output_tokens))
    # The API response has no cost field; do not mislabel it as a $0 receipt.
    print("cost_usd: unknown (Groq free-tier account limits apply)")


if __name__ == "__main__":
    main()
