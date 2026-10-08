import json

from coding_agent.contracts import ToolCall
from coding_agent.deepseek import (
    to_deepseek_messages,
    to_deepseek_tools,
)
from coding_agent.tools.base import make_tool_spec
from coding_agent.tools.bash import BashArgs, BashTool


def test_make_tool_spec_from_bash_tool():
    spec = make_tool_spec(BashTool())

    assert spec.name == "bash"
    assert spec.description == "Execute a shell command."
    assert spec.parameters == BashArgs.model_json_schema()


def test_to_deepseek_tools():
    spec = make_tool_spec(BashTool())

    tools = to_deepseek_tools([spec])

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command.",
                "parameters": BashArgs.model_json_schema(),
            },
        }
    ]


def test_to_deepseek_messages():
    history = [
        {
            "role": "user",
            "content": "list files",
        },
        {
            "role": "assistant",
            "tool_call": ToolCall(
                id="call_1",
                name="bash",
                arguments={"command": "ls"},
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "return_code": 0,
            "stdout": "a.py\nb.py\n",
            "stderr": "",
            "timed_out": False,
            "error": None,
        },
    ]

    messages = to_deepseek_messages(history)

    assert messages[0] == {
        "role": "user",
        "content": "list files",
    }

    assistant = messages[1]

    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"

    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }

    tool = messages[2]

    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"

    observation = json.loads(tool["content"])

    assert observation == {
        "return_code": 0,
        "stdout": "a.py\nb.py\n",
        "stderr": "",
        "timed_out": False,
        "error": None,
    }
