from coding_agent.contracts import ToolCall
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool


class FakeRuntime:
    def __init__(self):
        self.argv = None

    def run_process(self, argv: list[str]) -> ProcessResult:
        self.argv = argv
        return ProcessResult(
            return_code=0,
            stdout="hello\n",
            stderr="",
        )


def test_bash_tool_executes_valid_command():
    runtime = FakeRuntime()
    tool = BashTool()

    tool_call = ToolCall(
        id="call-1",
        name="bash",
        arguments={"command": "echo hello"},
    )

    result = tool.execute(tool_call, runtime)

    # 这里你自己写 assert
# ToolCall.arguments = {"command": "echo hello"}

# BashTool.execute(...)
# ↓
# FakeRuntime 收到
# ["bash", "-lc", "echo hello"]

# 并且返回的 ToolResult：
# tool_call_id 正确
# return_code == 0
# stdout == "hello\n"
    assert runtime.argv==["bash", "-lc", "echo hello"]
    assert result.tool_call_id == "call-1"
    assert result.return_code == 0
    assert result.stdout == "hello\n"


def test_bash_tool_rejects_invalid_arguments():
    runtime = FakeRuntime()
    tool = BashTool()

    tool_call = ToolCall(
        id="call-2",
        name="bash",
        arguments={},
    )

    result = tool.execute(tool_call, runtime)

    # 你自己验证：
    # 1. result.return_code is None
    # 2. result.error 不为空
    # 3. runtime.argv 仍然是 None
    assert result.return_code is None
    assert result.error is not None
    assert runtime.argv is None