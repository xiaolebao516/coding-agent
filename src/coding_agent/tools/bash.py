from pydantic import BaseModel, ValidationError

from coding_agent.contracts import ToolCall, ToolResult
from coding_agent.runtime.base import Runtime


class BashArgs(BaseModel):
    command: str


class BashTool:
    name = "bash"
    description = "Execute a shell command."
    args_model = BashArgs

    def execute(
        self,
        tool_call: ToolCall,
        runtime: Runtime,
    ) -> ToolResult:
        try:
            args = BashArgs.model_validate(tool_call.arguments)

        except ValidationError as exc:
            return ToolResult(
            tool_call_id=tool_call.id,
            return_code=None,
            stdout="",
            stderr="",
            error=f"invalid arguments: {exc}",
        )

        result = runtime.run_process(["bash", "-lc", args.command])
        return ToolResult(tool_call_id=tool_call.id,
                            return_code=result.return_code,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            timed_out=result.timed_out,
                            error=result.error)