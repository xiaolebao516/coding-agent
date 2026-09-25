from typing import Protocol, TYPE_CHECKING

from pydantic import BaseModel

from coding_agent.contracts import ToolCall, ToolResult

if TYPE_CHECKING:
    from coding_agent.runtime.base import Runtime


class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]

    def execute(
        self,
        tool_call: ToolCall,
        runtime: "Runtime",
    ) -> ToolResult:
        ...