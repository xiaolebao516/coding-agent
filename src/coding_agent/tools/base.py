from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from pydantic import BaseModel

from coding_agent.contracts import ToolCall, ToolResult

if TYPE_CHECKING:
    from coding_agent.runtime.base import Runtime


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


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


def make_tool_spec(tool: Tool) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        parameters=tool.args_model.model_json_schema(),
    )