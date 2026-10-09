from coding_agent.tools.base import Tool, ToolSpec, make_tool_spec

class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = { tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("duplicate tool name")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """Schemas of exactly the tools this registry can dispatch."""
        return [make_tool_spec(tool) for tool in self._tools.values()]
