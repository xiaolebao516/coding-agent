from coding_agent.tools.base import Tool

class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = { tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("duplicate tool name")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)