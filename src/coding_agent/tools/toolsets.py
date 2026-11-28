"""Toolset factory for ablation experiments (baseline bash vs SWE-agent ACI)."""

from __future__ import annotations

from coding_agent.tools.aci import EditFileTool, SearchRepoTool, ViewFileTool
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry

VALID_TOOLSETS = ("bash", "aci")


def build_tool_registry(toolset: str = "bash") -> ToolRegistry:
    """Build a ToolRegistry for the requested experiment profile.

    - ``bash``: Baseline single-tool setup (BashTool only).
    - ``aci``: Controlled ACI setup (BashTool + view_file + search_repo + edit_file).
    """
    if toolset == "bash":
        return ToolRegistry([BashTool()])
    if toolset == "aci":
        return ToolRegistry([
            BashTool(),
            ViewFileTool(),
            SearchRepoTool(),
            EditFileTool(),
        ])
    raise ValueError(f"unsupported toolset: {toolset!r} (expected one of {VALID_TOOLSETS})")
