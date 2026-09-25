import pytest

from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry


def test_registry_gets_tool_by_name():
    bash = BashTool()
    registry = ToolRegistry([bash])

    assert registry.get("bash") is bash


def test_registry_returns_none_for_unknown_tool():
    registry = ToolRegistry([BashTool()])

    assert registry.get("missing") is None


def test_registry_rejects_duplicate_tool_names():
    with pytest.raises(ValueError):
        ToolRegistry([
            BashTool(),
            BashTool(),
        ])