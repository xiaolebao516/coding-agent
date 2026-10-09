import pytest

from coding_agent.tools.bash import BashArgs, BashTool
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


def test_registry_specs_match_registered_tools():
    registry = ToolRegistry([BashTool()])

    specs = registry.specs()

    assert [spec.name for spec in specs] == ["bash"]
    assert specs[0].parameters == BashArgs.model_json_schema()
