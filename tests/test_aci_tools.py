"""Offline unit tests for M3 ACI tools (view_file, search_repo, edit_file) and toolsets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from coding_agent.contracts import ToolCall
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.aci import (
    MAX_VIEW_WINDOW,
    EditFileTool,
    SearchRepoTool,
    ViewFileTool,
)
from coding_agent.tools.toolsets import build_tool_registry


class LocalWorkspaceRuntime:
    """Offline test runtime executing python commands inside tmp_path without Docker."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_process(self, argv: list[str]) -> ProcessResult:
        cmd = [sys.executable if argv[0] == "python" else argv[0], *argv[1:]]
        proc = subprocess.run(
            cmd,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        return ProcessResult(
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


def test_view_file_reads_line_window_and_hints_remaining(tmp_path: Path):
    sample = tmp_path / "demo.py"
    sample.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 21)) + "\n", encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    result = ViewFileTool().execute(
        ToolCall("c1", "view_file", {"path": "demo.py", "start_line": 5, "end_line": 10}),
        runtime,
    )

    assert result.return_code == 0
    assert "[File: demo.py (lines 5-10 of 20)]" in result.stdout
    assert " 5 | line_5 = 5" in result.stdout
    assert "10 | line_10 = 10" in result.stdout
    assert "10 more lines remaining" in result.stdout


def test_view_file_caps_excessive_window_and_blocks_escape(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"row {i}" for i in range(1, 401)) + "\n", encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    capped = ViewFileTool().execute(
        ToolCall("c2", "view_file", {"path": "big.txt", "start_line": 1, "end_line": 9999}),
        runtime,
    )
    assert capped.return_code == 0
    assert f"(lines 1-{MAX_VIEW_WINDOW} of 400)" in capped.stdout

    escaped = ViewFileTool().execute(
        ToolCall("c3", "view_file", {"path": "../outside.txt"}),
        runtime,
    )
    assert escaped.return_code == 2
    assert "escapes workspace" in escaped.stderr


def test_view_file_rejects_inverted_range():
    runtime = LocalWorkspaceRuntime(Path("."))
    result = ViewFileTool().execute(
        ToolCall("c4", "view_file", {"path": "a.py", "start_line": 20, "end_line": 10}),
        runtime,
    )
    assert result.return_code is None
    assert result.error is not None and "end_line" in result.error


def test_search_repo_finds_matches_and_skips_noise_dirs(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def target_fn():\n    return 42\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.py").write_text("target_fn\n", encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    result = SearchRepoTool().execute(
        ToolCall("s1", "search_repo", {"pattern": "target_fn"}),
        runtime,
    )
    assert result.return_code == 0
    assert "pkg/mod.py:1: def target_fn():" in result.stdout
    assert "__pycache__" not in result.stdout


def test_edit_file_applies_unique_change_and_passes_ast_check(tmp_path: Path):
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    result = EditFileTool().execute(
        ToolCall(
            "e1",
            "edit_file",
            {"path": "calc.py", "old_str": "return a - b", "new_str": "return a + b"},
        ),
        runtime,
    )
    assert result.return_code == 0
    assert "AST check: PASS" in result.stdout
    assert target.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"


def test_edit_file_rejects_python_syntax_error_without_modifying_file(tmp_path: Path):
    target = tmp_path / "calc.py"
    original = "def add(a, b):\n    return a + b\n"
    target.write_text(original, encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    result = EditFileTool().execute(
        ToolCall(
            "e2",
            "edit_file",
            {"path": "calc.py", "old_str": "return a + b", "new_str": "return a + (b"},
        ),
        runtime,
    )
    assert result.return_code == 1
    assert "AST syntax guardrail" in result.stderr
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_rejects_ambiguous_or_missing_match(tmp_path: Path):
    target = tmp_path / "dup.py"
    target.write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    runtime = LocalWorkspaceRuntime(tmp_path)

    ambiguous = EditFileTool().execute(
        ToolCall("e3", "edit_file", {"path": "dup.py", "old_str": "x = 1", "new_str": "x = 2"}),
        runtime,
    )
    assert ambiguous.return_code == 1
    assert "matched 2 locations" in ambiguous.stderr

    missing = EditFileTool().execute(
        ToolCall("e4", "edit_file", {"path": "dup.py", "old_str": "z = 9", "new_str": "z = 0"}),
        runtime,
    )
    assert missing.return_code == 1
    assert "not found" in missing.stderr


def test_build_tool_registry_profiles():
    bash_reg = build_tool_registry("bash")
    assert [s.name for s in bash_reg.specs()] == ["bash"]

    aci_reg = build_tool_registry("aci")
    assert [s.name for s in aci_reg.specs()] == ["bash", "view_file", "search_repo", "edit_file"]

    with pytest.raises(ValueError, match="unsupported toolset"):
        build_tool_registry("unknown")
