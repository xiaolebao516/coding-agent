"""Controlled Agent-Computer Interface (ACI) tools inspired by SWE-agent.

All tools execute inside the configured Runtime sandbox via ``python -c`` so
they never touch the host filesystem directly and remain testable with
in-memory or local-subprocess FakeRuntime implementations.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from coding_agent.contracts import ToolCall, ToolResult
from coding_agent.runtime.base import Runtime

MAX_VIEW_WINDOW = 250
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_LINE_CHARS = 200


def _invalid_args_result(tool_call_id: str, exc: ValidationError) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call_id,
        return_code=None,
        stdout="",
        stderr="",
        error=f"invalid arguments: {exc}",
    )


def _run_python_script(
    tool_call_id: str,
    runtime: Runtime,
    script: str,
    payload: dict[str, Any],
) -> ToolResult:
    result = runtime.run_process(
        ["python", "-c", script, json.dumps(payload, ensure_ascii=False)]
    )
    return ToolResult(
        tool_call_id=tool_call_id,
        return_code=result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# 1. view_file: windowed, line-numbered file viewer
# ---------------------------------------------------------------------------

_VIEW_FILE_SCRIPT = r"""
import json, sys
from pathlib import Path

args = json.loads(sys.argv[1])
root = Path.cwd().resolve()
target = (root / args["path"]).resolve()
if root != target and root not in target.parents:
    print(f"error: path escapes workspace: {args['path']}", file=sys.stderr)
    sys.exit(2)
if not target.exists():
    print(f"error: file not found: {args['path']}", file=sys.stderr)
    sys.exit(1)
if target.is_dir():
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    print(f"[Directory: {args['path']} ({len(entries)} entries)]")
    for entry in entries[:200]:
        print(entry)
    if len(entries) > 200:
        print(f"...[{len(entries) - 200} more entries truncated]")
    sys.exit(0)

try:
    text = target.read_text(encoding="utf-8")
except UnicodeDecodeError:
    print(f"error: binary or non-UTF-8 file: {args['path']}", file=sys.stderr)
    sys.exit(1)

lines = text.splitlines()
total = len(lines)
start = args["start_line"]
end = min(args["end_line"], total)
if total == 0:
    print(f"[File: {args['path']} (empty file, 0 lines)]")
    sys.exit(0)
if start > total:
    print(
        f"error: start_line {start} exceeds file length ({total} lines)",
        file=sys.stderr,
    )
    sys.exit(1)

width = len(str(end))
print(f"[File: {args['path']} (lines {start}-{end} of {total})]")
for idx in range(start, end + 1):
    print(f"{idx:>{width}} | {lines[idx - 1]}")

remaining = total - end
if remaining > 0:
    next_start = end + 1
    next_end = min(total, next_start + 119)
    print(
        f"[{remaining} more lines remaining. Call view_file(path={args['path']!r}, "
        f"start_line={next_start}, end_line={next_end}) to continue.]"
    )
"""


class ViewFileArgs(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path to file or directory inside workspace.")
    start_line: int = Field(1, ge=1, description="1-based inclusive starting line number.")
    end_line: int = Field(120, ge=1, description="1-based inclusive ending line number (max window 250 lines).")

    @model_validator(mode="after")
    def _validate_window(self) -> "ViewFileArgs":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_line - self.start_line + 1 > MAX_VIEW_WINDOW:
            self.end_line = self.start_line + MAX_VIEW_WINDOW - 1
        return self


class ViewFileTool:
    name = "view_file"
    description = (
        "Read a bounded line window of a file with line numbers (max 250 lines per call), "
        "or list entries if path is a directory."
    )
    args_model = ViewFileArgs

    def execute(self, tool_call: ToolCall, runtime: Runtime) -> ToolResult:
        try:
            args = ViewFileArgs.model_validate(tool_call.arguments)
        except ValidationError as exc:
            return _invalid_args_result(tool_call.id, exc)
        return _run_python_script(
            tool_call.id,
            runtime,
            _VIEW_FILE_SCRIPT,
            args.model_dump(),
        )


# ---------------------------------------------------------------------------
# 2. search_repo: bounded code search with automatic noise filtering
# ---------------------------------------------------------------------------

_SEARCH_REPO_SCRIPT = r"""
import json, os, re, sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}
args = json.loads(sys.argv[1])
root = Path.cwd().resolve()
base = (root / args["path"]).resolve()
if root != base and root not in base.parents:
    print(f"error: path escapes workspace: {args['path']}", file=sys.stderr)
    sys.exit(2)
if not base.exists():
    print(f"error: search path not found: {args['path']}", file=sys.stderr)
    sys.exit(1)

pattern = args["pattern"]
use_regex = args["regex"]
max_results = args["max_results"]
max_line_chars = args["max_line_chars"]

if use_regex:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        print(f"error: invalid regex pattern: {exc}", file=sys.stderr)
        sys.exit(1)
else:
    compiled = None

def iter_files(target: Path):
    if target.is_file():
        yield target
        return
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fname in sorted(filenames):
            yield Path(dirpath) / fname

matches = []
truncated = False
for fpath in iter_files(base):
    try:
        raw = fpath.read_bytes()
    except OSError:
        continue
    if b"\x00" in raw[:4096]:
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        continue
    rel = fpath.relative_to(root).as_posix()
    for lineno, line in enumerate(text.splitlines(), start=1):
        hit = bool(compiled.search(line)) if compiled is not None else (pattern in line)
        if hit:
            snippet = line if len(line) <= max_line_chars else line[:max_line_chars] + "..."
            matches.append(f"{rel}:{lineno}: {snippet}")
            if len(matches) >= max_results:
                truncated = True
                break
    if truncated:
        break

if not matches:
    print(f"[No matches found for {pattern!r} under {args['path']}]")
    sys.exit(0)

for item in matches:
    print(item)
if truncated:
    print(f"[Truncated: reached max_results={max_results}. Narrow path or pattern for more.]")
"""


class SearchRepoArgs(BaseModel):
    pattern: str = Field(..., min_length=1, description="Substring or regex pattern to search for.")
    path: str = Field(".", min_length=1, description="Relative file or directory path to search under.")
    max_results: int = Field(30, ge=1, le=MAX_SEARCH_RESULTS, description="Maximum matching lines to return.")
    regex: bool = Field(False, description="Interpret pattern as a Python regular expression.")


class SearchRepoTool:
    name = "search_repo"
    description = (
        "Search repository files for a substring or regex, returning file:line matches "
        "while skipping .git/__pycache__/node_modules."
    )
    args_model = SearchRepoArgs

    def execute(self, tool_call: ToolCall, runtime: Runtime) -> ToolResult:
        try:
            args = SearchRepoArgs.model_validate(tool_call.arguments)
        except ValidationError as exc:
            return _invalid_args_result(tool_call.id, exc)
        payload = args.model_dump()
        payload["max_line_chars"] = MAX_SEARCH_LINE_CHARS
        return _run_python_script(
            tool_call.id,
            runtime,
            _SEARCH_REPO_SCRIPT,
            payload,
        )


# ---------------------------------------------------------------------------
# 3. edit_file: unique search-and-replace with Python AST syntax guardrail
# ---------------------------------------------------------------------------

_EDIT_FILE_SCRIPT = r"""
import ast, json, sys
from pathlib import Path

args = json.loads(sys.argv[1])
root = Path.cwd().resolve()
target = (root / args["path"]).resolve()
if root != target and root not in target.parents:
    print(f"error: path escapes workspace: {args['path']}", file=sys.stderr)
    sys.exit(2)
if not target.exists() or target.is_dir():
    print(f"error: file not found: {args['path']}", file=sys.stderr)
    sys.exit(1)

try:
    original = target.read_text(encoding="utf-8")
except UnicodeDecodeError:
    print(f"error: binary or non-UTF-8 file: {args['path']}", file=sys.stderr)
    sys.exit(1)

old_str = args["old_str"]
new_str = args["new_str"]

count = original.count(old_str)
if count == 0:
    print(
        f"error: old_str not found in {args['path']}. "
        "Use view_file to verify exact indentation and lines before editing.",
        file=sys.stderr,
    )
    sys.exit(1)
if count > 1:
    positions = []
    start_idx = 0
    while True:
        idx = original.find(old_str, start_idx)
        if idx == -1:
            break
        lineno = original.count("\n", 0, idx) + 1
        positions.append(str(lineno))
        start_idx = idx + len(old_str)
    print(
        f"error: old_str matched {count} locations in {args['path']} "
        f"(starting lines: {', '.join(positions)}). Provide more surrounding context for a unique match.",
        file=sys.stderr,
    )
    sys.exit(1)

match_pos = original.find(old_str)
start_line = original.count("\n", 0, match_pos) + 1
updated = original.replace(old_str, new_str, 1)

if target.suffix == ".py":
    try:
        ast.parse(updated, filename=args["path"])
    except SyntaxError as exc:
        err_line = exc.lineno or "?"
        err_col = exc.offset or "?"
        err_text = (exc.text or "").rstrip("\n")
        print(
            f"error: edit rejected by Python AST syntax guardrail in {args['path']} "
            f"(line {err_line}, col {err_col}): {exc.msg}\n"
            f"  offending line: {err_text}\n"
            "Original file was NOT modified. Fix the syntax and retry edit_file.",
            file=sys.stderr,
        )
        sys.exit(1)

target.write_text(updated, encoding="utf-8")

updated_lines = updated.splitlines()
new_span = max(1, len(new_str.splitlines()))
preview_start = max(1, start_line - 2)
preview_end = min(len(updated_lines), start_line + new_span + 2)
width = len(str(max(1, preview_end)))
print(f"[Edited {args['path']} around line {start_line}; AST check: {'PASS' if target.suffix == '.py' else 'N/A'}]")
for idx in range(preview_start, preview_end + 1):
    print(f"{idx:>{width}} | {updated_lines[idx - 1]}")
"""


class EditFileArgs(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path to the file to edit.")
    old_str: str = Field(..., min_length=1, description="Exact unique substring to replace.")
    new_str: str = Field(..., description="Replacement text.")

    @model_validator(mode="after")
    def _validate_change(self) -> "EditFileArgs":
        if self.old_str == self.new_str:
            raise ValueError("new_str must differ from old_str")
        return self


class EditFileTool:
    name = "edit_file"
    description = (
        "Replace an exact unique substring in a file. For .py files, validates syntax via "
        "ast.parse before writing and rejects edits that introduce syntax errors."
    )
    args_model = EditFileArgs

    def execute(self, tool_call: ToolCall, runtime: Runtime) -> ToolResult:
        try:
            args = EditFileArgs.model_validate(tool_call.arguments)
        except ValidationError as exc:
            return _invalid_args_result(tool_call.id, exc)
        return _run_python_script(
            tool_call.id,
            runtime,
            _EDIT_FILE_SCRIPT,
            args.model_dump(),
        )
