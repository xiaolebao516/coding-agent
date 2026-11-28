"""Offline unit tests for M5 ablation profiles, analyzer, and runner isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from coding_agent.contracts import ModelResponse, ToolCall, Usage
from coding_agent.eval.analyze import (
    load_eval_records,
    render_markdown_table,
    summarize_records,
)
from coding_agent.eval.profiles import ABLATION_PROFILES, get_profile
from coding_agent.eval.runner import EvalTaskSpec, run_ablation_task
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult


class LocalVerifyRuntime:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

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


def test_ablation_profiles_cover_expected_evolution():
    assert list(ABLATION_PROFILES.keys()) == [
        "baseline",
        "aci",
        "aci_per_step",
        "aci_epoch",
    ]
    assert get_profile("baseline").toolset == "bash"
    assert get_profile("aci_epoch").context_strategy == "epoch_masking"
    assert get_profile("aci_epoch").enable_reminders is True

    with pytest.raises(ValueError, match="unknown ablation profile"):
        get_profile("nonexistent")


def test_summarize_records_and_render_markdown_table(tmp_path: Path):
    records = [
        {
            "profile": "baseline",
            "task_id": "t1",
            "resolved": False,
            "steps": 12,
            "usage": {"input_tokens": 20000, "cache_read_tokens": 12000, "output_tokens": 1000, "cost_usd": 0.04},
            "middleware_stats": {"prefix_invalidations": 0},
            "invalid_action_events": 1,
        },
        {
            "profile": "baseline",
            "task_id": "t2",
            "resolved": True,
            "steps": 8,
            "usage": {"input_tokens": 10000, "cache_read_tokens": 6000, "output_tokens": 600, "cost_usd": 0.02},
            "middleware_stats": {"prefix_invalidations": 0},
            "invalid_action_events": 0,
        },
        {
            "profile": "aci_epoch",
            "task_id": "t1",
            "resolved": True,
            "steps": 7,
            "usage": {"input_tokens": 9000, "cache_read_tokens": 6300, "output_tokens": 500, "cost_usd": 0.015},
            "middleware_stats": {"prefix_invalidations": 1},
            "invalid_action_events": 0,
        },
        {
            "profile": "aci_epoch",
            "task_id": "t2",
            "resolved": True,
            "steps": 5,
            "usage": {"input_tokens": 7000, "cache_read_tokens": 4900, "output_tokens": 400, "cost_usd": 0.011},
            "middleware_stats": {"prefix_invalidations": 1},
            "invalid_action_events": 0,
        },
    ]
    for rec in records:
        fname = f"{rec['profile']}__{rec['task_id']}.eval.json"
        (tmp_path / fname).write_text(json.dumps(rec), encoding="utf-8")

    loaded = load_eval_records(tmp_path)
    summaries = summarize_records(loaded)
    by_name = {s.profile: s for s in summaries}

    assert by_name["baseline"].resolved_rate == pytest.approx(0.5)
    assert by_name["baseline"].avg_steps == pytest.approx(10.0)
    assert by_name["baseline"].cache_hit_rate == pytest.approx(0.6)
    assert by_name["aci_epoch"].resolved_rate == pytest.approx(1.0)
    assert by_name["aci_epoch"].cache_hit_rate == pytest.approx(0.7)
    assert by_name["aci_epoch"].avg_cost_usd == pytest.approx(0.013)

    table = render_markdown_table(summaries)
    assert "| `baseline` |" in table
    assert "| `aci_epoch` |" in table
    assert "70.0%" in table


def test_run_ablation_task_isolates_workspace_and_verifies_fix(tmp_path: Path):
    source_ws = tmp_path / "source_repo"
    source_ws.mkdir()
    (source_ws / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    task_spec = EvalTaskSpec(
        task_id="add-bug",
        problem_statement="Fix add(a, b) in app.py",
        workspace=source_ws,
        verify_command="python3 -c 'import app; assert app.add(2, 3) == 5'",
    )

    def fake_model_factory(specs):
        assert any(s.name == "edit_file" for s in specs)
        return FakeModel([
            ModelResponse(
                content=None,
                tool_call=ToolCall(
                    "call_1",
                    "edit_file",
                    {"path": "app.py", "old_str": "return a - b", "new_str": "return a + b"},
                ),
                usage=Usage(input_tokens=100, cache_read_tokens=50, output_tokens=20, cost_usd=0.001),
            ),
            ModelResponse(
                content="Fixed add()",
                tool_call=None,
                usage=Usage(input_tokens=120, cache_read_tokens=80, output_tokens=10, cost_usd=0.001),
            ),
        ])

    record = run_ablation_task(
        task_spec=task_spec,
        profile=get_profile("aci_epoch"),
        model_factory=fake_model_factory,
        runtime_factory=lambda ws, _img: LocalVerifyRuntime(ws),
        output_dir=out_dir,
        max_steps=5,
        budget=0.5,
    )

    assert record["resolved"] is True
    assert record["steps"] == 1
    assert (out_dir / "aci_epoch__add-bug.eval.json").exists()
    assert (out_dir / "aci_epoch__add-bug.trajectory.json").exists()
    # Original source workspace must remain untouched for subsequent profiles
    assert (source_ws / "app.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a - b\n"
