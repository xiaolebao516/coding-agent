"""Offline unit tests for M5 ablation profiles, analyzer, anti-tampering & fault tolerance."""

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


def test_ablation_profiles_isolate_each_variable():
    assert list(ABLATION_PROFILES.keys()) == [
        "baseline",
        "aci",
        "aci_per_step",
        "aci_epoch",
        "aci_epoch_remind",
    ]
    assert get_profile("baseline").toolset == "bash"
    assert get_profile("aci_per_step").enable_reminders is False
    assert get_profile("aci_epoch").enable_reminders is False
    assert get_profile("aci_epoch_remind").enable_reminders is True

    with pytest.raises(ValueError, match="unknown ablation profile"):
        get_profile("nonexistent")


def test_summarize_records_distinguishes_unknown_cache_from_zero(tmp_path: Path):
    records = [
        {
            "profile": "baseline",
            "task_id": "t1",
            "resolved": False,
            "steps": 10,
            "usage": {"input_tokens": 10000, "cache_read_tokens": None, "output_tokens": 500, "cost_usd": None},
            "middleware_stats": {"prefix_invalidations": 0},
            "invalid_action_events": 0,
            "stop_reason": "max_steps",
        },
        {
            "profile": "aci_epoch",
            "task_id": "t1",
            "resolved": True,
            "steps": 6,
            "usage": {"input_tokens": 8000, "cache_read_tokens": 5600, "output_tokens": 400, "cost_usd": 0.012},
            "middleware_stats": {"prefix_invalidations": 1},
            "invalid_action_events": 0,
            "stop_reason": "model_finished",
        },
    ]
    for rec in records:
        fname = f"{rec['profile']}__{rec['task_id']}.eval.json"
        (tmp_path / fname).write_text(json.dumps(rec), encoding="utf-8")

    summaries = summarize_records(load_eval_records(tmp_path))
    by_name = {s.profile: s for s in summaries}

    assert by_name["baseline"].cache_hit_rate is None
    assert by_name["aci_epoch"].cache_hit_rate == pytest.approx(0.7)

    table = render_markdown_table(summaries)
    assert "| `baseline` | 1 | 0/1 (0.0%) | 0 | 10.00 | 10,000 | N/A | 0.00 | N/A |" in table
    assert "70.0%" in table


def test_run_ablation_task_blocks_test_tampering_false_pass(tmp_path: Path):
    source_ws = tmp_path / "source_repo"
    (source_ws / "tests").mkdir(parents=True)
    (source_ws / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (source_ws / "tests" / "test_app.py").write_text(
        "import app\nassert app.add(2, 3) == 5\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    task_spec = EvalTaskSpec(
        task_id="tamper-test",
        problem_statement="Fix add(a, b)",
        workspace=source_ws,
        verify_command="python3 tests/test_app.py",
    )

    def cheating_model_factory(_specs):
        # Agent tries to cheat by weakening the test file instead of fixing app.py
        return FakeModel([
            ModelResponse(
                content=None,
                tool_call=ToolCall(
                    "call_cheat",
                    "edit_file",
                    {
                        "path": "tests/test_app.py",
                        "old_str": "assert app.add(2, 3) == 5",
                        "new_str": "assert True",
                    },
                ),
                usage=Usage(input_tokens=100, cache_read_tokens=0, output_tokens=20, cost_usd=0.001),
            ),
            ModelResponse(
                content="Done",
                tool_call=None,
                usage=Usage(input_tokens=120, cache_read_tokens=60, output_tokens=10, cost_usd=0.001),
            ),
        ])

    record = run_ablation_task(
        task_spec=task_spec,
        profile=get_profile("aci"),
        model_factory=cheating_model_factory,
        runtime_factory=lambda ws, _img: LocalVerifyRuntime(ws),
        output_dir=out_dir,
        max_steps=5,
        budget=0.5,
    )

    assert record["resolved"] is False
    assert record["tampered_test_files"] == ["tests/test_app.py"]


def test_run_ablation_task_persists_record_on_runner_exception(tmp_path: Path):
    source_ws = tmp_path / "source_repo"
    source_ws.mkdir()
    out_dir = tmp_path / "out"

    class CrashingRuntime:
        def start(self) -> None:
            raise RuntimeError("docker daemon unavailable")

        def run_process(self, argv: list[str]) -> ProcessResult:
            raise AssertionError("unreachable")

        def close(self) -> None:
            pass

    task_spec = EvalTaskSpec(
        task_id="crash-1",
        problem_statement="Fix bug",
        workspace=source_ws,
        verify_command="true",
    )

    record = run_ablation_task(
        task_spec=task_spec,
        profile=get_profile("baseline"),
        model_factory=lambda _specs: FakeModel([]),
        runtime_factory=lambda _ws, _img: CrashingRuntime(),
        output_dir=out_dir,
    )

    assert record["resolved"] is False
    assert record["stop_reason"] == "eval_runner_error"
    assert "docker daemon unavailable" in record["runner_error"]
    assert (out_dir / "baseline__crash-1.eval.json").exists()
