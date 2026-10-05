import json

import pytest

from coding_agent.cli import run_once
from coding_agent.contracts import ModelResponse, StopReason, Task, Usage
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult


class LifecycleFakeRuntime:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def run_process(self, argv: list[str]) -> ProcessResult:
        return ProcessResult(
            return_code=0,
            stdout="hello",
            stderr="",
        )

    def close(self) -> None:
        self.closed = True


def test_run_once_zero_paid_e2e(tmp_path):
    model = FakeModel(
        [
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.0,
                ),
            )
        ]
    )

    runtime = LifecycleFakeRuntime()

    task = Task(
        task_id="e2e-1",
        problem_statement="Say hello",
    )

    trajectory_path = tmp_path / "trajectory.json"

    result = run_once(
        task=task,
        model=model,
        runtime=runtime,
        max_steps=5,
        budget=1.0,
        trajectory_path=str(trajectory_path),
    )

    assert result.stop_reason == StopReason.MODEL_FINISHED

    assert runtime.started is True
    assert runtime.closed is True

    assert trajectory_path.exists()

    with open(trajectory_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == "e2e-1"
    assert data["stop_reason"] == StopReason.MODEL_FINISHED.value

    assert [event["type"] for event in data["events"]] == [
        "model_response",
        "terminal",
    ]


def test_run_once_closes_runtime_when_saving_fails(tmp_path):
    model = FakeModel(
        [
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.0,
                ),
            )
        ]
    )

    runtime = LifecycleFakeRuntime()

    task = Task(
        task_id="cleanup-test",
        problem_statement="Say hello",
    )

    # 父目录不存在，所以 open() 会失败
    trajectory_path = tmp_path / "missing" / "trajectory.json"

    with pytest.raises(FileNotFoundError):
        run_once(
            task=task,
            model=model,
            runtime=runtime,
            max_steps=5,
            budget=1.0,
            trajectory_path=str(trajectory_path),
        )

    assert runtime.started is True
    assert runtime.closed is True
