import json

import pytest

from coding_agent import cli
from coding_agent.cli import API_KEY_ENV, main, run_once
from coding_agent.contracts import ModelResponse, StopReason, Task, Usage
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from types import SimpleNamespace

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
        registry=ToolRegistry([BashTool()]),
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
            registry=ToolRegistry([BashTool()]),
            runtime=runtime,
            max_steps=5,
            budget=1.0,
            trajectory_path=str(trajectory_path),
        )

    assert runtime.started is True
    assert runtime.closed is True


# ---------------------------------------------------------------------------
# main(): composition root
# ---------------------------------------------------------------------------


def _forbid(name):
    def factory(*args, **kwargs):
        raise AssertionError(f"{name} must not be constructed")

    return factory


def test_main_without_api_key_stops_before_any_work(monkeypatch, tmp_path):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr(cli, "DockerRuntime", _forbid("DockerRuntime"))
    monkeypatch.setattr(cli, "DeepSeekModel", _forbid("DeepSeekModel"))

    with pytest.raises(SystemExit):
        main(["--task-id", "t", "--problem", "p", "--workspace", str(tmp_path)])


def test_main_composes_real_dependencies_without_leaking_key(monkeypatch, tmp_path):
    secret = "sk-test-should-never-appear"
    monkeypatch.setenv(API_KEY_ENV, secret)

    captured = {}

    class StubDeepSeekModel(FakeModel):
        provider = "deepseek"
        model = "deepseek-flash"

    def fake_deepseek_model(**kwargs):
        captured["model_kwargs"] = kwargs
        return StubDeepSeekModel(
            [ModelResponse(content="done", tool_call=None, usage=Usage(cost_usd=0.0))]
        )

    def fake_docker_runtime(**kwargs):
        captured["runtime_kwargs"] = kwargs
        return LifecycleFakeRuntime()

    monkeypatch.setattr(cli, "DeepSeekModel", fake_deepseek_model)
    monkeypatch.setattr(cli, "DockerRuntime", fake_docker_runtime)
    monkeypatch.setattr(
        cli,
        "DeepSeekBilling",
        lambda **kwargs: SimpleNamespace(
            is_available=lambda: True,
        ),
    )
    trajectory_path = tmp_path / "trajectory.json"
    main(
        [
            "--task-id", "t1",
            "--problem", "say hi",
            "--workspace", str(tmp_path),
            "--trajectory-path", str(trajectory_path),
        ]
    )

    # the model is told about exactly the tools the agent can run
    assert [spec.name for spec in captured["model_kwargs"]["tool_specs"]] == ["bash"]
    assert captured["model_kwargs"]["api_key"] == secret

    raw = trajectory_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-flash"
    assert secret not in raw


def test_main_rejects_unavailable_balance(monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_ENV, "test-key")

    monkeypatch.setattr(
        cli,
        "DeepSeekBilling",
        lambda **kwargs: SimpleNamespace(
            is_available=lambda: False,
        ),
    )
    monkeypatch.setattr(cli, "DockerRuntime", _forbid("DockerRuntime"))

    with pytest.raises(SystemExit):
        main(
            [
                "--task-id",
                "smoke",
                "--problem",
                "test",
                "--workspace",
                str(tmp_path),
            ]
        )



def test_gemini_cli_skips_deepseek_billing_and_uses_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.GEMINI_API_KEY_ENV, "gemini-test-key")
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr(cli, "DeepSeekBilling", _forbid("DeepSeekBilling"))
    captured = {}

    def fake_gemini_model(**kwargs):
        captured.update(kwargs)
        model = FakeModel([
            ModelResponse(content="done", tool_call=None, usage=Usage(input_tokens=5))
        ])
        model.provider = "gemini"
        model.model = kwargs["model"]
        return model

    monkeypatch.setattr(cli, "GeminiModel", fake_gemini_model)
    monkeypatch.setattr(cli, "DockerRuntime", lambda **kwargs: LifecycleFakeRuntime())

    path = tmp_path / "gemini-trajectory.json"
    main([
        "--provider", "gemini", "--task-id", "gemini-smoke",
        "--problem", "say hi", "--workspace", str(tmp_path),
        "--max-steps", "2", "--max-output-tokens", "128",
        "--trajectory-path", str(path),
    ])
    assert captured["model"] == "gemini-3.8-flash"
    assert captured["max_output_tokens"] == 128
    assert captured["api_key"] == "gemini-test-key"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provider"] == "gemini"
    assert data["budget"] is None
    assert "gemini-test-key" not in path.read_text(encoding="utf-8")


def test_gemini_cli_missing_key_aborts_before_docker(monkeypatch, tmp_path):
    monkeypatch.delenv(cli.GEMINI_API_KEY_ENV, raising=False)
    monkeypatch.setattr(cli, "DockerRuntime", _forbid("DockerRuntime"))
    monkeypatch.setattr(cli, "DeepSeekBilling", _forbid("DeepSeekBilling"))
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        main([
            "--provider", "gemini", "--task-id", "t",
            "--problem", "p", "--workspace", str(tmp_path),
        ])


def test_cli_gemini_thinking_level_and_first_run_limits(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.GEMINI_API_KEY_ENV, "test-key")
    captured = {}

    def fake_gemini_model(**kwargs):
        captured.update(kwargs)
        model = FakeModel([
            ModelResponse(content="done", tool_call=None,
                          usage=Usage(input_tokens=15, output_tokens=10))
        ])
        model.provider = "gemini"
        model.model = kwargs["model"]
        return model

    monkeypatch.setattr(cli, "GeminiModel", fake_gemini_model)
    monkeypatch.setattr(cli, "DockerRuntime", lambda **kwargs: LifecycleFakeRuntime())
    trajectory_path = tmp_path / "first-run.json"
    cli.main([
        "--provider", "gemini", "--task-id", "first-run",
        "--problem", "Fix a test failure", "--workspace", str(tmp_path),
        "--model", "gemini-3.8-flash", "--thinking-level", "medium",
        "--max-steps", "10", "--max-output-tokens", "8192",
        "--max-total-tokens", "60000", "--model-timeout-seconds", "180",
        "--trajectory-path", str(trajectory_path),
    ])
    assert captured["thinking_level"] == "medium"
    assert captured["max_output_tokens"] == 8192
    assert captured["timeout"] == 180
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["max_steps"] == 10
    assert trajectory["max_total_tokens"] == 60000


def test_cli_rejects_thinking_level_for_deepseek_before_external_work(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.API_KEY_ENV, "test-key")
    monkeypatch.setattr(cli, "DockerRuntime", _forbid("DockerRuntime"))
    with pytest.raises(SystemExit, match="only supported by Gemini"):
        cli.main(["--task-id", "t", "--problem", "p", "--workspace", str(tmp_path),
                  "--thinking-level", "medium"])


def test_cli_rejects_invalid_gemini_request_timeout_before_network(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.GEMINI_API_KEY_ENV, "fake-key")
    monkeypatch.setattr(cli, "GeminiModel", _forbid("GeminiModel"))
    with pytest.raises(SystemExit, match="must be positive"):
        cli.main([
            "--provider", "gemini", "--task-id", "t", "--problem", "p",
            "--workspace", str(tmp_path), "--model-timeout-seconds", "0",
        ])
