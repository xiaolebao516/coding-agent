"""Run-level token limiter contract. Offline FakeModel tests.

Agent aggregates provider usage before checking whether it can do more work.
"""
import pytest

from coding_agent import cli
from coding_agent.agent import Agent
from coding_agent.contracts import ModelResponse, StopReason, Task, ToolCall, Usage
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder


class RecordingRuntime:
    def __init__(self):
        self.commands = []

    def run_process(self, argv):
        self.commands.append(argv)
        return ProcessResult(return_code=0, stdout="OK", stderr="")


def run_agent(responses, *, cap, budget=None):
    model = FakeModel(responses)
    runtime = RecordingRuntime()
    recorder = TrajectoryRecorder()
    agent = Agent(
        model=model,
        tool_registry=ToolRegistry([BashTool()]),
        max_steps=5,
        budget=budget,
        max_total_tokens=cap,
    )
    result = agent.run(
        task=Task("token-cap", "Test a token cap"),
        runtime=runtime,
        recorder=recorder,
    )
    return result, model, runtime, recorder


def tool(call_id):
    return ToolCall(call_id, "bash", {"command": "printf OK"})


def test_cap_is_run_cumulative_and_stops_before_second_tool():
    # 8+2 first call, 17+5 second call => 32 >= cap 30.
    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content=None, tool_call=tool("one"),
            usage=Usage(input_tokens=8, cache_read_tokens=0, output_tokens=2),
        ),
        ModelResponse(
            content=None, tool_call=tool("two"),
            usage=Usage(input_tokens=17, cache_read_tokens=0, output_tokens=5),
        ),
    ], cap=30)

    assert result.stop_reason is StopReason.TOKEN_LIMIT_EXHAUSTED
    assert result.final_message is None
    assert result.steps == 1
    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 7
    assert len(runtime.commands) == 1
    assert model.index == 2
    assert recorder.trajectory.max_total_tokens == 30
    assert recorder.trajectory.to_dict()["stop_reason"] == "token_limit_exhausted"
    assert [e.type for e in recorder.trajectory.events] == [
        "model_response", "tool_result", "model_response", "terminal",
    ]


def test_unknown_token_usage_fails_closed_when_cap_is_active():
    # No output-token count means the total cannot be known.
    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content=None, tool_call=tool("one"),
            usage=Usage(input_tokens=100, output_tokens=None, cost_usd=None),
        ),
    ], cap=1000)
    assert result.stop_reason is StopReason.TOKEN_USAGE_UNKNOWN
    assert result.final_message is None
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens is None
    assert model.index == 1
    assert not runtime.commands
    assert recorder.trajectory.events[-1].data["stop_reason"] == "token_usage_unknown"


def test_cache_hit_tokens_are_not_counted_twice():
    # This fails if cache_read_tokens are counted in addition to input_tokens.
    # First call costs 100 prompt + 10 output = 110 tokens, NOT 200.
    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content=None, tool_call=tool("one"),
            usage=Usage(input_tokens=100, cache_read_tokens=90, output_tokens=10),
        ),
        ModelResponse(
            content="done", tool_call=None,
            usage=Usage(input_tokens=5, cache_read_tokens=0, output_tokens=5),
        ),
    ], cap=150)
    assert result.stop_reason is StopReason.MODEL_FINISHED
    assert result.final_message == "done"
    assert result.usage.input_tokens == 105
    assert result.usage.output_tokens == 15
    assert len(runtime.commands) == 1
    assert model.index == 2


def test_no_cap_allows_unknown_monetary_cost():
    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content="done", tool_call=None,
            usage=Usage(input_tokens=5, output_tokens=2, cost_usd=None),
        ),
    ], cap=None, budget=None)
    assert result.stop_reason is StopReason.MODEL_FINISHED
    assert result.usage.cost_usd is None
    assert recorder.trajectory.max_total_tokens is None


def test_complete_final_response_is_valid_even_if_last_call_exceeds_soft_cap():
    # A soft cap prevents FUTURE work; it does not invalidate a finished answer.
    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content="done", tool_call=None,
            usage=Usage(input_tokens=100, output_tokens=20, cost_usd=None),
        ),
    ], cap=50)
    assert result.stop_reason is StopReason.MODEL_FINISHED
    assert result.final_message == "done"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert model.index == 1


def test_invalid_token_limit_rejected_before_a_run():
    with pytest.raises(ValueError, match="max_total_tokens"):
        Agent(
            model=FakeModel([]),
            tool_registry=ToolRegistry([BashTool()]),
            max_steps=3, budget=None, max_total_tokens=0,
        )


def test_explicit_gemini_usd_budget_is_rejected_before_any_work(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DockerRuntime", lambda **kwargs: pytest.fail("no Docker"))
    with pytest.raises(SystemExit, match="--budget"):
        cli.main([
            "--provider", "gemini",
            "--budget", "1",
            "--task-id", "t",
            "--problem", "p",
            "--workspace", str(tmp_path),
        ])


def test_output_truncation_takes_priority_over_unknown_or_exceeded_tokens():
    from coding_agent.contracts import ModelFinishReason

    result, model, runtime, recorder = run_agent([
        ModelResponse(
            content="partial",
            tool_call=tool("never"),
            usage=Usage(input_tokens=500, output_tokens=None),
            finish_reason=ModelFinishReason.OUTPUT_TRUNCATED,
        ),
    ], cap=50)
    assert result.stop_reason is StopReason.OUTPUT_TRUNCATED
    assert result.usage.input_tokens == 500
    assert result.final_message is None
    assert model.index == 1
    assert not runtime.commands


def test_unknown_usage_in_final_response_does_not_fake_token_total():
    result, model, runtime, recorder = run_agent([
        ModelResponse(content="done", tool_call=None,
                      usage=Usage(input_tokens=10, output_tokens=None)),
    ], cap=50)
    assert result.stop_reason is StopReason.TOKEN_USAGE_UNKNOWN
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens is None
    assert not runtime.commands
