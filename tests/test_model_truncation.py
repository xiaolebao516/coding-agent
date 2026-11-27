"""Contract tests for truncated provider responses and Agent termination.

Offline only. Provider tests are supplied boilerplate; the two Agent tests
are intentionally RED until the student implements the Agent core branch.
"""
import pytest

from coding_agent import deepseek
from coding_agent.agent import Agent
from coding_agent.contracts import (
    ModelFinishReason,
    ModelResponse,
    StopReason,
    Task,
    ToolCall,
    Usage,
)
from coding_agent.deepseek import parse_deepseek_response
from coding_agent.gemini import GeminiModel
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder


def test_finished_is_the_default_model_response_reason():
    response = ModelResponse(content="done", tool_call=None)
    assert response.finish_reason is ModelFinishReason.FINISHED


def test_deepseek_truncation_skips_incomplete_tool_json_but_keeps_usage(monkeypatch):
    monkeypatch.setattr(deepseek, "_cost_usd", lambda *args: 0.0025)
    raw = {
        "choices": [{
            "finish_reason": "length",
            "message": {
                "content": "partial response",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "printf',
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 120,
            "prompt_cache_hit_tokens": 20,
            "completion_tokens": 30,
        },
    }

    result = parse_deepseek_response(raw, "deepseek-flash")
    assert result.finish_reason is ModelFinishReason.OUTPUT_TRUNCATED
    assert result.content == "partial response"
    assert result.tool_call is None
    assert result.usage == Usage(
        input_tokens=120,
        cache_read_tokens=20,
        output_tokens=30,
        cost_usd=0.0025,
    )


def test_gemini_truncation_keeps_usage_and_does_not_save_partial_call(monkeypatch):
    model = GeminiModel(api_key="test", tool_specs=[])
    monkeypatch.setattr(model, "_post", lambda _payload: {
        "candidates": [{
            "finishReason": "MAX_TOKENS",
            "content": {
                "parts": [
                    {"text": "partial explanation"},
                    {"functionCall": {
                        "name": "bash", "args": {"command": "printf PARTIAL"}
                    }},
                ]
            },
        }],
        "usageMetadata": {
            "promptTokenCount": 24,
            "candidatesTokenCount": 17,
        },
    })

    result = model.generate([{"role": "user", "content": "test"}])
    assert result.finish_reason is ModelFinishReason.OUTPUT_TRUNCATED
    assert result.content == "partial explanation"
    assert result.tool_call is None
    assert result.usage.input_tokens == 24
    assert result.usage.output_tokens == 17
    assert model._pending_calls == {}


def test_gemini_truncation_without_content_keeps_usage(monkeypatch):
    model = GeminiModel(api_key="test", tool_specs=[])
    monkeypatch.setattr(model, "_post", lambda _payload: {
        "candidates": [{"finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    })
    result = model.generate([{"role": "user", "content": "test"}])
    assert result.finish_reason is ModelFinishReason.OUTPUT_TRUNCATED
    assert result.content is None
    assert result.tool_call is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


class NoToolRuntime:
    def run_process(self, argv):
        pytest.fail("truncated model output must never cause tool execution")


class RecordingRuntime:
    def __init__(self):
        self.calls = []

    def run_process(self, argv):
        self.calls.append(argv)
        return ProcessResult(return_code=0, stdout="OK", stderr="")


def _run_agent(responses, runtime, *, budget=None):
    model = FakeModel(responses)
    agent = Agent(
        model=model,
        tool_registry=ToolRegistry([BashTool()]),
        max_steps=5,
        budget=budget,
    )
    recorder = TrajectoryRecorder()
    result = agent.run(
        Task(task_id="truncation-test", problem_statement="diagnose"),
        runtime=runtime,
        recorder=recorder,
    )
    return result, recorder, model


def test_agent_truncation_never_executes_tool_and_records_diagnostic():
    # RED: implement this stop branch in Agent, not in the provider adapter.
    result, recorder, model = _run_agent([
        ModelResponse(
            content="partial response for diagnosis only",
            tool_call=ToolCall(
                id="call_1", name="bash",
                arguments={"command": "printf PARTIAL"},
            ),
            usage=Usage(
                input_tokens=50,
                cache_read_tokens=0,
                output_tokens=20,
                cost_usd=0.004,
            ),
            finish_reason=ModelFinishReason.OUTPUT_TRUNCATED,
        ),
    ], NoToolRuntime())

    assert result.stop_reason is StopReason.OUTPUT_TRUNCATED
    assert result.final_message is None
    assert result.steps == 0
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd == pytest.approx(0.004)
    assert model.index == 1

    trajectory = recorder.trajectory
    assert trajectory is not None
    assert trajectory.final_message is None
    assert [event.type for event in trajectory.events] == [
        "model_response", "terminal",
    ]
    assert trajectory.events[0].data["content"] == "partial response for diagnosis only"
    assert trajectory.events[0].data["finish_reason"] == "output_truncated"
    assert trajectory.events[-1].data["stop_reason"] == "output_truncated"


def test_agent_truncation_after_tool_accumulates_usage_and_stops():
    # RED: finish reason needs checking after merging usage, before dispatch.
    runtime = RecordingRuntime()
    result, recorder, model = _run_agent([
        ModelResponse(
            content=None,
            tool_call=ToolCall("first", "bash", {"command": "printf OK"}),
            usage=Usage(
                input_tokens=10, cache_read_tokens=0,
                output_tokens=5, cost_usd=0.01,
            ),
        ),
        ModelResponse(
            content="unfinished",
            tool_call=None,
            usage=Usage(
                input_tokens=15, cache_read_tokens=0,
                output_tokens=6, cost_usd=0.002,
            ),
            finish_reason=ModelFinishReason.OUTPUT_TRUNCATED,
        ),
    ], runtime, budget=1.0)

    assert result.stop_reason is StopReason.OUTPUT_TRUNCATED
    assert result.final_message is None
    assert result.steps == 1
    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 11
    assert result.usage.cost_usd == pytest.approx(0.012)
    assert len(runtime.calls) == 1
    assert model.index == 2
    assert [e.type for e in recorder.trajectory.events] == [
        "model_response", "tool_result", "model_response", "terminal",
    ]
