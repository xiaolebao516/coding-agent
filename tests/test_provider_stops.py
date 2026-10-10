"""Provider rejection and malformed-response paths: offline, no Docker or API calls."""
import pytest

from coding_agent import deepseek
from coding_agent.agent import Agent
from coding_agent.contracts import (
    ModelFinishReason, ModelResponse, StopReason, Task, ToolCall, Usage,
)
from coding_agent.deepseek import parse_deepseek_response
from coding_agent.gemini import GeminiModel
from coding_agent.model import FakeModel
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder


@pytest.fixture
def ds_usage(monkeypatch):
    monkeypatch.setattr(
        deepseek, "usage_from_deepseek",
        lambda raw, model: Usage(input_tokens=24, output_tokens=8, cost_usd=0.002),
    )


@pytest.mark.parametrize("reason", [
    "content_filter", "aborted", "insufficient_system_resource", "new_reason",
])
def test_deepseek_failed_finish_keeps_usage_and_does_not_parse_tool(ds_usage, reason):
    raw = {"usage": {}, "choices": [{
        "finish_reason": reason,
        "message": {"content": "partial", "tool_calls": [{
            "id": "danger", "function": {"name": "bash", "arguments": "{broken"}
        }]},
    }]}
    response = parse_deepseek_response(raw, "deepseek-flash")
    assert response.finish_reason == ModelFinishReason.PROVIDER_STOPPED
    assert response.stop_detail == f"deepseek:{reason}"
    assert response.tool_call is None
    assert response.usage.cost_usd == 0.002


@pytest.mark.parametrize("arguments", ["{broken", "[]"])
def test_deepseek_invalid_tool_arguments_keep_usage(ds_usage, arguments):
    raw = {"usage": {}, "choices": [{
        "finish_reason": "tool_calls",
        "message": {"tool_calls": [{
            "id": "call1",
            "function": {"name": "bash", "arguments": arguments},
        }]},
    }]}
    response = parse_deepseek_response(raw, "deepseek-flash")
    assert response.finish_reason == ModelFinishReason.PROVIDER_STOPPED
    assert response.stop_detail == "deepseek:invalid_tool_call"
    assert response.tool_call is None
    assert response.usage.input_tokens == 24


def test_deepseek_empty_choices_preserve_usage(ds_usage):
    response = parse_deepseek_response({"choices": [], "usage": {}}, "deepseek-flash")
    assert response.finish_reason == ModelFinishReason.PROVIDER_STOPPED
    assert response.stop_detail == "deepseek:missing_choice"
    assert response.usage.output_tokens == 8


def test_deepseek_parallel_tool_calls_preserve_usage(ds_usage):
    call = {"id": "x", "function": {"name": "bash", "arguments": "{}"}}
    raw = {"usage": {}, "choices": [{
        "finish_reason": "tool_calls",
        "message": {"tool_calls": [call, call]},
    }]}
    response = parse_deepseek_response(raw, "deepseek-flash")
    assert response.stop_detail == "deepseek:unsupported_tool_calls"
    assert response.tool_call is None
    assert response.usage.input_tokens == 24


@pytest.mark.parametrize("raw,detail", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "gemini:SAFETY"),
    ({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "content": {
        "parts": [{"functionCall": {"id": "bad"}}]}}]}, "gemini:MALFORMED_FUNCTION_CALL"),
    ({"candidates": [{"finishReason": "SAFETY", "content": {
        "parts": [{"functionCall": {"id": "bad"}}]}}]}, "gemini:SAFETY"),
])
def test_gemini_provider_stops_keep_usage(monkeypatch, raw, detail):
    model = GeminiModel(api_key="dummy", tool_specs=[])
    raw["usageMetadata"] = {"promptTokenCount": 80, "candidatesTokenCount": 12}
    monkeypatch.setattr(model, "_post", lambda payload: raw)
    response = model.generate([{"role": "user", "content": "hi"}])
    assert response.finish_reason == ModelFinishReason.PROVIDER_STOPPED
    assert response.stop_detail == detail
    assert response.tool_call is None
    assert response.usage.input_tokens == 80
    assert response.usage.output_tokens == 12


def test_gemini_parallel_calls_keep_usage(monkeypatch):
    model = GeminiModel(api_key="dummy", tool_specs=[])
    call = {"functionCall": {"name": "bash", "id": "a", "args": {}}}
    monkeypatch.setattr(model, "_post", lambda payload: {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [call, call]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
    })
    response = model.generate([{"role": "user", "content": "hi"}])
    assert response.stop_detail == "gemini:unsupported_tool_calls"
    assert response.tool_call is None
    assert response.usage.output_tokens == 4


class NoExecution:
    def run_process(self, argv):
        raise AssertionError("provider-stopped call must never execute")


def test_agent_provider_stop_is_not_success_and_preserves_billing():
    model = FakeModel([ModelResponse(
        content="partial answer",
        tool_call=None,
        usage=Usage(input_tokens=10, output_tokens=3, cost_usd=0.01),
        finish_reason=ModelFinishReason.PROVIDER_STOPPED,
        stop_detail="deepseek:content_filter",
    )])
    recorder = TrajectoryRecorder()
    result = Agent(model, ToolRegistry([BashTool()]), max_steps=3, budget=1.0).run(
        Task("reject-1", "run something"), NoExecution(), recorder,
    )
    assert result.stop_reason == StopReason.PROVIDER_STOPPED
    assert result.final_message is None
    assert result.usage.input_tokens == 10
    assert result.usage.cost_usd == pytest.approx(0.01)
    assert result.steps == 0
    trajectory = recorder.trajectory
    assert trajectory is not None
    assert [e.type for e in trajectory.events] == ["model_response", "terminal"]
    assert trajectory.events[0].data["stop_detail"] == "deepseek:content_filter"
    assert trajectory.stop_reason == StopReason.PROVIDER_STOPPED
