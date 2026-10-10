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
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION
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
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION
    assert response.stop_detail == "deepseek:unsupported_tool_calls"
    assert response.tool_call is None
    assert response.usage.input_tokens == 24


@pytest.mark.parametrize("raw,detail", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "gemini:SAFETY"),
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
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION
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


# ---------------------------------------------------------------------------
# Recoverable invalid actions (model format errors) vs provider stops
# ---------------------------------------------------------------------------


def test_gemini_malformed_function_call_is_recoverable(monkeypatch):
    model = GeminiModel(api_key="dummy", tool_specs=[])
    monkeypatch.setattr(model, "_post", lambda payload: {
        "candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "content": {
            "parts": [{"functionCall": {"id": "bad"}}]}}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
    })
    response = model.generate([{"role": "user", "content": "hi"}])
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION
    assert response.stop_detail == "gemini:MALFORMED_FUNCTION_CALL"
    assert response.tool_call is None
    assert response.usage.input_tokens == 7


def test_gemini_merges_consecutive_user_turns(monkeypatch):
    model = GeminiModel(api_key="dummy", tool_specs=[])
    sent = []

    def fake_post(payload):
        sent.append(payload)
        return {"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": "ok"}]}}]}

    monkeypatch.setattr(model, "_post", fake_post)
    model.generate([
        {"role": "user", "content": "task"},
        {"role": "user", "content": "feedback"},
    ])
    contents = sent[0]["contents"]
    assert [c["role"] for c in contents] == ["user"]
    assert [p["text"] for p in contents[0]["parts"]] == ["task", "feedback"]


class HistoryRecordingModel(FakeModel):
    def __init__(self, responses):
        super().__init__(responses)
        self.seen = []

    def generate(self, history):
        self.seen.append([dict(item) for item in history])
        return super().generate(history)


INVALID = ModelResponse(
    content=None,
    tool_call=None,
    usage=Usage(input_tokens=10, output_tokens=2, cost_usd=0.001),
    finish_reason=ModelFinishReason.INVALID_ACTION,
    stop_detail="deepseek:invalid_tool_call",
)


def test_agent_feeds_invalid_action_back_and_continues():
    model = HistoryRecordingModel([
        INVALID,
        ModelResponse(content="done", tool_call=None,
                      usage=Usage(input_tokens=12, output_tokens=1, cost_usd=0.001)),
    ])
    result = Agent(model, ToolRegistry([BashTool()]), max_steps=5, budget=1.0).run(
        Task("invalid-1", "do it"), NoExecution(), TrajectoryRecorder(),
    )
    assert result.stop_reason == StopReason.MODEL_FINISHED
    assert result.final_message == "done"
    assert result.steps == 1
    assert result.usage.input_tokens == 22  # billed usage of the invalid call kept
    second_call_history = model.seen[1]
    assert second_call_history[-1]["role"] == "user"
    assert "deepseek:invalid_tool_call" in second_call_history[-1]["content"]


def test_agent_repeated_invalid_actions_are_bounded_by_max_steps():
    model = FakeModel([INVALID, INVALID, INVALID])
    result = Agent(model, ToolRegistry([BashTool()]), max_steps=3, budget=1.0).run(
        Task("invalid-2", "do it"), NoExecution(), TrajectoryRecorder(),
    )
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.steps == 3
    assert result.usage.input_tokens == 30


def test_invalid_action_cannot_bypass_cumulative_token_cap():
    """An invalid action has no ToolCall, but it would request another model turn."""
    model = FakeModel([ModelResponse(
        content=None,
        tool_call=None,
        usage=Usage(input_tokens=32, output_tokens=9, cost_usd=None),
        finish_reason=ModelFinishReason.INVALID_ACTION,
        stop_detail="gemini:MALFORMED_FUNCTION_CALL",
    )])
    runtime = NoExecution()
    recorder = TrajectoryRecorder()
    result = Agent(
        model, ToolRegistry([BashTool()]),
        max_steps=5, budget=None, max_total_tokens=40,
    ).run(Task("cap-invalid", "do it"), runtime, recorder)
    assert result.stop_reason == StopReason.TOKEN_LIMIT_EXHAUSTED
    assert result.steps == 0
    assert result.usage.input_tokens == 32
    assert model.index == 1
    assert [e.type for e in recorder.trajectory.events] == ["model_response", "terminal"]


def test_invalid_action_with_missing_usage_does_not_retry_under_token_cap():
    model = FakeModel([ModelResponse(
        content=None,
        tool_call=None,
        usage=Usage(input_tokens=None, output_tokens=9, cost_usd=None),
        finish_reason=ModelFinishReason.INVALID_ACTION,
        stop_detail="deepseek:invalid_tool_call",
    )])
    result = Agent(
        model, ToolRegistry([BashTool()]),
        max_steps=5, budget=None, max_total_tokens=40,
    ).run(Task("cap-unknown", "do it"), NoExecution(), TrajectoryRecorder())
    assert result.stop_reason == StopReason.TOKEN_USAGE_UNKNOWN
    assert result.steps == 0
    assert model.index == 1

