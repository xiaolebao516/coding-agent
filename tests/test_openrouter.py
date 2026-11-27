"""OpenRouter free API adapter: entirely offline tests."""
import io
import json

import pytest

from coding_agent import cli, openrouter
from coding_agent.agent import Agent
from coding_agent.contracts import ModelFinishReason, ModelResponse, StopReason, Task, ToolCall
from coding_agent.model import FakeModel
from coding_agent.openrouter import OpenRouterModel, parse_openrouter_response
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder
from coding_agent.runtime.base import ProcessResult


def raw(message, reason="stop", usage=None):
    return {"choices": [{"finish_reason": reason, "message": message}],
            "usage": usage or {"prompt_tokens": 18, "completion_tokens": 13,
                              "cost": 0, "prompt_tokens_details": {"cached_tokens": 0}}}


def test_user_verified_text_response_counts_zero_cost_and_reasoning():
    data = raw({"role": "assistant", "content": "OK", "reasoning": "OK because requested."})
    response = parse_openrouter_response(data)
    assert response.content == "OK" and response.tool_call is None
    assert response.usage.input_tokens == 18
    assert response.usage.output_tokens == 13
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cost_usd == 0.0


def test_free_model_lock_and_missing_usage():
    with pytest.raises(ValueError, match=":free"):
        OpenRouterModel(api_key="test", tool_specs=[], model="stepfun/step-5-preview")
    with pytest.raises(ValueError, match="nonzero"):
        parse_openrouter_response(raw({"content": "ok"}, usage={"cost": 0.12}))
    assert parse_openrouter_response({"choices": [], "usage": None}).finish_reason == ModelFinishReason.PROVIDER_STOPPED


def test_tool_call_round_trip_and_no_secret_in_payload(monkeypatch):
    model = OpenRouterModel(api_key="secret-value", tool_specs=ToolRegistry([BashTool()]).specs())
    sent = []
    def fake_post(payload):
        sent.append(payload)
        if len(sent) == 1:
            return raw({"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
                "tool_calls")
        return raw({"content": "done"}, "stop")

    monkeypatch.setattr(model, "_post", fake_post)
    response = model.generate([{"role": "user", "content": "list files"}])
    assert response.tool_call == ToolCall("call_1", "bash", {"command": "ls"})
    assert sent[0]["parallel_tool_calls"] is False
    assert sent[0]["tools"][0]["function"]["name"] == "bash"
    assert "secret-value" not in json.dumps(sent)

    reply = model.generate([
        {"role": "user", "content": "list files"},
        {"role": "assistant", "tool_call": response.tool_call},
        {"role": "tool", "tool_call_id": "call_1", "return_code": 0,
         "stdout": "a.py", "stderr": "", "timed_out": False, "error": None},
    ])
    assert reply.content == "done"
    messages = sent[1]["messages"]
    assert messages[-2]["tool_calls"][0]["id"] == "call_1"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"


@pytest.mark.parametrize("reason,expected", [
    ("length", ModelFinishReason.OUTPUT_TRUNCATED),
    ("content_filter", ModelFinishReason.PROVIDER_STOPPED),
    ("error", ModelFinishReason.PROVIDER_STOPPED),
])
def test_provider_stop_not_executed(reason, expected):
    response = parse_openrouter_response(raw({"content": "partial", "tool_calls": [
        {"id": "x", "function": {"name": "bash", "arguments": '{"command":"rm -rf /"}'}}]},
        reason=reason))
    assert response.tool_call is None
    assert response.finish_reason == expected
    assert response.usage.cost_usd == 0


@pytest.mark.parametrize("arg", ["{oops", "[]", "null"])
def test_invalid_tool_json_is_recoverable(arg):
    response = parse_openrouter_response(raw({"tool_calls": [
        {"id": "call1", "function": {"name": "bash", "arguments": arg}}]}, "tool_calls"))
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION
    assert response.tool_call is None
    assert response.usage.output_tokens == 13


def test_multiple_calls_rejected_even_if_provider_ignores_flag():
    call = {"id": "c", "function": {"name": "bash", "arguments": "{}"}}
    response = parse_openrouter_response(raw({"tool_calls": [call, call]}, "tool_calls"))
    assert response.finish_reason == ModelFinishReason.INVALID_ACTION


def test_http_uses_api_key_only_as_header_without_retry(monkeypatch):
    model = OpenRouterModel(api_key="secret-value", tool_specs=[])
    recorded = []
    def fake_urlopen(request, timeout):
        recorded.append((request, timeout))
        return io.BytesIO(json.dumps(raw({"content": "OK"})).encode())
    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen)
    result = model.generate([{"role": "user", "content": "hello"}])
    assert result.content == "OK"
    request, timeout = recorded[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret-value"
    assert "secret-value" not in request.data.decode()
    assert timeout == 120.0


def test_http_429_propagates_without_retry(monkeypatch):
    calls = []
    def reject(request, timeout):
        calls.append(1)
        raise OSError("simulated HTTP 429")
    monkeypatch.setattr(openrouter, "urlopen", reject)
    with pytest.raises(OSError, match="429"):
        OpenRouterModel(api_key="fake", tool_specs=[]).generate([{"role":"user","content":"x"}])
    assert len(calls) == 1


class FakeRuntime:
    def start(self): pass
    def close(self): pass
    def run_process(self, argv):
        return ProcessResult(return_code=0, stdout="hi", stderr="")


def test_cli_composes_openrouter_without_billing_or_actual_network(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.OPENROUTER_API_KEY_ENV, "secret-key")
    monkeypatch.setattr(cli, "DeepSeekBilling", lambda **k: pytest.fail("billing called"))
    captured = {}
    def fake_model(**kwargs):
        captured.update(kwargs)
        stub = FakeModel([ModelResponse(content="done", tool_call=None)])
        stub.provider = "openrouter"
        stub.model = kwargs["model"]
        return stub
    monkeypatch.setattr(cli, "OpenRouterModel", fake_model)
    monkeypatch.setattr(cli, "DockerRuntime", lambda **kwargs: FakeRuntime())
    path = tmp_path / "out.json"
    cli.main(["--provider", "openrouter", "--task-id", "smoke",
              "--problem", "say hi", "--workspace", str(tmp_path),
              "--trajectory-path", str(path), "--max-steps", "2"])
    assert captured["model"] == openrouter.DEFAULT_MODEL
    assert captured["api_key"] == "secret-key"
    assert len(captured["tool_specs"]) == 1
    data = json.loads(path.read_text())
    assert data["provider"] == "openrouter"
    assert data["budget"] is None
    assert data["max_total_tokens"] == 60000
    assert "secret-key" not in path.read_text()


def test_cli_rejects_paid_budget_and_missing_key_early(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.OPENROUTER_API_KEY_ENV, "fake")
    with pytest.raises(SystemExit, match="unavailable"):
        cli.main(["--provider","openrouter","--task-id","s","--problem","p",
                  "--workspace",str(tmp_path),"--budget","1"])
    monkeypatch.delenv(cli.OPENROUTER_API_KEY_ENV)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        cli.main(["--provider","openrouter","--task-id","s","--problem","p",
                  "--workspace",str(tmp_path)])


def test_agent_tracks_free_token_usage_and_tool_result():
    tool_call = ToolCall("a", "bash", {"command":"echo hi"})
    model = FakeModel([
        ModelResponse(None, tool_call, usage=openrouter.parse_openrouter_response(
            raw({"content": None})).usage),
        ModelResponse("done", None, usage=openrouter.parse_openrouter_response(
            raw({"content": "done"})).usage),
    ])
    model.provider = "openrouter"
    model.model = openrouter.DEFAULT_MODEL
    recorder = TrajectoryRecorder()
    result = Agent(model, ToolRegistry([BashTool()]), max_steps=3,
                   budget=None, max_total_tokens=100).run(
        Task("smoke", "echo hi"), FakeRuntime(), recorder)
    assert result.stop_reason == StopReason.MODEL_FINISHED
    assert result.usage.input_tokens == 36
    assert result.usage.output_tokens == 26
    assert result.usage.cost_usd == 0
    assert [e.type for e in recorder.trajectory.events] == [
        "model_response", "tool_result", "model_response", "terminal"
    ]
