"""Magpie adapter tests: mocked HTTP, no live gateway or Docker."""

import io
import json

import pytest

from coding_agent import cli
from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall
from coding_agent.magpie import DEFAULT_MODEL, MagpieModel
from coding_agent.model import FakeModel
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry


def sample(message, reason="stop"):
    return {
        "choices": [{"finish_reason": reason, "message": message}],
        "usage": {"prompt_tokens": 27, "completion_tokens": 12},
    }


def test_verified_tool_call_parses_without_inventing_cost():
    model = MagpieModel(
        api_key="test-key",
        base_url="http://127.0.0.1:3425/v1",
        tool_specs=[],
    )
    model._post = lambda _: sample(
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "bash", "arguments": '{"command":"printf MAGPIE_TOOL_OK"}'
            }
        }]},
        "tool_calls",
    )
    result = model.generate([{"role": "user", "content": "Call bash"}])
    assert result.tool_call == ToolCall(
        "call-1", "bash", {"command": "printf MAGPIE_TOOL_OK"}
    )
    assert result.finish_reason == ModelFinishReason.FINISHED
    assert result.usage.input_tokens == 27
    assert result.usage.output_tokens == 12
    assert result.usage.cost_usd is None


def test_tool_messages_round_trip_and_model_payload():
    model = MagpieModel(
        api_key="secret",
        base_url="http://100.94.210.64:3425/v1",
        tool_specs=ToolRegistry([BashTool()]).specs(),
    )
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return sample({"role": "assistant", "tool_calls": [{
                "id": "t1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'
                }
            }]}, "tool_calls")
        return sample({"role": "assistant", "content": "done"})

    model._post = fake_post
    first = model.generate([{"role": "user", "content": "show working directory"}])
    final = model.generate([
        {"role": "user", "content": "show working directory"},
        {"role": "assistant", "tool_call": first.tool_call},
        {"role": "tool", "tool_call_id": first.tool_call.id,
         "return_code": 0, "stdout": "/workspace", "stderr": "",
         "timed_out": False, "error": None},
    ])
    assert first.tool_call.name == "bash"
    assert final.content == "done"
    assert payloads[0]["model"] == DEFAULT_MODEL
    assert payloads[0]["max_tokens"] == 8192
    assert payloads[0]["stream"] is False
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[0]["parallel_tool_calls"] is False
    assert payloads[0]["tools"][0]["function"]["name"] == "bash"
    assert payloads[1]["messages"][-1]["role"] == "tool"
    assert payloads[1]["messages"][-1]["tool_call_id"] == "t1"
    assert "secret" not in json.dumps(payloads)


def test_http_headers_and_no_proxy():
    model = MagpieModel(
        api_key="test-secret",
        base_url="http://100.94.210.64:3425/v1",
        tool_specs=[],
    )
    assert any(getattr(handler, "proxies", None) == {} for handler in model._opener.handlers)
    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(json.dumps(sample({"role": "assistant", "content": "OK"})).encode())

    model._opener = FakeOpener()
    assert model.generate([{"role": "user", "content": "Hi"}]).content == "OK"
    request, timeout = requests[0]
    assert request.full_url == "http://100.94.210.64:3425/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert request.get_method() == "POST"
    assert timeout == 120.0
    assert "test-secret" not in request.data.decode()


@pytest.mark.parametrize("url", ["", "not-a-url", "ftp://host/v1", "http://host",
                                    "http://user:pass@host/v1", "http://host/v1?token=x"])
def test_invalid_base_url_rejected(url):
    with pytest.raises(ValueError, match="MAGPIE_BASE_URL"):
        MagpieModel(api_key="key", base_url=url, tool_specs=[])


@pytest.mark.parametrize("reason,expected", [
    ("length", ModelFinishReason.OUTPUT_TRUNCATED),
    ("error", ModelFinishReason.PROVIDER_STOPPED),
])
def test_unsafe_finish_reasons_do_not_execute_tools(reason, expected):
    model = MagpieModel(api_key="key", base_url="http://127.0.0.1:3425/v1", tool_specs=[])
    model._post = lambda _: sample({"tool_calls": [{
        "id": "t1", "function": {"name": "bash", "arguments": "{}"}
    }]}, reason)
    result = model.generate([{"role": "user", "content": "run"}])
    assert result.tool_call is None
    assert result.finish_reason == expected


class FakeRuntime:
    def start(self):
        pass

    def close(self):
        pass


def test_cli_magpie_composes_without_billing_or_network(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.MAGPIE_API_KEY_ENV, "fake-gateway-token")
    monkeypatch.setenv(cli.MAGPIE_BASE_URL_ENV, "http://100.94.210.64:3425/v1")
    monkeypatch.setattr(cli, "DeepSeekBilling", lambda **_: pytest.fail("billing called"))
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        result = FakeModel([ModelResponse(content="OK", tool_call=None)])
        result.provider, result.model = "magpie", kwargs["model"]
        return result

    monkeypatch.setattr(cli, "MagpieModel", fake_model)
    monkeypatch.setattr(cli, "DockerRuntime", lambda **_: FakeRuntime())
    path = tmp_path / "trajectory.json"
    cli.main([
        "--provider", "magpie",
        "--task-id", "preflight",
        "--problem", "Say OK",
        "--workspace", str(tmp_path),
        "--trajectory-path", str(path),
    ])
    data = json.loads(path.read_text())
    assert captured["model"] == DEFAULT_MODEL
    assert captured["base_url"] == "http://100.94.210.64:3425/v1"
    assert captured["api_key"] == "fake-gateway-token"
    assert [t.name for t in captured["tool_specs"]] == ["bash"]
    assert data["provider"] == "magpie"
    assert data["budget"] is None
    assert data["max_total_tokens"] == 60000
    assert "fake-gateway-token" not in path.read_text()


def test_cli_magpie_missing_credentials_and_budget_fail_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DockerRuntime", lambda **_: pytest.fail("Docker should not start"))
    argv = ["--provider", "magpie", "--task-id", "x", "--problem", "x",
            "--workspace", str(tmp_path)]
    monkeypatch.delenv(cli.MAGPIE_API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit, match="MAGPIE_API_KEY"):
        cli.main(argv)
    monkeypatch.setenv(cli.MAGPIE_API_KEY_ENV, "test")
    monkeypatch.delenv(cli.MAGPIE_BASE_URL_ENV, raising=False)
    with pytest.raises(SystemExit, match="MAGPIE_BASE_URL"):
        cli.main(argv)
    with pytest.raises(SystemExit, match="unavailable"):
        cli.main(argv + ["--budget", "1"])
