"""Groq provider adapter: no live API requests and no Docker required."""
import io
import json

import pytest

from coding_agent import cli, groq, groq_smoke
from coding_agent.contracts import ModelFinishReason, ModelResponse, ToolCall
from coding_agent.groq import GroqModel, parse_groq_response
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry


def sample(message, reason="stop"):
    return {
        "choices": [{"finish_reason": reason, "message": message}],
        "usage": {"prompt_tokens": 129, "completion_tokens": 54,
                  "total_tokens": 183,
                  "completion_tokens_details": {"reasoning_tokens": 23}},
    }


def test_real_single_tool_response_matches_user_verified_shape():
    raw = sample({
        "role": "assistant",
        "reasoning": "Call bash.",
        "tool_calls": [{
            "id": "fc_b09a",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": '{"command":"printf groq_tool_ok"}',
            },
        }],
    }, "tool_calls")
    response = parse_groq_response(raw)
    assert response.tool_call == ToolCall("fc_b09a", "bash", {"command": "printf groq_tool_ok"})
    assert response.usage.input_tokens == 129
    assert response.usage.output_tokens == 54
    assert response.usage.cost_usd is None  # Groq does not return dollar cost
    assert response.finish_reason == ModelFinishReason.FINISHED


def test_only_verified_free_plan_model_id_allowed():
    with pytest.raises(ValueError, match="verified model IDs"):
        GroqModel(api_key="fake", tool_specs=[], model="some/other-model")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqModel(api_key="", tool_specs=[])
    assert GroqModel(api_key="fake", tool_specs=[]).model == "openai/gpt-oss-120b"


def test_tool_payload_and_history_round_trip(monkeypatch):
    m = GroqModel(api_key="test-key", tool_specs=ToolRegistry([BashTool()]).specs())
    sent = []
    def fake_post(payload):
        sent.append(payload)
        if len(sent) == 1:
            return sample({"role": "assistant", "content": None, "tool_calls": [
                {"id": "fc1", "type": "function",
                 "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]}, "tool_calls")
        return sample({"role": "assistant", "content": "done"})
    monkeypatch.setattr(m, "_post", fake_post)
    first = m.generate([{"role":"user","content":"list files"}])
    assert first.tool_call.id == "fc1"
    assert sent[0]["parallel_tool_calls"] is False
    assert sent[0]["max_completion_tokens"] == 4096
    assert sent[0]["tools"][0]["function"]["name"] == "bash"
    assert "test-key" not in json.dumps(sent[0])
    final = m.generate([
        {"role":"user","content":"list files"},
        {"role":"assistant","tool_call":first.tool_call},
        {"role":"tool","tool_call_id":"fc1","return_code":0,
         "stdout":"x.py","stderr":"","timed_out":False,"error":None},
    ])
    assert final.content == "done"
    assert sent[1]["messages"][-1]["role"] == "tool"
    assert sent[1]["messages"][-1]["tool_call_id"] == "fc1"


@pytest.mark.parametrize("reason,finish", [
    ("length", ModelFinishReason.OUTPUT_TRUNCATED),
    ("content_filter", ModelFinishReason.PROVIDER_STOPPED),
    ("failed", ModelFinishReason.PROVIDER_STOPPED),
])
def test_truncation_and_stop_preserve_usage_without_tool_execution(reason,finish):
    out = parse_groq_response(sample({
        "content": "partial", "tool_calls": [
            {"id":"foo","function":{"name":"bash","arguments":"{}"}},
        ]}, reason))
    assert out.finish_reason == finish
    assert out.tool_call is None
    assert out.usage.input_tokens == 129


@pytest.mark.parametrize("arguments", ["{bad", "[]", "null"])
def test_invalid_tool_arguments_recoverable(arguments):
    out = parse_groq_response(sample({
        "tool_calls": [{"id":"c","function":{"name":"bash","arguments":arguments}}]},
        "tool_calls"))
    assert out.finish_reason == ModelFinishReason.INVALID_ACTION
    assert out.tool_call is None
    assert out.usage.output_tokens == 54


def test_unexpected_parallel_calls_rejected():
    call = {"id":"c","function":{"name":"bash","arguments":"{}"}}
    out = parse_groq_response(sample({"tool_calls":[call,call]}, "tool_calls"))
    assert out.finish_reason == ModelFinishReason.INVALID_ACTION
    assert out.tool_call is None


def test_groq_http_api_key_only_header_and_429_no_retry(monkeypatch):
    model = GroqModel(api_key="secret",tool_specs=[])
    sent = []
    def ok(request,timeout):
        sent.append((request,timeout))
        return io.BytesIO(json.dumps(sample({"content":"OK"})).encode())
    monkeypatch.setattr(groq,"urlopen",ok)
    assert model.generate([{"role":"user","content":"Hello"}]).content == "OK"
    request,timeout=sent[0]
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 120
    assert "secret" not in request.data.decode()

    sent.clear()
    def rate_limited(req,timeout):
        sent.append(1)
        raise OSError("simulated 429")
    monkeypatch.setattr(groq,"urlopen",rate_limited)
    with pytest.raises(OSError,match="429"):
        model.generate([{"role":"user","content":"Hello"}])
    assert len(sent)==1


class FakeRuntime:
    def start(self): pass
    def close(self): pass
    def run_process(self, argv):
        return ProcessResult(return_code=0,stdout="ok",stderr="")


def test_cli_groq_without_billing_or_network(monkeypatch,tmp_path):
    monkeypatch.setenv(cli.GROQ_API_KEY_ENV,"some-key")
    monkeypatch.setattr(cli,"DeepSeekBilling",lambda **kw: pytest.fail("Billing called"))
    kwargs=[]
    def fake_groq(**args):
        kwargs.append(args)
        m = FakeModel([ModelResponse("OK",None)])
        m.provider="groq"
        m.model=args["model"]
        return m
    monkeypatch.setattr(cli,"GroqModel",fake_groq)
    monkeypatch.setattr(cli,"DockerRuntime",lambda **kw:FakeRuntime())
    trajectory=tmp_path/"t.json"
    cli.main(["--provider","groq","--task-id","t","--problem","say hi",
              "--workspace",str(tmp_path),"--trajectory-path",str(trajectory)])
    assert kwargs[0]["model"]=="openai/gpt-oss-120b"
    assert kwargs[0]["api_key"]=="some-key"
    assert len(kwargs[0]["tool_specs"])==1
    result=json.loads(trajectory.read_text())
    assert result["provider"]=="groq"
    assert result["max_total_tokens"]==60000
    assert result["budget"] is None
    assert "some-key" not in trajectory.read_text()


def test_cli_missing_key_and_budget_fail_early(monkeypatch,tmp_path):
    monkeypatch.delenv(cli.GROQ_API_KEY_ENV,raising=False)
    with pytest.raises(SystemExit,match="GROQ_API_KEY"):
        cli.main(["--provider","groq","--task-id","t","--problem","p",
                  "--workspace",str(tmp_path)])
    monkeypatch.setenv(cli.GROQ_API_KEY_ENV,"fake")
    with pytest.raises(SystemExit,match="unavailable"):
        cli.main(["--provider","groq","--task-id","t","--problem","p",
                  "--workspace",str(tmp_path),"--budget","1"])


def test_safe_smoke_round_trip_uses_synthetic_result(monkeypatch,capsys):
    calls=[]
    class FakeModel:
        def __init__(self,**kwargs): pass
        def generate(self,history):
            calls.append(history)
            if len(calls)==1:
                return ModelResponse(None,ToolCall("t","bash",{"command":"printf groq_tool_ok"}))
            assert history[-1]["stdout"]=="groq_tool_ok"
            return ModelResponse("groq_tool_ok",None)
    monkeypatch.setenv("GROQ_API_KEY","fake")
    monkeypatch.setattr(groq_smoke,"GroqModel",FakeModel)
    groq_smoke.main()
    assert "PASS:" in capsys.readouterr().out
    assert len(calls)==2


def test_safe_smoke_rejects_unexpected_command(monkeypatch):
    class FakeModel:
        def __init__(self,**kwargs): pass
        def generate(self,history):
            return ModelResponse(None,ToolCall("t","bash",{"command":"rm -rf /"}))
    monkeypatch.setenv("GROQ_API_KEY","fake")
    monkeypatch.setattr(groq_smoke,"GroqModel",FakeModel)
    with pytest.raises(SystemExit,match="unexpected tool"):
        groq_smoke.main()



def test_smoke_displays_structured_403_reason_without_leaking_key(monkeypatch):
    from urllib.error import HTTPError

    class RejectingModel:
        def __init__(self, **kwargs): pass
        def generate(self, history):
            body = json.dumps({"error": {
                "type": "permissions_error",
                "code": "model_permission_blocked_project",
                "message": "Access denied for fake-secret",
            }}).encode()
            raise HTTPError(
                "https://api.groq.com/openai/v1/chat/completions",
                403, "Forbidden",
                {"Content-Type": "application/json"},
                io.BytesIO(body),
            )

    monkeypatch.setenv("GROQ_API_KEY", "fake-secret")
    monkeypatch.setattr(groq_smoke, "GroqModel", RejectingModel)
    with pytest.raises(SystemExit) as ex:
        groq_smoke.main()
    value = str(ex.value)
    assert "Groq HTTP 403" in value
    assert "model_permission_blocked_project" in value
    assert "fake-secret" not in value
    assert "[REDACTED]" in value


def test_smoke_html_403_does_not_dump_response_body(monkeypatch):
    from urllib.error import HTTPError

    class RejectingModel:
        def __init__(self, **kwargs): pass
        def generate(self, history):
            raise HTTPError(
                "https://api.groq.com/openai/v1/chat/completions",
                403, "Forbidden",
                {"Content-Type": "text/html", "Server": "cloudflare"},
                io.BytesIO(b"<html>secret diagnostic page</html>"),
            )

    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr(groq_smoke, "GroqModel", RejectingModel)
    with pytest.raises(SystemExit) as ex:
        groq_smoke.main()
    assert "non-JSON response" in str(ex.value)
    assert "server=cloudflare" in str(ex.value)
    assert "secret diagnostic page" not in str(ex.value)
