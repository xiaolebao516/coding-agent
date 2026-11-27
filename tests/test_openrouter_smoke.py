"""Smoke command is deterministic and never executes the generated command."""
import pytest

from coding_agent import openrouter_smoke
from coding_agent.contracts import ModelResponse, ToolCall, Usage


def test_live_smoke_happy_path_with_fake_model(monkeypatch, capsys):
    calls = []
    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def generate(self, history):
            if len(history) == 1:
                return ModelResponse(None, ToolCall("call1", "bash", {"command": "printf openrouter_tool_ok"}),
                                     usage=Usage(input_tokens=20, output_tokens=10, cost_usd=0.0))
            assert history[-1]["tool_call_id"] == "call1"
            assert history[-1]["stdout"] == openrouter_smoke.MARKER
            return ModelResponse("openrouter_tool_ok", None,
                                 usage=Usage(input_tokens=30, output_tokens=5, cost_usd=0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(openrouter_smoke, "OpenRouterModel", FakeModel)
    openrouter_smoke.main()
    assert calls[0]["api_key"] == "fake-key"
    assert "PASS: live tool call" in capsys.readouterr().out


def test_live_smoke_rejects_no_tool_call(monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs): pass
        def generate(self, history):
            return ModelResponse("just text", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(openrouter_smoke, "OpenRouterModel", FakeModel)
    with pytest.raises(SystemExit, match="did not emit"):
        openrouter_smoke.main()
