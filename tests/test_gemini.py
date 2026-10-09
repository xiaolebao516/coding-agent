"""Offline contract checks for Gemini: no network and no Docker."""
import json

import pytest

from coding_agent.contracts import ToolCall
from coding_agent.gemini import GeminiModel
from coding_agent.tools.bash import BashTool
from coding_agent.tools.base import make_tool_spec


def test_function_call_roundtrip_preserves_signature_and_id(monkeypatch):
    model = GeminiModel(api_key="test-only", tool_specs=[make_tool_spec(BashTool())])
    raw_call = {
        "role": "model",
        "parts": [{
            "functionCall": {
                "name": "bash",
                "args": {"command": "printf SMOKE_OK"},
                "id": "call_1",
            },
            "thoughtSignature": "opaque-signature",
        }],
    }
    sent = []

    def fake_post(payload):
        sent.append(payload)
        if len(sent) == 1:
            return {
                "candidates": [{"content": raw_call}],
                "usageMetadata": {"promptTokenCount": 52, "candidatesTokenCount": 18},
            }
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}],
            "usageMetadata": {"promptTokenCount": 32, "candidatesTokenCount": 1},
        }

    monkeypatch.setattr(model, "_post", fake_post)
    first = model.generate([{"role": "user", "content": "Run the command"}])
    assert first.tool_call == ToolCall(
        id="call_1", name="bash", arguments={"command": "printf SMOKE_OK"}
    )
    history = [
        {"role": "user", "content": "Run the command"},
        {"role": "assistant", "tool_call": first.tool_call},
        {"role": "tool", "tool_call_id": "call_1", "return_code": 0,
         "stdout": "SMOKE_OK", "stderr": "", "timed_out": False, "error": None},
    ]
    final = model.generate(history)
    assert final.content == "done"
    assert sent[1]["contents"][1] == raw_call
    response = sent[1]["contents"][2]["parts"][0]["functionResponse"]
    assert response["id"] == "call_1"
    assert response["name"] == "bash"
    assert response["response"]["stdout"] == "SMOKE_OK"
    assert "test-only" not in json.dumps(sent)


def test_missing_gemini_tool_state_fails_loudly():
    model = GeminiModel(api_key="fake", tool_specs=[])
    with pytest.raises(ValueError, match="missing Gemini tool-call state"):
        model._messages([{"role": "assistant", "tool_call": ToolCall("lost", "bash", {})}])


def test_missing_usage_not_falsely_zero(monkeypatch):
    model = GeminiModel(api_key="fake", tool_specs=[])
    monkeypatch.setattr(model, "_post", lambda _: {
        "candidates": [{"content": {"parts": [{"text": "done"}]}}]
    })
    result = model.generate([{"role": "user", "content": "hi"}])
    assert result.usage.output_tokens is None
    assert result.usage.input_tokens is None
    assert result.usage.cost_usd is None


def test_output_limit_is_configurable(monkeypatch):
    model = GeminiModel(api_key="fake", tool_specs=[], max_output_tokens=128)
    seen = []
    monkeypatch.setattr(model, "_post", lambda payload: (
        seen.append(payload) or {
            "candidates": [{"content": {"parts": [{"text": "done"}]}}]
        }
    ))
    model.generate([{"role": "user", "content": "hi"}])
    assert seen[0]["generationConfig"]["maxOutputTokens"] == 128
    with pytest.raises(ValueError, match="positive"):
        GeminiModel(api_key="fake", tool_specs=[], max_output_tokens=0)
