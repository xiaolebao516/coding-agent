import json

import pytest

from coding_agent import deepseek
from coding_agent.contracts import ToolCall, Usage
from coding_agent.deepseek import (
    DeepSeekModel,
    parse_deepseek_response,
    to_deepseek_messages,
    to_deepseek_tools,
    usage_from_deepseek,
)
from coding_agent.tools.base import make_tool_spec
from coding_agent.tools.bash import BashArgs, BashTool


def test_make_tool_spec_from_bash_tool():
    spec = make_tool_spec(BashTool())

    assert spec.name == "bash"
    assert spec.description == "Execute a shell command."
    assert spec.parameters == BashArgs.model_json_schema()


def test_to_deepseek_tools():
    spec = make_tool_spec(BashTool())

    tools = to_deepseek_tools([spec])

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command.",
                "parameters": BashArgs.model_json_schema(),
            },
        }
    ]


def test_to_deepseek_messages():
    history = [
        {
            "role": "user",
            "content": "list files",
        },
        {
            "role": "assistant",
            "tool_call": ToolCall(
                id="call_1",
                name="bash",
                arguments={"command": "ls"},
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "return_code": 0,
            "stdout": "a.py\nb.py\n",
            "stderr": "",
            "timed_out": False,
            "error": None,
        },
    ]

    messages = to_deepseek_messages(history)

    assert messages[0] == {
        "role": "user",
        "content": "list files",
    }

    assistant = messages[1]

    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"

    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }

    tool = messages[2]

    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"

    observation = json.loads(tool["content"])

    assert observation == {
        "return_code": 0,
        "stdout": "a.py\nb.py\n",
        "stderr": "",
        "timed_out": False,
        "error": None,
    }


# ---------------------------------------------------------------------------
# usage_from_deepseek  —— 练习：这 5 个测试就是规格
# ---------------------------------------------------------------------------

MODEL = "deepseek-flash"

RAW_USAGE = {
    "prompt_tokens": 1000,
    "prompt_cache_hit_tokens": 800,
    "prompt_cache_miss_tokens": 200,
    "completion_tokens": 50,
    "total_tokens": 1050,
}


def test_usage_maps_fields_and_cost(monkeypatch):
    calls = []

    def fake_cost(model, input_tokens, cache_read_tokens, output_tokens):
        calls.append((model, input_tokens, cache_read_tokens, output_tokens))
        return 0.5

    monkeypatch.setattr(deepseek, "_cost_usd", fake_cost)

    usage = usage_from_deepseek(RAW_USAGE, MODEL)

    assert usage == Usage(
        input_tokens=1000,
        cache_read_tokens=800,
        output_tokens=50,
        cost_usd=0.5,
    )
    assert calls == [(MODEL, 1000, 800, 50)]


def test_usage_missing_usage_is_all_unknown(monkeypatch):
    monkeypatch.setattr(deepseek, "_cost_usd", lambda *a: pytest.fail("no cost"))

    assert usage_from_deepseek(None, MODEL) == Usage()


def test_usage_missing_cache_hit_tokens_makes_cost_unknown(monkeypatch):
    monkeypatch.setattr(deepseek, "_cost_usd", lambda *a: pytest.fail("no cost"))
    raw = {k: v for k, v in RAW_USAGE.items() if k != "prompt_cache_hit_tokens"}

    usage = usage_from_deepseek(raw, MODEL)

    assert usage == Usage(
        input_tokens=1000,
        cache_read_tokens=None,
        output_tokens=50,
        cost_usd=None,
    )


def test_usage_cost_error_keeps_tokens(monkeypatch):
    def broken_cost(*args):
        raise RuntimeError("price table exploded")

    monkeypatch.setattr(deepseek, "_cost_usd", broken_cost)

    usage = usage_from_deepseek(RAW_USAGE, MODEL)

    assert usage == Usage(
        input_tokens=1000,
        cache_read_tokens=800,
        output_tokens=50,
        cost_usd=None,
    )


def test_usage_non_positive_cost_is_unknown(monkeypatch):
    monkeypatch.setattr(deepseek, "_cost_usd", lambda *a: 0.0)

    assert usage_from_deepseek(RAW_USAGE, MODEL).cost_usd is None


# ---------------------------------------------------------------------------
# parse_deepseek_response
# ---------------------------------------------------------------------------


def _raw(message):
    return {"choices": [{"message": message}], "usage": RAW_USAGE}


@pytest.fixture
def stub_usage(monkeypatch):
    monkeypatch.setattr(deepseek, "usage_from_deepseek", lambda raw, model: Usage())


def test_parse_final_answer(stub_usage):
    response = parse_deepseek_response(
        _raw({"role": "assistant", "content": "done"}), MODEL
    )

    assert response.content == "done"
    assert response.tool_call is None


def test_parse_single_tool_call(stub_usage):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
    }

    response = parse_deepseek_response(_raw(message), MODEL)

    assert response.tool_call == ToolCall(
        id="call_1", name="bash", arguments={"command": "ls"}
    )


def test_parse_rejects_multiple_tool_calls(stub_usage):
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }
    message = {"role": "assistant", "content": None, "tool_calls": [call, call]}

    response = parse_deepseek_response(_raw(message), MODEL)
    assert response.tool_call is None
    assert response.finish_reason == deepseek.ModelFinishReason.PROVIDER_STOPPED
    assert response.stop_detail == "deepseek:unsupported_tool_calls"


# ---------------------------------------------------------------------------
# DeepSeekModel
# ---------------------------------------------------------------------------


def test_model_rejects_model_without_price_entry():
    with pytest.raises(ValueError):
        DeepSeekModel(api_key="test", tool_specs=[], model="no-such-model")


def test_generate_sends_payload_and_parses(monkeypatch, stub_usage):
    model = DeepSeekModel(
        api_key="secret-key", tool_specs=[make_tool_spec(BashTool())]
    )
    sent = []

    def fake_post(payload):
        sent.append(payload)
        return _raw({"role": "assistant", "content": "hi"})

    monkeypatch.setattr(model, "_post", fake_post)

    response = model.generate([{"role": "user", "content": "say hi"}])

    assert response.content == "hi"
    payload = sent[0]
    assert payload["model"] == "deepseek-flash"
    assert payload["messages"] == [{"role": "user", "content": "say hi"}]
    assert payload["tools"][0]["function"]["name"] == "bash"
    assert "secret-key" not in json.dumps(payload)


def test_real_price_table_charges_cache_hits_less():
    """Uses the installed LiteLLM price table (no network, no API call)."""
    all_hit = deepseek._cost_usd(MODEL, 1_000_000, 1_000_000, 0)
    all_miss = deepseek._cost_usd(MODEL, 1_000_000, 0, 0)

    assert 0 < all_hit < all_miss
