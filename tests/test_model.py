from coding_agent.contracts import ModelResponse, ToolCall
from coding_agent.model import FakeModel


def test_fake_model_returns_responses_in_order():
    bash_call = ToolCall(
        id="bash_id",
        name="bash",
        arguments={"command": ""},
    )

    fake_model = FakeModel([
        ModelResponse(
            content=None,
            tool_call=bash_call,
        ),
        ModelResponse(
            content="done",
            tool_call=None,
        ),
    ])

    assert fake_model.generate(history="") == ModelResponse(
        content=None,
        tool_call=bash_call,
    )

    assert fake_model.generate(history="") == ModelResponse(
        content="done",
        tool_call=None,
    )