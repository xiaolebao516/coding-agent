import copy

from coding_agent.model import FakeModel
from coding_agent.agent import Agent
from coding_agent.contracts import (
    ModelResponse,
    StopReason,
    Task,
    ToolCall,
    Usage,
)
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry


class FakeRuntime:
    def run_process(self, argv: list[str]) -> ProcessResult:
        return ProcessResult(
            return_code=0,
            stdout="hello",
            stderr="",
        )

class FailingFakeRuntime:
    def run_process(self, argv):
        return ProcessResult(
            return_code=1,
            stdout="",
            stderr="2 tests failed",
        )


class RecordingFakeModel(FakeModel):
    def __init__(self, responses):
        super().__init__(responses)
        self.histories = []

    def generate(self, history):
        self.histories.append(copy.deepcopy(history))
        return super().generate(history)


def test_agent_happy_path():
    bash_call = ToolCall(
        id="call_1",
        name="bash",
        arguments={"command": "echo hello"},
    )

    model = FakeModel(
        [
            ModelResponse(
                content=None,
                tool_call=bash_call,
                usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.01),
            ),
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(input_tokens=150, output_tokens=30, cost_usd=None),
            ),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=5,
        budget=10.0,
    )

    task = Task(
        task_id="test-1",
        problem_statement="Say hello",
    )

    runtime = FakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )

    assert result.stop_reason == StopReason.MODEL_FINISHED
    assert result.final_message == "done"
    assert result.steps == 1
    assert result.usage.input_tokens == 250
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd is None


def test_agent_stops_at_max_steps():
    bash_call = ToolCall(
        id="call_2",
        name="bash",
        arguments={"command": "echo hello"},
    )

    model = FakeModel(
        [
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(0, 0, 0.0)),
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(0, 0, 0.0)),
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(0, 0, 0.0)),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=2,
        budget=10.0,
    )

    task = Task(
        task_id="test-2",
        problem_statement="Say hello",
    )

    runtime = FakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.steps == 2
    assert result.final_message is None


def test_agent_budget_control():
    bash_call = ToolCall(
        id="call_2",
        name="bash",
        arguments={"command": "echo hello"},
    )

    model = FakeModel(
        [
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(1000, 1000, 0.01)
            ),
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(1500, 1500, 0.015)
            ),
            ModelResponse(
                content=None,
                tool_call=bash_call,
            ),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=0.02,
    )

    task = Task(
        task_id="test-2",
        problem_statement="Say hello",
    )

    runtime = FakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )

    assert result.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert result.usage.cost_usd == 0.025
    assert result.steps == 2


def test_agent_stops_when_budget_becomes_unknown():
    bash_call = ToolCall(
        id="call_2",
        name="bash",
        arguments={"command": "echo hello"},
    )

    model = FakeModel(
        [
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(1000, 1000, 0.01)
            ),
            ModelResponse(
                content=None,
                tool_call=bash_call,
            ),
            ModelResponse(
                content=None,
                tool_call=bash_call,
            ),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=0.02,
    )

    task = Task(
        task_id="test-2",
        problem_statement="Say hello",
    )

    runtime = FakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )

    assert result.stop_reason == StopReason.BUDGET_UNKNOWN
    assert result.usage.cost_usd is None
    assert result.steps == 2


def test_agent_recovers_from_unknown_tool():
    bash_call = ToolCall(
        id="call_1",
        name="bash",
        arguments={"command": "echo hello"},
    )
    python_call = ToolCall(
        id="call_2",
        name="python",
        arguments={"command": "python Karina.py"},
    )

    model = RecordingFakeModel(
        [
            ModelResponse(
                content=None, tool_call=python_call, usage=Usage(1000, 1000, 0.01)
            ),
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(1000, 1000, 0.01)
            ),
            ModelResponse(
                content=None, tool_call=None, usage=Usage(1000, 1000, 0.01)
            ),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=0.02,
    )

    task = Task(
        task_id="test-2",
        problem_statement="Say hello",
    )

    runtime = FakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )
    second_history = model.histories[1]

    tool_message = second_history[-1]

    assert tool_message["role"] == "tool"   
    assert tool_message["return_code"] is None
    assert tool_message["error"] == "unknown tool: python"

def test_agent_feeds_command_failure_back_to_model(): 
    model = RecordingFakeModel(
        [
            ModelResponse(
                content=None, tool_call=ToolCall(
                    id="call_1",
                    name="bash",
                    arguments={"command":"pytest -q"}
                ), usage=Usage(1000, 1000, 0.01)
            ),
            ModelResponse(
                content="done",tool_call=None, usage=Usage(1000, 1000, 0.01)
            ),
        ]
    )

    registry = ToolRegistry(
        [
            BashTool(),
        ]
    )

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=1.0,
    )

    task = Task(
        task_id="test-feed-command-failure",
        problem_statement="pytest",
    )

    runtime = FailingFakeRuntime()

    result = agent.run(
        task=task,
        runtime=runtime,
    )
    second_history = model.histories[1]

    tool_message = second_history[-1]

    assert tool_message["role"] == "tool"
    assert tool_message["return_code"] == 1
    assert tool_message["stderr"] == "2 tests failed"
    assert tool_message["error"] is None

class FailingModel:
    def generate(self):
        raise RuntimeError("model down")


def test_agent_stops_on_model_error():
    model = FailingModel()

    registry = ToolRegistry([BashTool()])

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=1.0,
    )

    task = Task(
        task_id="test-model-error",
        problem_statement="anything",
    )

    runtime = FakeRuntime()

    result = agent.run(task=task, runtime=runtime)

    assert result.stop_reason == StopReason.MODEL_ERROR
    assert result.steps == 0
    assert result.final_message is None
    assert result.usage.cost_usd == 0.0


class FailingRuntime:
    def run_process(self, argv):
        raise RuntimeError("container unavailable")

def test_agent_stops_on_runtime_error():
    bash_call = ToolCall(
            id="call_1",
            name="bash",
            arguments={"command": "echo hello"},
        )
    model = FakeModel(
            [
                ModelResponse(
                    content=None, tool_call=bash_call, usage=Usage(1000, 1000, 0.01)
                ),
                ModelResponse(
                    content=None,
                    tool_call=bash_call,
                ),
                ModelResponse(
                    content=None,
                    tool_call=bash_call,
                ),
            ]
        )

    registry = ToolRegistry([BashTool()])

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=4,
        budget=1.0,
    )

    task = Task(
        task_id="test-model-error",
        problem_statement="anything",
    )

    runtime = FailingRuntime()

    result = agent.run(task=task, runtime=runtime)

    assert result.stop_reason == StopReason.RUNTIME_ERROR
    assert result.steps == 1
    assert result.final_message is None
