import pytest
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
from coding_agent.trajectory import RecorderState, TrajectoryRecorder


def make_task(
    task_id: str = "test-task",
    problem_statement: str = "anything",
) -> Task:
    return Task(
        task_id=task_id,
        problem_statement=problem_statement,
    )


def make_registry() -> ToolRegistry:
    return ToolRegistry([BashTool()])


def make_agent(
    model,
    *,
    max_steps: int = 4,
    budget: float | None = 1.0,
) -> Agent:
    return Agent(
        model=model,
        tool_registry=make_registry(),
        max_steps=max_steps,
        budget=budget,
    )


def bash_call(
    command: str = "echo hello",
    call_id: str = "call_1",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="bash",
        arguments={"command": command},
    )

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
        recorder=TrajectoryRecorder(),
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
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)),
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)),
            ModelResponse(content=None, tool_call=bash_call, usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)),
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
        recorder=TrajectoryRecorder(),
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
                content=None, tool_call=bash_call, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
            ),
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(input_tokens=1500, output_tokens=1500, cost_usd=0.015)
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
        recorder=TrajectoryRecorder(),
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
                content=None, tool_call=bash_call, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
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
        recorder=TrajectoryRecorder(),
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
                content=None, tool_call=python_call, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
            ),
            ModelResponse(
                content=None, tool_call=bash_call, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
            ),
            ModelResponse(
                content=None, tool_call=None, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
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
        recorder=TrajectoryRecorder(),
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
                ), usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
            ),
            ModelResponse(
                content="done",tool_call=None, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
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
        recorder=TrajectoryRecorder(),
    )
    second_history = model.histories[1]

    tool_message = second_history[-1]

    assert tool_message["role"] == "tool"
    assert tool_message["return_code"] == 1
    assert tool_message["stderr"] == "2 tests failed"
    assert tool_message["error"] is None

class FailingModel:
    provider = "fake"
    model = "fake"

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

    result = agent.run(
        task=task,
        runtime=runtime,
        recorder=TrajectoryRecorder(),
    )

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
                    content=None, tool_call=bash_call, usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01)
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

    result = agent.run(
        task=task,
        runtime=runtime,
        recorder=TrajectoryRecorder(),
    )

    assert result.stop_reason == StopReason.RUNTIME_ERROR
    assert result.steps == 1
    assert result.final_message is None


def test_agent_trajectory_start_and_finalize():
    model = FakeModel(
        [
            ModelResponse(
                content=None,
                tool_call=bash_call(),
                usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01),
            ),
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.01),
            ),
        ]
    )

    agent = make_agent(model)
    recorder = TrajectoryRecorder()

    result = agent.run(
        task=make_task(),
        runtime=FakeRuntime(),
        recorder=recorder,
    )

    trajectory = recorder.trajectory

    assert trajectory is not None
    assert trajectory.steps == result.steps
    assert trajectory.total_usage == result.usage
    assert trajectory.stop_reason == result.stop_reason
    assert trajectory.final_message == result.final_message
    assert recorder.state == RecorderState.FINISHED
    event_types = [event.type for event in trajectory.events]

    assert event_types == [
        "model_response",
        "tool_result",
        "model_response",
        "terminal",
    ]


def test_agent_records_ordered_trajectory_events():
    call = bash_call()

    model = FakeModel(
        [
            ModelResponse(
                content=None,
                tool_call=call,
                usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.01),
            ),
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(input_tokens=50, output_tokens=10, cost_usd=0.01),
            ),
        ]
    )

    agent = make_agent(model)
    recorder = TrajectoryRecorder()

    result = agent.run(
        task=make_task(),
        runtime=FakeRuntime(),
        recorder=recorder,
    )

    trajectory = recorder.trajectory
    assert trajectory is not None

    event_types = [event.type for event in trajectory.events]

    assert event_types == [
        "model_response",
        "tool_result",
        "model_response",
        "terminal",
    ]

    assert trajectory.stop_reason == result.stop_reason
    assert trajectory.steps == result.steps
    assert trajectory.total_usage == result.usage


def test_agent_records_unknown_tool_in_trajectory():
    unknown_call = ToolCall(
        id="unknown-1",
        name="python",
        arguments={"command": "python test.py"},
    )

    model = FakeModel(
        [
            ModelResponse(
                content=None,
                tool_call=unknown_call,
                usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.01),
            ),
            ModelResponse(
                content="done",
                tool_call=None,
                usage=Usage(input_tokens=50, output_tokens=10, cost_usd=0.01),
            ),
        ]
    )

    agent = make_agent(model)
    recorder = TrajectoryRecorder()

    result = agent.run(
        task=make_task(),
        runtime=FakeRuntime(),
        recorder=recorder,
    )

    trajectory = recorder.trajectory
    assert trajectory is not None

    assert [event.type for event in trajectory.events] == [
        "model_response",
        "tool_result",
        "model_response",
        "terminal",
    ]

    tool_event = trajectory.events[1]

    assert tool_event.data["return_code"] is None
    assert tool_event.data["error"] == "unknown tool: python"

    assert trajectory.events[-1].data["stop_reason"] == result.stop_reason.value


def test_agent_records_runtime_error_in_trajectory():
    model = FakeModel(
        [
            ModelResponse(
                content=None,
                tool_call=bash_call(),
                usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.01),
            ),
        ]
    )

    agent = make_agent(model)
    recorder = TrajectoryRecorder()

    result = agent.run(
        task=make_task(),
        runtime=FailingRuntime(),
        recorder=recorder,
    )

    trajectory = recorder.trajectory
    assert trajectory is not None

    assert [event.type for event in trajectory.events] == [
        "model_response",
        "runtime_error",
        "terminal",
    ]

    error_event = trajectory.events[1]

    assert error_event.data["error_type"] == "RuntimeError"
    assert error_event.data["message"] == "container unavailable"

    assert result.stop_reason == StopReason.RUNTIME_ERROR
    assert trajectory.stop_reason == StopReason.RUNTIME_ERROR


def test_merge_usage_sums_cache_read_tokens():
    from coding_agent.agent import _merge_usage

    total = _merge_usage(
        Usage(input_tokens=100, cache_read_tokens=80, output_tokens=10, cost_usd=0.01),
        Usage(input_tokens=200, cache_read_tokens=150, output_tokens=20, cost_usd=0.02),
    )

    assert total.input_tokens == 300
    assert total.cache_read_tokens == 230
    assert total.output_tokens == 30
    assert total.cost_usd == pytest.approx(0.03)


def test_merge_usage_unknown_cache_tokens_stay_unknown():
    from coding_agent.agent import _merge_usage

    total = _merge_usage(
        Usage(input_tokens=100, cache_read_tokens=0, output_tokens=10, cost_usd=0.0),
        Usage(input_tokens=100, cache_read_tokens=None, output_tokens=10, cost_usd=0.0),
    )

    assert total.cache_read_tokens is None


def test_usage_rejects_positional_arguments():
    with pytest.raises(TypeError):
        Usage(100, 20, 0.01)
