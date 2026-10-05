import pytest

from coding_agent import trajectory
from coding_agent.trajectory import *
import json

def test_trajectory_recorder_happy_path():
    recorder = TrajectoryRecorder()

    recorder.start(
        task_id="task-1",
        max_steps=5,
        budget=1.0,
    )

    event = TrajectoryEvent(
        type="model_response",
        data={"content": "hello"},
    )

    recorder.record(event)

    trajectory = recorder.finalize(
        steps=1,
        total_usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        ),
        stop_reason=StopReason.MODEL_FINISHED,
        final_message="done",
    )

    assert trajectory.task_id == "task-1"
    assert trajectory.max_steps == 5
    assert trajectory.budget == 1.0

    assert len(trajectory.events) == 1
    assert trajectory.events[0] == event

    assert trajectory.steps == 1
    assert trajectory.stop_reason == StopReason.MODEL_FINISHED
    assert trajectory.final_message == "done"

    assert trajectory.run_id
    assert trajectory.started_at
    assert trajectory.ended_at
    assert trajectory.wall_time is not None
    assert trajectory.wall_time >= 0

    assert recorder.state == RecorderState.FINISHED

    data = trajectory.to_dict()

    assert data["task_id"] == "task-1"
    assert data["stop_reason"] == "model_finished"

    json_text = json.dumps(data)
    assert isinstance(json_text, str)


def test_record_before_start_raises():
    recorder = TrajectoryRecorder()

    event = TrajectoryEvent(
        type="model_response",
        data={"content": "hello"},
    )

    with pytest.raises(RuntimeError):
        recorder.record(event)


def test_finalize_before_start_raises():
    recorder = TrajectoryRecorder()

    with pytest.raises(RuntimeError):
        recorder.finalize(
            steps=1,
            total_usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
            ),
            stop_reason=StopReason.MODEL_FINISHED,
            final_message="done",
        )


def test_start_twice_raises():
    recorder = TrajectoryRecorder()

    recorder.start(
        task_id="task-1",
        max_steps=5,
        budget=1.0,
    )

    with pytest.raises(RuntimeError):
        recorder.start(
            task_id="task-2",
            max_steps=5,
            budget=1.0,
        )


def test_finished_recorder_rejects_operations():
    recorder = TrajectoryRecorder()

    recorder.start(
        task_id="task-1",
        max_steps=5,
        budget=1.0,
    )

    recorder.finalize(
        steps=1,
        total_usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        ),
        stop_reason=StopReason.MODEL_FINISHED,
        final_message="done",
    )

    with pytest.raises(RuntimeError):
        recorder.record(
            TrajectoryEvent(
                type="model_response",
                data={"content": "too late"},
            )
        )

    with pytest.raises(RuntimeError):
        recorder.finalize(
            steps=1,
            total_usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.01,
            ),
            stop_reason=StopReason.MODEL_FINISHED,
            final_message="done",
        )


