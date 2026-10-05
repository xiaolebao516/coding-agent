from dataclasses import dataclass, field, asdict
from typing import Any

from coding_agent.contracts import StopReason, Usage
from enum import Enum
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

class RecorderState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class TrajectoryEvent:
    type: str
    data: dict[str, Any]

@dataclass
class Trajectory:
    run_id: str
    task_id: str
    started_at: str

    max_steps: int
    budget: float | None

    events: list[TrajectoryEvent] = field(default_factory=list)

    ended_at:str | None = None
    wall_time: float | None = None

    steps: int | None = None
    total_usage: Usage | None = None
    stop_reason: StopReason | None = None
    final_message: str | None = None
    patch_pointer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        if self.stop_reason is not None:
            data["stop_reason"] = self.stop_reason.value

        return data


class TrajectoryRecorder:

    def __init__(self):
        self.trajectory: Trajectory | None = None
        self._start_perf: float | None = None
        self.state = RecorderState.NEW

    def start(
        self,
        task_id: str,
        max_steps: int,
        budget: float | None,
    ) -> None:
        self._start_perf = perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        if self.state == RecorderState.NEW:
            self.trajectory = Trajectory(run_id=str(uuid4()),
                                         started_at=started_at,
                                         task_id=task_id,
                                         max_steps=max_steps,
                                         budget = budget,
                                        )

            self.state = RecorderState.RUNNING
        else:
            raise RuntimeError("recorder has already started")

    def record(self, event: TrajectoryEvent) -> None:
        if self.state != RecorderState.RUNNING or self.trajectory is None:
            raise RuntimeError("recorder is not running")

        self.trajectory.events.append(event)

    def finalize(
        self,
        steps: int,
        total_usage: Usage,
        stop_reason: StopReason,
        final_message: str | None,
        patch_pointer: str | None = None,
    ) -> Trajectory:
        if self.state != RecorderState.RUNNING or self.trajectory is None:
            raise RuntimeError("recorder is not running")
        if self._start_perf is None:
            raise RuntimeError("recorder start time is missing")

        self.trajectory.ended_at = datetime.now(timezone.utc).isoformat()
        self.trajectory.wall_time = perf_counter() - self._start_perf

        self.trajectory.steps = steps
        self.trajectory.total_usage = total_usage
        self.trajectory.stop_reason = stop_reason
        self.trajectory.final_message = final_message
        self.trajectory.patch_pointer = patch_pointer
        self.state = RecorderState.FINISHED

        return self.trajectory
