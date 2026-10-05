import argparse
import json

from coding_agent.agent import Agent
from coding_agent.contracts import Task
from coding_agent.model import Model
from coding_agent.runtime.base import Runtime
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder

def run_once(
    task: Task,
    model: Model,
    runtime: Runtime,
    max_steps: int,
    budget: float,
    trajectory_path: str,
):
    registry = ToolRegistry([BashTool()])

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=max_steps,
        budget=budget,
    )

    recorder = TrajectoryRecorder()

    runtime.start()

    try:
        result = agent.run(
            task=task,
            runtime=runtime,
            recorder=recorder,
        )

        trajectory = recorder.trajectory

        # 保存 trajectory JSON
        with open(trajectory_path, "w", encoding="utf-8") as f:
            assert trajectory is not None
            json.dump(
                trajectory.to_dict(),
                f,
                ensure_ascii=False,
                indent=2
            )

        return result

    finally:
        runtime.close()


# CLI 参数解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the coding agent.")

    parser.add_argument(
        "--task-id",
        required=True,
        help="Unique task identifier.",
    )

    parser.add_argument(
        "--problem",
        required=True,
        help="Problem statement for the agent.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=1.0,
        help="Maximum model cost in USD.",
    )

    parser.add_argument(
        "--trajectory-path",
        default="trajectory.json",
        help="Where to save the trajectory JSON.",
    )

    return parser


# 用户真正执行命令时进入这里
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    task = Task(
        task_id=args.task_id,
        problem_statement=args.problem
    )

    # runtime = DockerRuntime(...)
    # model = RealModel(...) # M2 real-model gate前补

    raise RuntimeError(
        "Real model adapter is not configured yet."
    )

if __name__ == "__main__":
    main()
