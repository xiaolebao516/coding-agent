import argparse
import json
import os
from pathlib import Path

from coding_agent.agent import Agent
from coding_agent.contracts import Task
from coding_agent.deepseek import DeepSeekModel
from coding_agent.model import Model
from coding_agent.runtime.base import Runtime
from coding_agent.runtime.docker import DockerRuntime
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder

API_KEY_ENV = "DEEPSEEK_API_KEY"

def run_once(
    task: Task,
    model: Model,
    registry: ToolRegistry,
    runtime: Runtime,
    max_steps: int,
    budget: float,
    trajectory_path: str,
):
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
        "--workspace",
        required=True,
        help="Host directory mounted into the task container.",
    )

    parser.add_argument(
        "--image",
        default="python:3.12-slim",
        help="Docker image for the task container.",
    )

    parser.add_argument(
        "--model",
        default="deepseek-flash",
        help="DeepSeek model name.",
    )

    parser.add_argument(
        "--trajectory-path",
        default="trajectory.json",
        help="Where to save the trajectory JSON.",
    )

    return parser


# 用户真正执行命令时进入这里
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Fail fast: before any container or model is created.
    # Never print the key value, not even a prefix.
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"{API_KEY_ENV} is not set or empty; aborting before any work.")

    task = Task(
        task_id=args.task_id,
        problem_statement=args.problem,
    )

    # Single source of truth for tools: the model sees exactly what the agent can run.
    registry = ToolRegistry([BashTool()])

    model = DeepSeekModel(
        api_key=api_key,
        tool_specs=registry.specs(),
        model=args.model,
    )
    runtime = DockerRuntime(
        workspace=Path(args.workspace),
        image=args.image,
    )

    result = run_once(
        task=task,
        model=model,
        registry=registry,
        runtime=runtime,
        max_steps=args.max_steps,
        budget=args.budget,
        trajectory_path=args.trajectory_path,
    )

    print(
        json.dumps(
            {
                "stop_reason": result.stop_reason.value,
                "steps": result.steps,
                "cost_usd": result.usage.cost_usd,
                "trajectory": args.trajectory_path,
            }
        )
    )


if __name__ == "__main__":
    main()
