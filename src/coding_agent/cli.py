import argparse
import json
import os
from pathlib import Path
from coding_agent.agent import Agent
from coding_agent.contracts import Task
from coding_agent.deepseek import DeepSeekModel
from coding_agent.gemini import GeminiModel
from coding_agent.groq import DEFAULT_MODEL as GROQ_DEFAULT_MODEL, GroqModel
from coding_agent.openrouter import DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL, OpenRouterModel
from coding_agent.model import Model
from coding_agent.runtime.base import Runtime
from coding_agent.runtime.docker import DockerRuntime
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trajectory import TrajectoryRecorder
from coding_agent.deepseek_billing import DeepSeekBilling
from coding_agent.workspace import (
    export_patch, prepare_workspace, validate_artifact_paths, write_predictions,
    restore_pytest_version, PYTEST_VERSION_FILE,
)

API_KEY_ENV = "DEEPSEEK_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
GROQ_API_KEY_ENV = "GROQ_API_KEY"

def run_once(
    task: Task,
    model: Model,
    registry: ToolRegistry,
    runtime: Runtime,
    max_steps: int,
    budget: float | None,
    trajectory_path: str,
    max_total_tokens: int | None = None,
    patch_workspace: Path | None = None,
    base_commit: str | None = None,
    patch_path: Path | None = None,
    predictions_path: Path | None = None,
    patch_exclude_paths: tuple[str, ...] = (),
):
    artifacts = (patch_workspace, base_commit, patch_path, predictions_path)
    if any(value is not None for value in artifacts):
        if not all(value is not None for value in artifacts):
            raise ValueError("patch export requires workspace, base commit, patch and predictions paths")
        validate_artifact_paths(patch_workspace, Path(trajectory_path), patch_path, predictions_path)
    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=max_steps,
        budget=budget,
        max_total_tokens=max_total_tokens,
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

        if patch_workspace is not None:
            patch = export_patch(patch_workspace, base_commit, patch_exclude_paths)
            patch_path.write_text(patch, encoding="utf-8")
            write_predictions(predictions_path, task.task_id, f"{model.provider}/{model.model}", patch)
            assert trajectory is not None
            trajectory.patch_pointer = str(patch_path.resolve())

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
        help="Problem statement for the agent.",
    )
    parser.add_argument("--problem-file", type=Path, help="UTF-8 public problem statement file.")

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )

    parser.add_argument(
        "--provider",
        choices=["deepseek", "gemini", "openrouter", "groq"],
        default="deepseek",
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum model cost in USD (DeepSeek default: 1.0; unavailable for Gemini).",
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
    parser.add_argument("--container-workdir", default="/workspace")
    parser.add_argument("--container-python-env", help="Python environment prefix applied inside bash -lc.")
    parser.add_argument("--match-workspace-owner", action="store_true")
    parser.add_argument("--restore-pytest-version", action="store_true", help="Reuse the pinned image's pytest build artifact.")
    parser.add_argument("--base-commit", help="Enable isolated Git checkout and patch export.")
    parser.add_argument("--task-workspace", type=Path, help="New isolated checkout directory.")
    parser.add_argument("--patch-path", type=Path)
    parser.add_argument("--predictions-path", type=Path)

    parser.add_argument(
        "--model",
        default=None,
        help="Model name.",
    )

    parser.add_argument(
        "--trajectory-path",
        default="trajectory.json",
        help="Where to save the trajectory JSON.",
    )

    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Soft run-level input + output token limit (Gemini default: 20000).",
    )

    parser.add_argument(
        "--model-timeout-seconds",
        type=float,
        default=None,
        help="Gemini HTTP request timeout in seconds (default: 60).",
    )

    parser.add_argument(
        "--thinking-level",
        choices=["low", "medium", "high"],
        default=None,
        help="Gemini reasoning level; defaults to low for backward compatibility.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
    )

    return parser


# 用户真正执行命令时进入这里
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if (args.problem is None) == (args.problem_file is None):
        raise SystemExit("provide exactly one of --problem or --problem-file")
    export_args = (args.base_commit, args.task_workspace, args.patch_path, args.predictions_path)
    if any(value is not None for value in export_args) and not all(value is not None for value in export_args):
        raise SystemExit("provide --base-commit, --task-workspace, --patch-path and --predictions-path together")
    if args.restore_pytest_version and args.base_commit is None:
        raise SystemExit("--restore-pytest-version requires an isolated --base-commit workspace")

    if args.max_total_tokens is not None and args.max_total_tokens <= 0:
        raise SystemExit("--max-total-tokens must be positive")
    if args.provider in ("gemini", "openrouter", "groq") and args.budget is not None:
        raise SystemExit("--budget is unavailable for this provider; use --max-total-tokens")
    if args.provider != "gemini" and args.thinking_level is not None:
        raise SystemExit("--thinking-level is only supported by Gemini")
    if args.model_timeout_seconds is not None:
        if args.model_timeout_seconds <= 0:
            raise SystemExit("--model-timeout-seconds must be positive")
        if args.provider != "gemini":
            raise SystemExit("--model-timeout-seconds is only supported by Gemini")

    # Fail fast: before any container or model is created.
    # Never print the key value, not even a prefix.
    api_key_env = {
        "deepseek": API_KEY_ENV,
        "gemini": GEMINI_API_KEY_ENV,
        "openrouter": OPENROUTER_API_KEY_ENV,
        "groq": GROQ_API_KEY_ENV,
    }[args.provider]

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is not set or empty")

    task = Task(
        task_id=args.task_id,
        problem_statement=args.problem if args.problem is not None else args.problem_file.read_text(encoding="utf-8"),
    )

    # Single source of truth for tools: the model sees exactly what the agent can run.
    registry = ToolRegistry([BashTool()])

    workspace = Path(args.workspace)
    base_commit = None
    if args.base_commit is not None:
        validate_artifact_paths(
            args.task_workspace, Path(args.trajectory_path), args.patch_path, args.predictions_path,
        )
        base_commit = prepare_workspace(workspace, args.task_workspace, args.base_commit)
        workspace = args.task_workspace

    patch_exclude_paths = ()
    if args.restore_pytest_version:
        restore_pytest_version(workspace, args.image)
        patch_exclude_paths = (PYTEST_VERSION_FILE,)

    if args.model is None:
        args.model = {
            "gemini": "gemini-3.8-flash",
            "deepseek": "deepseek-flash",
            "openrouter": OPENROUTER_DEFAULT_MODEL,
            "groq": GROQ_DEFAULT_MODEL,
        }[args.provider]

    if args.provider == "gemini":
        model = GeminiModel(
            api_key=api_key,
            tool_specs=registry.specs(),
            model=args.model,
            max_output_tokens=(
                args.max_output_tokens if args.max_output_tokens is not None else 512
            ),
            thinking_level=args.thinking_level or "low",
            timeout=(args.model_timeout_seconds if args.model_timeout_seconds is not None else 60.0),
        )
        budget = None
        max_total_tokens = (
            args.max_total_tokens if args.max_total_tokens is not None else 20_000
        )
    elif args.provider == "openrouter":
        model = OpenRouterModel(
            api_key=api_key,
            tool_specs=registry.specs(),
            model=args.model,
            max_output_tokens=(args.max_output_tokens if args.max_output_tokens is not None else 8192),
        )
        budget = None
        max_total_tokens = args.max_total_tokens if args.max_total_tokens is not None else 60_000
    elif args.provider == "groq":
        model = GroqModel(
            api_key=api_key,
            tool_specs=registry.specs(),
            model=args.model,
            max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else 4096,
        )
        budget = None
        max_total_tokens = args.max_total_tokens if args.max_total_tokens is not None else 60_000
    else:
        billing = DeepSeekBilling(api_key=api_key)
        if not billing.is_available():
            raise SystemExit("DeepSeek account has no available balance.")
        model = DeepSeekModel(
            api_key=api_key,
            tool_specs=registry.specs(),
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        budget = args.budget if args.budget is not None else 1.0
        max_total_tokens = args.max_total_tokens

    runtime = DockerRuntime(
        workspace=workspace,
        image=args.image,
        container_workdir=args.container_workdir,
        container_python_env=args.container_python_env,
        match_workspace_owner=args.match_workspace_owner,
    )

    result = run_once(
        task=task,
        model=model,
        registry=registry,
        runtime=runtime,
        max_steps=args.max_steps,
        budget=budget,
        trajectory_path=args.trajectory_path,
        max_total_tokens=max_total_tokens,
        patch_workspace=workspace if base_commit is not None else None,
        base_commit=base_commit,
        patch_path=args.patch_path,
        predictions_path=args.predictions_path,
        patch_exclude_paths=patch_exclude_paths,
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
