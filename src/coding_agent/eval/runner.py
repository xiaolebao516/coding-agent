"""Batch ablation runner for comparing baseline, ACI, per-step masking, and epoch masking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from coding_agent.agent import Agent
from coding_agent.contracts import ModelFinishReason, Task
from coding_agent.deepseek import DeepSeekModel
from coding_agent.eval.analyze import render_markdown_table, summarize_records
from coding_agent.eval.profiles import AblationProfile, get_profile
from coding_agent.gemini import GeminiModel
from coding_agent.middleware import MiddlewareModel, build_middleware_chain
from coding_agent.runtime.base import Runtime
from coding_agent.runtime.docker import DockerRuntime
from coding_agent.tools.toolsets import build_tool_registry
from coding_agent.trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class EvalTaskSpec:
    task_id: str
    problem_statement: str
    workspace: Path
    verify_command: str
    image: str = "python:3.12-slim"


def load_manifest(manifest_path: Path) -> list[EvalTaskSpec]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest must be a non-empty JSON list of task specs")
    base_dir = manifest_path.parent.resolve()
    tasks: list[EvalTaskSpec] = []
    for item in raw:
        ws = Path(item["workspace"])
        if not ws.is_absolute():
            ws = (base_dir / ws).resolve()
        tasks.append(
            EvalTaskSpec(
                task_id=str(item["task_id"]),
                problem_statement=str(item["problem_statement"]),
                workspace=ws,
                verify_command=str(item["verify_command"]),
                image=str(item.get("image", "python:3.12-slim")),
            )
        )
    return tasks


def run_ablation_task(
    *,
    task_spec: EvalTaskSpec,
    profile: AblationProfile,
    model_factory: Callable[[list[Any]], Any],
    runtime_factory: Callable[[Path, str], Runtime],
    output_dir: Path,
    max_steps: int = 15,
    budget: float | None = 0.5,
    max_total_tokens: int | None = None,
) -> dict[str, Any]:
    """Execute one (profile, task) pair in an isolated workspace copy and verify the result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = build_tool_registry(profile.toolset)
    base_model = model_factory(registry.specs())
    middlewares = build_middleware_chain(
        context_strategy=profile.context_strategy,
        keep_recent_observations=profile.keep_recent_observations,
        mask_epoch_steps=profile.mask_epoch_steps,
        enable_reminders=profile.enable_reminders,
    )
    wrapped_model = MiddlewareModel(base_model, middlewares, max_steps=max_steps)

    with tempfile.TemporaryDirectory(prefix=f"eval-{profile.name}-{task_spec.task_id}-") as tmp_dir:
        isolated_ws = Path(tmp_dir) / "workspace"
        shutil.copytree(task_spec.workspace, isolated_ws, symlinks=True)

        runtime = runtime_factory(isolated_ws, task_spec.image)
        recorder = TrajectoryRecorder()
        agent = Agent(
            model=wrapped_model,
            tool_registry=registry,
            max_steps=max_steps,
            budget=budget,
            max_total_tokens=max_total_tokens,
        )

        resolved = False
        verify_return_code: int | None = None
        verify_stderr = ""
        try:
            runtime.start()
            result = agent.run(
                task=Task(task_id=task_spec.task_id, problem_statement=task_spec.problem_statement),
                runtime=runtime,
                recorder=recorder,
            )
            verify_proc = runtime.run_process(["bash", "-lc", task_spec.verify_command])
            verify_return_code = verify_proc.return_code
            verify_stderr = verify_proc.stderr
            resolved = (verify_proc.return_code == 0) and (not verify_proc.timed_out)
        finally:
            runtime.close()

    stem = f"{profile.name}__{task_spec.task_id}"
    traj_path = output_dir / f"{stem}.trajectory.json"
    recorder.save(traj_path)

    invalid_action_events = 0
    if recorder.trajectory is not None:
        invalid_action_events = sum(
            1
            for ev in recorder.trajectory.events
            if ev.type == "model_response"
            and ev.data.get("finish_reason") == ModelFinishReason.INVALID_ACTION.value
        )

    record: dict[str, Any] = {
        "profile": profile.name,
        "task_id": task_spec.task_id,
        "provider": wrapped_model.provider,
        "model": wrapped_model.model,
        "resolved": resolved,
        "verify_return_code": verify_return_code,
        "verify_stderr_tail": verify_stderr[-500:] if verify_stderr else "",
        "stop_reason": result.stop_reason.value,
        "steps": result.steps,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "cache_read_tokens": result.usage.cache_read_tokens,
            "output_tokens": result.usage.output_tokens,
            "cost_usd": result.usage.cost_usd,
        },
        "invalid_action_events": invalid_action_events,
        "middleware_stats": wrapped_model.stats.to_dict(),
        "trajectory_path": str(traj_path),
    }

    eval_path = output_dir / f"{stem}.eval.json"
    eval_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _build_provider_factory(provider: str, model_name: str | None) -> Callable[[list[Any]], Any]:
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for --provider deepseek")
        chosen = model_name or "deepseek-chat"
        return lambda specs: DeepSeekModel(api_key=api_key, tool_specs=specs, model=chosen)
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required for --provider gemini")
        chosen = model_name or "gemini-2.5-flash"
        return lambda specs: GeminiModel(api_key=api_key, tool_specs=specs, model=chosen)
    raise SystemExit(f"Unsupported provider for eval runner: {provider!r} (use deepseek or gemini)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local coding-agent ablation experiments.")
    parser.add_argument("--manifest", required=True, help="Path to JSON task manifest.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "aci", "aci_per_step", "aci_epoch"],
        help="Ablation profiles to run.",
    )
    parser.add_argument("--provider", choices=["deepseek", "gemini"], default="deepseek")
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/eval_runs")
    args = parser.parse_args(argv)

    tasks = load_manifest(Path(args.manifest))
    profiles = [get_profile(p) for p in args.profiles]
    model_factory = _build_provider_factory(args.provider, args.model)
    budget = None if args.provider == "gemini" else args.budget
    max_total_tokens = args.max_total_tokens
    if args.provider == "gemini" and max_total_tokens is None:
        max_total_tokens = 40_000

    records: list[dict[str, Any]] = []
    for profile in profiles:
        for task_spec in tasks:
            rec = run_ablation_task(
                task_spec=task_spec,
                profile=profile,
                model_factory=model_factory,
                runtime_factory=lambda ws, img: DockerRuntime(workspace=ws, image=img),
                output_dir=Path(args.output_dir),
                max_steps=args.max_steps,
                budget=budget,
                max_total_tokens=max_total_tokens,
            )
            records.append(rec)
            print(
                f"[{profile.name}] {task_spec.task_id}: resolved={rec['resolved']} "
                f"steps={rec['steps']} stop={rec['stop_reason']}"
            )

    print("\n" + render_markdown_table(summarize_records(records)))


if __name__ == "__main__":
    main()
