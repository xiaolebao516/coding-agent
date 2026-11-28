"""Batch ablation runner with isolated verification workspace and fault-tolerant persistence."""

from __future__ import annotations

import argparse
import fnmatch
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

DEFAULT_PROTECTED_TEST_PATTERNS = ("tests/*", "test/*", "*/tests/*", "test_*.py", "*_test.py")
SKIP_SYNC_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache"})


@dataclass(frozen=True)
class EvalTaskSpec:
    task_id: str
    problem_statement: str
    workspace: Path
    verify_command: str
    image: str = "python:3.12-slim"
    protected_test_paths: tuple[str, ...] = ()
    protect_tests: bool = True


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
        protected = tuple(str(p) for p in (item.get("protected_test_paths") or ()))
        tasks.append(
            EvalTaskSpec(
                task_id=str(item["task_id"]),
                problem_statement=str(item["problem_statement"]),
                workspace=ws,
                verify_command=str(item["verify_command"]),
                image=str(item.get("image", "python:3.12-slim")),
                protected_test_paths=protected,
                protect_tests=bool(item.get("protect_tests", True)),
            )
        )
    return tasks


def _is_protected_relpath(rel_posix: str, spec: EvalTaskSpec) -> bool:
    if not spec.protect_tests:
        return False
    patterns = spec.protected_test_paths or DEFAULT_PROTECTED_TEST_PATTERNS
    name = Path(rel_posix).name
    for pat in patterns:
        norm = pat.rstrip("/")
        if rel_posix == norm or rel_posix.startswith(norm + "/"):
            return True
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _iter_workspace_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_SYNC_PARTS]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = fpath.relative_to(root).as_posix()
            files[rel] = fpath
    return files


def _sync_non_test_changes(
    *,
    clean_ws: Path,
    agent_ws: Path,
    verify_ws: Path,
    spec: EvalTaskSpec,
) -> list[str]:
    """Copy only non-protected file changes from agent_ws into fresh verify_ws.

    Returns a sorted list of protected test paths the agent attempted to modify,
    create, or delete (so test tampering is detected and cannot produce false passes).
    """
    clean_files = _iter_workspace_files(clean_ws)
    agent_files = _iter_workspace_files(agent_ws)
    all_rels = sorted(set(clean_files) | set(agent_files))
    tampered: list[str] = []

    for rel in all_rels:
        in_clean = rel in clean_files
        in_agent = rel in agent_files

        changed = False
        if in_clean and in_agent:
            changed = clean_files[rel].read_bytes() != agent_files[rel].read_bytes()
        else:
            changed = True

        if not changed:
            continue

        if _is_protected_relpath(rel, spec):
            tampered.append(rel)
            continue

        target = verify_ws / rel
        if in_agent:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agent_files[rel], target)
        elif target.exists():
            target.unlink()

    return tampered


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
    """Execute one (profile, task) pair with independent verification and fault-tolerant persistence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{profile.name}__{task_spec.task_id}"
    traj_path = output_dir / f"{stem}.trajectory.json"
    eval_path = output_dir / f"{stem}.eval.json"

    wrapped_model: MiddlewareModel | None = None
    recorder = TrajectoryRecorder()
    tampered_test_files: list[str] = []

    try:
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
            tmp_root = Path(tmp_dir)
            agent_ws = tmp_root / "agent_workspace"
            verify_ws = tmp_root / "verify_workspace"
            shutil.copytree(task_spec.workspace, agent_ws, symlinks=True)

            # 1. Run agent inside isolated agent_ws
            agent_runtime = runtime_factory(agent_ws, task_spec.image)
            agent = Agent(
                model=wrapped_model,
                tool_registry=registry,
                max_steps=max_steps,
                budget=budget,
                max_total_tokens=max_total_tokens,
            )
            try:
                agent_runtime.start()
                result = agent.run(
                    task=Task(task_id=task_spec.task_id, problem_statement=task_spec.problem_statement),
                    runtime=agent_runtime,
                    recorder=recorder,
                )
            finally:
                agent_runtime.close()

            # 2. Verify inside a fresh clean workspace with protected test files restored
            shutil.copytree(task_spec.workspace, verify_ws, symlinks=True)
            tampered_test_files = _sync_non_test_changes(
                clean_ws=task_spec.workspace,
                agent_ws=agent_ws,
                verify_ws=verify_ws,
                spec=task_spec,
            )

            verify_runtime = runtime_factory(verify_ws, task_spec.image)
            try:
                verify_runtime.start()
                verify_proc = verify_runtime.run_process(["bash", "-lc", task_spec.verify_command])
                verify_return_code = verify_proc.return_code
                verify_stderr = verify_proc.stderr
                resolved = (verify_proc.return_code == 0) and (not verify_proc.timed_out)
            finally:
                verify_runtime.close()

        saved_traj_path: str | None = None
        invalid_action_events = 0
        if recorder.trajectory is not None:
            traj_path.write_text(
                json.dumps(recorder.trajectory.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            saved_traj_path = str(traj_path)
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
            "tampered_test_files": tampered_test_files,
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
            "runner_error": None,
            "trajectory_path": saved_traj_path,
        }
    except Exception as exc:
        record = {
            "profile": profile.name,
            "task_id": task_spec.task_id,
            "provider": wrapped_model.provider if wrapped_model is not None else "unknown",
            "model": wrapped_model.model if wrapped_model is not None else "unknown",
            "resolved": False,
            "verify_return_code": None,
            "verify_stderr_tail": "",
            "tampered_test_files": tampered_test_files,
            "stop_reason": "eval_runner_error",
            "steps": 0,
            "usage": {
                "input_tokens": None,
                "cache_read_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
            },
            "invalid_action_events": 0,
            "middleware_stats": wrapped_model.stats.to_dict() if wrapped_model is not None else {},
            "runner_error": f"{type(exc).__name__}: {exc}",
            "trajectory_path": None,
        }

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
        default=["baseline", "aci", "aci_per_step", "aci_epoch", "aci_epoch_remind"],
        help="Ablation profiles to run.",
    )
    parser.add_argument("--provider", choices=["deepseek", "gemini"], default="deepseek")
    parser.add_argument("--model", default=None, help="Override provider model name.")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/eval_runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (profile, task) pairs that already have a saved .eval.json file.",
    )
    args = parser.parse_args(argv)

    tasks = load_manifest(Path(args.manifest))
    profiles = [get_profile(p) for p in args.profiles]
    model_factory = _build_provider_factory(args.provider, args.model)
    budget = None if args.provider == "gemini" else args.budget
    max_total_tokens = args.max_total_tokens
    if args.provider == "gemini" and max_total_tokens is None:
        max_total_tokens = 40_000

    out_dir = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    for profile in profiles:
        for task_spec in tasks:
            existing = out_dir / f"{profile.name}__{task_spec.task_id}.eval.json"
            if args.resume and existing.exists():
                rec = json.loads(existing.read_text(encoding="utf-8"))
                records.append(rec)
                print(f"[{profile.name}] {task_spec.task_id}: skipped (--resume)")
                continue

            rec = run_ablation_task(
                task_spec=task_spec,
                profile=profile,
                model_factory=model_factory,
                runtime_factory=lambda ws, img: DockerRuntime(workspace=ws, image=img),
                output_dir=out_dir,
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
