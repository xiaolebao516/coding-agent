import json
from pathlib import Path
import subprocess

import pytest

from coding_agent.cli import run_once
from coding_agent import cli
from coding_agent import workspace as workspace_module
from coding_agent.contracts import ModelResponse, StopReason, Task, ToolCall, Usage
from coding_agent.model import FakeModel
from coding_agent.runtime.base import ProcessResult
from coding_agent.runtime.docker import DockerRuntime
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.workspace import export_patch, prepare_workspace, validate_artifact_paths


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True,
    ).stdout


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "offline@example.invalid")
    git(repo, "config", "user.name", "Offline test")
    (repo / "tracked.txt").write_text("base\n")
    (repo / "deleted.txt").write_text("delete me\n")
    (repo / ".gitignore").write_text("ignored.txt\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    return repo, base


def test_isolation_and_patch_apply_roundtrip(source, tmp_path):
    repo, base = source
    (repo / "tracked.txt").write_text("source dirty\n")
    git(repo, "add", "tracked.txt")
    (repo / "private-untracked.txt").write_text("never copy\n")
    source_status = git(repo, "status", "--porcelain")
    task = tmp_path / "task"
    assert prepare_workspace(repo, task, base) == base
    assert (task / "tracked.txt").read_text() == "base\n"
    assert not (task / "private-untracked.txt").exists()
    assert (task / ".git").is_dir()
    assert git(task, "rev-parse", "HEAD").decode().strip() == base

    (task / "tracked.txt").write_text("fixed\n")
    (task / "deleted.txt").unlink()
    (task / "new file.txt").write_text("new\n")
    (task / "new.bin").write_bytes(b"\0\xff\x01")
    (task / "ignored.txt").write_text("ignore\n")
    git(task, "add", "tracked.txt")
    index = (task / ".git/index").read_bytes()
    patch = export_patch(task, base)
    assert "new file.txt" in patch
    assert "GIT binary patch" in patch
    assert "ignored.txt" not in patch
    assert (task / ".git/index").read_bytes() == index
    assert git(repo, "status", "--porcelain") == source_status
    assert (repo / "tracked.txt").read_text() == "source dirty\n"

    target = tmp_path / "apply"
    prepare_workspace(repo, target, base)
    patch_file = tmp_path / "generated.diff"
    patch_file.write_text(patch)
    git(target, "apply", "--check", str(patch_file))
    git(target, "apply", str(patch_file))
    assert (target / "tracked.txt").read_text() == "fixed\n"
    assert (target / "new file.txt").read_text() == "new\n"
    assert (target / "new.bin").read_bytes() == b"\0\xff\x01"
    assert not (target / "deleted.txt").exists()


def test_empty_patch_and_committed_changes(source, tmp_path):
    repo, base = source
    task = tmp_path / "task"
    prepare_workspace(repo, task, base)
    assert export_patch(task, base) == ""
    (task / "tracked.txt").write_text("committed fix\n")
    git(task, "add", ".")
    git(task, "-c", "user.name=Offline", "-c", "user.email=offline@example.invalid", "commit", "-m", "fix")
    assert "+committed fix" in export_patch(task, base)


def test_reviewed_exclusions_preserve_real_changes_and_index(source, tmp_path):
    repo, base = source
    task = tmp_path / "task"
    prepare_workspace(repo, task, base)
    (task / "tracked.txt").write_text("fixed\n")
    (task / "added.py").write_text("answer = 42\n")
    (task / "scratch.txt").write_text("temporary\n")
    (task / "debug").mkdir()
    (task / "debug/log.txt").write_text("temporary log\n")
    git(task, "add", "scratch.txt")
    index = (task / ".git/index").read_bytes()
    raw = export_patch(task, base)
    reviewed = export_patch(task, base, ("scratch.txt", "debug"))
    assert "scratch.txt" in raw and "debug/log.txt" in raw
    assert "scratch.txt" not in reviewed and "debug/log.txt" not in reviewed
    assert "+fixed" in reviewed and "added.py" in reviewed
    assert (task / "scratch.txt").exists()
    assert (task / "debug/log.txt").exists()
    assert (task / ".git/index").read_bytes() == index
    patch_file = tmp_path / "reviewed.diff"
    patch_file.write_text(reviewed)
    target = tmp_path / "review-apply"
    prepare_workspace(repo, target, base)
    git(target, "apply", "--check", str(patch_file))
    git(target, "apply", str(patch_file))
    assert (target / "added.py").exists()
    assert not (target / "scratch.txt").exists()


@pytest.mark.parametrize("path", ["", ".", "../outside", "/absolute"])
def test_reject_invalid_patch_exclusions(source, path):
    repo, base = source
    with pytest.raises(ValueError, match="repository-relative"):
        export_patch(repo, base, (path,))


def test_reject_reuse_and_invalid_base(source, tmp_path):
    repo, base = source
    with pytest.raises(ValueError, match="already exists"):
        prepare_workspace(repo, repo, base)
    with pytest.raises(ValueError, match="outside"):
        prepare_workspace(repo, repo / "task", base)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_workspace(repo, tmp_path / "invalid", "no-such-commit")
    assert not (tmp_path / "invalid").exists()


def test_forbidden_tree_rejected_from_names_only(monkeypatch, tmp_path):
    calls = []

    def names_only(repo, *args, **kwargs):
        calls.append(args[0])
        return b"a" * 40 + b"\n" if args[0] == "rev-parse" else b"eval.sh\0"

    monkeypatch.setattr(workspace_module, "_git", names_only)
    with pytest.raises(ValueError, match="forbidden"):
        prepare_workspace(tmp_path / "source", tmp_path / "task", "base")
    assert calls == ["rev-parse", "ls-tree"]
    assert not (tmp_path / "task").exists()


def test_artifacts_reject_mount_collisions_and_overwrite(tmp_path):
    task = tmp_path / "task"
    with pytest.raises(ValueError, match="outside"):
        validate_artifact_paths(task, task / "result.json")
    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="distinct"):
        validate_artifact_paths(task, path, path)
    path.write_text("keep")
    with pytest.raises(ValueError, match="overwrite"):
        validate_artifact_paths(task, path)


class LocalRuntime:
    """Test-only runtime; no Docker or network required."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.closed = False

    def start(self):
        pass

    def run_process(self, argv):
        result = subprocess.run(argv, cwd=self.workspace, capture_output=True, text=True)
        return ProcessResult(result.returncode, result.stdout, result.stderr)

    def close(self):
        self.closed = True


def test_fake_model_exports_patch_predictions_and_pointer(source, tmp_path):
    repo, base = source
    task = tmp_path / "task"
    prepare_workspace(repo, task, base)
    model = FakeModel([
        ModelResponse(
            content=None,
            tool_call=ToolCall(id="edit", name="bash", arguments={
                "command": "printf 'fixed\\n' > tracked.txt; printf 'new\\n' > added.txt",
            }),
            usage=Usage(cost_usd=0.0),
        ),
        ModelResponse(content="done", tool_call=None, usage=Usage(cost_usd=0.0)),
    ])
    runtime = LocalRuntime(task)
    trajectory = tmp_path / "trajectory.json"
    patch_path = tmp_path / "agent.diff"
    predictions = tmp_path / "predictions.jsonl"
    result = run_once(
        task=Task(task_id="offline-example", problem_statement="Update a file and add a file."),
        model=model, registry=ToolRegistry([BashTool()]), runtime=runtime,
        max_steps=3, budget=None, trajectory_path=str(trajectory),
        patch_workspace=task, base_commit=base,
        patch_path=patch_path, predictions_path=predictions,
    )
    assert result.stop_reason == StopReason.MODEL_FINISHED
    assert runtime.closed
    assert len(predictions.read_text().splitlines()) == 1
    row = json.loads(predictions.read_text())
    assert row == {
        "instance_id": "offline-example", "model_name_or_path": "fake/fake",
        "model_patch": patch_path.read_text(),
    }
    assert "added.txt" in row["model_patch"]
    assert json.loads(trajectory.read_text())["patch_pointer"] == str(patch_path.resolve())


def test_docker_workdir_and_no_implicit_pull(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "container-id\n", "")

    monkeypatch.setattr("coding_agent.runtime.docker.subprocess.run", run)
    runtime = DockerRuntime(tmp_path, "local-image", container_workdir="/testbed")
    runtime.start()
    runtime.run_process(["pwd"])
    runtime.close()
    assert "--pull=never" in calls[0]
    assert f"{tmp_path}:/testbed" in calls[0]
    assert calls[0][calls[0].index("-w") + 1] == "/testbed"
    assert calls[1] == ["docker", "exec", "-w", "/testbed", "container-id", "pwd"]


def test_missing_image_stops_before_model_generation(monkeypatch, tmp_path):
    def missing_image(argv, **kwargs):
        assert "--pull=never" in argv
        raise subprocess.CalledProcessError(125, argv, stderr="image not found locally")

    monkeypatch.setattr("coding_agent.runtime.docker.subprocess.run", missing_image)
    model = FakeModel([])
    with pytest.raises(subprocess.CalledProcessError):
        run_once(
            task=Task(task_id="offline", problem_statement="unused"), model=model,
            registry=ToolRegistry([BashTool()]), runtime=DockerRuntime(tmp_path, "missing"),
            max_steps=1, budget=None, trajectory_path=str(tmp_path / "trajectory.json"),
        )
    assert model.index == 0


def test_cli_wires_isolated_task_and_artifacts(monkeypatch, source, tmp_path):
    repo, base = source
    monkeypatch.setenv(cli.GEMINI_API_KEY_ENV, "offline-stub")
    monkeypatch.setattr(cli, "GeminiModel", lambda **kwargs: FakeModel([
        ModelResponse(content="done", tool_call=None, usage=Usage(cost_usd=0.0)),
    ]))
    captured = {}

    def runtime(**kwargs):
        captured.update(kwargs)
        return LocalRuntime(kwargs["workspace"])

    monkeypatch.setattr(cli, "DockerRuntime", runtime)
    problem = tmp_path / "public-problem.md"
    problem.write_text("Synthetic public task.")
    trajectory = tmp_path / "trajectory.json"
    cli.main([
        "--provider", "gemini", "--task-id", "offline-example",
        "--problem-file", str(problem), "--workspace", str(repo),
        "--base-commit", base, "--task-workspace", str(tmp_path / "isolated"),
        "--container-workdir", "/testbed", "--image", "offline-image",
        "--container-python-env", "/opt/miniconda3/envs/testbed",
        "--patch-path", str(tmp_path / "agent.diff"),
        "--predictions-path", str(tmp_path / "predictions.jsonl"),
        "--trajectory-path", str(trajectory),
    ])
    assert captured["workspace"] == tmp_path / "isolated"
    assert captured["container_workdir"] == "/testbed"
    assert captured["image"] == "offline-image"
    assert captured["container_python_env"] == "/opt/miniconda3/envs/testbed"
    assert json.loads(trajectory.read_text())["patch_pointer"] == str(tmp_path / "agent.diff")


def test_export_failure_still_closes_runtime(monkeypatch, tmp_path):
    runtime = LocalRuntime(tmp_path / "task")
    monkeypatch.setattr(cli, "export_patch", lambda *args: (_ for _ in ()).throw(RuntimeError("export failed")))
    with pytest.raises(RuntimeError, match="export failed"):
        run_once(
            task=Task(task_id="offline", problem_statement="unused"),
            model=FakeModel([ModelResponse(content="done", tool_call=None, usage=Usage(cost_usd=0.0))]),
            registry=ToolRegistry([BashTool()]), runtime=runtime, max_steps=1, budget=None,
            trajectory_path=str(tmp_path / "trajectory.json"),
            patch_workspace=tmp_path / "task", base_commit="unused",
            patch_path=tmp_path / "agent.diff", predictions_path=tmp_path / "predictions.jsonl",
        )
    assert runtime.closed


@pytest.mark.parametrize("path", ["relative", "/", "/testbed/../etc"])
def test_invalid_container_workdir(tmp_path, path):
    with pytest.raises(ValueError):
        DockerRuntime(tmp_path, "local-image", container_workdir=path)


def test_python_env_applies_inside_login_shell(monkeypatch, tmp_path):
    prefix = tmp_path / "testbed env"
    (prefix / "bin").mkdir(parents=True)
    # Synthetic interpreter verifies executable selection; it does not claim real Python 3.9.
    interpreter = prefix / "bin/python"
    interpreter.write_text('#!/bin/sh\nprintf "testbed-selected:%s\\n" "$CONDA_PREFIX"\n')
    interpreter.chmod(0o755)
    actual_run = subprocess.run
    calls = []

    def docker_exec(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == ["docker", "exec"]
        return actual_run(argv[5:], **kwargs)

    monkeypatch.setattr("coding_agent.runtime.docker.subprocess.run", docker_exec)
    runtime = DockerRuntime(tmp_path, "unused", container_python_env=str(prefix))
    runtime.container_id = "offline"
    command = "command -v python; python"
    result = runtime.run_process(["bash", "-lc", command])
    assert result.return_code == 0
    assert result.stdout.splitlines() == [str(interpreter), f"testbed-selected:{prefix}"]
    assert calls[0][5:7] == ["bash", "-lc"]
    assert "export PATH=" in calls[0][7]
    with pytest.raises(ValueError, match="bash -lc"):
        runtime.run_process(["python", "--version"])


@pytest.mark.parametrize("path", ["relative", "/", "/opt/../etc"])
def test_invalid_container_python_env(tmp_path, path):
    with pytest.raises(ValueError, match="container_python_env"):
        DockerRuntime(tmp_path, "unused", container_python_env=path)
