"""Offline coverage of the two first-task environment repairs.

Docker calls are mocked, and every Git repository is a temporary synthetic repo.
The tar fixture carries arbitrary bytes, not a fabricated pytest version.
"""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from coding_agent import workspace as workspace_module
from coding_agent.cli import run_once
from coding_agent.contracts import ModelResponse, StopReason, Task, Usage
from coding_agent.model import FakeModel
from coding_agent.runtime.docker import DockerRuntime
from coding_agent.tools.bash import BashTool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.workspace import export_patch


IMAGE = "official/offline-fixture@sha256:" + "a" * 64
GENERATED_PATH = "src/_pytest/_version.py"
GENERATED_BYTES = b"# synthetic opaque build-artifact fixture\n"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    git(path, "init")
    git(path, "config", "user.email", "offline@example.invalid")
    git(path, "config", "user.name", "Offline fixture")
    (path / "src/_pytest").mkdir(parents=True)
    (path / "src/_pytest/__init__.py").write_text("# base fixture\n")
    (path / "tracked.py").write_text("value = 'base'\n")
    (path / ".gitignore").write_text(GENERATED_PATH + "\n")
    git(path, "add", ".")
    git(path, "commit", "-m", "base fixture")
    return path


def artifact_tar() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("_version.py")
        member.size = len(GENERATED_BYTES)
        archive.addfile(member, io.BytesIO(GENERATED_BYTES))
    return stream.getvalue()


def mock_artifact_docker(monkeypatch, *, cp_error=False):
    original_run = subprocess.run
    calls = []

    def run(argv, **kwargs):
        if argv[0] != "docker":
            return original_run(argv, **kwargs)
        calls.append((list(argv), kwargs))
        if argv[1] == "create":
            stdout = "offline-container\n" if kwargs.get("text") else b"offline-container\n"
        elif argv[1] == "cp":
            if cp_error:
                raise subprocess.CalledProcessError(1, argv, stderr=b"fixture cp failed")
            stdout = artifact_tar()
        elif argv[1] == "rm":
            stdout = "" if kwargs.get("text") else b""
        else:
            raise AssertionError(f"unexpected Docker operation: {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout, "" if kwargs.get("text") else b"")

    monkeypatch.setattr(workspace_module.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("match_owner", [False, True])
def test_owner_match_is_explicit_and_keeps_network_isolation(monkeypatch, tmp_path, match_owner):
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "offline-container\n", "")

    monkeypatch.setattr("coding_agent.runtime.docker.subprocess.run", run)
    runtime = DockerRuntime(
        tmp_path, IMAGE, container_workdir="/testbed",
        container_python_env="/opt/miniconda3/envs/testbed",
        match_workspace_owner=match_owner,
    )
    runtime.start()
    runtime.run_process(["bash", "-lc", "python -V; git status --porcelain; test -w ."])
    runtime.close()
    command = calls[0]
    assert "--pull=never" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("-w") + 1] == "/testbed"
    assert command[command.index("-v") + 1] == f"{tmp_path}:/testbed"
    if match_owner:
        assert command[command.index("--user") + 1] == f"{tmp_path.stat().st_uid}:{tmp_path.stat().st_gid}"
        assert "HOME=/tmp" in command
    else:
        assert "--user" not in command
        assert "HOME=/tmp" not in command
    assert calls[1][:6] == ["docker", "exec", "-w", "/testbed", "offline-container", "bash"]
    assert calls[1][6] == "-lc"
    assert "export PATH=/opt/miniconda3/envs/testbed/bin:" in calls[1][7]
    assert calls[-1] == ["docker", "rm", "-f", "offline-container"]


def test_restore_copies_existing_image_artifact_without_start_or_network(monkeypatch, repo):
    calls = mock_artifact_docker(monkeypatch)
    index = (repo / ".git/index").read_bytes()
    metadata = workspace_module.restore_pytest_version(repo, IMAGE)
    assert (repo / GENERATED_PATH).read_bytes() == GENERATED_BYTES
    assert metadata["path"] == GENERATED_PATH
    assert metadata["image"] == IMAGE
    assert metadata["sha256"] == hashlib.sha256(GENERATED_BYTES).hexdigest()
    assert (repo / ".git/index").read_bytes() == index
    assert git(repo, "status", "--porcelain") == b""
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    assert export_patch(repo, base) == ""
    assert export_patch(repo, base, (GENERATED_PATH,)) == ""
    assert GENERATED_PATH in (repo / ".git/info/exclude").read_text()
    operations = [argv[1] for argv, _ in calls]
    assert operations == ["create", "cp", "rm"]
    create = calls[0][0]
    assert "--pull=never" in create
    assert IMAGE in create
    assert calls[1][0][2:] == ["offline-container:/testbed/" + GENERATED_PATH, "-"]
    assert calls[-1][0][-1] == "offline-container"


@pytest.mark.parametrize("image", ["official/offline-fixture:latest", "official/offline-fixture@sha256:short"])
def test_restore_rejects_unpinned_images_before_docker(monkeypatch, repo, image):
    calls = mock_artifact_docker(monkeypatch)
    with pytest.raises(ValueError):
        workspace_module.restore_pytest_version(repo, image)
    assert calls == []
    assert not (repo / GENERATED_PATH).exists()


@pytest.mark.parametrize("condition", ["existing", "tracked", "not-ignored"])
def test_restore_rejects_unsafe_artifact_destinations_before_docker(monkeypatch, repo, condition):
    path = repo / GENERATED_PATH
    if condition == "existing":
        path.write_bytes(b"existing fixture must remain unchanged\n")
    elif condition == "tracked":
        path.write_bytes(b"tracked fixture must remain unchanged\n")
        git(repo, "add", "-f", GENERATED_PATH)
        git(repo, "commit", "-m", "tracked fixture")
        path.unlink()
    else:
        (repo / ".gitignore").write_text("")
    calls = mock_artifact_docker(monkeypatch)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        workspace_module.restore_pytest_version(repo, IMAGE)
    assert calls == []
    if condition == "existing":
        assert path.read_bytes() == b"existing fixture must remain unchanged\n"
    else:
        assert not path.exists()


def test_restore_cleans_unstarted_container_when_copy_fails(monkeypatch, repo):
    calls = mock_artifact_docker(monkeypatch, cp_error=True)
    with pytest.raises(subprocess.CalledProcessError):
        workspace_module.restore_pytest_version(repo, IMAGE)
    assert [argv[1] for argv, _ in calls] == ["create", "cp", "rm"]
    assert not (repo / GENERATED_PATH).exists()


@pytest.mark.parametrize("change", ["modified", "deleted"])
def test_excluding_tracked_path_preserves_base_without_spurious_deletion(repo, change):
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    if change == "modified":
        (repo / "tracked.py").write_text("value = 'excluded change'\n")
    else:
        (repo / "tracked.py").unlink()
    (repo / "new_code.py").write_text("result = 42\n")
    git(repo, "add", "-A")
    index = (repo / ".git/index").read_bytes()
    assert "diff --git a/tracked.py b/tracked.py" in export_patch(repo, base)
    patch = export_patch(repo, base, ("tracked.py",))
    assert "diff --git a/tracked.py b/tracked.py" not in patch
    assert "diff --git a/new_code.py b/new_code.py" in patch
    assert "+result = 42" in patch
    assert (repo / ".git/index").read_bytes() == index
    if change == "modified":
        assert (repo / "tracked.py").read_text() == "value = 'excluded change'\n"
    else:
        assert not (repo / "tracked.py").exists()


class LocalRuntime:
    """No container or model request is needed for an artifact-export regression."""

    def __init__(self):
        self.closed = False

    def start(self):
        pass

    def run_process(self, argv):
        raise AssertionError("FakeModel completion should not execute a command")

    def close(self):
        self.closed = True


def test_patch_and_predictions_exclude_artifact_after_ignore_edit_and_force_stage(monkeypatch, repo, tmp_path):
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    mock_artifact_docker(monkeypatch)
    workspace_module.restore_pytest_version(repo, IMAGE)
    # An Agent may modify ignore rules and force-stage files. Exclusion must still
    # hold independently of those mutable rules and of the real repository index.
    (repo / ".gitignore").write_text("# updated ignore rules\n")
    (repo / ".git/info/exclude").write_text("")
    (repo / "tracked.py").write_text("value = 'fixed'\n")
    (repo / "new_code.py").write_text("result = 42\n")
    git(repo, "add", "-f", GENERATED_PATH)
    index = (repo / ".git/index").read_bytes()
    assert GENERATED_PATH in export_patch(repo, base)
    runtime = LocalRuntime()
    patch_path = tmp_path / "model.diff"
    predictions_path = tmp_path / "predictions.jsonl"
    trajectory_path = tmp_path / "trajectory.json"
    result = run_once(
        task=Task(task_id="offline-environment-repair", problem_statement="Synthetic offline task."),
        model=FakeModel([ModelResponse(content="done", tool_call=None, usage=Usage(cost_usd=0.0))]),
        registry=ToolRegistry([BashTool()]), runtime=runtime,
        max_steps=1, budget=None, trajectory_path=str(trajectory_path),
        patch_workspace=repo, base_commit=base, patch_path=patch_path,
        predictions_path=predictions_path, patch_exclude_paths=(GENERATED_PATH,),
    )
    assert result.stop_reason == StopReason.MODEL_FINISHED
    assert runtime.closed
    patch = patch_path.read_text()
    assert f"diff --git a/{GENERATED_PATH} " not in patch
    assert GENERATED_BYTES.decode().strip() not in patch
    assert "+value = 'fixed'" in patch
    assert "new_code.py" in patch
    assert "+result = 42" in patch
    assert ".gitignore" in patch
    assert (repo / GENERATED_PATH).read_bytes() == GENERATED_BYTES
    assert (repo / ".git/index").read_bytes() == index
    assert json.loads(predictions_path.read_text())["model_patch"] == patch
    assert json.loads(trajectory_path.read_text())["patch_pointer"] == str(patch_path.resolve())
