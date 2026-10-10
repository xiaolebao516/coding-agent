"""Isolated base-commit checkout and SWE-bench submission artifacts."""

import json
import hashlib
import io
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from tempfile import TemporaryDirectory


FORBIDDEN_NAMES = {"eval.sh", "gold.patch", "test.patch"}
PYTEST_VERSION_FILE = "src/_pytest/_version.py"


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True, env=env,
    ).stdout


def prepare_workspace(source: Path, destination: Path, base_commit: str) -> str:
    """Copy only a committed base tree, never the source's working files."""
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"task workspace already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("task workspace must be outside the source repository")
    if not base_commit or base_commit.startswith("-"):
        raise ValueError("base_commit must be a commit reference")
    base = _git(source, "rev-parse", "--verify", f"{base_commit}^{{commit}}").decode().strip()
    names = _git(source, "ls-tree", "-r", "--name-only", "-z", base).split(b"\0")
    if any(Path(os.fsdecode(name)).name in FORBIDDEN_NAMES for name in names if name):
        raise ValueError("base tree contains forbidden evaluation artifacts")
    destination.mkdir(parents=True)
    _git(destination, "init")
    # A standalone shallow repo works inside Docker without host worktree pointers.
    _git(destination, "fetch", "--no-tags", "--depth=1", source.as_uri(), base)
    _git(destination, "-c", "core.hooksPath=/dev/null", "checkout", "--detach", base)
    return base


def validate_artifact_paths(workspace: Path, *paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("artifact paths must be distinct")
    if any(path.is_relative_to(workspace.resolve()) for path in resolved):
        raise ValueError("artifacts must be outside the agent workspace")
    if any(path.exists() for path in resolved):
        raise ValueError("artifact paths must not overwrite existing files")


def restore_pytest_version(workspace: Path, image: str) -> dict[str, str]:
    """Restore only the installed build artifact from a pinned, local image."""
    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("pytest build artifact requires a pinned digest image")
    workspace = workspace.resolve()
    target = workspace / PYTEST_VERSION_FILE
    if target.exists() or target.is_symlink():
        raise ValueError("pytest build artifact already exists")
    if not target.parent.is_dir() or not target.resolve().is_relative_to(workspace):
        raise ValueError("pytest build artifact must be inside the checkout")
    if _git(workspace, "ls-files", "--", PYTEST_VERSION_FILE):
        raise ValueError("pytest build artifact must not be tracked")
    _git(workspace, "check-ignore", "--", PYTEST_VERSION_FILE)
    created = subprocess.run(
        ["docker", "create", "--pull=never", "--network", "none", image, "sleep", "2h"],
        capture_output=True, text=True, check=True, timeout=120,
    )
    container = created.stdout.strip()
    try:
        copied = subprocess.run(
            ["docker", "cp", f"{container}:/testbed/{PYTEST_VERSION_FILE}", "-"],
            capture_output=True, check=True, timeout=120,
        )
        with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isfile() or members[0].name != "_version.py":
                raise ValueError("unexpected pytest build artifact archive")
            artifact = archive.extractfile(members[0])
            assert artifact is not None
            content = artifact.read()
        target.write_bytes(content)
        # Keep environment state local to Git metadata, outside the submitted source.
        exclude = Path(_git(workspace, "rev-parse", "--git-path", "info/exclude").decode().strip())
        if not exclude.is_absolute():
            exclude = workspace / exclude
        with exclude.open("a", encoding="utf-8") as stream:
            stream.write(f"\n/{PYTEST_VERSION_FILE}\n")
        return {"path": PYTEST_VERSION_FILE, "image": image, "sha256": hashlib.sha256(content).hexdigest()}
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True, check=False, timeout=120)


def export_patch(workspace: Path, base_commit: str, exclude_paths: tuple[str, ...] = ()) -> str:
    """Diff final files against base, including untracked files, without staging."""
    exclusions = []
    for name in exclude_paths:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) == ".":
            raise ValueError("patch exclusions must be non-empty repository-relative paths")
        exclusions.append(f":(top,literal){path}")
    # Use a separate index so staged changes and the repository index are preserved.
    with TemporaryDirectory(prefix="coding-agent-index-") as temp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temp) / "index")}
        _git(workspace, "read-tree", base_commit, env=env)
        _git(
            workspace, "-c", "core.autocrlf=false", "add", "-A", "--", ".",
            ":(exclude)**/eval.sh", ":(exclude)**/gold.patch", ":(exclude)**/test.patch",
            env=env,
        )
        if exclusions:
            # Restore excluded paths to base in this temporary index only. Passing
            # ignored paths directly to git add can fail even with exclude magic.
            _git(workspace, "reset", "--quiet", base_commit, "--", *exclusions, env=env)
        return _git(
            workspace, "diff", "--cached", "--binary", "--no-ext-diff",
            "--no-textconv", base_commit, "--", env=env,
        ).decode("utf-8")


def write_predictions(path: Path, instance_id: str, model_name: str, patch: str) -> None:
    row = {"instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
