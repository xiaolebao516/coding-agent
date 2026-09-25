from pathlib import Path

from coding_agent.runtime.docker import DockerRuntime


def test_docker_runtime_success(tmp_path: Path):
    runtime = DockerRuntime(
        workspace=tmp_path,
        image="python:3.12-slim",
        process_timeout=5,
    )

    try:
        runtime.start()

        result = runtime.run_process(
            ["bash", "-lc", "echo hello"]
        )

        assert result.return_code == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.timed_out is False
        assert result.error is None

    finally:
        runtime.close()


def test_docker_runtime_nonzero_exit(tmp_path: Path):
    runtime = DockerRuntime(
        workspace=tmp_path,
        image="python:3.12-slim",
        process_timeout=5,
    )

    try:
        runtime.start()

        result = runtime.run_process(
            ["bash", "-lc", "echo boom >&2; exit 7"]
        )

        assert result.return_code == 7
        assert result.stdout == ""
        assert result.stderr == "boom\n"
        assert result.timed_out is False
        assert result.error is None

    finally:
        runtime.close()


def test_docker_runtime_timeout(tmp_path: Path):
    runtime = DockerRuntime(
        workspace=tmp_path,
        image="python:3.12-slim",
        process_timeout=1,
    )

    runtime.start()

    result = runtime.run_process(
        ["bash", "-lc", "sleep 5"]
    )

    assert result.return_code is None
    assert result.timed_out is True
    assert result.error is not None
    assert runtime.container_id is None

def test_docker_runtime_truncates_large_output(tmp_path: Path):
    runtime = DockerRuntime(
        workspace=tmp_path,
        image="python:3.12-slim",
        process_timeout=5,
    )

    try:
        runtime.start()

        result = runtime.run_process(
            ["python", "-c", "print('x' * 120000)"]
        )

        assert result.return_code == 0
        assert len(result.stdout) < 120000
        assert "[output truncated]" in result.stdout

    finally:
        runtime.close()