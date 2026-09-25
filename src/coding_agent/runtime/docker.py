from pathlib import Path
import subprocess
import uuid

from coding_agent.runtime.base import ProcessResult

MAX_OUTPUT_CHARS = 100_000

class DockerRuntime:
    def __init__(
        self,
        workspace: Path,
        image: str,
        process_timeout: float = 120,
        startup_timeout: float = 120,
    ):
        self.workspace = workspace.resolve()
        self.image = image
        self.process_timeout = process_timeout
        self.startup_timeout = startup_timeout
        self.container_id: str | None = None

    def start(self) -> None:
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")

        if self.container_id is not None:
            raise RuntimeError("runtime already started")

        container_name = f"coding-agent-{uuid.uuid4().hex[:8]}"

        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,

                "--network",
                "none",

                "--memory",
                "2g",

                "--cpus",
                "2",

                "--pids-limit",
                "256",

                "-v",
                f"{self.workspace}:/workspace",
                "-w",
                "/workspace",

                self.image,
                "sleep",
                "2h",
            ],
            # --memory 2g
            # → 容器最多用约 2 GB 内存

            # --cpus 2
            # → 最多使用约 2 个 CPU 核心的算力

            # --pids-limit 256
            # → 最多创建 256 个进程/线程级 PID
            # → 防止类似 fork bomb
            capture_output=True,
            text=True,
            check=True,
            timeout=self.startup_timeout,
        )

        self.container_id = result.stdout.strip()

    def run_process(self, argv: list[str]) -> ProcessResult:
        if self.container_id is None:
            raise RuntimeError("container is not started")
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", "/workspace", self.container_id, *argv,],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.process_timeout
            )
            return ProcessResult(
                    return_code=result.returncode,
                    stdout=_truncate_output(result.stdout, MAX_OUTPUT_CHARS),
                    stderr=_truncate_output(result.stderr, MAX_OUTPUT_CHARS),
            )
        except subprocess.TimeoutExpired as exc:
            self.close()

            return ProcessResult(
                return_code=None,
                stdout=_truncate_output(_to_text(exc.stdout), MAX_OUTPUT_CHARS),
                stderr=_truncate_output(_to_text(exc.stderr), MAX_OUTPUT_CHARS),
                timed_out=True,
                error=f"process timed out after {self.process_timeout} seconds",
            )

    def close(self) -> None:
        if self.container_id is None:
            return

        subprocess.run(
            ["docker", "rm", "-f", self.container_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.process_timeout,
        )

        self.container_id = None

def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

def _truncate_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value

    return value[:limit] + "\n...[output truncated]"