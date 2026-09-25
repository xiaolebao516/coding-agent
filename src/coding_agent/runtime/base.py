from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProcessResult:
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None


class Runtime(Protocol):
    def start(self) -> None:
        ...

    def run_process(self, argv: list[str]) -> ProcessResult:
        ...

    def close(self) -> None:
        ...