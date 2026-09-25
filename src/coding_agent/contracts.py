from dataclasses import dataclass, field
from typing import Any
from enum import Enum


@dataclass
class Task:
    task_id: str
    problem_statement: str

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

@dataclass
class ModelResponse:
    content: str | None
    tool_call: ToolCall | None
    usage: Usage = field(default_factory=Usage)

class StopReason(str, Enum):
    MODEL_FINISHED = "model_finished"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"
    RUNTIME_ERROR = "runtime_error"

@dataclass
class AgentResult:
    stop_reason: StopReason
    final_message: str | None
    steps: int
    usage: Usage = field(default_factory=Usage)

