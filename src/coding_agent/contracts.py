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

@dataclass(kw_only=True)
class Usage:
    input_tokens: int | None = None  # total prompt tokens, including cache hits
    cache_read_tokens: int | None = None  # subset of input_tokens served from cache
    output_tokens: int | None = None
    cost_usd: float | None = None  # computed from a price table; None = unknown

class ModelFinishReason(str, Enum):
    FINISHED = "finished"
    OUTPUT_TRUNCATED = "output_truncated"
    # Provider refused/aborted (safety block, content filter, ...): not recoverable here.
    PROVIDER_STOPPED = "provider_stopped"
    # The model produced an action we cannot execute (bad JSON args, several tool
    # calls, malformed call). Recoverable: tell the model and let it try again.
    INVALID_ACTION = "invalid_action"


@dataclass
class ModelResponse:
    content: str | None
    tool_call: ToolCall | None
    usage: Usage = field(default_factory=Usage)
    finish_reason: ModelFinishReason = ModelFinishReason.FINISHED
    stop_detail: str | None = None  # diagnostic label, never provider credentials

class StopReason(str, Enum):
    MODEL_FINISHED = "model_finished"
    OUTPUT_TRUNCATED = "output_truncated"
    PROVIDER_STOPPED = "provider_stopped"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BUDGET_UNKNOWN = "budget_unknown"
    MODEL_ERROR = "model_error"
    RUNTIME_ERROR = "runtime_error"

@dataclass
class AgentResult:
    stop_reason: StopReason
    final_message: str | None
    steps: int
    usage: Usage = field(default_factory=Usage)

