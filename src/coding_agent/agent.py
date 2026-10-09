import json
from typing import Any

from coding_agent.contracts import AgentResult, ModelFinishReason, StopReason, Task, ToolResult
from coding_agent.model import Model
from coding_agent.runtime.base import Runtime
from coding_agent.tools.registry import ToolRegistry
from coding_agent.contracts import Usage
from coding_agent.trajectory import TrajectoryEvent, TrajectoryRecorder


def _add_optional(a, b):
    if a is None or b is None:
        return None
    return a + b


def _merge_usage(total: Usage, current: Usage) -> Usage:
    return Usage(
        input_tokens=_add_optional(total.input_tokens, current.input_tokens),
        cache_read_tokens=_add_optional(
            total.cache_read_tokens, current.cache_read_tokens
        ),
        output_tokens=_add_optional(total.output_tokens, current.output_tokens),
        cost_usd=_add_optional(total.cost_usd, current.cost_usd),
    )


def _budget_stop_reason(
    usage: Usage,
    budget: float | None,
) -> StopReason | None:
    if budget is None:
        return None

    if usage.cost_usd is None:
        return StopReason.BUDGET_UNKNOWN

    if usage.cost_usd >= budget:
        return StopReason.BUDGET_EXHAUSTED

    return None


class Agent:
    def __init__(
        self,
        model: Model,
        tool_registry: ToolRegistry,
        max_steps: int,
        budget: float | None,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.budget = budget

    def run(
        self, task: Task, runtime: Runtime, recorder: TrajectoryRecorder
    ) -> AgentResult:
        recorder.start(
            task_id=task.task_id,
            provider=self.model.provider,
            model=self.model.model,
            max_steps=self.max_steps,
            budget=self.budget,
        )
        steps = 0
        total_usage = Usage(
            input_tokens=0, cache_read_tokens=0, output_tokens=0, cost_usd=0.0
        )
        history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": task.problem_statement,
            }
        ]
        result: AgentResult | None = None

        while steps < self.max_steps:
            final_message = None
            reason = _budget_stop_reason(total_usage, self.budget)
            if reason is not None:
                result = AgentResult(
                    reason,
                    final_message=final_message,
                    steps=steps,
                    usage=total_usage,
                )
                break
            try:
                response = self.model.generate(history)
                recorder.record(
                    TrajectoryEvent(
                        type="model_response",
                        data={
                            "content": response.content,
                            "tool_call": (
                                {
                                    "id": response.tool_call.id,
                                    "name": response.tool_call.name,
                                    "arguments": response.tool_call.arguments,
                                }
                                if response.tool_call is not None
                                else None
                            ),
                            "usage": response.usage,
                            "finish_reason": response.finish_reason.value
                        },
                    )
                )
            except Exception as exc:
                recorder.record(
                    TrajectoryEvent(
                        type="model_error",
                        data={"error_type": type(exc).__name__, "message": str(exc)},
                    )
                )
                result = AgentResult(
                    stop_reason=StopReason.MODEL_ERROR,
                    final_message=None,
                    steps=steps,
                    usage=total_usage,
                )
                break
            total_usage = _merge_usage(total_usage, response.usage)
            if response.finish_reason == ModelFinishReason.OUTPUT_TRUNCATED:
                result = AgentResult(
                    StopReason.OUTPUT_TRUNCATED,
                    final_message=None,
                    steps=steps,
                    usage=total_usage,
                )
                break
            final_message = response.content
            if response.tool_call is None:
                result = AgentResult(
                    StopReason.MODEL_FINISHED,
                    final_message=response.content,
                    steps=steps,
                    usage=total_usage,
                )
                break

            tool = self.tool_registry.get(response.tool_call.name)
            if tool is None:
                tool_result = ToolResult(
                    tool_call_id=response.tool_call.id,
                    return_code=None,
                    stdout="",
                    stderr="",
                    timed_out=False,
                    error=f"unknown tool: {response.tool_call.name}",
                )
                recorder.record(
                    TrajectoryEvent(
                        type="tool_result",
                        data={
                            "tool_call_id": tool_result.tool_call_id,
                            "return_code": tool_result.return_code,
                            "stdout": tool_result.stdout,
                            "stderr": tool_result.stderr,
                            "timed_out": tool_result.timed_out,
                            "error": tool_result.error,
                        },
                    )
                )
                history.append(
                    {
                        "role": "assistant",
                        "tool_call": response.tool_call,
                    }
                )

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "return_code": tool_result.return_code,
                        "stdout": tool_result.stdout,
                        "stderr": tool_result.stderr,
                        "timed_out": tool_result.timed_out,
                        "error": tool_result.error,
                    }
                )
                steps += 1
                continue
            history.append(
                {
                    "role": "assistant",
                    "tool_call": response.tool_call,
                }
            )
            try:
                tool_result = tool.execute(
                    tool_call=response.tool_call, runtime=runtime
                )
                recorder.record(
                    TrajectoryEvent(
                        type="tool_result",
                        data={
                            "tool_call_id": tool_result.tool_call_id,
                            "return_code": tool_result.return_code,
                            "stdout": tool_result.stdout,
                            "stderr": tool_result.stderr,
                            "timed_out": tool_result.timed_out,
                            "error": tool_result.error,
                        },
                    )
                )
            except Exception as exc:
                recorder.record(
                    TrajectoryEvent(
                        type="runtime_error",
                        data={
                            "tool_call_id": response.tool_call.id,
                            "tool_name": response.tool_call.name,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                )
                result = AgentResult(
                    StopReason.RUNTIME_ERROR,
                    final_message=None,
                    steps=steps + 1,
                    usage=total_usage,
                )
                break
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "return_code": tool_result.return_code,
                    "stdout": tool_result.stdout,
                    "stderr": tool_result.stderr,
                    "timed_out": tool_result.timed_out,
                    "error": tool_result.error,
                }
            )
            steps += 1

        if result is None:
            result = AgentResult(
                StopReason.MAX_STEPS,
                final_message=None,
                steps=steps,
                usage=total_usage,
            )
        recorder.record(
            TrajectoryEvent(
                type="terminal",
                data={
                    "stop_reason": result.stop_reason.value,
                    "steps": result.steps,
                },
            )
        )
        recorder.finalize(
            steps=result.steps,
            total_usage=result.usage,
            stop_reason=result.stop_reason,
            final_message=result.final_message,
        )
        return result
