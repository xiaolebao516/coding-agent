import json
from typing import Any

from coding_agent.contracts import AgentResult, StopReason, Task, ToolResult
from coding_agent.model import Model
from coding_agent.runtime.base import Runtime
from coding_agent.tools.registry import ToolRegistry
from coding_agent.contracts import Usage


def _add_optional(a, b):
    if a is None or b is None:
        return None
    return a + b


def _merge_usage(total: Usage, current: Usage) -> Usage:
    return Usage(
        input_tokens=_add_optional(
            total.input_tokens,
            current.input_tokens,
        ),
        output_tokens=_add_optional(
            total.output_tokens,
            current.output_tokens,
        ),
        cost_usd=_add_optional(
            total.cost_usd,
            current.cost_usd,
        ),
    )


def _budget_stop_reason(usage: Usage, budget: float) -> StopReason | None:
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
        budget: float,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.budget = budget

    def run(
        self,
        task: Task,
        runtime: Runtime,
    ) -> AgentResult:
        steps = 0
        total_usage = Usage(0, 0, 0.0)
        history:list[dict[str, Any]]= [
            {
                "role": "user",
                "content": task.problem_statement,
            }
        ]

        while steps < self.max_steps:
            final_message = None
            reason = _budget_stop_reason(total_usage, self.budget)
            if reason is not None:
                return AgentResult(
                    reason,
                    final_message=final_message,
                    steps=steps,
                    usage=total_usage,
                )
            try:
                response = self.model.generate(history)
            except Exception as exc:
                return AgentResult(
                    stop_reason=StopReason.MODEL_ERROR,
                    final_message=None,
                    steps=steps,
                    usage=total_usage,
                    )
            total_usage = _merge_usage(total_usage, response.usage)
            final_message = response.content
            if response.tool_call is None:
                return AgentResult(
                    StopReason.MODEL_FINISHED,
                    final_message=response.content,
                    steps=steps,
                    usage=total_usage,
                )

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
                tool_result = tool.execute(tool_call=response.tool_call, runtime=runtime)
            except Exception:
                return AgentResult(
                    StopReason.RUNTIME_ERROR,
                    final_message=None,
                    steps=steps + 1,
                    usage=total_usage,
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

        return AgentResult(
            StopReason.MAX_STEPS,
            final_message=None,
            steps=steps,
            usage=total_usage,
        )
