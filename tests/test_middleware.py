"""Offline unit tests for M4 pluggable middleware chain."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from coding_agent.contracts import ModelResponse, ToolCall, Usage
from coding_agent.middleware import (
    ContextMaskingMiddleware,
    EventReminderMiddleware,
    MiddlewareModel,
    build_middleware_chain,
)


class RecordingFakeModel:
    provider = "fake"
    model = "fake-m4"

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.seen_histories: list[list[dict[str, Any]]] = []

    def generate(self, history: list[dict[str, Any]]) -> ModelResponse:
        self.seen_histories.append(copy.deepcopy(history))
        return self.responses[len(self.seen_histories) - 1]


def _make_history(num_tool_turns: int, *, tool_name: str = "bash", rc: int = 0) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [{"role": "user", "content": "Fix bug"}]
    for i in range(1, num_tool_turns + 1):
        history.append({
            "role": "assistant",
            "tool_call": ToolCall(f"c{i}", tool_name, {"command": f"cmd_{i}"}),
        })
        history.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "return_code": rc,
            "stdout": f"output_{i}_" + ("x" * 600),
            "stderr": "",
            "timed_out": False,
            "error": None,
        })
    return history


def test_per_step_masking_masks_old_observations_and_preserves_recent():
    base = RecordingFakeModel([
        ModelResponse(content="done", tool_call=None, usage=Usage(input_tokens=500, cache_read_tokens=100, output_tokens=20))
    ])
    wrapped = MiddlewareModel(
        base,
        [ContextMaskingMiddleware(strategy="per_step_masking", keep_recent_observations=2)],
    )
    history = _make_history(4)
    original_snapshot = copy.deepcopy(history)

    wrapped.generate(history)

    assert history == original_snapshot

    seen = base.seen_histories[0]
    tool_msgs = [m for m in seen if m["role"] == "tool"]
    assert "Observation omitted by per_step_masking" in tool_msgs[0]["stdout"]
    assert "Observation omitted by per_step_masking" in tool_msgs[1]["stdout"]
    assert tool_msgs[2]["stdout"].startswith("output_3_")
    assert tool_msgs[3]["stdout"].startswith("output_4_")
    assert wrapped.stats.compaction_events == 1
    assert wrapped.stats.char_reduction_ratio > 0.35
    assert wrapped.stats.cache_hit_rate == pytest.approx(0.2)


def test_missing_cache_telemetry_reports_none_instead_of_zero():
    base = RecordingFakeModel([
        ModelResponse(content="done", tool_call=None, usage=Usage(input_tokens=500, cache_read_tokens=None, output_tokens=20))
    ])
    wrapped = MiddlewareModel(base, [ContextMaskingMiddleware(strategy="full")])
    wrapped.generate([{"role": "user", "content": "hi"}])

    assert wrapped.stats.cache_hit_rate is None
    assert wrapped.stats.to_dict()["cache_hit_rate"] is None


def test_epoch_masking_reduces_prefix_invalidations_compared_to_per_step():
    responses = [
        ModelResponse(content=None, tool_call=ToolCall("x", "bash", {"command": "ls"}), usage=Usage(input_tokens=100, cache_read_tokens=80, output_tokens=10))
        for _ in range(6)
    ]
    per_step_model = MiddlewareModel(
        RecordingFakeModel(responses),
        [ContextMaskingMiddleware(strategy="per_step_masking", keep_recent_observations=2)],
    )
    epoch_model = MiddlewareModel(
        RecordingFakeModel(responses),
        [ContextMaskingMiddleware(strategy="epoch_masking", keep_recent_observations=2, mask_epoch_steps=3)],
    )

    for turns in range(3, 9):
        hist = _make_history(turns)
        per_step_model.generate(hist)
        epoch_model.generate(hist)

    assert per_step_model.stats.prefix_invalidations == 6
    assert epoch_model.stats.prefix_invalidations == 2


def test_event_reminder_detects_read_only_loop_and_low_steps_without_mutating_prefix():
    base = RecordingFakeModel([
        ModelResponse(content="done", tool_call=None, usage=Usage(input_tokens=200, cache_read_tokens=150, output_tokens=10))
    ])
    wrapped = MiddlewareModel(
        base,
        [EventReminderMiddleware(read_only_threshold=3, low_steps_threshold=2)],
        max_steps=5,
    )
    history = _make_history(4, tool_name="view_file", rc=0)
    wrapped.generate(history)

    seen = base.seen_histories[0]
    assert seen[0]["content"] == "Fix bug"
    assert "consecutive read-only calls" in seen[-1]["stdout"]
    assert "Only 1 step(s) remaining" in seen[-1]["stdout"]
    assert wrapped.stats.reminders_injected == 2


def test_event_reminder_detects_repeated_identical_failure():
    base = RecordingFakeModel([
        ModelResponse(content="done", tool_call=None, usage=Usage(input_tokens=100, cache_read_tokens=0, output_tokens=10))
    ])
    wrapped = MiddlewareModel(
        base,
        [EventReminderMiddleware(repeat_failure_threshold=2)],
    )
    history = [
        {"role": "user", "content": "Fix tests"},
        {"role": "assistant", "tool_call": ToolCall("c1", "bash", {"command": "pytest"})},
        {"role": "tool", "tool_call_id": "c1", "return_code": 1, "stdout": "", "stderr": "fail", "timed_out": False, "error": None},
        {"role": "assistant", "tool_call": ToolCall("c2", "bash", {"command": "pytest"})},
        {"role": "tool", "tool_call_id": "c2", "return_code": 1, "stdout": "", "stderr": "fail", "timed_out": False, "error": None},
    ]
    wrapped.generate(history)
    assert "failed repeatedly" in base.seen_histories[0][-1]["stdout"]


def test_build_middleware_chain_validation():
    chain = build_middleware_chain("epoch_masking", enable_reminders=True)
    assert len(chain) == 2

    with pytest.raises(ValueError, match="unsupported context strategy"):
        build_middleware_chain("invalid_strategy")
