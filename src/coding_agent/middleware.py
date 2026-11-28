"""Pluggable Agent Middleware chain for Context Engineering & Cache-Aware Masking.

Implements three research-backed mechanisms without modifying core Agent state-machine invariants:
1. ContextMaskingMiddleware:
   - ``full``: Baseline unmodified history.
   - ``per_step_masking``: Masks tool observations older than the most recent M turns on every step
     (JetBrains Research, "The Complexity Trap", NeurIPS 2025 DL4Code, arXiv:2508.21433).
   - ``epoch_masking``: Advances the masking watermark only once every K steps so intermediate
     turns remain strictly append-only, preserving provider Prompt Prefix / KV Cache hits
     ("Cache Engineering for Agents", arXiv:2610.28577; Manus Context Engineering).
2. EventReminderMiddleware:
   - Appends lightweight, event-driven reminders to the latest tool observation (never mutating
     the leading prompt prefix) to counteract instruction fade-out and read-only/retry loops
     ("Building Effective AI Coding Agents for the Terminal", arXiv:2603.05344).
3. MiddlewareModel:
   - Composes an ordered middleware chain around any ``Model`` implementation via the
     Decorator / Chain-of-Responsibility pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from coding_agent.contracts import ModelResponse

VALID_CONTEXT_STRATEGIES = ("full", "per_step_masking", "epoch_masking")
READ_ONLY_TOOLS = frozenset({"view_file", "search_repo"})


@dataclass
class MiddlewareStats:
    """Observable telemetry for ablation studies (context savings vs prefix cache hits)."""

    total_calls: int = 0
    compaction_events: int = 0
    prefix_invalidations: int = 0
    masked_observations: int = 0
    reminders_injected: int = 0
    chars_before: int = 0
    chars_after: int = 0
    total_input_tokens: int = 0
    total_cache_read_tokens: int = 0
    cache_observed_input_tokens: int = 0
    cache_observed_calls: int = 0
    step_metrics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def char_reduction_ratio(self) -> float:
        if self.chars_before <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.chars_after / self.chars_before))

    @property
    def cache_hit_rate(self) -> float | None:
        """Return cache hit ratio only when cache_read_tokens telemetry is reported.

        Returns ``None`` (unknown) rather than ``0.0`` when the provider does not
        report ``cache_read_tokens``.
        """
        if self.cache_observed_calls == 0 or self.cache_observed_input_tokens <= 0:
            return None
        return self.total_cache_read_tokens / self.cache_observed_input_tokens

    def to_dict(self) -> dict[str, Any]:
        rate = self.cache_hit_rate
        return {
            "total_calls": self.total_calls,
            "compaction_events": self.compaction_events,
            "prefix_invalidations": self.prefix_invalidations,
            "masked_observations": self.masked_observations,
            "reminders_injected": self.reminders_injected,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "char_reduction_ratio": round(self.char_reduction_ratio, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "cache_observed_calls": self.cache_observed_calls,
            "cache_hit_rate": round(rate, 4) if rate is not None else None,
        }


@dataclass
class MiddlewareState:
    """Mutable per-run state shared across middlewares."""

    call_index: int = 0
    max_steps: int | None = None
    stats: MiddlewareStats = field(default_factory=MiddlewareStats)


class AgentMiddleware(Protocol):
    def before_model(
        self,
        history: list[dict[str, Any]],
        state: MiddlewareState,
    ) -> list[dict[str, Any]]:
        ...

    def after_model(
        self,
        history: list[dict[str, Any]],
        response: ModelResponse,
        state: MiddlewareState,
    ) -> ModelResponse:
        ...


def _estimate_history_chars(history: list[dict[str, Any]]) -> int:
    total = 0
    for msg in history:
        for key in ("content", "stdout", "stderr", "error"):
            val = msg.get(key)
            if isinstance(val, str):
                total += len(val)
    return total


def _mask_tool_message(msg: dict[str, Any], strategy: str) -> tuple[dict[str, Any], bool]:
    stdout = msg.get("stdout") or ""
    stderr = msg.get("stderr") or ""
    orig_chars = len(stdout) + len(stderr)
    if orig_chars == 0:
        return dict(msg), False

    rc = msg.get("return_code")
    placeholder = (
        f"[Observation omitted by {strategy}: {orig_chars} chars; return_code={rc}. "
        "Re-read file or re-run command if exact output is needed.]"
    )
    masked = dict(msg)
    masked["stdout"] = placeholder
    masked["stderr"] = ""
    return masked, True


class ContextMaskingMiddleware:
    """Observation masking with optional epoch-based watermarking to protect Prefix Cache."""

    def __init__(
        self,
        strategy: str = "full",
        keep_recent_observations: int = 3,
        mask_epoch_steps: int = 4,
    ):
        if strategy not in VALID_CONTEXT_STRATEGIES:
            raise ValueError(
                f"unsupported context strategy: {strategy!r} "
                f"(expected one of {VALID_CONTEXT_STRATEGIES})"
            )
        if keep_recent_observations < 1:
            raise ValueError("keep_recent_observations must be >= 1")
        if mask_epoch_steps < 1:
            raise ValueError("mask_epoch_steps must be >= 1")

        self.strategy = strategy
        self.keep_recent_observations = keep_recent_observations
        self.mask_epoch_steps = mask_epoch_steps
        self._watermark_tool_count = 0
        self._last_masked_indices: tuple[int, ...] = ()

    def before_model(
        self,
        history: list[dict[str, Any]],
        state: MiddlewareState,
    ) -> list[dict[str, Any]]:
        if self.strategy == "full":
            return [dict(m) for m in history]

        tool_indices = [idx for idx, m in enumerate(history) if m.get("role") == "tool"]
        num_tools = len(tool_indices)
        if num_tools <= self.keep_recent_observations:
            return [dict(m) for m in history]

        eligible_to_mask = num_tools - self.keep_recent_observations
        if self.strategy == "per_step_masking":
            target_mask_count = eligible_to_mask
        else:
            if eligible_to_mask - self._watermark_tool_count >= self.mask_epoch_steps:
                self._watermark_tool_count = eligible_to_mask
            target_mask_count = self._watermark_tool_count

        mask_indices = tuple(tool_indices[:target_mask_count])
        if mask_indices != self._last_masked_indices and mask_indices:
            state.stats.compaction_events += 1
            state.stats.prefix_invalidations += 1
            self._last_masked_indices = mask_indices

        mask_set = set(mask_indices)
        transformed: list[dict[str, Any]] = []
        masked_now = 0
        for idx, msg in enumerate(history):
            if idx in mask_set:
                new_msg, did_mask = _mask_tool_message(msg, self.strategy)
                if did_mask:
                    masked_now += 1
                transformed.append(new_msg)
            else:
                transformed.append(dict(msg))

        state.stats.masked_observations += masked_now
        return transformed

    def after_model(
        self,
        history: list[dict[str, Any]],
        response: ModelResponse,
        state: MiddlewareState,
    ) -> ModelResponse:
        return response


class EventReminderMiddleware:
    """Injects tail-only system reminders on stall/loop/budget events without breaking Prefix Cache."""

    def __init__(
        self,
        read_only_threshold: int = 4,
        repeat_failure_threshold: int = 2,
        low_steps_threshold: int = 2,
    ):
        self.read_only_threshold = read_only_threshold
        self.repeat_failure_threshold = repeat_failure_threshold
        self.low_steps_threshold = low_steps_threshold

    def before_model(
        self,
        history: list[dict[str, Any]],
        state: MiddlewareState,
    ) -> list[dict[str, Any]]:
        if not history or history[-1].get("role") != "tool":
            return history

        reminders: list[str] = []

        recent_tools = self._recent_assistant_tool_calls(history)
        if len(recent_tools) >= self.read_only_threshold:
            tail = recent_tools[-self.read_only_threshold :]
            if all(tc.name in READ_ONLY_TOOLS for tc in tail):
                reminders.append(
                    f"[System Reminder] You have made {self.read_only_threshold} consecutive "
                    "read-only calls without editing code. Formulate a concrete hypothesis "
                    "and test or apply a targeted fix."
                )

        if self._repeated_failing_call(history, self.repeat_failure_threshold):
            reminders.append(
                "[System Reminder] The exact same tool call has failed repeatedly. "
                "Do not repeat the identical command; inspect the error output and change strategy."
            )

        if state.max_steps is not None:
            completed_steps = sum(1 for m in history if m.get("role") == "tool")
            remaining = state.max_steps - completed_steps
            if 0 < remaining <= self.low_steps_threshold:
                reminders.append(
                    f"[System Reminder] Only {remaining} step(s) remaining before max_steps. "
                    "Run final verification tests and summarize your solution."
                )

        if not reminders:
            return history

        updated = [dict(m) for m in history]
        last = dict(updated[-1])
        suffix = "\n" + "\n".join(reminders)
        last["stdout"] = (last.get("stdout") or "") + suffix
        updated[-1] = last
        state.stats.reminders_injected += len(reminders)
        return updated

    def after_model(
        self,
        history: list[dict[str, Any]],
        response: ModelResponse,
        state: MiddlewareState,
    ) -> ModelResponse:
        return response

    @staticmethod
    def _recent_assistant_tool_calls(history: list[dict[str, Any]]) -> list[Any]:
        calls = []
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("tool_call") is not None:
                calls.append(msg["tool_call"])
        return calls

    @staticmethod
    def _repeated_failing_call(history: list[dict[str, Any]], threshold: int) -> bool:
        pairs: list[tuple[str, bool]] = []
        last_sig: str | None = None
        for msg in history:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_call") is not None:
                tc = msg["tool_call"]
                last_sig = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)}"
            elif role == "tool" and last_sig is not None:
                failed = (msg.get("return_code") not in (0, None)) or bool(msg.get("error"))
                pairs.append((last_sig, failed))
                last_sig = None

        if len(pairs) < threshold:
            return False
        tail = pairs[-threshold:]
        first_sig = tail[0][0]
        return all(sig == first_sig and failed for sig, failed in tail)


class MiddlewareModel:
    """Wraps any Model with an ordered AgentMiddleware chain and collects telemetry."""

    def __init__(
        self,
        base_model: Any,
        middlewares: list[AgentMiddleware] | None = None,
        *,
        max_steps: int | None = None,
    ):
        self._base_model = base_model
        self.provider: str = getattr(base_model, "provider", "unknown")
        self.model: str = getattr(base_model, "model", "unknown")
        self._middlewares: list[AgentMiddleware] = list(middlewares or [])
        self.state = MiddlewareState(max_steps=max_steps)

    @property
    def stats(self) -> MiddlewareStats:
        return self.state.stats

    def generate(self, history: list[dict[str, Any]]) -> ModelResponse:
        self.state.call_index += 1
        self.state.stats.total_calls += 1

        chars_before = _estimate_history_chars(history)
        effective_history: list[dict[str, Any]] = [dict(m) for m in history]
        for mw in self._middlewares:
            effective_history = mw.before_model(effective_history, self.state)

        chars_after = _estimate_history_chars(effective_history)
        self.state.stats.chars_before += chars_before
        self.state.stats.chars_after += chars_after

        response = self._base_model.generate(effective_history)

        for mw in reversed(self._middlewares):
            response = mw.after_model(effective_history, response, self.state)

        usage = response.usage
        if usage is not None:
            if usage.input_tokens is not None:
                self.state.stats.total_input_tokens += usage.input_tokens
            if usage.input_tokens is not None and usage.cache_read_tokens is not None:
                self.state.stats.cache_observed_calls += 1
                self.state.stats.cache_observed_input_tokens += usage.input_tokens
                self.state.stats.total_cache_read_tokens += usage.cache_read_tokens

        self.state.stats.step_metrics.append({
            "call_index": self.state.call_index,
            "chars_before": chars_before,
            "chars_after": chars_after,
            "input_tokens": usage.input_tokens if usage else None,
            "cache_read_tokens": usage.cache_read_tokens if usage else None,
        })
        return response


def build_middleware_chain(
    context_strategy: str = "full",
    *,
    keep_recent_observations: int = 3,
    mask_epoch_steps: int = 4,
    enable_reminders: bool = False,
) -> list[AgentMiddleware]:
    """Factory for standard ablation middleware chains."""
    chain: list[AgentMiddleware] = [
        ContextMaskingMiddleware(
            strategy=context_strategy,
            keep_recent_observations=keep_recent_observations,
            mask_epoch_steps=mask_epoch_steps,
        )
    ]
    if enable_reminders:
        chain.append(EventReminderMiddleware())
    return chain
