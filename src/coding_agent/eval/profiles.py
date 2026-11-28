"""Standard ablation experiment profiles for coding-agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationProfile:
    name: str
    toolset: str
    context_strategy: str
    keep_recent_observations: int = 3
    mask_epoch_steps: int = 4
    enable_reminders: bool = False
    description: str = ""


ABLATION_PROFILES: dict[str, AblationProfile] = {
    "baseline": AblationProfile(
        name="baseline",
        toolset="bash",
        context_strategy="full",
        enable_reminders=False,
        description="M2.5 Baseline: single BashTool + full unmodified context.",
    ),
    "aci": AblationProfile(
        name="aci",
        toolset="aci",
        context_strategy="full",
        enable_reminders=False,
        description="M3 (+ACI): view_file + search_repo + edit_file (AST guardrail) + full context.",
    ),
    "aci_per_step": AblationProfile(
        name="aci_per_step",
        toolset="aci",
        context_strategy="per_step_masking",
        keep_recent_observations=3,
        enable_reminders=True,
        description="M4a (+ACI + Per-step Observation Masking + Event Reminders).",
    ),
    "aci_epoch": AblationProfile(
        name="aci_epoch",
        toolset="aci",
        context_strategy="epoch_masking",
        keep_recent_observations=3,
        mask_epoch_steps=4,
        enable_reminders=True,
        description="M4b (+ACI + Epoch-based Masking for Prefix Cache + Event Reminders).",
    ),
}


def get_profile(name: str) -> AblationProfile:
    if name not in ABLATION_PROFILES:
        valid = ", ".join(ABLATION_PROFILES.keys())
        raise ValueError(f"unknown ablation profile: {name!r} (expected one of: {valid})")
    return ABLATION_PROFILES[name]
