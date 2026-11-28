"""Aggregate ablation evaluation records and render Markdown comparison tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileSummary:
    profile: str
    total_tasks: int
    resolved_tasks: int
    avg_steps: float
    avg_input_tokens: float
    avg_output_tokens: float
    cache_hit_rate: float
    avg_cost_usd: float | None
    avg_prefix_invalidations: float
    invalid_action_events: int

    @property
    def resolved_rate(self) -> float:
        if self.total_tasks <= 0:
            return 0.0
        return self.resolved_tasks / self.total_tasks


def load_eval_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.eval.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def summarize_records(records: list[dict[str, Any]]) -> list[ProfileSummary]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["profile"])].append(rec)

    summaries: list[ProfileSummary] = []
    for profile, items in grouped.items():
        n = len(items)
        resolved = sum(1 for r in items if r.get("resolved"))
        avg_steps = sum(int(r.get("steps", 0)) for r in items) / n

        total_in = sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in items)
        total_out = sum(int((r.get("usage") or {}).get("output_tokens") or 0) for r in items)
        total_cache = sum(int((r.get("usage") or {}).get("cache_read_tokens") or 0) for r in items)
        cache_hit = (total_cache / total_in) if total_in > 0 else 0.0

        costs = [
            (r.get("usage") or {}).get("cost_usd")
            for r in items
            if (r.get("usage") or {}).get("cost_usd") is not None
        ]
        avg_cost = (sum(float(c) for c in costs) / len(costs)) if len(costs) == n else None

        avg_invalidations = (
            sum(int((r.get("middleware_stats") or {}).get("prefix_invalidations") or 0) for r in items)
            / n
        )
        invalid_events = sum(int(r.get("invalid_action_events") or 0) for r in items)

        summaries.append(
            ProfileSummary(
                profile=profile,
                total_tasks=n,
                resolved_tasks=resolved,
                avg_steps=avg_steps,
                avg_input_tokens=total_in / n,
                avg_output_tokens=total_out / n,
                cache_hit_rate=cache_hit,
                avg_cost_usd=avg_cost,
                avg_prefix_invalidations=avg_invalidations,
                invalid_action_events=invalid_events,
            )
        )
    return summaries


def render_markdown_table(summaries: list[ProfileSummary]) -> str:
    header = (
        "| Profile | Tasks | Resolved Rate | Avg Steps | Avg Input Tokens | "
        "Prefix Cache Hit | Prefix Invalidations/Task | Avg Cost (USD) |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    rows = [header, sep]
    for s in summaries:
        cost_str = f"${s.avg_cost_usd:.4f}" if s.avg_cost_usd is not None else "N/A"
        rows.append(
            f"| `{s.profile}` | {s.total_tasks} | "
            f"{s.resolved_tasks}/{s.total_tasks} ({s.resolved_rate * 100:.1f}%) | "
            f"{s.avg_steps:.2f} | {s.avg_input_tokens:,.0f} | "
            f"{s.cache_hit_rate * 100:.1f}% | {s.avg_prefix_invalidations:.2f} | {cost_str} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze coding-agent ablation evaluation results.")
    parser.add_argument("--results-dir", required=True, help="Directory containing *.eval.json files.")
    args = parser.parse_args(argv)

    records = load_eval_records(Path(args.results_dir))
    if not records:
        raise SystemExit(f"No *.eval.json files found in {args.results_dir}")

    summaries = summarize_records(records)
    print(render_markdown_table(summaries))


if __name__ == "__main__":
    main()
