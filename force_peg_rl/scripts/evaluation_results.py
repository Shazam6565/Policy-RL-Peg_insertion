"""Aggregate and grade episode-level evaluation rows without Isaac Lab."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


def summarize_rows(rows: Sequence[Mapping]) -> dict:
    """Compute the portfolio-facing aggregate metrics for one suite run."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    n = len(rows)
    forces = sorted(float(row["max_contact_force"]) for row in rows)
    return {
        "episodes": n,
        "success_rate_pct": sum(bool(row["success"]) for row in rows) / n * 100.0,
        "median_completion_steps": statistics.median(float(row["episode_steps"]) for row in rows),
        "mean_episode_return": statistics.fmean(float(row["episode_return"]) for row in rows),
        "peak_force_p95_n": _linear_percentile(forces, 0.95),
        "mean_contact_force_n": statistics.fmean(float(row["mean_contact_force"]) for row in rows),
        "force_limit_label_rate_pct": (sum(row["termination_reason"] == "force_limit" for row in rows) / n * 100.0),
        "timeout_rate_pct": sum(row["termination_reason"] == "timeout" for row in rows) / n * 100.0,
    }


def grade_summary(summary: Mapping, rubric: Mapping) -> dict:
    """Grade numeric summary metrics against ``min``/``max`` gates."""
    checks = []
    for metric, gate in rubric.items():
        if metric not in summary:
            raise ValueError(f"Rubric metric {metric!r} is not in the evaluation summary")
        if not isinstance(gate, Mapping) or set(gate) not in ({"min"}, {"max"}):
            raise ValueError(f"Rubric metric {metric!r} must define exactly one of min or max")
        value = float(summary[metric])
        if "min" in gate:
            threshold = float(gate["min"])
            passed = value >= threshold
            comparison = ">="
        else:
            threshold = float(gate["max"])
            passed = value <= threshold
            comparison = "<="
        checks.append(
            {
                "metric": metric,
                "value": value,
                "comparison": comparison,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _linear_percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires at least one value")
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
