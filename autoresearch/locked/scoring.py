from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


def composite_score(metrics: dict[str, float]) -> float:
    return round(
        100.0 * metrics.get("flythrough_success_rate", 0.0)
        + 18.0 * metrics.get("first_acquisition_rate", 0.0)
        - 0.12 * metrics.get("mean_steps_to_acquire", 0.0)
        - 0.05 * metrics.get("mean_steps_to_intercept", 0.0)
        - 35.0 * metrics.get("crash_rate", 0.0)
        - 25.0 * metrics.get("out_of_bounds_rate", 0.0)
        - 20.0 * metrics.get("lost_target_rate", 0.0)
        - 100.0 * metrics.get("side_pass_false_success_rate", 0.0),
        6,
    )


def score_metrics(
    metrics: dict[str, float],
    *,
    tests_passed: bool,
    check_env_passed: bool,
    locked_files_modified: bool,
    baseline_metrics: dict[str, float] | None = None,
) -> ScoreResult:
    reasons: list[str] = []
    if metrics.get("side_pass_false_success_rate", 0.0) > 0.0:
        reasons.append("side_pass_false_success_rate > 0")
    if not tests_passed:
        reasons.append("tests failed")
    if not check_env_passed:
        reasons.append("check_env failed")
    if locked_files_modified:
        reasons.append("locked evaluator or benchmark files modified")
    if baseline_metrics is not None:
        baseline_crash = baseline_metrics.get("crash_rate", 0.0)
        crash_rate = metrics.get("crash_rate", 0.0)
        if crash_rate > baseline_crash + 0.05:
            reasons.append(
                f"crash_rate worsened significantly ({crash_rate:.3f} > {baseline_crash:.3f} + 0.050)"
            )
    return ScoreResult(
        score=composite_score(metrics),
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )

