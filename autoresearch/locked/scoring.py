from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoreResult:
    score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScoringWeights:
    flythrough: float = 100.0
    acquisition: float = 18.0
    steps_to_acquire: float = -0.12
    steps_to_intercept: float = -0.05
    crash: float = -35.0
    out_of_bounds: float = -25.0
    lost_target: float = -20.0
    side_pass_false_success: float = -100.0
    significant_crash_worsening: float = 0.05


DEFAULT_SCORING_WEIGHTS = ScoringWeights()


@dataclass(frozen=True)
class BaselineMetrics:
    crash_rate: float

    @classmethod
    def from_leaderboard(cls, leaderboard_path: Path) -> "BaselineMetrics | None":
        if not leaderboard_path.exists():
            return None
        with leaderboard_path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("accepted") == "true"]
        if not rows:
            return None
        best = max(rows, key=lambda row: float(row.get("score", "0") or 0.0))
        return cls(crash_rate=float(best.get("crash_rate", "0") or 0.0))


def composite_score(
    metrics: dict[str, float],
    weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS,
) -> float:
    return round(
        weights.flythrough * metrics.get("flythrough_success_rate", 0.0)
        + weights.acquisition * metrics.get("first_acquisition_rate", 0.0)
        + weights.steps_to_acquire * metrics.get("mean_steps_to_acquire", 0.0)
        + weights.steps_to_intercept * metrics.get("mean_steps_to_intercept", 0.0)
        + weights.crash * metrics.get("crash_rate", 0.0)
        + weights.out_of_bounds * metrics.get("out_of_bounds_rate", 0.0)
        + weights.lost_target * metrics.get("lost_target_rate", 0.0)
        + weights.side_pass_false_success * metrics.get("side_pass_false_success_rate", 0.0),
        6,
    )


def score_metrics(
    metrics: dict[str, float],
    *,
    tests_passed: bool,
    check_env_passed: bool,
    locked_files_modified: bool,
    baseline_metrics: BaselineMetrics | None = None,
    weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS,
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
        baseline_crash = baseline_metrics.crash_rate
        crash_rate = metrics.get("crash_rate", 0.0)
        if crash_rate > baseline_crash + weights.significant_crash_worsening:
            reasons.append(
                "crash_rate worsened significantly "
                f"({crash_rate:.3f} > {baseline_crash:.3f} + "
                f"{weights.significant_crash_worsening:.3f})"
            )
    return ScoreResult(
        score=composite_score(metrics, weights),
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )
