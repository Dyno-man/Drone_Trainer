from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    target_mode: str
    seed: int
    pursuer_pos: tuple[float, float, float]
    pursuer_vel: tuple[float, float, float]
    pursuer_heading: tuple[float, float, float]
    target_pos: tuple[float, float, float]
    target_vel: tuple[float, float, float]
    max_steps: int | None = None


QUICK_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        name="visible_straight_centerline",
        target_mode="straight",
        seed=100,
        pursuer_pos=(-35.0, 0.0, 18.0),
        pursuer_vel=(7.0, 0.0, 0.0),
        pursuer_heading=(1.0, 0.0, 0.0),
        target_pos=(15.0, 0.0, 18.0),
        target_vel=(4.0, 0.0, 0.0),
    ),
    BenchmarkScenario(
        name="hidden_then_acquire_left",
        target_mode="straight",
        seed=101,
        pursuer_pos=(-45.0, -8.0, 20.0),
        pursuer_vel=(5.0, 1.0, 0.0),
        pursuer_heading=(1.0, 0.05, 0.0),
        target_pos=(5.0, 18.0, 20.0),
        target_vel=(3.0, -0.5, 0.0),
    ),
    BenchmarkScenario(
        name="evasive_high_viewport",
        target_mode="evasive",
        seed=102,
        pursuer_pos=(-25.0, 18.0, 13.0),
        pursuer_vel=(6.0, -2.0, 0.8),
        pursuer_heading=(1.0, -0.2, 0.05),
        target_pos=(18.0, 2.0, 22.0),
        target_vel=(0.0, 5.0, -0.2),
    ),
    BenchmarkScenario(
        name="orbit_crossing",
        target_mode="orbit",
        seed=103,
        pursuer_pos=(-38.0, 12.0, 24.0),
        pursuer_vel=(7.0, -1.0, 0.0),
        pursuer_heading=(1.0, -0.1, 0.0),
        target_pos=(8.0, 4.0, 24.0),
        target_vel=(1.0, 5.5, 0.0),
    ),
)

MEDIUM_EXTRA_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        name="low_altitude_evasive",
        target_mode="evasive",
        seed=201,
        pursuer_pos=(-30.0, -22.0, 8.0),
        pursuer_vel=(6.0, 1.0, 0.2),
        pursuer_heading=(1.0, 0.15, 0.05),
        target_pos=(12.0, -4.0, 12.0),
        target_vel=(2.0, 4.0, 0.5),
    ),
    BenchmarkScenario(
        name="far_hidden_search",
        target_mode="straight",
        seed=202,
        pursuer_pos=(-58.0, 18.0, 21.0),
        pursuer_vel=(5.0, -1.0, 0.0),
        pursuer_heading=(1.0, -0.05, 0.0),
        target_pos=(22.0, -20.0, 21.0),
        target_vel=(4.0, 0.0, 0.0),
    ),
    BenchmarkScenario(
        name="vertical_offset_intercept",
        target_mode="straight",
        seed=203,
        pursuer_pos=(-32.0, 4.0, 12.0),
        pursuer_vel=(7.0, 0.0, 1.0),
        pursuer_heading=(1.0, 0.0, 0.15),
        target_pos=(12.0, 4.0, 25.0),
        target_vel=(3.0, 0.0, -0.3),
    ),
)

SIDE_PASS_PROBES: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        name="side_pass_above_radius",
        target_mode="straight",
        seed=900,
        pursuer_pos=(-1.0, 3.5, 18.0),
        pursuer_vel=(18.0, 0.0, 0.0),
        pursuer_heading=(1.0, 0.0, 0.0),
        target_pos=(0.0, 0.0, 18.0),
        target_vel=(0.0, 0.0, 0.0),
        max_steps=1,
    ),
)


def scenarios_for_mode(mode: str) -> tuple[BenchmarkScenario, ...]:
    if mode == "quick":
        return QUICK_SCENARIOS
    if mode == "medium":
        return QUICK_SCENARIOS + MEDIUM_EXTRA_SCENARIOS
    raise ValueError(f"Unsupported mode: {mode}")

