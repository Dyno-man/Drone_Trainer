from __future__ import annotations


def config_overrides(mode: str) -> dict[str, float | int | bool | str]:
    """Small environment setting changes that are allowed during research."""
    max_steps = 260 if mode == "quick" else 420
    return {
        "observation_mode": "viewport",
        "require_flythrough_success": True,
        "max_steps": max_steps,
        "lock_memory_steps": 24,
        "min_closing_speed": 0.0,
    }


def scenario_repeats(mode: str) -> int:
    return 1 if mode == "quick" else 3

