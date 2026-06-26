from __future__ import annotations

import math
from typing import Any

import numpy as np

from autoresearch.editable.policy_config import POLICY_CONFIG
from drone_env.utils.geometry import safe_unit


def select_action(obs: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
    """Viewport-only heuristic policy used as the editable research baseline."""
    cfg = POLICY_CONFIG
    velocity = np.asarray(obs[3:6], dtype=np.float32)
    heading = safe_unit(np.asarray(obs[6:9], dtype=np.float32))
    relative_pos = np.asarray(obs[9:12], dtype=np.float32)
    relative_vel = np.asarray(obs[12:15], dtype=np.float32)
    target_available = bool(obs[15] > 0.5 or obs[16] > 0.5)

    if target_available:
        desired = (
            cfg.intercept_gain * safe_unit(relative_pos)
            + cfg.velocity_gain * relative_vel
            - cfg.damping_gain * velocity
        )
    else:
        phase = step * 0.19
        lateral = np.array([-heading[1], heading[0], 0.0], dtype=np.float32)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        desired = (
            cfg.search_forward_gain * heading
            + cfg.search_lateral_gain * math.sin(phase) * safe_unit(lateral)
            + np.array([0.0, 0.0, cfg.search_vertical_gain * math.cos(phase * 0.7)])
            - cfg.damping_gain * velocity
        )

    return np.clip(desired, -cfg.max_action, cfg.max_action).astype(np.float32)


def hypothesis() -> str:
    return (
        "Slightly increased intercept and search gains for faster, more aggressive "
        "target pursuit while maintaining stability across all quick-mode scenarios."
    )


def what_to_try_next(metrics: dict[str, float]) -> str:
    if metrics.get("first_acquisition_rate", 0.0) < 0.8:
        return "Try broader search oscillation and a longer lock memory curriculum."
    if metrics.get("flythrough_success_rate", 0.0) < 0.5:
        return "Tune intercept gain and closing-speed shaping for cleaner centerline passes."
    return "Stress test with evasive and orbit scenarios before promoting the recipe."
