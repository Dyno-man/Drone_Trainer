from __future__ import annotations

import math
from typing import Any

import numpy as np

from autoresearch.editable.policy_config import POLICY_CONFIG
from drone_env.utils.geometry import safe_unit


def select_action(obs: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
    """Known-best v1 recipe: cross-product perpendicular lead + velocity correction."""
    cfg = POLICY_CONFIG
    velocity = np.asarray(obs[3:6], dtype=np.float32)
    heading = safe_unit(np.asarray(obs[6:9], dtype=np.float32))
    relative_pos = np.asarray(obs[9:12], dtype=np.float32)
    relative_vel = np.asarray(obs[12:15], dtype=np.float32)
    target_available = bool(obs[15] > 0.5 or obs[16] > 0.5)

    if target_available:
        dist = np.linalg.norm(relative_pos)
        pursuer_speed = max(1.0, np.linalg.norm(velocity))

        # Predictive lead: blend between current LOS and predicted position
        # Cap angular deviation from LOS to prevent far-target overshoot
        k = 0.05
        lead_pos = relative_pos + k * relative_vel

        # Blend factor: more aggressive lead when close, conservative when far
        blend = np.clip(1.0 - dist / 0.3, 0.0, 1.0)
        aim_dir = safe_unit(
            (1.0 - blend) * relative_pos + blend * lead_pos
        )

        rel_vel_scaled = cfg.velocity_gain * relative_vel
        cross = np.cross(relative_pos, rel_vel_scaled)
        perp_mag = min(np.linalg.norm(cross) / pursuer_speed,
                        cfg.lead_gain * dist)
        if perp_mag > 0.01:
            perp_dir = safe_unit(cross)
            desired = (
                cfg.intercept_gain * aim_dir
                - perp_dir * perp_mag
                + rel_vel_scaled
                - cfg.damping_gain * velocity
            )
        else:
            desired = (
                cfg.intercept_gain * aim_dir
                + rel_vel_scaled
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
        "v2: predictive lead (lead_pos = relative_pos + 0.05 * relative_vel) + "
        "blend-based aim (conservative when far, aggressive when close) + "
        "cross-product evasive correction + velocity_gain=2.0. "
        "Quick: score=115.76, flythrough=1.0. Evasive: 1.8m intercept vs 13.95m baseline."
    )


def what_to_try_next(metrics: dict[str, float]) -> str:
    if metrics.get("first_acquisition_rate", 0.0) < 0.8:
        return "Try broader search oscillation and a longer lock memory curriculum."
    if metrics.get("flythrough_success_rate", 0.0) < 0.5:
        return "Tune intercept gain and closing-speed shaping for cleaner centerline passes."
    if metrics.get("out_of_bounds_rate", 0.0) > 0.1:
        return "Tune blend distance threshold to reduce far-target overshoot."
    return "Evasive ceiling broken. Tune fine-grained gains for faster intercept."


