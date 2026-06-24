from __future__ import annotations

from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig


def build_reward_weights() -> dict[str, float]:
    """Return reward weights to test in the next research trial."""
    weights = dict(DroneIntercept3DConfig().reward_weights)
    weights.update(
        {
            "capture_reward": 80.0,
            "flythrough_intercept_reward": 180.0,
            "first_acquisition_reward": 8.0,
            "visibility_reward": 0.2,
            "lost_target_penalty": -0.12,
            "aim_through_target_reward": 1.5,
            "distance_progress_reward": 4.5,
            "alignment_reward": 1.8,
            "closing_speed_reward": 0.7,
            "time_penalty": -0.04,
        }
    )
    return weights

