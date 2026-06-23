"""Gymnasium registration for the drone intercept environments."""

from gymnasium.envs.registration import register, registry

ENV_ID = "DroneIntercept3D-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="drone_env.envs.drone_intercept_3d:DroneIntercept3DEnv",
    )

__all__ = ["ENV_ID"]
