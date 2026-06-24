"""Gymnasium registration for the drone intercept environments."""

from gymnasium.envs.registration import register, registry

ENV_ID = "DroneIntercept3D-v0"
ENV_ID_V2 = "DroneIntercept3D-v2"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="drone_env.envs.drone_intercept_3d:DroneIntercept3DEnv",
    )

if ENV_ID_V2 not in registry:
    register(
        id=ENV_ID_V2,
        entry_point="drone_env.envs.drone_intercept_3d_v2:DroneIntercept3DV2Env",
    )

__all__ = ["ENV_ID", "ENV_ID_V2"]
