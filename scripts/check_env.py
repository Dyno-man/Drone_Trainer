from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv


def main() -> None:
    env = DroneIntercept3DEnv()
    try:
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            print("stable-baselines3 is not installed; running Gymnasium API fallback checks.")
            obs, info = env.reset(seed=123)
            assert obs.shape == env.observation_space.shape
            assert env.observation_space.contains(obs)
            action = env.action_space.sample()
            result = env.step(action)
            assert len(result) == 5
            obs, reward, terminated, truncated, info = result
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "reward_breakdown" in info
            print("Fallback Gymnasium API checks passed.")
        else:
            check_env(env, warn=True)
            print("stable-baselines3 check_env passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
