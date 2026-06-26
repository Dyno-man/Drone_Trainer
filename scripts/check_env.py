from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drone_env


def assert_info_schema(env_id: str, info: dict) -> None:
    if env_id == drone_env.ENV_ID_V2:
        assert "reward_terms" in info, f"{env_id} info missing v2 reward_terms key: {sorted(info)}"
        return
    if env_id == drone_env.ENV_ID:
        assert "reward_breakdown" in info, f"{env_id} info missing v1 reward_breakdown key: {sorted(info)}"
        return
    assert (
        "reward_breakdown" in info or "reward_terms" in info
    ), f"{env_id} info must include a known reward schema key; got: {sorted(info)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=drone_env.ENV_ID)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym

    args = parse_args()
    env = gym.make(args.env_id)
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
            assert_info_schema(args.env_id, info)
            print("Fallback Gymnasium API checks passed.")
        else:
            check_env(env, warn=True)
            print("stable-baselines3 check_env passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
