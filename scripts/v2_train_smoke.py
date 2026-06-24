from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drone_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--curriculum-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def fallback_random_smoke(env, timesteps: int, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(timesteps):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise RuntimeError("non-finite value during v2 train smoke")
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1)
    return total_reward


def main() -> None:
    import gymnasium as gym

    args = parse_args()
    env = gym.make(drone_env.ENV_ID_V2, curriculum_level=args.curriculum_level)
    try:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            total_reward = fallback_random_smoke(env, args.timesteps, args.seed)
            print(
                "stable-baselines3 is not installed; "
                f"fallback v2 train smoke completed timesteps={args.timesteps} reward={total_reward:.3f}"
            )
            return

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=16,
            batch_size=8,
            gamma=0.98,
            seed=args.seed,
        )
        model.learn(total_timesteps=args.timesteps)
        if args.model_path:
            model.save(args.model_path)
        print(
            f"v2 PPO train smoke completed timesteps={args.timesteps} "
            f"curriculum_level={args.curriculum_level}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
