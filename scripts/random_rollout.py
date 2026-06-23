from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="evasive")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DroneIntercept3DEnv(target_mode=args.target_mode, render_mode=args.render_mode)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("non-finite rollout value")
                total_reward += reward
                length += 1
            print(
                f"episode={episode} reward={total_reward:.3f} length={length} "
                f"captured={info['captured']} final_distance={info['distance']:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
