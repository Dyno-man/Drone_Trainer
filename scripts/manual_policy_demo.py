from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv
from drone_env.utils.geometry import safe_unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="straight")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def pursuit_action(env: DroneIntercept3DEnv) -> np.ndarray:
    to_target = env.target_pos - env.pursuer_pos
    desired_velocity = safe_unit(to_target) * env.config.pursuer_max_speed
    velocity_error = desired_velocity - env.pursuer_vel
    action = velocity_error / max(env.config.max_accel * env.config.dt, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    env = DroneIntercept3DEnv(target_mode=args.target_mode, render_mode=args.render_mode)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            start_distance = info["distance"]
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(pursuit_action(env))
                total_reward += reward
                length += 1
            improved = info["distance"] < start_distance
            print(
                f"episode={episode} reward={total_reward:.3f} length={length} "
                f"captured={info['captured']} start_distance={start_distance:.3f} "
                f"final_distance={info['distance']:.3f} improved={improved}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
