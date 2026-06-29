from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drone_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=drone_env.ENV_ID)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="evasive")
    parser.add_argument("--curriculum-level", type=int, default=None)
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym

    args = parse_args()
    kwargs = {"render_mode": args.render_mode}
    if args.curriculum_level is not None:
        kwargs["curriculum_level"] = args.curriculum_level
    if "v2" in args.env_id:
        env = gym.make(args.env_id, **kwargs)
    else:
        env = gym.make(args.env_id, target_mode=args.target_mode, **kwargs)
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
                f"success={info.get('success', info.get('captured', False))} "
                f"collision={info.get('collision', False)} "
                f"final_distance={info.get('distance_to_target', info.get('distance', 0.0)):.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
