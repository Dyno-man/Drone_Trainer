from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manual_policy_demo import pursuit_action
from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv, REWARD_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="evasive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = DroneIntercept3DEnv(target_mode=args.target_mode)
    try:
        env.reset(seed=5)
        print("step,total," + ",".join(REWARD_KEYS))
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(pursuit_action(env))
            parts = [f"{info['reward_breakdown'][key]:.4f}" for key in REWARD_KEYS]
            print(f"{step},{reward:.4f}," + ",".join(parts))
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
