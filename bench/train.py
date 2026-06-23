import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import drone_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="evasive")
    parser.add_argument("--model-path", default="ppo_drone_intercept_3d")
    parser.add_argument("--skip-check-env", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make("DroneIntercept3D-v0", target_mode=args.target_mode)
    try:
        if not args.skip_check_env:
            check_env(env.unwrapped)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
        )

        model.learn(total_timesteps=args.timesteps)
        model.save(args.model_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
