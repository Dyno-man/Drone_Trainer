import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import drone_env
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_drone_intercept_3d")
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="straight")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(
        "DroneIntercept3D-v0",
        target_mode=args.target_mode,
        render_mode=args.render_mode,
    )
    model = PPO.load(args.model_path)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            done = False
            total = 0.0
            length = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                length += 1
                done = terminated or truncated
            print(
                f"episode={episode} reward={total:.3f} length={length} "
                f"captured={info['captured']} final_distance={info['distance']:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
