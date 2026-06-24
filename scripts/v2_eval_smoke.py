from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drone_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym

    args = parse_args()
    env = gym.make(drone_env.ENV_ID_V2, curriculum_level=args.curriculum_level)
    model = None
    if args.model_path:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to load --model-path") from exc
        model = PPO.load(args.model_path)

    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            visible_steps = 0
            clearances: list[float] = []
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("non-finite value during v2 eval smoke")
                total_reward += reward
                steps += 1
                visible_steps += int(info["target_visible"])
                clearances.append(float(info["nearest_obstacle_distance"]))
            row = {
                "seed": seed,
                "curriculum_level": args.curriculum_level,
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("collision", False)),
                "timeout": bool(truncated),
                "reward": round(total_reward, 6),
                "steps": steps,
                "visibility_ratio": round(visible_steps / max(1, steps), 6),
                "mean_clearance": round(float(np.mean(clearances)) if clearances else 0.0, 6),
            }
            rows.append(row)
            print(
                f"episode={episode} seed={seed} reward={total_reward:.3f} steps={steps} "
                f"success={row['success']} collision={row['collision']} timeout={row['timeout']}"
            )
    finally:
        env.close()

    if args.metrics_path:
        path = Path(args.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote_metrics={path}")

    success_rate = sum(int(row["success"]) for row in rows) / max(1, len(rows))
    collision_rate = sum(int(row["collision"]) for row in rows) / max(1, len(rows))
    mean_reward = float(np.mean([float(row["reward"]) for row in rows])) if rows else 0.0
    print(
        f"summary episodes={len(rows)} success_rate={success_rate:.3f} "
        f"collision_rate={collision_rate:.3f} mean_reward={mean_reward:.3f}"
    )


if __name__ == "__main__":
    main()
