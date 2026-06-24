from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig, DroneIntercept3DEnv
from drone_env.utils.geometry import safe_unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mode", choices=["straight", "evasive", "orbit"], default="straight")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-steps", type=int, default=240)
    return parser.parse_args()


def search_or_pursuit_action(env: DroneIntercept3DEnv) -> np.ndarray:
    if env.target_visible or env.has_target_lock:
        target_estimate = env.last_seen_target_pos
        desired_velocity = safe_unit(target_estimate - env.pursuer_pos) * env.config.pursuer_max_speed
    else:
        phase = env.step_count * 0.12
        sweep = np.array([np.cos(phase), np.sin(phase), 0.15 * np.sin(phase * 0.5)], dtype=np.float32)
        desired_velocity = safe_unit(sweep) * (env.config.pursuer_max_speed * 0.65)
    velocity_error = desired_velocity - env.pursuer_vel
    action = velocity_error / max(env.config.max_accel * env.config.dt, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = DroneIntercept3DConfig(max_steps=args.max_steps, observation_mode="viewport")
    env = DroneIntercept3DEnv(
        config=config,
        target_mode=args.target_mode,
        render_mode="human" if args.render else None,
    )
    acquired = False
    reacquired = False
    lost_after_acquire = False
    try:
        obs, info = env.reset(seed=args.seed)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(search_or_pursuit_action(env))
            total_reward += reward
            if info["target_visible"] and not acquired:
                acquired = True
            if acquired and not info["target_visible"] and not info["has_target_lock"]:
                lost_after_acquire = True
            if lost_after_acquire and info["target_visible"]:
                reacquired = True
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise RuntimeError("non-finite viewport search value")
        print(
            "viewport_search "
            f"target_mode={args.target_mode} acquired={acquired} reacquired={reacquired} "
            f"flythrough_intercept={info['flythrough_intercept']} captured={info['captured']} "
            f"steps={env.step_count} reward={total_reward:.3f} final_distance={info['distance']:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
