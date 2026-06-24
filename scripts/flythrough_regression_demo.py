from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig, DroneIntercept3DEnv


def configure_state(
    env: DroneIntercept3DEnv,
    pursuer_pos: tuple[float, float, float],
    previous_pursuer_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> None:
    env.pursuer_pos = np.array(pursuer_pos, dtype=np.float32)
    env.previous_pursuer_pos = np.array(previous_pursuer_pos, dtype=np.float32)
    env.target_pos = np.array(target_pos, dtype=np.float32)
    env.pursuer_vel = np.array([18.0, 0.0, 0.0], dtype=np.float32)
    env.target_vel = np.zeros(3, dtype=np.float32)
    env.pursuer_heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.previous_distance = float(np.linalg.norm(env.target_pos - env.pursuer_pos))
    env.previous_action = np.zeros(3, dtype=np.float32)
    env._update_visibility_state()


def run_case(
    name: str,
    pursuer_pos: tuple[float, float, float],
    previous_pursuer_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    expected_intercept: bool,
) -> bool:
    config = DroneIntercept3DConfig(max_steps=5, observation_mode="viewport")
    env = DroneIntercept3DEnv(config=config, target_mode="straight")
    try:
        env.reset(seed=5)
        configure_state(env, pursuer_pos, previous_pursuer_pos, target_pos)
        _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
        ok = info["flythrough_intercept"] is expected_intercept
        ok = ok and (terminated is expected_intercept)
        print(
            f"{name}: {'PASS' if ok else 'FAIL'} "
            f"flythrough_intercept={info['flythrough_intercept']} terminated={terminated} "
            f"distance={info['distance']:.3f}"
        )
        return ok
    finally:
        env.close()


def main() -> None:
    direct_ok = run_case(
        "direct_flythrough",
        pursuer_pos=(-1.0, 0.0, 10.0),
        previous_pursuer_pos=(-3.0, 0.0, 10.0),
        target_pos=(0.0, 0.0, 10.0),
        expected_intercept=True,
    )
    side_ok = run_case(
        "side_pass_rejected",
        pursuer_pos=(-1.0, 3.5, 10.0),
        previous_pursuer_pos=(-3.0, 3.5, 10.0),
        target_pos=(0.0, 0.0, 10.0),
        expected_intercept=False,
    )
    if not (direct_ok and side_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
