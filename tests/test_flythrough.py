import numpy as np

from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig, DroneIntercept3DEnv


def set_flythrough_state(
    env: DroneIntercept3DEnv,
    pursuer_pos: tuple[float, float, float],
    previous_pursuer_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    pursuer_vel: tuple[float, float, float] = (18.0, 0.0, 0.0),
) -> None:
    env.pursuer_pos = np.array(pursuer_pos, dtype=np.float32)
    env.previous_pursuer_pos = np.array(previous_pursuer_pos, dtype=np.float32)
    env.target_pos = np.array(target_pos, dtype=np.float32)
    env.pursuer_vel = np.array(pursuer_vel, dtype=np.float32)
    env.target_vel = np.zeros(3, dtype=np.float32)
    env.pursuer_heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.previous_distance = float(np.linalg.norm(env.target_pos - env.pursuer_pos))
    env.previous_action = np.zeros(3, dtype=np.float32)
    env._update_visibility_state()


def test_direct_flythrough_success_terminates() -> None:
    env = DroneIntercept3DEnv(config=DroneIntercept3DConfig(max_steps=20), target_mode="straight")
    env.reset(seed=1)
    set_flythrough_state(
        env,
        pursuer_pos=(-1.0, 0.0, 10.0),
        previous_pursuer_pos=(-3.0, 0.0, 10.0),
        target_pos=(0.0, 0.0, 10.0),
    )
    _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["flythrough_intercept"]
    assert info["captured"]
    assert info["reward_breakdown"]["flythrough_intercept_reward"] > 0.0
    assert reward > 0.0


def test_side_pass_is_rejected() -> None:
    env = DroneIntercept3DEnv(config=DroneIntercept3DConfig(max_steps=20), target_mode="straight")
    env.reset(seed=1)
    set_flythrough_state(
        env,
        pursuer_pos=(-1.0, 3.5, 10.0),
        previous_pursuer_pos=(-3.0, 3.5, 10.0),
        target_pos=(0.0, 0.0, 10.0),
    )
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert not info["flythrough_intercept"]
    assert not info["captured"]
