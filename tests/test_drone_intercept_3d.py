import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_env.envs.drone_intercept_3d import (
    REWARD_KEYS,
    DroneIntercept3DConfig,
    DroneIntercept3DEnv,
)


def make_env(**kwargs) -> DroneIntercept3DEnv:
    config = kwargs.pop("config", DroneIntercept3DConfig(max_steps=20))
    return DroneIntercept3DEnv(config=config, **kwargs)


def set_state(
    env: DroneIntercept3DEnv,
    pursuer_pos=(0.0, 0.0, 10.0),
    target_pos=(10.0, 0.0, 10.0),
    pursuer_vel=(0.0, 0.0, 0.0),
    target_vel=(0.0, 0.0, 0.0),
) -> None:
    env.pursuer_pos = np.array(pursuer_pos, dtype=np.float32)
    env.target_pos = np.array(target_pos, dtype=np.float32)
    env.pursuer_vel = np.array(pursuer_vel, dtype=np.float32)
    env.target_vel = np.array(target_vel, dtype=np.float32)
    env.previous_distance = float(np.linalg.norm(env.target_pos - env.pursuer_pos))
    env.previous_action = np.zeros(3, dtype=np.float32)


def test_reset_returns_valid_obs() -> None:
    env = make_env()
    obs, info = env.reset(seed=1)
    assert obs.shape == (20,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)
    assert info["distance"] > 0.0


def test_step_returns_valid_outputs() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert obs.shape == (20,)
    assert np.isfinite(obs).all()
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_breakdown" in info


def test_seed_reproducibility() -> None:
    env_a = make_env()
    env_b = make_env()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(env_a.pursuer_pos, env_b.pursuer_pos)
    np.testing.assert_allclose(env_a.target_pos, env_b.target_pos)


def test_reset_spawns_in_bounds() -> None:
    env = make_env()
    env.reset(seed=3)
    cfg = env.config
    assert -cfg.world_xy <= env.pursuer_pos[0] <= cfg.world_xy
    assert -cfg.world_xy <= env.pursuer_pos[1] <= cfg.world_xy
    assert env.pursuer_pos[2] > cfg.safe_min_z
    assert -cfg.world_xy <= env.target_pos[0] <= cfg.world_xy
    assert env.target_pos[2] > cfg.safe_min_z
    assert np.linalg.norm(env.target_pos - env.pursuer_pos) > cfg.capture_radius * 3


def test_previous_state_initialized() -> None:
    env = make_env()
    env.reset(seed=4)
    assert env.previous_distance == pytest.approx(np.linalg.norm(env.target_pos - env.pursuer_pos))
    np.testing.assert_array_equal(env.previous_action, np.zeros(3, dtype=np.float32))


def test_pursuer_kinematics() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    set_state(env, target_pos=(50.0, 0.0, 10.0))
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert env.pursuer_vel[0] == pytest.approx(env.config.max_accel * env.config.dt)
    assert env.pursuer_pos[0] == pytest.approx(env.config.max_accel * env.config.dt**2)


def test_pursuer_speed_clipped() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    set_state(env, pursuer_vel=(100.0, 0.0, 0.0), target_pos=(80.0, 0.0, 10.0))
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.linalg.norm(env.pursuer_vel) <= env.config.pursuer_max_speed + 1e-6


def test_target_straight_mode() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=2)
    for _ in range(30):
        env.step(np.zeros(3, dtype=np.float32))
        assert -env.config.world_xy <= env.target_pos[0] <= env.config.world_xy
        assert -env.config.world_xy <= env.target_pos[1] <= env.config.world_xy
        assert env.config.safe_min_z <= env.target_pos[2] <= env.config.safe_max_z


def test_target_evasive_mode() -> None:
    env = make_env(target_mode="evasive")
    env.reset(seed=2)
    set_state(env, target_pos=(5.0, 0.0, 10.0))
    env.step(np.zeros(3, dtype=np.float32))
    assert np.isfinite(env.target_vel).all()
    assert np.linalg.norm(env.target_vel) <= env.config.target_max_speed + 1e-6
    assert np.dot(env.target_vel, env.target_pos - env.pursuer_pos) > 0.0


def test_target_orbit_mode() -> None:
    env = make_env(target_mode="orbit")
    env.reset(seed=2)
    positions = []
    for _ in range(10):
        env.step(np.zeros(3, dtype=np.float32))
        positions.append(env.target_pos.copy())
    deltas = np.ptp(np.array(positions), axis=0)
    assert (deltas > 0.01).all()


def test_reward_breakdown_has_all_components() -> None:
    env = make_env()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert tuple(info["reward_breakdown"].keys()) == REWARD_KEYS


def test_capture_terminates() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    set_state(env, target_pos=(1.0, 0.0, 10.0))
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["captured"]
    assert info["reward_breakdown"]["capture_reward"] == 100.0
    assert reward > 0.0


def test_crash_terminates() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    set_state(env, pursuer_pos=(0.0, 0.0, 0.0))
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["crashed"]
    assert info["reward_breakdown"]["safety_failure_penalty"] <= -100.0


def test_out_of_bounds_terminates() -> None:
    env = make_env(target_mode="straight")
    env.reset(seed=1)
    set_state(env, pursuer_pos=(106.0, 0.0, 10.0), target_pos=(0.0, 0.0, 10.0))
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["out_of_bounds"]
    assert info["reward_breakdown"]["safety_failure_penalty"] <= -75.0


def test_timeout_truncates() -> None:
    env = make_env(config=DroneIntercept3DConfig(max_steps=2), target_mode="straight")
    env.reset(seed=1)
    set_state(env, target_pos=(50.0, 0.0, 10.0))
    env.step(np.zeros(3, dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert truncated


def test_distance_progress_reward_positive_when_closer() -> None:
    env = make_env()
    env.reset(seed=1)
    set_state(env, pursuer_pos=(0.0, 0.0, 10.0), target_pos=(10.0, 0.0, 10.0))
    env.previous_distance = 12.0
    reward, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["distance_progress_reward"] > 0.0
    env.previous_distance = 8.0
    reward, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["distance_progress_reward"] < 0.0


def test_distance_penalty() -> None:
    env = make_env()
    env.reset(seed=1)
    set_state(env, target_pos=(10.0, 0.0, 10.0))
    _, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["distance_penalty"] == pytest.approx(-0.3)


def test_time_penalty() -> None:
    env = make_env()
    env.reset(seed=1)
    _, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["time_penalty"] == pytest.approx(-0.05)


def test_alignment_reward() -> None:
    env = make_env()
    env.reset(seed=1)
    set_state(env, target_pos=(10.0, 0.0, 10.0), pursuer_vel=(1.0, 0.0, 0.0))
    _, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["alignment_reward"] > 0.0
    set_state(env, target_pos=(10.0, 0.0, 10.0), pursuer_vel=(-1.0, 0.0, 0.0))
    _, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["alignment_reward"] < 0.0


def test_closing_speed_reward() -> None:
    env = make_env()
    env.reset(seed=1)
    set_state(env, target_pos=(10.0, 0.0, 10.0), pursuer_vel=(5.0, 0.0, 0.0))
    _, breakdown = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert breakdown["closing_speed_reward"] > 0.0


def test_energy_penalty() -> None:
    env = make_env()
    env.reset(seed=1)
    _, zero = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    _, full = env._compute_reward(np.ones(3, dtype=np.float32), False, False, False)
    assert zero["energy_penalty"] == 0.0
    assert full["energy_penalty"] < 0.0


def test_smoothness_penalty() -> None:
    env = make_env()
    env.reset(seed=1)
    env.previous_action = np.zeros(3, dtype=np.float32)
    _, small = env._compute_reward(np.array([0.1, 0.0, 0.0], dtype=np.float32), False, False, False)
    _, large = env._compute_reward(np.ones(3, dtype=np.float32), False, False, False)
    assert large["smoothness_penalty"] < small["smoothness_penalty"]


def test_altitude_penalty() -> None:
    env = make_env()
    env.reset(seed=1)
    set_state(env, pursuer_pos=(0.0, 0.0, 10.0))
    _, safe = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    set_state(env, pursuer_pos=(0.0, 0.0, 1.0))
    _, low = env._compute_reward(np.zeros(3, dtype=np.float32), False, False, False)
    assert safe["altitude_penalty"] == 0.0
    assert low["altitude_penalty"] < 0.0


def test_render_rgb_array() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = make_env(render_mode="rgb_array")
    try:
        env.reset(seed=1)
        frame = env.render()
        assert frame is not None
        assert frame.shape == (env.config.render_height, env.config.render_width, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_render_close_reopen() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for _ in range(2):
        env = make_env(render_mode="rgb_array")
        env.reset(seed=1)
        assert env.render() is not None
        env.close()
