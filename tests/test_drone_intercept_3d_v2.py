import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drone_env
from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv
from drone_env.envs.drone_intercept_3d_v2 import (
    OBSERVATION_SCHEMA,
    V2_INFO_KEYS,
    V2_REWARD_KEYS,
    DroneIntercept3DConfigV2,
    DroneIntercept3DV2Env,
    Obstacle,
    make_curriculum_config,
    obstacle_clearance,
    obstacle_collides,
)


def make_env(config: DroneIntercept3DConfigV2 | None = None) -> DroneIntercept3DV2Env:
    return DroneIntercept3DV2Env(config=config or DroneIntercept3DConfigV2(max_steps=30))


def set_v2_state(
    env: DroneIntercept3DV2Env,
    agent_pos=(0.0, 0.0, 2.0),
    target_pos=(10.0, 0.0, 8.0),
    agent_vel=(0.0, 0.0, 0.0),
    target_vel=(0.0, 0.0, 0.0),
    yaw=0.0,
) -> None:
    env.reset_with_state(
        agent_pos=agent_pos,
        target_pos=target_pos,
        agent_vel=agent_vel,
        target_vel=target_vel,
        yaw=yaw,
    )


def obstacle_signature(env: DroneIntercept3DV2Env) -> list[tuple[str, tuple[float, ...], float, float]]:
    return [
        (
            obstacle.kind,
            tuple(np.round(obstacle.center.astype(float), 5)),
            round(obstacle.radius, 5),
            round(obstacle.height, 5),
        )
        for obstacle in env.obstacles
    ]


def test_v1_import_and_registration_still_work() -> None:
    assert DroneIntercept3DEnv is not None
    env_v0 = gym.make("DroneIntercept3D-v0")
    env_v2 = gym.make("DroneIntercept3D-v2")
    try:
        obs0, _ = env_v0.reset(seed=1)
        obs2, _ = env_v2.reset(seed=1)
        assert obs0.shape == env_v0.observation_space.shape
        assert obs2.shape == env_v2.observation_space.shape
    finally:
        env_v0.close()
        env_v2.close()


def test_observation_schema_is_stable() -> None:
    assert OBSERVATION_SCHEMA[0] == ("agent_position", 0, 3)
    assert OBSERVATION_SCHEMA[-1] == ("nearest_obstacle_clearance", 28, 29)
    assert OBSERVATION_SCHEMA[-1][2] == DroneIntercept3DConfigV2().observation_size


def test_reset_determinism_includes_obstacle_layout() -> None:
    cfg = DroneIntercept3DConfigV2(
        max_steps=30,
        enable_obstacles=True,
        obstacle_count=4,
        curriculum_level=2,
    )
    env_a = make_env(cfg)
    env_b = make_env(cfg)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(env_a.agent_pos, env_b.agent_pos)
    np.testing.assert_allclose(env_a.target_pos, env_b.target_pos)
    assert obstacle_signature(env_a) == obstacle_signature(env_b)


def test_different_seeds_usually_change_reset_layout() -> None:
    cfg = DroneIntercept3DConfigV2(enable_obstacles=True, obstacle_count=2, curriculum_level=2)
    env_a = make_env(cfg)
    env_b = make_env(cfg)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=124)
    assert not np.allclose(obs_a, obs_b)


def test_space_compliance_and_info_keys() -> None:
    env = make_env()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (4,)
    assert all(key in info for key in V2_INFO_KEYS)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert tuple(info["reward_terms"].keys()) == V2_REWARD_KEYS


def test_launch_and_target_spawn_geometry() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=30)
    env = make_env(cfg)
    env.reset(seed=2)
    assert abs(env.agent_pos[0]) <= cfg.launch_xy_jitter
    assert abs(env.agent_pos[1]) <= cfg.launch_xy_jitter
    assert cfg.launch_height_range[0] <= env.agent_pos[2] <= cfg.launch_height_range[1]
    assert env.target_pos[2] > env.agent_pos[2]
    assert np.linalg.norm(env.target_pos - env.agent_pos) >= cfg.target_distance_range[0] * 0.8


def test_dynamics_speed_yaw_and_hover_caps() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=120, max_speed=3.0, max_accel=10.0)
    env = make_env(cfg)
    env.reset(seed=3)
    start_z = float(env.agent_pos[2])
    env.step(np.zeros(4, dtype=np.float32))
    assert env.agent_pos[2] == pytest.approx(start_z)
    for _ in range(60):
        env.step(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert np.linalg.norm(env.agent_vel) <= cfg.max_speed + 1e-6
    assert abs(env.agent_yaw_rate) <= cfg.max_yaw_rate + 1e-6
    assert env.agent_yaw != pytest.approx(0.0)


def test_out_of_bounds_terminates() -> None:
    env = make_env(DroneIntercept3DConfigV2(max_steps=30))
    env.reset(seed=4)
    set_v2_state(env, agent_pos=(100.0, 0.0, 2.0), target_pos=(0.0, 0.0, 8.0))
    _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["out_of_bounds"]
    assert info["reward_terms"]["out_of_bounds"] < 0.0
    assert reward < 0.0


def test_rotor_span_collision_and_clearance_geometry() -> None:
    cfg = DroneIntercept3DConfigV2(rotor_span_radius=0.65)
    obstacle = Obstacle("cylinder", center=np.array([0.8, 0.0, 0.0], dtype=np.float32), radius=0.2, height=5.0)
    assert obstacle_collides(np.array([0.0, 0.0, 2.0], dtype=np.float32), obstacle, cfg.rotor_span_radius)
    assert obstacle_clearance(np.array([0.0, 0.0, 2.0], dtype=np.float32), obstacle, cfg.rotor_span_radius) < 0.0
    far = Obstacle("cylinder", center=np.array([1.2, 0.0, 0.0], dtype=np.float32), radius=0.2, height=5.0)
    assert not obstacle_collides(np.array([0.0, 0.0, 2.0], dtype=np.float32), far, cfg.rotor_span_radius)


def test_collision_terminates_and_reports_clearance() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=30, enable_obstacles=True, obstacle_count=0)
    env = make_env(cfg)
    env.reset(seed=5)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 8.0))
    env.obstacles = [Obstacle("cylinder", center=np.array([0.5, 0.0, 0.0], dtype=np.float32), radius=0.2, height=5.0)]
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated
    assert info["collision"]
    assert info["nearest_obstacle_distance"] <= 0.0
    assert info["reward_terms"]["collision"] < 0.0
    assert reward < 0.0


def test_obstacle_generation_respects_spawn_clearance() -> None:
    cfg = DroneIntercept3DConfigV2(
        enable_obstacles=True,
        obstacle_count=5,
        curriculum_level=2,
        spawn_clearance=2.0,
        target_spawn_clearance=2.0,
    )
    for seed in range(10):
        env = make_env(cfg)
        env.reset(seed=seed)
        for obstacle in env.obstacles:
            assert obstacle_clearance(env.agent_pos, obstacle, cfg.rotor_span_radius) >= cfg.spawn_clearance
            assert obstacle_clearance(env.target_pos, obstacle, cfg.rotor_span_radius) >= cfg.target_spawn_clearance


def test_visibility_range_fov_and_hidden_observation() -> None:
    cfg = DroneIntercept3DConfigV2(
        max_steps=30,
        enable_fov_limits=True,
        sensor_range=20.0,
        sensor_pitch_bias_deg=0.0,
    )
    env = make_env(cfg)
    env.reset(seed=6)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 2.0), yaw=0.0)
    assert env.target_visible
    obs = env._get_obs()
    assert obs[9] == 1.0
    assert not np.allclose(obs[10:13], 0.0)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(-10.0, 0.0, 2.0), yaw=0.0)
    assert not env.target_visible
    obs = env._get_obs()
    assert obs[9] == 0.0
    np.testing.assert_allclose(obs[10:13], np.zeros(3), atol=1e-6)


def test_upward_pitch_bias_changes_visibility() -> None:
    target = (10.0, 0.0, 7.77)
    narrow_flat = DroneIntercept3DConfigV2(
        enable_fov_limits=True,
        sensor_horizontal_fov_deg=20.0,
        sensor_vertical_fov_deg=20.0,
        sensor_pitch_bias_deg=0.0,
    )
    narrow_up = DroneIntercept3DConfigV2(
        enable_fov_limits=True,
        sensor_horizontal_fov_deg=20.0,
        sensor_vertical_fov_deg=20.0,
        sensor_pitch_bias_deg=30.0,
    )
    env_flat = make_env(narrow_flat)
    env_up = make_env(narrow_up)
    env_flat.reset(seed=7)
    env_up.reset(seed=7)
    set_v2_state(env_flat, agent_pos=(0.0, 0.0, 2.0), target_pos=target, yaw=0.0)
    set_v2_state(env_up, agent_pos=(0.0, 0.0, 2.0), target_pos=target, yaw=0.0)
    assert not env_flat.target_visible
    assert env_up.target_visible


def test_last_seen_persists_after_target_hidden() -> None:
    cfg = DroneIntercept3DConfigV2(enable_fov_limits=True, sensor_pitch_bias_deg=0.0)
    env = make_env(cfg)
    env.reset(seed=8)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 2.0), yaw=0.0)
    assert env.target_visible
    seen = env.last_seen_target_pos.copy()
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(-10.0, 0.0, 2.0), yaw=0.0)
    obs = env._get_obs()
    assert not env.target_visible
    np.testing.assert_allclose(env.last_seen_target_pos, seen)
    np.testing.assert_allclose(obs[10:13], np.zeros(3), atol=1e-6)
    assert not np.allclose(obs[13:16], 0.0)


def test_obstacle_ray_detects_forward_obstacle() -> None:
    cfg = DroneIntercept3DConfigV2(enable_obstacles=True, obstacle_ray_count=8, sensor_range=20.0)
    env = make_env(cfg)
    env.reset(seed=9)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 2.0), yaw=0.0)
    env.obstacles = [Obstacle("cylinder", center=np.array([5.0, 0.0, 0.0], dtype=np.float32), radius=0.5, height=6.0)]
    rays = env.obstacle_ray_distances()
    assert rays[4] < 1.0
    assert rays[4] == pytest.approx((5.0 - 0.5 - cfg.rotor_span_radius) / cfg.sensor_range)


def test_occlusion_blocks_target_visibility() -> None:
    cfg = DroneIntercept3DConfigV2(enable_occlusion=True, sensor_range=30.0)
    env = make_env(cfg)
    env.reset(seed=10)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 2.0), yaw=0.0)
    assert env.target_visible
    env.obstacles = [Obstacle("cylinder", center=np.array([5.0, 0.0, 0.0], dtype=np.float32), radius=0.8, height=5.0)]
    env._update_visibility()
    assert not env.target_visible


def test_reward_sum_and_signs() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=30, enable_fov_limits=True, sensor_pitch_bias_deg=0.0)
    env = make_env(cfg)
    env.reset(seed=11)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(10.0, 0.0, 2.0), yaw=0.0)
    env.previous_distance = 12.0
    reward, terms = env._compute_reward(np.zeros(4, dtype=np.float32), False, False, False, cfg.sensor_range)
    assert reward == pytest.approx(sum(terms.values()))
    assert terms["distance_progress"] > 0.0
    assert terms["target_visible"] > 0.0
    env.previous_distance = 8.0
    _, terms = env._compute_reward(np.ones(4, dtype=np.float32), False, False, False, 0.5)
    assert terms["distance_progress"] < 0.0
    assert terms["control"] < 0.0
    assert terms["clearance"] < 0.0


def test_reacquisition_reward_requires_prior_loss() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=30, enable_fov_limits=True, sensor_pitch_bias_deg=0.0)
    env = make_env(cfg)
    env.reset(seed=12)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(-10.0, 0.0, 2.0), yaw=0.0)
    env.has_seen_target = False
    env.time_since_seen = env.config.max_steps + 1
    env.last_seen_target_pos = np.zeros(3, dtype=np.float32)
    assert not env.target_visible
    env.target_pos = np.array([10.0, 0.0, 2.0], dtype=np.float32)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert info["target_visible"]
    assert info["reward_terms"]["reacquisition"] == 0.0
    env.target_pos = np.array([-10.0, 0.0, 2.0], dtype=np.float32)
    env.step(np.zeros(4, dtype=np.float32))
    env.target_pos = np.array([10.0, 0.0, 2.0], dtype=np.float32)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert info["reward_terms"]["reacquisition"] > 0.0


def test_success_requires_holding_capture_radius() -> None:
    cfg = DroneIntercept3DConfigV2(max_steps=30, success_hold_steps=3, capture_radius=3.0)
    env = make_env(cfg)
    env.reset(seed=13)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(2.0, 0.0, 2.0), yaw=0.0)
    for expected_hold in (1, 2):
        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert not terminated
        assert info["success_hold_steps"] == expected_hold
    _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["success"]
    assert info["success_hold_steps"] == 3
    assert info["reward_terms"]["success"] > 0.0
    assert reward > 0.0


def test_timeout_truncates() -> None:
    env = make_env(DroneIntercept3DConfigV2(max_steps=2))
    env.reset(seed=14)
    set_v2_state(env, agent_pos=(0.0, 0.0, 2.0), target_pos=(20.0, 0.0, 10.0), yaw=np.pi)
    env.step(np.zeros(4, dtype=np.float32))
    _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["time_limit"]


def test_curriculum_progressively_adds_difficulty() -> None:
    levels = [make_curriculum_config(level) for level in range(6)]
    assert levels[0].target_behavior == "hover"
    assert levels[1].target_behavior == "straight"
    assert not levels[1].enable_obstacles
    assert levels[2].enable_obstacles
    assert levels[3].enable_fov_limits
    assert levels[4].enable_occlusion
    assert levels[5].observation_noise_std > 0.0


def test_random_policy_rollout_no_nan() -> None:
    env = make_env(make_curriculum_config(2, max_steps=40))
    for episode in range(3):
        obs, _ = env.reset(seed=100 + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
