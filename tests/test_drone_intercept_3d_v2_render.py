"""Tests for DroneIntercept3D-v2 rendering (pygame-based 3D visualizer)."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from drone_env.envs.drone_intercept_3d_v2 import (
    DroneIntercept3DConfigV2,
    DroneIntercept3DV2Env,
    make_curriculum_config,
)


def _make_env(render_mode: str = "rgb_array", **overrides) -> DroneIntercept3DV2Env:
    cfg = DroneIntercept3DConfigV2(
        max_steps=20,
        observation_noise_std=0.0,
        **overrides,
    )
    return DroneIntercept3DV2Env(config=cfg, render_mode=render_mode)


# -- rgb_array mode --


def test_render_rgb_array_shape() -> None:
    env = _make_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        frame = env.render()
        assert frame is not None
        assert frame.shape == (env.config.render_height, env.config.render_width, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_render_rgb_array_after_steps() -> None:
    env = _make_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (env.observation_space.shape[0],)
            frame = env.render()
            assert frame is not None
            assert frame.shape == (env.config.render_height, env.config.render_width, 3)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_render_close_reopen() -> None:
    for _ in range(2):
        env = _make_env(render_mode="rgb_array")
        env.reset(seed=1)
        assert env.render() is not None
        env.close()


def test_render_with_obstacles() -> None:
    env = _make_env(
        render_mode="rgb_array",
        enable_obstacles=True,
        obstacle_count=5,
        curriculum_level=2,
    )
    try:
        env.reset(seed=42)
        assert len(env.obstacles) > 0
        frame = env.render()
        assert frame is not None
        assert frame.shape == (env.config.render_height, env.config.render_width, 3)
    finally:
        env.close()
def test_render_state_contains_drone_body_fields() -> None:
    """_render_state must include yaw and body params so the renderer draws the quadcopter body."""
    env = _make_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        state = env._render_state()
        assert "yaw" in state, "state must include yaw"
        assert isinstance(state["yaw"], float), f"yaw must be float, got {type(state['yaw'])}"
        assert "body_radius" in state, "state must include body_radius"
        assert isinstance(state["body_radius"], float)
        assert "body_height" in state
        assert "rotor_span_radius" in state
        # Verify they match config
        assert state["body_radius"] == env.config.body_radius
        assert state["body_height"] == env.config.body_height
        assert state["rotor_span_radius"] == env.config.rotor_span_radius
    finally:
        env.close()


def test_render_state_contains_obstacles() -> None:
    """When obstacles are configured, _render_state must include them."""
    env = _make_env(
        render_mode="rgb_array",
        enable_obstacles=True,
        obstacle_count=5,
        curriculum_level=2,
    )
    try:
        env.reset(seed=42)
        assert len(env.obstacles) > 0
        state = env._render_state()
        obstacles = state.get("obstacles", ())
        assert len(obstacles) > 0, "_render_state must include nonempty obstacles tuple"
        for obs in obstacles:
            assert "type" in obs
            assert "center" in obs
            assert "radius" in obs
            assert "height" in obs
    finally:
        env.close()


def test_render_state_empty_obstacles_when_disabled() -> None:
    """When no obstacles, _render_state obstacles tuple must be empty."""
    env = _make_env(render_mode="rgb_array", enable_obstacles=False)
    try:
        env.reset(seed=42)
        state = env._render_state()
        assert state.get("obstacles", ()) == ()
    finally:
        env.close()


def test_render_state_contains_viewport_fields() -> None:
    """_render_state must include heading and FOV data for the visibility cone."""
    env = _make_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        state = env._render_state()
        assert "pursuer_heading" in state
        assert "horizontal_fov_deg" in state
        assert "vertical_fov_deg" in state
        assert "viewport_range" in state
        assert state["horizontal_fov_deg"] == env.config.sensor_horizontal_fov_deg
        assert state["viewport_range"] == env.config.sensor_range
    finally:
        env.close()


# -- human mode --


def test_render_human_accepted() -> None:
    """human render_mode must be accepted in __init__."""
    env = _make_env(render_mode="human")
    try:
        env.reset(seed=42)
        assert env.render() is None  # human returns None
        for _ in range(5):
            env.step(env.action_space.sample())
            assert env.render() is None
    finally:
        env.close()


def test_render_human_invalid() -> None:
    """Reject unsupported render_mode."""
    with pytest.raises(ValueError, match="unsupported render_mode"):
        _make_env(render_mode="random")


# -- no render_mode --


def test_render_mode_none() -> None:
    """When render_mode is None, render() returns None."""
    env = _make_env(render_mode=None)
    env.reset(seed=42)
    assert env.render() is None
    env.close()
