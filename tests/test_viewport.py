import numpy as np

from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig, DroneIntercept3DEnv


def set_viewport_state(
    env: DroneIntercept3DEnv,
    target_pos: tuple[float, float, float],
    heading: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> None:
    env.pursuer_pos = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    env.pursuer_vel = np.zeros(3, dtype=np.float32)
    env.target_pos = np.array(target_pos, dtype=np.float32)
    env.target_vel = np.zeros(3, dtype=np.float32)
    env.pursuer_heading = np.array(heading, dtype=np.float32)
    env.steps_since_seen = env.config.lock_memory_steps + 1
    env.has_target_lock = False
    env.target_visible = False
    env._update_visibility_state()


def test_viewport_hidden_target_does_not_expose_fresh_relative_state() -> None:
    env = DroneIntercept3DEnv(
        config=DroneIntercept3DConfig(observation_mode="viewport", lock_memory_steps=2)
    )
    env.reset(seed=1)
    set_viewport_state(env, target_pos=(-10.0, 0.0, 10.0))
    obs = env._get_obs()
    assert not env.target_visible
    assert not env.has_target_lock
    np.testing.assert_allclose(obs[9:15], np.zeros(6), atol=1e-6)


def test_viewport_visible_target_sets_lock_and_finite_observation() -> None:
    env = DroneIntercept3DEnv(config=DroneIntercept3DConfig(observation_mode="viewport"))
    env.reset(seed=1)
    set_viewport_state(env, target_pos=(10.0, 0.0, 10.0))
    obs = env._get_obs()
    assert env.target_visible
    assert env.has_target_lock
    assert obs[15] == 1.0
    assert obs[16] == 1.0
    assert np.isfinite(obs).all()


def test_lock_memory_expires_after_configured_steps() -> None:
    env = DroneIntercept3DEnv(
        config=DroneIntercept3DConfig(observation_mode="viewport", lock_memory_steps=2)
    )
    env.reset(seed=1)
    set_viewport_state(env, target_pos=(10.0, 0.0, 10.0))
    assert env.has_target_lock
    env.target_pos = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    env._update_visibility_state()
    assert env.has_target_lock
    env._update_visibility_state()
    assert env.has_target_lock
    env._update_visibility_state()
    assert not env.has_target_lock
