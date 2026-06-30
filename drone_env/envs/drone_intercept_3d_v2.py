from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_env.utils.geometry import (
    angle_between_unit_vectors,
    clip_vector_norm,
    distance,
    in_bounds,
    norm,
    safe_unit,
)
from drone_env.utils.obstacle_geometry import (
    Obstacle,
    obstacle_clearance,
    obstacle_collides,
    segment_occluded,
)

TargetBehavior = Literal["hover", "straight", "random_patrol"]

V2_REWARD_KEYS = (
    "distance_progress",
    "target_visible",
    "reacquisition",
    "capture_hold",
    "success",
    "clearance",
    "launch_altitude",
    "time",
    "control",
    "collision",
    "out_of_bounds",
)

V2_INFO_KEYS = (
    "distance_to_target",
    "target_visible",
    "collision",
    "success",
    "success_hold_steps",
    "reward_terms",
    "nearest_obstacle_distance",
    "out_of_bounds",
    "time_limit",
    "curriculum_level",
    "obstacle_count",
)

# Stable observation schema for DroneIntercept3D-v2:
#  0:3   agent position [x/world_xy, y/world_xy, z/world_z]
#  3:6   agent velocity / max_speed
#  6     sin(yaw)
#  7     cos(yaw)
#  8     yaw_rate / max_yaw_rate
#  9     target_visible flag
# 10:13  visible target relative position normalized, or zeros if hidden
# 13:16  last seen target relative position normalized, or zeros before first sighting
# 16     normalized time since target last seen
# 17:25  obstacle ray distances, each in [0, 1]
# 25     previous distance to target / max_distance
# 26     success hold steps / success_hold_steps
# 27     step count / max_steps
# 28     nearest obstacle clearance / sensor_range, clipped to [-1, 1]
OBSERVATION_SCHEMA: tuple[tuple[str, int, int], ...] = (
    ("agent_position", 0, 3),
    ("agent_velocity", 3, 6),
    ("yaw_sin", 6, 7),
    ("yaw_cos", 7, 8),
    ("yaw_rate", 8, 9),
    ("target_visible", 9, 10),
    ("target_relative_position", 10, 13),
    ("last_seen_relative_position", 13, 16),
    ("time_since_seen", 16, 17),
    ("obstacle_rays", 17, 25),
    ("previous_target_distance", 25, 26),
    ("success_hold_fraction", 26, 27),
    ("step_fraction", 27, 28),
    ("nearest_obstacle_clearance", 28, 29),
)
OBSERVATION_NOISE_PROTECTED_END = next(
    end for name, _, end in OBSERVATION_SCHEMA if name == "target_visible"
)


@dataclass(frozen=True)
class DroneIntercept3DConfigV2:
    world_xy: float = 80.0
    world_z: float = 45.0
    render_width: int = 960
    render_height: int = 720
    bounds_margin: float = 0.0
    dt: float = 0.1
    max_steps: int = 500
    curriculum_level: int = 0
    max_speed: float = 14.0
    max_accel: float = 8.0
    max_yaw_rate: float = np.deg2rad(90.0)
    body_radius: float = 0.18
    body_height: float = 0.45
    rotor_span_radius: float = 0.65
    launch_xy_jitter: float = 2.0
    launch_height_range: tuple[float, float] = (1.2, 1.8)
    target_distance_range: tuple[float, float] = (18.0, 30.0)
    target_altitude_gap_range: tuple[float, float] = (8.0, 16.0)
    target_max_speed: float = 4.0
    target_behavior: TargetBehavior = "hover"
    sensor_range: float = 45.0
    sensor_horizontal_fov_deg: float = 90.0
    sensor_vertical_fov_deg: float = 70.0
    sensor_pitch_bias_deg: float = 18.0
    obstacle_ray_count: int = 8
    obstacle_count: int = 0
    obstacle_min_radius: float = 0.35
    obstacle_max_radius: float = 0.9
    obstacle_min_height: float = 7.0
    obstacle_max_height: float = 20.0
    spawn_clearance: float = 4.0
    target_spawn_clearance: float = 4.0
    enable_obstacles: bool = False
    enable_fov_limits: bool = False
    enable_occlusion: bool = False
    # Curriculum level 5 enables this by default; callers may also opt in for
    # robustness experiments at lower levels.
    observation_noise_std: float = 0.0
    capture_radius: float = 3.0
    success_hold_steps: int = 8
    clearance_penalty_distance: float = 2.0
    launch_altitude_window_steps: int = 30
    launch_altitude_target: float = 4.0
    terminate_on_collision: bool = True
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance_progress": 3.0,
            "target_visible": 0.2,
            "reacquisition": 1.5,
            "capture_hold": 0.6,
            "success": 80.0,
            "clearance": -1.0,
            "launch_altitude": 0.08,
            "time": -0.03,
            "control": -0.015,
            "collision": -80.0,
            "out_of_bounds": -60.0,
        }
    )

    @property
    def observation_size(self) -> int:
        return 21 + self.obstacle_ray_count

_LEVEL_UPDATES: dict[int, dict[str, Any]] = {
    1: {"target_behavior": "straight", "target_max_speed": 3.0},
    2: {"enable_obstacles": True, "obstacle_count": 3},
    3: {"enable_fov_limits": True},
    4: {"enable_occlusion": True, "obstacle_count": 5},
    5: {
        "target_behavior": "random_patrol",
        "obstacle_count": 7,
        "observation_noise_std": 0.01,
        "target_max_speed": 4.5,
    },
}

def make_curriculum_config(level: int, **overrides: Any) -> DroneIntercept3DConfigV2:
    """Build the staged v2 curriculum.

    Level 0 is a stationary target in open space. Later levels add straight
    target motion, obstacles, field-of-view limits, occlusion, and finally
    random patrol motion with light observation noise.
    """
    if level < 0 or level > 5:
        raise ValueError("curriculum level must be between 0 and 5")
    cfg = DroneIntercept3DConfigV2(curriculum_level=level)
    for lvl in range(1, level + 1):
        if lvl in _LEVEL_UPDATES:
            cfg = replace(cfg, **_LEVEL_UPDATES[lvl])
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def yaw_to_forward(yaw: float, pitch_bias_rad: float = 0.0) -> np.ndarray:
    horizontal = float(np.cos(pitch_bias_rad))
    return safe_unit(
        np.array(
            [
                np.cos(yaw) * horizontal,
                np.sin(yaw) * horizontal,
                np.sin(pitch_bias_rad),
            ],
            dtype=np.float32,
        )
    )


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class DroneIntercept3DV2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: DroneIntercept3DConfigV2 | None = None,
        curriculum_level: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if config is None:
            config = make_curriculum_config(0 if curriculum_level is None else curriculum_level)
        elif curriculum_level is not None:
            config = replace(config, curriculum_level=curriculum_level)
        self.config = config
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(config.observation_size, dtype=np.float32),
            high=np.ones(config.observation_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.agent_yaw = 0.0
        self.agent_yaw_rate = 0.0
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self.obstacles: list[Obstacle] = []
        self.last_seen_target_pos = np.zeros(3, dtype=np.float32)
        self.has_seen_target = False
        self.time_since_seen = config.max_steps + 1
        self.target_visible = False
        self.previous_target_visible = False
        self._had_seen_before_visibility = False
        self.previous_distance = 0.0
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.success_hold_steps = 0
        self.step_count = 0
        self.last_reward_terms = {key: 0.0 for key in V2_REWARD_KEYS}
        self.last_info: dict[str, Any] = {}
        self._target_turn_countdown = 0
        self._renderer: DroneRenderer | None = None

    @property
    def max_distance(self) -> float:
        cfg = self.config
        return float(np.sqrt((2.0 * cfg.world_xy) ** 2 + (2.0 * cfg.world_xy) ** 2 + cfg.world_z**2))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.step_count = 0
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.agent_yaw_rate = 0.0
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.success_hold_steps = 0
        self.has_seen_target = False
        self.time_since_seen = self.config.max_steps + 1
        self.last_seen_target_pos = np.zeros(3, dtype=np.float32)
        self.previous_target_visible = False
        self._had_seen_before_visibility = False
        self.target_visible = False
        self._target_turn_countdown = int(self.np_random.integers(10, 40))

        self.agent_pos = self._sample_launch_position()
        self.target_pos = self._sample_target_position()
        direction = self.target_pos - self.agent_pos
        self.agent_yaw = _normalize_angle(
            float(np.arctan2(direction[1], direction[0]) + self.np_random.uniform(-0.25, 0.25))
        )
        self.target_vel = self._initial_target_velocity()
        self.obstacles = self._generate_obstacles()
        self._update_visibility()
        self.previous_distance = distance(self.agent_pos, self.target_pos)
        self.last_reward_terms = {key: 0.0 for key in V2_REWARD_KEYS}
        nearest_clearance = self.nearest_obstacle_clearance()
        self.last_info = self._make_info(
            False,
            False,
            False,
            False,
            self.previous_distance,
            nearest_clearance,
        )
        return self._get_obs(), self.last_info.copy()

    def reset_with_state(
        self,
        *,
        agent_pos: tuple[float, float, float] | np.ndarray,
        target_pos: tuple[float, float, float] | np.ndarray,
        agent_vel: tuple[float, float, float] | np.ndarray | None = None,
        target_vel: tuple[float, float, float] | np.ndarray | None = None,
        yaw: float = 0.0,
        yaw_rate: float = 0.0,
        preserve_target_memory: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Install a deterministic scenario state and refresh derived fields.

        This is mainly intended for tests and scripted scenarios that need exact
        geometry without duplicating the environment's internal bookkeeping.
        """
        self.agent_pos = np.asarray(agent_pos, dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.agent_vel = (
            np.zeros(3, dtype=np.float32)
            if agent_vel is None
            else np.asarray(agent_vel, dtype=np.float32)
        )
        self.target_vel = (
            np.zeros(3, dtype=np.float32)
            if target_vel is None
            else np.asarray(target_vel, dtype=np.float32)
        )
        self.agent_yaw = _normalize_angle(float(yaw))
        self.agent_yaw_rate = float(yaw_rate)
        if not preserve_target_memory:
            self.has_seen_target = False
            self.time_since_seen = self.config.max_steps + 1
            self.last_seen_target_pos = np.zeros(3, dtype=np.float32)
        self.previous_distance = distance(self.agent_pos, self.target_pos)
        self.previous_target_visible = self.target_visible
        self._had_seen_before_visibility = self.has_seen_target
        self._update_visibility()
        nearest_clearance = self.nearest_obstacle_clearance()
        self.last_info = self._make_info(
            False,
            False,
            False,
            False,
            self.previous_distance,
            nearest_clearance,
        )
        return self._get_obs(), self.last_info.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        self.step_count += 1
        self.previous_target_visible = self.target_visible
        self._had_seen_before_visibility = self.has_seen_target
        self._integrate_agent(action)
        self._integrate_target()
        self._update_visibility()

        nearest_clearance = self.nearest_obstacle_clearance()
        collision = bool(nearest_clearance <= 0.0)
        out_of_bounds = not in_bounds(self.agent_pos, cfg.world_xy, cfg.world_z, cfg.bounds_margin, include_floor=True)
        current_distance = distance(self.agent_pos, self.target_pos)
        in_capture = current_distance <= cfg.capture_radius
        self.success_hold_steps = self.success_hold_steps + 1 if in_capture and not collision else 0
        success = self.success_hold_steps >= cfg.success_hold_steps
        terminated = bool(success or out_of_bounds or (collision and cfg.terminate_on_collision))
        truncated = bool(self.step_count >= cfg.max_steps and not terminated)

        reward, reward_terms = self._compute_reward(
            action,
            collision,
            out_of_bounds,
            success,
            nearest_clearance,
            current_distance,
        )
        self.previous_distance = current_distance
        self.previous_action = action.copy()
        self.last_reward_terms = reward_terms
        self.last_info = self._make_info(
            collision,
            out_of_bounds,
            success,
            truncated,
            current_distance,
            nearest_clearance,
        )
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self.last_info.copy()

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from drone_env.utils.rendering import DroneRenderer

            self._renderer = DroneRenderer(
                self.config.render_width,
                self.config.render_height,
                self.config.world_xy,
                self.config.world_z,
                self.render_mode,
            )
        state = self._render_state()
        return self._renderer.draw(state)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


    def _sample_launch_position(self) -> np.ndarray:
        cfg = self.config
        return np.array(
            [
                self.np_random.uniform(-cfg.launch_xy_jitter, cfg.launch_xy_jitter),
                self.np_random.uniform(-cfg.launch_xy_jitter, cfg.launch_xy_jitter),
                self.np_random.uniform(*cfg.launch_height_range),
            ],
            dtype=np.float32,
        )

    def _sample_target_position(self) -> np.ndarray:
        cfg = self.config
        for _ in range(1000):
            yaw = float(self.np_random.uniform(-np.pi, np.pi))
            horizontal_distance = float(self.np_random.uniform(*cfg.target_distance_range))
            altitude_gap = float(self.np_random.uniform(*cfg.target_altitude_gap_range))
            candidate = self.agent_pos + np.array(
                [np.cos(yaw) * horizontal_distance, np.sin(yaw) * horizontal_distance, altitude_gap],
                dtype=np.float32,
            )
            candidate[0] = np.clip(candidate[0], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
            candidate[1] = np.clip(candidate[1], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
            candidate[2] = np.clip(candidate[2], cfg.launch_height_range[1] + 2.0, cfg.world_z * 0.9)
            if distance(candidate, self.agent_pos) >= cfg.target_distance_range[0] * 0.8:
                return candidate.astype(np.float32)
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        horizontal_distance = float(self.np_random.uniform(*cfg.target_distance_range))
        altitude_gap = float(self.np_random.uniform(*cfg.target_altitude_gap_range))
        fallback = np.array(
            [np.cos(yaw) * horizontal_distance, np.sin(yaw) * horizontal_distance, altitude_gap],
            dtype=np.float32,
        )
        fallback = self.agent_pos + fallback
        fallback[0] = np.clip(fallback[0], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
        fallback[1] = np.clip(fallback[1], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
        fallback[2] = np.clip(fallback[2], cfg.launch_height_range[1] + 2.0, cfg.world_z * 0.9)
        return fallback.astype(np.float32)

    def _initial_target_velocity(self) -> np.ndarray:
        cfg = self.config
        if cfg.target_behavior == "hover":
            return np.zeros(3, dtype=np.float32)
        direction = safe_unit(self.np_random.normal(0.0, 1.0, size=3).astype(np.float32))
        direction[2] *= 0.25
        direction = safe_unit(direction)
        return direction * cfg.target_max_speed

    def _generate_obstacles(self) -> list[Obstacle]:
        cfg = self.config
        if not cfg.enable_obstacles or cfg.obstacle_count <= 0:
            return []

        is_tree_mode = cfg.curriculum_level >= 2
        obstacles: list[Obstacle] = []
        attempts = 0
        max_attempts = cfg.obstacle_count * 400

        while len(obstacles) < cfg.obstacle_count and attempts < max_attempts:
            attempts += 1
            radius = float(self.np_random.uniform(cfg.obstacle_min_radius, cfg.obstacle_max_radius))
            height = float(self.np_random.uniform(cfg.obstacle_min_height, cfg.obstacle_max_height))

            center = (
                float(self.np_random.uniform(-cfg.world_xy * 0.75, cfg.world_xy * 0.75)),
                float(self.np_random.uniform(-cfg.world_xy * 0.75, cfg.world_xy * 0.75)),
                0.0,
            )

            trunk = Obstacle("cylinder", center=center, radius=radius, height=height)
            agent_clearance = obstacle_clearance(self.agent_pos, trunk, cfg.rotor_span_radius)
            target_clearance = obstacle_clearance(self.target_pos, trunk, cfg.rotor_span_radius)
            if agent_clearance < cfg.spawn_clearance or target_clearance < cfg.target_spawn_clearance:
                continue

            if is_tree_mode and len(obstacles) < cfg.obstacle_count:
                canopy_center = (center[0], center[1], height)
                canopy = Obstacle("sphere", center=canopy_center, radius=radius * 3.0)
                canopy_agent_clear = obstacle_clearance(self.agent_pos, canopy, cfg.rotor_span_radius)
                canopy_target_clear = obstacle_clearance(self.target_pos, canopy, cfg.rotor_span_radius)
                if (canopy_agent_clear >= cfg.spawn_clearance
                        and canopy_target_clear >= cfg.target_spawn_clearance):
                    obstacles.append(trunk)
                    obstacles.append(canopy)
                else:
                    obstacles.append(trunk)
            else:
                obstacles.append(trunk)

        return obstacles[: cfg.obstacle_count]

    def _integrate_agent(self, action: np.ndarray) -> None:
        cfg = self.config
        self.agent_yaw_rate = float(np.clip(action[3] * cfg.max_yaw_rate, -cfg.max_yaw_rate, cfg.max_yaw_rate))
        self.agent_yaw = _normalize_angle(self.agent_yaw + self.agent_yaw_rate * cfg.dt)
        forward = yaw_to_forward(self.agent_yaw, 0.0)
        right = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        accel = (forward * action[0] + right * action[1] + up * action[2]) * cfg.max_accel
        accel = clip_vector_norm(accel, cfg.max_accel)
        self.agent_vel = clip_vector_norm(self.agent_vel + accel * cfg.dt, cfg.max_speed)
        self.agent_pos = (self.agent_pos + self.agent_vel * cfg.dt).astype(np.float32)

    def _integrate_target(self) -> None:
        cfg = self.config
        self._update_target_velocity_for_behavior()
        self.target_vel = clip_vector_norm(self.target_vel, cfg.target_max_speed)
        self.target_pos = (self.target_pos + self.target_vel * cfg.dt).astype(np.float32)
        self._reflect_target_at_bounds()

    def _update_target_velocity_for_behavior(self) -> None:
        cfg = self.config
        if cfg.target_behavior == "hover":
            self.target_vel = np.zeros(3, dtype=np.float32)
        elif cfg.target_behavior == "straight":
            if norm(self.target_vel) < 1e-6:
                self.target_vel = np.array([cfg.target_max_speed, 0.0, 0.0], dtype=np.float32)
        elif cfg.target_behavior == "random_patrol":
            self._target_turn_countdown -= 1
            if self._target_turn_countdown <= 0 or norm(self.target_vel) < 1e-6:
                self.target_vel = self._initial_target_velocity()
                self._target_turn_countdown = int(self.np_random.integers(10, 40))

    def _reflect_target_at_bounds(self) -> None:
        cfg = self.config
        for axis in (0, 1):
            if self.target_pos[axis] < -cfg.world_xy:
                self.target_pos[axis] = -cfg.world_xy
                self.target_vel[axis] = abs(self.target_vel[axis])
            elif self.target_pos[axis] > cfg.world_xy:
                self.target_pos[axis] = cfg.world_xy
                self.target_vel[axis] = -abs(self.target_vel[axis])
        if self.target_pos[2] < cfg.launch_height_range[1] + 2.0:
            self.target_pos[2] = cfg.launch_height_range[1] + 2.0
            self.target_vel[2] = abs(self.target_vel[2])
        elif self.target_pos[2] > cfg.world_z:
            self.target_pos[2] = cfg.world_z
            self.target_vel[2] = -abs(self.target_vel[2])

    def _sensor_forward(self) -> np.ndarray:
        return yaw_to_forward(self.agent_yaw, np.deg2rad(self.config.sensor_pitch_bias_deg))

    def _target_in_sensor_cone(self) -> bool:
        cfg = self.config
        relative = self.target_pos - self.agent_pos
        target_distance = norm(relative)
        if target_distance <= 1e-8 or target_distance > cfg.sensor_range:
            return False
        if not cfg.enable_fov_limits:
            return True
        direction = safe_unit(relative)
        angle = angle_between_unit_vectors(self._sensor_forward(), direction)
        half_angle = np.deg2rad(min(cfg.sensor_horizontal_fov_deg, cfg.sensor_vertical_fov_deg) * 0.5)
        return bool(angle <= half_angle)

    def _target_occluded(self) -> bool:
        if not self.config.enable_occlusion:
            return False
        return any(
            obstacle.affects_occlusion
            and segment_occluded(self.agent_pos, self.target_pos, obstacle, self.config.rotor_span_radius * 0.25)
            for obstacle in self.obstacles
        )

    def _update_visibility(self) -> None:
        visible = self._target_in_sensor_cone() and not self._target_occluded()
        self.target_visible = visible
        if visible:
            self.has_seen_target = True
            self.time_since_seen = 0
            self.last_seen_target_pos = self.target_pos.copy()
        else:
            self.time_since_seen += 1

    def nearest_obstacle_clearance(self) -> float:
        if not self.obstacles:
            return self.config.sensor_range
        return float(
            min(
                obstacle_clearance(self.agent_pos, obstacle, self.config.rotor_span_radius)
                for obstacle in self.obstacles
            )
        )

    def obstacle_ray_distances(self) -> np.ndarray:
        return self._obstacle_sensor_readings()[1]

    def _obstacle_sensor_readings(self) -> tuple[float, np.ndarray]:
        cfg = self.config
        if not self.obstacles:
            return cfg.sensor_range, np.ones(cfg.obstacle_ray_count, dtype=np.float32)
        nearest = cfg.sensor_range
        best_rays = np.full(cfg.obstacle_ray_count, cfg.sensor_range, dtype=np.float32)
        relative_angles = np.linspace(-np.pi, np.pi, cfg.obstacle_ray_count, endpoint=False)
        origin_xy = self.agent_pos[:2]
        directions = np.stack(
            [
                np.cos(self.agent_yaw + relative_angles),
                np.sin(self.agent_yaw + relative_angles),
            ],
            axis=1,
        ).astype(np.float32)
        for obstacle in self.obstacles:
            if obstacle.kind not in ("cylinder", "sphere"):
                continue
            nearest = min(nearest, obstacle_clearance(self.agent_pos, obstacle, cfg.rotor_span_radius))
            center_xy = np.asarray(obstacle.center, dtype=np.float32)[:2]
            to_center = center_xy - origin_xy
            inflated = obstacle.radius + cfg.rotor_span_radius
            for idx, direction in enumerate(directions):
                projection = float(np.dot(to_center, direction))
                if projection <= 0.0 or projection > cfg.sensor_range:
                    continue
                lateral = abs(float(direction[0] * to_center[1] - direction[1] * to_center[0]))
                if lateral <= inflated:
                    best_rays[idx] = min(best_rays[idx], np.float32(max(0.0, projection - inflated)))
        values = np.clip(best_rays / cfg.sensor_range, 0.0, 1.0).astype(np.float32)
        return float(nearest), values

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        visible_rel = self.target_pos - self.agent_pos if self.target_visible else np.zeros(3, dtype=np.float32)
        last_rel = self.last_seen_target_pos - self.agent_pos if self.has_seen_target else np.zeros(3, dtype=np.float32)
        nearest_clearance, obstacle_rays = self._obstacle_sensor_readings()
        nearest = np.clip(nearest_clearance / cfg.sensor_range, -1.0, 1.0)
        obs = np.concatenate(
            [
                self._normalize_position(self.agent_pos),
                self.agent_vel / cfg.max_speed,
                np.array(
                    [
                        np.sin(self.agent_yaw),
                        np.cos(self.agent_yaw),
                        self.agent_yaw_rate / cfg.max_yaw_rate,
                        float(self.target_visible),
                    ],
                    dtype=np.float32,
                ),
                self._normalize_relative(visible_rel),
                self._normalize_relative(last_rel),
                np.array(
                    [
                        min(float(self.time_since_seen), float(cfg.max_steps)) / max(1.0, float(cfg.max_steps)),
                    ],
                    dtype=np.float32,
                ),
                obstacle_rays,
                np.array(
                    [
                        np.clip(self.previous_distance / self.max_distance, 0.0, 1.0),
                        np.clip(self.success_hold_steps / max(1, cfg.success_hold_steps), 0.0, 1.0),
                        np.clip(self.step_count / max(1, cfg.max_steps), 0.0, 1.0),
                        nearest,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if cfg.observation_noise_std > 0.0:
            noise = self.np_random.normal(0.0, cfg.observation_noise_std, size=obs.shape).astype(np.float32)
            noise[:OBSERVATION_NOISE_PROTECTED_END] = 0.0
            obs = np.clip(obs + noise, self.observation_space.low, self.observation_space.high).astype(np.float32)
        return obs

    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        out_of_bounds: bool,
        success: bool,
        nearest_clearance: float,
        current_distance: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        weights = cfg.reward_weights
        if current_distance is None:
            current_distance = distance(self.agent_pos, self.target_pos)
        progress = float(np.clip((self.previous_distance - current_distance) / self.max_distance, -1.0, 1.0))
        reacquired = (
            self.target_visible
            and not self.previous_target_visible
            and self._had_seen_before_visibility
        )
        clearance_fraction = max(0.0, cfg.clearance_penalty_distance - nearest_clearance) / max(
            1e-6,
            cfg.clearance_penalty_distance,
        )
        launch_fraction = 0.0
        if self.step_count <= cfg.launch_altitude_window_steps:
            launch_fraction = min(float(self.agent_pos[2]) / max(1e-6, cfg.launch_altitude_target), 1.0)
        terms = {
            "distance_progress": weights["distance_progress"] * progress,
            "target_visible": weights["target_visible"] if self.target_visible else 0.0,
            "reacquisition": weights["reacquisition"] if reacquired else 0.0,
            "capture_hold": weights["capture_hold"] if self.success_hold_steps > 0 and not success else 0.0,
            "success": weights["success"] if success else 0.0,
            "clearance": weights["clearance"] * clearance_fraction,
            "launch_altitude": weights["launch_altitude"] * launch_fraction,
            "time": weights["time"],
            "control": weights["control"] * float(np.dot(action, action)),
            "collision": weights["collision"] if collision else 0.0,
            "out_of_bounds": weights["out_of_bounds"] if out_of_bounds else 0.0,
        }
        ordered = {key: float(terms[key]) for key in V2_REWARD_KEYS}
        return float(sum(ordered.values())), ordered

    def _make_info(
        self,
        collision: bool,
        out_of_bounds: bool,
        success: bool,
        truncated: bool,
        distance_to_target: float,
        nearest_clearance: float,
    ) -> dict[str, Any]:
        info = {
            "distance_to_target": float(distance_to_target),
            "target_visible": bool(self.target_visible),
            "collision": bool(collision),
            "success": bool(success),
            "success_hold_steps": int(self.success_hold_steps),
            "reward_terms": self.last_reward_terms.copy(),
            "nearest_obstacle_distance": float(nearest_clearance),
            "out_of_bounds": bool(out_of_bounds),
            "time_limit": bool(truncated),
            "curriculum_level": int(self.config.curriculum_level),
            "obstacle_count": len(self.obstacles),
        }
        return info

    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        return self._normalize_by(position, (self.config.world_xy, self.config.world_xy, self.config.world_z))

    def _normalize_relative(self, relative: np.ndarray) -> np.ndarray:
        return self._normalize_by(
            relative,
            (2.0 * self.config.world_xy, 2.0 * self.config.world_xy, self.config.world_z),
        )

    @staticmethod
    def _normalize_by(values: np.ndarray, denominators: tuple[float, float, float]) -> np.ndarray:
        return np.clip(
            np.asarray(values, dtype=np.float32) / np.asarray(denominators, dtype=np.float32),
            -1.0,
            1.0,
        ).astype(np.float32)

    def _render_state(self) -> dict[str, Any]:
        """Build a render-state dict compatible with DroneRenderer.draw()."""
        obstacles = []
        for obs in self.obstacles:
            obstacles.append({
                "type": obs.kind,
                "center": list(obs.center),
                "radius": obs.radius,
                "height": obs.height,
            })

        return {
            "pursuer_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "yaw": self.agent_yaw,
            "body_radius": self.config.body_radius,
            "body_height": self.config.body_height,
            "rotor_span_radius": self.config.rotor_span_radius,
            "pursuer_heading": yaw_to_forward(self.agent_yaw),
            "horizontal_fov_deg": self.config.sensor_horizontal_fov_deg,
            "vertical_fov_deg": self.config.sensor_vertical_fov_deg,
            "viewport_range": self.config.sensor_range,
            "obstacles": tuple(obstacles),
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "distance": distance(self.agent_pos, self.target_pos),
            "reward": float(sum(self.last_reward_terms.values())),
            "target_visible": self.target_visible,
            "has_target_lock": False,
            "steps_since_seen": int(self.time_since_seen),
            "collision": self.last_info.get("collision", False),
            "out_of_bounds": self.last_info.get("out_of_bounds", False),
            "success": self.last_info.get("success", False),
        }
