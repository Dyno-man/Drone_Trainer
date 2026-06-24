from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_env.utils.geometry import (
    angle_between_unit_vectors,
    clip_vector_norm,
    closest_point_on_segment,
    distance,
    in_bounds,
    norm,
    point_in_cone_or_frustum,
    safe_unit,
    segment_sphere_intersection,
)

TargetMode = Literal["straight", "evasive", "orbit"]
ObservationMode = Literal["privileged", "viewport"]

REWARD_KEYS = (
    "capture_reward",
    "flythrough_intercept_reward",
    "first_acquisition_reward",
    "visibility_reward",
    "reacquisition_reward",
    "lost_target_penalty",
    "aim_through_target_reward",
    "distance_progress_reward",
    "distance_penalty",
    "time_penalty",
    "alignment_reward",
    "closing_speed_reward",
    "energy_penalty",
    "smoothness_penalty",
    "altitude_penalty",
    "safety_failure_penalty",
)


@dataclass
class DroneIntercept3DConfig:
    world_xy: float = 100.0
    world_z: float = 50.0
    bounds_margin: float = 5.0
    dt: float = 0.1
    pursuer_max_speed: float = 18.0
    target_max_speed: float = 10.0
    max_accel: float = 14.0
    capture_radius: float = 3.0
    observation_mode: ObservationMode = "viewport"
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    viewport_range: float = 80.0
    lock_memory_steps: int = 20
    intercept_radius: float = 2.0
    flythrough_plane_radius: float = 2.0
    min_closing_speed: float = 0.0
    require_los_for_pursuit_reward: bool = False
    require_flythrough_success: bool = True
    max_steps: int = 500
    safe_min_z: float = 2.0
    safe_max_z: float = 48.0
    min_initial_distance: float = 25.0
    render_width: int = 960
    render_height: int = 720
    trail_length: int = 80
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "capture_reward": 100.0,
            "flythrough_intercept_reward": 140.0,
            "first_acquisition_reward": 5.0,
            "visibility_reward": 0.15,
            "reacquisition_reward": 2.5,
            "lost_target_penalty": -0.08,
            "aim_through_target_reward": 1.0,
            "distance_progress_reward": 4.0,
            "distance_penalty": -0.03,
            "time_penalty": -0.05,
            "alignment_reward": 1.5,
            "closing_speed_reward": 0.5,
            "energy_penalty": -0.02,
            "smoothness_penalty": -0.01,
            "altitude_penalty": -2.0,
            "crash_penalty": -100.0,
            "out_of_bounds_penalty": -75.0,
        }
    )


class DroneIntercept3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: DroneIntercept3DConfig | None = None,
        target_mode: TargetMode = "evasive",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if target_mode not in ("straight", "evasive", "orbit"):
            raise ValueError(f"Unsupported target_mode: {target_mode}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.config = config or DroneIntercept3DConfig()
        if self.config.observation_mode not in ("privileged", "viewport"):
            raise ValueError(f"Unsupported observation_mode: {self.config.observation_mode}")
        self.target_mode: TargetMode = target_mode
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self.pursuer_pos = np.zeros(3, dtype=np.float32)
        self.pursuer_vel = np.zeros(3, dtype=np.float32)
        self.previous_pursuer_pos = np.zeros(3, dtype=np.float32)
        self.pursuer_heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)
        self.last_seen_target_pos = np.zeros(3, dtype=np.float32)
        self.last_seen_target_vel = np.zeros(3, dtype=np.float32)
        self.steps_since_seen = self.config.lock_memory_steps + 1
        self.target_visible = False
        self.has_target_lock = False
        self._has_ever_seen_target = False
        self._lost_target_this_step = False
        self._first_acquisition_this_step = False
        self._reacquisition_this_step = False
        self.previous_distance = 0.0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}
        self._renderer = None
        self._pursuer_trail: list[np.ndarray] = []
        self._target_trail: list[np.ndarray] = []
        self._orbit_angle = 0.0

    @property
    def max_distance(self) -> float:
        cfg = self.config
        return float(np.sqrt((2 * cfg.world_xy) ** 2 + (2 * cfg.world_xy) ** 2 + cfg.world_z**2))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "target_mode" in options:
            self.target_mode = options["target_mode"]

        cfg = self.config
        self.step_count = 0
        self.pursuer_vel = np.zeros(3, dtype=np.float32)
        self.target_vel = self._sample_target_velocity()
        self.previous_action = np.zeros(3, dtype=np.float32)
        self._orbit_angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        for _ in range(1000):
            self.pursuer_pos = self._sample_position()
            self.target_pos = self._sample_position()
            if distance(self.pursuer_pos, self.target_pos) >= cfg.min_initial_distance:
                break
        else:
            self.pursuer_pos = np.array([-30.0, 0.0, 15.0], dtype=np.float32)
            self.target_pos = np.array([30.0, 0.0, 20.0], dtype=np.float32)

        self.previous_pursuer_pos = self.pursuer_pos.copy()
        self.pursuer_heading = safe_unit(self.target_pos - self.pursuer_pos)
        if norm(self.pursuer_heading) == 0.0:
            self.pursuer_heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.last_seen_target_pos = np.zeros(3, dtype=np.float32)
        self.last_seen_target_vel = np.zeros(3, dtype=np.float32)
        self.steps_since_seen = cfg.lock_memory_steps + 1
        self.target_visible = False
        self.has_target_lock = False
        self._has_ever_seen_target = False
        self._lost_target_this_step = False
        self._first_acquisition_this_step = False
        self._reacquisition_this_step = False
        self._update_visibility_state()
        self.previous_distance = distance(self.pursuer_pos, self.target_pos)
        self._pursuer_trail = [self.pursuer_pos.copy()]
        self._target_trail = [self.target_pos.copy()]
        self.last_reward = 0.0
        self.last_info = self._base_info(False, False, False)
        return self._get_obs(), self.last_info.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        self.step_count += 1
        self.previous_pursuer_pos = self.pursuer_pos.copy()
        accel = action * cfg.max_accel
        self.pursuer_vel = clip_vector_norm(self.pursuer_vel + accel * cfg.dt, cfg.pursuer_max_speed)
        self._update_pursuer_heading()
        self.pursuer_pos = (self.pursuer_pos + self.pursuer_vel * cfg.dt).astype(np.float32)
        self._update_target()
        self._update_visibility_state()

        current_distance = distance(self.pursuer_pos, self.target_pos)
        flythrough_intercept = self._flythrough_intercept()
        proximity_capture = current_distance <= cfg.capture_radius
        captured = bool(
            flythrough_intercept
            if cfg.require_flythrough_success
            else (flythrough_intercept or proximity_capture)
        )
        crashed = self.pursuer_pos[2] <= 0.0
        out_of_bounds = not in_bounds(
            self.pursuer_pos,
            cfg.world_xy,
            cfg.world_z,
            cfg.bounds_margin,
            include_floor=True,
        )
        terminated = bool(captured or crashed or out_of_bounds)
        truncated = bool(self.step_count >= cfg.max_steps and not terminated)

        reward, breakdown = self._compute_reward(
            action,
            captured,
            crashed,
            out_of_bounds,
            flythrough_intercept=flythrough_intercept,
        )
        self.previous_distance = current_distance
        self.previous_action = action.copy()
        self._append_trails()

        info = self._base_info(captured, crashed, out_of_bounds, flythrough_intercept)
        info["reward_breakdown"] = breakdown
        self.last_reward = reward
        self.last_info = info.copy()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from drone_env.utils.rendering import DroneRenderer

            cfg = self.config
            self._renderer = DroneRenderer(
                cfg.render_width, cfg.render_height, cfg.world_xy, cfg.world_z, self.render_mode
            )
        return self._renderer.draw(self._render_state())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        target_pos, target_vel, target_available = self._observation_target_state()
        relative_pos = target_pos - self.pursuer_pos if target_available else np.zeros(3, dtype=np.float32)
        relative_vel = target_vel - self.pursuer_vel if target_available else np.zeros(3, dtype=np.float32)
        angle_to_target = 0.0
        distance_to_estimate = 0.0
        if target_available:
            direction = safe_unit(relative_pos)
            angle_to_target = angle_between_unit_vectors(self.pursuer_heading, direction) / np.pi
            distance_to_estimate = distance(self.pursuer_pos, target_pos) / self.max_distance
        obs = np.concatenate(
            [
                self._normalize_pos(self.pursuer_pos),
                self.pursuer_vel / cfg.pursuer_max_speed,
                self.pursuer_heading,
                np.array(
                    [
                        relative_pos[0] / (2.0 * cfg.world_xy),
                        relative_pos[1] / (2.0 * cfg.world_xy),
                        relative_pos[2] / cfg.world_z,
                    ],
                    dtype=np.float32,
                ),
                relative_vel / (cfg.pursuer_max_speed + cfg.target_max_speed),
                np.array(
                    [
                        float(self.target_visible),
                        float(self.has_target_lock),
                        min(self.steps_since_seen, cfg.lock_memory_steps + 1)
                        / max(1, cfg.lock_memory_steps + 1),
                        angle_to_target,
                        distance_to_estimate,
                        self.previous_distance / self.max_distance,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(
        self,
        action: np.ndarray,
        captured: bool,
        crashed: bool,
        out_of_bounds: bool,
        flythrough_intercept: bool = False,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        weights = cfg.reward_weights
        current_distance = distance(self.pursuer_pos, self.target_pos)
        direction_to_target = safe_unit(self.target_pos - self.pursuer_pos)
        pursuer_direction = safe_unit(self.pursuer_vel)
        alignment = float(np.dot(pursuer_direction, direction_to_target))
        relative_vel = self.target_vel - self.pursuer_vel
        closing_speed = float(-np.dot(relative_vel, direction_to_target))
        unsafe_low = max(0.0, cfg.safe_min_z - float(self.pursuer_pos[2]))
        unsafe_high = max(0.0, float(self.pursuer_pos[2]) - cfg.safe_max_z)

        safety_penalty = 0.0
        if crashed:
            safety_penalty += weights["crash_penalty"]
        if out_of_bounds:
            safety_penalty += weights["out_of_bounds_penalty"]

        closest = closest_point_on_segment(
            self.target_pos,
            self.pursuer_pos,
            self.pursuer_pos + safe_unit(self.pursuer_vel) * max(cfg.intercept_radius, cfg.viewport_range),
        )
        miss_distance = distance(closest, self.target_pos)
        reward_has_los = self.target_visible or self.has_target_lock or not cfg.require_los_for_pursuit_reward
        aim_reward = 0.0
        if reward_has_los and norm(self.pursuer_vel) > 1e-6:
            aim_reward = weights["aim_through_target_reward"] * float(
                np.clip(1.0 - miss_distance / max(cfg.intercept_radius * 4.0, 1e-6), 0.0, 1.0)
            )

        breakdown = {
            "capture_reward": weights["capture_reward"] if captured else 0.0,
            "flythrough_intercept_reward": weights["flythrough_intercept_reward"]
            if flythrough_intercept
            else 0.0,
            "first_acquisition_reward": weights["first_acquisition_reward"]
            if self._first_acquisition_this_step
            else 0.0,
            "visibility_reward": weights["visibility_reward"] if self.target_visible else 0.0,
            "reacquisition_reward": weights["reacquisition_reward"]
            if self._reacquisition_this_step
            else 0.0,
            "lost_target_penalty": weights["lost_target_penalty"]
            if (not self.target_visible and not self.has_target_lock)
            else 0.0,
            "aim_through_target_reward": aim_reward,
            "distance_progress_reward": weights["distance_progress_reward"]
            * float(np.clip(self.previous_distance - current_distance, -5.0, 5.0)),
            "distance_penalty": weights["distance_penalty"] * current_distance,
            "time_penalty": weights["time_penalty"],
            "alignment_reward": float(np.clip(weights["alignment_reward"] * alignment, -1.5, 1.5)),
            "closing_speed_reward": float(
                np.clip(weights["closing_speed_reward"] * closing_speed, -5.0, 5.0)
            ),
            "energy_penalty": weights["energy_penalty"] * float(np.dot(action, action)),
            "smoothness_penalty": weights["smoothness_penalty"]
            * float(np.dot(action - self.previous_action, action - self.previous_action)),
            "altitude_penalty": weights["altitude_penalty"] * (unsafe_low + unsafe_high),
            "safety_failure_penalty": safety_penalty,
        }
        return float(sum(breakdown.values())), {key: float(breakdown[key]) for key in REWARD_KEYS}

    def _update_pursuer_heading(self) -> None:
        velocity_heading = safe_unit(self.pursuer_vel)
        if norm(velocity_heading) > 0.0:
            self.pursuer_heading = velocity_heading

    def _target_is_visible(self) -> bool:
        cfg = self.config
        return point_in_cone_or_frustum(
            self.pursuer_pos,
            self.pursuer_heading,
            self.target_pos,
            cfg.horizontal_fov_deg,
            cfg.vertical_fov_deg,
            cfg.viewport_range,
        )

    def _update_visibility_state(self) -> None:
        was_visible = self.target_visible
        had_lock = self.has_target_lock
        visible = self._target_is_visible()
        self._first_acquisition_this_step = False
        self._reacquisition_this_step = False
        self._lost_target_this_step = False
        self.target_visible = visible
        if visible:
            self.last_seen_target_pos = self.target_pos.copy()
            self.last_seen_target_vel = self.target_vel.copy()
            self.steps_since_seen = 0
            self.has_target_lock = True
            if not self._has_ever_seen_target:
                self._first_acquisition_this_step = True
            elif not was_visible:
                self._reacquisition_this_step = True
            self._has_ever_seen_target = True
        else:
            self.steps_since_seen += 1
            self.has_target_lock = self.steps_since_seen <= self.config.lock_memory_steps
            self._lost_target_this_step = had_lock and not self.has_target_lock

    def _observation_target_state(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.config.observation_mode == "privileged":
            return self.target_pos, self.target_vel, True
        if self.target_visible or self.has_target_lock:
            return self.last_seen_target_pos, self.last_seen_target_vel, True
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), False

    def _flythrough_intercept(self) -> bool:
        cfg = self.config
        intersects = segment_sphere_intersection(
            self.previous_pursuer_pos,
            self.pursuer_pos,
            self.target_pos,
            cfg.intercept_radius,
        )
        if not intersects:
            return False
        if cfg.min_closing_speed <= 0.0:
            return True
        direction_to_target = safe_unit(self.target_pos - self.previous_pursuer_pos)
        relative_vel = self.pursuer_vel - self.target_vel
        closing_speed = float(np.dot(relative_vel, direction_to_target))
        return bool(closing_speed >= cfg.min_closing_speed)

    def _update_target(self) -> None:
        if self.target_mode == "straight":
            self._target_straight()
        elif self.target_mode == "evasive":
            self._target_evasive()
        else:
            self._target_orbit()
        self.target_vel = clip_vector_norm(self.target_vel, self.config.target_max_speed)
        self.target_pos = (self.target_pos + self.target_vel * self.config.dt).astype(np.float32)
        self._reflect_target_at_bounds()

    def _target_straight(self) -> None:
        if norm(self.target_vel) < 1e-6:
            self.target_vel = np.array([self.config.target_max_speed, 0.0, 0.0], dtype=np.float32)

    def _target_evasive(self) -> None:
        away = safe_unit(self.target_pos - self.pursuer_pos)
        lateral = np.array([-away[1], away[0], 0.25], dtype=np.float32)
        lateral = safe_unit(lateral)
        jitter = self.np_random.normal(0.0, 0.15, size=3).astype(np.float32)
        desired = away * 0.8 + lateral * 0.3 + jitter
        self.target_vel = safe_unit(desired) * self.config.target_max_speed
        self._bias_target_back_inside()

    def _target_orbit(self) -> None:
        cfg = self.config
        prev = self.target_pos.copy()
        radius = cfg.world_xy * 0.45
        self._orbit_angle += 0.45 * cfg.dt
        desired = np.array(
            [
                radius * np.cos(self._orbit_angle),
                radius * np.sin(self._orbit_angle),
                cfg.world_z * 0.5 + 8.0 * np.sin(self._orbit_angle * 0.7),
            ],
            dtype=np.float32,
        )
        self.target_vel = (desired - prev) / cfg.dt

    def _reflect_target_at_bounds(self) -> None:
        cfg = self.config
        for axis in (0, 1):
            low = -cfg.world_xy
            high = cfg.world_xy
            if self.target_pos[axis] < low:
                self.target_pos[axis] = low
                self.target_vel[axis] = abs(self.target_vel[axis])
            elif self.target_pos[axis] > high:
                self.target_pos[axis] = high
                self.target_vel[axis] = -abs(self.target_vel[axis])
        if self.target_pos[2] < cfg.safe_min_z:
            self.target_pos[2] = cfg.safe_min_z
            self.target_vel[2] = abs(self.target_vel[2])
        elif self.target_pos[2] > cfg.safe_max_z:
            self.target_pos[2] = cfg.safe_max_z
            self.target_vel[2] = -abs(self.target_vel[2])

    def _bias_target_back_inside(self) -> None:
        cfg = self.config
        bias = np.zeros(3, dtype=np.float32)
        for axis in (0, 1):
            if self.target_pos[axis] > cfg.world_xy * 0.8:
                bias[axis] -= 1.0
            elif self.target_pos[axis] < -cfg.world_xy * 0.8:
                bias[axis] += 1.0
        if self.target_pos[2] > cfg.safe_max_z - 3.0:
            bias[2] -= 1.0
        elif self.target_pos[2] < cfg.safe_min_z + 3.0:
            bias[2] += 1.0
        if norm(bias) > 0.0:
            self.target_vel = safe_unit(self.target_vel + bias * self.config.target_max_speed)
            self.target_vel *= self.config.target_max_speed

    def _sample_position(self) -> np.ndarray:
        cfg = self.config
        return np.array(
            [
                self.np_random.uniform(-cfg.world_xy * 0.55, cfg.world_xy * 0.55),
                self.np_random.uniform(-cfg.world_xy * 0.55, cfg.world_xy * 0.55),
                self.np_random.uniform(cfg.safe_min_z + 3.0, cfg.safe_max_z - 3.0),
            ],
            dtype=np.float32,
        )

    def _sample_target_velocity(self) -> np.ndarray:
        direction = safe_unit(self.np_random.normal(0.0, 1.0, size=3).astype(np.float32))
        if abs(float(direction[2])) > 0.4:
            direction[2] *= 0.4
            direction = safe_unit(direction)
        return direction * self.config.target_max_speed

    def _normalize_pos(self, position: np.ndarray) -> np.ndarray:
        cfg = self.config
        return np.array(
            [position[0] / cfg.world_xy, position[1] / cfg.world_xy, position[2] / cfg.world_z],
            dtype=np.float32,
        )

    def _append_trails(self) -> None:
        cfg = self.config
        self._pursuer_trail.append(self.pursuer_pos.copy())
        self._target_trail.append(self.target_pos.copy())
        self._pursuer_trail = self._pursuer_trail[-cfg.trail_length :]
        self._target_trail = self._target_trail[-cfg.trail_length :]

    def _base_info(
        self,
        captured: bool,
        crashed: bool,
        out_of_bounds: bool,
        flythrough_intercept: bool = False,
    ) -> dict[str, Any]:
        return {
            "captured": bool(captured),
            "crashed": bool(crashed),
            "out_of_bounds": bool(out_of_bounds),
            "distance": distance(self.pursuer_pos, self.target_pos),
            "target_mode": self.target_mode,
            "target_visible": bool(self.target_visible),
            "has_target_lock": bool(self.has_target_lock),
            "steps_since_seen": int(self.steps_since_seen),
            "last_seen_target_pos": self.last_seen_target_pos.copy(),
            "flythrough_intercept": bool(flythrough_intercept),
            "pursuer_heading": self.pursuer_heading.copy(),
        }

    def _render_state(self) -> dict[str, Any]:
        return {
            "pursuer_pos": self.pursuer_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "pursuer_trail": [point.copy() for point in self._pursuer_trail],
            "target_trail": [point.copy() for point in self._target_trail],
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "distance": distance(self.pursuer_pos, self.target_pos),
            "reward": self.last_reward,
            "captured": self.last_info.get("captured", False),
            "flythrough_intercept": self.last_info.get("flythrough_intercept", False),
            "crashed": self.last_info.get("crashed", False),
            "out_of_bounds": self.last_info.get("out_of_bounds", False),
            "target_visible": self.last_info.get("target_visible", False),
            "has_target_lock": self.last_info.get("has_target_lock", False),
            "steps_since_seen": self.last_info.get("steps_since_seen", 0),
            "pursuer_heading": self.pursuer_heading.copy(),
            "horizontal_fov_deg": self.config.horizontal_fov_deg,
            "vertical_fov_deg": self.config.vertical_fov_deg,
            "viewport_range": self.config.viewport_range,
        }
