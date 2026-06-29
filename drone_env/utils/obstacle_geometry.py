from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from drone_env.utils.geometry import closest_point_on_segment, distance

ObstacleKind = Literal["cylinder", "sphere", "box"]


@dataclass(frozen=True)
class Obstacle:
    kind: ObstacleKind
    center: tuple[float, float, float]
    radius: float = 0.0
    height: float = 0.0
    half_extents: tuple[float, float, float] | None = None
    affects_occlusion: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", tuple(self.center))
        if self.half_extents is not None:
            object.__setattr__(self, "half_extents", tuple(self.half_extents))

def obstacle_clearance(position: np.ndarray, obstacle: Obstacle, rotor_radius: float) -> float:
    position = np.asarray(position, dtype=np.float32)
    center = np.asarray(obstacle.center, dtype=np.float32)
    if obstacle.kind == "cylinder":
        horizontal = float(np.linalg.norm(position[:2] - center[:2]))
        inflated = obstacle.radius + rotor_radius
        z_min = float(center[2])
        z_max = z_min + obstacle.height
        if z_min <= position[2] <= z_max:
            return horizontal - inflated
        dz = z_min - float(position[2]) if position[2] < z_min else float(position[2]) - z_max
        dx = max(0.0, horizontal - inflated)
        return float(np.hypot(dx, dz))
    if obstacle.kind == "sphere":
        return distance(position, center) - (obstacle.radius + rotor_radius)
    if obstacle.kind == "box":
        if obstacle.half_extents is None:
            raise ValueError("box obstacle requires half_extents")
        half = np.asarray(obstacle.half_extents, dtype=np.float32) + rotor_radius
        q = np.abs(position - center) - half
        outside = np.maximum(q, 0.0)
        # Signed distance for an inflated axis-aligned box: outside distance is
        # Euclidean, while inside distance is the least negative separating axis.
        inside = min(float(np.max(q)), 0.0)
        return float(np.linalg.norm(outside) + inside)
    raise ValueError(f"unsupported obstacle kind: {obstacle.kind}")


def obstacle_collides(position: np.ndarray, obstacle: Obstacle, rotor_radius: float) -> bool:
    return obstacle_clearance(position, obstacle, rotor_radius) <= 0.0



def _solve_quadratic_interval(a: float, b: float, c: float) -> tuple[float | None, float | None]:
    """Return (t_enter, t_exit) for where a*t**2 + b*t + c <= 0, or (None, None)."""
    if a < 1e-15:
        if abs(b) < 1e-15:
            if c <= 0.0:
                return 0.0, 1.0
            return None, None
        t = -c / b
        if b > 0:
            return 0.0, t
        return t, 1.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    sqrt_disc = disc ** 0.5
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    return min(t1, t2), max(t1, t2)


def _point_in_cylinder(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
    height: float,
    rotor_radius: float = 0.0,
) -> bool:
    dist_xy = float(np.linalg.norm(point[:2] - center[:2]))
    if dist_xy > radius + rotor_radius:
        return False
    z_min = float(center[2])
    z_max = z_min + height
    return z_min <= float(point[2]) <= z_max

def segment_occluded(start: np.ndarray, end: np.ndarray, obstacle: Obstacle, rotor_radius: float = 0.0) -> bool:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if obstacle.kind == "sphere":
        closest = closest_point_on_segment(obstacle.center, start, end)
        return distance(closest, obstacle.center) <= obstacle.radius + rotor_radius
    if obstacle.kind == "cylinder":
        center = np.asarray(obstacle.center, dtype=np.float32)
        seg = end - start
        seg_xy = seg[:2]
        m = start[:2] - center[:2]

        r = obstacle.radius + rotor_radius
        a = float(np.dot(seg_xy, seg_xy))
        b = float(2.0 * np.dot(seg_xy, m))
        c = float(np.dot(m, m)) - r * r

        t_xy_enter, t_xy_exit = _solve_quadratic_interval(a, b, c)

        dz = float(seg[2])
        z_min = float(center[2])
        z_max = z_min + obstacle.height

        if abs(dz) < 1e-12:
            if z_min <= float(start[2]) <= z_max:
                t_z_enter, t_z_exit = 0.0, 1.0
            else:
                return False
        else:
            t_z_enter = (z_min - float(start[2])) / dz
            t_z_exit = (z_max - float(start[2])) / dz
            if t_z_enter > t_z_exit:
                t_z_enter, t_z_exit = t_z_exit, t_z_enter
            t_z_enter = max(0.0, t_z_enter)
            t_z_exit = min(1.0, t_z_exit)

        if t_xy_enter is not None and t_xy_exit is not None:
            t_enter = max(0.0, t_xy_enter, t_z_enter)
            t_exit = min(1.0, t_xy_exit, t_z_exit)
            if t_enter <= t_exit:
                return True

        return _point_in_cylinder(start, center, obstacle.radius, obstacle.height, rotor_radius) \
            or _point_in_cylinder(end, center, obstacle.radius, obstacle.height, rotor_radius)
    if obstacle.kind == "box":
        return obstacle_clearance(closest_point_on_segment(obstacle.center, start, end), obstacle, rotor_radius) <= 0.0
    return False
