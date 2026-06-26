from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from drone_env.utils.geometry import closest_point_on_segment, distance

ObstacleKind = Literal["cylinder", "sphere", "box"]


@dataclass(frozen=True)
class Obstacle:
    kind: ObstacleKind
    center: np.ndarray
    radius: float = 0.0
    height: float = 0.0
    half_extents: np.ndarray | None = None
    affects_occlusion: bool = True


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
        length_sq = float(np.dot(seg_xy, seg_xy))
        if length_sq <= 1e-12:
            return False
        t = float(np.clip(np.dot(center[:2] - start[:2], seg_xy) / length_sq, 0.0, 1.0))
        point = start + t * seg
        z_min = center[2]
        z_max = center[2] + obstacle.height
        return bool(
            z_min <= point[2] <= z_max
            and np.linalg.norm(point[:2] - center[:2]) <= obstacle.radius + rotor_radius
        )
    if obstacle.kind == "box":
        return obstacle_clearance(closest_point_on_segment(obstacle.center, start, end), obstacle, rotor_radius) <= 0.0
    return False
