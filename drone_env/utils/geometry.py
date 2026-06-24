from __future__ import annotations

import numpy as np


def safe_norm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(vector).all():
        return 0.0
    return float(np.linalg.norm(vector))


def norm(vector: np.ndarray) -> float:
    return safe_norm(vector)


def safe_unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    magnitude = safe_norm(vector)
    if magnitude < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / magnitude).astype(np.float32)


def angle_between_unit_vectors(a: np.ndarray, b: np.ndarray) -> float:
    a = safe_unit(a)
    b = safe_unit(b)
    if safe_norm(a) == 0.0 or safe_norm(b) == 0.0:
        return float(np.pi)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def point_in_cone_or_frustum(
    origin: np.ndarray,
    heading: np.ndarray,
    point: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    max_range: float,
) -> bool:
    # MVP uses a symmetric cone with the narrower FOV axis.
    relative = np.asarray(point, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    dist = safe_norm(relative)
    if dist <= 1e-8 or dist > max_range:
        return False
    direction = safe_unit(relative)
    heading_unit = safe_unit(heading)
    if safe_norm(heading_unit) == 0.0:
        return False
    half_angle = np.deg2rad(max(0.0, min(horizontal_fov_deg, vertical_fov_deg)) * 0.5)
    return bool(np.dot(heading_unit, direction) >= np.cos(half_angle))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))


def closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return a.copy()
    t = float(np.clip(np.dot(point - a, segment) / length_sq, 0.0, 1.0))
    return (a + t * segment).astype(np.float32)


def segment_sphere_intersection(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    closest = closest_point_on_segment(center, start, end)
    return bool(distance(closest, center) <= radius)


def crossed_intercept_plane(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    plane_radius: float,
) -> bool:
    normal_unit = safe_unit(normal)
    if safe_norm(normal_unit) == 0.0 or plane_radius < 0.0:
        return False
    start_rel = np.asarray(start, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    end_rel = np.asarray(end, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    start_side = float(np.dot(start_rel, normal_unit))
    end_side = float(np.dot(end_rel, normal_unit))
    if start_side == 0.0:
        crossing_t = 0.0
    elif end_side == 0.0:
        crossing_t = 1.0
    elif start_side * end_side > 0.0:
        return False
    else:
        crossing_t = start_side / (start_side - end_side)
    crossing = np.asarray(start, dtype=np.float32) + crossing_t * (
        np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    )
    return bool(distance(crossing, center) <= plane_radius)


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    magnitude = norm(vector)
    if magnitude <= max_norm or magnitude == 0.0:
        return vector.astype(np.float32)
    return (vector * (max_norm / magnitude)).astype(np.float32)


def in_bounds(
    position: np.ndarray,
    world_xy: float,
    world_z: float,
    margin: float = 0.0,
    include_floor: bool = True,
) -> bool:
    x, y, z = np.asarray(position, dtype=np.float32)
    min_z = 0.0 if include_floor else np.finfo(np.float32).eps
    return bool(
        -world_xy - margin <= x <= world_xy + margin
        and -world_xy - margin <= y <= world_xy + margin
        and min_z <= z <= world_z + margin
    )
