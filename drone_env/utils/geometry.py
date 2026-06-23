from __future__ import annotations

import numpy as np


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def safe_unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / magnitude).astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))


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
