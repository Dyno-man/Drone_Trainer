from __future__ import annotations

import numpy as np
import pytest

from drone_env.utils.obstacle_geometry import (
    Obstacle,
    obstacle_clearance,
    obstacle_collides,
    segment_occluded,
)


# --- Bug 2: Frozen dataclass with immutable tuple center ---

def test_obstacle_center_is_tuple() -> None:
    """Obstacle.center is a tuple, not ndarray."""
    obs = Obstacle(kind="cylinder", center=np.array([1.0, 2.0, 3.0]), radius=0.5, height=10.0)
    assert isinstance(obs.center, tuple)
    assert obs.center == (1.0, 2.0, 3.0)


def test_obstacle_center_immutable() -> None:
    """Mutation of center raises TypeError."""
    obs = Obstacle(kind="cylinder", center=(1.0, 2.0, 3.0), radius=0.5, height=10.0)
    with pytest.raises(TypeError):
        obs.center[0] = 999.0


def test_obstacle_half_extents_tuple() -> None:
    """Box obstacle stores half_extents as tuple."""
    obs = Obstacle(kind="box", center=(0, 0, 0), half_extents=np.array([1.0, 2.0, 3.0]))
    assert isinstance(obs.half_extents, tuple)
    assert obs.half_extents == (1.0, 2.0, 3.0)


# --- Bug 1: Cylinder occlusion false negative ---

def test_segment_occluded_cylinder_slope() -> None:
    """Exact BUGFIXES_V2 repro: sloped segment passes through cylinder.

    Segment (0,0,0)->(10,0,10), cylinder at (5,0,0) r=4 h=2.
    Segment parametrically: (10t, 0, 10t). Inside cylinder when t in [0.1, 0.2].
    """
    start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 10.0], dtype=np.float32)
    blocker = Obstacle(kind="cylinder", center=(5.0, 0.0, 0.0), radius=4.0, height=2.0)
    assert segment_occluded(start, end, blocker) is True


def test_segment_occluded_cylinder_horizontal() -> None:
    """Horizontal segment perpendicular to cylinder axis."""
    start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 1.0], dtype=np.float32)
    blocker = Obstacle(kind="cylinder", center=(5.0, 0.0, 0.0), radius=4.0, height=2.0)
    # Segment at z=1, cylinder z in [0, 2] -> overlaps. XY at distance 0 < 4 -> occludes.
    assert segment_occluded(start, end, blocker) is True


def test_segment_occluded_cylinder_tangent() -> None:
    """Segment tangent to cylinder surface returns True (endpoint touch)."""
    start = np.array([0.0, 4.0, 0.0], dtype=np.float32)
    end = np.array([10.0, 4.0, 0.0], dtype=np.float32)
    blocker = Obstacle(kind="cylinder", center=(5.0, 0.0, 0.0), radius=4.0, height=2.0)
    # At x=1, y=4: dist_xy = sqrt(16 + 16) > 4 -> no occlusion from walls
    # Actually the segment passes through at y=4, cylinder center at (5,0), r=4
    # Closest point: (5, 4, z) -> dist_xy = 4.0 = radius -> tangent
    result = segment_occluded(start, end, blocker)
    assert result is True  # disc ~ 0, should catch tangent


def test_segment_occluded_cylinder_clear_miss() -> None:
    """Segment passing far from cylinder returns False."""
    start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 1.0], dtype=np.float32)
    blocker = Obstacle(kind="cylinder", center=(5.0, 10.0, 0.0), radius=1.0, height=2.0)
    assert segment_occluded(start, end, blocker) is False


def test_segment_occluded_cylinder_vertical() -> None:
    """Vertical segment through cylinder (a~0, b~0, c<=0)."""
    start = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    end = np.array([5.0, 0.0, 10.0], dtype=np.float32)
    blocker = Obstacle(kind="cylinder", center=(5.0, 0.0, 0.0), radius=4.0, height=2.0)
    # Segment is exactly through cylinder axis, z in [0, 2] for cylinder
    assert segment_occluded(start, end, blocker) is True


def test_segment_occluded_sphere() -> None:
    """Sphere occlusion still works after changes."""
    start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 5.0], dtype=np.float32)
    blocker = Obstacle(kind="sphere", center=(5.0, 0.0, 5.0), radius=3.0)
    assert segment_occluded(start, end, blocker) is True


def test_segment_occluded_sphere_clear() -> None:
    """Sphere occlusion for non-intersecting segment."""
    start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 5.0], dtype=np.float32)
    blocker = Obstacle(kind="sphere", center=(5.0, 20.0, 5.0), radius=1.0)
    assert segment_occluded(start, end, blocker) is False


def test_segment_occluded_box() -> None:
    """Box occlusion still works."""
    start = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 5.0], dtype=np.float32)
    blocker = Obstacle(kind="box", center=(5.0, 0.0, 5.0), half_extents=(1.0, 1.0, 1.0))
    assert segment_occluded(start, end, blocker) is True


# --- Clearance helpers ---

def test_obstacle_clearance_cylinder() -> None:
    """Clearance inside vs outside cylinder."""
    obs = Obstacle(kind="cylinder", center=(0.0, 0.0, 0.0), radius=1.0, height=5.0)
    inside = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    outside = np.array([3.0, 0.0, 2.5], dtype=np.float32)
    assert obstacle_clearance(inside, obs, 0.0) < 0.0  # inside -> negative clearance
    assert obstacle_clearance(outside, obs, 0.0) > 0.0  # outside -> positive


def test_obstacle_clearance_sphere() -> None:
    """Clearance for sphere."""
    obs = Obstacle(kind="sphere", center=(0.0, 0.0, 0.0), radius=2.0)
    assert obstacle_clearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), obs, 0.0) == pytest.approx(-1.0)
    assert obstacle_clearance(np.array([4.0, 0.0, 0.0], dtype=np.float32), obs, 0.0) == pytest.approx(2.0)


def test_obstacle_collides() -> None:
    """obstacle_collides is clearance <= 0."""
    obs = Obstacle(kind="sphere", center=(0.0, 0.0, 0.0), radius=1.0)
    assert obstacle_collides(np.array([0.0, 0.0, 0.0], dtype=np.float32), obs, 0.0) is True
    assert obstacle_collides(np.array([2.0, 0.0, 0.0], dtype=np.float32), obs, 0.0) is False
