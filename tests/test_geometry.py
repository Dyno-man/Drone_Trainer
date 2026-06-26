import numpy as np

from drone_env.utils.geometry import (
    angle_between_unit_vectors,
    clip_vector_norm,
    closest_point_on_segment,
    distance,
    in_bounds,
    point_in_cone_or_frustum,
    safe_unit,
    segment_sphere_intersection,
)
from drone_env.utils.obstacle_geometry import (
    Obstacle,
    obstacle_clearance,
    obstacle_collides,
    segment_occluded,
)


def test_geometry_helpers_are_stable() -> None:
    assert distance(np.array([0, 0, 0]), np.array([3, 4, 0])) == 5.0
    np.testing.assert_array_equal(safe_unit(np.zeros(3, dtype=np.float32)), np.zeros(3))
    clipped = clip_vector_norm(np.array([3, 4, 0], dtype=np.float32), 2.0)
    assert np.linalg.norm(clipped) <= 2.0 + 1e-6
    assert in_bounds(np.array([0, 0, 1], dtype=np.float32), 100.0, 50.0)
    assert not in_bounds(np.array([106, 0, 1], dtype=np.float32), 100.0, 50.0, margin=5.0)


def test_angle_between_unit_vectors_handles_degenerate_inputs() -> None:
    assert angle_between_unit_vectors(np.zeros(3), np.array([1.0, 0.0, 0.0])) == np.pi
    assert angle_between_unit_vectors(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.0


def test_point_in_cone_or_frustum() -> None:
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert point_in_cone_or_frustum(origin, heading, np.array([10.0, 0.0, 0.0]), 90.0, 60.0, 20.0)
    assert not point_in_cone_or_frustum(origin, heading, np.array([-10.0, 0.0, 0.0]), 90.0, 60.0, 20.0)
    assert not point_in_cone_or_frustum(origin, heading, np.array([30.0, 0.0, 0.0]), 90.0, 60.0, 20.0)


def test_segment_helpers() -> None:
    point = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    start = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    end = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(closest_point_on_segment(point, start, end), np.zeros(3), atol=1e-6)
    assert segment_sphere_intersection(start, end, np.zeros(3), 0.1)
    assert not segment_sphere_intersection(start + np.array([0.0, 2.0, 0.0]), end + np.array([0.0, 2.0, 0.0]), np.zeros(3), 0.5)


def test_obstacle_geometry_canonical_module_path() -> None:
    cylinder = Obstacle("cylinder", center=np.array([0.8, 0.0, 0.0], dtype=np.float32), radius=0.2, height=5.0)
    assert obstacle_collides(np.array([0.0, 0.0, 2.0], dtype=np.float32), cylinder, 0.65)
    assert obstacle_clearance(np.array([0.0, 0.0, 2.0], dtype=np.float32), cylinder, 0.65) < 0.0

    sphere = Obstacle("sphere", center=np.array([2.0, 0.0, 0.0], dtype=np.float32), radius=0.5)
    assert obstacle_clearance(np.array([4.0, 0.0, 0.0], dtype=np.float32), sphere, 0.25) == 1.25

    start = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    end = np.array([10.0, 0.0, 2.0], dtype=np.float32)
    blocker = Obstacle("cylinder", center=np.array([5.0, 0.0, 0.0], dtype=np.float32), radius=0.8, height=5.0)
    assert segment_occluded(start, end, blocker)
