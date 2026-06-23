import numpy as np

from drone_env.utils.geometry import clip_vector_norm, distance, in_bounds, safe_unit


def test_geometry_helpers_are_stable() -> None:
    assert distance(np.array([0, 0, 0]), np.array([3, 4, 0])) == 5.0
    np.testing.assert_array_equal(safe_unit(np.zeros(3, dtype=np.float32)), np.zeros(3))
    clipped = clip_vector_norm(np.array([3, 4, 0], dtype=np.float32), 2.0)
    assert np.linalg.norm(clipped) <= 2.0 + 1e-6
    assert in_bounds(np.array([0, 0, 1], dtype=np.float32), 100.0, 50.0)
    assert not in_bounds(np.array([106, 0, 1], dtype=np.float32), 100.0, 50.0, margin=5.0)
