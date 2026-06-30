# BUGFIXES_V2 Implementation Plan

## Dependency Graph

```
Bug 2 (Obstacle tuple) ──┐
    ├── Bug 1 (cylinder occlusion)
    ├── Bug 4 (obstacle counting)
    ├── Bug 5 (ray/nearest)
    └── Bug 6 (observation_size)
Bug 3 (dead import) ─ independent
Bug 7 (curriculum map) ─ independent
Bug 8 (random_rollout) ─ independent
Bug 9 (target fallback) ─ independent
All ────────────────────► run tests + smoke
```

- **Bug 2** is the foundation - changes Obstacle type contract used by everything.
- **Bug 1** must pass the exact repro from the bug doc.

---

## Bug 1 CRITICAL: Cylinder occlusion false negative

**File:** `drone_env/utils/obstacle_geometry.py:61-75`
**Function:** `segment_occluded`

### Problem

Current code finds the XY-closest point on the segment and checks if that point is inside the cylinder volume. For sloped sightlines the XY-closest point can lie outside the cylinder's z-range while the 3D segment genuinely passes through the cylinder.

### Fix: 3D segment-vs-cylinder via interval intersection

**Math.** Segment P(t) = start + t*seg for t in [0,1]. Cylinder XY-circle: ||P_xy(t) - center|| <= r where r = radius + rotor_radius.

Let `m = start[:2] - center[:2]`. Then `P_xy(t) - center[:2] = m + t*seg_xy`.

Condition: ||m + t*seg_xy||^2 <= r^2
-> (seg_xy*seg_xy)*t^2 + 2(m*seg_xy)*t + (m*m - r^2) <= 0

Call this `a*t^2 + b*t + c <= 0` and solve for the t-interval where it holds. Also solve for z-interval where z is in [z_min, z_max]. Occlusion = these two intervals overlap.

Verified correct with the exact repro: start=(0,0,0), end=(10,0,10), cylinder at (5,0,0) r=4 h=2
- a=100, b=-100, c=9 -> roots at t=0.1, t=0.9 (t_xy=[0.1, 0.9])
- z: t_z=[0.0, 0.2]
- overlap: [0.1, 0.2] -> occludes=True (correct)

### Code

```python
def _solve_quadratic_interval(a, b, c):
    """Return (t_enter, t_exit) for where a*t^2 + b*t + c <= 0, or (None, None)."""
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


def _point_in_cylinder(point, center, radius, height, rotor_radius=0.0):
    dist_xy = float(np.linalg.norm(point[:2] - center[:2]))
    if dist_xy > radius + rotor_radius:
        return False
    z_min = float(center[2])
    z_max = z_min + height
    return z_min <= float(point[2]) <= z_max
```

In `segment_occluded`, replace the cylinder branch (lines 61-75):

```python
if obstacle.kind == "cylinder":
    center = np.asarray(obstacle.center, dtype=np.float32)  # kept from original line 62
    seg = end - start
    seg_xy = seg[:2]
    m = start[:2] - center[:2]  # key: start - center, NOT center - start

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
```

### Test cases
- Exact repro from BUGFIXES_V2.md: (0,0,0)->(10,0,10) through cylinder at (5,0,0) r=4 h=2 -> True
- Regression: horizontal cylinder, perpendicular segment, sphere occlusion unchanged
- Vertical segment through cylinder (a~0, b~0, c<=0) returns (0.0, 1.0)
- Tangent touch: disc~0, returns True

---

## Bug 2 CRITICAL: Frozen dataclass with mutable np.ndarray

**File:** `drone_env/utils/obstacle_geometry.py:13-21`

### Problem

`frozen=True` prevents reassignment but not mutation: `obstacle.center[:] = new_values` silently corrupts state. Type annotations alone don't coerce - passing `np.array(...)` still stores ndarray.

### Fix: `__post_init__` coercion to immutable tuples

```python
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
```

`object.__setattr__` bypasses the frozen guard (required in `__post_init__`). `tuple()` coerces both ndarrays and lists to tuples.

### Caller Updates - tuple minus ndarray raises TypeError

Every arithmetic site must convert tuple to ndarray via `np.asarray()`.

| File | Line | Change |
|------|------|--------|
| `obstacle_geometry.py` | 24-25 | `np.asarray(obstacle.center, ...)` - already handles tuples |
| `obstacle_geometry.py` | 27 | `center` already `np.asarray`'d at line 25, ok |
| `obstacle_geometry.py` | 62 | `np.asarray(obstacle.center, ...)` - already present |
| `obstacle_geometry.py` | 41 | `np.asarray(obstacle.half_extents, ...)` - already handles tuples |
| `drone_intercept_3d_v2.py` | 480-486 | `center = np.array([...])` -> `center = (float(...), float(...), 0.0)` |
| `drone_intercept_3d_v2.py` | 497 | `canopy_center = center + np.array(...)` -> `canopy_center = (center[0], center[1], height)` |
| `drone_intercept_3d_v2.py` | 621 | `center_xy = obstacle.center[:2]` -> `center_xy = np.asarray(obstacle.center)[:2]` |

### Test cases
- `Obstacle(center=np.array([1,2,3]))` stores center == (1.0, 2.0, 3.0) (tuple, not ndarray)
- `obstacle.center[:] = ...` raises TypeError (tuples have no __setitem__)
- All existing callers still work (verified by test suite)

---

## Bug 3 HIGH: Dead import of `ObstacleKind`

**File:** `drone_env/envs/drone_intercept_3d_v2.py:20`

### Fix

Remove `ObstacleKind` from the import list. One-line edit.

---

## Bug 4 HIGH: Tree-style obstacle counting is counterintuitive

**File:** `drone_env/envs/drone_intercept_3d_v2.py:470-504`

### Problem

At level >= 2, canopy spheres are appended as separate obstacles, but `[:cfg.obstacle_count]` slices the final list. `obstacle_count=3` could produce 2 cylinders + 1 canopy or 3 cylinders with no deterministic mapping.

### Fix: Count canopies toward the limit

Use `obstacle_count` as the total cap including canopies. Each tree (cylinder + optional canopy) adds 2 items toward the count:

```python
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
                obstacles.append(canopy)  # 2 items toward count
            else:
                obstacles.append(trunk)  # 1 item
        else:
            obstacles.append(trunk)

    return obstacles[: cfg.obstacle_count]
```

### Test cases
- `obstacle_count=3, level=2` returns exactly 3 items, composition includes canopy spheres
- `obstacle_count=5, level=4` returns exactly 5 items

---

## Bug 5 HIGH: Ray/nearest clearance inconsistency

**File:** `drone_env/envs/drone_intercept_3d_v2.py:602-632`

### Problem

Line 618 computes `nearest` for all obstacle types, but lines 619-630 only compute `best_rays` for cylinders and spheres. Box obstacles show small clearance in `nearest` but full `sensor_range` in rays.

### Fix

Add the same `obstacle.kind in ("cylinder", "sphere")` guard before the `nearest = min(...)` line:

```python
for obstacle in self.obstacles:
    if obstacle.kind in ("cylinder", "sphere"):
        nearest = min(nearest, obstacle_clearance(...))
    else:
        continue
```

One-line addition + one `else: continue`.

### Test cases
- With a box obstacle: `nearest` and all ray values should both report `sensor_range` (box is ignored by both)
- With only cylinders/spheres: no behavioral change

---

## Bug 6 MEDIUM: `observation_size` / observation size mismatch

**File:** `drone_env/envs/drone_intercept_3d_v2.py:152-154`

### Fix: Make observation_size dynamic (Option 2)

```python
@property
def observation_size(self) -> int:
    return 21 + self.obstacle_ray_count
```

The schema is 17 dims + obstacle_ray_count + 4 dims = 21 + obstacle_ray_count. The hardcoded 29 is just 21 + 8 (the default). Gymnasium observation_space is built using `self.config.observation_size` so it will match automatically.

### Test cases
- `obstacle_ray_count=8` (default): obs shape (29,)
- `obstacle_ray_count=16`: obs shape (37,)

---

## Bug 7 MEDIUM: `make_curriculum_config` chained replace()

**File:** `drone_env/envs/drone_intercept_3d_v2.py:157-185`

### Fix: Level->overrides mapping

```python
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
    if level < 0 or level > 5:
        raise ValueError("curriculum level must be between 0 and 5")
    cfg = DroneIntercept3DConfigV2(curriculum_level=level)
    for lvl in range(1, level + 1):
        if lvl in _LEVEL_UPDATES:
            cfg = replace(cfg, **_LEVEL_UPDATES[lvl])
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
```

### Test cases
- Verify each level's overrides (behavior, speed, obstacles, FOV, occlusion, noise)
- Verify overrides parameter still works at each level

---

## Bug 8 MEDIUM: `random_rollout.py` try/except for env compatibility

**File:** `scripts/random_rollout.py:32-37`

### Fix

```python
if "v2" in args.env_id:
    env = gym.make(args.env_id, **kwargs)
else:
    env = gym.make(args.env_id, target_mode=args.target_mode, **kwargs)
```

Replace the try/except block. Simple, explicit, no brittle string matching on exception messages.

---

## Bug 9 MEDIUM: `_sample_target_position` deterministic fallback

**File:** `drone_env/envs/drone_intercept_3d_v2.py:452-459`

### Problem

1000-retry loop fallback creates fixed position at `(min_distance, 0, ...)` - deterministic and always the same.

### Fix: Random fallback within bounds

```python
yaw = float(self.np_random.uniform(-np.pi, np.pi))
horizontal_distance = float(self.np_random.uniform(*cfg.target_distance_range))
altitude_gap = float(self.np_random.uniform(*cfg.target_altitude_gap_range))
fallback = self.agent_pos + np.array(
    [np.cos(yaw) * horizontal_distance, np.sin(yaw) * horizontal_distance, altitude_gap],
    dtype=np.float32,
)
fallback[0] = np.clip(fallback[0], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
fallback[1] = np.clip(fallback[1], -cfg.world_xy * 0.85, cfg.world_xy * 0.85)
fallback[2] = np.clip(fallback[2], cfg.launch_height_range[1] + 2.0, cfg.world_z * 0.9)
return fallback.astype(np.float32)
```

Same random logic as the retry loop, no distance validation (accept whatever lands in bounds).

### Test cases
- Fallback is not deterministic (two calls with exhausted retries give different positions)

---

## Test Strategy

### New tests to add

**`tests/test_obstacle_geometry.py`** (new file or section):
- `test_segment_occluded_cylinder_slope` - exact BUGFIXES_V2 repro
- `test_segment_occluded_cylinder_horizontal` - dz~0 segment
- `test_segment_occluded_cylinder_tangent` - disc~0, returns True
- `test_segment_occluded_cylinder_clear_miss` - near miss returns False
- `test_obstacle_center_immutable` - tuple center cannot be mutated

**`tests/test_drone_intercept_3d_v2.py`** (add sections):
- `test_observation_size_dynamic` - ray_count=16 -> obs shape (37,)
- `test_generate_obstacles_count` - level>=2 with obstacle_count=N returns exactly N items
- `test_curriculum_config_levels` - verify each level's overrides
- `test_sample_target_fallback_random` - fallback is not deterministic

### Run
```bash
pytest tests/
scripts/v2_train_smoke.py
scripts/v2_eval_smoke.py
```

### Quick check for Bug 1
```bash
python -c "
import numpy as np
from drone_env.utils.obstacle_geometry import segment_occluded, Obstacle
start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
end   = np.array([10.0, 0.0, 10.0], dtype=np.float32)
blocker = Obstacle(kind='cylinder', center=(5.0, 0.0, 0.0), radius=4.0, height=2.0)
assert segment_occluded(start, end, blocker) == True
print('PASS')
"
```

---

## Implementation Order

1. Bug 2 - Obstacle tuple migration with __post_init__ (foundation)
2. Bug 1 - Cylinder occlusion (critical, uses Bug 2's type)
3. Bug 3 - Dead import (trivial, independent)
4. Bug 4 - Tree obstacle counting (uses Bug 2's tuple center)
5. Bug 5 - Ray/nearest inconsistency (independent)
6. Bug 6 - observation_size dynamic (independent)
7. Bug 7 - Curriculum mapping (independent)
8. Bug 8 - random_rollout fix (independent)
9. Bug 9 - Target fallback (independent)
10. Write new tests
11. Run full test suite + smoke tests
12. Verify exact BUGFIXES_V2 repro case

---

## Files Changed

| File | Bugs |
|------|------|
| `drone_env/utils/obstacle_geometry.py` | 1, 2 |
| `drone_env/envs/drone_intercept_3d_v2.py` | 3, 4, 5, 6, 7, 9 |
| `scripts/random_rollout.py` | 8 |
| `tests/test_obstacle_geometry.py` | new |
| `tests/test_drone_intercept_3d_v2.py` | new tests |
