# DroneIntercept3D-v2 Bug Fixes

All bugs discovered during the thermo-nuclear code quality review of branch `More-Realistic-Env`.

---

## [CRITICAL] Cylinder occlusion false negative in `segment_occluded`

**File:** `drone_env/utils/obstacle_geometry.py:61-75`
**Function:** `segment_occluded`

### Problem

Cylinder occlusion checks only the segment's **XY-closest point** and its z-coordinate. For sloped sightlines the XY-closest point can be outside the cylinder's height range while the 3D segment genuinely passes through the cylinder volume.

### Repro

```python
import numpy as np
from drone_env.utils.obstacle_geometry import segment_occluded, Obstacle

start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
end   = np.array([10.0, 0.0, 10.0], dtype=np.float32)
blocker = Obstacle(
    kind="cylinder",
    center=np.array([5.0, 0.0, 0.0], dtype=np.float32),
    radius=4.0,
    height=2.0,
)
# Segment parametrically: (10t, 0, 10t)
# Inside cylinder when t ∈ [0.1, 0.2]
# But the XY-closest point is at t=0.5, z=5.0, outside z∈[0,2]
result = segment_occluded(start, end, blocker)
assert result == True  # FAILS — returns False
```

### Fix

Perform a proper 3D segment-vs-cylinder intersection using interval overlap:
1. **Side wall:** Solve the quadratic for where the segment's XY projection intersects the infinite cylinder's XY circle. This gives a `t`-interval `[t_side_min, t_side_max]` where the segment is within radius.
2. **Z overlap:** Solve for the `t`-interval where the segment's z is within `[z_min, z_max]` — trivially `[(z_min - start_z)/dz, (z_max - start_z)/dz]` intersected with `[0, 1]` (handle `dz ≈ 0` as full overlap).
3. **Result:** The segment occludes if the two `t`-intervals overlap (including endpoint touch). Also check whether either endpoint lies inside the cylinder volume.

---

## [CRITICAL] Frozen dataclass with mutable `np.ndarray` fields

**File:** `drone_env/utils/obstacle_geometry.py:13-21`
**Class:** `Obstacle`

### Problem

```python
@dataclass(frozen=True)
class Obstacle:
    kind: ObstacleKind
    center: np.ndarray           # ← mutable
    ...
    half_extents: np.ndarray | None = None  # ← mutable
```

`frozen=True` prevents reassignment but not mutation of the array contents. `obstacle.center[:] = new_values` silently corrupts state.

### Fix

Pick one:
- **Option A:** Remove `frozen=True` and document the class as mutable.
- **Option B:** Convert arrays to immutable types on construction: store as `tuple[float, ...]` instead of `np.ndarray`.

Option B is preferred if `Obstacle` is intended to be a pure value type.

---

## [HIGH] Dead import of `ObstacleKind`

**File:** `drone_env/envs/drone_intercept_3d_v2.py:20`

```python
from drone_env.utils.obstacle_geometry import (
    Obstacle,
    ObstacleKind,  # ← never used in this file
    obstacle_clearance,
    obstacle_collides,
    segment_occluded,
)
```

### Fix

Remove `ObstacleKind` from the import.

---

## [HIGH] Tree-style obstacle counting is counterintuitive

**File:** `drone_env/envs/drone_intercept_3d_v2.py:470-504`
**Function:** `_generate_obstacles`

### Problem

At `curriculum_level >= 2`, a canopy sphere is appended as a separate obstacle, but the final list is sliced to `[:cfg.obstacle_count]`. This means `obstacle_count=3` could produce 2 cylinders + 1 canopy or 3 cylinders, with no deterministic mapping.

### Fix

Either:
- Document `obstacle_count` as "number of trunk cylinders" and let canopies be bonus objects, or
- Count canopies toward the limit from the start by using `obstacle_count - 1` as the cylinder target when level >= 2.

---

## [HIGH] Ray/nearest clearance inconsistency

**File:** `drone_env/envs/drone_intercept_3d_v2.py:602-632`
**Function:** `_obstacle_sensor_readings`

### Problem

`nearest` clearance (line 618) is computed for **all** obstacle types, but `best_rays` only processes `"cylinder"` and `"sphere"`. If the nearest obstacle is a box, `nearest` reports a small clearance while all rays show `sensor_range`.

### Fix

Either compute rays for boxes too, or only consider the same obstacle subset for `nearest`.

---

## [MEDIUM] `obstacle_ray_count` / observation size mismatch

**File:** `drone_env/envs/drone_intercept_3d_v2.py`

### Problem

`observation_size` is hardcoded to `29` (line 154), which assumes exactly 8 obstacle rays. The observation layout is:

```
17 dims before rays + obstacle_ray_count + 4 trailing dims = 21 + obstacle_ray_count
```

If `cfg.obstacle_ray_count != 8`, the observation vector won't match `observation_space`.

### Repro

```python
from drone_env.envs.drone_intercept_3d_v2 import DroneIntercept3DConfigV2, DroneIntercept3DV2Env

cfg = DroneIntercept3DConfigV2(obstacle_ray_count=16, enable_obstacles=True, obstacle_count=1)
env = DroneIntercept3DV2Env(config=cfg)
obs, _ = env.reset(seed=0)
assert obs.shape == (29,)  # FAILS — obs.shape is (37,)
```

### Fix

Pick one:
- **Option 1 (freeze):** Remove `obstacle_ray_count` as a public field; hardcode to 8. The 29-dim schema is the documented contract.
- **Option 2 (dynamic):** Make `DroneIntercept3DConfigV2.observation_size` return `21 + self.obstacle_ray_count` and build `observation_space` to match dynamically.
- **Option 3 (validate):** In `__init__`, assert `cfg.obstacle_ray_count == 8` if the schema is fixed.

---

## [MEDIUM] `make_curriculum_config` chained `replace()`

**File:** `drone_env/envs/drone_intercept_3d_v2.py:157-185`

### Problem

Chained `replace()` calls create a linear dependency where each level builds on the previous. This works but is fragile when adding new levels.

### Fix

Use a level→overrides mapping:

```python
_LEVEL_UPDATES: dict[int, dict[str, Any]] = {
    1: {"target_behavior": "straight", "target_max_speed": 3.0},
    2: {"enable_obstacles": True, "obstacle_count": 3},
    3: {"enable_fov_limits": True},
    4: {"enable_occlusion": True, "obstacle_count": 5},
    5: {"target_behavior": "random_patrol", "obstacle_count": 7,
        "observation_noise_std": 0.01, "target_max_speed": 4.5},
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

---

## [MEDIUM] `random_rollout.py` try/except for env compatibility

**File:** `scripts/random_rollout.py:33-37`

### Problem

Relies on the specific `TypeError` message containing `"target_mode"` to detect v1 vs v2. Fragile across env changes.

### Fix

```python
if "v2" in args.env_id:
    env = gym.make(args.env_id, **kwargs)
else:
    env = gym.make(args.env_id, target_mode=args.target_mode, **kwargs)
```

---

## [MEDIUM] `_sample_target_position` deterministic fallback

**File:** `drone_env/envs/drone_intercept_3d_v2.py:437-459`

### Problem

If the 1000-retry loop fails, the fallback creates a fixed position at `(min_distance, 0, ...)` — deterministic and always the same. This could be a problem for edge-case reproducibility.

### Fix

Use a random fallback within bounds instead of a fixed position.

---

## Tracking

| # | Severity | Status |
|---|----------|--------|
| 1 | Critical | Open |
| 2 | Critical | Open |
| 3 | High | Open |
| 4 | High | Open |
| 5 | High | Open |
| 6 | Medium | Open |
| 7 | Medium | Open |
| 8 | Medium | Open |
| 9 | Medium | Open |
