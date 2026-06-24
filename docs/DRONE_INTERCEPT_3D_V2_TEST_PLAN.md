# DroneIntercept3D-v2 Test Plan

## Definition Of Done For Atomic Tasks

- New behavior is deterministic under a fixed seed.
- Observation and action spaces contain reset and step outputs.
- Reward additions expose named terms through `info["reward_terms"]`.
- Collision and clearance checks use `rotor_span_radius`.
- V1 import and smoke behavior still work.
- Docs are updated when state, action, observation, reward, termination, or curriculum changes.
- At least one focused pytest or smoke command covers the changed behavior.

## Pytest Coverage

Run:

```bash
.venv/bin/python -m pytest -q
```

Focused v2 coverage lives in `tests/test_drone_intercept_3d_v2.py` and checks:

- v0 and v2 registration
- deterministic reset including obstacle layout
- action and observation space compliance
- launch and target spawn geometry
- speed, acceleration, yaw-rate, and hover assumptions
- out-of-bounds lifecycle behavior
- rotor-span collision and clearance
- seeded obstacle generation clearance
- target visibility, FOV limits, upward pitch bias, hidden observations, and last-seen memory
- obstacle proximity rays and occlusion
- reward term sum and signs
- success hold, collision, timeout, and random rollout behavior

## Smoke Commands

Gymnasium API check:

```bash
.venv/bin/python scripts/check_env.py --env-id DroneIntercept3D-v2
```

Random rollout:

```bash
.venv/bin/python scripts/random_rollout.py --env-id DroneIntercept3D-v2 --curriculum-level 2 --episodes 3
```

Train smoke:

```bash
.venv/bin/python scripts/v2_train_smoke.py --timesteps 64 --curriculum-level 0
```

Eval smoke with metrics CSV:

```bash
.venv/bin/python scripts/v2_eval_smoke.py --episodes 3 --curriculum-level 0 --metrics-path /tmp/drone_intercept_v2_eval_metrics.csv
```
