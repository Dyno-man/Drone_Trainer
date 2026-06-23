# Codex Goal: Build a Lightweight 3D Gymnasium Drone Intercept Environment

You are working in my drone RL project. Build a clean MVP custom Gymnasium environment named `DroneIntercept3DEnv` for training a pursuer drone to intercept an adversarial target drone in 3D.

## Operating Mode

Use the CSV task backlog as the source of truth. Work through tasks in dependency order. For each task:

1. Implement the smallest correct change.
2. Run that task's verification command if possible.
3. Mark the task mentally as complete only if the acceptance criterion is met.
4. Do not skip tests.
5. Do not claim success for checks you did not run.

## MVP Scope

Single-agent Gymnasium environment:

- Trainable agent: pursuer drone
- Scripted adversarial agent: target drone

Do not implement full multi-agent PettingZoo yet. Keep the structure future-compatible, but stay focused.

## Environment Summary

World:

- x range: [-100, 100]
- y range: [-100, 100]
- z range: [0, 50]
- floor is z = 0

State per drone:

- position: x, y, z
- velocity: vx, vy, vz

Action space:

```python
spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

Action means desired 3D acceleration direction scaled by max acceleration.

Observation:

20-dimensional flat float32 vector containing normalized:

1. pursuer position
2. pursuer velocity
3. target position
4. target velocity
5. relative position
6. relative velocity
7. current distance
8. previous distance

Target modes:

- straight
- evasive
- orbit

Default target mode: evasive.

Termination:

- success if distance <= capture_radius
- crash if pursuer z <= 0
- out of bounds if pursuer exits arena beyond margin
- truncation if step_count >= max_steps

## Reward Function

Implement exactly 10 reward components and expose them in `info["reward_breakdown"]`:

1. capture_reward
2. distance_progress_reward
3. distance_penalty
4. time_penalty
5. alignment_reward
6. closing_speed_reward
7. energy_penalty
8. smoothness_penalty
9. altitude_penalty
10. safety_failure_penalty

Use the `drone_env_reward_weights.csv` values as the initial formula/weight source.

Final reward:

```python
reward = (
    capture_reward
    + distance_progress_reward
    + distance_penalty
    + time_penalty
    + alignment_reward
    + closing_speed_reward
    + energy_penalty
    + smoothness_penalty
    + altitude_penalty
    + safety_failure_penalty
)
```

## Rendering

Add lightweight 3D visualization using pygame, not a heavy game engine.

Support:

- `render_mode="human"`
- `render_mode="rgb_array"`

Render:

- arena bounding box
- floor grid
- pursuer drone
- target drone
- trails
- line between drones
- current distance
- step count
- reward/debug HUD

A simple fixed isometric projection is enough.

## Required Scripts

Create or update:

- `scripts/check_env.py`
- `scripts/random_rollout.py`
- `scripts/manual_policy_demo.py`
- optional but preferred: `scripts/reward_debug.py`

## Required Tests

Add pytest coverage for:

- reset returns valid observation
- step returns valid Gymnasium outputs
- capture termination
- crash termination
- out-of-bounds termination
- timeout truncation
- reward breakdown contains all components
- distance progress reward positive when closer
- alignment/closing/energy/smoothness/altitude penalties
- rgb_array rendering if possible

## Final Acceptance Commands

Run these before claiming completion:

```bash
python scripts/check_env.py
pytest -q
python scripts/random_rollout.py --episodes 3
python scripts/manual_policy_demo.py --target-mode straight --episodes 3
python scripts/manual_policy_demo.py --target-mode orbit --render-mode human --episodes 1
```

If a command cannot run because of a missing optional dependency or headless environment, say exactly which command failed and why.

## Final Response Format

When done, report:

1. Files created/modified.
2. Commands run.
3. Test/check results.
4. What assumptions were made.
5. What TODOs remain.

Be brutally honest. Do not pretend checks passed unless they actually passed.
