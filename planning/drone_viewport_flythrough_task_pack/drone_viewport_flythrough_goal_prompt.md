# Slash Goal: Fine-Tune Drone Env for Viewport Search + Fly-Through Interception

You are improving an existing 3D Gymnasium drone interception environment that is already training well.

Do not rebuild the whole project.

The current issue is:

- The pursuer drone often flies directly next to the adversarial drone.
- The desired behavior is that the pursuer flies through the adversarial drone.
- The pursuer should not have perfect omniscient target awareness forever.
- The pursuer should have a camera-like viewport/field-of-view.
- The agent should first search for the adversarial drone, acquire it in its viewport, pursue it, and then succeed only by flying through it.

Use the CSV files in this planning pack as the source of truth:

```text
planning/drone_viewport_flythrough_task_backlog.csv
planning/drone_viewport_flythrough_reward_weights.csv
planning/drone_viewport_flythrough_acceptance_checks.csv
```

Work through the backlog in dependency order.

Update task statuses as you complete them.

## Non-Negotiable Design Rules

1. Preserve the current environment behavior that is already training well where possible.
2. Do not rewrite the whole environment.
3. Keep this as a lightweight Gymnasium single-agent env for now.
4. The trainable agent is the pursuer drone.
5. The adversarial drone may remain scripted for this phase.
6. Do not add heavy 3D engines.
7. Pygame/simple projection rendering is acceptable.
8. Use tests for every important behavior change.
9. Do not claim success unless you actually ran the checks.
10. Make reward weights easy to tune.

## New Success Definition

Old behavior:
- Success could happen when the pursuer gets near the target.

New behavior:
- Success should happen when the pursuer path passes through the target's capture volume.

Implementation guidance:
- Use previous pursuer position and current pursuer position.
- Treat the target as a small sphere or capture volume.
- If the movement segment intersects that target volume, this is a fly-through intercept.
- A side-by-side pass should not count.
- Overshooting is fine if the path crossed through the target during the step.

## New Viewport Definition

The pursuer drone should have a camera-like field of view.

Minimum viable implementation:
- Track a normalized pursuer heading vector.
- Derive heading from velocity when moving.
- Keep previous heading when nearly stationary.
- Use a cone-style FOV check for MVP.
- Target is visible only if:
  - it is in front of the pursuer,
  - inside FOV angle,
  - inside viewport range.

Expose in `info`:
- `target_visible`
- `has_target_lock`
- `steps_since_seen`
- `last_seen_target_pos`
- `flythrough_intercept`

## Observation Mode

Keep a debug/legacy privileged mode if it exists or is useful.

Add viewport mode:

```python
observation_mode = "viewport"
```

In viewport mode:
- Do not expose perfect fresh target state when the target is hidden.
- If visible, expose target-relative state.
- If recently seen, expose last-seen estimate and lock flag.
- If not visible and lock expired, expose no-target/search features.
- Observation shape must stay constant.

## Reward Direction

Keep the current useful dense shaping, but retune success around fly-through.

The highest reward should be true fly-through intercept.

Add/adjust reward components for:
- fly-through intercept
- aim-through-target / low projected miss distance
- first visual acquisition
- maintaining visibility
- reacquisition
- lost target penalty
- distance progress
- closing speed
- time penalty
- safety penalties

Every component should appear in `info["reward_breakdown"]` every step.

## Required Final Checks

Run these before saying the work is complete:

```bash
python scripts/check_env.py
pytest -q
python scripts/random_rollout.py
python scripts/manual_policy_demo.py
python scripts/viewport_search_demo.py --target-mode straight
python scripts/flythrough_regression_demo.py
```

If a command cannot be run, explain exactly why.

## Final Response Required

When finished, report:

1. Files created/modified.
2. What changed about success behavior.
3. What changed about viewport/search behavior.
4. What reward components were added or adjusted.
5. Commands run and results.
6. Any known limitations or TODOs.

Be honest. Do not claim unverified success.
