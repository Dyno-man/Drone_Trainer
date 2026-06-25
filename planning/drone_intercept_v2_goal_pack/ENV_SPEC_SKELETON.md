# ENV_SPEC.md Skeleton for DroneIntercept3D-v2

Codex should fill this in while implementing.

## Scope
Simulation-only, non-destructive RL environment. Success means reaching and holding a capture/tracking radius. No impact, payload, damage, or hardware flight code.

## Environment ID
- Proposed: `DroneIntercept3D-v2`

## Agent model
- Body abstraction:
- Rotor-span radius:
- Height:
- State variables:
- Dynamics assumptions:

## Target model
- State variables:
- Supported behaviors:
- Spawn constraints:

## Obstacles
- Supported primitives:
- Random generation rules:
- Spawn clearance rules:
- Collision rules:
- Occlusion rules:

## Action space
Document shape, dtype, bounds, and meaning of each element.

## Observation space
Document shape, dtype, bounds, and exact index/name for each field.

## Sensor model
- Vision cone:
- Upward bias:
- Range:
- Occlusion:
- Last-seen memory:
- Noise:

## Reward terms
Every term should be named and exposed in `info["reward_terms"]`.

| Term | Meaning | Sign | Notes |
|---|---|---|---|
| time_penalty | | negative | |
| distance_progress | | positive/negative | |
| target_visible | | positive | |
| reacquire_target | | positive | |
| clearance_penalty | | negative | |
| smooth_control_penalty | | negative | |
| launch_altitude | | positive | expires early |
| collision_penalty | | negative terminal | |
| out_of_bounds_penalty | | negative terminal | |
| success_reward | | positive terminal | hold-for-N |

## Termination/truncation
- Success:
- Collision:
- Out of bounds:
- Timeout:
- Target lost:

## Curriculum levels
| Level | Features enabled |
|---|---|
| 0 | 3D, no obstacles, visible/easy target |
| 1 | moving target |
| 2 | simple obstacles, no occlusion |
| 3 | limited FOV / upward cone |
| 4 | occlusion and last-seen behavior |
| 5 | noise, denser obstacles, harder target motion |

## Debug/rendering
- Top-down view:
- Side view:
- Episode recorder:

## Metrics
- success
- collision
- timeout
- total reward
- steps
- visibility ratio
- mean/min clearance
- curriculum level
- seed
