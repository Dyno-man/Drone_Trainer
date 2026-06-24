# DroneIntercept3D-v2 Environment Spec

`DroneIntercept3D-v2` is a simulation-only Gymnasium environment for non-destructive drone tracking. It does not model payloads, damage, impact, weapons, individual motors, or real hardware flight code. Success means reaching and holding a safe capture/tracking radius around a larger target drone.

## Environment Contract

- Gymnasium id: `DroneIntercept3D-v2`
- Entry point: `drone_env.envs.drone_intercept_3d_v2:DroneIntercept3DV2Env`
- Config: `DroneIntercept3DConfigV2`
- Curriculum factory: `make_curriculum_config(level)`
- Action space: `Box(-1, 1, shape=(4,), dtype=float32)`
- Observation space: `Box(-1, 1, shape=(29,), dtype=float32)`

The v1 environment remains registered as `DroneIntercept3D-v0` and is not replaced.

## State

The agent is a compact cylindrical drone abstraction with:

- 3D position and velocity
- yaw and yaw-rate
- body radius, body height, and conservative `rotor_span_radius`

The target has independent 3D position and velocity. The default target behavior is hover; higher curriculum levels add straight and random patrol motion.

Tree-like obstacles are represented by simple primitives:

- vertical cylinders for trunks
- spheres for canopies
- boxes reserved for simple block volumes

Collision and clearance use the agent `rotor_span_radius`, not just the center point.

## Action

Action indices:

- `0`: forward acceleration command in agent yaw frame
- `1`: lateral/right acceleration command in agent yaw frame
- `2`: vertical acceleration command
- `3`: yaw-rate command

Commands are normalized to `[-1, 1]`. Acceleration, speed, and yaw-rate are clipped by config limits. The dynamics assume gravity-neutral hover; a zero action does not make the drone fall.

## Observation

Stable observation indices:

- `0:3`: agent position `[x/world_xy, y/world_xy, z/world_z]`
- `3:6`: agent velocity divided by `max_speed`
- `6`: `sin(yaw)`
- `7`: `cos(yaw)`
- `8`: yaw-rate divided by `max_yaw_rate`
- `9`: target-visible flag
- `10:13`: visible target relative position, normalized, or zeros when hidden
- `13:16`: last-seen target relative position, normalized, or zeros before first sighting
- `16`: normalized time since target was last seen
- `17:25`: obstacle ray distances in `[0, 1]`
- `25`: previous distance to target divided by maximum world distance
- `26`: success hold steps divided by required hold steps
- `27`: step count divided by max steps
- `28`: nearest obstacle clearance divided by sensor range, clipped to `[-1, 1]`

The target's exact relative position is hidden unless it is inside sensor range/cone and not occluded when occlusion is enabled.

## Perception

The sensor points forward from yaw with a configurable upward pitch bias. Visibility requires:

- target within `sensor_range`
- target inside the cone when FOV limits are enabled
- no obstacle blocking line of sight when occlusion is enabled

Obstacle rays are simple horizontal proximity rays around the drone and are normalized by sensor range.

## Reward Terms

Every step exposes `info["reward_terms"]` with these stable keys:

- `distance_progress`
- `target_visible`
- `reacquisition`
- `capture_hold`
- `success`
- `clearance`
- `launch_altitude`
- `time`
- `control`
- `collision`
- `out_of_bounds`

The scalar reward is the sum of the named terms.

## Termination

`terminated=True` when:

- success hold reaches `success_hold_steps`
- collision occurs and `terminate_on_collision=True`
- the agent leaves world bounds

`truncated=True` when `max_steps` is reached without termination.

The success condition is non-destructive: the drone must remain within `capture_radius` for the configured number of steps.

## Curriculum

Levels are intentionally simple and progressive:

- `0`: 3D kinematics, hover target, no obstacles, no FOV limits
- `1`: straight target motion
- `2`: randomized tree-like obstacles
- `3`: forward/upward FOV limits
- `4`: obstacle occlusion
- `5`: random target patrol and seeded observation noise

## Safe Scope

This pass intentionally excludes motor-level physics, real flight control, swarm behavior, impact semantics, payload behavior, damage, or weapon behavior. Future swarm fields may add friendly drone bearings, spacing, and communication observations after single-drone v2 metrics are stable.
