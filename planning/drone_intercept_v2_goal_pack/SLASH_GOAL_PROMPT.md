/goal Build DroneIntercept3D-v2 as a simulation-only, non-destructive RL environment upgrade.

Context:
- The previous drone/research-agent phase was successful. Do not break the known-good v1 behavior.
- This v2 phase should move from a dot-chaser toward a small embodied cylindrical drone abstraction: central cylinder/capsule body, rotor-span collision radius, 3D motion, upward-forward sensing, randomized obstacles, and target tracking.
- The physical drone concept is a compact cylindrical body with protruding rotors. Do NOT simulate individual motors yet. Use simplified but realistic kinematics: velocity, acceleration limits, yaw/heading, and conservative collision radius.
- The success behavior is non-destructive: reaching and holding a capture/tracking radius around a larger target drone. Do not implement impact, payload, damage, weapon behavior, or real hardware flight code in this pass.

Primary objective:
Implement a clean, testable DroneIntercept3D-v2 Gymnasium/PufferLib-compatible environment where a hand-launched small drone starts near low altitude, scans upward-forward, acquires a target, avoids randomized tree-like obstacles, and succeeds by holding a safe capture radius for N steps.

Inputs:
- Use atomic_goals_drone_intercept_3d_v2.csv as the implementation backlog.
- Treat each row as an atomic task with its own acceptance criteria and test/check.
- Prioritize P0 first, then P1, then P2/P3 only after the core environment is stable.

Implementation rules:
1. Preserve v1 behavior. Create v2 files/classes/registration without replacing the old env.
2. Keep the implementation simple and inspectable. Prefer clear geometry helpers and small pure functions over clever code.
3. Centralize tunable values in a config dataclass or equivalent config object.
4. Every new environment feature must be deterministic under a fixed seed.
5. Every reward component must be named in reward_terms and exposed through info for debugging.
6. The observation schema must be documented with stable indices or named fields.
7. Use a conservative rotor-span radius for collisions and clearance, not just the drone center point.
8. The target should be visible only when inside sensor range/cone and not occluded when occlusion is enabled.
9. Curriculum levels should progressively add difficulty: 3D movement, target motion, obstacles, FOV limits, occlusion, and noise.
10. Add tests as you build, not at the end.

Definition of done:
- v1 still imports/runs.
- v2 env instantiates and passes reset/step smoke tests.
- Same seed produces same reset layout and observation.
- Action and observation spaces are valid under Gymnasium checks.
- Dynamics caps, collision, clearance, visibility, reward signs, termination, and success-hold behavior have tests.
- A short random-policy rollout completes without NaNs or crashes.
- A short train smoke command and eval smoke command complete.
- Docs/specs explain state, action, observation, reward terms, termination, curriculum, and safe simulation scope.

Execution plan:
1. Read the CSV backlog and group tasks by phase.
2. Implement P0 tasks in dependency order.
3. After each small cluster, run the relevant tests or smoke checks.
4. Commit/record progress after stable milestones.
5. Implement P1 tasks once P0 is stable.
6. Stop before over-engineering P2/P3 unless the core is passing.
7. Produce a final summary with completed task IDs, skipped/deferred task IDs, commands run, test results, and known risks.

Review yourself before finishing:
- Look for reward hacking opportunities.
- Look for coordinate-frame mistakes in yaw, FOV, and upward bias.
- Look for collision false negatives around rotor span.
- Look for nondeterministic reset behavior.
- Look for observation_space mismatches.
- Look for terminal/truncation mistakes.
- Look for docs drifting from implementation.
