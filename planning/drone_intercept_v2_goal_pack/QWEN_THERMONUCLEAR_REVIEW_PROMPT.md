# Qwen / Thermonuclear Debugging Review Prompt

You are reviewing the DroneIntercept3D-v2 implementation after Codex completes the backlog.

Goal:
Perform a hard, adversarial review of the implementation. Do not assume it is correct. Identify correctness bugs, reward hacking risks, deterministic seeding failures, geometry mistakes, test gaps, and docs mismatches.

Scope/safety:
This is a simulation-only non-destructive RL environment. Success means holding a capture/tracking radius around a target. Do not suggest payload, damage, impact, or real-world flight deployment logic.

Review checklist:
1. Environment registration
   - v1 still works.
   - v2 has a distinct id/class/path.
   - reset and step obey Gymnasium API.

2. Determinism
   - Same seed creates same spawn, obstacle layout, target path, and noisy observations if noise is enabled.
   - Different seeds actually vary layouts.

3. Observation/action spaces
   - observation_space contains every reset/step observation.
   - action_space matches actual accepted action shape/range.
   - Observation docs match implementation indices/order.

4. Dynamics
   - dt integration is consistent.
   - Velocity, acceleration, yaw, and bounds caps are enforced.
   - No-action hover assumption is intentional and documented.

5. Geometry/collision
   - Drone collision uses rotor_span_radius, not only body center.
   - Clearance distances are correct and become unsafe before collision.
   - Cylinder/sphere/box obstacle helpers have obvious unit tests.

6. Perception
   - Forward/upward vision cone math is correct.
   - Target visibility respects range, angle, and occlusion when enabled.
   - Last-seen fields reset/update correctly.
   - Hidden target observations do not leak exact current target location.

7. Rewards
   - Reward terms are named and exposed in info.
   - Total reward equals sum of named terms.
   - Distance-progress cannot be exploited by oscillation.
   - Visibility/reacquisition rewards cannot dominate the actual task.
   - Collision/OOB penalties are strong and terminal.
   - Success requires hold-for-N-steps, not a single lucky touch.

8. Curriculum/training
   - Curriculum levels enable features in the intended order.
   - Training and eval commands use the correct env id/version.
   - Metrics CSV includes success, collision, timeout, reward, steps, visibility ratio, and clearance.

9. Tests/docs
   - Test coverage maps to the atomic goals CSV.
   - ENV_SPEC.md, TEST_PLAN.md, and README do not contradict the code.
   - Any skipped tasks are explicitly listed and justified.

Output format:
- Critical blockers
- Major issues
- Minor issues
- Reward hacking risks
- Missing tests
- Suggested fixes in priority order
- Final verdict: PASS / PASS WITH FIXES / FAIL
