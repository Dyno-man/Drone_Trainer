# Recommended Implementation Order

Use the CSV as source of truth, but this order should keep Codex from overbuilding too early.

## Milestone A — Safe baseline and contracts
D2-000 through D2-016

Goal: v2 exists, v1 still works, config/action/observation/info/reward contracts are clear.

## Milestone B — 3D body and dynamics
D2-020 through D2-034

Goal: the agent is no longer a dot. It has 3D motion, yaw, acceleration limits, body size, and rotor-span collision radius.

## Milestone C — Launch and target
D2-040 through D2-054

Goal: hand-launch start and target-above/away behavior works with simple target motion.

## Milestone D — Obstacles and perception
D2-060 through D2-077

Goal: random tree-like obstacles, clearance, FOV, last-seen state, and optional occlusion/noise exist.

## Milestone E — Reward and success semantics
D2-080 through D2-089

Goal: reward is inspectable, success requires hold-for-N steps, and failure cases are clean.

## Milestone F — Curriculum/training/eval
D2-090 through D2-095

Goal: train/eval smoke commands exist and log useful metrics.

## Milestone G — Rendering/tests/docs/review
D2-100 through D2-135

Goal: debug renders, tests, specs, and Qwen review handoff are ready.

## Stop condition
Stop and report once P0 and P1 are complete and passing. Defer P2/P3 if they threaten stability.
