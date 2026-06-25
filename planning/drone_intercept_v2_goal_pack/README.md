# DroneIntercept3D-v2 Goal Pack

This pack contains the implementation backlog and Codex/Qwen prompts for the next phase of the drone RL environment.

## Files

- `atomic_goals_drone_intercept_3d_v2.csv` — 90 atomic implementation goals with acceptance criteria, dependencies, tests, priority, and complexity.
- `SLASH_GOAL_PROMPT.md` — copy/paste prompt for Codex using `/goal`.
- `QWEN_THERMONUCLEAR_REVIEW_PROMPT.md` — handoff prompt for your Qwen model after Codex finishes.
- `ENV_SPEC_SKELETON.md` — a spec skeleton Codex can fill as it implements.
- `IMPLEMENTATION_ORDER.md` — recommended order to reduce breakage and keep tasks atomic.

## Core v2 direction

Build a simulation-only, non-destructive RL environment where a small cylindrical drone abstraction launches from low altitude, scans upward-forward, avoids randomized tree-like obstacles, and succeeds by holding a capture/tracking radius around a target drone.

The important design choice is not photorealism yet. The important jump is from dot-chaser to embodied 3D agent with:

- 3D position/velocity/yaw
- acceleration and turn-rate limits
- cylindrical body and rotor-span collision radius
- forward/upward sensor cone
- target visibility and last-seen memory
- obstacle clearance and collision penalties
- curriculum progression
- deterministic seeding and tests

## Priority use

Do P0 rows first. Then P1. P2/P3 are polish/future-proofing.
