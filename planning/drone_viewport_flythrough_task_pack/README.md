# Drone Viewport + Fly-Through Codex Task Pack

This pack is for fine-tuning the existing 3D Gymnasium drone interception environment.

Main goals:

1. Change success from proximity/side-by-side capture to true fly-through interception.
2. Add a camera-like viewport so the pursuer must search, acquire, pursue, and intercept.

## Files

- `drone_viewport_flythrough_task_backlog.csv`
  - Atomic task list for Codex.
- `drone_viewport_flythrough_reward_weights.csv`
  - Proposed reward components and weights.
- `drone_viewport_flythrough_acceptance_checks.csv`
  - Commands and checks Codex must run.
- `drone_viewport_flythrough_goal_prompt.md`
  - Copy-paste slash goal prompt.

## Suggested repo placement

```text
planning/
  drone_viewport_flythrough_task_backlog.csv
  drone_viewport_flythrough_reward_weights.csv
  drone_viewport_flythrough_acceptance_checks.csv
  drone_viewport_flythrough_goal_prompt.md
```

## Suggested Codex instruction

```text
Use planning/drone_viewport_flythrough_goal_prompt.md as the goal.
Use planning/drone_viewport_flythrough_task_backlog.csv as the source of truth.
Work through tasks in dependency order.
Update task statuses as you complete them.
Run all acceptance checks before claiming success.
```
