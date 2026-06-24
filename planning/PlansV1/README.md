# Drone 3D Gymnasium Codex Task Pack

This pack gives Codex a mechanical implementation plan for a lightweight 3D drone-intercept Gymnasium environment.

Files:

- `drone_env_codex_goal_prompt.md`: copy/paste goal prompt for Codex.
- `drone_env_task_backlog.csv`: atomic task backlog with dependencies, verification commands, and acceptance criteria.
- `drone_env_reward_weights.csv`: the 10 reward components and suggested starting weights.
- `drone_env_acceptance_checks.csv`: final acceptance checklist.

Recommended usage:

1. Put these files in the repo root or a `planning/` folder.
2. Paste the goal prompt into Codex.
3. Tell Codex to follow the CSV backlog in dependency order.
4. Have Codex update task statuses as it completes work.
