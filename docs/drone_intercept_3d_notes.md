# DroneIntercept3D MVP Notes

Environment id: `DroneIntercept3D-v0`

Chosen layout:

- `drone_env/envs/drone_intercept_3d.py` contains the Gymnasium environment and config dataclass.
- `drone_env/utils/geometry.py` keeps pure vector helpers testable.
- `drone_env/utils/rendering.py` contains the lightweight pygame renderer.
- `scripts/` contains check, rollout, manual policy, and reward debug utilities.
- `tests/` contains focused pytest coverage for API, kinematics, rewards, episode logic, and rendering.

Assumptions:

- The workspace started with planning files only, so this creates a minimal Python package layout.
- `.venv/bin/python` is available in this environment; `python` is not currently on PATH.
- `stable-baselines3` is optional. `scripts/check_env.py` uses it when present and otherwise runs direct Gymnasium API checks.
- Human rendering may be limited by headless execution environments; `rgb_array` uses SDL's dummy driver.

Useful commands:

```bash
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/random_rollout.py --episodes 3
.venv/bin/python scripts/manual_policy_demo.py --target-mode straight --episodes 3
.venv/bin/python scripts/manual_policy_demo.py --target-mode orbit --render-mode human --episodes 1
.venv/bin/python scripts/reward_debug.py --steps 20
```

Verification run on 2026-06-23:

- `.venv/bin/python -c "import gymnasium, numpy, pygame, pytest; print('ok')"` passed.
- `.venv/bin/python -c "from drone_env.envs.drone_intercept_3d import DroneIntercept3DEnv; print(DroneIntercept3DEnv)"` passed.
- `.venv/bin/python - <<'PY' ... gym.make('DroneIntercept3D-v0') ... PY` passed.
- `.venv/bin/python -m pytest -q` passed: 26 tests.
- `.venv/bin/python scripts/check_env.py` passed using the Gymnasium fallback because `stable-baselines3` was not installed.
- `.venv/bin/python scripts/random_rollout.py --episodes 3` passed.
- `.venv/bin/python scripts/manual_policy_demo.py --target-mode straight --episodes 3` passed and captured in all 3 episodes.
- `.venv/bin/python scripts/manual_policy_demo.py --target-mode orbit --render-mode human --episodes 1` passed.
- `.venv/bin/python scripts/reward_debug.py --steps 20` passed.

Known environment caveat:

- The exact prompt commands using `python ...` fail in this container because `python` is not on PATH. Use `.venv/bin/python ...` here, or activate `.venv` in a shell that exposes `python`.
