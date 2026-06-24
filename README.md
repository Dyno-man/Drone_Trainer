# Drone Trainer Gym Environment

Lightweight Gymnasium environment for training a pursuer drone to intercept a scripted target drone in 3D.
The current default task uses a camera-like viewport and succeeds when the pursuer's movement segment flies through the target capture volume.

Environment id:

```bash
DroneIntercept3D-v0
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "stable-baselines3>=2.3"
```

## Quick Checks

```bash
python -m pytest -q
python scripts/check_env.py
python scripts/random_rollout.py --episodes 3
python scripts/manual_policy_demo.py --target-mode straight --episodes 3
python scripts/viewport_search_demo.py --target-mode straight
python scripts/flythrough_regression_demo.py
python scripts/manual_policy_demo.py --target-mode orbit --render-mode human --episodes 1
```

## Task Definition

The trainable agent is the pursuer drone. The adversarial drone is scripted with `straight`, `evasive`, or `orbit` target modes.

Success is a fly-through intercept: the previous-to-current pursuer position segment must intersect the target sphere defined by `intercept_radius`. A side-by-side pass no longer counts as success when `require_flythrough_success=True`.

The default `observation_mode` is `viewport`. In viewport mode the observation contains pursuer state, pursuer heading, target estimate features, visibility flags, lock state, normalized `steps_since_seen`, angle-to-estimate, and distance-to-estimate. Fresh target-relative state is exposed only while the target is visible or while target lock memory is active. Set `DroneIntercept3DConfig(observation_mode="privileged")` to keep the debug mode with direct target state.

Key viewport and intercept knobs live in `DroneIntercept3DConfig`:

- `horizontal_fov_deg`, `vertical_fov_deg`, `viewport_range`
- `lock_memory_steps`
- `intercept_radius`, `flythrough_plane_radius`, `min_closing_speed`
- `require_los_for_pursuit_reward`, `require_flythrough_success`

Reward breakdown entries are stable every step. In addition to the original dense shaping, the environment reports `flythrough_intercept_reward`, `first_acquisition_reward`, `visibility_reward`, `reacquisition_reward`, `lost_target_penalty`, and `aim_through_target_reward`.

## Train

Start with straight target mode before trying evasive mode:

```bash
python bench/train.py --timesteps 200000 --target-mode straight --model-path models/ppo_drone_straight
```

Evaluate with rendering:

```bash
python bench/eval.py --model-path models/ppo_drone_straight --target-mode straight --render-mode human --episodes 5
```

Then train on the default evasive target:

```bash
python bench/train.py --timesteps 500000 --target-mode evasive --model-path models/ppo_drone_evasive
python bench/eval.py --model-path models/ppo_drone_evasive --target-mode evasive --render-mode human --episodes 5
```

## Project Layout

- `drone_env/envs/drone_intercept_3d.py` - Gymnasium environment and config.
- `drone_env/utils/geometry.py` - vector math helpers.
- `drone_env/utils/rendering.py` - lightweight pygame renderer.
- `scripts/` - environment checks, rollouts, demos, reward debugging.
- `bench/` - PPO train/eval entry points.
- `tests/` - pytest coverage for API, kinematics, rewards, terminations, and rendering.
- `planning/` - original task backlog and acceptance planning files.

## Notes

- Trained model archives are ignored by git by default. Save important weights outside the repo or publish them separately.
- `stable-baselines3` is optional for environment checks but required for `bench/train.py` and `bench/eval.py`.
