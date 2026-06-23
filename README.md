# Drone Trainer Gym Environment

Lightweight Gymnasium environment for training a pursuer drone to intercept a scripted target drone in 3D.

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
python scripts/manual_policy_demo.py --target-mode orbit --render-mode human --episodes 1
```

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
