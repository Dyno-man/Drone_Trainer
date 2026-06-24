from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.editable.curriculum import config_overrides, scenario_repeats
from autoresearch.editable.recipe import select_action
from autoresearch.editable.reward_weights import build_reward_weights
from autoresearch.locked.benchmark_scenarios import (
    SIDE_PASS_PROBES,
    BenchmarkScenario,
    scenarios_for_mode,
)
from drone_env.envs.drone_intercept_3d import DroneIntercept3DConfig, DroneIntercept3DEnv
from drone_env.utils.geometry import safe_unit


class PredictModel(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        ...


def _build_config(mode: str, scenario: BenchmarkScenario) -> DroneIntercept3DConfig:
    kwargs = config_overrides(mode)
    if scenario.max_steps is not None:
        kwargs["max_steps"] = scenario.max_steps
    kwargs["reward_weights"] = build_reward_weights()
    return DroneIntercept3DConfig(**kwargs)


class BenchmarkSetup:
    """Applies benchmark scenario state to the current env implementation."""

    def __init__(self, scenario: BenchmarkScenario) -> None:
        self.scenario = scenario

    def apply(self, env: DroneIntercept3DEnv) -> None:
        scenario = self.scenario
        env.pursuer_pos = np.asarray(scenario.pursuer_pos, dtype=np.float32)
        env.previous_pursuer_pos = env.pursuer_pos.copy()
        env.pursuer_vel = np.asarray(scenario.pursuer_vel, dtype=np.float32)
        env.pursuer_heading = self._heading(scenario.pursuer_heading)
        env.target_pos = np.asarray(scenario.target_pos, dtype=np.float32)
        env.target_vel = np.asarray(scenario.target_vel, dtype=np.float32)
        env.previous_distance = float(np.linalg.norm(env.target_pos - env.pursuer_pos))
        env.previous_action = np.zeros(3, dtype=np.float32)
        env.steps_since_seen = env.config.lock_memory_steps + 1
        env.has_target_lock = False
        env.target_visible = False
        env._has_ever_seen_target = False
        env._update_visibility_state()
        env.last_info = env._base_info(False, False, False)

    @staticmethod
    def _heading(values: tuple[float, float, float]) -> np.ndarray:
        heading = safe_unit(np.asarray(values, dtype=np.float32))
        if float(np.linalg.norm(heading)) == 0.0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return heading


def _episode_action(model: PredictModel | None, obs: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
    if model is not None:
        model_obs = _adapt_observation_for_model(model, obs)
        action, _ = model.predict(model_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    return select_action(obs, info, step)


def _adapt_observation_for_model(model: PredictModel, obs: np.ndarray) -> np.ndarray:
    model_space = getattr(model, "observation_space", None)
    expected_shape = getattr(model_space, "shape", None)
    if not expected_shape or tuple(expected_shape) == tuple(obs.shape):
        return obs
    expected_size = int(np.prod(expected_shape))
    flat_obs = obs.reshape(-1)
    if expected_size <= flat_obs.size:
        return flat_obs[:expected_size].reshape(expected_shape).astype(np.float32)
    padded = np.zeros(expected_size, dtype=np.float32)
    padded[: flat_obs.size] = flat_obs
    return padded.reshape(expected_shape).astype(np.float32)


def run_episode(
    scenario: BenchmarkScenario,
    *,
    mode: str,
    repeat_index: int,
    model: PredictModel | None = None,
) -> dict[str, Any]:
    env = DroneIntercept3DEnv(config=_build_config(mode, scenario), target_mode=scenario.target_mode)
    try:
        obs, info = env.reset(seed=scenario.seed + repeat_index)
        BenchmarkSetup(scenario).apply(env)
        obs = env._get_obs()
        info = env.last_info.copy()
        acquired = bool(info["target_visible"])
        first_acquire_step = 0 if acquired else None
        lost_target = False
        total_reward = 0.0
        final_info = info
        for step in range(env.config.max_steps):
            action = _episode_action(model, obs, info, step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_info = info
            if info["target_visible"] and first_acquire_step is None:
                first_acquire_step = step + 1
                acquired = True
            if acquired and not info["target_visible"] and not info["has_target_lock"]:
                lost_target = True
            if terminated or truncated:
                break
        return {
            "scenario": scenario.name,
            "repeat_index": repeat_index,
            "target_mode": scenario.target_mode,
            "steps": int(env.step_count),
            "reward": float(total_reward),
            "first_acquired": bool(first_acquire_step is not None),
            "steps_to_acquire": first_acquire_step,
            "flythrough_success": bool(final_info["flythrough_intercept"] and final_info["captured"]),
            "steps_to_intercept": int(env.step_count)
            if final_info["flythrough_intercept"] and final_info["captured"]
            else None,
            "crashed": bool(final_info["crashed"]),
            "out_of_bounds": bool(final_info["out_of_bounds"]),
            "lost_target": bool(lost_target),
            "final_distance": float(final_info["distance"]),
            "final_info": {
                key: value
                for key, value in final_info.items()
                if key
                in {
                    "captured",
                    "crashed",
                    "out_of_bounds",
                    "distance",
                    "target_mode",
                    "target_visible",
                    "has_target_lock",
                    "steps_since_seen",
                    "flythrough_intercept",
                }
            },
            "scenario_definition": asdict(scenario),
        }
    finally:
        env.close()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@lru_cache(maxsize=None)
def run_side_pass_probes(mode: str) -> tuple[dict[str, Any], ...]:
    probes = []
    for probe in SIDE_PASS_PROBES:
        env = DroneIntercept3DEnv(config=_build_config(mode, probe), target_mode=probe.target_mode)
        try:
            env.reset(seed=probe.seed)
            BenchmarkSetup(probe).apply(env)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            probes.append(
                {
                    "scenario": probe.name,
                    "false_success": bool(info["captured"] or info["flythrough_intercept"]),
                    "captured": bool(info["captured"]),
                    "flythrough_intercept": bool(info["flythrough_intercept"]),
                    "distance": float(info["distance"]),
                }
            )
        finally:
            env.close()
    return tuple(probes)


def side_pass_false_success_rate(side_pass_probes: tuple[dict[str, Any], ...]) -> float:
    false_successes = [float(probe["false_success"]) for probe in side_pass_probes]
    return _mean(false_successes)


def summarize_episodes(
    episodes: list[dict[str, Any]],
    side_pass_probes: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    acquire_steps = [float(ep["steps_to_acquire"]) for ep in episodes if ep["steps_to_acquire"] is not None]
    intercept_steps = [
        float(ep["steps_to_intercept"]) for ep in episodes if ep["steps_to_intercept"] is not None
    ]
    return {
        "episodes": float(len(episodes)),
        "flythrough_success_rate": _mean([float(ep["flythrough_success"]) for ep in episodes]),
        "first_acquisition_rate": _mean([float(ep["first_acquired"]) for ep in episodes]),
        "mean_steps_to_acquire": _mean(acquire_steps),
        "mean_steps_to_intercept": _mean(intercept_steps),
        "crash_rate": _mean([float(ep["crashed"]) for ep in episodes]),
        "out_of_bounds_rate": _mean([float(ep["out_of_bounds"]) for ep in episodes]),
        "lost_target_rate": _mean([float(ep["lost_target"]) for ep in episodes]),
        "side_pass_false_success_rate": side_pass_false_success_rate(side_pass_probes),
    }


def evaluate(mode: str = "quick", model_path: str | None = None) -> dict[str, Any]:
    model: PredictModel | None = None
    if model_path:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to evaluate a saved model") from exc
        model = PPO.load(model_path)

    episodes: list[dict[str, Any]] = []
    for scenario in scenarios_for_mode(mode):
        for repeat_index in range(scenario_repeats(mode)):
            episodes.append(run_episode(scenario, mode=mode, repeat_index=repeat_index, model=model))
    side_pass_probes = run_side_pass_probes(mode)
    metrics = summarize_episodes(episodes, side_pass_probes)
    result: dict[str, Any] = {
        "mode": mode,
        "metrics": metrics,
        "episodes": episodes,
        "side_pass_probes": list(side_pass_probes),
    }
    if model_path is not None:
        result["model_path"] = model_path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium"], default="quick")
    parser.add_argument("--model", default=None, help="Optional Stable-Baselines3 PPO .zip model")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(mode=args.mode, model_path=args.model)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
