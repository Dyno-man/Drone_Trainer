# Autoresearch Journal

This journal is appended by `python autoresearch/run_experiment.py --mode quick`
and `python autoresearch/run_experiment.py --mode medium`.

Codex research trials may modify only:

- `autoresearch/editable/recipe.py`
- `autoresearch/editable/reward_weights.py`
- `autoresearch/editable/curriculum.py`
- `autoresearch/editable/policy_config.py`

Locked evaluator files are guarded by `autoresearch/locked_manifest.json`.


## 20260624T141756Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T141928Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T142245Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.
