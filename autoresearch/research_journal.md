# Autoresearch Journal

This journal is appended by `python autoresearch/run_experiment.py --mode quick`
and `python autoresearch/run_experiment.py --mode medium`.

Codex research trials may modify only:

- `autoresearch/editable/recipe.py`
- `autoresearch/editable/reward_weights.py`
- `autoresearch/editable/curriculum.py`
- `autoresearch/editable/policy_config.py`

Locked evaluator files are guarded by `autoresearch/locked_manifest.json`.

To start a trial from the stable autoresearch baseline branch:

```bash
python autoresearch/run_experiment.py --mode quick --create-branch
```

By default this creates `codex/autoresearch-<run_id>` from `Auto-Research`.


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

## 20260624T145140Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: none
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T151654Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T151745Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153323Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: none
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153326Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `medium`
- Files changed: none
- Result: accepted
- Score: `56.103571`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153842Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: rejected
- Score: `70.283333`
- Accepted or rejected: rejected
- Rejection reasons: crash_rate worsened significantly (0.250 > 0.000 + 0.050)
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153854Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153907Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `84.916667`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T153941Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `78.700000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154010Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/reward_weights.py
- Result: accepted
- Score: `78.700000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154109Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `78.700000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154136Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `78.700000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154315Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: none
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154343Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154404Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `78.700000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154423Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154440Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py
- Result: accepted
- Score: `84.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154502Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `medium`
- Files changed: autoresearch/editable/policy_config.py
- Result: rejected
- Score: `43.976407`
- Accepted or rejected: rejected
- Rejection reasons: crash_rate worsened significantly (0.143 > 0.000 + 0.050)
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154556Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154627Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/curriculum.py, autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154657Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/curriculum.py, autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154735Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: none
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154803Z

- Hypothesis: Use a spiral search pattern when target is hidden instead of simple sine sweep, covering 3D space more effectively for reacquisition.
- Mode: `quick`
- Files changed: autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154845Z

- Hypothesis: Predictive intercept with velocity-based lead factor, plus spiral search for reacquisition when target is hidden.
- Mode: `quick`
- Files changed: autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T154946Z

- Hypothesis: Direct pursuit with extra push when far from target, spiral search when hidden.
- Mode: `quick`
- Files changed: autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T155019Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `medium`
- Files changed: none
- Result: accepted
- Score: `56.103571`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T155100Z

- Hypothesis: Use a viewport-only heuristic that sweeps when the target is hidden, then accelerates through the remembered target bearing with stronger fly-through reward shaping.
- Mode: `quick`
- Files changed: autoresearch/editable/reward_weights.py
- Result: accepted
- Score: `89.900000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T155133Z

- Hypothesis: Slightly increased intercept and search gains for faster, more aggressive target pursuit while maintaining stability across all quick-mode scenarios.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## Session Research Summary (20260624T155200Z)

- Baseline: score=89.9, flythrough_success_rate=0.75
- Accepted improvement: intercept_gain 1.35→1.52 (+12.6%), other gains +5-15%, damping -0.16→-0.15
- Net delta: score 89.9 → 89.95 (+0.05), flythrough stable at 0.75
- Rejected: intercept_gain ≥1.53 causes target loss in evasive scenario, crash_rate spike at gain 2.0

**Negative results** (score=89.90, same as baseline):
- Spiral search pattern (replacing sine sweep)
- Predictive intercept with velocity-based lead factor
- Extra forward push when far from target (dist > 5.0)
- Reward weight modifications (no effect on hand-coded heuristic, as expected)
- Target bearing memory across visibility changes
- lock_memory_steps 24→32 (same metrics)
- min_closing_speed 0.0→1.5 (same metrics)

**Critical finding**: intercept_gain boundary is narrow (1.52 OK, 1.53 target loss). Evasive_high_viewport is the bottleneck for flythrough_success_rate - the heuristic cannot achieve flythrough on this scenario within 260 steps.

**What to try next**: Improve evasive scenario handling via curriculum progression (start simpler scenarios, increase difficulty), or switch from hand-coded heuristic to learned policy that can discover non-linear intercept strategies.

## Next Hypothesis

- Hypothesis: Add state-aware lead pursuit for evasive targets. When the target is visible or locked and lateral relative velocity is high, steer toward a bounded lead point along the target velocity instead of steering directly at relative position.
- Goal: Improve the evasive bottleneck without increasing crash_rate, out_of_bounds_rate, lost_target_rate, or side_pass_false_success_rate.
- Edit target: `autoresearch/editable/recipe.py` first; only touch `autoresearch/editable/policy_config.py` if a new bounded lead gain is needed.
- Keep: `intercept_gain=1.52` unless the lead term makes it unstable.
- Avoid: global aggression increases; prior rounds showed they quickly cause target loss or crashes.
- Quick acceptance signal: `flythrough_success_rate` exceeds `0.75`, or `mean_steps_to_intercept` improves while all hard gates remain clean.
- Medium acceptance signal: medium `flythrough_success_rate` improves above `0.5714285714` without significant crash-rate worsening.
- First command: `python autoresearch/run_experiment.py --mode quick`
- Promotion command: `python autoresearch/run_experiment.py --mode medium`

## 20260624T183035Z

- Hypothesis: Slightly increased intercept and search gains for faster, more aggressive target pursuit while maintaining stability across all quick-mode scenarios.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `89.950000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T183406Z

- Hypothesis: Slightly increased intercept and search gains for faster, more aggressive target pursuit while maintaining stability across all quick-mode scenarios.
- Mode: `quick`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `115.750000`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T183419Z

- Hypothesis: Slightly increased intercept and search gains for faster, more aggressive target pursuit while maintaining stability across all quick-mode scenarios.
- Mode: `medium`
- Files changed: autoresearch/editable/policy_config.py, autoresearch/editable/recipe.py
- Result: accepted
- Score: `95.451984`
- Accepted or rejected: accepted
- Rejection reasons: none
- What to try next: Stress test with evasive and orbit scenarios before promoting the recipe.

## 20260624T183430Z Session note

- Primary adaptive pursuit attempt `20260624T183035Z` was a no-op despite mechanical acceptance: it compared normalized relative velocity against `maneuver_threshold=2.0`, so the maneuver branch did not activate and quick stayed at score `89.95`, flythrough `0.75`.
- Corrected implementation converts viewport-normalized relative position/velocity back to raw units for maneuver detection and lead-point pursuit while leaving the non-maneuver and search branches unchanged.
- Corrected quick run `20260624T183406Z`: score `115.75`, flythrough `1.0`, `evasive_high_viewport` succeeds at step `58`, all quick guardrails remain `0.0`.
- Promotion run `20260624T183419Z`: medium score `95.451984`, flythrough `0.8571428571428571`, crash/lost-target/side-pass rates `0.0`; remaining medium failure is `far_hidden_search` with out-of-bounds rate `0.14285714285714285`.
- Next useful experiment: address far hidden search acquisition/out-of-bounds separately; do not retune evasive pursuit unless a regression appears.
