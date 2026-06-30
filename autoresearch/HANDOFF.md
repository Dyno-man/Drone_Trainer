# Autoresearch Handoff

Goal: improve viewport-based search and true fly-through interception without rebuilding the environment.

Current best (quick mode): score=115.76, flythrough_success_rate=1.0 (4/4 scenarios).
Evasive solved: `evasive_high_viewport` — final_distance=1.8m, ~57 steps to intercept.

Current best config (verified on quick & medium mode):
- `intercept_gain=1.52`, `velocity_gain=2.0`, `lead_gain=10.0`, `damping_gain=0.15`
- Recipe: predictive lead (`lead_pos = relative_pos + 0.05 * relative_vel`) + blend aim + cross-product perpendicular lead + velocity correction
- Blend: `blend = clip(1.0 - dist / 0.3, 0, 1)` — conservative when far, aggressive when close
- Search: spiral pattern with sine oscillation (unchanged from v1)

Primary metric: `flythrough_success_rate`

Hard rejection gates:
- `side_pass_false_success_rate > 0`
- `scripts/check_env.py` fails
- `pytest -q` fails
- locked evaluator or benchmark files are modified
- `crash_rate` gets significantly worse than the best accepted run

## Edit Only These Files

- `autoresearch/editable/recipe.py`
- `autoresearch/editable/reward_weights.py`
- `autoresearch/editable/curriculum.py`
- `autoresearch/editable/policy_config.py`

Do not edit locked files during research trials:

- `autoresearch/locked/evaluator.py`
- `autoresearch/locked/scoring.py`
- `autoresearch/locked/benchmark_scenarios.py`
- `autoresearch/locked/anti_cheat_checks.py`
- `autoresearch/locked_manifest.json`

## Standard Trial Loop

1. Inspect the leaderboard:

```bash
python autoresearch/compare_runs.py
```

2. Form one small hypothesis.

3. Modify only files in `autoresearch/editable/`.

4. Run a quick trial:

```bash
python autoresearch/run_experiment.py --mode quick
```

5. If accepted and promising, run a medium trial:

```bash
python autoresearch/run_experiment.py --mode medium
```

6. Compare results:

```bash
python autoresearch/compare_runs.py
```

7. Keep accepted changes only if they improve the objective and do not trip any rejection gate.

## Branch-Based Trial

Use this when starting a new isolated research attempt from the stable baseline branch.

```bash
python autoresearch/run_experiment.py --mode quick --create-branch
```

Default behavior:
- base branch: `Auto-Research`
- new branch: `codex/autoresearch-<run_id>`

Optional explicit branch:

```bash
python autoresearch/run_experiment.py --mode quick --create-branch --branch-name codex/my-research-idea
```

The worktree must be clean before using `--create-branch`.

## Evaluate A Saved Model

```bash
python autoresearch/locked/evaluator.py --model path/to/model.zip
```

Use this for model inspection only. Research changes should still go through `run_experiment.py`.

## Generated Outputs

Each experiment writes an ignored run folder:

```text
autoresearch/runs/<run_id>/
```

Expected files:
- `proposal.md`
- `metrics.json`
- `diff.patch`
- `eval_summary.md`
- `validation.log`

Persistent tracked summaries:
- `autoresearch/leaderboard.csv`
- `autoresearch/research_journal.md`

## Decision Rules

Accept:
- run is marked `accepted=true`
- `flythrough_success_rate` improves or supports the current research direction
- `side_pass_false_success_rate` remains `0.0`
- crash and out-of-bounds rates do not regress materially

Reject:
- any hard gate trips
- improvement is only proximity-like behavior rather than fly-through interception
- changes require touching locked files
- results are noisy without a clear hypothesis

## Known Research History

**What worked:** `velocity_gain=2.0` with cross-product perpendicular lead. Improved evasive from 18.6m→13.95m.

**v2 breakthrough:** Predictive lead (`lead_pos = relative_pos + 0.05 * relative_vel`) with blend-based aim broke the evasive ceiling. Quick score: 90.47→115.76 (+28%). Evasive: 13.95m→1.8m intercept. All 4 quick scenarios flythrough.

**What failed (v2 ablations):**
- Predictive lead at k=0.3: crashed, out-of-bounds
- Predictive lead at k=0.1: crashed, out-of-bounds
- Predictive lead at k=0.05 WITHOUT velocity_gain=2.0: marginal improvement (14m→still no flythrough)
- Distance-adaptive k (inverted): evasive got worse (21.9m final distance)
- Pure in-plane lateral correction without velocity term: out-of-bounds

**Key insight:** The evasive target's motion is predictable (80% away + 30% lateral). The linear policy's cross-product term captures perpendicular motion but lacks predictive positioning. Adding a small (5%) predictive lead to the aim direction allows the pursuer to cut across the target's evasive path instead of chasing it. The blend factor prevents far-target overshoot while keeping full lead when close.

## Minimal Fresh-Context Prompt

```text
Use autoresearch/HANDOFF.md. Evasive ceiling already broken by v2 (score=115.76).
Current: velocity_gain=2.0 + predictive lead (k=0.05) + blend aim + cross-product.
Quick: flythrough=1.0, mean_steps_to_intercept=44.8.
Medium: flythrough=0.857, OOB=0.143 (far_hidden_search issue).
Only edit autoresearch/editable/*.py. Run quick experiments, compare leaderboard.
Consider tuning blend threshold or velocity_gain for far-target stability.
```
